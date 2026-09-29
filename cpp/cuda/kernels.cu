// CUDA kernels of the GPU backend. This file is not compiled with the extension: CMake embeds it (with math.hpp)
// into the extension, and cuda.hpp compiles it at run time with NVRTC for the GPU that is present.
//
// Every kernel follows the rule of the CPU kernels (docs/kernels.md): each output element is produced by one thread
// that runs that element's whole accumulation, in the documented order, in double, rounded to float once. Threads
// split outputs, never a sum; there are no atomics, no split-K and no warp shuffles in any reduction. NVRTC runs
// with --fmad=false, so the compiler cannot fuse a multiply and an add, and double +, -, *, / and sqrt are IEEE
// round-to-nearest on the GPU, as on the CPU. The results are therefore the CPU bits, not merely reproducible ones
// (tests/test_cuda.py). The one explicit fma() (in linear) multiplies two floats, whose product is exact in double,
// so it rounds exactly like the separate multiply and add of linear_reference().
#include "dllm/math.hpp"

typedef unsigned long long u64;

__device__ inline u64 thread_index() { return blockIdx.x * static_cast<u64>(blockDim.x) + threadIdx.x; }

__device__ inline u64 thread_stride() { return gridDim.x * static_cast<u64>(blockDim.x); }

// out[r, n] = sum_k x[r, k] * w[n, k] (+ bias[n]), k ascending. The weights are stored transposed, wt[k][n], so
// neighbouring threads (neighbouring n) read neighbouring words. Grid: x over outputs, y over rows.
extern "C" __global__ void linear_f32(const float* x, const float* wt, const float* bias, float* out, u64 rows,
                                      u64 in_features, u64 out_features) {
    const u64 n = thread_index();
    if (n >= out_features) {
        return;
    }
    for (u64 r = blockIdx.y; r < rows; r += gridDim.y) {
        const float* xr = x + r * in_features;
        double acc = 0.0;
#pragma unroll 8
        for (u64 k = 0; k < in_features; ++k) {
            acc = fma(static_cast<double>(xr[k]), static_cast<double>(wt[k * out_features + n]), acc);
        }
        if (bias != nullptr) {
            acc += bias[n];
        }
        out[r * out_features + n] = static_cast<float>(acc);
    }
}

// Q8_0 linear (quant.hpp). The activations arrive quantised by quantize_q8 below. Weights are stored block-major,
// wq[b][n][32] and ws[b][n], so neighbouring threads read neighbouring 32-byte blocks. The 32 products of a block
// are an exact int32 sum (any order); blocks are combined in double, ascending, exactly as on the CPU.
extern "C" __global__ void linear_q8(const signed char* xq, const float* xs, const signed char* wq, const float* ws,
                                     const float* bias, float* out, u64 rows, u64 in_features, u64 out_features) {
    const u64 n = thread_index();
    if (n >= out_features) {
        return;
    }
    const u64 blocks = in_features / 32;
    for (u64 r = blockIdx.y; r < rows; r += gridDim.y) {
        const signed char* xr = xq + r * in_features;
        const float* xsr = xs + r * blocks;
        double acc = 0.0;
        for (u64 b = 0; b < blocks; ++b) {
            const int* xb = reinterpret_cast<const int*>(xr + b * 32);
            const int* wb = reinterpret_cast<const int*>(wq + (b * out_features + n) * 32);
            int isum = 0;
            for (int i = 0; i < 8; ++i) {
#if __CUDA_ARCH__ >= 610
                // dp4a: four signed 8-bit products added to isum, exactly (NVRTC has no header for __dp4a).
                asm("dp4a.s32.s32 %0, %1, %2, %0;" : "+r"(isum) : "r"(xb[i]), "r"(wb[i]));
#else
                const int xw = xb[i];
                const int ww = wb[i];
                for (int s = 0; s < 32; s += 8) {
                    isum += static_cast<int>(static_cast<signed char>(xw >> s)) *
                            static_cast<int>(static_cast<signed char>(ww >> s));
                }
#endif
            }
            acc += (static_cast<double>(xsr[b]) * static_cast<double>(ws[b * out_features + n])) *
                   static_cast<double>(isum);
        }
        if (bias != nullptr) {
            acc += bias[n];
        }
        out[r * out_features + n] = static_cast<float>(acc);
    }
}

// RMSNorm: one block per row. Thread 0 sums the squares in index order; then every thread scales its elements.
extern "C" __global__ void rms_norm(const float* x, const float* weight, float* out, u64 dim, double eps,
                                    int add_unit_offset) {
    __shared__ double inv_rms;
    const float* xr = x + blockIdx.x * dim;
    if (threadIdx.x == 0) {
        double sum_sq = 0.0;
        for (u64 i = 0; i < dim; ++i) {
            sum_sq += static_cast<double>(xr[i]) * static_cast<double>(xr[i]);
        }
        inv_rms = 1.0 / __dsqrt_rn(sum_sq / static_cast<double>(dim) + eps);
    }
    __syncthreads();
    for (u64 i = threadIdx.x; i < dim; i += blockDim.x) {
        double scale = 1.0;
        if (weight != nullptr) {
            scale = add_unit_offset ? 1.0 + static_cast<double>(weight[i]) : static_cast<double>(weight[i]);
        }
        out[blockIdx.x * dim + i] = static_cast<float>(static_cast<double>(xr[i]) * inv_rms * scale);
    }
}

// Elementwise activations, as nn.hpp: in double from math.hpp, rounded once. kind: 0 silu, 1 gelu, 2 gelu_tanh.
extern "C" __global__ void activation(const float* x, float* out, u64 n, int kind) {
    for (u64 i = thread_index(); i < n; i += thread_stride()) {
        const double d = x[i];
        double y;
        if (kind == 0) {
            y = d * dllm::sigmoid(d);
        } else if (kind == 1) {
            y = 0.5 * d * dllm::erfc(-d * 7.07106781186547524401e-01);
        } else {
            y = 0.5 * d * (1.0 + dllm::tanh(7.97884560802865355879e-01 * (d + 0.044715 * d * d * d)));
        }
        out[i] = static_cast<float>(y);
    }
}

// RoPE: one thread per (token, head, pair) plus one per pass-through dimension; the angle and its sine and cosine
// are computed per thread, with the same operations as the CPU table.
extern "C" __global__ void rope(const float* x, const long long* positions, const double* inv_freq, float* out,
                                u64 tokens, u64 heads, u64 head_dim, u64 rotary_dim, int interleaved, int inverse) {
    const u64 half = rotary_dim / 2;
    const u64 per_head = half + (head_dim - rotary_dim);
    const u64 total = tokens * heads * per_head;
    for (u64 index = thread_index(); index < total; index += thread_stride()) {
        const u64 th = index / per_head;
        const u64 i = index % per_head;
        const u64 t = th / heads;
        const float* xh = x + th * head_dim;
        float* oh = out + th * head_dim;
        if (i >= half) {
            const u64 d = rotary_dim + (i - half);
            oh[d] = xh[d];
            continue;
        }
        const double angle = static_cast<double>(positions[t]) * inv_freq[i];
        const double c = dllm::cos(angle);
        const double s = inverse ? -dllm::sin(angle) : dllm::sin(angle);
        const u64 i0 = interleaved ? 2 * i : i;
        const u64 i1 = interleaved ? 2 * i + 1 : i + half;
        const double a = xh[i0];
        const double b = xh[i1];
        oh[i0] = static_cast<float>(a * c - b * s);
        oh[i1] = static_cast<float>(a * s + b * c);
    }
}

// Attention: one block per (query, head) row, rows [row0, row0 + gridDim.x). Scores are one thread per key (each its
// own dot product over head_dim ascending); the maximum and the softmax total are taken by thread 0 over the keys
// ascending; the output is one thread per value dimension, summing over the keys ascending. scratch holds
// kv_len doubles per block. A non-zero window keeps only the last `window` visible keys (sliding-window attention).
extern "C" __global__ void attention(const float* q, const float* k, const float* v, float* out, double* scratch,
                                     u64 row0, u64 kv_len, u64 q_heads, u64 kv_heads, u64 head_dim, u64 value_dim,
                                     double scale, int causal, u64 q_offset, u64 window) {
    __shared__ double max_score;
    __shared__ double inv_total;
    const u64 row = row0 + blockIdx.x;
    const u64 t = row / q_heads;
    const u64 h = row % q_heads;
    const u64 kvh = h / (q_heads / kv_heads);
    float* oh = out + row * value_dim;
    u64 end = kv_len;
    if (causal) {
        const u64 last = q_offset + t + 1;
        end = last < kv_len ? last : kv_len;
    }
    const u64 first = window != 0 && end > window ? end - window : 0;
    const u64 visible = end - first;
    k += first * kv_heads * head_dim;
    v += first * kv_heads * value_dim;
    if (visible == 0) {
        for (u64 i = threadIdx.x; i < value_dim; i += blockDim.x) {
            oh[i] = 0.0f;
        }
        return;
    }
    double* scores = scratch + static_cast<u64>(blockIdx.x) * kv_len;
    const float* qh = q + row * head_dim;
    for (u64 j = threadIdx.x; j < visible; j += blockDim.x) {
        const float* kj = k + (j * kv_heads + kvh) * head_dim;
        double dot = 0.0;
        for (u64 i = 0; i < head_dim; ++i) {
            dot += static_cast<double>(qh[i]) * static_cast<double>(kj[i]);
        }
        scores[j] = dot * scale;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        double max = scores[0];
        for (u64 j = 1; j < visible; ++j) {
            if (scores[j] > max) {
                max = scores[j];
            }
        }
        max_score = max;
    }
    __syncthreads();
    for (u64 j = threadIdx.x; j < visible; j += blockDim.x) {
        scores[j] = dllm::exp(scores[j] - max_score);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        double total = 0.0;
        for (u64 j = 0; j < visible; ++j) {
            total += scores[j];
        }
        inv_total = 1.0 / total;
    }
    __syncthreads();
    for (u64 i = threadIdx.x; i < value_dim; i += blockDim.x) {
        double acc = 0.0;
        for (u64 j = 0; j < visible; ++j) {
            acc += scores[j] * static_cast<double>(v[(j * kv_heads + kvh) * value_dim + i]);
        }
        oh[i] = static_cast<float>(acc * inv_total);
    }
}

// quantize_q8_0 of quant.hpp for x[rows][in_features]: one thread per 32-block, the same float operations.
__device__ inline bool finite(float v) { return (__float_as_uint(v) & 0x7f800000u) != 0x7f800000u; }

__device__ inline float round_half_even(float v) {
    const float f = floorf(v);
    const float diff = v - f;
    if (diff > 0.5f) {
        return f + 1.0f;
    }
    if (diff < 0.5f) {
        return f;
    }
    return fmodf(f, 2.0f) == 0.0f ? f : f + 1.0f;
}

extern "C" __global__ void quantize_q8(const float* x, signed char* q, float* scales, u64 rows, u64 in_features) {
    const u64 blocks = in_features / 32;
    for (u64 index = thread_index(); index < rows * blocks; index += thread_stride()) {
        const float* xb = x + index * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) {
            const float a = fabsf(xb[i]);
            if (a > amax && finite(a)) {
                amax = a;
            }
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        scales[index] = d;
        for (int i = 0; i < 32; ++i) {
            float v = finite(xb[i]) ? round_half_even(xb[i] * id) : 0.0f;
            v = v > 127.0f ? 127.0f : (v < -127.0f ? -127.0f : v);
            q[index * 32 + i] = static_cast<signed char>(v);
        }
    }
}

// The decoder's float32 elementwise steps: the SwiGLU product float(silu(gate)) * up and the residual addition.
extern "C" __global__ void swiglu(const float* gate, const float* up, float* out, u64 n) {
    for (u64 i = thread_index(); i < n; i += thread_stride()) {
        const double d = gate[i];
        const float s = static_cast<float>(d * dllm::sigmoid(d));
        out[i] = s * up[i];
    }
}

extern "C" __global__ void add(const float* a, const float* b, float* out, u64 n) {
    for (u64 i = thread_index(); i < n; i += thread_stride()) {
        out[i] = a[i] + b[i];
    }
}

extern "C" __global__ void copy_words(const unsigned int* source, unsigned int* destination, u64 n) {
    for (u64 i = thread_index(); i < n; i += thread_stride()) {
        destination[i] = source[i];
    }
}
