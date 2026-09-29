# Kernels

The numeric kernels live in `cpp/include/dllm/` (header-only C++17) and are exposed through
`etalii_dllm.numerics`. This page documents the evaluation order each one commits to, because that order is what
makes the output reproducible. Changing it is a deliberate, golden-value-changing act (see `CLAUDE.md`).

## Common rules

- Inputs are `float32`; every reduction accumulates in `double`, over the reduced index in ascending order, and
  rounds to `float32` once at the end.
- Each output element has its own accumulation. Loops may be tiled or reordered across outputs, but never inside
  one accumulation, so an output's bits do not depend on the batch size, the sequence length, the tile sizes or
  which other rows are computed alongside it (**batch invariance**).
- No kernel switches strategy on input size. Threads partition outputs, never reductions (see
  [threads and SIMD](#threads-and-simd)).
- Transcendentals come from `math.hpp` (built from `+ - * /` and `sqrt` only), never from the C runtime.

## Tensor

`etalii_dllm.tensor.Tensor` is the data type the kernels and models share: read-only, C-contiguous, little-endian
`float32`, with storage aligned to 64 bytes (kernel outputs are allocated the same way). `Tensor(x)` copies only when
`x` is not already in that form. `tensor == other` compares bits, and `fingerprint()` hashes shape and bits.

## Transcendentals (`math.hpp`)

| Function | Method | Accuracy (vs. glibc, double) |
| --- | --- | --- |
| `exp` | `x = k ln2 + r`, two-part ln2, degree-13 Taylor, exact scaling by `2^k` | a few ulp |
| `log` | `x = m 2^e`, `m` in `[sqrt(1/2), sqrt(2))`, `2 atanh((m-1)/(m+1))` series to `s^25` | ~1 ulp |
| `sin`, `cos` | Cody-Waite reduction by `pi/2` (three 33-bit parts from fdlibm), Taylor to degree 15/16 on `[-pi/4, pi/4]` | ~1 ulp for `|x| < 1.6e6` |
| `tanh` | Taylor to `x^15` below 0.125, else `1 - 2 / (e^(2x) + 1)` | < 2e-15 relative |
| `sigmoid` | `1 / (1 + e^-x)` or `e^x / (1 + e^x)`, never exponentiating a large positive number | ~1e-15 relative |
| `erf`, `erfc` | 60-term Maclaurin series below 2.5, Laplace continued fraction (depth 80) for erfc above | erf < 1e-14 absolute, erfc < 1e-13 relative |

Beyond `|x| = 1.6e6` the sine/cosine reduction loses accuracy but stays deterministic; RoPE angles
(`position * inv_freq`) stay far below that for any realistic context length.

`softmax` and `log_softmax` subtract the maximum (first index on ties), sum `exp(l_j - max)` over `j` ascending in
double, and round each output once: `softmax = e_i / total`, `log_softmax = (l_i - max) - log(total)`. The sampler
uses the first, API `logprobs` the second.

## Linear algebra (`nn.hpp`)

- **`linear(x, weight, bias)`**: `x[..., in] @ weight[out, in]^T + bias` (PyTorch `nn.Linear` layout; leading
  dimensions are flattened into rows). Output `[r, n]` is `sum_k x[r, k] * w[n, k]` over `k` ascending, plus
  `bias[n]` in double, rounded once. `linear_reference` in `nn.hpp` states exactly that as a plain loop; the fast
  `linear` computes the same bits (see [threads and SIMD](#threads-and-simd)). `weight` may also be a
  `PackedWeight` (the same matrix pre-arranged for the SIMD kernel, same bits) or a `QuantizedWeight`
  ([Q8_0](#q8_0-quantisation), different bits).
- **`matmul(a, b)`**: `a[m, k] @ b[k, n]`. Tiled over 64 columns and 256-deep slices of `k`, with one double
  accumulator per output kept across the depth slices, so the sum still runs over `k` ascending and each output
  equals `dot(a[i, :], b[:, j])` bit for bit.

## Normalisation and activations

- **`rms_norm(x, weight, eps, add_unit_offset)`**: per row, `sum x^2` ascending in double, `inv = 1 / sqrt(sum/dim +
  eps)` (IEEE `sqrt` is correctly rounded), `out = x * inv * w` (or `* (1 + w)` for Gemma), rounded once.
- **`silu`**: `x * sigmoid(x)`. **`gelu`**: exact `0.5 x (1 + erf(x / sqrt 2))` (evaluated as `0.5 x erfc(-x / sqrt 2)` so the negative tail does not cancel), or with `approximate="tanh"` the
  GPT-2 form. All elementwise in double, rounded once.

## RoPE

- **`rope_inv_freq(head_dim, theta, rotary_dim, scaling)`** computes `theta^(-2i/rotary_dim)` as
  `exp(-(2i/rotary_dim) log theta)` in double with the kernels above, then applies Hugging Face `rope_scaling`:
  `linear` or `llama3`. (Hugging Face computes these in float32; our values are closer to exact, so they will not
  match its bits, only its values to float precision.)
- **`rope(x, positions, inv_freq, interleaved)`** rotates `x[tokens, heads, head_dim]`. The angle
  `position * inv_freq[i]` is formed in double and fed to `dllm::sin`/`dllm::cos`; the rotation is done in double and
  rounded once. `interleaved=False` pairs `(i, i + rotary_dim/2)` (Hugging Face checkpoints), `True` pairs
  `(2i, 2i+1)` (Meta/GGUF). Dimensions past `rotary_dim` (partial rotary) pass through. A token's output depends
  only on its own position.

## Attention

`attention(q, k, v, scale, causal, q_offset)` with `q[q_len, q_heads, d]`, `k[kv_len, kv_heads, d]`,
`v[kv_len, kv_heads, dv]`. Query head `h` uses key/value head `h / (q_heads / kv_heads)` (MHA, GQA and MQA). With
`causal`, query `t` is at position `q_offset + t` (default `kv_len - q_len`) and sees keys `0 ..= q_offset + t`.

For each (query, head), independently:

1. scores `s_j = (sum_i q_i k_ji) * scale`, dot over `i` ascending, kept in double;
2. `m = max_j s_j`; `p_j = exp(s_j - m)`; `Z = sum_j p_j` over `j` ascending;
3. `out_i = (sum_j p_j v_ji) / Z`, the sum over `j` ascending, rounded once.

Keys beyond the causal horizon are never read, so the result does not depend on how long the KV cache is, and a
prefill of `n` tokens gives exactly the same bits as decoding them one at a time (`tests/test_kernels.py`).

## Threads and SIMD

Phase 6 made the kernels fast without touching the order above. Two facts make that possible:

1. **Every output element owns its accumulation.** Work is split by output element, never inside a sum, so how the
   work is split, how many threads there are and which thread runs what cannot change a bit. `parallel.hpp` is a
   small persistent pool (`DLLM_THREADS`, `--threads`, or `numerics.set_threads`; default: all cores). A kernel
   called while the pool is busy (another request's kernel) runs on the calling thread, which gives the same bits.
2. **SIMD lanes are different outputs.** `linear` groups outputs in panels of 16 whose weights are repacked to
   `[in][16]` (`PackedWeight` does it once at load time). One SIMD register then holds the accumulators of 4 (AVX2)
   or 2 (SSE2, NEON) *different* outputs, which all advance through `k` ascending together. Each lane still does
   exactly `acc += x[k] * w[k]` in double. A product of two floats is exact in double (24 + 24 significant bits
   fit in 53), so a fused multiply-add rounds exactly once, like the separate multiply and add: FMA gives the same
   bits here. (The compiler is still forbidden to contract anything on its own.)

`attention` runs one task per (query, head) row and scores four keys at a time, each with its own dot-product
accumulator; `linear_backward` runs one task per row of `dx` and per row of `dweight`.

The code path is chosen once per process from the CPU (`simd.hpp`): `avx2` (AVX2 + FMA, x86-64 with GCC/Clang)
or `portable` (SSE2 on x86-64, NEON on arm64, plain C++ elsewhere). `numerics.instruction_set()` reports it, and
the tests force every supported path and thread count and require the bits of `linear_reference`
(`tests/test_batch_invariance.py`). All golden hashes from before Phase 6 are unchanged.

Measured on a 4-core cloud VM (Xeon, AVX2) with SmolLM2-135M, float32 weights:

| | before Phase 6 | 1 thread | 4 threads | 4 threads, Q8_0 |
| --- | --- | --- | --- | --- |
| Prompt, 64 tokens | 15 s | 1.0 s | 0.43 s | 0.53 s |
| Generation, per token | 346 ms | 99 ms | 46 ms | 27 ms |

Generation is limited by memory bandwidth (every weight is read once per token), which is what Q8_0 helps with.

## Q8_0 quantisation

`--quantize q8_0` (or `DLLM_QUANTIZE=q8_0`, `QuantizedWeight`) runs the linear layers on 8-bit weights, a quarter
of the memory traffic. The layout is fixed: blocks of 32 consecutive inputs of one output row share a float32 scale.

- **Quantising** a block (weights at load time, activations on every call): `amax = max |x_i|`, `d = amax / 127`
  (float), `q_i = round(x_i * (1 / d))` with round-half-to-even implemented with exact operations (independent of
  the floating point environment), clamped to ±127. An all-zero block has `d = 0` and `q = 0`.
- **Accumulating**: per block the 32 products are summed in `int32`. Integer addition is associative and cannot
  overflow here (32 · 127 · 127 < 2³¹), so that sum may run in any order and is vectorised freely. Blocks are then
  combined per output in double, blocks ascending: `acc += (double(d_x) * double(d_w)) * double(isum)`, plus the
  bias, rounded once.

Quantised output is deterministic but not the float output: it gets its own `system_fingerprint` (the weights
fingerprint hashed with `:q8_0`), and the tests check it against an exact reference and its golden hash. Layers
whose input size is not a multiple of 32 stay float32. Embedding lookups stay float32; the LM head is quantised.

## GPU

Not yet (issue #31). The design carries over: one GPU thread per output element running the documented order,
no atomics and no split-K reductions, with `math.hpp` compiled for the device, would even reproduce the CPU bits.
What is missing is hardware to prove it: neither the CI runners nor the cloud sessions have a GPU.

## Gradients (`grad.hpp`)

The backward kernels used for [fine-tuning](training.md) follow the same rules: one double accumulator per output
element, a fixed order, one rounding.

- **`linear_backward`**: `dx[r, k] = sum_n dy[r, n] w[n, k]` (`n` ascending), `dw[n, k] = sum_r dy[r, n] x[r, k]`
  and `db[n] = sum_r dy[r, n]` (`r` ascending).
- **`rms_norm_backward`**: `inv` recomputed as in the forward kernel; `dx_i = inv w_i dy_i - x_i inv^3 / dim *
  sum_j w_j dy_j x_j` (`j` ascending); `dw_i = sum_r dy x inv` over rows ascending.
- **`silu_backward`**: `dy * s (1 + x (1 - s))` with `s = sigmoid(x)`, in double.
- **`rope(..., inverse=True)`**: the same rotation with `sin` negated (exactly), i.e. the transpose.
- **`attention_backward`**: per (query, head), in query-then-head order, the probabilities are recomputed exactly as
  in the forward pass; `dp_j = sum_i dout_i v_ji`, `D = sum_j p_j dp_j`, `ds_j = p_j (dp_j - D)`,
  `dq = scale sum_j ds_j k_j`; `dk_j` and `dv_j` accumulate `scale ds_j q` and `p_j dout` in double over queries
  ascending, then heads ascending.
- **`cross_entropy`**: `logsumexp = max + log(sum_j exp(l_j - max))` (`j` ascending, `dllm` exp/log); the loss is
  summed over rows ascending; `dlogits = (softmax - onehot) * scale`. Negative targets are skipped.
- **`embedding_backward`**: rows grouped per token id by a counting sort that keeps positions ascending, then
  summed in double.
- **`sum_squares`** and **`adamw_step`**: see [training](training.md).
