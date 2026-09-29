# Numeric kernels and compute backends

Every floating point reduction in EtAlii.Dllm runs in a small C++ library with one documented evaluation order per
kernel. This page shows how that library is organised: the Python wrappers, the bindings, the header-only kernels,
the dispatch to threads, SIMD and the GPU, and the build. The per-kernel formulas and orders are in
[kernels](../kernels.md); why they matter is in [determinism by design](determinism.md).

## Layers

```mermaid
flowchart TB
    subgraph py["Python (src/etalii_dllm/)"]
        direction LR
        tensor["tensor.py<br/>Tensor: aligned float32"]
        numerics["numerics.py<br/>linear, rms_norm, rope, attention,<br/>softmax, argmax, DeterministicRandom,<br/>PackedWeight, QuantizedWeight"]
        cudapy["cuda.py<br/>NVRTC discovery, CudaTensor,<br/>CudaWeight, device ops"]
        training["training/backprop.py<br/>gradient wrappers"]
    end

    subgraph bind["Bindings (cpp/kernels.cpp, nanobind)"]
        kernels["etalii_dllm._kernels<br/>checks shapes, allocates aligned outputs,<br/>releases the GIL"]
    end

    subgraph hdr["Header-only kernels (cpp/include/dllm/)"]
        direction LR
        random["random.hpp"]
        math["math.hpp<br/>reductions, exp, log,<br/>sin, cos, tanh, erf"]
        nn["nn.hpp<br/>linear, matmul, rms_norm,<br/>rope, attention"]
        grad["grad.hpp<br/>backward kernels,<br/>cross-entropy, AdamW"]
        quant["quant.hpp<br/>Q8_0"]
    end

    subgraph exec["Execution"]
        direction LR
        simd["simd.hpp<br/>AVX2+FMA / SSE2 / NEON / scalar"]
        pool["parallel.hpp<br/>persistent thread pool"]
        cudahpp["cuda.hpp<br/>driver + NVRTC loaded at run time"]
        cu["cuda/kernels.cu<br/>(embedded source)"]
    end

    tensor & numerics & training --> kernels
    cudapy --> kernels
    kernels --> random & math & nn & grad & quant
    nn & grad & quant --> simd & pool
    kernels --> cudahpp --> cu
    cu -.->|"compiled with"| math
```

- **Python decides what to compute, C++ computes it.** `numerics.py` is a thin layer: it converts inputs to
  `float32` `Tensor`s, calls `_kernels` and wraps results. Elementwise work (residual additions, the SwiGLU product,
  temperature scaling) may stay in NumPy because it has no order to get wrong; sums, dot products and norms may not.
- **The bindings stay thin.** `kernels.cpp` validates shapes, allocates 64-byte aligned outputs and releases the
  GIL around long kernels, so requests on several Python threads can run kernels at the same time.
- **Kernels are header-only**, in the `dllm` namespace, so the CPU and GPU builds share `math.hpp` verbatim.

## Splitting work without changing a bit

The kernels are fast because they compute many outputs at once, never because they split one sum.

```mermaid
flowchart LR
    call["linear(x[rows, in], W[out, in])"] --> tasks["tasks = panels of 16 outputs × blocks of rows<br/>(count from the data size only)"]
    tasks --> pool{"thread pool free?"}
    pool -->|yes| workers["worker threads take tasks<br/>through an atomic counter"]
    pool -->|"no (another request)"| caller["all tasks on the calling thread"]
    workers & caller --> panel["one task: 16 outputs × one block of rows"]
    panel --> isa{"instruction set<br/>(fixed at start-up)"}
    isa -->|avx2| v4["4 outputs per register"]
    isa -->|"sse2 / neon"| v2["2 outputs per register"]
    isa -->|scalar| v1["1 output at a time"]
    v4 & v2 & v1 --> acc["each output: acc += x[k] · w[k]<br/>k ascending, in double, one rounding"]
```

- **`parallel.hpp`**: a process-wide pool (`DLLM_THREADS`, `--threads`, `numerics.set_threads`; default all cores).
  Each task owns a disjoint set of outputs and nothing is combined across tasks, so the thread count and scheduling
  are speed settings only. When the pool is busy with another caller's kernel, the tasks run on the calling thread,
  with the same result. After `fork()` a child starts its own workers.
- **`fpenv.hpp`**: every binding runs under `FpEnvGuard`, which sets round-to-nearest with subnormals kept for the
  call and restores the caller's state; pool workers enter that state when they start. A library that turned
  flush-to-zero on for the process therefore cannot change a kernel's bits.
- **`simd.hpp`**: the widest supported variant is chosen once per process. SIMD lanes hold *different* outputs,
  each advancing through `k` in the same order as the scalar loop. The product of two floats is exact in double, so
  an FMA rounds once exactly like a multiply plus an add, and every variant equals `linear_reference` bit for bit.
  `PackedWeight` rearranges a weight once at load time into the `[in][16]` panels the vector loop reads.
- **Tiling** (for example `matmul`'s 64-column, 256-deep tiles) only changes the order in which outputs are
  visited; each output keeps one accumulator across all tiles.
- **Q8_0** (`quant.hpp`) sums int8 products within a block in int32, which is exact and so order-free, then combines
  blocks in ascending order in double.

`tests/test_batch_invariance.py` forces every supported instruction set and several thread counts, and requires the
bits of the scalar reference, of a lone row, and of a lone request.

## GPU backend

```mermaid
sequenceDiagram
    participant P as cuda.py
    participant K as _kernels (cuda.hpp)
    participant D as NVIDIA driver
    participant N as NVRTC

    P->>P: find_nvrtc(): DLLM_NVRTC, nvidia-* wheels,<br/>PyTorch, CUDA_PATH, system paths
    P->>K: cuda_initialize(nvrtc path, device)
    K->>D: dlopen / LoadLibrary libcuda, pick device
    K->>N: dlopen / LoadLibrary NVRTC
    K->>N: compile kernels.cu + math.hpp<br/>--fmad=false, IEEE div/sqrt, no FTZ,<br/>sm_XX (or compute_XX PTX)
    N-->>K: cubin / PTX
    K->>D: load module
    Note over P,K: weights uploaded once (CudaWeight),<br/>activations and KV cache stay on the device
    P->>K: linear / rms_norm / rope / attention / swiglu on CudaTensor
    K->>D: launch on one stream, one thread per output element
    P->>K: download logits
```

- **Nothing CUDA at build time.** CMake embeds `cuda/kernels.cu` and `math.hpp` in the extension as byte arrays.
  At run time the driver and NVRTC are loaded dynamically, so one wheel works with or without a GPU and without a
  CUDA toolkit; without them `--device cuda` fails with a message naming what is missing.
- **The CPU's order on the GPU.** Each kernel in `kernels.cu` mirrors a CPU kernel with one GPU thread per output
  element and the same accumulation. No atomics, no warp-shuffle reductions, no tensor cores, no CUDA math library:
  transcendentals are `math.hpp` compiled for the device. The GPU therefore returns the CPU's bits
  (`tests/test_cuda.py`), and a model has the same `system_fingerprint` on both devices.
- **Device-resident decoding.** `transformer.py` keeps weights, activations and the KV cache on the GPU; only the
  embedding rows go up and the logits come back.

## Build

```mermaid
flowchart LR
    pip["pip install -e .[dev]"] --> skb["scikit-build-core<br/>(pyproject.toml)"]
    skb --> cmake["CMakeLists.txt"]
    cmake --> embed["embed kernels.cu + math.hpp<br/>→ generated/dllm_cuda_sources.hpp"]
    cmake --> flags["-ffp-contract=off -fno-fast-math<br/>(MSVC: /fp:precise)"]
    embed & flags --> nb["nanobind_add_module(_kernels)"]
    nb --> so["etalii_dllm/_kernels.*.so / .pyd"]
```

The compiler flags are part of the determinism contract: they stop the compiler from reassociating sums or fusing
multiplies and adds on its own. AVX2 variants are enabled per function through target attributes, not through
`-march=native`, so one binary runs on every x86-64 CPU and picks its path at start-up.

## Adding a kernel

1. Write it in `cpp/include/dllm/` with one accumulation per output, index ascending, `double` accumulator, and
   no branch on input size.
2. Bind it in `kernels.cpp` (shape checks, aligned output, GIL released) and wrap it in `numerics.py`.
3. If the decoder uses it on the GPU, add the mirror kernel to `cpp/cuda/kernels.cu` and a CPU-vs-GPU test.
4. Document its order in [kernels](../kernels.md) and test it against a plain reference loop (and, for new
   transcendentals, for accuracy).
