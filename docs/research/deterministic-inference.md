# Deterministic inference: research notes

Goal: identical output tokens for identical (weights, prompt, context window) inputs on the same hardware, on every run.
This note lists where non-determinism comes from in LLM inference, what mainstream stacks do about it, and the rules
EtAlii.Dllm adopts. References at the end are starting points for deeper reading.

## Three levels of determinism

1. **Run-to-run on one machine.** Same process, same hardware, same request gives the same output. Most serving
   stacks do *not* guarantee even this under load.
2. **Batch invariance.** The output of one request does not depend on which other requests share its batch, or on
   how long the prompt prefix in the KV cache is.
3. **Cross-platform bit-exactness.** Same output on x64 and Arm64, Windows, Linux and macOS, across runtime versions.

EtAlii.Dllm targets **all three levels**. It started with levels 1 and 2 only, expecting that portability would have
to be given up for speed. It did not: with double accumulators in one fixed order, its own transcendentals and SIMD
lanes that hold different outputs, the AVX2, SSE2, NEON and CUDA paths already gave the same bits, so Phase 10
made level 3 a guarantee (see [portable determinism](../kernels.md#portable-determinism)). What it rules out is
reductions whose order or intermediate precision the hardware decides (tensor cores, split-K, warp shuffles).

## Where non-determinism comes from

| Source | Why it breaks reproducibility |
| --- | --- |
| Floating point non-associativity | `(a + b) + c ≠ a + (b + c)` in IEEE 754. Any change in summation order changes the last bits, and after dozens of layers and an argmax, the chosen token. |
| Parallel reductions | Atomics, split-K matrix multiplication and tree reductions whose shape depends on thread count or scheduling reorder sums. |
| Batch-size dependent kernels | Kernels pick different tiling or reduction strategies by batch size, so a request's result depends on its neighbours. This, not GPU "randomness", is the main cause of non-determinism in production LLM serving (Thinking Machines, 2025). |
| Kernel autotuning | Libraries benchmark several algorithms at start-up and pick the fastest, which may differ per run or machine. |
| SIMD width and BLAS | A vectorised sum over 4, 8 or 16 lanes groups terms differently. NumPy selects SIMD kernels at runtime by CPU and uses pairwise summation; BLAS libraries (OpenBLAS, MKL) choose block sizes and thread splits by matrix shape and thread count, so the same row can give different bits in a different batch. |
| Fused multiply-add | `fma(a, b, c)` rounds once, `a * b + c` twice. Compilers that contract expressions on some targets and not others give different bits. |
| Transcendental functions | `exp`, `log`, `sin`, `tanh` come from the platform C runtime; they are not required to be correctly rounded and differ between glibc, MSVC and Apple libm. |
| Random number generators | Library RNGs are not specified across versions (e.g. derived distributions in `numpy.random` and `std::` distributions), and parallel draws consume the stream in scheduling order. |
| Sampling | Unstable sorts with ties, float-summed cumulative probabilities (top-p) and threshold comparisons amplify tiny differences. |
| Denormals / flush-to-zero | CPU and GPU modes that flush subnormals change results for tiny values. |
| Text handling | Unicode normalisation, culture-sensitive parsing and hash-randomised dictionary order in tokenizers. |
| Environment | Driver, library and hardware versions change kernels silently. |

## What mainstream stacks offer

- **OpenAI API**: a `seed` parameter plus a `system_fingerprint` in the response. Determinism is documented as
  best effort; the fingerprint only tells you when the backend changed.
- **PyTorch**: `torch.use_deterministic_algorithms(True)` and cuBLAS workspace settings give run-to-run
  determinism on one GPU type, at a performance cost, but not across hardware.
- **Batch-invariant kernels**: Thinking Machines (He et al., 2025) showed that batch-invariant matmul, RMSNorm and
  attention kernels make vLLM inference fully reproducible on one hardware type, with moderate overhead. vLLM and
  SGLang have since added batch-invariant / deterministic modes.
- **Reproducible BLAS** (Demmel & Nguyen, ReproBLAS): summation algorithms whose result is independent of order,
  using pre-rounding into fixed bins. Portable but costlier.
- **Correctly rounded math libraries** (CORE-MATH, CRlibm): transcendental functions with a unique correct result,
  which makes them portable by definition.
- **Integer inference**: int8/int4 weights with int32 accumulation. Integer addition is associative, so the result is
  independent of order and parallelism, as long as dequantisation happens in a fixed place.

## Rules adopted by EtAlii.Dllm

0. **Arithmetic that reduces lives in C++**, compiled with `-ffp-contract=off` / `/fp:precise` and no fast-math.
   Python and NumPy may orchestrate and do elementwise work (which is correctly rounded per element), but never sums,
   matmuls or other reductions in inference paths; no BLAS.

1. **Prefer own transcendental functions** (`cpp/include/dllm/math.hpp`: range reduction + fixed polynomial) over
   `std::exp`, `math.exp`, `numpy.exp` and friends. On one machine the platform versions are repeatable too, but a runtime or
   C library update can silently change them; our own keep results stable across upgrades.
2. **Hardware-specific instructions are allowed** (SIMD of any width, FMA), provided the code path chosen on a
   given machine never varies between runs, e.g. selected once from CPU capabilities, never by timing or autotuning.
3. **Fixed reduction order**, documented per kernel. Accumulate in `double` for `float` data. Parallelism is allowed
   only with fixed partitioning: chunks defined by the data size alone, partial sums combined in chunk order,
   independent of thread count and scheduling.
4. **Batch invariance by construction.** Kernels never branch on batch size or sequence position for their
   reduction strategy.
5. **Specified RNG.** xoshiro256\*\* seeded by SplitMix64 for sequential streams; a counter-based generator
   (Philox) when draws must be parallel, so each draw depends only on (seed, counter).
6. **Total-order sampling.** Candidates sorted by (probability desc, token id asc). Cumulative sums in `double`.
7. **Culture-invariant, ordinal text handling** in tokenizers; no reliance on set iteration order or `PYTHONHASHSEED`.
8. **Golden hashes.** Tests fix SHA-256 hashes of weights, logits and generated tokens, plus tests that the same
   request gives the same output alone, in a batch and under concurrency. A change in any hash is either a bug or an
   intentional, documented change. Once kernels become hardware-specific, hashes may be keyed per CPU/GPU family.

## Open questions

- Performance budget: how close can deterministic CPU kernels get to llama.cpp on the same hardware?
- GPU: answered in part (issue #31). Running the CPU's per-output order in double precision, without contraction,
  makes the GPU reproduce the CPU bits exactly, so it needs no fingerprint of its own (see
  [kernels](../kernels.md#gpu)). The cost is double-precision throughput; whether a float32 GPU "determinism domain"
  with its own fingerprint is worth the extra golden values remains open.
- Quantisation: is integer-only inference (including softmax and normalisation in fixed point) accurate enough
  for small models? Integer accumulation is associative, which makes parallel kernels deterministic for free.

## References

- H. He and Thinking Machines Lab, *Defeating Nondeterminism in LLM Inference*, 2025.
- D. Goldberg, *What Every Computer Scientist Should Know About Floating-Point Arithmetic*, ACM Computing Surveys, 1991.
- J. Demmel, H. D. Nguyen, *Parallel Reproducible Summation*, IEEE Transactions on Computers, 2015 (ReproBLAS).
- D. Blackman, S. Vigna, *Scrambled Linear Pseudorandom Number Generators*, ACM TOMS, 2021 (xoshiro256\*\*).
- J. Salmon et al., *Parallel Random Numbers: As Easy as 1, 2, 3*, SC 2011 (Philox).
- CORE-MATH project, correctly rounded mathematical functions.
- PyTorch documentation, *Reproducibility*.
- OpenAI API documentation, `seed` and `system_fingerprint`.
