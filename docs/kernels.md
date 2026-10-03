# Kernels

[The determinism specification](specification.md) states the exact operation sequence of every kernel in one place;
this page explains how the kernels compute it quickly.

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
- Every kernel runs in the IEEE default floating point environment, whatever the caller set (see
  [floating point environment](#floating-point-environment)).

## Floating point environment

IEEE 754 fixes the result of `+ - * /` and `sqrt` only for a given rounding mode and subnormal handling. The kernels
assume the default: round to nearest even, subnormals kept, exceptions masked. A process can be in another state
without knowing it: a library built with `-ffast-math` turns flush-to-zero (FTZ) and denormals-are-zero (DAZ) on for
the whole process when it loads, and a host application may change the rounding mode. `fpenv.hpp` makes that
irrelevant: every binding in `_kernels` runs under `FpEnvGuard`, which puts the calling thread into the default state
(MXCSR on x86-64, FPCR on arm64) for the duration of the call and restores the caller's state afterwards, and the
thread pool's workers enter the default state when they start. The cost is one control-register read per kernel
call (two writes only when the state differs). The GPU kernels are compiled with `--ftz=false`,
`--prec-div=true` and `--prec-sqrt=true` for the same reason.

NumPy's own elementwise operations outside the kernels still follow the caller's environment;
`numerics.fp_environment_is_canonical()` reports whether it is the default. `tests/test_fp_environment.py` turns
FTZ/DAZ on and requires the kernels' subnormal results and golden fingerprints to stay the same.

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
| `atan`, `acos` | `atan`: `pi/2 - atan(1/x)` above 1, `pi/6 + atan((sqrt3 x - 1)/(sqrt3 + x))` above `tan(pi/12)`, Taylor to `r^35`; `acos(x) = 2 atan(sqrt((1-x)/(1+x)))` (SLERP merges) | ~2 ulp |
| `tanh` | Taylor to `x^15` below 0.125, else `1 - 2 / (e^(2x) + 1)` | < 2e-15 relative |
| `sigmoid` | `1 / (1 + e^-x)` or `e^x / (1 + e^x)`, never exponentiating a large positive number | ~1e-15 relative |
| `erf`, `erfc` | 60-term Maclaurin series below 2.5, Laplace continued fraction (depth 80) for erfc above | erf < 1e-14 absolute, erfc < 1e-13 relative |

Beyond `|x| = 1.6e6` the sine/cosine reduction loses accuracy but stays deterministic; RoPE angles
(`position * inv_freq`) stay far below that for any realistic context length. From `|x| = 2^52 · π/2` on, the
multiple of `π/2` is used as it is rather than converted to an integer, because converting a double beyond 2^63
gives different results on x86-64 and arm64 (found by the [reference implementation](specification.md), Phase 20).

`softmax` and `log_softmax` subtract the maximum (first index on ties), sum `exp(l_j - max)` over `j` ascending in
double, and round each output once: `softmax = e_i * (1 / total)`, `log_softmax = (l_i - max) - log(total)`. The sampler
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
- **`layer_norm(x, weight, bias, eps)`** (BERT encoders): per row, `mean = sum x / dim` ascending in double, then
  `var = sum (x - mean)^2 / dim` ascending in double, `inv = 1 / sqrt(var + eps)`, `out = (x - mean) * inv * w + b`
  in double, rounded once. One thread; CPU only (encoders run on the CPU).
- **`silu`**: `x * sigmoid(x)`. **`gelu`**: exact `0.5 x (1 + erf(x / sqrt 2))` (evaluated as `0.5 x erfc(-x / sqrt 2)` so the negative tail does not cancel), or with `approximate="tanh"` the
  GPT-2 form. All elementwise in double, rounded once.
- **`swiglu(gate, up, activation)`**: the gated MLP activation `act(gate) * up` in one pass. `act(gate)` is
  rounded to float first and then multiplied by `up` in float, exactly what the two separate steps (and the CUDA
  kernel) do, so fusing them changes no bit.
- **`softcap(x, cap)`**: `cap * tanh(x / cap)` in double, rounded once (Gemma 2's final logits).
- **`sigmoid_elementwise(x)`**: `sigmoid(x)` in double, rounded once (the gate of Qwen2-MoE's shared expert).
- **`moe_route(logits, k, normalize)`**: mixture-of-experts routing, row by row on one thread: `softmax` of the router
  logits, then the `k` largest probabilities by repeated selection in a total order (larger first, equal ones to the
  lower expert), then, with `normalize`, each divided by their total (double, rank order) and rounded once. The
  experts themselves are ordinary `linear` and `swiglu` calls on the rows routed to them; every row is computed on
  its own, so how tokens are grouped by expert (which depends on the batch) cannot change a bit, and each row adds
  its experts' scaled outputs in increasing expert order.

## RoPE

- **`rope_inv_freq(head_dim, theta, rotary_dim, scaling)`** computes `theta^(-2i/rotary_dim)` as
  `exp(-(2i/rotary_dim) log theta)` in double with the kernels above, then applies Hugging Face `rope_scaling`:
  `linear`, `llama3` or `longrope` (each frequency divided by its `short_factor`). (Hugging Face computes these in float32; our values are closer to exact, so they will not
  match its bits, only its values to float precision.)
- **`rope(x, positions, inv_freq, interleaved)`** rotates `x[tokens, heads, head_dim]`. The angle
  `position * inv_freq[i]` is formed in double and fed to `dllm::sin`/`dllm::cos`; the rotation is done in double and
  rounded once. `interleaved=False` pairs `(i, i + rotary_dim/2)` (Hugging Face checkpoints), `True` pairs
  `(2i, 2i+1)` (Meta/GGUF). Dimensions past `rotary_dim` (partial rotary) pass through. A token's output depends
  only on its own position.

## Attention

`attention(q, k, v, scale, causal, q_offset, window, softcap)` with `q[q_len, q_heads, d]`, `k[kv_len, kv_heads, d]`,
`v[kv_len, kv_heads, dv]`. Query head `h` uses key/value head `h / (q_heads / kv_heads)` (MHA, GQA and MQA). With
`causal`, query `t` is at position `q_offset + t` (default `kv_len - q_len`) and sees keys `0 ..= q_offset + t`. A
non-zero `window` (sliding-window attention, Mistral) keeps only the last `window` of those keys,
`q_offset + t - window + 1 ..= q_offset + t`; the sums below then run over that range, from its first key on, which is
exactly plain attention over those keys. Without `causal`, a `window` keeps the keys closer than `window` on either
side (local attention, ModernBERT): `max(q_offset + t - window + 1, 0) ..= min(q_offset + t + window - 1, kv_len - 1)`.
ModernBERT's `local_attention` of `L` sees keys at most `L / 2` away, so its window is `L / 2 + 1`. The CUDA kernel,
`attention_backward` and the interpretability kernels (`attention_weights`) use the same range.

For each (query, head), independently:

1. scores `s_j = (sum_i q_i k_ji) * scale`, dot over `i` ascending, kept in double; with a positive `softcap`
   (Gemma 2), `s_j = softcap * tanh(s_j / softcap)` with `math.hpp`'s `tanh`, still in double;
2. `m = max_j s_j`; `p_j = exp(s_j - m)`; `Z = sum_j p_j` over `j` ascending;
3. `out_i = (sum_j p_j v_ji) * (1 / Z)`, the sum over `j` ascending, rounded once.

Keys beyond the causal horizon (or before the window) are never read, so the result does not depend on how long the
KV cache is, and a
prefill of `n` tokens gives exactly the same bits as decoding them one at a time (`tests/test_kernels.py`).

DeBERTa's disentangled attention adds position terms to the scores: `biased_attention(q, k, v, bias, scale)` takes
them as a float32 `bias[heads, q_len, kv_len]` and scores `s_j = (sum_i q_i k_ji + bias_j) * scale`, the dot over
`i` ascending in double and the bias added in double before scaling, then steps 2 and 3 above. Every key is
visible. Threads split the (query, head) rows; there is no SIMD tile, so every machine runs the same scalar loop.
The bias itself comes from `linear` and an elementwise gather (`docs/specification.md#6-the-encoder`).

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

`attention` runs one task per (key/value head, tile of four query rows); see [Phase 13](#phase-13-tiles) below.
`linear_backward` runs one task per row of `dx` and per row of `dweight`.

The code path is chosen once per process from the CPU (`simd.hpp`): `avx2` (AVX2 + FMA, x86-64 with GCC/Clang)
or `portable` (SSE2 on x86-64, NEON on arm64, plain C++ elsewhere). `numerics.instruction_set()` reports it,
`DLLM_ISA=portable` forces the portable path (for checks; it never changes the bits), and
the tests force every supported path and thread count and require the bits of `linear_reference`
(`tests/test_batch_invariance.py`). All golden hashes from before Phase 6 are unchanged.

Measured on a 4-core cloud VM (Xeon, AVX2) with SmolLM2-135M, float32 weights:

| | before Phase 6 | 1 thread | 4 threads | 4 threads, Q8_0 |
| --- | --- | --- | --- | --- |
| Prompt, 64 tokens | 15 s | 1.0 s | 0.43 s | 0.53 s |
| Generation, per token | 346 ms | 99 ms | 46 ms | 27 ms |

Generation is limited by memory bandwidth (every weight is read once per token), which is what Q8_0 helps with.

### Phase 13: tiles

Phase 13 made the kernels faster again, with the same rule. Every output still owns its accumulator and its order.
Only the way outputs are grouped onto threads, SIMD lanes and registers changed, so all golden hashes stayed the
same.

- **Register tiles in `linear`.** A panel's outputs are computed for several rows of `x` at once, so each weight
  vector that is loaded and converted to double serves several rows. AVX2 tiles are up to 6 rows by 8 outputs
  (12 accumulators in the 16 vector registers). Tiles of 1 or 2 rows use the whole panel, so there are enough
  independent accumulators to hide the FMA latency. SSE2 tiles are 4 outputs wide, NEON tiles 8. `x` is converted
  to double once per call, so the inner loop only broadcasts. The tile height depends only on the kernel and on
  how many rows are left (full tiles of 6, then one tile of the remaining rows), never on the request.
- **Q8_0 tiles (AVX2).** Two rows by four outputs: each weight block that is loaded serves both rows, and the four
  block sums of a row are reduced together into one vector. The four outputs' doubles then advance in the lanes of
  one register with separate multiplies and adds, `acc + (d_x * d_w) * isum`, the exact scalar order (no FMA,
  because these products are not exact). The activations are quantised on the pool.
- **Tiled attention.** One task per (key/value head, tile of 4 query rows). The rows of a tile are the query heads
  of a group (GQA), or consecutive queries, in (query, head) order. Each SIMD lane holds a different row. The
  scores of the four rows are computed over the union of their visible keys, but a row's softmax and output only
  ever use its own keys, in its own order. `attention(..., reference=True)` keeps the row-by-row statement, and
  `tests/test_batch_invariance.py` compares the two on every path, including causal and sliding-window spans that
  differ within a tile.
- **Parallel elementwise.** Activations, `swiglu` and soft-capping run on the pool in fixed chunks of 8192 values.
- **A spinning pool.** Workers and the caller busy-wait for up to 200 µs before they sleep on a condition
  variable. A decode step runs hundreds of jobs that each take tens of microseconds, and waking sleeping threads
  for each cost more than the job.
- **`-O3` for real.** nanobind's CMake helper added `-Os` after `-O3`, which built every kernel for size, at up to
  half the speed. `NOMINSIZE` removes it. The optimisation level never changes a bit: the floating point flags
  (`-ffp-contract=off -fno-fast-math`, `/fp:precise`) fix the semantics.

Measured in the same cloud VM (4 vCPUs, Xeon, AVX2) with SmolLM2-135M, a 256-token prompt and single-token decode
steps, all threads, best or median of repeated runs:

| | float32, before | float32, Phase 13 | Q8_0, before | Q8_0, Phase 13 |
| --- | --- | --- | --- | --- |
| Prompt, tokens/s | 93 | 280 to 290 | 77 | 275 to 305 |
| Generation, ms per token | 54 to 59 | 33 to 34 | 38 to 41 | 20 to 22 |

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

## Q4_0 quantisation

`--quantize q4_0` stores the linear layers in 4 bits, half of Q8_0's weight memory. Blocks are the same 32 inputs
with a float32 scale: `d = amax / 7`, `q_i = round_half_even(x_i * (1 / d))` clamped to ±7, computed with the same
exact operations as Q8_0. Two values share a byte: byte `j` of a block holds `(q_j + 8) | ((q_(j+16) + 8) << 4)`,
llama.cpp's `Q4_0` order.

The kernel does not have an arithmetic of its own. It unpacks the weights of a panel of outputs back to `int8` (an
exact operation) and runs exactly the Q8_0 computation on them: activations quantised to Q8_0, exact `int32` block
sums, blocks combined in double in ascending order. So `linear_q4` equals `linear_q8` on the unpacked values bit for
bit on every SIMD path (`tests/test_batch_invariance.py`), the GPU uploads the unpacked values and runs its Q8_0
kernel, and Q4_0 has its own `system_fingerprint` (`:q4_0`) and golden hashes. Unpacking happens per group of four
outputs for every call, whatever the number of rows, so the strategy does not depend on the batch.

## Speculative decoding

`Transformer.forward_cached_last(tokens, cache, n)` returns the logits of the last `n` positions of one pass. Each
row is the bits a one-at-a-time decode of that prefix gives, because the tiles and attention compute every row on
its own (the property continuous batching relies on). `generation.py` uses it to check drafted tokens: it asks the
sampler for each position in turn, with that position's own random draw, and keeps a draft token only when it is
the chosen token. The KV cache rows of rejected tokens are dropped at the next call by the usual common-prefix
match. See [speculative decoding](api.md#speculative-decoding).

## Memory

`ModelFile` maps the file read-only. Once the `Transformer` has made the kernels' copy of a matrix (packed panels,
Q8_0/Q4_0 blocks or a GPU buffer) it calls `ModelFile.release(name)`, which tells the operating system with
`madvise(MADV_DONTNEED)` that those file pages are no longer needed, so only one copy of each weight stays
resident. Nothing is lost: a page that is touched again is read back from the file, byte for byte. On Windows,
which has no `madvise`, release does nothing. SmolLM2-135M after a chat: float32 1.09 GB to 607 MB, Q8_0 713 MB to
229 MB, Q4_0 163 MB.

## Portable determinism

Since Phase 10 the same inputs give the same bits on every supported machine, not only run to run on one. Nothing in
the kernels had to change for that; the rules that already made them reproducible also make them portable:

- **Only basic IEEE operations.** `+ - * /` and `sqrt` are correctly rounded on every IEEE 754 machine; everything
  else (`exp`, `sin`, `erf`, ...) is built from them in `math.hpp`. No libm, no `-ffast-math`, no compiler
  contraction (`-ffp-contract=off`, `/fp:precise`, NVRTC `--fmad=false`).
- **One order, double accumulators.** Every reduction runs in the order above on every path. SIMD lanes and GPU
  threads hold different outputs, never parts of one sum, so AVX2, SSE2, NEON, scalar and CUDA do the same
  operations per output. The only fused operation, the FMA in `linear`, multiplies two floats, which is exact in
  double, so it rounds exactly like the separate multiply and add. Q8_0's integer sums are exact in any order.
- **A known floating point environment** ([above](#floating-point-environment)) on every call.
- **Pinned text handling.** The tokenizer's Unicode normalisation, lower-casing and `\p{..}` classes come from
  Unicode 15.1 tables shipped in the package (`etalii_dllm.unicode`), not from the installed Python or `regex`.
  What still comes from them (`\s`, `str.isspace`, the case folding of the contraction patterns) is pinned by
  `tests/test_unicode.py`, so a version that changes it fails the tests rather than the output.

CI checks it: the golden hashes on Linux, Windows and macOS, and the real SmolLM2-135M and Qwen2.5-0.5B logits (float32
and Q8_0), greedy and sampled answers on all five release platforms (Linux x86-64 and arm64, Windows, macOS arm64
and Intel), each with the best and the portable SIMD path (`DLLM_ISA=portable` forces the latter). `dllm verify`
prints one fingerprint to compare two machines by hand.

Out of scope: accelerators without IEEE float64 (Apple GPUs through Metal, most NPUs) and fast paths whose
reduction order the hardware chooses (tensor cores, split-K, warp shuffles). The price of portability is the price
of determinism itself: double accumulators and a fixed order, which is why the GPU backend runs in double precision.

## GPU

`--device cuda` (or `DLLM_DEVICE=cuda`, `Transformer(..., device="cuda")`) runs the decoder on an NVIDIA GPU and
**gives the CPU's bits**: the same logits, the same tokens and the same `system_fingerprint`, float32 or Q8_0. All
golden hashes, including the real SmolLM2 and Qwen2.5 answers, are reproduced on the GPU (`tests/test_cuda.py`,
`tests/test_reference_models.py`). Measured on an RTX 4080 (Windows, driver 616) against a 32-thread CPU:

| | SmolLM2-135M, 64-token prompt | SmolLM2-135M, per token | Qwen2.5-0.5B, 64-token prompt | Qwen2.5-0.5B, per token |
| --- | --- | --- | --- | --- |
| CPU, float32 | 160 ms | 27 ms | 426 ms | 48 ms |
| GPU, float32 | 70 ms | 17 ms | 202 ms | 25 ms |
| CPU, Q8_0 | 165 ms | 22 ms | 353 ms | 25 ms |
| GPU, Q8_0 | 133 ms | 8 ms | 201 ms | 9 ms |

How it works:

- **The same order.** `cpp/cuda/kernels.cu` has one kernel per CPU kernel, and each GPU thread computes whole output
  elements with exactly the accumulation of the CPU kernel: `linear` sums over `k` ascending in double (one thread
  per output, weights stored transposed so a warp reads consecutive words), attention computes each score in its own
  thread and takes the maximum and the softmax total in one thread over the keys ascending, RMSNorm sums its squares
  in one thread. There are no atomics, no split-K, no warp-shuffle reductions and no tensor cores, so nothing depends
  on scheduling.
- **The same arithmetic.** The kernels are compiled at run time by NVRTC with `--fmad=false` (no multiply-add
  contraction), IEEE division and square root and no flush-to-zero. Double `+ - * /` and `sqrt` are then correctly
  rounded on the GPU as on the CPU. The one explicit `fma` (in `linear`) multiplies two floats, whose product is
  exact in double, so it rounds like the separate multiply and add. The transcendentals are `math.hpp` itself,
  compiled for the device (it is built from `+ - * /` only), never CUDA's `exp` or `sin`.
- **Q8_0** quantises the activations on the GPU with the same float operations as `quantize_q8_0` (explicit round
  half to even); the block sums are exact int32 (`dp4a`) and the blocks are combined in double, ascending.
- **Device-resident.** Weights are uploaded once; activations and the KV cache stay on the GPU; only the embedding
  rows go up and the logits come back. Operations are queued on one stream without waiting (the download of the
  logits waits), and callers on several threads take turns queueing, which cannot change any result.
- **Mixtures of experts.** The router logits come back to the host, which routes them with `moe_route`; each expert
  runs its rows on the GPU and its outputs are scaled and added on the host in increasing expert order, the CPU's
  float32 operations.
- **No CUDA at build time.** The extension loads the NVIDIA driver and NVRTC dynamically, so the same wheel builds
  and runs everywhere; without a GPU `--device cuda` fails with a message saying what is missing.
- **Checked in CI without a GPU.** Compiling needs no GPU, so the `cuda extra` CI job installs `.[dev,cuda]` on
  Linux and Windows and compiles the embedded kernels with NVRTC for every architecture from sm_50 to sm_90
  (`cuda.compile_kernels`, `tests/test_cuda.py`); on macOS the extra installs nothing. Bit-exactness itself is
  only checked where a GPU is present.

Double precision is slow on consumer GPUs (1/64 of float32 on the RTX 40 series), and the fixed per-output order
leaves a GPU partly idle when a layer has few outputs, so this is far from the speed of a float32 GPU engine. It is
still faster than the CPU, and much faster with Q8_0, where most of the work is exact integer arithmetic. Obvious
next steps: tiling `linear` through shared memory for prompts, CUDA graphs to cut launch overhead, and sampling on
the GPU.

## Gradients (`grad.hpp`)

The backward kernels used for [fine-tuning](training.md) follow the same rules: one double accumulator per output
element, a fixed order, one rounding.

- **`linear_backward`**: `dx[r, k] = sum_n dy[r, n] w[n, k]` (`n` ascending), `dw[n, k] = sum_r dy[r, n] x[r, k]`
  and `db[n] = sum_r dy[r, n]` (`r` ascending).
- **`rms_norm_backward`**: `inv` recomputed as in the forward kernel; `dx_i = inv g_i dy_i - x_i inv^3 / dim *
  sum_j g_j dy_j x_j` (`j` ascending) with `g = w`, or `g = 1 + w` in double with `add_unit_offset` (Gemma);
  `dw_i = sum_r dy x inv` over rows ascending.
- **`silu_backward`**: `dy * s (1 + x (1 - s))` with `s = sigmoid(x)`, in double.
- **`gelu_tanh_backward`**: with `u = sqrt(2/pi) (x + 0.044715 x^3)` and `t = tanh(u)` (`dllm` tanh),
  `dy * (0.5 (1 + t) + 0.5 x (1 - t^2) sqrt(2/pi) (1 + 3 * 0.044715 x^2))`, in double.
- **`softcap_backward`**: `dy * (1 - t^2)` with `t = tanh(x / cap)`, in double (logit soft-capping).
- **`rope(..., inverse=True)`**: the same rotation with `sin` negated (exactly), i.e. the transpose.
- **`attention_backward`**: per (query, head), in query-then-head order, the probabilities are recomputed exactly as
  in the forward pass; `dp_j = sum_i dout_i v_ji`, `D = sum_j p_j dp_j`, `ds_j = p_j (dp_j - D)` (times
  `1 - tanh(s_j / softcap)^2` of the scaled score `s_j` when a soft-cap is set),
  `dq = scale sum_j ds_j k_j`; `dk_j` and `dv_j` accumulate `scale ds_j q` and `p_j dout` in double over queries
  ascending, then heads ascending.
- **`biased_attention_backward`** (DeBERTa): the same over every key with the scores of `biased_attention`
  (`(q . k_j + bias_j) * scale`, no soft-cap); `g_j = p_j (dp_j - D) * scale` gives `dq = sum_j g_j k_j`, the
  accumulations `dk_j += g_j q` and `dv_j += p_j dout`, and `dbias[h, t, j] = g_j` rounded once. One thread, the
  (query, head) pairs in query-then-head order, so the accumulations have one order whatever the thread count.
- **`cross_entropy`**: `logsumexp = max + log(sum_j exp(l_j - max))` (`j` ascending, `dllm` exp/log); the loss is
  summed over rows ascending; `dlogits = (softmax - onehot) * scale`. Negative targets are skipped.
- **`embedding_backward`**: rows grouped per token id by a counting sort that keeps positions ascending, then
  summed in double.
- **`layer_norm_backward`** (encoders): the mean and `inv_std` recomputed as in the forward kernel; with
  `xhat_i = (x_i - mean) inv_std` and `g_i = w_i dy_i`, `dx_i = inv_std (g_i - (sum_j g_j) / dim - xhat_i
  (sum_j g_j xhat_j) / dim)` (`j` ascending, all in double); `dw_i = sum_r dy xhat` and `db_i = sum_r dy` over rows
  ascending.
- **`gelu_backward`** (the exact GELU): `dy * (0.5 erfc(-x / sqrt(2)) + x exp(-x^2 / 2) / sqrt(2 pi))` with the
  portable `erfc` and `exp`, in double.
- **`moe_route_backward`**: per row, in row order, the probabilities `p` are recomputed exactly as `moe_route` does;
  then in double `S = sum_j p[i_j]` and `C = sum_j dw_j p[i_j]` over the chosen experts in rank order (renormalised
  routing only), `dp_e` = the optional gradient of the probabilities plus `dw_j / S - C / S^2` (renormalised) or
  `dw_j` for a chosen expert `e = i_j`, `D = sum_e p_e dp_e` (`e` ascending) and `dlogits_e = float(p_e (dp_e - D))`.
  The choice of the top `k` has no gradient.
- **`sum_squares`** and **`adamw_step`**: see [training](training.md).

## Interpretability kernels

`interp.hpp` holds the kernels behind the [interpretability tools](interpretability.md), with the same rules: one
double accumulator per output element, a fixed order, one rounding; threads split outputs.

- **`attention_weights`**: the probabilities of [attention](#attention), written out instead of applied: steps 1 and 2
  above in the same order, then `p_j * (1 / Z)` rounded once to float. Keys a query cannot see (causal horizon,
  window) get 0. Output `[q_len, heads, kv_len]`.
- **`cosine_similarity`**: for every row `r` of `matrix[rows, dim]`, `(sum_i m_ri q_i) / (sqrt(sum_i m_ri^2) *
  sqrt(sum_i q_i^2))`, each sum over `i` ascending in double; a zero row or query gives 0.
- **`column_mean`**: `(sum_r x[r, c]) / rows`, rows ascending in double.
- **`cholesky_solve`**: solves `a x = b` for a symmetric positive definite `a` (lower triangle read), single-threaded
  in double: the factor `L` column by column with every inner sum ascending, then forward and back substitution
  row by row, rounded to float once at the end (for [model editing](interpretability.md)).
