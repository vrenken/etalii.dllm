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
- No kernel switches strategy on input size. There is no threading yet; when it comes (Phase 6) it will partition
  outputs, not reductions.
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

## Linear algebra (`nn.hpp`)

- **`linear(x, weight, bias)`**: `x[..., in] @ weight[out, in]^T + bias` (PyTorch `nn.Linear` layout; leading
  dimensions are flattened into rows). Output `[r, n]` is `sum_k x[r, k] * w[n, k]` over `k` ascending, plus
  `bias[n]` in double, rounded once. Tiles of 16 rows by 64 columns only order the visits.
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
