# Determinism specification

EtAlii.Dllm gives the same bits for the same weights, prompt and options on every run and on every supported
machine. This page says exactly which bits. It lists every arithmetic step from the model file's float32 tensors to
the chosen token, so that another implementation (in another language, on other hardware) can produce the same
output and prove it.

The specification is checked three ways:

- **`etalii_dllm.reference`** implements this page a second time, in Python and elementwise NumPy only, sharing no
  code with the C++ kernels. `tests/test_reference.py` requires it to give the kernels' bits for every kernel and
  every architecture option.
- **`dllm verify --reference`** runs a model through both implementations on the machine itself and compares the
  logits and a greedy and a sampled answer bit for bit (see [checking an implementation](#checking-an-implementation)).
- **`dllm conformance write DIR`** writes test vectors: inputs and exact outputs for every kernel, the random number
  generator, the sampler and two small decoders. A port checks itself against these files.

Where this page and the code disagree, it is a bug in one of them; the conformance vectors decide which bits are
meant. The kernels' design and speed are covered in [kernels](kernels.md). This page covers only the results.

## 1. Arithmetic

- **Formats.** Tensors are IEEE 754 binary32 (float32), little-endian. Accumulators and intermediate values are
  binary64 (double).
- **Environment.** Round to nearest, ties to even. Subnormal inputs and results are kept (no flush-to-zero or
  denormals-are-zero). Exceptions are masked, so overflow gives ±inf and invalid operations give NaN.
- **Operations.** Every `+ - * /` and `sqrt` is one correctly rounded IEEE operation. Expressions are evaluated
  left to right as written (`a + b + c` is `(a + b) + c`, `a * b * c` is `(a * b) * c`), and nothing is contracted:
  `a * b + c` is two roundings, never a fused multiply-add. `a - c` and `a + (-c)` are the same operation.
- **Conversions.** float32 to double is exact. double to float32 rounds to nearest even, written `f32(x)` below.
  The product of two float32 values is exact in double. Converting a double to an integer truncates toward zero,
  and is used only where the value fits.
- **Sums.** A "sum over `k` ascending" starts at `+0.0` and adds one term at a time in increasing `k`: `acc = acc +
  term_k`. Each output element has its own sum, so the result never depends on how many outputs, rows or tokens are
  computed together.
- **NaN.** Results that are NaN may carry any payload or sign. Comparisons with NaN are false.

## 2. Transcendentals

All functions take and return doubles. Constants are the double nearest to the decimal given. Fractions written
`1/n!` or `a/b` are the correctly rounded double quotient of the two doubles. `horner(p0; c1, …, cn; t)` means
`p = p0`, then `p = p * t + c_i` for `i = 1 … n`.

**`exp(x)`**

1. NaN gives NaN, `x > 709.78` gives +inf, `x < -745.2` gives +0.
2. `kd = x * 1.44269504088896338700`. `k = trunc(kd + 0.5)` if `kd >= 0`, else `trunc(kd - 0.5)`.
3. `r = (x - k * 6.93147180369123816490e-01) - k * 1.90821492927058770002e-10`.
4. `p = horner(1/13!; 1/12!, 1/11!, …, 1/2!, 1, 1; r)`.
5. Scale by `2^k`. While `k > 1023`: `p = p * 2^1023` and `k -= 1023`. While `k < -1022`: `p = p * 2^-1022` and
   `k += 1022`. The result is `p * 2^k`. Every power of two here is an exact normal double.

**`log(x)`**

1. NaN or `x < 0` gives NaN, `x == 0` (either sign) gives -inf, +inf gives +inf.
2. If `x < 2.2250738585072014e-308` (subnormal): `x = x * 2^54` and `e = -54`, else `e = 0`.
3. Take `x`'s biased exponent field `b`: `e += b - 1023`. Replace the exponent field with 1023 to get `m` in
   `[1, 2)`.
4. If `m >= 2.0 * 7.07106781186547524401e-01`: `m = m * 0.5` and `e += 1`.
5. `s = (m - 1) / (m + 1)`, `s2 = s * s`, `p = horner(1/25; 1/23, 1/21, …, 1/5, 1/3; s2)`.
6. `log_m = 2 * s + 2 * s * s2 * p`. The result is `(e * 6.93147180369123816490e-01 + log_m) + e *
   1.90821492927058770002e-10`, with `e` as a double.

**`sin(x)`, `cos(x)`**

1. NaN or ±inf gives NaN.
2. Reduce: `nd = x * 6.36619772367581382433e-01`.
   - If `|nd| < 2^52`, `n = trunc(nd + 0.5)` for `nd >= 0`, else `trunc(nd - 0.5)`.
   - Otherwise `n = nd`, which is already an integer.
3. `r = (((x - n * 1.57079632673412561417) - n * 6.07710050630396597660e-11) - n * 2.02226624871116645580e-21) - n *
   8.47842766036889956997e-32`.
4. Quadrant `q = n mod 4`, taken from `n` as a two's complement integer. For `|n| >= 2^62`, `q = 0`.
5. `S(r) = r + r * r2 * horner(-1/15!; 1/13!, -1/11!, 1/9!, -1/7!, 1/5!, -1/3!; r2)` and `C(r) = (1 - 0.5 * r2) +
   r2 * r2 * horner(1/16!; -1/14!, 1/12!, -1/10!, 1/8!, -1/6!, 1/4!; r2)`, with `r2 = r * r`.
6. By `q = 0, 1, 2, 3`: `sin` is `S, C, -S, -C`, and `cos` is `C, -S, -C, S`.

**`atan(x)`**

1. NaN gives NaN, and `atan(x) = -atan(-x)` for `x < 0`.
2. If `x > 1`, the result is `1.57079632679489661923 - atan(1 / x)`.
3. If `x > 0.26794919243112270647`: `offset = 0.52359877559829887308` and `r = (s * x - 1) / (s + x)` with `s =
   1.73205080756887729353`. Otherwise `offset = 0` and `r = x`.
4. `p = horner(-1/35; c_16, …, c_1; r2)` with `c_n = (n odd ? -1 : 1) / (2n + 1)` and `r2 = r * r`. The result is
   `offset + (r + r * r2 * p)`.

**`acos(x)`**: outside `[-1, 1]` (or NaN) gives NaN. `acos(-1) = 3.14159265358979323846`. Otherwise `2 * atan(sqrt((1
- x) / (1 + x)))`.

**`tanh(x)`**

1. NaN gives NaN. Let `a = |x|`.
2. If `a < 0.125`: `t = a + a * a2 * horner(-929569/638512875; 21844/6081075, -1382/155925, 62/2835, -17/315, 2/15,
   -1/3; a2)`, with `a2 = a * a`.
3. Else if `a > 22`: `t = 1`. Otherwise `t = 1 - 2 / (exp(2 * a) + 1)`.
4. The result is `-t` for `x < 0`, else `t`.

**`sigmoid(x)`**: for `x >= 0`, `1 / (1 + exp(-x))`. Otherwise, with `e = exp(x)`, `e / (1 + e)`.

**`erf(x)`, `erfc(x)`**

- `T(a)` for `a >= 2.5`: `k = a`, then `k = a + (0.5 * n) / k` for `n = 80, 79, …, 1`. `T(a) = exp(-a * a) *
  5.64189583547756286948e-01 / k`.
- `erf(x)`: NaN gives NaN. With `a = |x|`:
  - If `a < 2.5`: `a2 = a * a`, `term = total = a`, then for `n = 1 … 59`: `term = -term * a2 / n` and `total =
    total + term / (2n + 1)`. The result is `1.12837916709551257390 * total`.
  - If `a > 6`, the result is 1.
  - Otherwise it is `1 - T(a)`.
  - Negate the result for `x < 0`.
- `erfc(x)`: NaN gives NaN. For `x < 2.5` it is `1 - erf(x)`, for `x > 27.3` it is 0, otherwise `T(x)`.

## 3. Kernels

`x`, `w` and similar are float32 inputs, converted to double where they enter an expression.

| Kernel | Result |
| --- | --- |
| `linear(x[r, in], w[out, in], bias)` | `f32(sum_k x[r,k] * w[n,k] + bias[n])`: the sum over `k` ascending, then the bias added to it. |
| `matmul(a[m, k], b[k, n])` | `f32(sum_k a[i,k] * b[k,j])`, the sum over `k` ascending. |
| `rms_norm(x, w, eps)` | Per row: `ss = sum_i x_i * x_i` ascending, `inv = 1 / sqrt(ss / dim + eps)`, `f32(x_i * inv * s_i)` with `s_i = w_i`, or `1 + w_i` with a unit offset (Gemma), or 1 without weights. |
| `silu(x)` | `f32(x * sigmoid(x))`. |
| `gelu(x)` | `f32(0.5 * x * erfc(-x * 7.07106781186547524401e-01))`. |
| `gelu_tanh(x)` | `f32(0.5 * x * (1 + tanh(7.97884560802865355879e-01 * (x + 0.044715 * x * x * x))))`. |
| `swiglu(gate, up, act)` | `act(gate)` as above (rounded to float32), times `up`: one float32 multiplication. |
| `softcap(x, cap)` | `f32(cap * tanh(x / cap))`. |
| `argmax(x)` | The lowest index of the largest value (a later value must be strictly greater). |
| `softmax(l)` | `m = l[argmax(l)]`, `e_i = exp(l_i - m)`, `total = sum_i e_i` ascending, `f32(e_i * (1 / total))`. |
| `log_softmax(l)` | `f32((l_i - m) - log(total))`, with `total` as above. |

**Q8_0 and Q4_0.** Quantise a float32 row in blocks of 32 values:

1. `amax` is the largest `|x_i|` among the finite values, or 0.
2. `d = f32(amax / 127)`, or `/ 7` for Q4_0, a float32 division. `id = 1 / d` in float32, or 0 when `d == 0`.
3. Each value is `q_i = round_half_even(x_i * id)` (a float32 product), clamped to ±127 (±7 for Q4_0). A non-finite
   `x_i` gives 0.

`linear` with quantised weights quantises each activation row to Q8_0 the same way. Per output, over blocks `b`
ascending: `acc = acc + (d_x[b] * d_w[b]) * isum_b`, where `isum_b = sum_i q_x[i] * q_w[i]` is the block's exact
integer dot product. Then the bias is added and the result is `f32(acc)`. Only weights whose input size is a
multiple of 32 are quantised. The embedding stays float32, and the LM head is quantised.

**RoPE.**

- `rope_inv_freq(head_dim, theta, rotary_dim, scaling)`: frequency `i < rotary_dim / 2` is `exp(-(2i / rotary_dim) *
  log(theta))`, where `2i / rotary_dim` is a double division. Then `rope_scaling` is applied in double:
  - `linear`: `f / factor`.
  - `llama3`: with `wavelen = 2π / f`:
    - `f` itself when `wavelen < original / high`;
    - `f / factor` when `wavelen > original / low`;
    - otherwise `(1 - s) * f / factor + s * f`, with `s = (original / wavelen - low) / (high - low)`.
    - Defaults: `low = 1`, `high = 4`, `original = 8192`.
  - `longrope`: `f / short_factor[i]`.
- `rope(x[t, h, d], positions, inv_freq)`:
  - The angle is `position * inv_freq[i]`, in double.
  - With `c = cos(angle)` and `s = sin(angle)`, pair `(a, b)` becomes `(f32(a * c - b * s), f32(a * s + b * c))`.
  - Pairs are `(i, i + rotary_dim / 2)`, the layout every imported model uses (or `(2i, 2i + 1)` interleaved).
  - Dimensions from `rotary_dim` on are copied unchanged.

**Attention.**

- Shapes: `q[t, qh, d]`, `k[j, kvh, d]`, `v[j, kvh, dv]`. Query head `h` reads key/value head `h / (qh / kvh)`.
- Query `t` sits at position `q_offset + t` and sees keys `first … end - 1`:
  - `end = min(q_offset + t + 1, kv_len)` (or `kv_len` when not causal);
  - `first = end - window` when a window is set and `end > window`, else 0.
- For each query and head:
  1. `s_j = (sum_i q_i * k_ji) * scale`, the sum over `i` ascending. With a soft-cap, `s_j = cap * tanh(s_j / cap)`.
  2. `m = max_j s_j`, `p_j = exp(s_j - m)` and `Z = sum_j p_j`, over `j` ascending.
  3. `out_i = f32((sum_j p_j * v_ji) * (1 / Z))`, the sum over `j` ascending.

Keys before `first` and after `end` are never read, so a prefill and token-by-token decoding give the same bits.

## 4. Random numbers and sampling

**Generator.** xoshiro256\*\* seeded through SplitMix64, with 64-bit unsigned wrapping arithmetic.

- Seeding: the state words are four SplitMix64 outputs from the seed. Each output is `z = (state +=
  0x9E3779B97F4A7C15)`, `z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9`, `z = (z ^ (z >> 27)) * 0x94D049BB133111EB`, `z ^
  (z >> 31)`.
- `next_u64`: the result is `rotl(s1 * 5, 7) * 9`. Then the state advances: `t = s1 << 17`, `s2 ^= s0`, `s3 ^= s1`,
  `s1 ^= s2`, `s0 ^= s3`, `s2 ^= t`, `s3 = rotl(s3, 45)`.
- `next_double`: `(next_u64() >> 11) * 2^-53`.
- `next_gaussian`: the sum of 12 `next_double()` values (ascending, from 0), minus 6, rounded to float32.
- `fill_gaussian(seed, n)`: `n` values of `next_gaussian` from a generator seeded with `seed`. All test weights and
  conformance inputs come from it.

**Sampler.**

- With temperature 0, the token is `argmax(logits)`.
- Otherwise the sampler, seeded with `seed mod 2^64`, makes one draw per token:
  1. `z = f32(logits / temperature)` (a float32 division) and `p = softmax(z)`.
  2. Order the candidates by probability descending, then id ascending.
  3. Keep the first `top_k` candidates (all when `top_k` is 0).
  4. With `top_p < 1`, keep the shortest prefix whose running double sum of `p` reaches `top_p`.
  5. `total` is the kept probabilities summed in order. With `u = next_double() * total`, the token is the first
     candidate whose running sum exceeds `u`, or the last kept candidate if none does.
- Constrained decoding restricts the logits to the allowed ids (ascending) before these steps.

## 5. The decoder

The model file ([format](model-format.md)) gives a `TransformerConfig` and float32 tensors. Two weight changes are
made once at load time, as float32 multiplications:

- Granite multiplies `attention.o.weight` and `mlp.down.weight` by `residual_multiplier`.
- LongRoPE multiplies the rows of `attention.q`/`k` (weights and biases) that feed the rotated dimensions of each head
  by `attention_factor`.

For new tokens at positions `start …`, all in float32 unless stated:

1. `x` is the embedding rows, times `embedding_multiplier` when it is not 1.
2. For each layer:
   1. `h = norm(x)` with `attention_norm` (pre and sandwich placements), else `h = x`.
   2. `q, k, v = linear(h, W, bias)`.
      - OLMo 2 (`qk_norm_scope = all`) normalises `q` and `k` over the whole projection.
      - Qwen3 and Gemma 3 (`head`) normalise each head.
   3. RoPE `q` and `k` with the global or local (Gemma 3 sliding layers) frequencies. Append `k` and `v` to the
      cache.
   4. `a = attention(q, cache_k, cache_v)` with `scale = attention_multiplier` (default `1 / sqrt(head_dim)`, a double),
      the layer's window and the attention soft-cap.
   5. `o = linear(a, Wo)`, then `norm` with `attention_post_norm` (post and sandwich placements). `x = x + o`.
   6. `h = norm(x)` with `mlp_norm` (pre and sandwich placements), else `h = x`. `m = linear(swiglu(linear(h, Wgate),
      linear(h, Wup)), Wdown)`, then `mlp_post_norm` (post and sandwich placements). `x = x + m`.
   7. A steering vector for this layer is added: `x = x + v`.
3. `logits = linear(norm(x, final_norm), head)`, where `head` is the embedding when tied. Divide by `logits_scaling`
   in float32 when it is not 1, then soft-cap.

`norm` is `rms_norm` with the model's `rms_norm_eps`, and with `(1 + w)` for Gemma.

**Generation.** The prompt's tokens are fed and the next token is sampled from the last position's logits. A stop
token (the tokenizer's end of sequence or the model's own end ids) ends the answer and is not part of it.
Otherwise the token is appended and fed. The answer also ends after `max_tokens` tokens.

Not covered here, but just as fixed:

- tokenization: byte-level BPE and SentencePiece-style tokenizers, with Unicode handling pinned to version 15.1
  (`etalii_dllm.unicode`);
- the chat templates;
- receipts ([receipts](receipts.md)) and the model file format ([model format](model-format.md)).

## Checking an implementation

**On one machine.** `dllm --model m.dllm verify --reference` compares this machine's compiled kernels (whatever
SIMD path, thread count or GPU is in use) with the reference implementation. It checks the prompt's logits and a
greedy and a sampled 16-token answer, and prints `equal` for each part, or where the two first differ. CI runs it
for SmolLM2-135M on every release platform and SIMD path, in float32 and Q8_0.

**Elsewhere.** `dllm conformance write DIR` writes the vectors:

- **Layout.** `DIR/manifest.json` and one raw little-endian file per array.
- **Manifest.** Canonical JSON: sorted keys, no whitespace. It holds `format` (`etalii-dllm-conformance`), `version`
  (1) and `cases`.
- **Cases.** Each case has a `name`, a `kernel`, its `params`, and `inputs` and `outputs`, each mapping a name to
  `{file, dtype, shape, sha256}`.
- **Kernels covered.** The transcendentals, `linear` (float32 and quantised), `quantize`, `matmul`, `rms_norm`, the
  activations, `softmax`, `rope_inv_freq`, `rope`, `attention`, `random` (the `next_u64`, `next_double` and
  `next_gaussian` streams), `sample`, and `decoder`.
- **The decoder cases.** Each holds a config, its tensors (inputs named `tensor:<name>`), and the logits after each
  token fed one at a time.

An implementation conforms when it reproduces every output: the same shape and bits, where any NaN matches any NaN.
`dllm conformance check DIR [--implementation reference]` runs this build or the reference implementation against
vectors. The manifest's SHA-256 is the same on every machine and is a golden value
(`tests/golden_values.py::CONFORMANCE_MANIFEST_SHA256`).
