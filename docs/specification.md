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
  logits and a greedy, a sampled and a fully controlled (penalties, min-p, logit bias) answer bit for bit (see [checking an implementation](#checking-an-implementation)).
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
  - `longrope`: `f / short_factor[i]`, or `f / long_factor[i]` when `factor_set` is `long`.
  - `yarn`: with `d = rotary_dim`, `c(r) = d * log(original / (r * 2π)) / (2 * log(theta))` (`2π` is the double
    product `2 * π`), `low = c(beta_fast)` and `high = c(beta_slow)`, rounded down and up with `truncate`, then
    `low = max(low, 0)`, `high = min(high, d - 1)`, and `high + 0.001` when the two are equal. With
    `ramp = min(max((i - low) / (high - low), 0), 1)`, the frequency is `f / factor * ramp + f * (1 - ramp)`.
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

**Logit adjustments.** Before any sampling (greedy included), the float32 logits change in this order. Each token
is adjusted on its own, so the order tokens are visited in does not matter:

1. Logit bias: `x = f32(x + f32(bias))` for every `(token, bias)` pair with `token` below the vocabulary size.
2. Repetition penalty `r` (when `r != 1` and `repeat_last_n != 0`): for every distinct token among the last
   `repeat_last_n` tokens of prompt plus output (all of them when `repeat_last_n` is -1), `x = f32(x / f32(r))` if
   `x > 0`, else `x = f32(x * f32(r))`.
3. Frequency and presence penalties: for every token the output (not the prompt) contains `c > 0` times,
   `x = f32(x - f32(c * frequency_penalty + presence_penalty))`, the penalty computed in double.
4. Watermark (when a key is set; [watermarks](watermarks.md)): with `K` the first 8 bytes (little-endian) of
   `SHA-256("dllm-watermark/1\0" + key)`, `mix` the SplitMix64 finalizer and `G = 0x9E3779B97F4A7C15`, the seed after
   the previous token `p` (the last prompt token for the first output token, -1 without one) is
   `s = mix(K + (p + 1) * G)`, all modulo 2^64. Every token `t` with `mix(s + (t + 1) * G) < floor(gamma * 2^64)` gets
   `x = f32(x + f32(delta))`.

Log-probabilities reported with an answer come from the unadjusted logits.

**Sampler.**

- With temperature 0, the token is `argmax(logits)`.
- Otherwise the sampler, seeded with `seed mod 2^64`, makes one draw per token:
  1. `z = f32(logits / temperature)` (a float32 division) and `p = softmax(z)`.
  2. Order the candidates by probability descending, then id ascending.
  3. Keep the first `top_k` candidates (all when `top_k` is 0).
  4. With `top_p < 1`, keep the shortest prefix whose running double sum of `p` reaches `top_p`.
  5. With `min_p > 0`, cut the kept prefix before the first candidate (after the first) whose `p` is below
     `min_p * p[first]` (a double product).
  6. `total` is the kept probabilities summed in order. With `u = next_double() * total`, the token is the first
     candidate whose running sum exceeds `u`, or the last kept candidate if none does.
- Constrained decoding restricts the adjusted logits to the allowed ids (ascending) before these steps.
- **Several choices.** Choice `i` of a request for `n` is the request with the seed `(seed + i) mod 2^64`; nothing
  else differs, so each choice is exactly the answer a single request with that seed gets.

**Regular expressions.** A regex constraint allows exactly the outputs whose UTF-8 bytes, in full, are the
encoding of a string the pattern matches, where `\d`, `\w` and `\s` have their ASCII meanings (`[0-9]`,
`[A-Za-z0-9_]`, `[ \t\n\r\f\v]`), `.` is any code point but `\n`, and classes hold code points (never
surrogates). The supported syntax is listed in `etalii_dllm.regexp`. A token is allowed when its bytes extend the
output to a prefix of such an encoding; a stop token is allowed when the output so far is a full match.

**JSON schemas.** A schema constraint allows a subset of the JSON texts the schema accepts: properties in schema
order, constrained strings without escapes, constrained numbers as decimal texts without an exponent. Before
compiling, `allOf`, `not` and `if`/`then`/`else` are rewritten into equivalent schemas as `etalii_dllm.schema_algebra`
describes (keys and branches in the order the schema gives them); a schema that cannot be rewritten exactly is
refused rather than approximated.

**Grammars.** A GBNF grammar constraint allows exactly the outputs whose UTF-8 bytes are the encoding of a string
the `root` rule derives, where literals and classes hold code points (never surrogates) and `.` is any code point.
Before decoding, alternatives that derive no finite string are removed; grammars with left recursion are refused.
The supported syntax is listed in `etalii_dllm.gbnf`. Tokens and stop tokens are allowed as for regular expressions.

## 5. The decoder

The model file ([format](model-format.md)) gives a `TransformerConfig` and float32 tensors. Two weight changes are
made once at load time, as float32 multiplications:

- Granite multiplies `attention.o.weight` and `mlp.down.weight` by `residual_multiplier`.
- LongRoPE and YaRN multiply the rows of `attention.q`/`k` (weights and biases) that feed the rotated dimensions of
  each head by `attention_factor`. A model with QK-norm multiplies the entries of `attention.q_norm`/`k_norm` that
  feed the rotated dimensions instead (the norm comes after the projection).

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

### The context window

`W` is the model's `context_length`. The sequence (prompt and answer so far) never holds more than `W` tokens:

- A prompt of `W` tokens or more is refused (the API's `context_length_exceeded` error).
- When the sequence reaches `W` tokens, the answer ends with finish reason `length` (`overflow = stop`, the default),
  or the window **rolls** (`overflow = roll`): the sequence becomes its first 4 tokens followed by its last `floor(W /
  2)` tokens, the cache is emptied and the kept tokens are fed afresh from position 0. The next token is sampled from
  the last position's logits, so every token after a roll is exactly what a fresh generation over the kept tokens
  would choose. The sampler is not reset: the penalties keep counting over the whole history (prompt and every answer
  token, rolled away or not). A window of 8 tokens or fewer cannot roll.
- Truncation (`truncation = auto`) happens before tokenization, on messages: while the rendered prompt holds `W`
  tokens or more, the earliest message that is neither a system message nor the last message is dropped, together with
  the tool messages directly after it. When only system messages and the last message are left, the prompt is refused
  as above.

### Reasoning

A thinking model is one whose chat template contains `<think>`. For its chat answers (not raw prompts, structured
output or forced tool calls, which constrain the answer from its first token), the output text is split by fixed
text matching (the text is the decoded output bytes, without a trailing incomplete UTF-8 sequence):

- The output **starts in thinking** when the prompt, without trailing whitespace, ends with `<think>`. Otherwise it
  thinks when its text, without leading whitespace, starts with `<think>`.
- The thinking runs to the first `</think>`. The **reasoning** is the text in between without surrounding whitespace;
  the **answer** is the rest without leading whitespace. When `</think>` never comes, there is no answer. An output
  that does not think is all answer, unchanged.
- **Switch.** `thinking = true/false` renders the template with `enable_thinking` set. When thinking is off and that
  gives the same prompt as thinking on (the template ignores it), the engine appends `\n</think>\n\n` to a prompt
  that starts the output in thinking, else `<think>\n\n</think>\n\n`.
- **Budget.** A generated token counts toward the budget when the block is open after it and was not closed before
  it: from the first output token when the output starts in thinking, else from the token that completes `<think>`,
  to the token that completes `</think>`, included. Before a token is sampled, when the block is open and has counted
  `max_reasoning_tokens` tokens, the engine appends the tokens of `\n</think>\n\n` (without the `\n` in front when
  the text ends with one), encoded by the model's tokenizer, as output tokens: the sampler records them (penalties)
  and they count toward `max_tokens` and the context window, whichever ends first. Sampling then continues.
  Usage reports the counted tokens as `reasoning_tokens`.

### Prompt scoring

The score of a text ([scoring](api.md#scoring)) tokenizes it as a prompt is tokenized. Token `i >= 1` gets
`log_softmax(z_i)[t_i]`, where `z_i` are the logits after tokens `0 .. i-1` (one forward pass over the text; the
decoder's logits do not depend on how the tokens were batched) and `log_softmax` is section 3's kernel; the first
token is not scored. Alternatives are ordered by log-probability descending, then id ascending. The log-likelihood is
the float32 log-probabilities summed in token order in double, and the perplexity `exp(-log_likelihood / n)` with the
portable `exp`, for the `n` scored tokens.

### Voting

A vote over `n` answers ([voting](api.md#voting)) samples choices `0 .. n-1` (seeds `seed + i`). Each answer is
normalised: with an `extract` regex, its last match (group 1 when the pattern has groups) is the answer, and no match
casts no vote; then NFKC and lower case from the pinned tables, runs of `[ \t\n\r\f\v]` become one space and the
ends are stripped; an empty result casts no vote. The winner is the answer with the most votes, ties going to the
answer whose first vote has the lowest choice index; the response is that choice's answer, or choice 0 when nobody
voted.

### Beam search

A beam search ([beam search](api.md#beam-search)) of width `W` starts from one empty hypothesis. A hypothesis's
log-likelihood is the sum in double, in token order, of the float32 `log_softmax` (section 3) values of its tokens,
including the stop token that ended it. Each step:

1. Every live hypothesis takes the `2W` tokens with the highest log-probability after its context (ties: lower id
   first) as candidates, with log-likelihood `parent + log_softmax[t]`.
2. All candidates are ranked by log-likelihood descending, then by token sequence ascending.
3. They are taken in that order until `W` live hypotheses are chosen. A candidate whose token is a stop token finishes
   (without that token) if its rank is below `W`, and is dropped otherwise; so does a candidate whose text now
   contains a stop sequence (with that token, the text cut before the stop sequence). Every other candidate becomes
   live.

The search ends when `W` hypotheses have finished or none is live, or after `max_tokens` steps (or when the context
window is full), when every live hypothesis finishes with `length`. Finished hypotheses are ranked by
`log_likelihood / exp(length_penalty * log(length))` (`length` counts the scored tokens; portable `exp` and `log` in
double) descending, then by token sequence ascending, and the first `n_best` are the answers. The sampler takes no
part.

### Token healing

With token healing ([token healing](api.md#token-healing)), if the prompt's tokens are non-empty and the last token
`t` decodes to a non-empty byte string `b` (special tokens decode to nothing), the context is the prompt without `t`
and `b` must be written first. While a remainder `r` of `b` is unwritten (starting with `r = b`), a step's allowed
tokens are those whose non-empty bytes `d` satisfy `d` is a prefix of `r` or `r` is a prefix of `d`; every other
token, stop tokens included, gets `-inf` before section 4's logit adjustments and sampling, exactly as a grammar mask
does. Writing `d` leaves `r` minus `d` when `d` is shorter, and otherwise ends healing, handing the bytes of `d`
after `r` to the answer's own constraint (a JSON schema, regex or grammar), if any. Stop sequences match only the
answer's text, which is its decoded bytes without the first `len(b)` bytes; its tokens, including the
healing ones, count as completion tokens.

### Length and stop controls

An answer ends at the first stop token: the model's own (none with `ignore_eos`) and the request's
`stop_token_ids`. While it has fewer than `min_tokens` tokens, the stop tokens are taken out of the step's choice, as
a grammar mask takes out tokens: the sampler of section 4 chooses among the remaining token ids (in ascending order),
and with a grammar among the grammar's allowed tokens without the stop tokens. A stop sequence ends the answer at its
first occurrence in the text; the text keeps it with `include_stop_str_in_output` and ends before it otherwise.

### Fill-in-the-middle

With a suffix ([fill-in-the-middle](api.md#fill-in-the-middle)), each FIM token is the token whose text is `<|name|>`
in the vocabulary, else the one whose text is `<name>`. The context is `fim_prefix`, the prompt's tokens,
`fim_suffix`, the suffix's tokens and `fim_middle`, the prompt and the suffix each tokenized on its own; a vocabulary
without all three refuses the request. The answer (the middle) is generated from that context as any answer is, and
it also ends, like at a stop token, at any of `fim_prefix`, `fim_suffix`, `fim_middle`, `fim_pad`, `file_sep`,
`repo_name` and `endoftext` the vocabulary holds.

### Guided decoding

A guided answer ([guided decoding](api.md#guided-decoding)) replaces each step's float32 logits `l` by a combination
before section 4's logit adjustments and sampling; the reported log-probabilities stay those of `l`. Every operation
below is float32 (each result rounded to float32) unless it says double.

- **Classifier-free guidance.** A second context, the negative prompt's tokens followed by the answer so far, runs on
  the same model; with its logits `n`, decoding uses `n + f32(scale) * (l - n)`.
- **Contrastive decoding.** The amateur model runs on the request's own context; with `p = softmax(l)` (section 3)
  and the amateur's logits `a`, a token with `p_t < alpha * max(p)` (both factors as double, the product in double)
  gets `-inf`, and every other token `f32(1 + beta) * l - f32(beta) * a`.
- **Ensembles.** Every model runs on the request's own context; decoding uses
  `f32(sum_i (w_i / W) * log_softmax(l_i))`, where `W` is the sum of the weights in double, `w_i / W` and each
  product are double, and the sum runs in double in model order, the served model first.

A guided answer ends when any of its contexts fills the window; it never rolls.

Not covered here, but just as fixed:

- tokenization: byte-level BPE and SentencePiece-style tokenizers, with Unicode handling pinned to version 15.1
  (`etalii_dllm.unicode`);
- the chat templates;
- receipts ([receipts](receipts.md)) and the model file format ([model format](model-format.md)).

## Checking an implementation

**On one machine.** `dllm --model m.dllm verify --reference` compares this machine's compiled kernels (whatever
SIMD path, thread count or GPU is in use) with the reference implementation. It checks the prompt's logits and a
greedy, a sampled and a controlled 16-token answer (every logit adjustment and `min_p` at once) and a greedy answer in a window just
longer than the prompt, which rolls it ([the context window](#the-context-window)), and a greedy answer that starts in a
thinking block with a budget of 2 tokens ([reasoning](#reasoning)), and the prompt's scores ([prompt scoring](#prompt-scoring)), and a sampled answer with classifier-free guidance ([guided decoding](#guided-decoding)), and the three ranked answers of a beam search of width 3 with their score bits ([beam search](#beam-search)), and prints `equal` for each part, or where the two first differ. CI runs it
for SmolLM2-135M on every release platform and SIMD path, in float32 and Q8_0.

**Elsewhere.** `dllm conformance write DIR` writes the vectors:

- **Layout.** `DIR/manifest.json` and one raw little-endian file per array.
- **Manifest.** Canonical JSON: sorted keys, no whitespace. It holds `format` (`etalii-dllm-conformance`), `version`
  (1) and `cases`.
- **Cases.** Each case has a `name`, a `kernel`, its `params`, and `inputs` and `outputs`, each mapping a name to
  `{file, dtype, shape, sha256}`.
- **Kernels covered.** The transcendentals, `linear` (float32 and quantised), `quantize`, `matmul`, `rms_norm`, the
  activations, `softmax`, `rope_inv_freq`, `rope`, `attention`, `random` (the `next_u64`, `next_double` and
  `next_gaussian` streams), `sample`, `sample_controls` (logit adjustments, the watermark included, and `min_p` over a sequence of steps that
  starts from a `prompt` input), `guided`, `contrasted` and `ensembled` (the [guided decoding](#guided-decoding) combinations), and `decoder`.
- **The decoder cases.** Each holds a config, its tensors (inputs named `tensor.<name>`), and the logits after each
  token fed one at a time.

An implementation conforms when it reproduces every output: the same shape and bits, where any NaN matches any NaN.
`dllm conformance check DIR [--implementation reference]` runs this build or the reference implementation against
vectors. The manifest's SHA-256 is the same on every machine and is a golden value
(`tests/golden_values.py::CONFORMANCE_MANIFEST_SHA256`).
