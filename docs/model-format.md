# The model.dllm format

`model.dllm` is the single file an imported model lives in: weights in the layout our kernels read, plus everything
needed to run and attribute the model. `dllm import` writes it (`src/etalii_dllm/importing/`); `ModelFile`
(`src/etalii_dllm/modelfile.py`) memory-maps it. `dllm inspect model.dllm` prints its header.

## Layout (format version 1)

| Offset | Size | Content |
| --- | --- | --- |
| 0 | 4 | Magic `DLLM` |
| 4 | 4 | Format version, uint32 little-endian (`1`) |
| 8 | 8 | Header length `H`, uint64 little-endian |
| 16 | `H` | Header: canonical JSON (UTF-8, sorted keys, no whitespace, no NaN), padded with spaces so the data starts at a multiple of 64 |
| 16 + `H` | rest | Tensor data |

Tensors are C-order little-endian float32. Each starts at a multiple of 64 bytes from the start of the data
section (so a memory map gives kernel-ready, cache-line aligned arrays) and is followed by zero bytes up to the next
multiple of 64. They are stored in natural order of their names (`layers.2` before `layers.10`).

**Fingerprint.** The SHA-256 of the whole data section, padding included, is the model's fingerprint and its
`system_fingerprint`. It is written into the header's `fingerprint` field (fixed width, patched in after the data is
hashed), and `ModelFile` re-hashes the data on open unless `verify=False`.

**Reproducibility.** Nothing in the file comes from a clock, the environment or iteration order, so importing the
same source twice gives byte-identical files (`tests/test_import.py`).

## Header

| Key | Content |
| --- | --- |
| `format`, `format_version` | `"dllm"`, `1` |
| `architecture` | `TransformerConfig` (`src/etalii_dllm/architecture.py`): family (`gemma2`, `gemma3`, `granite`, `llama`, `mistral`, `olmo2`, `phi3`, `qwen2`, `qwen3`), sizes, heads and KV heads, head dim, context length, RMSNorm epsilon, RoPE theta and scaling, attention bias, `qk_norm` (written only when true), `qk_norm_scope` (`head` or `all`), `norm_placement` (`pre`, `post` or `sandwich`) and `norm_unit_offset` (RMSNorms scale by `1 + weight`, Gemma), each written only when not the default, `activation` (`silu` or `gelu_tanh`), `local_rope_theta` (the RoPE base of the sliding-window layers, Gemma 3; written only when set), the Granite multipliers `embedding_multiplier`, `attention_multiplier` (the attention score scale), `residual_multiplier` and `logits_scaling` (each written only when not the plain value), tied embeddings, BOS/EOS token ids, `sliding_window` and `sliding_window_layers` (written only when set; no layer list means every layer), `rotary_dim` (partial rotary, written only when set), `attention_softcap` and `logits_softcap` (Gemma 2 soft-capping, written only when set) |
| `tensors` | List of `{name, shape, dtype: "F32", offset, nbytes, source_dtype}`; `source_dtype` is what the source stored (`BF16`, `F16`, `Q8_0`, ...) |
| `fingerprint` | SHA-256 of the data section (hex) |
| `source` | `format` (`safetensors`/`gguf`), `repository` and `revision` (the commit hash for `hf:` imports), `url` when known, and `files`: path, SHA-256 and size of every source file read |
| `licence` | `spdx` id, full licence `text`, `attribution` line, `redistributable` (true only for Apache-2.0 and MIT), optional `link` |
| `tokenizer` | Hugging Face: `tokenizer.json`, `tokenizer_config.json` and `special_tokens_map.json` verbatim. GGUF: the `tokenizer.*` metadata. The BPE tokenizer (#14) reads this |
| `chat_template` | The model's Jinja chat template, or `null` |
| `fine_tuning` | Only in models written by `dllm finetune`: base model fingerprint, data fingerprint, run settings, steps completed and final loss (see [training](training.md)) |
| `adapter` | Only in models written by `dllm import ADAPTER --base BASE`: the base model fingerprint, the LoRA settings (`rank`, `alpha`, `targets`, `rslora`), the adapter's licence and its source files (see [training](training.md#lora-adapters)) |
| `embedding` | Only in models imported from sentence-transformers: `pooling` (`last_token` or `mean`), `normalize`, `prompts` (name to the text put in front of the input, such as `query`) and `default_prompt_name` (see [retrieval](retrieval.md#embedding-models)) |
| `edits` | Only in edited models (`dllm edit`): a list, oldest first, of the edits applied to the weights: method (`rome`), layer, prompt, subject, target, contexts, base model fingerprint, covariance and optimiser settings, the norm of the value change and the target probability before and after (see [interpretability](interpretability.md#model-editing-rome)) |

## Tensor names

Every import maps onto one naming scheme, and weight matrices use the `[out, in]` layout of `linear`:

| Name | Shape |
| --- | --- |
| `token_embedding.weight` | `[vocab, hidden]` |
| `layers.N.attention_norm.weight` | `[hidden]` |
| `layers.N.attention.{q,k,v}.weight` | `[heads * head_dim, hidden]`, `[kv_heads * head_dim, hidden]` |
| `layers.N.attention.{q,k,v}.bias` | Qwen2 only |
| `layers.N.attention.{q,k}_norm.weight` | `[head_dim]`, Qwen3: RMSNorm over each query and key head before RoPE; OLMo 2: `[heads * head_dim]` and `[kv_heads * head_dim]`, over the whole projection |
| `layers.N.{attention,mlp}_post_norm.weight` | `[hidden]`, OLMo 2 only (it has no `attention_norm`/`mlp_norm`): RMSNorm on the output of attention and of the MLP, before the residual add |
| `layers.N.attention.o.weight` | `[hidden, heads * head_dim]` |
| `layers.N.mlp_norm.weight` | `[hidden]` |
| `layers.N.mlp.{gate,up}.weight`, `layers.N.mlp.down.weight` | `[intermediate, hidden]`, `[hidden, intermediate]` |
| `final_norm.weight` | `[hidden]` |
| `lm_head.weight` | `[vocab, hidden]`, absent when embeddings are tied |

Rotary embeddings pair dimensions `(i, i + head_dim/2)` as in Hugging Face checkpoints. llama.cpp permutes the Q and
K rows of Llama models to pair `(2i, 2i+1)`; the importer undoes that reindexing, so an F32/F16/BF16 GGUF and the
original checkpoint import to the same bytes and the same fingerprint.

## Conversion rules

- **Lossless widening.** BF16 is the upper half of a float32 and every float16 is a float32, so F16/BF16 → F32
  is exact. F64 and integer weights are refused.
- **Quantised GGUF.** `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q4_K`, `Q5_K` and `Q6_K` are dequantised with
  elementwise float32 operations in llama.cpp's order; tests check the result is bit-identical to `gguf-py`. The
  import is deterministic, but a quantised source is of course only as precise as its quantisation.
  Running quantised (`--quantize q8_0` or `q4_0`) requantises the float32 tensors at load time; the file itself
  stays float32, so one file serves every precision.
- **Fail loudly.** Unknown tensors, unsupported families (anything but Gemma 2, Gemma 3 text, Granite, Llama, Mistral, OLMo 2, Phi-3, Qwen2 and Qwen3),
  activations other than SiLU and GELU (tanh), MLP biases and RoPE scaling other than `linear`/`llama3`/`longrope` stop the import. Sliding-window
  attention comes from `sliding_window` (Mistral: every layer; Qwen2/Qwen3 with `use_sliding_window`: the layers from
  `max_window_layers` on; a `layer_types` list names them explicitly); a window at least as long as the context is
  dropped.
- **Gemma 3.** Text-only checkpoints (`model_type` `gemma3_text`) map to family `gemma3`: RMSNorm before and after
  attention and the MLP (`sandwich`, from `input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`
  and `post_feedforward_layernorm`), every norm scaling by `1 + weight` (the stored weights are Gemma's own), GELU
  (tanh) gating, QK-norm per head, embeddings times `sqrt(hidden_size)` and attention scores times
  `query_pre_attn_scalar ** -0.5` (as `embedding_multiplier` and `attention_multiplier`). The sliding-window layers
  (from `layer_types`, or every layer but each `sliding_window_pattern`-th) rotate with `rope_local_base_freq` and no
  scaling; the full-attention layers with `rope_theta` and `rope_scaling`.
- **Gemma 2.** Family `gemma2` is Gemma 3 without QK-norm and with a single RoPE base: the same sandwich norms,
  `1 + weight` scaling, GELU (tanh), embedding and attention multipliers. Every other layer slides (from
  `layer_types`, or the even layers). `attn_logit_softcapping` becomes `attention_softcap` (each scaled attention
  score `s` becomes `cap * tanh(s / cap)` before the softmax) and `final_logit_softcapping` becomes `logits_softcap`
  (the same on the output logits).
- **Phi-3.** The fused `qkv_proj` and `gate_up_proj` are split into our q/k/v and gate/up tensors (rows stacked in
  that order). `partial_rotary_factor` becomes `rotary_dim`. LongRoPE (`longrope`, or Phi-3's older `su`/`yarn`) is
  stored with its `short_factor`, `long_factor`, `original_max_position_embeddings` and `attention_factor` (computed
  as transformers does when absent). transformers switches to the long factors once a sequence outgrows the original
  context, which would make earlier tokens' output depend on the sequence length; dllm always uses the short factors
  and caps `context_length` at the original context, so it matches transformers everywhere it runs. The attention
  factor scales the rotated query and key dimensions; the decoder folds it into the rows of the q/k projections when
  the model loads.
- **Licences.** Apache-2.0 and MIT import directly. Anything else needs `--accept-licence` and is recorded as not
  redistributable. A source with no stated licence needs `--licence`; a licence with no text in the source and no
  standard text bundled needs `--licence-file`.
