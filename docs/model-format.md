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
| `architecture` | `TransformerConfig` (`src/etalii_dllm/architecture.py`): family (`llama`, `mistral`, `olmo2`, `qwen2`, `qwen3`), sizes, heads and KV heads, head dim, context length, RMSNorm epsilon, RoPE theta and scaling, attention bias, `qk_norm` (written only when true), `qk_norm_scope` (`head` or `all`) and `norm_placement` (`pre` or `post`), both written only when not the default, tied embeddings, BOS/EOS token ids, `sliding_window` and `sliding_window_layers` (written only when set; no layer list means every layer) |
| `tensors` | List of `{name, shape, dtype: "F32", offset, nbytes, source_dtype}`; `source_dtype` is what the source stored (`BF16`, `F16`, `Q8_0`, ...) |
| `fingerprint` | SHA-256 of the data section (hex) |
| `source` | `format` (`safetensors`/`gguf`), `repository` and `revision` (the commit hash for `hf:` imports), `url` when known, and `files`: path, SHA-256 and size of every source file read |
| `licence` | `spdx` id, full licence `text`, `attribution` line, `redistributable` (true only for Apache-2.0 and MIT), optional `link` |
| `tokenizer` | Hugging Face: `tokenizer.json`, `tokenizer_config.json` and `special_tokens_map.json` verbatim. GGUF: the `tokenizer.*` metadata. The BPE tokenizer (#14) reads this |
| `chat_template` | The model's Jinja chat template, or `null` |
| `fine_tuning` | Only in models written by `dllm finetune`: base model fingerprint, data fingerprint, run settings, steps completed and final loss (see [training](training.md)) |
| `adapter` | Only in models written by `dllm import ADAPTER --base BASE`: the base model fingerprint, the LoRA settings (`rank`, `alpha`, `targets`, `rslora`), the adapter's licence and its source files (see [training](training.md#lora-adapters)) |

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
- **Fail loudly.** Unknown tensors, unsupported families (anything but Llama, Mistral, OLMo 2, Qwen2 and Qwen3), non-SiLU
  activations, MLP biases, partial rotary and RoPE scaling other than `linear`/`llama3` stop the import. Sliding-window
  attention comes from `sliding_window` (Mistral: every layer; Qwen2/Qwen3 with `use_sliding_window`: the layers from
  `max_window_layers` on; a `layer_types` list names them explicitly).
- **Licences.** Apache-2.0 and MIT import directly. Anything else needs `--accept-licence` and is recorded as not
  redistributable. A source with no stated licence needs `--licence`; a licence with no text in the source and no
  standard text bundled needs `--licence-file`.
