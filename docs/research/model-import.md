# Importing existing open-weight models

EtAlii.Dllm will not train its large statistical base from scratch. Training even a small useful model needs far
more data and compute than this project should spend. Instead we **import the weights of existing small open-weight
LLMs, convert them to our own format and run them deterministically**. Our value is the deterministic engine, not
new weights.

## Candidate models

Small (≤ 2B parameters) so they run on a CPU and fit in CI caches. Licences are as understood at the time of writing
(2026-09); **re-check the licence on the model card at import time**. The importer records it in the converted file.

| Model | Sizes | Architecture | Licence | Notes |
| --- | --- | --- | --- | --- |
| SmolLM2 (Hugging Face) | 135M, 360M, 1.7B | Llama-style | Apache 2.0 | **First target.** Tiny, permissive, instruct variants, standard safetensors + `tokenizer.json` |
| Qwen2.5 | 0.5B, 1.5B | Qwen2 (Llama-like, QKV bias) | Apache 2.0 (the 3B and 72B sizes use a different licence) | Strong for its size, multilingual, tool-calling chat template |
| Qwen3 | 0.6B, 1.7B | Qwen3 | Apache 2.0 | Newer; thinking/non-thinking modes |
| TinyLlama | 1.1B | Llama 2 | Apache 2.0 | Well known reference model |
| OLMo 2 (AI2) | 1B | OLMo | Apache 2.0 | Training data and code are open too, useful for research comparisons |
| Phi-3.5-mini / Phi-4-mini (Microsoft) | 3.8B | Phi-3 | MIT | Larger; good reasoning for the size |
| GPT-2 (OpenAI) | 124M–1.5B | GPT-2 | MIT | Classic baseline; simplest architecture to implement first |
| Llama 3.2 (Meta) | 1B, 3B | Llama 3 | Llama 3.2 Community Licence | Usable but with attribution, acceptable-use and scale conditions; avoid as a default |
| Gemma 3 (Google) | 270M, 1B | Gemma 3 | Gemma Terms of Use | Custom terms with a prohibited-use policy; avoid as a default |

Default: **Apache 2.0 or MIT models only**, so converted weights can be redistributed with attribution and a copy of
the licence. Models with custom licences are opt-in and never redistributed by this project.

## Conversion pipeline (planned, roadmap phase 2)

```
Hugging Face repo (config.json, tokenizer.json, *.safetensors)  ─┐
GGUF file (llama.cpp / Ollama)                                   ─┴─►  dllm import  ─►  model.dllm
```

1. **Read.** safetensors is a JSON header followed by raw little-endian tensors, easy to parse in C# with no
   dependencies. GGUF (key/value metadata + tensors, including quantised block formats) is the second reader.
2. **Map.** Translate the source architecture (`config.json`: hidden size, heads, KV heads, RoPE θ and scaling, norm
   epsilon, tied embeddings) to our transformer description. Unsupported features fail the import loudly.
3. **Convert.** bf16 → fp32 and fp16 → fp32 are exact, so conversion loses nothing. Any quantisation we add uses
   round-to-nearest-even with a fixed block layout, so converting the same source twice gives the same bytes.
4. **Write `model.dllm`.** Our own container: a header with architecture, tokenizer, chat template, **source repo +
   revision, licence text and attribution**, then tensors in the layout our kernels want (pre-transposed, aligned).
   The SHA-256 of the tensor data becomes the model's `system_fingerprint`.
5. **Verify.** Compare our logits for a set of prompts against the reference implementation (Hugging Face
   transformers, or llama.cpp) within a tolerance. Bit equality with them is not expected, because they do not use
   our reduction order. Then pin our own outputs as golden hashes.

The same importer brings in the tokenizer (`tokenizer.json` BPE, byte fallback, special tokens) and the model's
Jinja chat template, which also defines its tool-calling format and so feeds the OpenAI/MCP tool support.

## Beyond inference

- **Fine-tuning** (optional, later) on top of imported weights with deterministic training, instead of training
  from zero.
- **Distillation data**: an imported model can generate deterministic training data or logits for our own smaller
  experimental models, with the source model's licence applying to that data.

## Practical constraints

- Weights never go into git (`*.safetensors`, `*.gguf` are ignored). Converted models live in a local cache; CI uses
  a small cached model (SmolLM2-135M, ~270 MB in bf16) or synthetic weights for unit tests.
- The Claude Code cloud environment's default network policy blocks `huggingface.co` and its CDN hosts. Downloading
  models there requires adding those hosts to the environment's allowed domains, or supplying the files another way.
