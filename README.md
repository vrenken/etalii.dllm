# EtAlii.Dllm

**A deterministic large language model, built from scratch in Python with C++ kernels.**

Mainstream LLM inference is not reproducible: the same prompt, the same weights and even the same `seed` can
give different answers from one request to the next. EtAlii.Dllm treats bit-exact reproducibility as a hard
requirement instead of a best-effort hint:

> Same weights + same prompt and context window ⇒ the same tokens, bit for bit, on every run and on every machine.

That holds regardless of server load, batch composition or thread scheduling, and across machines: x86-64 and arm64
CPUs on Linux, Windows and macOS, every SIMD path, and NVIDIA GPUs give the same bits, float32 or Q8_0 alike
(checked in CI with the real models on all five release platforms). `dllm verify` prints one fingerprint to
compare two machines. What it takes: [portable determinism](docs/kernels.md#portable-determinism).

The model speaks the protocols the rest of the ecosystem already uses (the OpenAI, Anthropic and Ollama HTTP APIs, with
streaming, tool calling and JSON-schema structured output, and the Model Context Protocol), so existing clients and
agents can use it without changes.

## Status

Phases 1 to 6 (kernels, importing models, fine-tuning, API parity, MCP, performance) done, including a CUDA backend
that reproduces the CPU bits on NVIDIA GPUs. The full pipeline (tokenizer →
model → sampler → CLI / HTTP API / MCP server) runs end to end and is proven bit-exact by CI, run to run and across Linux, Windows and macOS on
x86-64 and arm64 (Phase 10). The transformer building blocks (aligned `Tensor`, batch-invariant matmul,
RMSNorm, SiLU/GELU, RoPE and grouped-query attention, see [docs/kernels.md](docs/kernels.md)) are in place, and so are
`dllm import` (safetensors/GGUF to our [model.dllm](docs/model-format.md) format, with source and licence recorded)
a Llama/Qwen2/Qwen3 decoder whose KV cache cannot change its output, the models' own BPE tokenizers and chat templates, and
`--model` on every front end. The real SmolLM2-135M-Instruct, Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct and Qwen3-0.6B imports match Hugging Face
`transformers` (logits within about 2e-5, identical greedy answers), checked in CI. `dllm finetune` trains an imported model further with AdamW, reproducibly: equal
runs, and runs resumed from a checkpoint, write byte-identical models (see [docs/training.md](docs/training.md)).
The HTTP API speaks OpenAI, Anthropic and Ollama, with streaming, tool calling, JSON-schema structured output
(constrained decoding), logprobs and embeddings; streamed and non-streamed answers are identical (see
[docs/api.md](docs/api.md)).
The kernels run on all cores with SIMD (AVX2, SSE2 or NEON) and still give the same bits as the plain scalar loops,
whatever the thread count or batch, and `--quantize q8_0` runs 8-bit weights with exact integer accumulation (see
[docs/kernels.md](docs/kernels.md#threads-and-simd)).
Without a model file the engine falls back to a seeded placeholder (a bigram table).

## Which models can it run?

Not every model, but any model in a supported family. Determinism is a property of the engine, not of the weights:
`dllm import` only converts a model's weights, tokenizer and chat template to our [model.dllm](docs/model-format.md)
format, and every model the engine runs is deterministic. So a model becomes deterministic as soon as it imports.
What limits the choice is which architectures the engine implements.

| | Supported | Refused at import (for now) |
| --- | --- | --- |
| Architecture | Llama-style decoders (`model_type` `llama`: SmolLM2, TinyLlama, Llama 2, Llama 3.x, ...), Mistral (sliding-window attention), OLMo 2, IBM Granite 3.x (dense), Phi-3/Phi-3.5-mini/Phi-4-mini/Phi-4 (fused projections, partial rotary, LongRoPE up to the original context), Qwen2/Qwen2.5, dense Qwen3 | Gemma, Phi-1/Phi-2, Phi-3-small, mixture-of-experts models (Mixtral, Qwen3-MoE), GELU MLPs, MLP biases, YaRN RoPE scaling |
| Tokenizer | BPE from `tokenizer.json` (the model's own, with its Jinja chat template): byte-level (GPT-2, Llama 3, Qwen) and SentencePiece-style with `▁` and byte fallback (Llama 2, TinyLlama, Mistral, Phi-3) | Unigram and WordPiece tokenizers (T5, BERT), SentencePiece `tokenizer.model` files without a `tokenizer.json`, GGUF files with a SentencePiece (`llama`) vocabulary |
| Files | Hugging Face safetensors (F32/F16/BF16), GGUF (F32/F16/BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K), PEFT LoRA adapters | Other GGUF quantisations |
| Size | Weights are held in memory as float32 (or Q8_0 with `--quantize q8_0`), so memory and CPU speed set the limit; about 1.5B parameters is practical today | |

Verified against Hugging Face `transformers` in CI: SmolLM2-135M-Instruct, Qwen2.5-0.5B-Instruct,
Qwen2.5-1.5B-Instruct, Qwen3-0.6B, TinyLlama-1.1B-Chat and OLMo-2-1B-Instruct. Other models in these families should work but are not checked. An unsupported
model fails at `dllm import` with an error that names the missing feature, never with silently wrong output. A server
serves one model, chosen at start-up with `--model`/`DLLM_MODEL`.

## Quick start

New here? [Getting started](docs/getting-started.md) walks through installing, importing a real model and using it
from the command line, the HTTP API and MCP.

Pre-built wheels for Linux, Windows and macOS are attached to every [release](https://github.com/vrenken/etalii.dllm/releases)
(see [getting started](docs/getting-started.md#1-install)). Building from source requires Python 3.11+, CMake and
a C++17 compiler (the numeric kernels are a C++ extension built on install).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest

# Command line
dllm info
dllm generate --prompt "Hello" --temperature 0.8 --seed 7

# OpenAI-, Anthropic- and Ollama-compatible HTTP server on http://localhost:5080 (see docs/api.md)
dllm-server
curl http://localhost:5080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"dllm","messages":[{"role":"user","content":"Hi"}],"temperature":0.7,"seed":5}'

# MCP server over stdio, e.g. registered with Claude Code
claude mcp add dllm -- dllm-mcp

# Import an open-weight model to our own format (source, revision and licence are recorded in the file)
dllm import hf:HuggingFaceTB/SmolLM2-135M-Instruct -o smollm2-135m.dllm
dllm import ./qwen2.5-0.5b-instruct-q8_0.gguf -o qwen2.5-0.5b.dllm
dllm inspect smollm2-135m.dllm

# Run it: every front end takes --model (or the DLLM_MODEL environment variable) and uses the model's own
# tokenizer and chat template
dllm --model smollm2-135m.dllm chat "What is the capital of France?"
dllm-server --model smollm2-135m.dllm
claude mcp add dllm -- dllm-mcp --model /path/to/smollm2-135m.dllm

# Let the model call the tools of other MCP servers during a chat (mcp.json in the usual mcpServers format)
dllm --model qwen2.5-0.5b.dllm chat "What time is it in Amsterdam?" --mcp-config mcp.json

# Fine-tune it on your own text; the same data and options give a byte-identical model
dllm finetune smollm2-135m.dllm --data my-data.jsonl -o smollm2-135m-tuned.dllm --steps 50
```

Run the same request twice and compare: the responses, including `id` and `system_fingerprint`, are identical.

## Why Python and C++

Python is where the model ecosystem lives: loading safetensors/GGUF, tokenizers, reference implementations to
compare against, the official MCP SDK and FastAPI for the HTTP API. Deterministic arithmetic cannot be left to
NumPy or BLAS, whose reduction order depends on array size, thread count and library version, so every kernel that
reduces (sums, dot products, softmax, later matmul and attention) is written in C++ with a fixed order and exposed
to Python through [nanobind](https://github.com/wjakob/nanobind). Python orchestrates; C++ computes.

## Repository layout

| Path | What it is |
| --- | --- |
| `cpp/` | C++ kernels: `include/dllm/random.hpp` (RNG), `include/dllm/math.hpp` (exp, log, sin, cos, tanh, erf, sum, dot, softmax, log-softmax), `include/dllm/nn.hpp` (linear/matmul, RMSNorm, SiLU/GELU, RoPE, attention), `include/dllm/grad.hpp` (their gradients, cross-entropy, AdamW), `kernels.cpp` (Python bindings). Evaluation orders: [docs/kernels.md](docs/kernels.md) |
| `src/etalii_dllm/` | Python package: `tensor` (aligned float32 `Tensor`), `numerics`, `sampling`, `tokenization`, `models`, `generation` (streaming, stop sequences, logprobs), `grammar` (constrained decoding), `tools` (tool calling), `chat`, `engine` (shared facade) |
| `src/etalii_dllm/training/` | Fine-tuning: decoder gradients, AdamW, fixed data order, checkpoints. See [docs/training.md](docs/training.md) |
| `src/etalii_dllm/server/` | HTTP API, FastAPI: OpenAI (`/v1/models`, `/v1/chat/completions`, `/v1/responses`, `/v1/embeddings`) Anthropic (`/v1/messages`) and Ollama (`/api/chat`, `/api/generate`, ...); a chat page at `/` (`static/chat.html`). See [docs/api.md](docs/api.md) |
| `src/etalii_dllm/mcp_server.py` | Model Context Protocol server (stdio, official `mcp` SDK): `chat`, `generate` and `model_info` tools, model card/chat template/determinism resources, prompts. See [docs/mcp.md](docs/mcp.md) |
| `src/etalii_dllm/mcp_host.py` | MCP client host: the model calls external MCP servers' tools during a chat (`dllm chat --mcp-config`) |
| `src/etalii_dllm/cli.py` | `dllm` command line tool |
| `src/etalii_dllm/importing/` | Model import: safetensors and GGUF readers, GGUF dequantisation, Hugging Face download, `dllm import` |
| `src/etalii_dllm/bpe.py`, `chat_template.py` | BPE tokenizer (byte-level or SentencePiece-style) from `tokenizer.json` (or GGUF metadata) and the model's Jinja chat template |
| `src/etalii_dllm/transformer.py` | Llama/Qwen2/Qwen3 decoder with a KV cache that cannot change the logits |
| `src/etalii_dllm/modelfile.py` | The [`model.dllm`](docs/model-format.md) container that imported models are stored in |
| `tests/` | pytest suite, including golden-hash reproducibility tests |
| `docs/research/` | Research notes: [deterministic inference](docs/research/deterministic-inference.md), [compatibility targets](docs/research/compatibility.md), [model import](docs/research/model-import.md) |

## How determinism is achieved

Summarised from the [research notes](docs/research/deterministic-inference.md):

- **Own random numbers.** A specified generator (xoshiro256\*\* seeded by SplitMix64) instead of `random` or `numpy.random`.
- **Own transcendental functions.** `exp`, `log`, `sin`, `cos`, `tanh` and `erf` built from IEEE basic operations only.
  Not strictly needed on one machine, but cheap, and it keeps runtime or library updates from shifting results.
- **Fixed reduction order.** Sums, dot products, softmax, matmul, RMSNorm and attention run in C++ and accumulate in one documented order, never in an order
  chosen by thread scheduling or batch size (batch invariance), so concurrent requests cannot change each other's output.
- **Pinned Unicode.** Normalisation, lower-casing and the `\p{L}`-style classes in pre-tokenizer patterns use Unicode
  15.1 tables shipped in the package, not the ones of the installed Python or `regex`, so the same text gives the same
  tokens on every installation.
- **Total-order sampling.** Ties break on token id, so sorting never depends on algorithm stability.
- **Golden hashes in CI.** Tests assert SHA-256 hashes of weights and generated tokens; any drift fails the build.
  SIMD and threads only ever split work between output elements, so they have not changed a single hash.

## Architecture

[docs/architecture](docs/architecture/README.md) describes how the pieces fit together (system context, layers,
module map), with Mermaid diagrams.

## Roadmap

| Phase | Goal |
| --- | --- |
| 0. Bootstrap ✅ | Solution skeleton, deterministic RNG and math, sampler, byte tokenizer, placeholder model, OpenAI-style API, MCP server, CI with golden hashes |
| 1. Kernels ✅ | Tensor type, deterministic matmul with fixed tiling, RMSNorm, RoPE with deterministic `sin`/`cos`, SiLU/GELU, attention with fixed-order softmax |
| 2. Import existing models ✅ | Llama-style decoder with KV cache; `dllm import` converting small open-weight models (SmolLM2, Qwen2.5, ...) from safetensors/GGUF to our own format with licence metadata; BPE tokenizer and chat templates. SmolLM2-135M and Qwen2.5-0.5B verified against `transformers` in CI. See [model import](docs/research/model-import.md) |
| 3. Fine-tuning ✅ | Deterministic backprop and AdamW on top of imported weights, fixed data order, reproducible checkpoints |
| 4. API parity ✅ | Streaming (SSE), tool/function calling, JSON-schema structured output, logprobs, Anthropic Messages endpoint, embeddings. See [HTTP API](docs/api.md) |
| 5. MCP, both directions ✅ | Richer MCP server (prompts, resources); MCP client host so the model can call external tools during a chat. See [MCP](docs/mcp.md) |
| 6. Performance ✅ | ✅ SIMD and multi-threading with fixed, batch-invariant reduction order; ✅ integer quantisation (Q8_0, associative int32 accumulation); ✅ batch-invariance stress tests; ✅ CUDA kernels that give the CPU's bits (`--device cuda`). See [kernels](docs/kernels.md#threads-and-simd) and [GPU](docs/kernels.md#gpu) |
| 7. Usability and releases | ✅ Pre-built wheels for Linux, Windows and macOS and tagged GitHub releases ([releasing](docs/releasing.md)); PyPI; ✅ the CUDA install route checked in CI; ✅ a Docker image (`ghcr.io/vrenken/etalii-dllm`); ✅ a web chat UI at `/` of `dllm-server`; ✅ a larger verified model (Qwen2.5-1.5B) |
| 8. Serving and ecosystem ✅ | ✅ Prompt caching across requests with the same bits as a cold run ([prompt caching](docs/api.md#prompt-caching)); ✅ concurrent requests decoded as one batch, each keeping its solo bits ([batching](docs/api.md#concurrent-requests)); ✅ Ollama-compatible API ([Ollama API](docs/api.md#ollama-api)); ✅ OpenAI Responses API ([Responses API](docs/api.md#responses-api)); ✅ Qwen3 (verified Qwen3-0.6B); ✅ LoRA fine-tuning and PEFT adapter import ([LoRA](docs/training.md#lora-adapters)) |
| 9. Mainstream model families | ✅ SentencePiece-style tokenizers (TinyLlama, Llama 2, Mistral, Phi-3; verified TinyLlama-1.1B-Chat); ✅ Mistral and sliding-window attention; Gemma 2/3; ✅ Phi-3/Phi-4-mini; ✅ OLMo 2; ✅ Granite 3.x; a verified Llama 3.2. Each family is checked against `transformers`, see [which models can it run](#which-models-can-it-run) |
| 10. Portable determinism ✅ | Identical output across machines as a guarantee, not a bonus: ✅ kernels run in the IEEE default floating point environment whatever the process set ([kernels](docs/kernels.md#floating-point-environment)); ✅ tokenizer and chat template independent of the Python, `regex` and `jinja2` versions; ✅ real-model golden bits checked on every release platform and SIMD path; ✅ `dllm verify` to compare two machines; ✅ the guarantee documented ([portable determinism](docs/kernels.md#portable-determinism)) |

## Working with Claude Code

This repository is developed largely by Claude Code cloud sessions. [CLAUDE.md](CLAUDE.md) holds the build commands,
conventions and determinism rules those sessions follow.

## License

[Apache 2.0](LICENSE)
