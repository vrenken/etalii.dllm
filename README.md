# EtAlii.Dllm

**A deterministic large language model, built from scratch in Python with C++ kernels.**

Mainstream LLM inference is not reproducible: the same prompt, the same weights and even the same `seed` can
give different answers from one request to the next. EtAlii.Dllm treats bit-exact reproducibility as a hard
requirement instead of a best-effort hint:

> Same hardware + same weights + same prompt and context window ⇒ the same tokens, bit for bit, on every run.

That holds regardless of server load, batch composition or thread scheduling. Identical output across *different*
hardware is not a goal; where it comes for free (as it does today) it is a bonus, not a promise.

The model speaks the protocols the rest of the ecosystem already uses (an OpenAI-compatible chat API and the
Model Context Protocol), so existing clients and agents can use it without changes.

## Status

Bootstrap. The full pipeline (tokenizer → model → sampler → CLI / HTTP API / MCP server) runs end to end and is
proven run-to-run bit-exact by CI (on Linux, Windows and macOS, which today even agree with each other). The model itself is still a seeded placeholder (a bigram
table), so its output is noise. Next on the roadmap: a transformer that runs imported small open-weight models.

## Quick start

Requires Python 3.11+, CMake and a C++17 compiler (the numeric kernels are a C++ extension built on install).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest

# Command line
dllm info
dllm generate --prompt "Hello" --temperature 0.8 --seed 7

# OpenAI-compatible HTTP server on http://localhost:5080
dllm-server
curl http://localhost:5080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"dllm","messages":[{"role":"user","content":"Hi"}],"temperature":0.7,"seed":5}'

# MCP server over stdio, e.g. registered with Claude Code
claude mcp add dllm -- dllm-mcp

# Import an open-weight model to our own format (source, revision and licence are recorded in the file)
dllm import hf:HuggingFaceTB/SmolLM2-135M-Instruct -o smollm2-135m.dllm
dllm import ./qwen2.5-0.5b-instruct-q8_0.gguf -o qwen2.5-0.5b.dllm
dllm inspect smollm2-135m.dllm
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
| `cpp/` | C++ kernels: `include/dllm/random.hpp` (RNG), `include/dllm/math.hpp` (exp, sum, dot, softmax), `kernels.cpp` (Python bindings) |
| `src/etalii_dllm/` | Python package: `numerics`, `sampling`, `tokenization`, `models`, `generation`, `chat`, `engine` (shared facade) |
| `src/etalii_dllm/server/` | OpenAI-compatible HTTP API (`/v1/models`, `/v1/chat/completions`), FastAPI |
| `src/etalii_dllm/mcp_server.py` | Model Context Protocol server (stdio, official `mcp` SDK) exposing `generate` and `model_info` tools |
| `src/etalii_dllm/cli.py` | `dllm` command line tool |
| `src/etalii_dllm/importing/` | Model import: safetensors and GGUF readers, GGUF dequantisation, Hugging Face download, `dllm import` |
| `src/etalii_dllm/modelfile.py` | The [`model.dllm`](docs/model-format.md) container that imported models are stored in |
| `tests/` | pytest suite, including golden-hash reproducibility tests |
| `docs/research/` | Research notes: [deterministic inference](docs/research/deterministic-inference.md), [compatibility targets](docs/research/compatibility.md), [model import](docs/research/model-import.md) |

## How determinism is achieved

Summarised from the [research notes](docs/research/deterministic-inference.md):

- **Own random numbers.** A specified generator (xoshiro256\*\* seeded by SplitMix64) instead of `random` or `numpy.random`.
- **Own transcendental functions.** `exp` (and later `sin`, `cos`, `log`) built from IEEE basic operations only.
  Not strictly needed on one machine, but cheap, and it keeps runtime or library updates from shifting results.
- **Fixed reduction order.** Sums, dot products and softmax run in C++ and accumulate in one documented order, never in an order
  chosen by thread scheduling or batch size (batch invariance), so concurrent requests cannot change each other's output.
- **Total-order sampling.** Ties break on token id, so sorting never depends on algorithm stability.
- **Golden hashes in CI.** Tests assert SHA-256 hashes of weights and generated tokens; any drift fails the build.
  Hashes may become per-hardware once kernels use hardware-specific instructions.

## Roadmap

| Phase | Goal |
| --- | --- |
| 0. Bootstrap ✅ | Solution skeleton, deterministic RNG and math, sampler, byte tokenizer, placeholder model, OpenAI-style API, MCP server, CI with golden hashes |
| 1. Kernels | Tensor type, deterministic matmul with fixed tiling, RMSNorm, RoPE with deterministic `sin`/`cos`, SiLU/GELU, attention with fixed-order softmax |
| 2. Import existing models | Llama-style decoder with KV cache; `dllm import` converting small open-weight models (SmolLM2, Qwen2.5, TinyLlama, ...) from safetensors/GGUF to our own format with licence metadata; BPE tokenizer and chat templates. See [model import](docs/research/model-import.md) |
| 3. Fine-tuning (optional) | Deterministic backprop and AdamW on top of imported weights, fixed data order, reproducible checkpoints |
| 4. API parity | Streaming (SSE), tool/function calling, JSON-schema structured output, Anthropic Messages endpoint, embeddings |
| 5. MCP, both directions | Richer MCP server (prompts, resources); MCP client host so the model can call external tools during a chat |
| 6. Performance | SIMD and multi-threading with fixed, batch-invariant reduction order, integer quantisation (associative int32 accumulation), GPU kernels that keep bit-exactness |

## Working with Claude Code

This repository is developed largely by Claude Code cloud sessions. [CLAUDE.md](CLAUDE.md) holds the build commands,
conventions and determinism rules those sessions follow.

## License

[Apache 2.0](LICENSE)
