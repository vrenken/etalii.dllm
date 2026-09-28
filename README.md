# EtAlii.Dllm

**A deterministic large language model, built from scratch in C#/.NET.**

Mainstream LLM inference is not reproducible: the same prompt, the same weights and even the same `seed` can
give different answers from one request to the next. EtAlii.Dllm treats bit-exact reproducibility as a hard
requirement instead of a best-effort hint:

> Same weights + same request ⇒ the same tokens, bit for bit, on every run, every machine and every OS.

The model speaks the protocols the rest of the ecosystem already uses (an OpenAI-compatible chat API and the
Model Context Protocol), so existing clients and agents can use it without changes.

## Status

Bootstrap. The full pipeline (tokenizer → model → sampler → CLI / HTTP API / MCP server) runs end to end and is
proven bit-exact across Linux, Windows and macOS by CI. The model itself is still a seeded placeholder (a bigram
table), so its output is noise; the transformer is the next milestone on the roadmap.

## Quick start

Requires the [.NET 10 SDK](https://dotnet.microsoft.com/download).

```bash
dotnet build
dotnet test

# Command line
dotnet run --project src/EtAlii.Dllm.Cli -- info
dotnet run --project src/EtAlii.Dllm.Cli -- generate --prompt "Hello" --temperature 0.8 --seed 7

# OpenAI-compatible HTTP server on http://localhost:5080
dotnet run --project src/EtAlii.Dllm.Server
curl http://localhost:5080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"dllm","messages":[{"role":"user","content":"Hi"}],"temperature":0.7,"seed":5}'

# MCP server over stdio, e.g. registered with Claude Code
claude mcp add dllm -- dotnet run --project src/EtAlii.Dllm.Mcp
```

Run the same request twice and compare: the responses, including `id` and `system_fingerprint`, are identical.

## Repository layout

| Path | What it is |
| --- | --- |
| `src/EtAlii.Dllm.Core` | Deterministic numerics, RNG, sampler, tokenizer, models, generation loop |
| `src/EtAlii.Dllm.Server` | OpenAI-compatible HTTP API (`/v1/models`, `/v1/chat/completions`) |
| `src/EtAlii.Dllm.Mcp` | Model Context Protocol server (stdio) exposing `generate` and `model_info` tools |
| `src/EtAlii.Dllm.Cli` | `dllm` command line tool |
| `tests/` | xUnit tests, including golden-hash reproducibility tests |
| `docs/research/` | Research notes: [deterministic inference](docs/research/deterministic-inference.md), [compatibility targets](docs/research/compatibility.md) |

## How determinism is achieved

Summarised from the [research notes](docs/research/deterministic-inference.md):

- **Own random numbers.** A specified generator (xoshiro256\*\* seeded by SplitMix64) instead of `System.Random`.
- **Own transcendental functions.** `exp` (and later `sin`, `cos`, `log`) built from IEEE basic operations only,
  because platform math libraries differ in the last bit.
- **Fixed reduction order.** Sums, dot products and softmax accumulate in one documented order, never in an order
  chosen by thread scheduling, SIMD width or batch size.
- **Total-order sampling.** Ties break on token id, so sorting never depends on algorithm stability.
- **Golden hashes in CI.** Tests assert SHA-256 hashes of weights and generated tokens on three operating systems
  and two CPU architectures; any drift fails the build.

## Roadmap

| Phase | Goal |
| --- | --- |
| 0. Bootstrap ✅ | Solution skeleton, deterministic RNG and math, sampler, byte tokenizer, placeholder model, OpenAI-style API, MCP server, CI with golden hashes |
| 1. Kernels | Tensor type, deterministic matmul with fixed tiling, RMSNorm, RoPE with portable `sin`/`cos`, SiLU/GELU, attention with fixed-order softmax |
| 2. Transformer inference | Llama-style decoder with KV cache, BPE tokenizer (`tokenizer.json`/tiktoken), loading safetensors and GGUF weights, run small open models (e.g. SmolLM, TinyStories) bit-exactly |
| 3. Training from scratch | Deterministic backprop and AdamW, fixed data order, train a small model end to end with reproducible checkpoints |
| 4. API parity | Streaming (SSE), tool/function calling, JSON-schema structured output, Anthropic Messages endpoint, embeddings |
| 5. MCP, both directions | Richer MCP server (prompts, resources); MCP client host so the model can call external tools during a chat |
| 6. Performance | Fixed-lane SIMD, multi-threading with deterministic partitioning, integer quantisation (associative int32 accumulation), GPU kernels that keep bit-exactness |

## Working with Claude Code

This repository is developed largely by Claude Code cloud sessions. [CLAUDE.md](CLAUDE.md) holds the build commands,
conventions and determinism rules those sessions follow.

## License

[Apache 2.0](LICENSE)
