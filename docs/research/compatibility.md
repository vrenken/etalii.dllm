# Compatibility targets

EtAlii.Dllm should drop into existing tooling. These are the interfaces it implements or plans to implement, and
how determinism shows up in each.

## OpenAI Chat Completions (implemented, subset)

The de facto standard: most SDKs, agent frameworks and IDE plugins can point at a custom `base_url`.

| Feature | Status |
| --- | --- |
| `GET /v1/models` | ✅ |
| `POST /v1/chat/completions`, non-streaming | ✅ |
| `temperature`, `top_p`, `seed`, `max_tokens` / `max_completion_tokens` | ✅ |
| `system_fingerprint` | ✅ derived from a hash of the weights |
| Deterministic `id` and `created` | ✅ derived from the output instead of the clock, so identical requests give byte-identical responses |
| Streaming (`stream: true`, SSE chunks) | Phase 4 |
| `tools` / `tool_choice`, `tool_calls` in responses | Phase 4 |
| `response_format` with JSON schema (constrained decoding) | Phase 4 |
| `logprobs` | Phase 4 |
| Responses API (`/v1/responses`) | Later |
| Embeddings (`/v1/embeddings`) | Later |

Difference from OpenAI: `seed` is a guarantee, not a hint. A changed `system_fingerprint` is the only way the same
request can produce a different answer.

## Anthropic Messages API (planned)

`POST /v1/messages` with `system`, `messages`, `max_tokens`, `temperature`, `tools` and streaming events, so
Anthropic SDK clients can use the model. Phase 4.

## Model Context Protocol

Two directions:

1. **Dllm as MCP server** (implemented). `src/EtAlii.Dllm.Mcp` uses the official C# SDK (`ModelContextProtocol`
   NuGet package) over stdio and exposes `generate` and `model_info`, both annotated read-only and idempotent,
   which is literally true here. Next: MCP prompts and resources, and the HTTP (streamable) transport.
2. **Dllm as the model behind an MCP host** (planned, Phase 5). The model emits tool calls, a host connects to MCP
   servers, runs the calls and feeds results back. Determinism then covers the model's decisions; tool results are
   external inputs and are part of the "request" when reproducing a conversation.

## Model and tokenizer formats (planned, Phase 2)

- **safetensors** (Hugging Face) and **GGUF** (llama.cpp / Ollama) weight loading, so existing small open models
  can be run bit-exactly.
- **Tokenizers**: Hugging Face `tokenizer.json` BPE and tiktoken vocabularies, with byte-level fallback.
- **Chat templates**: the current template is a fixed internal format; loading a model's own Jinja chat template
  comes with Phase 2.
