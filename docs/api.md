# HTTP API

`dllm-server` speaks the OpenAI and the Anthropic wire formats, so the official SDKs of both work against it
unchanged. Every endpoint is a thin translation layer over the same engine (`DllmEngine.chat_stream`), which is also
what the CLI and the MCP server use, so all front ends give the same answer for the same request.

| Endpoint | Compatible with | Notes |
| --- | --- | --- |
| `GET /v1/models` | OpenAI | The one model being served |
| `POST /v1/chat/completions` | OpenAI Chat Completions | Streaming, tools, structured output, logprobs, stop sequences |
| `POST /v1/embeddings` | OpenAI Embeddings | Mean-pooled final hidden states, L2-normalised |
| `POST /v1/messages` | Anthropic Messages | Streaming, tools, structured output (`output_config.format`), stop sequences |
| `POST /v1/messages/count_tokens` | Anthropic | Input tokens of a request, tools included |
| `GET /` | Browser | A chat page over the streamed `/v1/chat/completions`, with temperature, seed and system prompt; it marks a regenerated answer that is identical to the earlier one. Self-contained, nothing loaded from elsewhere |

## Determinism guarantees

- The same model file, request and seed give the same tokens on the same machine, under any load. Temperature 0 is
  greedy decoding; with a temperature above 0 the sampler draws from a seeded `DeterministicRandom` (`seed` defaults
  to 0), so sampled answers repeat too.
- Streamed and non-streamed answers are identical: the non-streamed response is assembled from the same event
  stream. The concatenated `content` deltas equal `message.content`, and the tool calls are the same calls with the
  same ids.
- Response ids (`chatcmpl-...`, `msg_...`), tool call ids (`call_...`, `toolu_...`) are hashes of the request and the
  model fingerprint; `created` is 0. Identical requests get byte-identical responses, and a streamed response has the
  same id as its non-streamed twin. The one exception is the prompt cache counters in `usage` (see
  [Prompt caching](#prompt-caching)), which say how much work earlier requests saved; start the server with
  `--prompt-cache 0` to have those always 0 as well.
- `system_fingerprint` (OpenAI) identifies the exact weights.

## Defaults and differences

- `temperature` defaults to 0 on every endpoint (OpenAI and Anthropic default to 1). `top_k` is accepted on the
  OpenAI endpoint too, and `seed` on the Anthropic endpoint (not part of its API: pass it with
  `extra_body={"seed": 7}`). Recent Anthropic SDKs no longer have `temperature`, `top_p` and `top_k` parameters;
  send them with `extra_body` as well.
- `max_tokens` defaults to 64 on the OpenAI endpoint; it is required on the Anthropic endpoint, as there.
- Only `n=1`. No images, audio, documents or server tools (requests using them get a 400 error). `developer`
  messages are treated as `system` messages. Unknown request fields are ignored.
- A final assistant message is a prefill: the answer continues its text (Anthropic semantics, on both endpoints).
- Errors use each API's error shape: `{"error": {"type": "invalid_request_error", "message": ...}}` for OpenAI and
  `{"type": "error", "error": {...}}` for Anthropic, with HTTP status 400 (also for schema validation errors).

## Prompt caching

The server keeps the KV caches (the attention keys and values) of recent requests and reuses the longest shared
token prefix for the next one, as OpenAI and Anthropic do with prompt caching. A multi-turn chat, a long shared
system prompt or an MCP tool loop only computes the new tokens: with SmolLM2-135M and a 260-token system prompt,
follow-up turns take 0.8 s instead of 3.5 s. It is automatic (no `cache_control` needed) and cannot change the
answer: the KV cache rows are exactly what a recompute gives, so tokens, logprobs, ids and `system_fingerprint` are
the same as without it (`tests/test_prompt_cache.py`).

- OpenAI: `usage.prompt_tokens_details.cached_tokens` is the number of prompt tokens read from the cache;
  `prompt_tokens` still counts all of them.
- Anthropic: `usage.cache_read_input_tokens` is the number read from the cache and `input_tokens` the rest, as
  Anthropic counts them; `cache_creation_input_tokens` is always 0 (nothing is billed).
- `--prompt-cache N` (or `DLLM_PROMPT_CACHE=N`) sets how many KV caches are kept, 4 by default; 0 turns caching
  off. Each cache holds one conversation and takes `2 x layers x kv_heads x head_dim x 4` bytes per token (about
  45 KB per token for SmolLM2-135M, 56 KB for Qwen2.5-1.5B). A cache serves one request at a time; a request whose
  prefix is cached but in use starts afresh. When the pool is full the least recently used cache goes.

## Streaming

`"stream": true` returns server-sent events. OpenAI: `chat.completion.chunk` objects, a first chunk with the role,
content deltas, tool call deltas (each call arrives whole in one delta), a final chunk with `finish_reason`, the usage
chunk when `stream_options.include_usage` is set, then `data: [DONE]`. Anthropic: `message_start`,
`content_block_start`/`content_block_delta`/`content_block_stop` per text or `tool_use` block (tool input arrives as
one `input_json_delta`), `message_delta` with `stop_reason` and usage, and `message_stop`.

Text is released as soon as it is final: bytes of an incomplete UTF-8 character, and text that may be the start of
a stop sequence, wait for the next token.

## Structured output

OpenAI `response_format` (`{"type": "json_object"}` or `{"type": "json_schema", "json_schema": {"schema": ...}}`)
and Anthropic `output_config.format` (`{"type": "json_schema", "schema": ...}`) switch on constrained decoding: at
every step only tokens that keep the output a valid prefix of a matching JSON document can be sampled, and
generation ends when the document is complete. The answer is therefore always valid JSON for the schema, unless it
is cut off by `max_tokens` (`finish_reason: "length"`).

How it works: the schema is compiled to a byte-level pushdown automaton (`src/etalii_dllm/grammar.py`); the bytes of
every vocabulary token sit in a trie, and a walk over the trie that stops as soon as the automaton rejects a prefix
yields the allowed tokens. With greedy decoding the most likely token is checked first, so the full walk is only
needed when the model wants something the schema forbids. Masking changes which tokens are eligible, not how they
are ranked: the sampler's (probability, token id) order and its random stream are unchanged.

Supported schema keywords: `type` (also as a list), `properties`, `required`, `additionalProperties` (no extra
properties are generated; a schema without `properties` is a free-form object), `items`, `minItems`, `maxItems`,
`enum`, `const`, `anyOf`, `oneOf` (as `anyOf`), single-schema `allOf`, local `$ref` (`#/$defs/...`,
`#/definitions/...`, recursion allowed) and `nullable`. Annotations such as `title`, `description` and `format` are
ignored. Keywords that constrain values in ways the automaton does not check (`pattern`, `minLength`, `minimum`,
...) are refused with a 400 error rather than silently ignored. Properties are generated in schema order; optional
ones may be skipped. At most 16 whitespace bytes may appear between JSON tokens.

## Tools

OpenAI `tools`/`tool_choice` and Anthropic `tools`/`tool_choice` map onto one mechanism. Models call tools in the
Hermes format used by Qwen2.5 and most tool-trained small models:

```text
<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris"}}
</tool_call>
```

- If the model's chat template supports tools, it presents them, exactly as `transformers` would. Otherwise the
  engine adds the Hermes tool instructions to the system message, shows earlier calls as `<tool_call>` blocks and
  tool results as `<tool_response>` user turns.
- `auto`: the model may answer in text. Once it writes `<tool_call>`, the call is constrained by a grammar built
  from the tool definitions (a known name and arguments that fit its `parameters`/`input_schema`, leniently: keywords
  the grammar cannot check are ignored for tools). Several calls in one answer are fine.
- `required` (OpenAI) / `any` (Anthropic), or a named tool: the answer is exactly one call, constrained from the
  first token.
- `none`: the tools are not shown to the model.
- A reply that is nothing but a JSON object `{"name": ..., "arguments"|"parameters": ...}` naming a known tool (the
  Llama 3 style) also counts as a call.
- `finish_reason` is `tool_calls` (OpenAI) and `stop_reason` is `tool_use` (Anthropic) when the answer has calls.
  Send results back as `tool` messages (OpenAI) or `tool_result` blocks (Anthropic).

When streaming with tools, text is streamed until the model starts a tool call (or starts its answer with `{`, which
could be a bare JSON call); the rest is decided when generation ends.

## Logprobs

OpenAI `logprobs: true` with `top_logprobs` 0 to 20 returns, per generated token, its log-probability under the
model's unmodified distribution (temperature 1, before top-k/top-p and constraints), computed by the fixed-order
`log_softmax` kernel, and the most likely alternatives ordered by (logprob descending, token id ascending). Token
strings are the token's bytes decoded as UTF-8 (with replacement characters for partial characters); `bytes` holds
the exact bytes.

## Embeddings

`/v1/embeddings` embeds each input (a string, a list of strings, a token array or a list of token arrays) as the
mean of the model's final-norm hidden states over all positions, L2-normalised. The mean sums every component over
the positions in ascending order in double (the `linear` kernel), so embeddings are bit-exact. `dimensions` keeps the
first components and normalises again; `encoding_format: "base64"` (the OpenAI SDK default) returns little-endian
float32. A decoder-only chat model is not a trained embedding model: expect usable but not state-of-the-art
similarity scores.

## Examples

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:5080/v1", api_key="unused")
for chunk in client.chat.completions.create(
    model="dllm", messages=[{"role": "user", "content": "Count to five."}], stream=True, seed=1
):
    print(chunk.choices[0].delta.content or "", end="")

reply = client.chat.completions.create(
    model="dllm",
    messages=[{"role": "user", "content": "Invent a cat."}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "cat",
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
            },
        },
    },
)
print(reply.choices[0].message.content)  # always valid JSON for the schema
```

```python
import anthropic

client = anthropic.Anthropic(base_url="http://127.0.0.1:5080", api_key="unused")
message = client.messages.create(
    model="dllm",
    max_tokens=200,
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[
        {
            "name": "get_weather",
            "description": "Current weather for a city",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        }
    ],
    extra_body={"temperature": 0.7, "seed": 42},
)
print(message.stop_reason, message.content)
```
