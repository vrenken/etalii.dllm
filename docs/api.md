# HTTP API

`dllm-server` speaks the OpenAI, the Anthropic and the Ollama wire formats, so the official clients of all three
work against it unchanged. Every endpoint is a thin translation layer over the same engine (`DllmEngine.chat_stream`), which is also
what the CLI and the MCP server use, so all front ends give the same answer for the same request.

| Endpoint | Compatible with | Notes |
| --- | --- | --- |
| `GET /v1/models` | OpenAI | The one model being served |
| `POST /v1/chat/completions` | OpenAI Chat Completions | Streaming, tools, structured output, logprobs, stop sequences |
| `POST /v1/responses`, `GET`/`DELETE /v1/responses/{id}` | OpenAI Responses | Streaming (typed `response.*` events), function tools, `text.format` JSON schema, logprobs, `previous_response_id`. See [Responses API](#responses-api) |
| `POST /v1/embeddings` | OpenAI Embeddings | Final hidden states pooled as the model says (mean, or last token for embedding models), L2-normalised |
| `POST /v1/messages` | Anthropic Messages | Streaming, tools, structured output (`output_config.format`), stop sequences |
| `POST /v1/messages/count_tokens` | Anthropic | Input tokens of a request, tools included |
| `POST /api/chat`, `POST /api/generate` | Ollama | Streaming (NDJSON, the default), tools, `format` (JSON or a schema), logprobs, `raw` prompts. See [Ollama API](#ollama-api) |
| `POST /api/embed`, `POST /api/embeddings` | Ollama | Same vectors as `/v1/embeddings` |
| `GET /api/tags`, `POST /api/show`, `GET /api/ps`, `GET /api/version` | Ollama | The served model, its template and architecture |
| `POST /v1/receipts/verify` | Extension | Re-runs a [generation receipt](receipts.md) and returns `{"ok", "reasons", "notes", "receipt"}`; a list is a [receipt chain](agents.md#receipt-chains) (`{"ok", "reasons", "notes", "turns"}`) |
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
- Every chat endpoint returns a [receipt](receipts.md) when the request says `"receipt": true`: the engine request and
  hashes of the output, which anyone with the same weights can replay (`dllm replay`, `POST /v1/receipts/verify`).
  `"previous_receipt": "rcpt_..."` links it to the previous turn's receipt, and `previous_response_id` does so
  automatically; `POST /v1/receipts/verify` with a list checks the whole [chain](agents.md#receipt-chains).
  Asking for one changes neither the answer nor the response id.

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
- `--persistent-cache DIR` (or `DLLM_PROMPT_CACHE_DIR`) also keeps the idle caches in `DIR`, so a restarted server
  still skips the prefixes it has seen: with SmolLM2-135M on a 4-core cloud VM, the first answer after a restart
  to a 1275-token prompt that shares a 1260-token system prompt with an earlier request takes 0.4 s instead of
  6.3 s (the stored cache is 59 MB; loading it does not measurably slow the start). A stored cache is used only by the same weights (quantisation and steering
  included) and engine version, and only when its SHA-256 checksum matches, so a damaged file is ignored rather than
  read. The files hold the prompts' keys and values, from which the prompt text can be partly recovered: keep the
  directory as private as the conversations.

## Concurrent requests

Requests that arrive together are decoded together (continuous batching, as vLLM and TGI do): at every step the new
tokens of all running requests, prompts being read and single tokens being generated alike, go through the model as
one batch, so the weights are read once per step instead of once per request. A request joins at the next step after
it arrives and leaves when it is done. With SmolLM2-135M on a 4-core VM, eight concurrent greedy requests of 32 tokens
finish in 8.6 s instead of 16.7 s one after another.

Unlike other batching servers, this cannot change anyone's answer. Every kernel computes each row on its own in a
fixed order and attention runs per request, so a request gets exactly the bits it gets alone, whichever requests
share its steps and whenever they arrived (`tests/test_batching.py`, and `tests/test_batch_invariance.py` fires 36
overlapping requests at the server and compares every response with a lone run). Nothing needs configuring.

## Speculative decoding

`--speculate [N]` (or `DLLM_SPECULATE=N`) guesses up to N next tokens cheaply, checks them all in one forward pass
and keeps the ones the model would have chosen, so repetitive output (code, quotes, edits of the prompt, JSON) comes
out several tokens per pass. The guesses come from the text so far (the tokens that followed the last earlier
occurrence of the current two or three tokens), or, with `--draft-model FILE` (`DLLM_DRAFT_MODEL`), from a smaller
model with the same tokenizer, which turns speculation on with N = 8.

Other engines accept a draft token when a random test says its probability is high enough, which keeps the
distribution but not the tokens. Here a drafted token is kept only when it *is* the token plain decoding picks at that
position, from the same logits and that position's own random draw, so tokens, text, logprobs, finish reasons and
`system_fingerprint` are identical with and without it, at any temperature, with structured output, tools and stop
sequences (`tests/test_speculative.py`). The check pass gives each position the bits of one-at-a-time decoding
because every kernel computes each row on its own. Speculation only changes the speed: on SmolLM2-135M an answer that
repeats its prompt is about 1.6× faster, code about 1.3×, free chat about the same. A step that checks a draft runs
on its own rather than in the shared batch of concurrent requests; either way every request keeps its solo bits.

## Responses API

OpenAI's newer Responses API works with the official SDK's `client.responses`:

```python
first = client.responses.create(model="dllm", instructions="Be brief.", input="My name is Ada.")
follow_up = client.responses.create(model="dllm", input="What is my name?", previous_response_id=first.id)
print(follow_up.output_text)
```

It gives the same answer as `/v1/chat/completions` for the same conversation and options.

- `input` is a string or a list of items: messages (`user`, `assistant`, `system`, `developer`, with `input_text`/
  `output_text` parts), `function_call` and `function_call_output`. `instructions` becomes the system message and,
  as in OpenAI's API, does not carry over to a follow-up.
- `previous_response_id` continues a stored response. Responses are kept in memory (`store`, default true; the most
  recent 256), can be fetched with `GET /v1/responses/{id}` and removed with `DELETE`. Their ids (`resp_...`, item ids
  `msg_...`/`fc_...`) are hashes of the request and the weights, so a conversation replayed from its start gets the
  same ids; they do not survive a server restart.
- Function tools with `tool_choice` `auto`, `none`, `required` or `{"type": "function", "name": ...}`;
  `text.format` of type `json_object` or `json_schema`; `include=["message.output_text.logprobs"]` with
  `top_logprobs`. `max_output_tokens` defaults to 1024; a response cut off by it has `status: "incomplete"`.
- Streaming sends `response.created`, `response.in_progress`, then per output item `response.output_item.added`,
  the text (`response.content_part.added`, `response.output_text.delta`, `.done`, `response.content_part.done`) or
  the call (`response.function_call_arguments.delta`, `.done`), `response.output_item.done`, and finally
  `response.completed` or `response.incomplete` carrying the whole response, which is exactly the non-streamed body.
- `usage.input_tokens_details.cached_tokens` reports the [prompt cache](#prompt-caching). Built-in tools (web
  search, file search, ...), images, files, reasoning items and background mode are not supported.

## Ollama API

Tools that talk to Ollama (the `ollama` Python and JavaScript clients, Open WebUI, LangChain's `ChatOllama`,
Continue, ...) work when pointed at `http://127.0.0.1:5080` instead of Ollama's `http://127.0.0.1:11434`:

```python
import ollama

client = ollama.Client(host="http://127.0.0.1:5080")
reply = client.chat(
    model="dllm", messages=[{"role": "user", "content": "Hi!"}], options={"seed": 7, "temperature": 0.7}
)
print(reply.message.content)
```

The answer is the same as the OpenAI and Anthropic endpoints give for the same conversation and options.

- `options`: `temperature` (default 0, Ollama's is 0.8), `seed`, `top_k`, `top_p`, `num_predict` (unset or negative:
  up to 2048 tokens or the model's context, whichever is smaller) and `stop`; other options are ignored.
- `format`: `"json"` for any JSON object, or a JSON schema, enforced by constrained decoding.
- Tools use Ollama's format (arguments as objects). Ollama's calls carry no ids, so the conversation's calls are
  numbered `call_0`, `call_1`, ...; a `tool` message answers the earliest open call to the tool it names in
  `tool_name`.
- `/api/generate` renders `system` and `prompt` with the chat template; with `raw: true` the prompt is used as is.
  An empty `prompt` returns `done_reason: "load"`, as Ollama does. `suffix`, `template`, `context` and images are
  refused.
- `created_at`/`modified_at` are always the Unix epoch and every `*_duration` is 0 (nothing clock-derived);
  `prompt_eval_count` counts the prompt tokens not read from the [prompt cache](#prompt-caching), as Ollama does.
- One model is served whatever `model` names; pulling, creating, copying and deleting models is not supported
  (`dllm import` does that job).

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

Embedding models imported from sentence-transformers (such as Qwen3-Embedding-0.6B) use their own pooling instead:
the text is encoded with the tokenizer's special tokens and the last token's hidden state (or the mean) is the
embedding. The extension field `input_type` names one of the model's prompts (`"query"` for search queries, or
`"document"`), which is put in front of the text; an unknown name is a 400 error. See [retrieval](retrieval.md).

A server started with `--index` grounds every chat request in the passages its document index finds for the last
user message, and its `system_fingerprint` includes the index ([retrieval](retrieval.md#grounded-chat)).

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
