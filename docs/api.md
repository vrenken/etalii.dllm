# HTTP API

`dllm-server` speaks the OpenAI, the Anthropic and the Ollama wire formats, so the official clients of all three
work against it unchanged. Every endpoint is a thin translation layer over the same engine (`DllmEngine.chat_stream`), which is also
what the CLI and the MCP server use, so all front ends give the same answer for the same request.

| Endpoint | Compatible with | Notes |
| --- | --- | --- |
| `GET /v1/models` | OpenAI | The one model being served |
| `POST /v1/chat/completions` | OpenAI Chat Completions | Streaming, tools, structured output, logprobs, stop sequences |
| `POST /v1/completions` | OpenAI Completions (legacy) | Raw prompts continued without a chat template, streaming, `n`, logprobs, `echo` with exact prompt scores. See [completions API](#completions-api) |
| `POST /v1/responses`, `GET`/`DELETE /v1/responses/{id}` | OpenAI Responses | Streaming (typed `response.*` events), function tools, `text.format` JSON schema, logprobs, `previous_response_id`. See [Responses API](#responses-api) |
| `POST /v1/embeddings` | OpenAI Embeddings | Final hidden states pooled as the model says (mean, or last token for embedding models), L2-normalised |
| `POST /v1/messages` | Anthropic Messages | Streaming, tools, structured output (`output_config.format`), stop sequences |
| `POST /v1/messages/count_tokens` | Anthropic | Input tokens of a request, tools included |
| `POST /api/chat`, `POST /api/generate` | Ollama | Streaming (NDJSON, the default), tools, `format` (JSON or a schema), logprobs, `raw` prompts. See [Ollama API](#ollama-api) |
| `POST /api/embed`, `POST /api/embeddings` | Ollama | Same vectors as `/v1/embeddings` |
| `POST /v1/rerank`, `POST /rerank` | Cohere/Jina rerank | Documents ranked for a query by the model as a yes/no judge ([reranking](retrieval.md#reranking)) |
| `GET /api/tags`, `POST /api/show`, `GET /api/ps`, `GET /api/version` | Ollama | The served model, its template and architecture |
| `POST /v1/receipts/verify` | Extension | Re-runs a [generation receipt](receipts.md) (or re-scores a score receipt, re-votes a vote receipt) and returns `{"ok", "reasons", "notes", "receipt"}`; a list is a [receipt chain](agents.md#receipt-chains) (`{"ok", "reasons", "notes", "turns"}`) |
| `POST /v1/watermark/detect` | Extension | Green tokens, z-score and verdict of a text for a watermark key ([watermarks](watermarks.md#detecting)) |
| `GET /v1/audit` | Extension | What `--audit-every` found, the response cache's counters and how many requests were coalesced. See [serving at scale](serving.md) |
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
- `--response-cache DIR` answers repeated requests from storage, and identical requests in flight share one
  generation. Both give the same bytes as a fresh run, apart from the cached-token counters; see
  [serving at scale](serving.md).

## Defaults and differences

- `temperature` defaults to 0 on every endpoint (OpenAI and Anthropic default to 1). `top_k` is accepted on the
  OpenAI endpoint too, and `seed` on the Anthropic endpoint (not part of its API: pass it with
  `extra_body={"seed": 7}`). Recent Anthropic SDKs no longer have `temperature`, `top_p` and `top_k` parameters;
  send them with `extra_body` as well.
- `max_tokens` defaults to 64 on the OpenAI endpoint; it is required on the Anthropic endpoint, as there.
- `n` (up to 16) on chat completions returns several choices; see [decoding controls](#decoding-controls). No
  images, audio, documents or server tools (requests using them get a 400 error). `developer`
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
  An empty `prompt` returns `done_reason: "load"`, as Ollama does. A `suffix` [fills in the middle](#fill-in-the-middle)
  between the prompt and it. `template`, `context` and images are refused.
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
properties are generated; a schema without `properties` is a free-form object, whose values follow
`additionalProperties` when that is a schema), `propertyNames`, `minProperties`, `maxProperties`, `items`,
`prefixItems` (and `items` arrays with `additionalItems`), `minItems`, `maxItems`, `uniqueItems`, `enum`, `const`,
`anyOf`, `oneOf` (as `anyOf`), `allOf`, `not`, `if`/`then`/`else`, `patternProperties`, `contains` (with
`minContains`/`maxContains`), local `$ref` (`#/$defs/...`, `#/definitions/...`, recursion allowed) and `nullable`.
Annotations such as `title` and `description` are ignored. Keywords the automaton does not check
(`dependentSchemas`, `unevaluatedProperties`, ...) are refused with a 400 error rather than silently ignored.
Properties are generated in schema order; optional ones may be skipped. At most 16 whitespace bytes may appear
between JSON tokens.

Value constraints are exact too (Phases 31, 33, 34 and 35), each compiled to a byte automaton or checked by the
automaton of the object or array:

| Keyword | Rule |
| --- | --- |
| `pattern` | The [regex syntax](#decoding-controls) of `guided_regex`, found anywhere in the string as JSON Schema says, unless a leading `^` or trailing `$` anchors it. `^a\|b` is refused as ambiguous: group the alternation. |
| `format` | `date` (`YYYY-MM-DD`, month 01 to 12, day 01 to 31), `time` (`hh:mm:ss`, optional fraction, `Z` or an offset), `date-time` (both, joined by `T`), `uuid` and `ipv4`. Other formats stay annotations. |
| `minLength`, `maxLength` | Counted in code points. |
| `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum` | On integers and numbers, as numbers (or the draft 4 booleans). The answer is a decimal text in the range, compared exactly with the bound as written (a float bound as its shortest decimal, so `0.1` means 0.1), never with an exponent and never `-0`. |
| `multipleOf` | An integer or a decimal such as `0.01`: the answer's decimal value divided by it is an integer, in exact decimal arithmetic. Divisors with too many digits (more than the automaton's 10,000 states) are refused. |
| `prefixItems` | One schema per leading element (a tuple, as pydantic writes it); `items` then applies to the rest, and `items: false` closes the tuple. The older `items` array with `additionalItems` works the same. |
| `propertyNames` | A string schema (`pattern`, `format`, lengths, `enum`, `const`) for the names of a free-form object. Declared properties whose names break it are left out; a required one is refused. |
| `uniqueItems` | Enforced when the items come from a finite set (`enum`, `const`, booleans, `null` or unions of those): the automaton remembers the values written, comparing JSON values (`1` equals `1.0`, never `true`). Other items with `uniqueItems: true` are refused. |
| `minProperties`, `maxProperties` | The number of members. On an object without `properties` (whose names may repeat), `minProperties` above 1 is refused. |
| `patternProperties` | A name that matches patterns (found anywhere, as for `pattern`) gets the merged schemas of all of them; other names get `additionalProperties`. Names are split by automaton intersection and difference, so every name has exactly one rule. Declared properties also obey the patterns they match. |
| `contains`, `minContains`, `maxContains` | At least `minContains` (default 1) elements match `contains`, for any items. `maxContains` is enforced when the items come from a finite set (it works together with `uniqueItems`) and refused otherwise. |
| `allOf` | Merged keyword by keyword into one schema: types intersect, `enum` values intersect, bounds take the tighter value, `multipleOf` the exact least common multiple, every `pattern` applies, properties merge per name, tuples per position and `anyOf` branches distribute. Two different `contains` or two sets of `patternProperties` are refused. |
| `not` | Exact when the schema can be negated: types (not `integer` alone), `enum`/`const` (values removed from strings and numbers by automaton difference, so `not: {"const": 1}` also rules out `1.0`), string lengths, patterns and formats, number bounds, `multipleOf`, and object conditions of `properties` and `required`; `anyOf`, `allOf` and `not` negate by De Morgan's laws. Anything else is refused. |
| `if`, `then`, `else` | Rewritten as (`if` and `then`) or (`not if` and `else`), so `if` must be negatable as for `not`. A discriminator such as `"if": {"properties": {"kind": {"const": "circle"}}}` picks which properties are required. |

`enum` and `const` values are also checked against the schema's other keywords: `{"type": "string", "enum": ["a", 1]}`
allows only `"a"`. A keyword that constrains one type leaves the other types free when the schema has no `type`, as
JSON Schema says: `{"minimum": 5}` takes any string but no number below 5. An optional property or `anyOf` branch no
value can satisfy is left out; a required one is refused. The rewrites are pure functions of the schema
(`src/etalii_dllm/schema_algebra.py`), so a schema gives the same automaton on every machine.

A constrained string is written without escape sequences, so it holds no quote, backslash or control character. Its
automaton is the intersection of the automata of each keyword, with every state that cannot reach a match removed, so
decoding never runs into a dead end; a schema no string (or number) can satisfy is refused with a 400 error. Tool parameters only
guide the model, so their value constraints are ignored.

```bash
dllm chat "Book a flight" --json-schema '{"type": "object", "properties": {"from": {"type": "string", "pattern": "^[A-Z]{3}$"},
  "date": {"type": "string", "format": "date"}, "seats": {"type": "integer", "minimum": 1, "maximum": 9}},
  "required": ["from", "date", "seats"]}'
```

## Grammars

A GBNF grammar (the format of llama.cpp) constrains an answer to the text its `root` rule derives, matched in full
(Phase 32). It goes in `response_format: {"type": "grammar", "grammar": ...}` or the llama.cpp-style `grammar` field
on chat completions and completions, or `--grammar FILE` (or the grammar itself) on `dllm generate` and `dllm chat`.
The grammar is compiled to the same byte-level pushdown automaton as a JSON schema, so the answer always follows it
unless `max_tokens` cuts it off, streamed and non-streamed answers are identical, and receipts record the grammar so
`dllm replay` repeats it.

```
root   ::= "Colours: " colour (", " colour){1,3} "."
colour ::= "red" | "green" | "blue" | "yellow"
```

| Syntax | Meaning |
| --- | --- |
| `name ::= ...` | A rule; names hold letters, digits, `-` and `_`. A rule ends where the next `name ::=` begins, so it may span lines. `#` starts a comment. |
| `a b`, `a \| b`, `( ... )` | Sequence, alternatives (an empty one matches nothing), group. |
| `"text"` | A literal, with the escapes `\n \r \t \\ \" \[ \] \-`, `\xHH`, `\uHHHH` and `\UHHHHHHHH`. |
| `[a-z_]`, `[^"\n]`, `.` | One character of a class (ranges, negation, the same escapes), or any character. Characters are Unicode code points, never surrogates, written as UTF-8. |
| `x*`, `x+`, `x?`, `x{m}`, `x{m,}`, `x{m,n}` | Repetitions, counts up to 1000. |
| `name` | A rule reference; rules may refer to each other and to themselves, on the left too. |
| `<[42]>`, `<think>`, `!<[42]>` | A token reference (Phase 51): one whole token by id, the token whose text is exactly `<think>`, or any one token except it. |

**Left recursion** (Phase 51) is rewritten exactly before compiling: `expr ::= expr "+" term | term` derives the same
strings as `expr ::= term ("+" term)*` and is compiled as such (Paull's algorithm over the rules of each
left-recursive cycle, in their written order), so grammars written for parser generators work unchanged.

**Token references** (Phase 51, as in llama.cpp) match tokens rather than text: `<[id]>` the token with that id,
`<name>` the token whose text in the model's vocabulary is exactly `<name>` (such as `<think>` or `<|im_start|>`), and
`!<[id]>`/`!<name>` any one token except that one. A token reference reads exactly one token, at a token boundary,
and any token it reads is allowed whatever its bytes (so a negation can produce bytes that are not UTF-8 on their
own). The model's stop tokens always end the answer, so a grammar cannot ask for one and a negation never includes
them. A special token that decodes to no text adds none.

```
root     ::= <think> thinking </think> answer
thinking ::= !</think>*
answer   ::= " " [A-Za-z ]+ "."
```

**Lazy grammars** (Phase 51, llama.cpp's `grammar_lazy` and `grammar_triggers`) leave the answer free until one of
the trigger words appears; from the start of the earliest one on, the rest of the answer must follow the grammar
(which must therefore start with the trigger word), and the answer ends when the grammar's match cannot be extended.
If the token that completes a trigger already runs past it with bytes the grammar refuses, the rest stays free.
Triggers go in `"grammar_lazy": true, "grammar_triggers": [{"type": "word", "value": "<tool>"}]` (or plain strings)
on chat completions and completions, or `--grammar-trigger WORD` (repeatable) on `dllm generate` and `dllm chat`.
Receipts record the trigger words, so `dllm replay` repeats them, and requests without them keep their ids.

Refused with a 400 error (a `GrammarError` in Python): left recursion behind something that can match nothing
(`a ::= b a "x"` where `b` can match nothing), unbounded repetitions of something that can match nothing, undefined
or doubly defined rules, a missing `root`, unknown token references or ones naming a stop token, a grammar that grows
too large when its left recursion is rewritten, and a lazy grammar that cannot start with its trigger word or that is
combined with tools. Alternatives that can never finish (a rule with no way out of its recursion) are dropped, so
decoding never runs into an answer it cannot end, and a grammar that derives no text at all is refused. A grammar
cannot be combined with another structured output, a `guided_regex` or beam search.

```bash
dllm chat "Name some colours" --grammar colours.gbnf
curl -s localhost:5080/v1/completions -H 'content-type: application/json' -d '{
  "prompt": "2 + 2 =", "max_tokens": 20, "temperature": 0.7, "seed": 1,
  "grammar": "root ::= \" \" [0-9]+ \".\""
}'
```

## Decoding controls

Penalties, min-p, logit bias, several choices and regexes (Phase 21) keep every answer bit-reproducible: each one is
an exact, documented step ([specification](specification.md#4-random-numbers-and-sampling)) that the independent
reference implementation repeats bit for bit.

| Control | OpenAI chat | Ollama `options` | CLI (`generate`, `chat`) |
| --- | --- | --- | --- |
| Frequency and presence penalties (count only the answer's tokens) | `frequency_penalty`, `presence_penalty` | `frequency_penalty`, `presence_penalty` | `--frequency-penalty`, `--presence-penalty` |
| Repetition penalty over the last N tokens of prompt and answer (-1: all) | `repetition_penalty`, `repeat_last_n` (extensions) | `repeat_penalty`, `repeat_last_n` | `--repetition-penalty`, `--repeat-last-n` |
| Min-p: drop tokens less likely than `min_p` times the top one | `min_p` (extension) | `min_p` | `--min-p` |
| Logit bias added to chosen tokens | `logit_bias` | | `--logit-bias TOKEN=BIAS` |
| Several choices | `n` | | `--n` (`generate`) |
| Output matching a regex in full | `response_format: {"type": "regex", "regex": ...}` or `guided_regex` | | `--regex` |
| Output a [GBNF grammar](#grammars) derives | `response_format: {"type": "grammar", "grammar": ...}` or `grammar` | | `--grammar` |
| A [lazy grammar](#grammars) from a trigger word on | `grammar_lazy`, `grammar_triggers` (extensions, as in llama.cpp; chat completions and completions) | | `--grammar-trigger` |
| [Guided decoding](#guided-decoding): a negative prompt, or contrast against an amateur model | `guidance: {negative_prompt, scale}`, `contrast: {beta, alpha}` (extensions; also on Anthropic messages and completions) | | `--negative-prompt`, `--guidance-scale`, `--contrast BETA`, `--contrast-alpha` |
| [Length and stop controls](#length-and-stop-controls): a minimum length, writing past the end-of-sequence token, extra stop token ids, the stop string kept | `min_tokens`, `ignore_eos`, `stop_token_ids`, `include_stop_str_in_output` (extensions, as in vLLM; chat completions, completions and Responses) | | `--min-tokens`, `--ignore-eos`, `--stop-token-id`, `--stop`, `--include-stop` |
| [Token healing](#token-healing) of a prompt or prefill that ends inside a word | `token_healing: true` (extension; also on Responses, Anthropic messages, completions and Ollama) | | `--token-healing` |
| [Modern samplers](#modern-samplers): DRY, XTC, locally typical, top-n-sigma | `dry_multiplier`, `dry_base`, `dry_allowed_length`, `dry_penalty_last_n`, `dry_sequence_breakers`, `xtc_probability`, `xtc_threshold`, `typical_p`, `top_n_sigma` (extensions, as in llama.cpp; chat completions and completions) | the same names | `--dry-multiplier`, `--dry-base`, `--dry-allowed-length`, `--dry-penalty-last-n`, `--dry-sequence-breaker`, `--xtc-probability`, `--xtc-threshold`, `--typical-p`, `--top-n-sigma` |
| [Adaptive samplers](#adaptive-samplers): Mirostat 1 and 2, dynamic temperature | `mirostat`, `mirostat_tau`, `mirostat_eta`, `dynatemp_range`, `dynatemp_exponent` (extensions, as in llama.cpp; chat completions and completions) | the same names | `--mirostat`, `--mirostat-tau`, `--mirostat-eta`, `--dynatemp-range`, `--dynatemp-exponent` |
| A keyed [watermark](watermarks.md) | `watermark: {key, gamma, delta}` (extension; also on Anthropic messages) | `watermark_key`, `watermark_gamma`, `watermark_delta` | `--watermark-key`, `--watermark-gamma`, `--watermark-delta` |

- The defaults change nothing: a repetition penalty of 1 (Ollama's own default is 1.1; here it stays off unless asked
  for, so answers equal those of the other endpoints), no bias, `min_p` 0. A `logit_bias` id outside the vocabulary
  is a 400 error. Receipts record every control, so `dllm replay` repeats them.
- Choice `i` samples with the seed `seed + i`, so each choice is exactly the answer of a single request with that
  seed. Non-streamed choices are decoded concurrently (sharing batched steps, which never changes a bit); streamed ones
  arrive one after another, never interleaved by timing. `usage.completion_tokens` adds up all choices. With
  `"receipt": true` the top-level `receipt` belongs to choice 0, each choice carries its own `receipt`, and streamed
  choices have theirs on their finishing chunk.
- A regex constrains decoding like a JSON schema does ([structured output](#structured-output)): the answer is
  always a full match unless `max_tokens` cuts it off. `\d`, `\w` and `\s` mean `[0-9]`, `[A-Za-z0-9_]` and ASCII
  whitespace; classes, `.`, groups, alternation and `* + ? {m,n}` are supported. Backreferences, lookaround, word
  boundaries and flags are refused with a 400 error.

```bash
curl -s localhost:5080/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [{"role": "user", "content": "Give me a date"}],
  "temperature": 0.8, "seed": 3, "n": 2, "frequency_penalty": 0.5,
  "response_format": {"type": "regex", "regex": "\\d{4}-\\d{2}-\\d{2}"}
}'
```

## Modern samplers

The samplers llama.cpp and text-generation-webui users reach for (Phase 49) are exact here too, with llama.cpp's
parameter names and defaults, and are repeated bit for bit by the reference implementation
([specification](specification.md#4-random-numbers-and-sampling)):

- **DRY** ("don't repeat yourself", `dry_multiplier` > 0): a token that would extend a run already seen in the
  prompt or answer is penalised by `dry_multiplier * dry_base^(n - dry_allowed_length)` for a run of `n >=
  dry_allowed_length` tokens. A token whose text contains one of `dry_sequence_breakers` (default newline, `:`,
  `"` and `*`) ends a run; runs count up to 256 tokens; `dry_penalty_last_n` limits how far back it looks (-1: all).
  DRY changes the logits, so it works with greedy decoding too.
- **XTC** (exclude top choices, `xtc_probability` > 0): with that chance per token, every candidate at least
  `xtc_threshold` likely (default 0.1) except the least likely of them is dropped, so the answer avoids the most
  obvious words. The coin flip comes from the request's seeded stream and is drawn only when XTC is on.
- **Locally typical sampling** (`typical_p` < 1): keeps the candidates whose surprise is closest to the
  distribution's entropy, up to `typical_p` of the probability.
- **Top-n-sigma** (`top_n_sigma` > 0): keeps the tokens whose logit is within `top_n_sigma` standard deviations of
  the top logit, which makes high temperatures usable.

With a temperature, the steps run in this order: top-k, top-n-sigma, typical-p, top-p, min-p, XTC. They are recorded in
receipts and cache keys only when set, so earlier receipts keep their bytes.

```bash
dllm generate --prompt "Once upon a time" --temperature 1.2 --seed 4 --top-n-sigma 1.5 --xtc-probability 0.5 \
    --dry-multiplier 0.8
```

## Adaptive samplers

Samplers that adapt to the distribution token by token (Phase 50), with llama.cpp's and Ollama's names and defaults,
exact and repeated bit for bit by the reference implementation
([specification](specification.md#4-random-numbers-and-sampling)):

- **Mirostat** (`mirostat` 1 or 2): keeps the answer's surprise near `mirostat_tau` bits (default 5) by cutting the
  candidates and moving its limit `mu` after every token by `mirostat_eta` (default 0.1) times the error. Mirostat 2
  cuts candidates whose surprise exceeds `mu`; Mirostat 1 fits a Zipf exponent to the top 100 candidates and keeps a
  matching number. Mirostat replaces top-k, top-n-sigma, typical-p, top-p, min-p and XTC, as in llama.cpp; logit
  bias, penalties and DRY still apply.
- **Dynamic temperature** (`dynatemp_range` > 0): the temperature moves between `temperature - dynatemp_range` and
  `temperature + dynatemp_range` with the distribution's normalised entropy raised to `dynatemp_exponent` (default
  1): confident steps run cool, uncertain ones warm.

```bash
dllm generate --prompt "Once upon a time" --temperature 0.9 --seed 4 --mirostat 2 --mirostat-tau 3 \
    --dynatemp-range 0.5
```

## Completions API

`POST /v1/completions` continues a prompt as it is, without a chat template, on the same event stream as every other
endpoint: a completion equals `dllm generate` with the same prompt and options, bit for bit. `prompt` is a string or
a list of strings (choice `index` is `prompt_index * n + i`); `max_tokens` defaults to 16 as in OpenAI's API and
`temperature` to 0 as everywhere here. `stop`, `seed`, `n`, the [decoding controls](#decoding-controls),
`guided_regex`, `grammar` (lazy too), `watermark`, `stream` (with `stream_options.include_usage`) and `receipt` (a receipt on each choice)
work as on chat completions. `suffix` [fills in the middle](#fill-in-the-middle) between the prompt and it (an empty
`suffix` is no suffix, as in OpenAI's API). A `best_of` other than `n` is refused with a 400 error.

`logprobs: N` returns OpenAI's `logprobs` object (`tokens`, `token_logprobs`, `top_logprobs` with N alternatives,
`text_offset` in characters). With `echo: true` the prompt comes first in `text` and in `logprobs`, its tokens
[scored exactly](#scoring) (the first prompt token has `null`, as in OpenAI's API).

## Length and stop controls

Four controls (Phase 38, named as in vLLM) shape where an answer ends, each an exact step that receipts record and
the reference implementation repeats:

- `min_tokens`: until the answer has that many tokens, no stop token can be chosen. The sampler picks among the
  other tokens exactly as a grammar restricts its choice, so the answer is the one the restricted distribution gives;
  stop sequences still end it. `max_tokens` still caps it.
- `ignore_eos`: the model's own end-of-sequence tokens no longer end the answer. They can be chosen like any token
  (they add no text), so the answer runs to `max_tokens`, a stop sequence or a `stop_token_ids` token: useful for
  benchmarks and fixed-length outputs.
- `stop_token_ids`: token ids that end the answer as well as the model's own stop tokens (not part of the answer).
  Ids outside the vocabulary are a 400 error.
- `include_stop_str_in_output`: the stop sequence that ended the answer stays in its text, streamed or not.

Unset, they change nothing (and leave response ids as they were). Beam search refuses them.

```bash
dllm generate --prompt "Once upon a time" --min-tokens 40 --max-tokens 60 --temperature 0.8 --seed 1
curl -s localhost:5080/v1/completions -H 'content-type: application/json' -d '{
  "prompt": "Q: 2 + 2 =", "max_tokens": 20, "stop": ["\n"], "include_stop_str_in_output": true
}'
```

## Fill-in-the-middle

Code models such as Qwen2.5 (and Qwen2.5-Coder) are trained to fill a gap: given the code before and after it, they
write what goes in between. A `suffix` on `/v1/completions`, Ollama's `/api/generate` or `dllm generate --suffix`
makes the answer that middle. The prompt is built from the model's own FIM tokens,
`<|fim_prefix|> prompt <|fim_suffix|> suffix <|fim_middle|>` (StarCoder-style vocabularies spell them `<fim_prefix>`
and so on), with the prompt and the suffix tokenized separately, and the middle ends at the model's stop tokens or
at the first token that starts another FIM part, pads, or begins a new file or text (`<|fim_pad|>`,
`<|file_sep|>`, `<|repo_name|>`, `<|endoftext|>`).

- A middle is a completion like any other: the same sampler, seeds, stop sequences, logprobs, `n` choices, regexes
  and grammars, streamed or not, and the same bits on every machine. `prompt_tokens` counts the FIM prompt.
- A model without the three FIM tokens refuses a suffix with a 400 error (exit code 2 in the CLI). `echo`, token
  healing, a negative prompt and beam search cannot be combined with a suffix.
- Receipts record the suffix, `dllm replay` repeats the middle, and the
  [specification](specification.md#fill-in-the-middle) and the reference implementation (`dllm verify --reference`,
  which checks a sampled middle on models with FIM tokens) define the same bits.

```bash
dllm --model qwen2.5-0.5b.dllm generate --prompt $'def add(a, b):\n    return ' --suffix $'\n\nprint(add(1, 2))\n'
curl -s localhost:5080/v1/completions -H 'content-type: application/json' -d '{
  "prompt": "def add(a, b):\n    return ", "suffix": "\n\nprint(add(1, 2))\n", "max_tokens": 16
}'
```

## Token healing

A prompt that ends inside a word ("The quick brown fo") pins the tokenizer to a split the model rarely saw in
training, so answers continuing it read badly. With `token_healing: true` (CLI `--token-healing`) the prompt's last
token is taken back and the answer is constrained to start with that token's bytes: the model may write `fox` as one
token, or `f` then `ox`, whichever it prefers. The taken-back bytes are not repeated in the answer's text, so the text
still continues the prompt where it ended. It works the same way on raw prompts (completions, `dllm generate`, Ollama
`raw`) and on chat prefills, an assistant message at the end of the conversation (`dllm chat --prefill TEXT`).

- Healing is exact: the healing steps sample from the model's distribution restricted to the tokens whose bytes fit
  what remains of the taken-back token (either a prefix of it, or it followed by more), with the same sampler and
  seed. It combines with JSON schemas, regexes and grammars, which see the answer after the healed bytes.
- A prompt that is empty or ends in a special token (such as a chat template's turn marker) is left alone, so
  healing a chat without a prefill changes nothing.
- The taken-back token counts as a completion token: `prompt_tokens` is one less, and `completion_tokens` includes
  the healing tokens. Stop tokens cannot end the answer while it is still healing, and stop sequences match only the text after the healed bytes.
- Receipts record `token_healing`, `dllm replay` repeats it, and the [specification](specification.md#token-healing)
  and the reference implementation (`dllm verify --reference`) define the same bits. Beam search refuses it.

```bash
dllm generate --prompt "The quick brown fo" --token-healing --temperature 0.9 --seed 11
dllm chat "Name a colour." --prefill "My favourite colour is gre" --token-healing
curl -s localhost:5080/v1/completions -H 'content-type: application/json' -d '{
  "prompt": "The quick brown fo", "max_tokens": 12, "token_healing": true
}'
```

## Scoring

The log-probability of every token of a given text, with the same bits everywhere
([specification](specification.md#prompt-scoring)). Three ways to get it:

```bash
dllm --model qwen2.5-0.5b.dllm score story.txt --top 3        # per token, log-likelihood and perplexity
dllm --model qwen2.5-0.5b.dllm score story.txt --json --receipt score.json
curl -s localhost:5080/v1/completions -H 'content-type: application/json' \
  -d '{"prompt": "The capital of France is Paris.", "max_tokens": 0, "echo": true, "logprobs": 0}'
```

The scores are the logprobs generation reports for the same tokens: scoring a prompt plus an answer gives the
answer's `logprobs` exactly. `dllm score --receipt` writes a score receipt (`"score": "dllm-score/1"`, the text and
a fingerprint of every log-probability) that `dllm replay` and `POST /v1/receipts/verify` check by scoring again.
[`dllm eval`](evaluation.md) uses the same log-likelihoods for perplexity and multiple-choice tasks.

## Voting

Self-consistency: sample several answers and return the most common one. A vote is exact too
([specification](specification.md#voting)): the answers are the choices with seeds `seed + i`, each is normalised
(NFKC, lower case, whitespace collapsed; optionally only the last match of an `extract` regex counts), and the most
frequent answer wins, a tie going to the one that appeared first.

```bash
dllm chat "What is 17 * 3? End with 'Answer: N'." --temperature 0.8 --vote 7 --vote-extract "Answer: (\d+)"
curl -s localhost:5080/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [{"role": "user", "content": "What is 17 * 3?"}], "temperature": 0.8,
  "vote": {"n": 7, "extract": "(\\d+)"}, "receipt": true}'
```

The response's one choice is the winning answer in full; `vote` holds every normalised answer, the ballots (answer,
votes, choice indexes) and the winner's index, and `usage.completion_tokens` counts all the samples. With
`"receipt": true` the receipt is a vote receipt (`"vote": "dllm-vote/1"`: the request, every answer's receipt id, the
ballots and the winner), which `dllm replay` and `POST /v1/receipts/verify` check by voting again. `vote` cannot be
combined with `stream` or `n`; `n` in a vote is 1 to 16.

## Beam search

The best answers by exact log-likelihood instead of a sampled one ([specification](specification.md#beam-search)).
`beam: {"width": W, "n_best": K, "length_penalty": A}` on chat completions and completions (`dllm generate --beams W
--n-best K --length-penalty A`) keeps the W most likely partial answers, extends them together and returns the K
best finished ones as choices, best first. Every step has a fixed rule: a hypothesis's log-likelihood is the double
sum of its float32 token log-probabilities, candidates are ranked by log-likelihood and then by token sequence (a total
order, so ties never depend on sort stability), and finished answers are ranked by
`log_likelihood / length ** length_penalty`. Width 1 is greedy decoding.

```bash
dllm --model qwen2.5-0.5b.dllm generate --prompt "The capital of France is" --max-tokens 12 --beams 4 --n-best 2
curl -s localhost:5080/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [{"role": "user", "content": "Name a prime number."}], "max_tokens": 16,
  "beam": {"width": 4, "n_best": 2}, "receipt": true}'
```

The response's `beam` holds every returned answer's log-likelihood, score, finish reason and fingerprint;
`logprobs: true` gives each answer's token log-probabilities. `stop` and `max_tokens` work as usual; temperature,
top-k, top-p and seeds play no part. Penalties, logit bias, min-p, watermarks, guides, tools, structured output,
`top_logprobs`, `stream`, `n` and rolling are refused with a 400 error, since they would change what the search ranks.
With `"receipt": true` the receipt is a beam receipt (`"beam": "dllm-beam/1"`: the request, the settings and every
answer's fingerprint and score bits; on completions each choice carries its prompt's), which `dllm replay` and
`POST /v1/receipts/verify` check by searching again. A search runs `width` sequences per step, each with its own
KV cache, in one batched pass.

## Guided decoding

Decoding from a combination of several next-token distributions, each one an exactly defined float32 computation
([specification](specification.md#guided-decoding)) that the reference implementation repeats bit for bit. A request
uses at most one guide; logit bias, penalties, the watermark, temperature and the sampler then apply to the combined
logits as usual, and reported `logprobs` stay those of the model's own logits.

- **Classifier-free guidance** steers away from a negative prompt: `guidance: {"negative_prompt": "...", "scale": 1.5}`
  (`--negative-prompt`, `--guidance-scale`). On chat requests the negative prompt replaces the content of the last
  user message (the system prompt and earlier turns stay); on completions it replaces the prompt. Scale 1 is
  (up to float32 rounding) the unguided answer; larger scales push further away from the negative prompt.
- **Contrastive decoding** favours what the served model knows better than a smaller amateur model with the same
  tokenizer: start the server (or CLI) with `--contrast-model small.dllm` (`$DLLM_CONTRAST_MODEL`), then ask for
  `contrast: {"beta": 0.5, "alpha": 0.1}` (`--contrast 0.5`, `--contrast-alpha`). Tokens less likely than `alpha`
  times the top token are dropped.
- **Ensembles** average the log-probabilities of models that share a tokenizer, for every request:
  `--ensemble-model other.dllm=0.5` (repeatable; `$DLLM_ENSEMBLE_MODELS` holds `PATH[=WEIGHT]` items separated by
  the path separator) and `--ensemble-weight` for the served model (default 1). An ensemble changes the
  `system_fingerprint`, so receipts and cached answers of the plain model never mix with it, and it cannot be
  combined with a per-request guide.

```bash
dllm generate --prompt "Once upon a time" --temperature 0.7 --seed 3 --negative-prompt "It was a dark night" --guidance-scale 3
dllm --model qwen2.5-1.5b.dllm --contrast-model qwen2.5-0.5b.dllm chat "Explain entropy" --contrast 0.5
curl -s localhost:5080/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [{"role": "user", "content": "Write a cheerful sentence"}], "temperature": 0.7, "seed": 3,
  "guidance": {"negative_prompt": "Write a sad sentence", "scale": 2}}'
```

Each guide runs one more forward pass per token and model, with its own KV cache. Guided requests do not use
speculative decoding (the answer is the same either way) and cannot roll the context window
([long conversations](#long-conversations)): they stop when the longest context is full. Receipts record the guide,
so `dllm replay` repeats it.

## Reasoning

Thinking models (Qwen3 and others whose chat template writes `<think>` blocks) get their reasoning separated from the
answer, by a fixed rule ([specification](specification.md#reasoning)), in streamed and non-streamed responses alike:

| | OpenAI chat | Responses | Anthropic | Ollama | CLI (`chat`) |
| --- | --- | --- | --- | --- | --- |
| The reasoning | `message.reasoning_content`, streamed as `delta.reasoning_content` | a `reasoning` output item (`reasoning_text`), streamed as `response.reasoning_text.delta` | a `thinking` content block, streamed as `thinking_delta` | `message.thinking` (`thinking` for `/api/generate`) | on stderr after `thinking:` |
| Thinking off | `reasoning_effort: "none"` or `chat_template_kwargs: {"enable_thinking": false}` | `reasoning: {"effort": "none"}` | `thinking: {"type": "disabled"}` | `think: false` | `--no-think` |
| Thinking on | any other `reasoning_effort` | any other effort | `enabled` or `adaptive` | `think: true` or an effort | `--think` |
| Budget | `max_reasoning_tokens` (extension) | `max_reasoning_tokens` (extension) | `thinking.budget_tokens` | `max_reasoning_tokens` (extension) | `--max-reasoning-tokens N` |
| Usage | `completion_tokens_details.reasoning_tokens` | `output_tokens_details.reasoning_tokens` | part of `output_tokens` | part of `eval_count` | |

- Without a switch the model does what its template does by default (Qwen3 thinks). Switching thinking off goes
  through the model's own template (`enable_thinking`), so the prompt is exactly what the model's authors render.
- A budget is a token count, never a time: when the thinking has taken `max_reasoning_tokens` tokens, the engine
  closes the block with fixed tokens and the answer follows, so the same request gives the same bits everywhere.
- Structured output and forced tool calls answer at once (their grammar applies from the first token), so they are not
  split; switch thinking off for them on models that think by default.
- The reasoning is not part of the conversation history (Qwen3's template drops it too). Receipts record the switch
  and the budget and hash the reasoning, so `dllm replay` repeats them. Anthropic's `signature` is a hash of the
  thinking, derived from content. Responses without reasoning are unchanged, field for field.

```bash
curl -s localhost:5080/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [{"role": "user", "content": "Is 391 prime?"}],
  "max_tokens": 512, "max_reasoning_tokens": 256
}'
```

## Long conversations

A prompt and its answer never hold more tokens than the model's context window (`context_length` in `dllm inspect`).
What happens at the edge is fixed by the request, never by load or timing ([specification](specification.md#the-context-window)):

| Behaviour | OpenAI chat | Responses | Ollama | CLI |
| --- | --- | --- | --- | --- |
| A prompt that does not fit is refused | 400, `code: context_length_exceeded` | 400, `code: context_length_exceeded` | 400 | exit code 2 (`generate`), 1 (`chat`) |
| Drop the oldest turns until it fits | `truncation: "auto"` (extension) | `truncation: "auto"` | `truncate: true` | `chat --truncate` |
| A full window ends the answer (`finish_reason: length`) | default | default | default | default |
| Keep going on a rolled window | `context_overflow: "roll"` (extension) | `context_overflow: "roll"` (extension) | `shift: true` | `--context-overflow roll` |

- Truncation drops the earliest message that is neither a system message nor the last one, together with the tool
  results right after it, and repeats until the prompt fits. System messages and the last message always stay.
- Rolling keeps the first 4 tokens (attention sinks) and the latest half window, and computes them afresh. Each
  token after a roll is exactly the token a new request over the kept tokens would give, with and without the
  prompt cache and in a batch, so a rolled answer is as reproducible as any other. The penalties keep counting over
  the whole history.
- Both are off by default (unlike Ollama, where truncation and shifting are on), so a request that fits is unchanged
  and a request that does not fit says so. Receipts record both, so `dllm replay` repeats them.

```bash
curl -s localhost:5080/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [{"role": "user", "content": "Tell me a long story"}],
  "max_tokens": 100000, "truncation": "auto", "context_overflow": "roll"
}'
```

## Batches

`POST /v1/files` and `/v1/batches` run OpenAI batch files with output that is the same bytes on every machine and at
every concurrency; `dllm batch` does the same from the command line. See [batch jobs](batches.md).

## Tools

OpenAI `tools`/`tool_choice` and Anthropic `tools`/`tool_choice` map onto one mechanism. Models call tools in the
format they were trained on, read from the model's own chat template:

| Format | Models | A call looks like |
|---|---|---|
| `hermes` | Qwen2.5, Qwen3, Hermes and most tool-trained small models | `<tool_call>{"name": "get_weather", "arguments": {...}}</tool_call>` |
| `llama3` | Llama 3.1, 3.2 and 3.3 | the whole reply is `{"name": "get_weather", "parameters": {...}}` |
| `mistral` | Mistral, Mixtral, Ministral | `[TOOL_CALLS][{"name": "get_weather", "arguments": {...}}]` |
| `granite` | IBM Granite 3 | `<|tool_call|>[{"name": "get_weather", "arguments": {...}}]` |
| `xml` | Qwen3-Coder | `<tool_call>` `<function=get_weather>` `<parameter=city>` Paris `</parameter>` `</function>` `</tool_call>`, one tag per line |
| `deepseek` | DeepSeek V3 and R1, the R1 distills of Qwen and Llama | `<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather`, the arguments in a fenced `json` block, `<｜tool▁call▁end｜><｜tool▁calls▁end｜>` |
| `pythonic` | Llama 3.2/4-style templates that ask for Python calls | the whole reply is `[get_weather(city="Paris"), is_noon()]` |

- The format is detected from the template in a fixed order: `[TOOL_CALLS]` means Mistral, DeepSeek's
  `<｜tool▁calls▁begin｜>` DeepSeek, `<|tool_call|>` Granite, `<function=` the XML format, `<tool_call>` Hermes, the
  Python call example `func_name1(` the Python format, and a template with an `ipython` role and a `"parameters"` key
  Llama 3; anything else is Hermes. `DllmEngine.tool_format` shows the result and may be set (`"hermes"`,
  `"llama3"`, `"mistral"`, `"granite"`, `"xml"`, `"deepseek"`, `"pythonic"`) for a template the detection gets
  wrong.
- The XML format (Phase 52) writes string parameters as raw text (the grammar keeps them free of `<`, so they cannot
  run into the closing tag) and every other value as JSON; parameters come in schema order, required ones always.
  DeepSeek's arguments are a JSON object in a fenced block, several calls one per line between the begin and end
  markers. Python calls are keyword arguments in schema order with Python's `True`, `False` and `None`; other values
  are JSON (valid Python literals), and the parser reads double-quoted strings as JSON strings and accepts JSON's
  `true`/`false`/`null` inside lists and objects. Templates that expect earlier calls' arguments as JSON text
  (DeepSeek's) get them as text.
- If the model's chat template supports tools, it presents them, exactly as `transformers` would. Otherwise the
  engine adds the Hermes tool instructions to the system message, shows earlier calls as `<tool_call>` blocks and
  tool results as `<tool_response>` user turns, and the model calls tools in the Hermes format.
- Mistral's, Granite's and DeepSeek's markers are special tokens, which decoding normally leaves out; the engine keeps
  the format's own markers in the generated text, so the constraint and the parser see them.
- `auto`: the model may answer in text. Once it writes the format's marker, the call is constrained by a grammar
  built from the tool definitions (a known name and arguments that fit its `parameters`/`input_schema`, leniently:
  keywords the grammar cannot check are ignored for tools). Several calls in one answer are fine (one `<tool_call>`
  block each, or several entries in the Mistral, Granite, DeepSeek and Python lists). Llama 3 and Python calls have
  no marker, so their answer is constrained from the first token: either text that does not start with `{` (`[` for
  Python calls), or exactly one call (one list).
- `required` (OpenAI) / `any` (Anthropic), or a named tool: the answer is exactly one call (one list for Mistral
  and Granite), constrained from the first token.
- `none`: the tools are not shown to the model.
- In every format, a reply that is nothing but a JSON object `{"name": ..., "arguments"|"parameters": ...}` naming
  a known tool also counts as a call.
- `finish_reason` is `tool_calls` (OpenAI) and `stop_reason` is `tool_use` (Anthropic) when the answer has calls.
  Send results back as `tool` messages (OpenAI) or `tool_result` blocks (Anthropic).
- Call ids come from the request and the call's position. Mistral's template accepts only ids of nine letters and
  digits, so earlier calls and their results are shown to it with nine-character ids derived from the ids you send
  (`tools.template_id`); the ids in the API stay as they are.

When streaming with tools, text is streamed until the model starts a tool call (or starts its answer with `{`, which
could be a bare JSON call, or `[` in the Python format); the rest is decided when generation ends.

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
`--index-mode` chooses dense, lexical (BM25) or hybrid search, and `--rerank-model` reranks the passages first.
`/v1/rerank` ranks documents for a query with the served model ([reranking](retrieval.md#reranking)).

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
