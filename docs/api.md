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
`#/definitions/...`, recursion allowed) and `nullable`. Annotations such as `title` and `description` are ignored.
Keywords the automaton does not check (`multipleOf`, `uniqueItems`, `minProperties`, bounds on non-integer numbers,
...) are refused with a 400 error rather than silently ignored. Properties are generated in schema order; optional
ones may be skipped. At most 16 whitespace bytes may appear between JSON tokens.

Value constraints are exact too (Phase 31), each compiled to a byte automaton:

| Keyword | Rule |
| --- | --- |
| `pattern` | The [regex syntax](#decoding-controls) of `guided_regex`, found anywhere in the string as JSON Schema says, unless a leading `^` or trailing `$` anchors it. `^a\|b` is refused as ambiguous: group the alternation. |
| `format` | `date` (`YYYY-MM-DD`, month 01 to 12, day 01 to 31), `time` (`hh:mm:ss`, optional fraction, `Z` or an offset), `date-time` (both, joined by `T`), `uuid` and `ipv4`. Other formats stay annotations. |
| `minLength`, `maxLength` | Counted in code points. |
| `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum` | On integers, as numbers (or the draft 4 booleans). The answer is the decimal text of an integer in the range, never `-0`. |

A constrained string is written without escape sequences, so it holds no quote, backslash or control character. Its
automaton is the intersection of the automata of each keyword, with every state that cannot reach a match removed, so
decoding never runs into a dead end; a schema no string can satisfy is refused with a 400 error. Tool parameters only
guide the model, so their value constraints are ignored.

```bash
dllm chat "Book a flight" --json-schema '{"type": "object", "properties": {"from": {"type": "string", "pattern": "^[A-Z]{3}$"},
  "date": {"type": "string", "format": "date"}, "seats": {"type": "integer", "minimum": 1, "maximum": 9}},
  "required": ["from", "date", "seats"]}'
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
| [Guided decoding](#guided-decoding): a negative prompt, or contrast against an amateur model | `guidance: {negative_prompt, scale}`, `contrast: {beta, alpha}` (extensions; also on Anthropic messages and completions) | | `--negative-prompt`, `--guidance-scale`, `--contrast BETA`, `--contrast-alpha` |
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

## Completions API

`POST /v1/completions` continues a prompt as it is, without a chat template, on the same event stream as every other
endpoint: a completion equals `dllm generate` with the same prompt and options, bit for bit. `prompt` is a string or
a list of strings (choice `index` is `prompt_index * n + i`); `max_tokens` defaults to 16 as in OpenAI's API and
`temperature` to 0 as everywhere here. `stop`, `seed`, `n`, the [decoding controls](#decoding-controls),
`guided_regex`, `watermark`, `stream` (with `stream_options.include_usage`) and `receipt` (a receipt on each choice)
work as on chat completions. `suffix` and a `best_of` other than `n` are refused with a 400 error.

`logprobs: N` returns OpenAI's `logprobs` object (`tokens`, `token_logprobs`, `top_logprobs` with N alternatives,
`text_offset` in characters). With `echo: true` the prompt comes first in `text` and in `logprobs`, its tokens
[scored exactly](#scoring) (the first prompt token has `null`, as in OpenAI's API).

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
