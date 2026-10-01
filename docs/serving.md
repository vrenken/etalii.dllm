# Serving at scale

An answer from EtAlii.Dllm is a pure function of the weights, the engine version and the request. Ordinary LLM
servers can only approximate some things, and that purity makes them exact:

- a **response cache** that is never stale and never approximate;
- **coalescing**: identical requests that arrive together share one generation;
- **audits** that check a server, or a whole fleet, really gives the same bits.

None of these changes a bit of any answer. Receipts, response ids and tool call ids stay the same too.

## The response cache

```bash
dllm-server --model qwen2.5-0.5b.dllm --response-cache /var/cache/dllm     # or $DLLM_RESPONSE_CACHE
dllm --model qwen2.5-0.5b.dllm --response-cache ~/.cache/dllm chat "Hi"   # the CLI and dllm-mcp too
dllm cache stats /var/cache/dllm
dllm cache clear /var/cache/dllm
```

Every finished answer is stored under a **content key**. A repeated request is then answered from storage instead of
running the model. The key is a SHA-256 over everything that can change the output:

- the engine version;
- the full weights fingerprint, which includes quantisation and steering;
- the system fingerprint;
- the document index, if one is used;
- the engine request. That is everything a [receipt](receipts.md) records about the request: messages, sampling
  settings and seed, stop sequences, tools, the response format, logprobs and the request id that tool call ids come
  from.

Change any of them and the key changes, so a cached answer can never be wrong for the request it answers. Nothing
needs invalidating. A new model or engine version simply uses new keys. The receipt chain link (`previous_receipt`)
is left out of the key because it never changes the output. A cached answer still gets its own receipt with the
right `previous`, signed with `--sign-key` if one is set.

A cached answer is the same stream of events as a fresh one, logprobs included. The only visible difference is
`usage.prompt_tokens_details.cached_tokens` (and its equivalents in the other APIs), which then covers the whole
prompt. With `--prompt-cache 0` the counters are otherwise 0, so check them if you need to know whether an answer
came from the cache.

On disk, each answer is one file, `<key>.response.json`:

- it holds the events as canonical JSON, with floats as exact hex strings, plus a SHA-256 of the content;
- it is written to a temporary name and renamed into place, so a crash never leaves a half-written file;
- a damaged file is ignored and rewritten;
- several servers can share one directory: they all write the same bytes.

Replays never use the cache, because they must run the model: `dllm replay`, `POST /v1/receipts/verify` and audits
all bypass it.

## Coalescing

When identical requests arrive while the first is still generating, they share its generation. Each request sees
every event from the start, so a follower that arrives late first catches up with what has been generated so far.
Whichever reader is furthest ahead drives the model, so a client that disconnects holds up no one. Coalescing is
always on. It changes nothing but speed, and `GET /v1/audit` counts how often it happened (`coalesced`).

## Self-audit

```bash
dllm-server --model qwen2.5-0.5b.dllm --audit-every 100     # or $DLLM_AUDIT_EVERY
curl localhost:5080/v1/audit
```

```json
{
  "audit": {"every": 100, "checked": 12, "passed": 12, "failed": 0, "failures": []},
  "response_cache": {"hits": 40, "misses": 1200, "responses": 1200, "bytes": 5341120},
  "coalesced": 3
}
```

With `--audit-every N`, a server re-runs about one response in N on a background thread, after answering it. The
re-run bypasses the response cache and coalescing. It then compares the outputs the way `dllm replay` does. A
mismatch lists the receipt id and what differed. The last 100 mismatches are kept. Which responses get checked
follows from their receipt ids (the hash modulo N), not from a random choice. So replicas that serve the same
traffic check the same responses, and a run can be repeated.

A mismatch on one machine means something is wrong with the machine or the build: memory errors, a broken
installation, or a determinism bug worth reporting.

## Auditing a fleet

```bash
dllm audit --url http://replica-1:5080 --url http://replica-2:5080 --prompts prompts.txt
dllm --model qwen2.5-0.5b.dllm audit --local --url http://replica-1:5080 --prompts prompts.txt --json
```

```text
request 1: same (a3f09c...)
request 2: same (5be1d2...)
2 targets, 2 requests: the same bits
```

`dllm audit` sends every request in `prompts.txt` to every target and compares the receipts. A line is either a user
message or a whole chat completions request in JSON, for example
`{"messages": [...], "temperature": 0.8, "seed": 3}`. `--local` adds this machine's own engine as a target, using
`--model` and the other runtime options, and runs it fresh. The receipts are compared on:

- the system fingerprint;
- the prompt size;
- the token hash;
- the text, naming the first character that differs;
- the tool calls;
- the finish reason.

The exit code is 0 when every target gave the same bits for every request, 1 when they did not, and 2 for a usage
error. This makes it a deployment check: run it after rolling out a new machine, a new GPU or a new build, against a
replica you trust.
