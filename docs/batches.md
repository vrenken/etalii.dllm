# Batch jobs

`dllm batch` and the OpenAI Files and Batches API run many requests at once and give an output file that is the
same bytes on every machine, at every concurrency, and after any interruption (Phase 22).

```bash
dllm --model smollm2-135m.dllm batch requests.jsonl -o results.jsonl --workers 8
dllm --model smollm2-135m.dllm batch requests.jsonl -o results.jsonl --verify --sample 20
```

## Input and output

The input is an [OpenAI batch file](https://platform.openai.com/docs/guides/batch): one JSON object per line with a
unique `custom_id`, `"method": "POST"`, a `url` (`/v1/chat/completions` or `/v1/embeddings`) and the request `body`.

```json
{"custom_id": "q1", "method": "POST", "url": "/v1/chat/completions", "body": {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 32}}
```

The output has one line per input line, in input order:

```json
{"id": "batch_req_...", "custom_id": "q1", "response": {"status_code": 200, "request_id": "req_...", "body": {...}}, "error": null}
```

- `body` is exactly what the HTTP endpoint answers for that request (`POST /v1/chat/completions`), with every
  [decoding control](api.md#decoding-controls) and `"receipt": true` available.
- A request the endpoint rejects has its 400 answer as the `response`. A line that is not a valid request (not JSON, a
  duplicate or missing `custom_id`, another url or method, `"stream": true`) has `"response": null` and an `error`.
  Neither stops the batch.
- `id` is a hash of the system fingerprint, the line number and the line itself.

## Why the bytes never change

- Requests run concurrently (`--workers N`, default 1). On a model file they share batched decoding steps, which never
  changes a bit ([concurrent requests](api.md#concurrent-requests)).
- Lines are written in input order as soon as the next one is ready, never in the order requests finish.
- Every id is derived from content, and responses hold no clock values.

So the output's SHA-256 is the same for `--workers 1` and `--workers 64`, on Linux, Windows and macOS. The test suite
pins one (`BATCH_OUTPUT_SHA256`).

## Resuming

Run the same command again after an interruption (a killed process, a crash, a full disk). `dllm batch` checks every
finished line's `id` against the input and the model, keeps them, drops a torn last line and computes the rest. The
finished file is byte-identical to an uninterrupted run. An output written for another input or another model is
refused instead of being overwritten.

## Digests and verification

A finished batch writes `results.jsonl.digest.json`:

```json
{"format": "dllm-batch/1", "engine": "0.2.0", "system_fingerprint": "fp_...", "input_sha256": "...",
 "output_sha256": "...", "requests": 7, "completed": 7, "failed": 0, "digest": "bdig_..."}
```

With `--sign-key` the digest is signed ([provenance](provenance.md)). `dllm batch IN -o OUT --verify` re-runs the
requests (`--sample K`: K evenly spread lines) and compares each result with `OUT` byte for byte. It also checks the
digest against the files, so anyone with the same weights can confirm a batch result.

## The Files and Batches API

`dllm-server` serves the OpenAI endpoints the SDK's batch workflow uses:

| Endpoint | |
| --- | --- |
| `POST /v1/files` | Upload (multipart, `purpose` `batch`). The id is `file-` + the SHA-256 of the bytes. |
| `GET /v1/files`, `GET /v1/files/{id}`, `GET /v1/files/{id}/content`, `DELETE /v1/files/{id}` | List, describe, download, delete. |
| `POST /v1/batches` | Start a batch (`input_file_id`, `endpoint`, `completion_window`, `metadata`). The id is a hash of these and the system fingerprint, so posting the same batch again returns it. |
| `GET /v1/batches`, `GET /v1/batches/{id}`, `POST /v1/batches/{id}/cancel` | Poll, list, cancel. |

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:5080/v1", api_key="unused")
file = client.files.create(file=open("requests.jsonl", "rb"), purpose="batch")
batch = client.batches.create(input_file_id=file.id, endpoint="/v1/chat/completions", completion_window="24h")
# poll client.batches.retrieve(batch.id) until status == "completed", then:
results = client.files.content(client.batches.retrieve(batch.id).output_file_id).read()
```

- The output file holds the bytes `dllm batch` writes, so its id is the same on every server.
- A completed batch carries its `digest`. Timestamps are 0.
- What a poll sees (`in_progress` or `completed`) depends on timing, and nothing else does.
- A cancelled batch keeps the lines it finished. Posting it again resumes it.
- Files and batches live in `DLLM_BATCH_DIR` when it is set, where they survive restarts. Otherwise they live in a
  temporary directory.
