# Generation receipts

Every answer EtAlii.Dllm gives is a pure function of the weights, the engine version and the request. A **receipt**
writes exactly those inputs down, next to hashes of the output, so anyone with the same weights can re-run the
request later, on any supported machine, and check that the answer is the same, bit for bit.

## Asking for a receipt

| Front end | How |
| --- | --- |
| `dllm chat` | `--receipt FILE` writes the receipt as JSON |
| OpenAI `/v1/chat/completions` | `"receipt": true` (with the SDK: `extra_body={"receipt": True}`); the response, or the finishing chunk when streamed, gets a `receipt` field |
| OpenAI `/v1/responses` | `"receipt": true`; the response object (`response.completed` when streamed) gets `receipt` |
| Anthropic `/v1/messages` | `"receipt": true`; the message, or the `message_delta` event when streamed, gets `receipt` |
| Ollama `/api/chat`, `/api/generate` | `"receipt": true`; the final (`done`) object gets `receipt` |
| MCP `chat` tool | `receipt: true`; the result is JSON `{"content", "receipt"}` |
| Python | `engine.chat_completion(request).receipt`, or the `Finished` event of `engine.chat_stream` |

Asking for a receipt changes nothing else: the answer and the response id are the same with or without it, and a
streamed answer carries the same receipt as the non-streamed one.

## What is in it

```json
{
  "receipt": "dllm/1",
  "id": "rcpt_6f1c...",
  "engine": "0.2.0",
  "model": "SmolLM2-135M-Instruct",
  "system_fingerprint": "fp_9b2e...",
  "request": {
    "messages": [{"role": "user", "content": "Hi", "tool_calls": [], "tool_call_id": null, "name": null}],
    "prompt": null, "max_tokens": 256,
    "options": {"temperature": 0.0, "top_k": 0, "top_p": 1.0, "seed": 0},
    "stop": [], "tools": [], "tool_choice": {"mode": "auto", "name": null},
    "response_format": {"type": "text", "schema": null},
    "top_logprobs": null, "call_id_prefix": "call_", "request_id": "chatcmpl-..."
  },
  "output": {
    "tokens": "3a7f...", "content": "e4c2...", "tool_calls": "4f53...",
    "finish_reason": "stop", "prompt_tokens": 31, "completion_tokens": 9
  }
}
```

- `system_fingerprint` identifies the weights and everything else that changes the bits: quantisation, LoRA
  adapters, steering vectors and the document index (`--index`).
- `request` is the engine request after the front end translated it, so a receipt from the Anthropic API replays
  the same way as one from the CLI. A grounded chat (`--index`) records the conversation before the retrieved
  passages are added; the index is part of the fingerprint.
- `output.tokens` is the SHA-256 fingerprint of the generated token ids (the one `dllm generate` prints),
  `output.content` the SHA-256 of the answer text and `output.tool_calls` of the canonical JSON of the tool calls.
- `id` is `rcpt_` plus the first 32 hex digits of the SHA-256 of the canonical JSON (sorted keys, no spaces) of
  everything else, so any edit to a receipt is detected.

Nothing comes from a clock or a random source: the same request gives the same receipt, byte for byte. Settings that
never change a bit are not recorded: thread count, device (CPU or GPU), prompt cache, speculative decoding and batching.

A receipt holds the conversation in plain text. Treat it like the conversation itself.

## Checking a receipt

```bash
dllm --model smollm2-135m.dllm chat "Name three colours." --receipt colours.json
dllm --model smollm2-135m.dllm replay colours.json          # exit code 0: verified, 1: differs
dllm --model smollm2-135m.dllm replay colours.json --json   # {"ok", "reasons", "notes", "receipt"}
curl -s localhost:5080/v1/receipts/verify -d @colours.json  # the same check on a running server
```

The MCP server has a `verify_receipt` tool for the same check. A replay fails, with the reasons listed, when

- the receipt was edited (its id does not match its content);
- the engine runs other weights or settings (another `system_fingerprint`; pass the same `--model`, `--quantize`,
  `--adapter`, `--steer` and `--index` options as when the receipt was made);
- the replayed output differs: the prompt, the tokens, the text, the tool calls, the finish reason or the number of
  tokens.

The request is replayed in every case, so the reasons say what actually differs. Another engine version is only a
note: versions that keep the kernels and the sampler give the same output, and the output hashes decide.

## Limits

- A receipt proves that these weights give this answer to this request. It does not prove who ran the request; sign
  receipts with your own key if you need that.
- With the MCP host (`dllm chat --mcp-server`), the receipt covers the last round: its request holds the tool
  results the earlier rounds produced.
- Logprobs are not hashed; they follow from the same computation as the tokens.
