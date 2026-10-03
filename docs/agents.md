# Reproducible agents

An agent run is a chat in which the model calls tools in a loop ([MCP host](mcp.md#client-host-the-model-calls-mcp-tools)).
The model's side of that loop is as deterministic as any chat; only the tool results come from outside. EtAlii.Dllm
records those results in a **transcript**, so anyone can replay the whole run offline, bit for bit, and find the first
round where a different run diverges. With the **built-in tools** even the tool results repeat, and **receipt
chains** do the same for an ordinary multi-turn conversation over the HTTP APIs.

## Transcripts

```bash
dllm --model qwen2.5-0.5b.dllm chat "What is 12.5 times 8.25?" --tool calculator --transcript run.json
dllm --model qwen2.5-0.5b.dllm replay run.json          # exit code 0: every round verified, 1: differs
dllm --model qwen2.5-0.5b.dllm replay run.json --json   # {"ok", "reasons", "notes", "diverged_at", "transcript"}
```

`--transcript FILE` works with every MCP tool source (`--mcp-config`, `--mcp-server`, `--tool`). The file is plain
JSON, with nothing from a clock or a random source, so the same run writes the same bytes:

```json
{
  "transcript": "dllm-agent/1",
  "id": "trn_...",
  "engine": "0.2.0", "model": "Qwen/Qwen2.5-0.5B-Instruct", "system_fingerprint": "fp_...",
  "request": {"messages": [...], "max_tokens": 256, ...},
  "tools": [{"name": "calculate", "description": "...", "parameters": {...}, "server": "tools"}],
  "max_rounds": 8,
  "rounds": [
    {"receipt": {...}, "content": "", "tool_calls": [{"id": "call_...", "name": "calculate", "arguments": "..."}],
     "results": [{"id": "call_...", "name": "calculate", "server": "tools", "content": "825/8 = 103.125",
                  "is_error": false}]},
    {"receipt": {...}, "content": "12.5 times 8.25 is 103.125.", "tool_calls": [], "results": []}
  ]
}
```

- `request` is the starting engine request, in the same form as a [receipt](receipts.md)'s.
- Every round holds its [receipt](receipts.md), the assistant text and tool calls in full, and the tool results.
- `id` is `trn_` plus the first 32 hex digits of the SHA-256 of the canonical JSON of everything else, so an edit
  is detected.
- `server_requests` (only when a server asked) lists every [sampling and elicitation](mcp.md#elicitation-and-roots)
  request the engine answered during the run, in the order they arrived: the kind, the server, the engine request
  it became (as in a receipt), the answer and its fingerprint (`message` and `action` for elicitations; a declined
  elicitation has no request).

`dllm replay` recognises a transcript by its `transcript` key. It runs the same loop again, but no MCP server is
started: every tool call is answered with the recorded result of the same position in the run. It then compares
each round's receipt with the recorded one and fails, with the reasons listed, when

- the transcript was edited (its id does not match its content), or a round's text or calls do not match its own
  receipt;
- the engine runs other weights or settings (another `system_fingerprint`);
- a round diverges: `round N diverges: its conversation differs` when the request differs (an edited tool result,
  say), `the model's answer differs` when only the output does; `diverged_at` is that round;
- the run has a different number of rounds;
- `server request N (sampling|elicitation for SERVER): the engine's answer differs`: replay runs each recorded
  server request again on the engine (offline: the request is in the transcript) and compares the answer.

From Python: `transcripts.Recorder` builds a transcript from the events of `mcp_host.chat`, `transcripts.replay`
checks one, and `transcripts.RecordedTools` stands in for an `McpHost` with the recorded results.

## Built-in tools

Tools whose results are deterministic, so an agent run with them repeats bit for bit on every machine, with no
external server:

| `--tool` | MCP tools | What it does |
| --- | --- | --- |
| `calculator` | `calculate(expression)` | Exact rational arithmetic: `+ - * / // % **` (integer exponents), parentheses, decimal literals. Decimals are taken as written (`0.1 + 0.2` is exactly `3/10`); the answer is the exact fraction and its decimal expansion, truncated after 30 digits and marked `≈` when it does not terminate. Never a binary float. |
| `files=DIR` | `list_files(path)`, `read_file(path, start, lines)` | Read-only access to `DIR`: listings sorted by name (directories end with `/`), at most 200 lines per read; paths outside `DIR` are refused. |
| `documents` | `search_documents(query, top)` | The exact search over the engine's [document index](retrieval.md) (`--index`). |

```bash
dllm --model qwen2.5-0.5b.dllm chat "What is in notes/todo.md?" --tool files=. --tool calculator
dllm-tools --tool calculator --tool files=.     # the same tools as an MCP server over stdio, for any MCP client
```

`--tool` can be repeated and combined with `--mcp-config` and `--mcp-server`; the built-in tools are served as the
in-process MCP server `tools`. A tool that cannot answer (a division by zero, an exponent over 1024, a result over
16384 bits, a path outside the directory) gives an error result the model can read, as any MCP tool does.

## Receipt chains

A multi-turn conversation over the HTTP APIs gets one receipt per turn. Linking them lets anyone check the whole
conversation: each turn's receipt names the receipt of the turn before as `previous`.

| Front end | How |
| --- | --- |
| OpenAI `/v1/responses` | Automatic: a response with `previous_response_id` chains its receipt to the stored response's |
| OpenAI `/v1/chat/completions`, Anthropic `/v1/messages`, Ollama `/api/chat` and `/api/generate` | `"previous_receipt": "rcpt_..."` next to `"receipt": true` |
| MCP `chat` tool | `previous_receipt: "rcpt_..."` |
| Python | `ChatRequest(..., previous_receipt="rcpt_...")` |

`previous_receipt` changes nothing but the receipt: the answer and the response id stay the same. Save the receipts
of a conversation as a JSON list, oldest first, and check them all at once:

```bash
dllm --model smollm2-135m.dllm replay conversation.json   # exit code 0: every turn verified and linked
curl -s localhost:5080/v1/receipts/verify -d @conversation.json
```

`receipts.verify_chain` verifies every receipt and checks that each turn

- names the receipt before it as `previous`;
- continues the previous turn exactly: its messages start with the previous turn's messages, followed by an
  assistant message whose text and tool calls hash to the previous receipt's output. A leading system message may
  differ between turns, since the Responses API's `instructions` do not carry over.

So an answer that was edited before it was sent back is caught, even though every receipt on its own verifies. A
chain whose first receipt names a `previous` it does not include is checked from there, with a note.
