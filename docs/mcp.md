# MCP, both directions

EtAlii.Dllm speaks the [Model Context Protocol](https://modelcontextprotocol.io) both ways, with the official `mcp`
Python SDK:

- **Server** (`dllm-mcp`): other MCP clients (Claude Code, Claude Desktop, IDEs) use the deterministic model through
  tools, resources and prompts.
- **Client host** (`dllm chat --mcp-config`, `etalii_dllm.mcp_host`): the deterministic model calls the tools of
  external MCP servers during a chat.

## Server: `dllm-mcp`

```bash
claude mcp add dllm -- dllm-mcp --model /absolute/path/to/smollm2-135m.dllm
```

Runs over stdio. `--model` (or `DLLM_MODEL`) selects the model; without it the placeholder model is served.

### Tools

| Tool | Arguments | Returns |
| --- | --- | --- |
| `chat` | `messages` (`[{"role", "content"}]`), `max_tokens` (256), `temperature` (0), `seed` (0), `json_schema` (optional) | The answer, rendered with the model's chat template; with `json_schema`, JSON valid under it (constrained decoding) |
| `generate` | `prompt`, `max_tokens` (64), `temperature` (0), `seed` (0) | A continuation of the prompt |
| `model_info` | | Model id and system fingerprint |

All three are annotated read-only and idempotent: the same arguments always return the same text.

### Resources

| URI | Type | Content |
| --- | --- | --- |
| `dllm://model` | `application/json` | Model card: id, system fingerprint, vocabulary size, and for an imported model its architecture, source (repository, revision) and licence (SPDX id, attribution; the full text stays in the file) |
| `dllm://model/chat-template` | `text/plain` | The Jinja chat template conversations are rendered with (or a description of the fixed fallback format) |
| `dllm://determinism` | `text/markdown` | What the reproducibility guarantee covers and what may differ between machines |

### Prompts

Ready-made tasks whose message you send to the `chat` tool (or to any model):

| Prompt | Arguments | Message |
| --- | --- | --- |
| `summarize` | `text`, `max_words` (60) | Summarize the text in at most `max_words` words |
| `translate` | `text`, `language` | Translate the text, answering with the translation only |
| `extract_json` | `text`, `fields` (comma-separated) | Answer with one JSON object with exactly these keys; pair it with the `chat` tool's `json_schema` to guarantee the shape |

## Client host: the model calls MCP tools

```bash
# mcp.json in the format Claude Desktop and Claude Code use
cat > mcp.json <<'JSON'
{
  "mcpServers": {
    "time": {"command": "uvx", "args": ["mcp-server-time", "--local-timezone", "Europe/Amsterdam"]},
    "docs": {"url": "https://example.com/mcp"}
  }
}
JSON
dllm --model qwen2.5-0.5b.dllm chat "What time is it in Amsterdam?" --mcp-config mcp.json

# or name servers on the command line (repeatable): [NAME=]COMMAND ARGS... or [NAME=]URL
dllm --model qwen2.5-0.5b.dllm chat "What time is it?" --mcp-server "time=uvx mcp-server-time"
```

The host starts every stdio server (or connects to every Streamable HTTP URL), lists their tools and offers them to
the model like any function tools (see [tool calling](api.md)). Then it loops:

1. The model answers; constrained decoding guarantees that every call names a real tool with arguments that fit its
   input schema.
2. If the answer contains calls, the host runs them one at a time, in the order the model made them, and appends
   the assistant turn and one `tool` message per result. A tool that fails (the server reports an error, raises, or
   the tool is unknown) gives an `Error: ...` result the model can read; the chat does not stop.
3. Repeat until the model answers without a call, or `--max-tool-rounds` (8) rounds have run.

The answer streams to stdout; each call (`-> name(arguments)`) and result (`<- result: ...`) is printed to stderr,
followed by the last round's fingerprint.

From Python:

```python
import anyio

from etalii_dllm import mcp_host
from etalii_dllm.chat import ChatMessage
from etalii_dllm.engine import ChatRequest, DllmEngine, TextDelta


async def main() -> None:
    engine = DllmEngine.from_model_file("qwen2.5-0.5b.dllm")
    request = ChatRequest([ChatMessage("user", "What time is it in Amsterdam?")], max_tokens=256)
    async with mcp_host.McpHost(mcp_host.load_config("mcp.json")) as host:
        async for event in mcp_host.chat(engine, request, host):
            if isinstance(event, TextDelta):
                print(event.text, end="")
            elif isinstance(event, mcp_host.ToolResult):
                print(f"\n[{event.call.name} -> {event.content}]")


anyio.run(main)
```

`chat` yields the engine's own events (`TextDelta`, `ToolCallEvent`, `Finished`) for every round, and a `ToolResult`
after each executed call. Tools in the request itself (`ChatRequest.tools`) are offered next to the MCP tools but
not executed: a call to one ends the chat with finish reason `tool_calls`, as in the HTTP API. A `required` or named
`tool_choice` applies to the first round only, so the model can answer once it has the result.

### Tool names

Tools keep their own names. A name offered by more than one server is qualified with the server name
(`time.now`, `clock.now`), for every server that offers it.

### Determinism

The model side is exactly as reproducible as a normal chat:

- Servers are connected in name order and tools are listed in (server, tool) order, so the prompt that presents
  them is the same on every run, whatever order a server lists them in.
- Calls run sequentially in the order the model made them; tool call ids are derived from the conversation and the
  round, never from a clock.
- So the same conversation plus the same tool results always gives the same answer, token for token.

External tools are outside that guarantee: a clock, a search engine or a database can return something different on
the next run, and the answer after it then differs too. Every `ToolResult` carries what the tool returned, so a
run can be reproduced exactly by passing the full transcript (the user turn, the assistant turns with their
`tool_calls`, and the `tool` messages) to `DllmEngine.chat_completion`.

### Limits

- Only tools are used from the servers; their resources and prompts are not offered to the model.
- Servers that ask the client for sampling, elicitation or roots are not supported.
- Small models call tools clumsily; Qwen2.5-Instruct is trained on the `<tool_call>` format the engine uses,
  SmolLM2-135M is not.
