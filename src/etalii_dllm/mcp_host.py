"""MCP client host: lets the model call the tools of external MCP servers during a chat.

The host connects to the configured servers (stdio commands or Streamable HTTP URLs, in the ``mcpServers`` format
Claude Desktop and Claude Code use), offers their tools to the model as ordinary function tools, and runs the chat
loop: generate, execute the calls the model made, append the results as ``tool`` messages, generate again, until the
model answers without a call or ``max_rounds`` is reached.

Determinism: the model side is as reproducible as any other chat. Servers are connected in name order, tools are
listed in (server, tool) order, calls run one at a time in the order the model made them, and tool call ids are
derived from the conversation. So the same conversation and the same tool results always give the same answer; if
an external tool returns something different, the answer after it may differ (and :attr:`ToolResult` records what
was returned, so a run can be replayed with :meth:`DllmEngine.chat_completion` on the transcript).
"""

from __future__ import annotations

import json
import shlex
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from etalii_dllm.chat import ChatMessage, ToolCall
from etalii_dllm.engine import ChatEvent, ChatRequest, DllmEngine, Finished, TextDelta, ToolCallEvent
from etalii_dllm.tools import AUTO, Tool

DEFAULT_MAX_ROUNDS = 8


class McpHostError(Exception):
    """A server could not be configured, started or listed."""


@dataclass(frozen=True)
class McpServerConfig:
    """One MCP server: a stdio ``command`` with ``args`` and ``env``, or a Streamable HTTP ``url``."""

    name: str
    command: str | None = None
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    url: str | None = None

    def __post_init__(self) -> None:
        if (self.command is None) == (self.url is None):
            raise McpHostError(f"MCP server {self.name!r} needs exactly one of 'command' or 'url'")


def load_config(source: str | Path | Mapping[str, Any]) -> list[McpServerConfig]:
    """Servers from an ``{"mcpServers": {name: {"command", "args", "env"} | {"url"}}}`` file or mapping, in name
    order."""
    if isinstance(source, Mapping):
        document: Any = source
    else:
        try:
            document = json.loads(Path(source).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise McpHostError(f"{source}: {error}") from error
    servers = document.get("mcpServers") if isinstance(document, Mapping) else None
    if not isinstance(servers, Mapping):
        raise McpHostError("an MCP configuration needs an 'mcpServers' object")
    configs = []
    for name in sorted(servers):
        entry = servers[name]
        if not isinstance(entry, Mapping):
            raise McpHostError(f"MCP server {name!r} must be an object")
        configs.append(
            McpServerConfig(
                name=str(name),
                command=entry.get("command"),
                args=tuple(str(a) for a in entry.get("args", ())),
                env={str(k): str(v) for k, v in (entry.get("env") or {}).items()},
                url=entry.get("url"),
            )
        )
    return configs


def parse_server(spec: str) -> McpServerConfig:
    """A server from the command line: ``[name=]command args...`` or ``[name=]http(s)://...``. Without a name,
    the server is named after its command (or host)."""
    name, separator, rest = spec.partition("=")
    if not separator or not name.isidentifier():
        name, rest = "", spec
    if rest.startswith(("http://", "https://")):
        return McpServerConfig(name or rest.split("/")[2].split(":")[0], url=rest)
    words = shlex.split(rest)
    if not words:
        raise McpHostError(f"empty MCP server specification {spec!r}")
    return McpServerConfig(name or Path(words[0]).stem, command=words[0], args=tuple(words[1:]))


@dataclass(frozen=True)
class ToolResult:
    """What an MCP tool returned for a call (``is_error`` when the server reported a failure)."""

    call: ToolCall
    server: str
    content: str
    is_error: bool


HostEvent = ChatEvent | ToolResult


def result_text(content: Sequence[Any], structured: Any = None) -> str:
    """The text the model sees for a tool result: text blocks joined by newlines, other blocks (images, resources)
    as canonical JSON, and the structured content when there are no blocks."""
    parts = []
    for block in content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
        else:
            dumped = block.model_dump(mode="json", by_alias=True, exclude_none=True)
            parts.append(json.dumps(dumped, sort_keys=True, ensure_ascii=False))
    if not parts and structured is not None:
        parts.append(json.dumps(structured, sort_keys=True, ensure_ascii=False))
    return "\n".join(parts)


class McpHost:
    """Connections to MCP servers and their tools. Use as ``async with McpHost(configs) as host:``.

    ``servers`` may also hold in-process ``MCPServer`` objects (by name), which tests use.
    """

    def __init__(self, servers: Sequence[McpServerConfig] | Mapping[str, Any]) -> None:
        if isinstance(servers, Mapping):
            self._targets = {name: servers[name] for name in sorted(servers)}
        else:
            names = [s.name for s in servers]
            if len(set(names)) != len(names):
                raise McpHostError("MCP server names must be unique")
            self._targets = {s.name: s for s in sorted(servers, key=lambda s: s.name)}
        self._stack = AsyncExitStack()
        self._clients: dict[str, Any] = {}
        self._routes: dict[str, tuple[str, str]] = {}
        """Tool name as the model sees it -> (server, the server's own tool name)."""
        self.tools: list[Tool] = []

    async def __aenter__(self) -> McpHost:
        from mcp import Client
        from mcp.client.stdio import StdioServerParameters

        try:
            for name, target in self._targets.items():
                if isinstance(target, McpServerConfig):
                    if target.url is not None:
                        target = target.url
                    else:
                        assert target.command is not None
                        target = StdioServerParameters(
                            command=target.command, args=list(target.args), env=dict(target.env) or None
                        )
                try:
                    self._clients[name] = await self._stack.enter_async_context(Client(target, cache=None))
                except Exception as error:  # any start-up failure names the server
                    raise McpHostError(f"cannot connect to MCP server {name!r}: {error}") from error
            await self._list_tools()
        except BaseException:
            await self._stack.aclose()
            raise
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._stack.aclose()

    async def _list_tools(self) -> None:
        listed: list[tuple[str, Any]] = []
        for server, client in self._clients.items():
            tools: list[Any] = []
            cursor = None
            while True:
                page = await client.list_tools(cursor=cursor)
                tools.extend(page.tools)
                cursor = page.next_cursor
                if cursor is None:
                    break
            listed.extend((server, tool) for tool in sorted(tools, key=lambda t: t.name))
        counts: dict[str, int] = {}
        for _, tool in listed:
            counts[tool.name] = counts.get(tool.name, 0) + 1
        for server, tool in listed:
            # A name offered by more than one server is qualified with its server's name.
            exposed = tool.name if counts[tool.name] == 1 else f"{server}.{tool.name}"
            self._routes[exposed] = (server, tool.name)
            schema = dict(tool.input_schema or {}) or {"type": "object", "properties": {}}
            self.tools.append(Tool(exposed, tool.description or "", schema))

    @property
    def servers(self) -> dict[str, str]:
        """Each offered tool name and the server it belongs to."""
        return {exposed: server for exposed, (server, _) in self._routes.items()}

    async def call(self, call: ToolCall) -> ToolResult:
        """Runs one tool call. Failures (unknown tool, bad arguments, server errors) become error results the model
        can read, so a chat never stops on a tool."""
        route = self._routes.get(call.name)
        if route is None:
            return ToolResult(call, "", f"unknown tool {call.name!r}", True)
        server, name = route
        arguments = call.arguments_object()
        if not isinstance(arguments, dict):
            return ToolResult(call, server, "the arguments are not a JSON object", True)
        try:
            result = await self._clients[server].call_tool(name, arguments)
        except Exception as error:  # reported to the model
            return ToolResult(call, server, f"{type(error).__name__}: {error}", True)
        return ToolResult(call, server, result_text(result.content, result.structured_content), bool(result.is_error))


async def chat(
    engine: DllmEngine,
    request: ChatRequest,
    host: McpHost,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
) -> AsyncIterator[HostEvent]:
    """Runs a chat in which the model may call the host's tools (and the request's own tools, which are not
    executed: a call to one ends the chat like a normal ``tool_calls`` answer).

    Yields the engine's events of every round and a :class:`ToolResult` after each executed call. A ``required``
    or named ``tool_choice`` applies to the first round; later rounds let the model decide, so it can answer. The
    last :class:`Finished` has finish reason ``tool_calls`` only when the model still called a tool in round
    ``max_rounds`` (those calls are not executed). Raises ``ValueError`` for invalid requests, like
    :meth:`DllmEngine.chat_stream`.
    """
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")
    tools = [*host.tools, *request.tools]
    own = {t.name for t in request.tools}
    messages = list(request.messages)
    choice = request.tool_choice
    for round_index in range(max_rounds):
        current = replace(
            request,
            messages=list(messages),
            tools=tools,
            tool_choice=choice,
            request_id=f"{request.request_id}\n{round_index}",
        )
        content: list[str] = []
        calls: list[ToolCall] = []
        finished: Finished | None = None
        for event in engine.chat_stream(current):
            if isinstance(event, ToolCallEvent):
                calls.append(event.call)
            elif isinstance(event, Finished):
                finished = event
            elif isinstance(event, TextDelta):  # a thinking model's reasoning stays out of the history
                content.append(event.text)
            yield event
        assert finished is not None
        last_round = round_index == max_rounds - 1
        if finished.finish_reason != "tool_calls" or last_round or any(c.name in own for c in calls):
            return
        messages.append(ChatMessage("assistant", "".join(content), tool_calls=tuple(calls)))
        for call in calls:
            result = await host.call(call)
            yield result
            text = f"Error: {result.content}" if result.is_error else result.content
            messages.append(ChatMessage("tool", text, tool_call_id=call.id, name=call.name))
        if choice.mode in ("required", "named"):
            choice = AUTO
