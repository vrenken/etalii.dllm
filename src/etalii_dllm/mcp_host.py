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

Given an engine, the host also answers servers' sampling requests (``sampling/createMessage``) with it, exactly: the
seed derives from the request's content, so an identical request always gets the identical answer
(:func:`sampling_request`). Prompts and resources are listed in (server, name) and (server, URI) order and turn into
chat messages (:meth:`McpHost.prompt`, :meth:`McpHost.resource`); docs/mcp.md#sampling-prompts-and-resources.

Servers' form elicitations (``elicitation/create``) are answered by the engine too, greedily and constrained by the
requested schema (:func:`elicitation_request`), and the host offers the filesystem roots it was given sorted by URI
(``roots/list``); docs/mcp.md#elicitation-and-roots. :attr:`McpHost.answers` lists every sampling and elicitation in
the order the servers asked, for agent transcripts (:mod:`etalii_dllm.transcripts`).
"""

from __future__ import annotations

import hashlib
import json
import shlex
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from etalii_dllm.chat import ChatMessage, ToolCall
from etalii_dllm.engine import ChatEvent, ChatRequest, DllmEngine, Finished, ResponseFormat, TextDelta, ToolCallEvent
from etalii_dllm.sampling import SamplingOptions
from etalii_dllm.tools import Tool, ToolChoice

DEFAULT_MAX_ROUNDS = 8
ELICITATIONS = ("engine", "decline")
"""How the host answers form elicitations: with the engine, or by declining every one."""
ELICITATION_MAX_TOKENS = 512
"""The longest answer the engine gives to an elicitation; one that does not fit is cancelled."""
ELICITATION_SYSTEM = "An MCP server asks for information. Answer with a JSON object that fits this schema:\n{schema}"


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


@dataclass(frozen=True)
class Sampling:
    """A sampling request a server made and the engine's answer to it."""

    server: str
    request: ChatRequest
    content: str
    stop_reason: str
    fingerprint: str


@dataclass(frozen=True)
class Elicitation:
    """An elicitation a server made and the host's answer: ``accept`` with the engine's ``content``, ``decline``
    (URL mode, or the host declines every elicitation) or ``cancel`` (the answer did not fit in
    :data:`ELICITATION_MAX_TOKENS`). ``request`` and ``fingerprint`` are ``None`` when the engine did not run."""

    server: str
    message: str
    action: str
    content: Mapping[str, Any] | None
    request: ChatRequest | None
    fingerprint: str | None


@dataclass(frozen=True)
class RootInfo:
    """A filesystem root the host offers its servers."""

    uri: str
    name: str


@dataclass(frozen=True)
class PromptInfo:
    """A prompt a server offers: its name as the host exposes it, the server and the prompt's own name."""

    name: str
    server: str
    title: str
    description: str
    arguments: tuple[str, ...]
    """The argument names, required ones first in the server's order."""


@dataclass(frozen=True)
class ResourceInfo:
    """A resource a server offers."""

    uri: str
    server: str
    name: str
    description: str
    mime_type: str | None


HostEvent = ChatEvent | ToolResult


def _text_blocks(content: Any, what: str) -> str:
    """The text of one content block or a list of them; anything but text is refused."""
    blocks = content if isinstance(content, list) else [content]
    texts = []
    for block in blocks:
        if getattr(block, "type", None) != "text":
            raise ValueError(f"{what} can only hold text, not {getattr(block, 'type', type(block).__name__)!r}")
        texts.append(block.text)
    return "".join(texts)


def sampling_request(params: Any) -> ChatRequest:
    """The engine request for an MCP ``sampling/createMessage`` request: its system prompt and messages, its
    temperature (0, greedy, when absent), ``max_tokens`` and stop sequences, and a seed from the SHA-256 of the
    request's canonical JSON, so the same request always gets the same answer. Model preferences, included context
    and metadata are hints the engine (one model, no other context) leaves aside. Raises ``ValueError`` for what it
    cannot honour: content other than text and sampling with tools."""
    if params.tools:
        raise ValueError("sampling with tools is not supported")
    messages = [ChatMessage("system", params.system_prompt)] if params.system_prompt else []
    for message in params.messages:
        messages.append(ChatMessage(message.role, _text_blocks(message.content, "a sampling message")))
    seed = _canonical_seed(params)
    options = SamplingOptions(temperature=params.temperature or 0.0, seed=seed)
    return ChatRequest(messages, params.max_tokens, options, stop=tuple(params.stop_sequences or ()),
                       request_id=f"mcp-sampling-{seed:08x}")  # fmt: skip


def _canonical_seed(params: Any) -> int:
    canonical = json.dumps(
        params.model_dump(mode="json", by_alias=True, exclude_none=True, exclude={"meta"}),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return int(hashlib.sha256(canonical.encode()).hexdigest()[:8], 16)


def form_schema(requested: Mapping[str, Any]) -> dict[str, Any]:
    """The JSON schema an elicitation's answer is constrained by: the requested schema (a flat object of strings,
    numbers, booleans and enums) without the display-only ``enumNames`` and, unless it says otherwise, without
    properties it does not name."""
    if requested.get("type") != "object" or not isinstance(requested.get("properties", {}), Mapping):
        raise ValueError("an elicitation's requested schema must be an object schema")
    schema = json.loads(json.dumps(requested))
    for value in schema.get("properties", {}).values():
        if isinstance(value, dict):
            value.pop("enumNames", None)
    schema.setdefault("additionalProperties", False)
    return schema


def elicitation_request(params: Any) -> ChatRequest:
    """The engine request for an MCP form elicitation: a system prompt holding the requested schema, the server's
    message as the user message, greedy decoding constrained by :func:`form_schema` and at most
    :data:`ELICITATION_MAX_TOKENS` tokens, with a request id from the SHA-256 of the request's canonical JSON. Raises
    ``ValueError`` for URL-mode elicitations and schemas that are not object schemas."""
    if getattr(params, "mode", "form") != "form":
        raise ValueError("only form elicitations can be answered by the engine")
    schema = form_schema(params.requested_schema)
    seed = _canonical_seed(params)
    text = json.dumps(params.requested_schema, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    messages = [ChatMessage("system", ELICITATION_SYSTEM.format(schema=text)), ChatMessage("user", params.message)]
    return ChatRequest(messages, ELICITATION_MAX_TOKENS, SamplingOptions(temperature=0.0, seed=seed),
                       response_format=ResponseFormat("json_schema", schema),
                       request_id=f"mcp-elicitation-{seed:08x}")  # fmt: skip


def answer_sampling(engine: DllmEngine, server: str, request: ChatRequest) -> Sampling:
    """The engine's answer to a sampling request (:func:`sampling_request`) with its MCP stop reason."""
    result = engine.chat_completion(request)
    stop_reason = "maxTokens" if result.finish_reason == "length" else "endTurn"
    if result.stop_sequence is not None:
        stop_reason = "stopSequence"
    return Sampling(server, request, result.content, stop_reason, result.fingerprint)


def answer_elicitation(engine: DllmEngine, server: str, message: str, request: ChatRequest) -> Elicitation:
    """The engine's answer to a form elicitation (:func:`elicitation_request`): ``accept`` with the JSON object it
    wrote, or ``cancel`` when the object did not fit in the request's tokens."""
    result = engine.chat_completion(request)
    if result.finish_reason == "length":
        return Elicitation(server, message, "cancel", None, request, result.fingerprint)
    return Elicitation(server, message, "accept", json.loads(result.content), request, result.fingerprint)


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

    def __init__(
        self,
        servers: Sequence[McpServerConfig] | Mapping[str, Any],
        engine: DllmEngine | None = None,
        *,
        elicitation: str = "engine",
        roots: Sequence[str | Path] = (),
    ) -> None:
        if elicitation not in ELICITATIONS:
            raise McpHostError(f"elicitation must be one of {', '.join(ELICITATIONS)}")
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
        self.prompts: list[PromptInfo] = []
        self._prompt_routes: dict[str, tuple[str, str]] = {}
        self.resources: list[ResourceInfo] = []
        self._engine = engine
        """Answers the servers' sampling requests; without one, the host does not offer sampling."""
        self.samplings: list[Sampling] = []
        """Every sampling request answered, in the order the servers made them."""
        self._elicitation = elicitation
        self.elicitations: list[Elicitation] = []
        """Every elicitation answered, in the order the servers made them."""
        self.answers: list[Sampling | Elicitation] = []
        """Every sampling and elicitation, in the order the servers made them."""
        self.roots = _roots(roots)
        """The roots offered to the servers, sorted by URI; without any the host does not offer roots."""

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
                sampler = self._sampler(name) if self._engine is not None else None
                elicitor = self._elicitor(name) if self._engine is not None else None
                lister = self._list_roots if self.roots else None
                try:
                    client = Client(target, cache=None, sampling_callback=sampler, elicitation_callback=elicitor,
                                    list_roots_callback=lister)  # fmt: skip
                    self._clients[name] = await self._stack.enter_async_context(client)
                except Exception as error:  # any start-up failure names the server
                    raise McpHostError(f"cannot connect to MCP server {name!r}: {error}") from error
            await self._list_tools()
            await self._list_prompts()
            await self._list_resources()
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

    def _sampler(self, server: str) -> Any:
        from mcp import types

        engine = self._engine
        assert engine is not None

        async def sample(context: Any, params: types.CreateMessageRequestParams) -> Any:
            try:
                sampling = answer_sampling(engine, server, sampling_request(params))
            except ValueError as error:
                return types.ErrorData(code=types.INVALID_PARAMS, message=str(error))
            self.samplings.append(sampling)
            self.answers.append(sampling)
            return types.CreateMessageResult(
                role="assistant",
                content=types.TextContent(type="text", text=sampling.content),
                model=engine.model.id,
                stop_reason=sampling.stop_reason,
            )

        return sample

    def _elicitor(self, server: str) -> Any:
        from mcp import types

        engine = self._engine
        assert engine is not None

        async def elicit(context: Any, params: Any) -> Any:
            if self._elicitation == "decline" or getattr(params, "mode", "form") != "form":
                elicitation = Elicitation(server, params.message, "decline", None, None, None)
            else:
                try:
                    elicitation = answer_elicitation(engine, server, params.message, elicitation_request(params))
                except ValueError as error:
                    return types.ErrorData(code=types.INVALID_PARAMS, message=str(error))
            self.elicitations.append(elicitation)
            self.answers.append(elicitation)
            return types.ElicitResult(action=elicitation.action, content=elicitation.content)  # type: ignore[arg-type]

        return elicit

    async def _list_roots(self, context: Any) -> Any:
        from mcp import types

        return types.ListRootsResult(roots=[types.Root(uri=root.uri, name=root.name) for root in self.roots])  # type: ignore[arg-type]

    async def _list_prompts(self) -> None:
        listed: list[tuple[str, Any]] = []
        for server, client in self._clients.items():
            if getattr(client.server_capabilities, "prompts", None) is None:
                continue
            prompts: list[Any] = []
            cursor = None
            while True:
                page = await client.list_prompts(cursor=cursor)
                prompts.extend(page.prompts)
                cursor = page.next_cursor
                if cursor is None:
                    break
            listed.extend((server, prompt) for prompt in sorted(prompts, key=lambda p: p.name))
        counts: dict[str, int] = {}
        for _, prompt in listed:
            counts[prompt.name] = counts.get(prompt.name, 0) + 1
        for server, prompt in listed:
            exposed = prompt.name if counts[prompt.name] == 1 else f"{server}.{prompt.name}"
            self._prompt_routes[exposed] = (server, prompt.name)
            arguments = sorted(prompt.arguments or [], key=lambda a: not a.required)  # stable: server order kept
            self.prompts.append(
                PromptInfo(
                    exposed, server, prompt.title or "", prompt.description or "", tuple(a.name for a in arguments)
                )
            )

    async def _list_resources(self) -> None:
        for server, client in self._clients.items():
            if getattr(client.server_capabilities, "resources", None) is None:
                continue
            resources: list[Any] = []
            cursor = None
            while True:
                page = await client.list_resources(cursor=cursor)
                resources.extend(page.resources)
                cursor = page.next_cursor
                if cursor is None:
                    break
            for resource in sorted(resources, key=lambda r: str(r.uri)):
                self.resources.append(
                    ResourceInfo(
                        str(resource.uri), server, resource.name, resource.description or "", resource.mime_type
                    )
                )

    async def prompt(self, name: str, arguments: Mapping[str, str] | None = None) -> list[ChatMessage]:
        """The messages of the prompt the host exposes as ``name``, filled in with ``arguments``. Raises
        :class:`McpHostError` for an unknown prompt, a server error or content other than text and text
        resources."""
        route = self._prompt_routes.get(name)
        if route is None:
            raise McpHostError(f"unknown MCP prompt {name!r}")
        server, own = route
        try:
            result = await self._clients[server].get_prompt(own, dict(arguments or {}))
        except Exception as error:
            raise McpHostError(f"MCP prompt {name!r}: {error}") from error
        messages = []
        for message in result.messages:
            content = message.content
            if getattr(content, "type", None) == "resource":
                text = getattr(content.resource, "text", None)
                if text is None:
                    raise McpHostError(f"MCP prompt {name!r} holds a binary resource")
            elif getattr(content, "type", None) == "text":
                text = content.text
            else:
                raise McpHostError(f"MCP prompt {name!r} holds {content.type!r} content; only text is supported")
            messages.append(ChatMessage(message.role, text))
        return messages

    async def resource(self, uri: str) -> str:
        """The text of the resource at ``uri`` (from the first server, in name order, that lists it; any server
        when none does), its text parts joined by newlines. Raises :class:`McpHostError` for a server error or a
        binary resource."""
        servers = [info.server for info in self.resources if info.uri == uri] or list(self._clients)
        error: Exception | None = None
        for server in servers:
            try:
                result = await self._clients[server].read_resource(uri)
            except Exception as problem:  # the next server may have it
                error = problem
                continue
            texts = []
            for part in result.contents:
                text = getattr(part, "text", None)
                if text is None:
                    raise McpHostError(f"MCP resource {uri!r} is binary; only text resources are supported")
                texts.append(text)
            return "\n".join(texts)
        raise McpHostError(f"cannot read MCP resource {uri!r}: {error}")

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


def _roots(paths: Sequence[str | Path]) -> list[RootInfo]:
    """Each directory as its absolute ``file://`` URI and name, without duplicates, sorted by URI."""
    roots: dict[str, RootInfo] = {}
    for path in paths:
        directory = Path(path).resolve()
        if not directory.is_dir():
            raise McpHostError(f"MCP root {str(path)!r} is not a directory")
        uri = directory.as_uri()
        roots[uri] = RootInfo(uri, directory.name or uri)
    return [roots[uri] for uri in sorted(roots)]


async def chat(
    engine: DllmEngine,
    request: ChatRequest,
    host: McpHost,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
) -> AsyncIterator[HostEvent]:
    """Runs a chat in which the model may call the host's tools (and the request's own tools, which are not
    executed: a call to one ends the chat like a normal ``tool_calls`` answer).

    Yields the engine's events of every round and a :class:`ToolResult` after each executed call. A ``required``
    or named ``tool_choice`` applies to the first round; later rounds let the model decide (among the allowed tools,
    with the same limit on parallel calls), so it can answer. The
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
            choice = ToolChoice("auto", None, choice.allowed, choice.parallel)
