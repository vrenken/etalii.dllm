"""Phase 39: deterministic MCP sampling (#257), prompts (#258) and resources (#259) in the MCP host and dllm chat,
with a golden sampling answer (#260)."""

from __future__ import annotations

import sys
import textwrap

import anyio
import pytest
from golden_values import MCP_SAMPLING_FINGERPRINT
from mcp import types
from mcp.server.mcpserver import Context, MCPServer
from test_cli import isolated_environment  # noqa: F401 - autouse fixture: the CLI tests set DLLM_MODEL

from etalii_dllm import mcp_host
from etalii_dllm.chat import ChatMessage, ToolCall
from etalii_dllm.cli import main as cli_main
from etalii_dllm.engine import DllmEngine
from etalii_dllm.mcp_host import McpHost, McpHostError, sampling_request

SERVER_SOURCE = textwrap.dedent(
    '''
    from mcp import types
    from mcp.server.mcpserver import Context, MCPServer

    server = MCPServer("notes")


    @server.prompt()
    def review(language: str, style: str = "short") -> str:
        """Asks for a code review."""
        return f"Review this {language} code in a {style} style."


    @server.resource("notes://today")
    def today() -> str:
        """Today's notes."""
        return "Buy milk."


    @server.tool()
    async def summarize(text: str, ctx: Context) -> str | types.InputRequiredResult:
        """Summarizes a text with the client's model."""
        answers = ctx.input_responses
        if not answers:
            message = types.SamplingMessage(role="user", content=types.TextContent(type="text", text=text))
            params = types.CreateMessageRequestParams(messages=[message], max_tokens=6)
            return types.InputRequiredResult(input_requests={"s": types.CreateMessageRequest(params=params)})
        return answers["s"].content.text


    server.run()
    '''
)


def _sampling_message(text: str, kind: str = "text") -> types.SamplingMessage:
    content: types.TextContent | types.ImageContent = (
        types.TextContent(type="text", text=text)
        if kind == "text"
        else types.ImageContent(type="image", data="AAAA", mime_type="image/png")
    )
    return types.SamplingMessage(role="user", content=content)


def context_server(name: str = "ctx") -> MCPServer:
    server = MCPServer(name)

    @server.prompt()
    def greet(person: str) -> str:
        """A greeting."""
        return f"Say hello to {person}."

    @server.prompt()
    def dialogue() -> list[dict]:
        """A short exchange."""
        return [
            {"role": "user", "content": {"type": "text", "text": "Hi"}},
            {"role": "assistant", "content": {"type": "text", "text": "Hello!"}},
            {"role": "user", "content": {"type": "resource", "resource": {"uri": "notes://a", "text": "A note."}}},
        ]

    @server.prompt()
    def picture() -> list[dict]:
        """An image, which the host refuses."""
        return [{"role": "user", "content": {"type": "image", "data": "AAAA", "mimeType": "image/png"}}]

    @server.prompt()
    def blob() -> list[dict]:
        """A binary resource, which the host refuses."""
        return [{"role": "user", "content": {"type": "resource", "resource": {"uri": "b://x", "blob": "AAAA"}}}]

    @server.resource("notes://today")
    def today() -> str:
        """Today's notes."""
        return "Buy milk."

    @server.resource("data://bytes", mime_type="application/octet-stream")
    def raw() -> bytes:
        """Binary data."""
        return b"\x00\x01"

    @server.tool()
    async def summarize(text: str, ctx: Context) -> str | types.InputRequiredResult:
        """Summarizes a text with the client's model."""
        answers = ctx.input_responses
        if not answers:
            params = types.CreateMessageRequestParams(
                messages=[_sampling_message(f"Summarize: {text}")], max_tokens=8, system_prompt="Be brief."
            )
            return types.InputRequiredResult(input_requests={"s": types.CreateMessageRequest(params=params)})
        return answers["s"].content.text

    @server.tool()
    async def describe(ctx: Context) -> str | types.InputRequiredResult:
        """Asks the client to sample from an image."""
        answers = ctx.input_responses
        if not answers:
            params = types.CreateMessageRequestParams(messages=[_sampling_message("", "image")], max_tokens=4)
            return types.InputRequiredResult(input_requests={"s": types.CreateMessageRequest(params=params)})
        return answers["s"].content.text

    return server


def run(servers, engine, body):
    async def go():
        async with McpHost(servers, engine) as host:
            return await body(host)

    return anyio.run(go)


# Sampling


def test_sampling_requests_become_exact_engine_requests():
    params = types.CreateMessageRequestParams(
        messages=[_sampling_message("Hi")], max_tokens=5, system_prompt="Be kind.", temperature=0.7,
        stop_sequences=["\n"], model_preferences=types.ModelPreferences(hints=[types.ModelHint(name="any")]),
    )  # fmt: skip
    request = sampling_request(params)
    assert request.messages == [ChatMessage("system", "Be kind."), ChatMessage("user", "Hi")]
    assert (request.max_tokens, request.options.temperature, tuple(request.stop)) == (5, 0.7, ("\n",))
    assert sampling_request(params) == request  # the seed derives from the content alone
    other = sampling_request(params.model_copy(update={"temperature": 0.8}))
    assert other.options.seed != request.options.seed
    plain = sampling_request(types.CreateMessageRequestParams(messages=[_sampling_message("Hi")], max_tokens=5))
    assert plain.options.temperature == 0.0 and plain.messages == [ChatMessage("user", "Hi")]
    multi = types.SamplingMessage(role="user", content=[types.TextContent(type="text", text=t) for t in ("a", "b")])
    assert (
        sampling_request(types.CreateMessageRequestParams(messages=[multi], max_tokens=1)).messages[0].content == "ab"
    )
    with pytest.raises(ValueError, match="only hold text"):
        sampling_request(types.CreateMessageRequestParams(messages=[_sampling_message("", "image")], max_tokens=1))
    tool = types.Tool(name="t", input_schema={"type": "object"})
    with pytest.raises(ValueError, match="tools"):
        sampling_request(
            types.CreateMessageRequestParams(messages=[_sampling_message("x")], max_tokens=1, tools=[tool])
        )


def test_the_host_answers_sampling_with_the_engine():
    engine = DllmEngine.create_default()

    async def body(host: McpHost):
        first = await host.call(ToolCall("c1", "summarize", '{"text": "a long story"}'))
        second = await host.call(ToolCall("c2", "summarize", '{"text": "a long story"}'))
        refused = await host.call(ToolCall("c3", "describe", "{}"))
        return first, second, refused, list(host.samplings)

    first, second, refused, samplings = run({"ctx": context_server()}, engine, body)
    assert not first.is_error and first.content == second.content
    assert refused.is_error and "only hold text" in refused.content
    assert len(samplings) == 2 and samplings[0] == samplings[1]
    sampling = samplings[0]
    assert sampling.server == "ctx" and sampling.content == first.content
    assert sampling.request.messages[0] == ChatMessage("system", "Be brief.")
    assert sampling.stop_reason in ("maxTokens", "endTurn")
    assert sampling.fingerprint == MCP_SAMPLING_FINGERPRINT
    assert engine.chat_completion(sampling.request).fingerprint == sampling.fingerprint


def test_without_an_engine_the_host_does_not_sample():
    async def body(host: McpHost):
        return await host.call(ToolCall("c1", "summarize", '{"text": "x"}')), host.samplings

    result, samplings = run({"ctx": context_server()}, None, body)
    assert result.is_error and samplings == []


# Prompts and resources


def test_prompts_and_resources_are_listed_in_a_fixed_order():
    async def body(host: McpHost):
        return host.prompts, host.resources

    prompts, resources = run({"b": context_server("b"), "a": context_server("a")}, None, body)
    assert [p.name for p in prompts] == [f"{s}.{n}" for s in "ab" for n in ("blob", "dialogue", "greet", "picture")]
    assert prompts[2].arguments == ("person",) and prompts[2].description == "A greeting."
    assert [(r.server, r.uri) for r in resources] == [(s, u) for s in "ab" for u in ("data://bytes", "notes://today")]


def test_prompts_become_chat_messages():
    async def body(host: McpHost):
        greet = await host.prompt("greet", {"person": "Ann"})
        dialogue = await host.prompt("dialogue")
        errors = []
        for name, arguments in (("picture", {}), ("blob", {}), ("nothing", {}), ("greet", {})):
            try:
                await host.prompt(name, arguments)
            except McpHostError as error:
                errors.append(str(error))
        return greet, dialogue, errors

    greet, dialogue, errors = run({"ctx": context_server()}, None, body)
    assert greet == [ChatMessage("user", "Say hello to Ann.")]
    assert dialogue == [ChatMessage("user", "Hi"), ChatMessage("assistant", "Hello!"), ChatMessage("user", "A note.")]
    assert "'image' content" in errors[0] and "binary resource" in errors[1]
    assert "unknown MCP prompt" in errors[2] and "MCP prompt 'greet'" in errors[3]


def test_resources_are_read_as_text():
    async def body(host: McpHost):
        text = await host.resource("notes://today")
        errors = []
        for uri in ("data://bytes", "notes://missing"):
            try:
                await host.resource(uri)
            except McpHostError as error:
                errors.append(str(error))
        return text, errors

    text, errors = run({"ctx": context_server()}, None, body)
    assert text == "Buy milk."
    assert "binary" in errors[0] and "cannot read MCP resource 'notes://missing'" in errors[1]


# dllm chat


@pytest.fixture
def notes_server(tmp_path):
    path = tmp_path / "notes_server.py"
    path.write_text(SERVER_SOURCE, encoding="utf-8")
    return f"notes={sys.executable} {path}"


def test_cli_lists_and_uses_prompts_and_resources(notes_server, capsys):
    assert cli_main(["chat", "--mcp-server", notes_server, "--mcp-list"]) == 0
    out = capsys.readouterr().out
    assert "summarize (notes)" in out and "review (notes) [language, style]" in out and "notes://today (notes)" in out
    args = ["chat", "--mcp-server", notes_server, "--max-tokens", "4"]
    assert cli_main([*args, "--mcp-prompt", "review", "--mcp-arg", "language=Python", "--mcp-resource",
                     "notes://today"]) == 0  # fmt: skip
    assert "fingerprint:" in capsys.readouterr().err
    assert cli_main([*args, "Summarize 'a long story'"]) == 0


def test_cli_context_errors(notes_server, capsys):
    assert cli_main(["chat", "Hi", "--mcp-prompt", "review"]) == 1
    assert "need --mcp-config or --mcp-server" in capsys.readouterr().err
    assert cli_main(["chat", "--mcp-server", notes_server, "--mcp-prompt", "review", "--mcp-arg", "language"]) == 1
    assert "KEY=VALUE" in capsys.readouterr().err
    assert cli_main(["chat", "Hi", "--mcp-server", notes_server, "--mcp-resource", "notes://missing"]) == 1
    assert "cannot read MCP resource" in capsys.readouterr().err


def test_module_exports():
    assert mcp_host.Sampling and mcp_host.PromptInfo and mcp_host.ResourceInfo
