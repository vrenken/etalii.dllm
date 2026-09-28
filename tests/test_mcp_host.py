import json
import sys
from typing import Literal

import anyio
import pytest
from mcp.server.mcpserver import MCPServer

from etalii_dllm import mcp_host
from etalii_dllm.chat import ChatMessage, ToolCall
from etalii_dllm.cli import main as cli_main
from etalii_dllm.engine import ChatRequest, DllmEngine, Finished, ToolCallEvent
from etalii_dllm.mcp_host import McpHost, McpHostError, McpServerConfig, ToolResult
from etalii_dllm.tools import Tool, ToolChoice

DLLM_MCP = McpServerConfig(
    "dllm", command=sys.executable, args=("-c", "from etalii_dllm.mcp_server import main; main()")
)


def fixture_server(name: str = "fixture") -> MCPServer:
    server = MCPServer(name)

    @server.tool()
    def add(a: int, b: int) -> str:
        """Adds two integers."""
        return str(a + b)

    @server.tool()
    def get_weather(city: Literal["Paris", "Rome"]) -> str:
        """Current weather for a city."""
        return {"Paris": "12 C, rain", "Rome": "24 C, sun"}[city]

    @server.tool()
    def fail() -> str:
        """Always fails."""
        raise RuntimeError("out of order")

    return server


def run_chat(servers, request: ChatRequest, max_rounds: int = 3) -> list:
    async def go() -> list:
        async with McpHost(servers) as host:
            return [event async for event in mcp_host.chat(DllmEngine.create_default(), request, host, max_rounds)]

    return anyio.run(go)


def ask(choice: ToolChoice, tools: tuple[Tool, ...] = ()) -> ChatRequest:
    return ChatRequest([ChatMessage("user", "What is the weather in Rome?")], 160, tools=tools, tool_choice=choice)


def test_load_config_and_parse_server(tmp_path):
    path = tmp_path / "mcp.json"
    config = {
        "mcpServers": {
            "time": {"command": "uvx", "args": ["mcp-server-time"], "env": {"TZ": "UTC"}},
            "docs": {"url": "https://example.com/mcp"},
        }
    }
    path.write_text(json.dumps(config))
    docs, time = mcp_host.load_config(path)
    assert docs == McpServerConfig("docs", url="https://example.com/mcp")
    assert time == McpServerConfig("time", command="uvx", args=("mcp-server-time",), env={"TZ": "UTC"})

    assert mcp_host.parse_server("t=uvx mcp-server-time --local-timezone 'Europe/Amsterdam'") == McpServerConfig(
        "t", command="uvx", args=("mcp-server-time", "--local-timezone", "Europe/Amsterdam")
    )
    assert mcp_host.parse_server("/usr/bin/fetcher -v").name == "fetcher"
    assert mcp_host.parse_server("https://example.com:8443/mcp") == McpServerConfig(
        "example.com", url="https://example.com:8443/mcp"
    )
    with pytest.raises(McpHostError):
        mcp_host.load_config({"mcpServers": {"x": {"command": "a", "url": "b"}}})
    with pytest.raises(McpHostError):
        mcp_host.load_config({"servers": {}})


def test_host_lists_tools_in_a_fixed_order_and_qualifies_duplicates():
    async def go() -> list[str]:
        async with McpHost({"b": fixture_server("b"), "a": fixture_server("a")}) as host:
            return [t.name for t in host.tools]

    names = anyio.run(go)
    assert names == ["a.add", "a.fail", "a.get_weather", "b.add", "b.fail", "b.get_weather"]


def test_host_reports_tool_failures_as_error_results():
    async def go() -> list[ToolResult]:
        async with McpHost({"fixture": fixture_server()}) as host:
            return [
                await host.call(ToolCall("c1", "add", '{"a": 2, "b": 3}')),
                await host.call(ToolCall("c2", "fail", "{}")),
                await host.call(ToolCall("c3", "nope", "{}")),
                await host.call(ToolCall("c4", "add", "[1]")),
            ]

    added, failed, unknown, bad = anyio.run(go)
    assert (added.content, added.is_error, added.server) == ("5", False, "fixture")
    assert failed.is_error and "fail" in failed.content
    assert unknown.is_error and unknown.server == ""
    assert bad.is_error


def test_chat_executes_calls_and_continues_deterministically():
    request = ask(ToolChoice("named", "get_weather"))
    events = run_chat({"fixture": fixture_server()}, request)
    calls = [e for e in events if isinstance(e, ToolCallEvent)]
    results = [e for e in events if isinstance(e, ToolResult)]
    finishes = [e for e in events if isinstance(e, Finished)]
    assert calls[0].call.name == "get_weather"
    assert results[0].call == calls[0].call
    assert results[0].content in ("12 C, rain", "24 C, sun")
    assert finishes[0].finish_reason == "tool_calls"
    assert len(finishes) >= 2  # the first round's named choice does not carry over, so the model gets to answer
    assert run_chat({"fixture": fixture_server()}, request) == events


def test_calls_to_the_requests_own_tools_end_the_chat():
    own = Tool("lookup", "Client-side lookup", {"type": "object", "properties": {"q": {"enum": ["x"]}}})
    events = run_chat({"fixture": fixture_server()}, ask(ToolChoice("named", "lookup"), (own,)))
    assert not any(isinstance(e, ToolResult) for e in events)
    assert events[-1] == next(e for e in events if isinstance(e, Finished))
    assert events[-1].finish_reason == "tool_calls"


def test_max_rounds_stops_before_executing_more_calls():
    events = run_chat({"fixture": fixture_server()}, ask(ToolChoice("named", "add")), max_rounds=1)
    assert not any(isinstance(e, ToolResult) for e in events)
    assert events[-1].finish_reason == "tool_calls"
    with pytest.raises(ValueError):
        run_chat({"fixture": fixture_server()}, ask(ToolChoice("auto")), max_rounds=0)


def test_host_runs_stdio_servers():
    events = run_chat([DLLM_MCP], ask(ToolChoice("named", "model_info")), max_rounds=2)
    (result,) = [e for e in events if isinstance(e, ToolResult)]
    engine = DllmEngine.create_default()
    assert result.content == f"model: {engine.model.id}\nsystem_fingerprint: {engine.system_fingerprint}"


def test_host_names_the_server_that_cannot_start():
    missing = McpServerConfig("ghost", command=sys.executable, args=("-c", "import sys; sys.exit(3)"))
    with pytest.raises(McpHostError, match="ghost"):
        run_chat([missing], ask(ToolChoice("auto")))


def test_cli_chat_with_mcp_servers(tmp_path, capsys):
    config = tmp_path / "mcp.json"
    config.write_text(json.dumps({"mcpServers": {"dllm": {"command": DLLM_MCP.command, "args": list(DLLM_MCP.args)}}}))
    arguments = ["chat", "Hi", "--max-tokens", "8", "--mcp-config", str(config)]
    assert cli_main(arguments) == 0
    first = capsys.readouterr()
    assert "tools: chat, generate, model_info" in first.err
    assert "fingerprint: " in first.err
    assert cli_main(arguments) == 0
    assert capsys.readouterr() == first
    assert cli_main(["chat", "Hi", "--mcp-config", str(tmp_path / "missing.json")]) == 1
