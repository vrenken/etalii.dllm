"""Phase 46: MCP elicitations answered exactly by the engine (#292), roots offered in a fixed order (#293), and the
engine's sampling and elicitation answers in agent transcripts, checked by dllm replay (#294), with a golden
elicitation answer (#295)."""

from __future__ import annotations

import dataclasses
import json

import anyio
import pytest
from golden_values import MCP_ELICITATION_FINGERPRINT
from mcp import types
from mcp.server.mcpserver import Context, MCPServer
from test_cli import isolated_environment  # noqa: F401 - autouse fixture: the CLI tests set DLLM_MODEL
from test_mcp_context import _sampling_message

from etalii_dllm import mcp_host, transcripts
from etalii_dllm.chat import ChatMessage, ToolCall
from etalii_dllm.cli import main
from etalii_dllm.engine import ChatRequest, DllmEngine
from etalii_dllm.mcp_host import McpHost, McpHostError, elicitation_request, form_schema
from etalii_dllm.tools import ToolChoice

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "title": "Name", "maxLength": 12},
        "age": {"type": "integer", "minimum": 0, "maximum": 120},
        "color": {"type": "string", "enum": ["red", "green"], "enumNames": ["Red", "Green"]},
        "subscribe": {"type": "boolean"},
    },
    "required": ["name", "age", "color"],
}


def _form(message: str = "Who are you?", schema: dict | None = None) -> types.ElicitRequestFormParams:
    return types.ElicitRequestFormParams(mode="form", message=message, requested_schema=schema or SCHEMA)


def _elicit(params: types.ElicitRequestFormParams | types.ElicitRequestURLParams) -> types.InputRequiredResult:
    return types.InputRequiredResult(input_requests={"e": types.ElicitRequest(params=params)})


def form_server(name: str = "forms") -> MCPServer:
    server = MCPServer(name)

    @server.tool()
    async def register(ctx: Context) -> str | types.InputRequiredResult:
        """Registers the user, asking for their details."""
        if not ctx.input_responses:
            return _elicit(_form())
        answer = ctx.input_responses["e"]
        return json.dumps({"action": answer.action, "content": answer.content}, sort_keys=True)

    @server.tool()
    async def essay(ctx: Context) -> str | types.InputRequiredResult:
        """Asks for an answer too long for the engine's limit."""
        if not ctx.input_responses:
            schema = {
                "type": "object",
                "properties": {"text": {"type": "string", "minLength": 100000}},
                "required": ["text"],
            }
            return _elicit(_form("Write a long essay.", schema))
        return ctx.input_responses["e"].action

    @server.tool()
    async def login(ctx: Context) -> str | types.InputRequiredResult:
        """Sends the user to a web page."""
        if not ctx.input_responses:
            return _elicit(types.ElicitRequestURLParams(mode="url", message="Sign in", url="https://example.com/a"))
        return ctx.input_responses["e"].action

    @server.tool()
    async def listing(ctx: Context) -> str | types.InputRequiredResult:
        """Asks for a list, which is not a form."""
        if not ctx.input_responses:
            return _elicit(_form("Items?", {"type": "array"}))
        return ctx.input_responses["e"].action

    @server.tool()
    async def roots(ctx: Context) -> str | types.InputRequiredResult:
        """The client's roots."""
        if not ctx.input_responses:
            return types.InputRequiredResult(input_requests={"r": types.ListRootsRequest()})
        return json.dumps([[str(root.uri), root.name] for root in ctx.input_responses["r"].roots])

    @server.tool()
    async def profile(ctx: Context) -> str | types.InputRequiredResult:
        """Samples a greeting, then asks for the user's details."""
        answers = ctx.input_responses or {}
        if "e" in answers:  # each round carries only its own answers; the greeting comes back as the state
            return f"{ctx.request_state} {answers['e'].action}"
        if "s" in answers:
            return types.InputRequiredResult(
                input_requests={"e": types.ElicitRequest(params=_form())}, request_state=answers["s"].content.text
            )
        params = types.CreateMessageRequestParams(messages=[_sampling_message("Say hi.")], max_tokens=6)
        return types.InputRequiredResult(input_requests={"s": types.CreateMessageRequest(params=params)})

    return server


def run(body, engine=None, servers=None, **options):
    async def go():
        async with McpHost(servers or {"forms": form_server()}, engine, **options) as host:
            return await body(host)

    return anyio.run(go)


def call(name: str):
    async def body(host):
        return await host.call(ToolCall("c1", name, "{}")), host

    return body


# Elicitation


def test_elicitations_become_exact_engine_requests():
    schema = form_schema(SCHEMA)
    assert "enumNames" not in schema["properties"]["color"] and "enumNames" in SCHEMA["properties"]["color"]
    assert schema["additionalProperties"] is False
    assert form_schema({**SCHEMA, "additionalProperties": True})["additionalProperties"] is True
    with pytest.raises(ValueError, match="object schema"):
        form_schema({"type": "array"})
    with pytest.raises(ValueError, match="object schema"):
        form_schema({"type": "object", "properties": []})

    request = elicitation_request(_form())
    assert request == elicitation_request(_form())  # the request id derives from the content alone
    assert request.request_id != elicitation_request(_form("Who else?")).request_id
    assert request.messages[1] == ChatMessage("user", "Who are you?")
    assert request.messages[0].content.endswith(json.dumps(SCHEMA, sort_keys=True, separators=(",", ":")))
    assert request.response_format.schema == schema and request.options.temperature == 0.0
    assert request.max_tokens == mcp_host.ELICITATION_MAX_TOKENS
    with pytest.raises(ValueError, match="only form elicitations"):
        elicitation_request(types.ElicitRequestURLParams(mode="url", message="m", url="https://example.com"))


def test_the_host_answers_elicitations_with_the_engine():
    engine = DllmEngine.create_default()

    async def body(host):
        first = await host.call(ToolCall("c1", "register", "{}"))
        second = await host.call(ToolCall("c2", "register", "{}"))
        return first, second, list(host.elicitations), list(host.answers)

    first, second, elicitations, answers = run(body, engine)
    assert not first.is_error and first.content == second.content
    answer = json.loads(first.content)
    assert answer["action"] == "accept"
    content = answer["content"]
    assert {"name", "age", "color"} <= set(content) <= set(SCHEMA["properties"])
    assert len(content["name"]) <= 12 and 0 <= content["age"] <= 120 and content["color"] in ("red", "green")
    assert len(elicitations) == 2 and elicitations[0] == elicitations[1] and answers == elicitations
    elicitation = elicitations[0]
    assert elicitation.server == "forms" and elicitation.message == "Who are you?"
    assert elicitation.content == content and elicitation.fingerprint == MCP_ELICITATION_FINGERPRINT
    assert elicitation.request is not None
    assert mcp_host.answer_elicitation(engine, "forms", "Who are you?", elicitation.request) == elicitation


def test_declined_cancelled_and_refused_elicitations():
    engine = DllmEngine.create_default()
    result, host = run(call("login"), engine)
    assert result.content == "decline"
    assert host.elicitations == [mcp_host.Elicitation("forms", "Sign in", "decline", None, None, None)]

    result, host = run(call("register"), engine, elicitation="decline")
    assert json.loads(result.content) == {"action": "decline", "content": None}
    assert host.elicitations[0].request is None

    result, host = run(call("essay"), engine)
    assert result.content == "cancel"
    cancelled = host.elicitations[0]
    assert cancelled.action == "cancel" and cancelled.content is None and cancelled.fingerprint is not None

    result, host = run(call("listing"), engine)
    assert result.is_error and host.answers == []  # refused: not an object schema

    result, host = run(call("register"))  # without an engine the host does not offer elicitation
    assert result.is_error and host.elicitations == []

    with pytest.raises(McpHostError, match="engine, decline"):
        McpHost({}, engine, elicitation="ask")


# Roots


def test_roots_are_offered_sorted_by_uri(tmp_path):
    (tmp_path / "b").mkdir()
    (tmp_path / "a").mkdir()
    paths = [tmp_path / "b", str(tmp_path / "a"), tmp_path / "a" / ".." / "b"]
    result, host = run(call("roots"), roots=paths)
    expected = [[(tmp_path / name).resolve().as_uri(), name] for name in ("a", "b")]
    assert json.loads(result.content) == expected
    assert [[root.uri, root.name] for root in host.roots] == expected

    result, host = run(call("roots"))  # without roots the host does not offer them
    assert result.is_error and host.roots == []

    (tmp_path / "file.txt").write_text("x", encoding="utf-8")
    with pytest.raises(McpHostError, match="is not a directory"):
        McpHost({}, roots=[tmp_path / "file.txt"])
    root = mcp_host._roots([tmp_path.anchor])[0]
    assert root.name == root.uri  # the filesystem root has no name of its own


# Transcripts


def record(engine: DllmEngine, **options) -> dict:
    request = ChatRequest([ChatMessage("user", "Set up my profile.")], 160, tool_choice=ToolChoice("named", "profile"))

    async def go():
        async with McpHost({"forms": form_server()}, engine, **options) as host:
            recorder = transcripts.Recorder(engine, request, host.tools, 2, host.servers, host.answers)
            async for event in mcp_host.chat(engine, request, host, 2):
                recorder.add(event)
            return recorder.transcript()

    return anyio.run(go)


def test_transcripts_record_and_replay_the_engines_answers():
    engine = DllmEngine.create_default()
    transcript = record(engine)
    assert record(engine) == transcript
    sampling, elicitation = transcript["server_requests"]
    assert sampling["kind"] == "sampling" and sampling["server"] == "forms" and sampling["fingerprint"]
    assert elicitation["kind"] == "elicitation" and elicitation["action"] == "accept"
    assert elicitation["fingerprint"] == MCP_ELICITATION_FINGERPRINT
    assert transcript["rounds"][0]["results"][0]["content"] == f"{sampling['content']} accept"

    outcome = transcripts.replay(engine, transcript)
    assert outcome.ok and outcome.reasons == () and outcome.transcript == transcript

    edited = json.loads(json.dumps(transcript))
    edited["server_requests"][1]["content"]["name"] = "Mallory"
    edited["id"] = transcripts.transcript_id(edited)
    outcome = transcripts.replay(engine, edited)
    assert outcome.reasons == (
        "server request 1 (elicitation for forms): the engine's answer differs from the transcript",
    )
    assert outcome.diverged_at is None  # the rounds still replay from the recorded tool results

    declined = record(engine, elicitation="decline")
    assert declined["server_requests"][1] == {
        "kind": "elicitation", "server": "forms", "message": "Who are you?", "action": "decline", "content": None,
        "request": None, "fingerprint": None,
    }  # fmt: skip
    assert transcripts.replay(engine, declined).ok


def test_transcripts_without_server_requests_keep_their_format():
    engine = DllmEngine.create_default()
    request = ChatRequest([ChatMessage("user", "Hi")], 4)
    recorder = transcripts.Recorder(engine, request, [], 1)

    async def go():
        async for event in mcp_host.chat(engine, request, McpHost({}), 1):
            recorder.add(event)

    anyio.run(go)
    assert "server_requests" not in recorder.transcript()


# The command line


def test_chat_elicits_lists_roots_and_records(tmp_path, capsys, monkeypatch):
    from etalii_dllm.engine import default_engine

    monkeypatch.delenv("DLLM_MODEL", raising=False)
    default_engine.cache_clear()
    real_host = mcp_host.McpHost
    monkeypatch.setattr(
        mcp_host, "McpHost", lambda servers, *rest, **kw: real_host({"forms": form_server()}, *rest, **kw)
    )
    base = ["chat", "Set up my profile.", "--max-tokens", "160", "--mcp-server", "forms=unused"]
    assert main([*base, "--mcp-root", str(tmp_path), "--mcp-list"]) == 0
    out = capsys.readouterr().out
    assert f"roots:\n  {tmp_path.resolve().as_uri()}: {tmp_path.name}\n" in out

    # The model is made to call the profile tool (dllm chat itself leaves the choice to the model).
    real_chat = mcp_host.chat
    named = ToolChoice("named", "profile")
    monkeypatch.setattr(
        mcp_host, "chat", lambda e, r, h, n: real_chat(e, dataclasses.replace(r, tool_choice=named), h, n)
    )
    path = tmp_path / "run.json"
    assert main([*base, "--max-tool-rounds", "2", "--transcript", str(path)]) == 0
    err = capsys.readouterr().err
    transcript = json.loads(path.read_text(encoding="utf-8"))
    sampling, elicitation = transcript["server_requests"]
    assert f"   sampled for forms: {sampling['fingerprint']}" in err
    assert f"   elicited for forms: {json.dumps(elicitation['content'], ensure_ascii=False, sort_keys=True)}" in err
    assert main(["replay", str(path)]) == 0
    assert "verified" in capsys.readouterr().out

    assert main([*base, "--max-tool-rounds", "2", "--mcp-elicit", "decline"]) == 0
    assert "   elicitation for forms: decline" in capsys.readouterr().err
    assert main(["chat", "Hi", "--mcp-root", str(tmp_path)]) == 1
    assert "--mcp-root and --mcp-list need --mcp-config" in capsys.readouterr().err
    default_engine.cache_clear()
