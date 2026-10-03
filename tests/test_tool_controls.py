"""Phase 53: exact tool call controls: strict tools (#327), at most one call per answer (#328), allowed tools (#329),
and receipts, the response cache, the APIs and a golden value (#330)."""

from __future__ import annotations

import hashlib
import json
import re

import jsonschema
import pytest
import test_more_tool_formats as more
import test_tool_formats as base
from fastapi.testclient import TestClient
from golden_values import TOOL_CONTROL_CALLS
from test_tool_formats import NOON, TOOLS, WEATHER

from etalii_dllm import receipts, serving
from etalii_dllm.chat import ChatMessage
from etalii_dllm.engine import ChatRequest, DllmEngine
from etalii_dllm.grammar import Grammar, GrammarError, TokenConstraint
from etalii_dllm.sampling import SamplingOptions
from etalii_dllm.server import anthropic_api, responses_api
from etalii_dllm.server.app import _chat_request, app
from etalii_dllm.server.contracts import ChatCompletionRequest
from etalii_dllm.tools import (
    AUTO,
    DEEPSEEK,
    GRANITE,
    HERMES,
    LLAMA3,
    MISTRAL,
    PYTHONIC,
    XML,
    Tool,
    ToolChoice,
    auto_grammar,
    forced_grammar,
    parse_calls,
    validate_tools,
)

BOOKING = Tool(
    "book",
    "Book a table",
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "pattern": "^[A-Z][a-z]+$", "maxLength": 6},
            "guests": {"type": "integer", "minimum": 2, "maximum": 4},
            "time": {"type": "string", "format": "time"},
            "seat": {"$ref": "#/$defs/seat"},
        },
        "required": ["name", "guests", "seat"],
        "additionalProperties": False,
        "$defs": {"seat": {"type": "string", "enum": ["in", "out"]}},
    },
    strict=True,
)
LOOSE = Tool(BOOKING.name, BOOKING.description, BOOKING.parameters)
TIME = r"(?:[01]\d|2[0-3]):[0-5]\d:(?:[0-5]\d|60)(?:\.\d+)?(?:[Zz]|[+-](?:[01]\d|2[0-3]):[0-5]\d)"
FORMATS = ("hermes", "llama3", "mistral", "granite", "xml", "deepseek", "pythonic")


def engine(name: str) -> DllmEngine:
    return more.engine(name) if name in more.TEMPLATES else base.engine(name)


def ask(choice: ToolChoice, tools=(WEATHER, NOON, BOOKING), seed: int = 3) -> ChatRequest:
    messages = [ChatMessage("user", "Book a table for Ann?")]
    options = SamplingOptions(temperature=0.9, seed=seed)
    return ChatRequest(messages, 800, options, tools=list(tools), tool_choice=choice, request_id="req")


# -- the choice ---------------------------------------------------------------------------------------------------


def test_tool_choice_fields():
    for bad in (
        lambda: ToolChoice("named", "x", allowed=("x",)),
        lambda: ToolChoice("none", allowed=("x",)),
        lambda: ToolChoice("auto", allowed=()),
        lambda: ToolChoice("auto", allowed=("x", "x")),
    ):
        with pytest.raises(ValueError):
            bad()
    choice = ToolChoice("required", allowed=("is_noon",), parallel=False)
    assert choice.callable(TOOLS) == [NOON]
    assert ToolChoice("named", "get_weather").callable(TOOLS) == [WEATHER]
    assert AUTO.callable(TOOLS) == TOOLS
    assert choice.record() == {"mode": "required", "name": None, "allowed": ["is_noon"], "parallel": False}
    assert ToolChoice.from_record(choice.record()) == choice
    assert AUTO.record() == {"mode": "auto", "name": None}  # older receipts stay the same
    assert ToolChoice.from_record({"mode": "auto", "name": None}) == AUTO
    with pytest.raises(ValueError, match="unknown tool 'rm'"):
        validate_tools(TOOLS, ToolChoice("auto", allowed=("rm",)))


# -- strict tools -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", FORMATS)
def test_strict_arguments_satisfy_the_whole_schema(name):
    chat = engine(name)
    for seed in range(4):
        result = chat.chat_completion(ask(ToolChoice("named", "book"), seed=seed))
        assert result.tool_calls
        for call in result.tool_calls:
            arguments = json.loads(call.arguments)
            jsonschema.validate(arguments, BOOKING.parameters)
            assert "time" not in arguments or re.fullmatch(TIME, arguments["time"]), arguments


def test_strict_and_lenient_grammars():
    strict = forced_grammar([BOOKING], ToolChoice("required"), HERMES).matcher()
    loose = forced_grammar([LOOSE], ToolChoice("required"), HERMES).matcher()
    good = '<tool_call>{"name": "book", "arguments": {"name": "Ann", "guests": 2, "seat": "in"}}</tool_call>'
    for text, valid in [
        (good, True),
        (good.replace("Ann", "ann"), False),
        (good.replace("Ann", "Annabel"), False),
        (good.replace('"guests": 2', '"guests": 5'), False),
        (good.replace('"in"', '"up"'), False),
    ]:
        assert strict.matches(text.encode()) == valid, text
        assert loose.matches(text.encode()) or "up" in text, text
    xml = forced_grammar([BOOKING], ToolChoice("required"), XML).matcher()
    call = (
        "<tool_call>\n<function=book>\n<parameter=name>\nAnn\n</parameter>\n<parameter=guests>\n3\n</parameter>\n"
        '<parameter=time>\n12:30:00Z\n</parameter>\n<parameter=seat>\n"out"\n</parameter>\n</function>\n</tool_call>'
    )
    assert xml.matches(call.encode())
    assert not xml.matches(call.replace("12:30:00Z", "noon").encode())
    assert not xml.matches(call.replace("Ann", "Ann Lee").encode())
    assert not xml.matches(call.replace("\n3\n", "\n9\n").encode())


def test_strict_raw_strings_in_the_xml_format():
    def tool(prop, **extra):
        schema = {"type": "object", "properties": {"x": prop}, "required": ["x"], **extra}
        return Tool("t", parameters=schema, strict=True)

    def matcher(prop):
        return forced_grammar([tool(prop)], ToolChoice("required"), XML).matcher()

    def text(value):
        return f"<tool_call>\n<function=t>\n<parameter=x>\n{value}\n</parameter>\n</function>\n</tool_call>".encode()

    pick = matcher({"type": "string", "enum": ["ab", "abc", "x<y"], "maxLength": 2})
    assert pick.matches(text("ab")) and not pick.matches(text("abc")) and not pick.matches(text("x<y"))
    fixed = matcher({"type": "string", "const": "on", "enum": ["on", "off"]})
    assert fixed.matches(text("on")) and not fixed.matches(text("off"))
    assert matcher({"type": "string", "const": "on"}).matches(text("on"))
    with pytest.raises(GrammarError, match="cannot use anyOf"):
        matcher({"type": "string", "anyOf": [{"maxLength": 1}]})
    with pytest.raises(GrammarError, match="cannot enforce minProperties"):
        forced_grammar([tool({"type": "string"}, minProperties=1)], ToolChoice("required"), XML)
    with pytest.raises(GrammarError, match="required arguments without properties: y"):
        forced_grammar([Tool("t", parameters={"type": "object", "required": ["y"]}, strict=True)], AUTO, PYTHONIC)
    for bad in ({"type": "string", "pattern": 1}, {"type": "string", "minLength": 3, "maxLength": 2}):
        with pytest.raises(GrammarError):
            Grammar.raw_string(bad, r"[^<]*")
    # Formats that write the arguments object as JSON enforce object keywords.
    counted = tool({"type": "string"}, minProperties=1)
    assert forced_grammar([counted], ToolChoice("required"), HERMES).matcher()


def test_strict_tools_refuse_what_they_cannot_enforce():
    chat = engine("hermes")
    odd = Tool("odd", parameters={"type": "object", "dependentSchemas": {}}, strict=True)
    with pytest.raises(ValueError, match="dependentSchemas"):
        chat.chat_stream(ask(ToolChoice("required"), tools=[odd]))
    lenient = Tool("odd", parameters={"type": "object", "dependentSchemas": {}})
    assert chat.chat_completion(ask(ToolChoice("required"), tools=[lenient])).tool_calls


def test_python_calls_of_strict_tools():
    grammar = forced_grammar([BOOKING, NOON], ToolChoice("required"), PYTHONIC).matcher()
    assert grammar.matches(b'[book(name="Ann", guests=2, seat="in")]')
    assert not grammar.matches(b'[book(name="Ann", guests=7, seat="in")]')
    flag = Tool("f", parameters={"type": "object", "properties": {"on": {"type": "boolean"}}}, strict=True)
    assert forced_grammar([flag], ToolChoice("required"), PYTHONIC).matcher().matches(b"[f(on=True)]")


# -- one call per answer ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", [MISTRAL, GRANITE, DEEPSEEK, PYTHONIC])
def test_one_call_lists(fmt):
    def call(tool_name: str) -> str:
        if fmt is PYTHONIC:
            return f"{tool_name}()"
        if fmt is DEEPSEEK:
            head = f"{more.DEEPSEEK_CALL_BEGIN}function{more.DEEPSEEK_SEP}{tool_name}"
            return f"{head}\n```json\n{{}}\n```{more.DEEPSEEK_CALL_END}"
        return json.dumps({"name": tool_name, "arguments": {}})

    def reply(*names: str) -> bytes:
        if fmt is PYTHONIC:
            return ("[" + ", ".join(call(n) for n in names) + "]").encode()
        if fmt is DEEPSEEK:
            return (fmt.open + "\n".join(call(n) for n in names) + fmt.close).encode()
        return (fmt.open + "[" + ", ".join(call(n) for n in names) + "]").encode()

    for tools in ([NOON], [NOON, Tool("other", strict=True)]):
        many = forced_grammar(tools, ToolChoice("required"), fmt).matcher()
        one = forced_grammar(tools, ToolChoice("required", parallel=False), fmt).matcher()
        assert many.matches(reply("is_noon", "is_noon")) and many.matches(reply("is_noon"))
        assert one.matches(reply("is_noon")) and not one.matches(reply("is_noon", "is_noon"))


def test_auto_answers_end_after_one_call():
    trie = engine("hermes")._token_trie()
    grammar, trigger = auto_grammar(TOOLS, HERMES, ToolChoice("auto", parallel=False))
    once = TokenConstraint(grammar, trie, trigger=trigger, once=True)
    once.accept_bytes(b'Checking. <tool_call>{"name": "is_noon", "arguments": {}}</tool_call>')
    assert once.finished and once.may_stop
    again = TokenConstraint(grammar, trie, trigger=trigger)
    again.accept_bytes(b'<tool_call>{"name": "is_noon", "arguments": {}}</tool_call>')
    assert not again.finished and not again.active
    with pytest.raises(ValueError, match="'once' needs a trigger"):
        TokenConstraint(grammar, trie, once=True)


@pytest.mark.parametrize("name", FORMATS)
def test_answers_have_at_most_one_call(name):
    chat = engine(name)
    for mode in ("auto", "required"):
        for seed in range(3):
            result = chat.chat_completion(ask(ToolChoice(mode, parallel=False), seed=seed))
            assert len(result.tool_calls) <= 1 if mode == "auto" else len(result.tool_calls) == 1


# -- allowed tools ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", FORMATS)
def test_allowed_tools_keep_the_prompt(name):
    chat = engine(name)
    narrowed = ask(ToolChoice("required", allowed=("is_noon",)))
    assert chat.request_prompt(narrowed, narrowed.tools) == chat.request_prompt(ask(AUTO), narrowed.tools)
    for seed in range(3):
        result = chat.chat_completion(ask(ToolChoice("required", allowed=("is_noon",)), seed=seed))
        assert [c.name for c in result.tool_calls] and {c.name for c in result.tool_calls} == {"is_noon"}
        auto = chat.chat_completion(ask(ToolChoice("auto", allowed=("is_noon",)), seed=seed))
        assert {c.name for c in auto.tool_calls} <= {"is_noon"}


def test_allowed_tools_in_the_grammar():
    grammar, _ = auto_grammar(TOOLS, HERMES, ToolChoice("auto", allowed=("is_noon",)))
    matcher = grammar.matcher()
    assert matcher.matches(b'{"name": "is_noon", "arguments": {}}</tool_call>')
    assert not matcher.matches(b'{"name": "get_weather", "arguments": {"city": "Rome", "unit": "C"}}</tool_call>')
    bare = auto_grammar(TOOLS, LLAMA3, ToolChoice("auto", allowed=("is_noon",)))[0].matcher()
    assert bare.matches(b'{"name": "is_noon", "parameters": {}}') and bare.matches(b"Hello")
    assert not bare.matches(b'{"name": "get_weather", "parameters": {"city": "Rome", "unit": "C"}}')


def test_a_bare_call_to_a_tool_that_is_not_allowed_stays_text():
    text = '{"name": "get_weather", "arguments": {"city": "Rome", "unit": "C"}}'
    assert parse_calls(text, ToolChoice("auto", allowed=("is_noon",)).callable(TOOLS), HERMES) == (text, [])


# -- receipts, the cache and the MCP host -------------------------------------------------------------------------


def test_receipts_record_the_controls_only_when_set():
    plain = receipts.request_record(ask(AUTO, tools=TOOLS))
    assert "strict" not in json.dumps(plain["tools"]) and plain["tool_choice"] == {"mode": "auto", "name": None}
    controlled = ask(ToolChoice("auto", allowed=("book",), parallel=False))
    record = receipts.request_record(controlled)
    assert record["tools"][2]["strict"] is True
    assert record["tool_choice"] == {"mode": "auto", "name": None, "allowed": ["book"], "parallel": False}
    assert receipts.request_from_record(record).tools == tuple(controlled.tools)
    assert receipts.request_from_record(record).tool_choice == controlled.tool_choice
    chat = engine("hermes")
    keys = {serving.response_key(chat, r) for r in (ask(AUTO), controlled, ask(ToolChoice("auto", parallel=False)))}
    assert len(keys) == 3


def test_receipts_replay_tool_controls():
    chat = engine("granite")
    result = chat.chat_completion(ask(ToolChoice("required", allowed=("book",), parallel=False)))
    assert result.receipt["request"]["tool_choice"]["parallel"] is False
    assert receipts.verify(chat, result.receipt).ok


# -- the APIs -----------------------------------------------------------------------------------------------------


def openai_request(**fields) -> ChatCompletionRequest:
    tools = [{"type": "function", "function": {"name": t.name, "parameters": dict(t.parameters)}} for t in TOOLS]
    tools[0]["function"]["strict"] = True
    body = {"model": "m", "messages": [{"role": "user", "content": "Hi"}], "tools": tools, **fields}
    return ChatCompletionRequest.model_validate(body)


def test_openai_tool_controls():
    chat = engine("hermes")
    request = _chat_request(openai_request(parallel_tool_calls=False), chat)
    assert request.tools[0].strict and not request.tools[1].strict
    assert request.tool_choice == ToolChoice("auto", parallel=False)
    allowed = {"type": "allowed_tools", "allowed_tools": {"mode": "required", "tools": [
        {"type": "function", "function": {"name": "is_noon"}}]}}  # fmt: skip
    request = _chat_request(openai_request(tool_choice=allowed), chat)
    assert request.tool_choice == ToolChoice("required", allowed=("is_noon",))
    named = _chat_request(openai_request(tool_choice="none", parallel_tool_calls=False), chat)
    assert named.tool_choice == ToolChoice("none", parallel=False)


def test_openai_endpoint_allowed_tools():
    client = TestClient(app)
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "Noon?"}],
        "tools": [{"type": "function", "function": {"name": t.name, "parameters": dict(t.parameters)}} for t in TOOLS],
        "tool_choice": {"type": "allowed_tools", "allowed_tools": {"mode": "required", "tools": [
            {"type": "function", "function": {"name": "is_noon"}}]}},
        "parallel_tool_calls": False,
        "max_tokens": 600,
    }  # fmt: skip
    response = client.post("/v1/chat/completions", json=body).json()
    calls = response["choices"][0]["message"]["tool_calls"]
    assert [c["function"]["name"] for c in calls] == ["is_noon"]
    body["tool_choice"]["allowed_tools"]["tools"][0]["function"]["name"] = "rm"
    assert client.post("/v1/chat/completions", json=body).status_code == 400


def test_responses_tool_controls():
    def tools(**fields):
        listed = [{"type": "function", "name": t.name, "parameters": dict(t.parameters)} for t in TOOLS]
        listed[0]["strict"] = True
        return responses_api._tools(responses_api.ResponsesRequest.model_validate({"tools": listed, **fields}))

    converted, choice = tools(parallel_tool_calls=False)
    assert converted[0].strict and choice == ToolChoice("auto", parallel=False)
    allowed = {"type": "allowed_tools", "mode": "required", "tools": [{"type": "function", "name": "is_noon"}]}
    assert tools(tool_choice=allowed)[1] == ToolChoice("required", allowed=("is_noon",))
    assert tools(tool_choice={"type": "allowed_tools", "tools": [{"type": "function", "name": "is_noon"}]})[
        1
    ] == ToolChoice("auto", allowed=("is_noon",))
    with pytest.raises(ValueError, match="functions with a name"):
        tools(tool_choice={"type": "allowed_tools", "tools": [{"type": "web_search"}]})
    with pytest.raises(ValueError, match="allowed_tools"):
        tools(tool_choice={"type": "web_search"})
    assert tools(tool_choice="none", parallel_tool_calls=False)[1] == ToolChoice("none", parallel=False)
    assert tools(tool_choice={"type": "function", "name": "is_noon"})[1] == ToolChoice("named", "is_noon")


def test_responses_endpoint_shows_strict_tools():
    client = TestClient(app)
    responses_api.store.clear()
    tool = {"type": "function", "name": "is_noon", "parameters": dict(NOON.parameters), "strict": True}
    body = {"model": "m", "input": "Noon?", "tools": [tool], "tool_choice": "required", "max_output_tokens": 600}
    response = client.post("/v1/responses", json=body).json()
    assert response["tools"][0]["strict"] is True
    assert [o["name"] for o in response["output"] if o["type"] == "function_call"] == ["is_noon"]
    responses_api.store.clear()


def test_anthropic_tool_controls():
    from etalii_dllm.server.anthropic_contracts import ToolChoiceModel, ToolDefinition

    definitions = [ToolDefinition(name=t.name, input_schema=dict(t.parameters), strict=t is WEATHER) for t in TOOLS]
    for kind, mode in (("auto", "auto"), ("any", "required")):
        converted, choice = anthropic_api._tools(
            definitions, ToolChoiceModel(type=kind, disable_parallel_tool_use=True)
        )
        assert converted[0].strict and not converted[1].strict
        assert choice == ToolChoice(mode, parallel=False)
    choice = anthropic_api._tools(
        definitions, ToolChoiceModel(type="tool", name="is_noon", disable_parallel_tool_use=True)
    )[1]
    assert choice == ToolChoice("named", "is_noon", parallel=False)
    assert anthropic_api._tools(definitions, None)[1] == AUTO


# -- the golden value ---------------------------------------------------------------------------------------------


def test_golden_tool_control_calls():
    calls = []
    for name in FORMATS:
        chat = engine(name)
        for choice in (
            ToolChoice("named", "book"),
            ToolChoice("required", allowed=("is_noon", "book"), parallel=False),
            ToolChoice("auto", allowed=("book",), parallel=False),
        ):
            result = chat.chat_completion(ask(choice))
            calls.append(
                [name, result.finish_reason, result.content, [[c.id, c.name, c.arguments] for c in result.tool_calls]]
            )
    digest = hashlib.sha256(json.dumps(calls, sort_keys=True).encode()).hexdigest()
    assert digest == TOOL_CONTROL_CALLS, json.dumps(calls)


def test_mcp_host_rounds_keep_the_controls():
    import asyncio

    from etalii_dllm import mcp_host

    class Host:
        tools = TOOLS

        async def call(self, call):
            return mcp_host.ToolResult(call, "fake", "yes", False)

    chat = engine("hermes")
    seen: list[ToolChoice] = []
    original = chat.chat_stream

    def recording(request, **kwargs):
        seen.append(request.tool_choice)
        return original(request, **kwargs)

    chat.chat_stream = recording
    request = ask(ToolChoice("required", allowed=("is_noon",), parallel=False), tools=())

    async def run():
        return [event async for event in mcp_host.chat(chat, request, Host(), max_rounds=2)]

    asyncio.run(run())
    assert seen == [ToolChoice("required", allowed=("is_noon",), parallel=False),
                    ToolChoice("auto", allowed=("is_noon",), parallel=False)]  # fmt: skip
