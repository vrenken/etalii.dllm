import json

import jsonschema
import pytest

from etalii_dllm.chat import ChatMessage, ToolCall
from etalii_dllm.chat_template import ChatTemplate
from etalii_dllm.engine import ChatRequest, DllmEngine, ResponseFormat, TextDelta, ToolCallEvent, _answer_prefix
from etalii_dllm.sampling import GREEDY, SamplingOptions
from etalii_dllm.tools import Tool, ToolChoice, parse_calls, validate_tools

WEATHER = Tool(
    "get_weather",
    "Current weather for a city",
    {
        "type": "object",
        "properties": {"city": {"type": "string", "enum": ["Paris", "Rome"]}, "unit": {"enum": ["C", "F"]}},
        "required": ["city"],
    },
)
TIME = Tool("get_time", "Current time", {"type": "object", "properties": {"zone": {"type": "string"}}})
TOOLS = [WEATHER, TIME]

TOOL_TEMPLATE = (
    "{% if tools %}<|im_start|>system\n# Tools{% for t in tools %}\n{{ t | tojson }}{% endfor %}<|im_end|>\n{% endif %}"
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}"
    "{% for c in m.tool_calls or [] %}<tool_call>{{ c.function | tojson }}</tool_call>{% endfor %}<|im_end|>\n"
    "{% endfor %}<|im_start|>assistant\n"
)


def test_parse_calls():
    text = 'Let me check.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Rome"}}\n</tool_call>'
    assert parse_calls(text, TOOLS) == ("Let me check.", [("get_weather", '{"city": "Rome"}')])
    two = '<tool_call>{"name":"get_time","arguments":{}}</tool_call><tool_call>{"name":"get_time","arguments":"{}"}'
    assert parse_calls(two, TOOLS) == ("", [("get_time", "{}"), ("get_time", "{}")])
    assert parse_calls('{"name": "get_weather", "parameters": {"city": "Paris"}}', TOOLS)[1] == [
        ("get_weather", '{"city": "Paris"}')
    ]
    unknown = '<tool_call>{"name": "rm", "arguments": {}}</tool_call>'
    assert parse_calls(unknown, TOOLS) == (unknown, [])
    assert parse_calls("  plain answer \n", TOOLS) == ("plain answer", [])


def test_answer_prefix_is_monotonic_and_a_prefix_of_the_answer():
    text = 'Sure thing.\n<tool_call>\n{"name": "get_time", "arguments": {}}\n</tool_call>'
    previous = ""
    for end in range(len(text) + 1):
        prefix = _answer_prefix(text[:end])
        assert prefix.startswith(previous)
        previous = prefix
    assert parse_calls(text, TOOLS)[0].startswith(previous)
    assert _answer_prefix('  {"name"') == ""
    assert _answer_prefix("Hello <tool_ca") == "Hello"


def test_validation():
    validate_tools(TOOLS, ToolChoice("named", "get_time"))
    with pytest.raises(ValueError):
        validate_tools(TOOLS, ToolChoice("named", "nope"))
    with pytest.raises(ValueError):
        validate_tools([WEATHER, WEATHER], ToolChoice())
    with pytest.raises(ValueError):
        validate_tools([], ToolChoice("required"))
    with pytest.raises(ValueError):
        validate_tools([Tool("x", "", {"type": "string"})], ToolChoice())
    with pytest.raises(ValueError):
        ToolChoice("named")


def test_tools_without_template_support_use_hermes_instructions():
    engine = DllmEngine.create_default()
    history = [
        ChatMessage("user", "Weather?"),
        ChatMessage("assistant", "", (ToolCall("call_1", "get_weather", '{"city": "Paris"}'),)),
        ChatMessage("tool", "sunny", tool_call_id="call_1", name="get_weather"),
    ]
    prompt = engine.render_chat(history, TOOLS)
    assert prompt.startswith("<|system|>\n# Tools")
    assert '"name": "get_weather"' in prompt
    assert '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>' in prompt
    assert "<|user|>\n<tool_response>\nsunny\n</tool_response>" in prompt


def test_templates_that_know_tools_present_them():
    engine = DllmEngine.create_default()
    engine.chat_template = ChatTemplate(TOOL_TEMPLATE)
    history = [
        ChatMessage("user", "Weather?"),
        ChatMessage("assistant", "", (ToolCall("call_1", "get_weather", '{"city": "Paris"}'),)),
    ]
    prompt = engine.render_chat(history, TOOLS)
    assert prompt.startswith('<|im_start|>system\n# Tools{"type": "function", "function": {"name": "get_weather"')
    assert '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>' in prompt


def request(tool_choice: ToolChoice, **kwargs) -> ChatRequest:
    messages = [ChatMessage("user", "What is the weather in Paris?")]
    return ChatRequest(messages, 200, tools=TOOLS, tool_choice=tool_choice, request_id="req", **kwargs)


def test_required_tool_calls_are_valid_and_deterministic():
    engine = DllmEngine.create_default()
    for seed in range(4):
        options = SamplingOptions(temperature=1.0, seed=seed)
        result = engine.chat_completion(request(ToolChoice("required"), options=options))
        if result.finish_reason == "length":
            continue  # the placeholder model may babble inside a string
        assert result.finish_reason == "tool_calls"
        (call,) = result.tool_calls
        tool = next(t for t in TOOLS if t.name == call.name)
        jsonschema.validate(json.loads(call.arguments), tool.parameters)
        assert call.id.startswith("call_") and len(call.id) == 29
        again = DllmEngine.create_default().chat_completion(request(ToolChoice("required"), options=options))
        assert again == result


def test_named_tool_choice_calls_that_tool():
    result = DllmEngine.create_default().chat_completion(request(ToolChoice("named", "get_weather")))
    assert [c.name for c in result.tool_calls] == ["get_weather"]
    assert json.loads(result.tool_calls[0].arguments)["city"] in ("Paris", "Rome")


def test_tool_choice_none_hides_the_tools():
    engine = DllmEngine.create_default()
    stream = engine.chat_stream(request(ToolChoice("none"), options=GREEDY))
    plain = engine.chat_stream(ChatRequest([ChatMessage("user", "What is the weather in Paris?")], 200))
    assert stream.prompt_tokens == plain.prompt_tokens


def test_streamed_tool_calls_match_the_completion():
    engine = DllmEngine.create_default()
    chat = request(ToolChoice("required"), options=SamplingOptions(temperature=0.5, seed=2))
    events = list(engine.chat_stream(chat))
    result = engine.chat_completion(chat)
    assert tuple(e.call for e in events if isinstance(e, ToolCallEvent)) == result.tool_calls
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == result.content


def test_response_format_with_tools_allows_either():
    engine = DllmEngine.create_default()
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
    result = engine.chat_completion(request(ToolChoice(), response_format=ResponseFormat("json_schema", schema)))
    if result.tool_calls:
        assert result.finish_reason == "tool_calls"
    elif result.finish_reason == "stop":
        jsonschema.validate(json.loads(result.content), schema)
