"""Phase 54: tool calling in the remaining formats: Phi-4-mini's functools lists with the tools on the system message
(#332), DeepSeek V3.1's markers (#333) and Command R7B's action blocks (#334), with the tool controls, in every API,
docs and a golden value (#335)."""

from __future__ import annotations

import hashlib
import json

import jsonschema
import pytest
from golden_values import REMAINING_TOOL_FORMAT_CALLS
from test_more_tool_formats import DEEPSEEK_TEMPLATE
from test_tool_controls import BOOKING
from test_tool_formats import TOOLS, WEATHER

from etalii_dllm.bpe import BpeTokenizer, bytes_to_unicode
from etalii_dllm.chat import ChatMessage, ToolCall
from etalii_dllm.chat_template import ChatTemplate
from etalii_dllm.engine import ChatRequest, DllmEngine, TextDelta, ToolCallEvent
from etalii_dllm.models import BigramModel
from etalii_dllm.sampling import SamplingOptions
from etalii_dllm.tools import (
    COMMAND_R7B,
    COMMAND_R7B_CLOSE,
    COMMAND_R7B_OPEN,
    DEEPSEEK,
    DEEPSEEK_CALL_BEGIN,
    DEEPSEEK_CALL_END,
    DEEPSEEK_CALLS_BEGIN,
    DEEPSEEK_CALLS_END,
    DEEPSEEK_SEP,
    DEEPSEEK_V31,
    FORMATS,
    PHI4_MINI,
    Tool,
    ToolChoice,
    auto_grammar,
    detect_format,
    forced_grammar,
    parse_calls,
    template_messages,
)

# Templates modelled on each model's own.
PHI4_TEMPLATE = (
    "{% for message in messages %}{% if message['role'] == 'system' and 'tools' in message and message['tools'] is "
    "not none %}{{ '<|' + message['role'] + '|>' + message['content'] + '<|tool|>' + message['tools'] + '<|/tool|>' + "
    "'<|end|>' }}{% else %}{{ '<|' + message['role'] + '|>' + message['content'] + '<|end|>' }}{% endif %}"
    "{% endfor %}{{ '<|assistant|>' }}"
)
DEEPSEEK_V31_TEMPLATE = (
    "{% if tools %}<\uff5cSystem\uff5c>Tools:{% for t in tools %}{{ '\\n' }}{{ t | tojson }}{% endfor %}{% endif %}"
    "{% for m in messages %}{% if m.role == 'user' %}<\uff5cUser\uff5c>{{ m.content }}"
    "{% elif m.tool_calls %}<\uff5cAssistant\uff5c>{{ m.content }}" + DEEPSEEK_CALLS_BEGIN +
    "{% for c in m.tool_calls %}" + DEEPSEEK_CALL_BEGIN + "{{ c.function.name }}" + DEEPSEEK_SEP +
    "{{ c.function.arguments }}" + DEEPSEEK_CALL_END + "{% endfor %}" + DEEPSEEK_CALLS_END +
    "{% elif m.role == 'tool' %}<\uff5ctool\u2581output\uff5c>{{ m.content }}"
    "{% else %}{{ m.content }}{% endif %}{% endfor %}<\uff5cAssistant\uff5c>"
)  # fmt: skip
R7B_TEMPLATE = (
    "{% if tools %}<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>## Available Tools{% for t in tools %}{{ '\\n' }}"
    "{{ t | tojson }}{% endfor %}<|END_OF_TURN_TOKEN|>{% endif %}{% for m in messages %}<|START_OF_TURN_TOKEN|>"
    "{% if m.role == 'user' %}<|USER_TOKEN|>{{ m.content }}{% elif m.role == 'assistant' %}<|CHATBOT_TOKEN|>"
    '{% if m.tool_calls %}<|START_ACTION|>[{% for c in m.tool_calls %}{"tool_call_id": "{{ loop.index0 }}", '
    '"tool_name": "{{ c.function.name }}", "parameters": {{ c.function.arguments | tojson }}}'
    "{% if not loop.last %}, {% endif %}{% endfor %}]<|END_ACTION|>{% else %}{{ m.content }}{% endif %}"
    "{% elif m.role == 'tool' %}<|SYSTEM_TOKEN|><|START_TOOL_RESULT|>{{ m.content }}<|END_TOOL_RESULT|>"
    "{% else %}{{ m.content }}{% endif %}<|END_OF_TURN_TOKEN|>{% endfor %}<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
)
TEMPLATES = {"phi4-mini": PHI4_TEMPLATE, "deepseek-v3.1": DEEPSEEK_V31_TEMPLATE, "command-r7b": R7B_TEMPLATE}
MARKERS = ["</s>", "<|tool|>", "<|/tool|>", COMMAND_R7B_OPEN, COMMAND_R7B_CLOSE, DEEPSEEK_CALLS_BEGIN,
           DEEPSEEK_CALLS_END, DEEPSEEK_CALL_BEGIN, DEEPSEEK_CALL_END, DEEPSEEK_SEP]  # fmt: skip


def tokenizer() -> BpeTokenizer:
    """A byte-level tokenizer whose markers are special tokens, as in these models' vocabularies."""
    characters = bytes_to_unicode()
    vocab = {characters[byte]: byte for byte in range(256)}
    added = [{"id": 256 + i, "content": text, "special": True} for i, text in enumerate(MARKERS)]
    spec = {
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "use_regex": True},
        "decoder": {"type": "ByteLevel"},
        "added_tokens": added,
    }
    return BpeTokenizer(spec, end_of_sequence="</s>")


def engine(name: str) -> DllmEngine:
    tokens = tokenizer()
    model = BigramModel(tokens.vocabulary_size, 54)
    return DllmEngine(model, tokens, "fp_remaining_tool_formats", chat_template=ChatTemplate(TEMPLATES[name]))


def request(choice: ToolChoice, tools=TOOLS, **options) -> ChatRequest:
    messages = [ChatMessage("user", "Weather in Rome?")]
    return ChatRequest(messages, 800, tools=list(tools), tool_choice=choice, request_id="req", **options)


# -- detection, presentation and markers --------------------------------------------------------------------------


def test_formats_are_detected_from_the_template():
    for name, source in TEMPLATES.items():
        assert detect_format(source) is FORMATS[name]
        assert engine(name).tool_format is FORMATS[name]
    assert detect_format(DEEPSEEK_TEMPLATE) is DEEPSEEK  # V3 and R1 write a fenced json block
    assert detect_format("{{ tools }}" + DEEPSEEK_CALLS_BEGIN) is DEEPSEEK_V31


def test_markers_become_visible():
    r7b = engine("command-r7b")
    assert r7b.tokenizer.decode([259, 260]) == COMMAND_R7B_OPEN + COMMAND_R7B_CLOSE
    v31 = engine("deepseek-v3.1")
    assert v31.tokenizer.decode([261, 263]) == DEEPSEEK_CALLS_BEGIN + DEEPSEEK_CALL_BEGIN
    assert engine("phi4-mini").tokenizer.decode([257]) == ""  # Phi-4-mini's calls are plain text


def test_phi4_mini_takes_the_tools_on_the_system_message():
    chat = engine("phi4-mini")
    prompt = chat.render_chat([ChatMessage("user", "Noon?")], TOOLS)
    functions = json.dumps([t.to_openai()["function"] for t in TOOLS], ensure_ascii=False)
    assert prompt == f"<|system|><|tool|>{functions}<|/tool|><|end|><|user|>Noon?<|end|><|assistant|>"
    with_system = chat.render_chat([ChatMessage("system", "Be brief."), ChatMessage("user", "Noon?")], TOOLS)
    assert with_system.startswith(f"<|system|>Be brief.<|tool|>{functions}<|/tool|><|end|>")
    assert chat.render_chat([ChatMessage("user", "Noon?")]) == "<|user|>Noon?<|end|><|assistant|>"
    call = ToolCall("call_1", "get_weather", '{"city": "Rome", "unit": "C"}')
    entries = template_messages([ChatMessage("assistant", "Checking.", (call,))], PHI4_MINI)
    assert entries == [
        {"role": "assistant", "content": 'Checking.functools[{"name": "get_weather", "arguments": {"city": "Rome", '
         '"unit": "C"}}]'}
    ]  # fmt: skip


# -- constraints --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_required_calls_are_valid_in_each_format(name):
    chat = engine(name)
    for seed in range(3):
        options = SamplingOptions(temperature=0.8, seed=seed)
        result = chat.chat_completion(request(ToolChoice("named", "get_weather", parallel=False), options=options))
        assert result.finish_reason == "tool_calls" and result.tool_calls, result
        for call in result.tool_calls:
            assert call.name == "get_weather"
            jsonschema.validate(json.loads(call.arguments), WEATHER.parameters)
    single = ToolChoice("named", "get_weather", parallel=False)
    stream = list(chat.chat_stream(request(single)))
    result = chat.chat_completion(request(single))
    assert result.tool_calls
    assert [e.call for e in stream if isinstance(e, ToolCallEvent)] == list(result.tool_calls)
    assert not any(e.text for e in stream if isinstance(e, TextDelta))


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_tool_controls_in_each_format(name):
    chat = engine(name)
    for seed in range(3):
        options = SamplingOptions(temperature=0.9, seed=seed)
        book = ToolChoice("named", "book", parallel=False)
        strict = chat.chat_completion(request(book, tools=[*TOOLS, BOOKING], options=options))
        for call in strict.tool_calls:
            jsonschema.validate(json.loads(call.arguments), BOOKING.parameters)
        single = chat.chat_completion(request(ToolChoice("required", parallel=False), options=options))
        assert len(single.tool_calls) == 1
        choice = ToolChoice("required", allowed=("is_noon",), parallel=False)
        allowed = chat.chat_completion(request(choice, options=options))
        assert {c.name for c in allowed.tool_calls} == {"is_noon"}


def test_auto_calls_start_at_each_marker():
    assert auto_grammar(TOOLS, PHI4_MINI)[1] == "functools"
    assert auto_grammar(TOOLS, DEEPSEEK_V31)[1] == DEEPSEEK_CALLS_BEGIN
    assert auto_grammar(TOOLS, COMMAND_R7B)[1] == COMMAND_R7B_OPEN


def test_call_grammars():
    v31 = forced_grammar(TOOLS, ToolChoice("required"), DEEPSEEK_V31).matcher()
    call = f'{DEEPSEEK_CALL_BEGIN}get_weather{DEEPSEEK_SEP}{{"city": "Rome", "unit": "C"}}{DEEPSEEK_CALL_END}'
    noon = f"{DEEPSEEK_CALL_BEGIN}is_noon{DEEPSEEK_SEP}{{}}{DEEPSEEK_CALL_END}"
    assert v31.matches(f"{DEEPSEEK_CALLS_BEGIN}{call}{noon}{DEEPSEEK_CALLS_END}".encode())
    assert not v31.matches(f"{DEEPSEEK_CALLS_BEGIN}{DEEPSEEK_CALLS_END}".encode())
    single = forced_grammar(TOOLS, ToolChoice("required", parallel=False), DEEPSEEK_V31).matcher()
    assert not single.matches(f"{DEEPSEEK_CALLS_BEGIN}{call}{noon}{DEEPSEEK_CALLS_END}".encode())
    r7b = forced_grammar(TOOLS, ToolChoice("required"), COMMAND_R7B).matcher()
    action = '[{"tool_call_id": "0", "tool_name": "is_noon", "parameters": {}}]'
    assert r7b.matches(f"{COMMAND_R7B_OPEN}{action}{COMMAND_R7B_CLOSE}".encode())
    assert not r7b.matches(f"{COMMAND_R7B_OPEN}{action}".encode())
    assert not r7b.matches(f'{COMMAND_R7B_OPEN}[{{"name": "is_noon", "arguments": {{}}}}]{COMMAND_R7B_CLOSE}'.encode())
    phi = forced_grammar(TOOLS, ToolChoice("required"), PHI4_MINI).matcher()
    assert phi.matches(b'functools[{"name": "is_noon", "arguments": {}}]')


# -- parsing ------------------------------------------------------------------------------------------------------


def test_parse_deepseek_v31_calls():
    call = f'{DEEPSEEK_CALL_BEGIN}get_weather{DEEPSEEK_SEP}{{"city": "Rome"}}{DEEPSEEK_CALL_END}'
    noon = f"{DEEPSEEK_CALL_BEGIN}is_noon{DEEPSEEK_SEP}{{}}{DEEPSEEK_CALL_END}"
    text = f"Checking.{DEEPSEEK_CALLS_BEGIN}{call}{noon}{DEEPSEEK_CALLS_END}"
    assert parse_calls(text, TOOLS, DEEPSEEK_V31) == ("Checking.", [("get_weather", '{"city": "Rome"}'),
                                                                   ("is_noon", "{}")])  # fmt: skip
    broken = f"{DEEPSEEK_CALLS_BEGIN}{DEEPSEEK_CALL_BEGIN}rm{DEEPSEEK_SEP}{{}}{DEEPSEEK_CALL_END}{DEEPSEEK_CALLS_END}"
    assert parse_calls(broken, TOOLS, DEEPSEEK_V31) == (broken, [])
    assert parse_calls(text, TOOLS, DEEPSEEK) == (text, [])  # V3's parser wants the fenced block


def test_parse_command_r7b_calls():
    action = '[{"tool_call_id": "0", "tool_name": "get_weather", "parameters": {"city": "Rome", "unit": "C"}}]'
    text = f"I'll look.{COMMAND_R7B_OPEN}{action}{COMMAND_R7B_CLOSE}"
    expected = ("I'll look.", [("get_weather", json.dumps({"city": "Rome", "unit": "C"}))])
    assert parse_calls(text, TOOLS, COMMAND_R7B) == expected
    assert parse_calls(text + " Done.", TOOLS, COMMAND_R7B) == expected
    wrong_key = f'{COMMAND_R7B_OPEN}[{{"name": "is_noon", "parameters": {{}}}}]{COMMAND_R7B_CLOSE}'
    assert parse_calls(wrong_key, TOOLS, COMMAND_R7B) == (wrong_key, [])


def test_parse_phi4_mini_calls():
    text = 'functools[{"name": "is_noon", "arguments": {}}, {"name": "get_weather", "arguments": {"city": "Paris"}}]'
    assert parse_calls(text, TOOLS, PHI4_MINI) == ("", [("is_noon", "{}"), ("get_weather", '{"city": "Paris"}')])
    assert parse_calls("No functools here.", TOOLS, PHI4_MINI) == ("No functools here.", [])


# -- history, round trips and the golden value --------------------------------------------------------------------


def test_history_renders_in_each_format():
    call = ToolCall("call_0123456789abcdef01234567", "is_noon", "{}")
    history = [
        ChatMessage("user", "Noon?"),
        ChatMessage("assistant", "", (call,)),
        ChatMessage("tool", "yes", tool_call_id=call.id, name="is_noon"),
    ]
    prompts = {name: engine(name).render_chat(history, TOOLS) for name in TEMPLATES}
    assert (
        '<|assistant|>functools[{"name": "is_noon", "arguments": {}}]<|end|><|tool|>yes<|end|>' in prompts["phi4-mini"]
    )
    assert f"{DEEPSEEK_CALL_BEGIN}is_noon{DEEPSEEK_SEP}{{}}{DEEPSEEK_CALL_END}" in prompts["deepseek-v3.1"]
    assert '"tool_name": "is_noon", "parameters": {}}]<|END_ACTION|>' in prompts["command-r7b"]


def test_calls_round_trip_through_the_conversation():
    for name in TEMPLATES:
        chat = engine(name)
        first = chat.chat_completion(request(ToolChoice("named", "get_weather", parallel=False)))
        messages = [
            ChatMessage("user", "Weather in Rome?"),
            ChatMessage("assistant", first.content, first.tool_calls),
            ChatMessage("tool", "sunny", tool_call_id=first.tool_calls[0].id, name="get_weather"),
        ]
        follow = ChatRequest(messages, 8, tools=TOOLS, tool_choice=ToolChoice("none"), request_id="req2")
        assert chat.chat_completion(follow).finish_reason in ("stop", "length")


def test_golden_remaining_tool_format_calls():
    calls = []
    for name in sorted(TEMPLATES):
        chat = engine(name)
        for choice in (ToolChoice("required"), ToolChoice("named", "get_weather", parallel=False), ToolChoice("auto")):
            options = SamplingOptions(temperature=0.7, seed=3)
            result = chat.chat_completion(request(choice, tools=[*TOOLS, Tool("ping")], options=options))
            calls.append(
                [name, result.finish_reason, result.content, [[c.id, c.name, c.arguments] for c in result.tool_calls]]
            )
    digest = hashlib.sha256(json.dumps(calls, sort_keys=True).encode()).hexdigest()
    assert digest == REMAINING_TOOL_FORMAT_CALLS, json.dumps(calls)
