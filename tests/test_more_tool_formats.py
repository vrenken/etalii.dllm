"""Phase 52: tool calling in more model formats: Qwen3-Coder's XML parameters (#322), DeepSeek's markers (#323) and
Python-style call lists (#324), in every API, docs and golden values (#325)."""

from __future__ import annotations

import hashlib
import json

import jsonschema
import pytest
from golden_values import MORE_TOOL_FORMAT_CALLS
from test_tool_formats import NOON, TOOLS, WEATHER

from etalii_dllm.bpe import BpeTokenizer, bytes_to_unicode
from etalii_dllm.chat import ChatMessage, ToolCall
from etalii_dllm.chat_template import ChatTemplate
from etalii_dllm.engine import ChatRequest, DllmEngine, TextDelta, ToolCallEvent, _answer_prefix
from etalii_dllm.grammar import Grammar, TokenConstraint
from etalii_dllm.models import BigramModel
from etalii_dllm.sampling import GREEDY, SamplingOptions
from etalii_dllm.tools import (
    DEEPSEEK,
    DEEPSEEK_CALL_BEGIN,
    DEEPSEEK_CALL_END,
    DEEPSEEK_CALLS_BEGIN,
    DEEPSEEK_CALLS_END,
    DEEPSEEK_SEP,
    FORMATS,
    HERMES,
    LLAMA3,
    PYTHONIC,
    XML,
    Tool,
    ToolChoice,
    auto_grammar,
    detect_format,
    forced_grammar,
    parse_calls,
)

SEARCH = Tool(
    "search",
    "Search the documents",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 9},
            "exact": {"type": "boolean"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "lang": {"type": "string", "enum": ["en", "nl"]},
        },
        "required": ["query"],
    },
)
ALL = [*TOOLS, SEARCH]

# Templates modelled on each family's own.
XML_TEMPLATE = (
    "{% if tools %}<|im_start|>system\n# Tools\n<tools>{% for t in tools %}{{ '\\n' }}{{ t | tojson }}{% endfor %}"
    "{{ '\\n' }}</tools>{{ '\\n' }}If you choose to call a function ONLY reply in the following format:{{ '\\n' }}"
    "<tool_call>{{ '\\n' }}<function=example_function_name>{{ '\\n' }}<parameter=example_parameter_1>{{ '\\n' }}"
    "value_1{{ '\\n' }}</parameter>{{ '\\n' }}</function>{{ '\\n' }}</tool_call><|im_end|>{{ '\\n' }}{% endif %}"
    "{% for m in messages %}<|im_start|>{{ m.role }}{{ '\\n' }}{{ m.content }}"
    "{% for c in m.tool_calls or [] %}<tool_call>{{ '\\n' }}<function={{ c.function.name }}>{{ '\\n' }}"
    "{% for k, v in c.function.arguments.items() %}<parameter={{ k }}>{{ '\\n' }}"
    "{% if v is string %}{{ v }}{% else %}{{ v | tojson }}{% endif %}{{ '\\n' }}</parameter>{{ '\\n' }}{% endfor %}"
    "</function>{{ '\\n' }}</tool_call>{% endfor %}<|im_end|>{{ '\\n' }}{% endfor %}<|im_start|>assistant{{ '\\n' }}"
)
DEEPSEEK_TEMPLATE = (
    "{% if tools %}<\uff5cSystem\uff5c>Tools:{% for t in tools %}{{ '\\n' }}{{ t | tojson }}{% endfor %}{% endif %}"
    "{% for m in messages %}{% if m.role == 'user' %}<\uff5cUser\uff5c>{{ m.content }}"
    "{% elif m.tool_calls %}<\uff5cAssistant\uff5c>{{ m.content }}" + DEEPSEEK_CALLS_BEGIN +
    "{% for c in m.tool_calls %}" + DEEPSEEK_CALL_BEGIN + "function" + DEEPSEEK_SEP +
    "{{ c.function.name + '\\n' + '```json' + '\\n' + c.function.arguments + '\\n' + '```' }}" + DEEPSEEK_CALL_END +
    "{% if not loop.last %}{{ '\\n' }}{% endif %}{% endfor %}" + DEEPSEEK_CALLS_END +
    "{% elif m.role == 'tool' %}<\uff5ctool\u2581output\uff5c>{{ m.content }}"
    "{% else %}{{ m.content }}{% endif %}{% endfor %}<\uff5cAssistant\uff5c>"
)  # fmt: skip
PYTHONIC_TEMPLATE = (
    "{% if tools %}<|start_header_id|>system<|end_header_id|>{{ '\\n\\n' }}If you decide to invoke any of the "
    "function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, "
    "params_name2=params_value2...), func_name2(params)]{% for t in tools %}{{ '\\n' }}{{ t | tojson }}{% endfor %}"
    "<|eot_id|>{% endif %}{% for m in messages %}<|start_header_id|>{{ m.role }}<|end_header_id|>{{ '\\n\\n' }}"
    "{% if m.tool_calls %}[{% for c in m.tool_calls %}{{ c.function.name }}("
    "{% for k, v in c.function.arguments.items() %}{{ k }}={{ v | tojson }}{% if not loop.last %}, {% endif %}"
    "{% endfor %}){% if not loop.last %}, {% endif %}{% endfor %}]{% else %}{{ m.content }}{% endif %}<|eot_id|>"
    "{% endfor %}<|start_header_id|>assistant<|end_header_id|>{{ '\\n\\n' }}"
)
TEMPLATES = {"xml": XML_TEMPLATE, "deepseek": DEEPSEEK_TEMPLATE, "pythonic": PYTHONIC_TEMPLATE}
MARKERS = ["</s>", DEEPSEEK_CALLS_BEGIN, DEEPSEEK_CALLS_END, DEEPSEEK_CALL_BEGIN, DEEPSEEK_CALL_END, DEEPSEEK_SEP]


def tokenizer() -> BpeTokenizer:
    """A byte-level tokenizer whose DeepSeek markers are special tokens, as in DeepSeek's vocabulary."""
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
    model = BigramModel(tokens.vocabulary_size, 52)
    return DllmEngine(model, tokens, "fp_more_tool_formats", chat_template=ChatTemplate(TEMPLATES[name]))


def request(choice: ToolChoice, tools=TOOLS, **options) -> ChatRequest:
    messages = [ChatMessage("user", "Weather in Rome?")]
    return ChatRequest(messages, 800, tools=tools, tool_choice=choice, request_id="req", **options)


# -- detection and markers ----------------------------------------------------------------------------------------


def test_formats_are_detected_from_the_template():
    for name, source in TEMPLATES.items():
        assert detect_format(source) is FORMATS[name]
        assert engine(name).tool_format is FORMATS[name]
    # Qwen3-Coder's template also writes <tool_call>: its <function= decides.
    assert detect_format("{{ tools }}<tool_call>") is HERMES
    assert detect_format("{{ tools }}<tool_call><function=") is XML
    assert detect_format("{{ tools }} ipython parameters func_name1(") is PYTHONIC
    assert detect_format("{{ tools }} ipython parameters") is LLAMA3


def test_deepseek_markers_become_visible():
    chat = engine("deepseek")
    for token in range(257, 262):
        assert chat.tokenizer.decode([token]) == MARKERS[token - 256]
    chat.tool_format = "hermes"
    assert chat.tokenizer.decode([257]) == ""


# -- grammar building blocks --------------------------------------------------------------------------------------


def test_grammar_parts():
    ab = Grammar.one_of([Grammar.literal("a"), Grammar.either([Grammar.literal("b"), Grammar.literal("bb")])])
    many = Grammar.sequence([Grammar.repeat(ab, Grammar.literal(",")), Grammar.optional(Grammar.literal("!"))])
    matcher = many.matcher()
    for text, expected in [("a", True), ("a,bb,b!", True), ("", False), ("a,", False), ("ab", False)]:
        assert matcher.matches(text.encode()) == expected, text
    loose = Grammar.repeat(Grammar.either([Grammar.literal("x"), Grammar.literal("y")])).matcher()
    assert loose.matches(b"xyx") and not loose.matches(b"")


# -- constraints --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_required_calls_are_valid_in_each_format(name):
    chat = engine(name)
    result = chat.chat_completion(request(ToolChoice("named", "get_weather")))
    assert result.finish_reason == "tool_calls", result
    (call,) = result.tool_calls
    assert call.name == "get_weather" and result.content == ""
    jsonschema.validate(json.loads(call.arguments), WEATHER.parameters)
    stream = list(chat.chat_stream(request(ToolChoice("named", "get_weather"))))
    assert [e.call for e in stream if isinstance(e, ToolCallEvent)] == [call]
    assert not any(e.text for e in stream if isinstance(e, TextDelta))


def test_auto_calls_start_at_each_marker():
    assert auto_grammar(TOOLS, XML)[1] == "<tool_call>"
    assert auto_grammar(TOOLS, DEEPSEEK)[1] == DEEPSEEK_CALLS_BEGIN
    grammar, trigger = auto_grammar(TOOLS, PYTHONIC)
    assert trigger is None
    trie = engine("pythonic")._token_trie()
    text = TokenConstraint(grammar, trie)
    text.accept_bytes(b"Sure, [here] it is")
    assert text.may_stop
    call = TokenConstraint(grammar, trie)
    call.accept_bytes(b'[get_weather(city="Rome", unit="C")]')
    assert call.finished
    assert not TokenConstraint(grammar, trie).allows_bytes(b"[rm(")


def test_xml_values():
    matcher = forced_grammar([SEARCH], ToolChoice("required"), XML).matcher()
    good = (
        "<tool_call>\n<function=search>\n<parameter=query>\nrain in\nRome\n</parameter>\n<parameter=limit>\n3\n"
        "</parameter>\n<parameter=lang>\nnl\n</parameter>\n</function>\n</tool_call>"
    )
    assert matcher.matches(good.encode())
    assert not matcher.matches(good.replace("<parameter=limit>\n3", "<parameter=limit>\nx").encode())
    assert not matcher.matches(good.replace("nl\n", "de\n").encode())
    no_query = "<tool_call>\n<function=search>\n<parameter=limit>\n3\n</parameter>\n</function>\n</tool_call>"
    assert not matcher.matches(no_query.encode())


def test_python_values():
    matcher = forced_grammar([SEARCH, NOON], ToolChoice("required"), PYTHONIC).matcher()
    for text, expected in [
        ('[search(query="a", exact=True)]', True),
        ('[search(query="a", tags=["x", "y"], lang="en"), is_noon()]', True),
        ('[search(query="a", exact=true)]', False),
        ('[search(limit=2, query="a")]', False),
        ("[is_noon(), is_noon()]", True),
        ("[]", False),
    ]:
        assert matcher.matches(text.encode()) == expected, text


# -- parsing ------------------------------------------------------------------------------------------------------


def test_parse_xml_calls():
    text = (
        'Looking.<tool_call>\n<function=search>\n<parameter=query>\n "quoted" text\n</parameter>\n<parameter=limit>\n'
        "3\n</parameter>\n<parameter=tags>\nnot json\n</parameter>\n</function>\n</tool_call> Done."
    )
    content, calls = parse_calls(text, ALL, XML)
    assert content == "Looking. Done."
    assert calls == [("search", json.dumps({"query": ' "quoted" text', "limit": 3, "tags": "not json"}))]
    unknown = "<tool_call>\n<function=rm>\n</function>\n</tool_call>"
    assert parse_calls(unknown, ALL, XML) == (unknown, [])
    assert parse_calls("<tool_call>junk</tool_call>", ALL, XML) == ("<tool_call>junk</tool_call>", [])


def test_parse_deepseek_calls():
    def call(name: str, arguments: str) -> str:
        return f"{DEEPSEEK_CALL_BEGIN}function{DEEPSEEK_SEP}{name}\n```json\n{arguments}\n```{DEEPSEEK_CALL_END}"

    weather = call("get_weather", '{"city": "Rome"}')
    text = f"Checking.{DEEPSEEK_CALLS_BEGIN}{weather}\n{call('is_noon', '{}')}"
    expected = ("Checking.", [("get_weather", '{"city": "Rome"}'), ("is_noon", "{}")])
    assert parse_calls(text + DEEPSEEK_CALLS_END, ALL, DEEPSEEK) == expected
    assert parse_calls(text, ALL, DEEPSEEK) == expected  # cut off before the end marker
    for broken in (call("rm", "{}"), call("is_noon", "{"), call("is_noon", "{}") + " and more", ""):
        reply = DEEPSEEK_CALLS_BEGIN + broken + DEEPSEEK_CALLS_END
        assert parse_calls(reply, ALL, DEEPSEEK) == (reply, [])
    assert parse_calls(' {"name": "is_noon", "arguments": {}}', ALL, DEEPSEEK) == ("", [("is_noon", "{}")])


def test_parse_python_calls():
    text = '[search(query="a\\/b", limit=-2, exact=False, tags=["x"], lang=None), is_noon()]'
    content, calls = parse_calls(text, ALL, PYTHONIC)
    assert content == ""
    expected = {"query": "a/b", "limit": -2, "exact": False, "tags": ["x"], "lang": None}
    assert calls == [("search", json.dumps(expected)), ("is_noon", "{}")]
    nested = "[search(query='single', tags={\"k\": [true, null, 1.5]})]"
    assert json.loads(parse_calls(nested, ALL, PYTHONIC)[1][0][1]) == {
        "query": "single",
        "tags": {"k": [True, None, 1.5]},
    }
    for broken in (
        "[rm()]",
        "[is_noon(1)]",
        "[is_noon(**x)]",
        "[is_noon(x=y)]",
        "[is_noon(x=-True)]",
        "[is_noon(x={1: 2})]",
        "[is_noon(x={**y})]",
        "[x.y()]",
        "[1]",
        "[]",
        "[is_noon(",
        "[search(query=f(x))]",
    ):
        assert parse_calls(broken, ALL, PYTHONIC) == (broken, []), broken
    assert parse_calls("Hello [there]", ALL, PYTHONIC) == ("Hello [there]", [])


def test_answer_prefix_holds_back_python_calls():
    assert _answer_prefix(" [get_weather(", "", "[") == ""
    assert _answer_prefix("Hi there [", "", "[") == "Hi there ["
    assert _answer_prefix(" [x", "", "{") == "[x"
    assert PYTHONIC.bare == "[" and LLAMA3.bare == "{" and XML.bare == ""


# -- history, round trips and the golden value --------------------------------------------------------------------


def test_history_renders_in_each_format():
    call = ToolCall("call_0123456789abcdef01234567", "search", '{"query": "rain", "limit": 2}')
    history = [
        ChatMessage("user", "Rain?"),
        ChatMessage("assistant", "", (call,)),
        ChatMessage("tool", "wet", tool_call_id=call.id, name="search"),
    ]
    prompts = {name: engine(name).render_chat(history, ALL) for name in TEMPLATES}
    xml = "<function=search>\n<parameter=query>\nrain\n</parameter>\n<parameter=limit>\n2\n</parameter>\n</function>"
    assert xml in prompts["xml"]
    assert f'function{DEEPSEEK_SEP}search\n```json\n{{"query": "rain", "limit": 2}}\n```' in prompts["deepseek"]
    assert '[search(query="rain", limit=2)]<|eot_id|>' in prompts["pythonic"]
    assert 257 in engine("deepseek").tokenizer.encode(prompts["deepseek"])


def test_calls_round_trip_through_the_conversation():
    for name in TEMPLATES:
        chat = engine(name)
        first = chat.chat_completion(request(ToolChoice("named", "get_weather")))
        messages = [
            ChatMessage("user", "Weather in Rome?"),
            ChatMessage("assistant", first.content, first.tool_calls),
            ChatMessage("tool", "sunny", tool_call_id=first.tool_calls[0].id, name="get_weather"),
        ]
        follow = ChatRequest(messages, 8, tools=TOOLS, tool_choice=ToolChoice("none"), request_id="req2")
        assert chat.chat_completion(follow).finish_reason in ("stop", "length")


def test_golden_more_tool_format_calls():
    calls = []
    for name in sorted(TEMPLATES):
        chat = engine(name)
        for choice in (ToolChoice("required"), ToolChoice("named", "get_weather")):
            result = chat.chat_completion(request(choice, options=SamplingOptions(temperature=0.7, seed=3)))
            calls.append(
                [name, result.finish_reason, result.content, [[c.id, c.name, c.arguments] for c in result.tool_calls]]
            )
    digest = hashlib.sha256(json.dumps(calls, sort_keys=True).encode()).hexdigest()
    assert digest == MORE_TOOL_FORMAT_CALLS, json.dumps(calls)


def test_greedy_runs_repeat():
    for name in TEMPLATES:
        a = engine(name).chat_completion(request(ToolChoice("required"), options=GREEDY))
        b = engine(name).chat_completion(request(ToolChoice("required"), options=GREEDY))
        assert a == b and a.tool_calls
