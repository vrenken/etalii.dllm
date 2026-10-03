"""Phase 48: tool calls in each model's own format, detected from its chat template (#302), constrained in that
format (#303), read back with ids the template accepts (#304), with a golden value (#305)."""

from __future__ import annotations

import hashlib
import json

import jsonschema
import pytest
from golden_values import TOOL_FORMAT_CALLS

from etalii_dllm.bpe import BpeTokenizer, bytes_to_unicode
from etalii_dllm.chat import ChatMessage, ToolCall
from etalii_dllm.chat_template import ChatTemplate
from etalii_dllm.engine import ChatRequest, DllmEngine, ResponseFormat, TextDelta, ToolCallEvent, _answer_prefix
from etalii_dllm.grammar import TokenConstraint
from etalii_dllm.models import BigramModel
from etalii_dllm.sampling import GREEDY, SamplingOptions
from etalii_dllm.tools import (
    FORMATS,
    GRANITE,
    HERMES,
    LLAMA3,
    MISTRAL,
    Tool,
    ToolChoice,
    auto_grammar,
    detect_format,
    parse_calls,
    template_id,
    tool_format,
)

WEATHER = Tool(
    "get_weather",
    "Current weather for a city",
    {
        "type": "object",
        "properties": {"city": {"enum": ["Paris", "Rome"]}, "unit": {"enum": ["C", "F"]}},
        "required": ["city", "unit"],
        "additionalProperties": False,
    },
)
NOON = Tool("is_noon", "Whether it is noon", {"type": "object", "properties": {}, "additionalProperties": False})
TOOLS = [WEATHER, NOON]

# Templates modelled on each family's own: how it presents tools, writes earlier calls and returns results.
LLAMA3_TEMPLATE = (
    "<|begin_of_text|>{% if tools %}<|start_header_id|>user<|end_header_id|>\n\nGiven the following functions, "
    "respond with a JSON for a function call. Respond in the format "
    '{"name": function name, "parameters": dictionary of argument name and its value}.'
    "{% for t in tools %}\n{{ t | tojson }}{% endfor %}<|eot_id|>{% endif %}"
    "{% for m in messages %}{% if m.role == 'tool' %}<|start_header_id|>ipython<|end_header_id|>\n\n"
    "{{ m.content }}<|eot_id|>{% elif m.tool_calls %}<|start_header_id|>assistant<|end_header_id|>\n\n"
    '{% for c in m.tool_calls %}{"name": "{{ c.function.name }}", "parameters": {{ c.function.arguments | tojson }}}'
    "{% endfor %}<|eot_id|>{% else %}<|start_header_id|>{{ m.role }}<|end_header_id|>\n\n{{ m.content }}<|eot_id|>"
    "{% endif %}{% endfor %}<|start_header_id|>assistant<|end_header_id|>\n\n"
)
MISTRAL_TEMPLATE = (
    "<s>{% if tools %}[AVAILABLE_TOOLS]{{ tools | tojson }}[/AVAILABLE_TOOLS]{% endif %}"
    "{% for m in messages %}{% if m.role == 'user' %}[INST]{{ m.content }}[/INST]"
    "{% elif m.tool_calls %}[TOOL_CALLS][{% for c in m.tool_calls %}"
    "{% if c.id is not defined or c.id | length != 9 %}"
    "{{ raise_exception('Tool call IDs should be alphanumeric strings with length 9!') }}{% endif %}"
    '{"name": "{{ c.function.name }}", "arguments": {{ c.function.arguments | tojson }}, "id": "{{ c.id }}"}'
    "{% if not loop.last %}, {% endif %}{% endfor %}]</s>"
    "{% elif m.role == 'tool' %}{% if m.tool_call_id | length != 9 %}"
    "{{ raise_exception('Tool call IDs should be alphanumeric strings with length 9!') }}{% endif %}"
    '[TOOL_RESULTS]{"content": {{ m.content | tojson }}, "call_id": "{{ m.tool_call_id }}"}[/TOOL_RESULTS]'
    "{% else %}{{ m.content }}</s>{% endif %}{% endfor %}"
)
GRANITE_TEMPLATE = (
    "{% if tools %}<|start_of_role|>available_tools<|end_of_role|>{% for t in tools %}\n{{ t | tojson }}{% endfor %}"
    "<|end_of_text|>\n{% endif %}{% for m in messages %}{% if m.tool_calls %}"
    "<|start_of_role|>assistant<|end_of_role|><|tool_call|>{{ m.tool_calls | map(attribute='function') | list | "
    "tojson }}<|end_of_text|>\n{% elif m.role == 'tool' %}<|start_of_role|>tool_response<|end_of_role|>"
    "{{ m.content }}<|end_of_text|>\n{% else %}<|start_of_role|>{{ m.role }}<|end_of_role|>{{ m.content }}"
    "<|end_of_text|>\n{% endif %}{% endfor %}<|start_of_role|>assistant<|end_of_role|>"
)
HERMES_TEMPLATE = (
    "{% if tools %}<|im_start|>system\n# Tools{% for t in tools %}\n{{ t | tojson }}{% endfor %}<|im_end|>\n{% endif %}"
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}"
    "{% for c in m.tool_calls or [] %}<tool_call>{{ c.function | tojson }}</tool_call>{% endfor %}<|im_end|>\n"
    "{% endfor %}<|im_start|>assistant\n"
)
TEMPLATES = {"hermes": HERMES_TEMPLATE, "llama3": LLAMA3_TEMPLATE, "mistral": MISTRAL_TEMPLATE,
             "granite": GRANITE_TEMPLATE}  # fmt: skip
MARKERS = ["</s>", "[TOOL_CALLS]", "<|tool_call|>"]


def tokenizer() -> BpeTokenizer:
    """A byte-level tokenizer whose tool call markers are special tokens, as in Mistral's and Granite's vocabularies
    (Qwen's ``<tool_call>`` is an ordinary added token)."""
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
    model = BigramModel(tokens.vocabulary_size, 48)
    return DllmEngine(model, tokens, "fp_tool_formats", chat_template=ChatTemplate(TEMPLATES[name]))


def request(choice: ToolChoice, **options) -> ChatRequest:
    messages = [ChatMessage("user", "Weather in Rome?")]
    return ChatRequest(messages, 800, tools=TOOLS, tool_choice=choice, request_id="req", **options)


# -- detection ---------------------------------------------------------------------------------------------------


def test_formats_are_detected_from_the_template():
    for name, source in TEMPLATES.items():
        assert detect_format(source).name == name
        assert engine(name).tool_format == FORMATS[name]
    assert detect_format(None) is HERMES and detect_format("{{ messages }}") is HERMES
    assert detect_format("{% if tools %}{{ tools }}{% endif %}") is HERMES  # tools but no known marker
    # Hermes markers win over Llama 3 hints (Hermes 3 is a Llama 3 fine-tune with an ipython-free template).
    assert detect_format(HERMES_TEMPLATE + "ipython parameters") is HERMES
    assert tool_format("mistral") is MISTRAL
    with pytest.raises(ValueError, match="unknown tool call format"):
        tool_format("phi")


def test_the_format_can_be_overridden_and_markers_become_visible():
    chat = engine("hermes")
    assert chat.tokenizer.decode([257]) == ""
    chat.tool_format = "mistral"
    assert chat.tokenizer.decode([257]) == "[TOOL_CALLS]" and chat.tokenizer.decode([256]) == ""
    assert chat.tokenizer.decode_bytes([258]) == b""
    chat.tool_format = GRANITE
    assert chat.tokenizer.decode_bytes([258]) == b"<|tool_call|>"
    assert chat.tokenizer.showing(["<|tool_call|>"]) is chat.tokenizer
    chat.chat_template = None
    assert chat.tool_format is HERMES and chat.tokenizer.decode([258]) == ""


# -- constraints -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_required_calls_are_valid_in_each_format(name):
    chat = engine(name)
    result = chat.chat_completion(request(ToolChoice("named", "get_weather")))
    assert result.finish_reason == "tool_calls", result
    (call,) = result.tool_calls
    assert call.name == "get_weather" and call.id.startswith("call_")
    jsonschema.validate(json.loads(call.arguments), WEATHER.parameters)
    fmt = FORMATS[name]
    assert result.content == ""
    stream = list(chat.chat_stream(request(ToolChoice("named", "get_weather"))))
    assert [e.call for e in stream if isinstance(e, ToolCallEvent)] == [call]
    assert not any(e.text for e in stream if isinstance(e, TextDelta))
    assert fmt.name == name


def test_calls_are_written_in_the_models_own_format():
    for name, start in [("hermes", "<tool_call>"), ("mistral", "[TOOL_CALLS]"), ("granite", "<|tool_call|>")]:
        chat = engine(name)
        assert auto_grammar(TOOLS, chat.tool_format)[1] == start
        constraint = chat._constraint(request(ToolChoice("required")), TOOLS)
        assert constraint is not None
        # The call starts with the marker: for Mistral and Granite its special token (or the marker's first byte).
        expected = {"hermes": [ord("<")], "mistral": [ord("["), 257], "granite": [ord("<"), 258]}
        assert constraint.allowed() == expected[name]


def test_llama3_auto_answers_in_text_or_calls_from_the_start():
    chat = engine("llama3")
    grammar, trigger = auto_grammar(TOOLS, LLAMA3)
    assert trigger is None
    trie = chat._token_trie()
    text = TokenConstraint(grammar, trie)
    text.accept_bytes(b"Sure, {here} it is")
    assert text.active and text.may_stop and not text.finished
    call = TokenConstraint(grammar, trie)
    call.accept_bytes(b' {"name": "is_noon", "parameters": {}}')
    assert call.finished
    wrong = TokenConstraint(grammar, trie)
    assert not wrong.allows_bytes(b'{"name": "rm"')
    assert not wrong.allows_bytes(b'{"name": "is_noon", "arguments"')


def test_response_format_or_call_in_each_format():
    schema = {"type": "object", "properties": {"answer": {"enum": ["yes", "no"]}}, "required": ["answer"]}
    for name in TEMPLATES:
        result = engine(name).chat_completion(
            request(ToolChoice(), response_format=ResponseFormat("json_schema", schema))
        )
        if result.tool_calls:
            assert result.finish_reason == "tool_calls"
        elif result.finish_reason == "stop":
            jsonschema.validate(json.loads(result.content), schema)


# -- parsing and ids ---------------------------------------------------------------------------------------------


def test_parse_calls_in_each_format():
    weather = ("get_weather", '{"city": "Rome", "unit": "C"}')
    noon = ("is_noon", "{}")
    mistral = 'Checking.[TOOL_CALLS][{"name": "get_weather", "arguments": {"city": "Rome", "unit": "C"}}, ' \
        '{"name": "is_noon", "arguments": {}}]'  # fmt: skip
    assert parse_calls(mistral, TOOLS, MISTRAL) == ("Checking.", [weather, noon])
    granite = '<|tool_call|> [{"name": "is_noon", "arguments": "{}"}]'
    assert parse_calls(granite, TOOLS, GRANITE) == ("", [noon])
    assert parse_calls('<|tool_call|>{"name": "is_noon", "arguments": {}}', TOOLS, GRANITE) == ("", [noon])
    assert parse_calls('{"name": "get_weather", "parameters": {"city": "Rome", "unit": "C"}}', TOOLS, LLAMA3) == (
        "",
        [weather],
    )
    assert parse_calls("Hello there", TOOLS, LLAMA3) == ("Hello there", [])
    # Unknown tools, broken JSON and trailing text keep the whole reply as text.
    for broken in ['[TOOL_CALLS][{"name": "rm", "arguments": {}}]', "[TOOL_CALLS][{", "[TOOL_CALLS][] x",
                   '[TOOL_CALLS][{"name": "is_noon", "arguments": {}}] and more']:  # fmt: skip
        assert parse_calls(broken, TOOLS, MISTRAL) == (broken.strip(), [])
    assert parse_calls("[TOOL_CALLS][]", TOOLS, MISTRAL) == ("[TOOL_CALLS][]", [])
    # A bare JSON call counts in every format.
    assert parse_calls(' {"name": "is_noon", "arguments": {}}', TOOLS, MISTRAL) == ("", [noon])


def test_answer_prefix_holds_back_each_marker():
    assert _answer_prefix("Hi [TOOL_CA", "[TOOL_CALLS]") == "Hi"
    assert _answer_prefix('Hi [TOOL_CALLS][{"name"', "[TOOL_CALLS]") == "Hi"
    assert _answer_prefix("Hi <|tool", "<|tool_call|>") == "Hi"
    assert _answer_prefix("Hi there ", "") == "Hi there"
    assert _answer_prefix(' {"name"', "") == ""


def test_template_ids():
    assert template_id("abcDEF123", 9) == "abcDEF123"
    derived = template_id("call_0123456789abcdef01234567", 9)
    assert len(derived) == 9 and derived.isalnum() and derived == template_id("call_0123456789abcdef01234567", 9)
    assert template_id("toolu_1", 9) != template_id("toolu_2", 9)


def test_history_renders_in_each_format():
    history = [
        ChatMessage("user", "Weather in Rome?"),
        ChatMessage("assistant", "", (ToolCall("call_0123456789abcdef01234567", "get_weather", '{"city": "Rome"}'),)),
        ChatMessage("tool", "sunny", tool_call_id="call_0123456789abcdef01234567", name="get_weather"),
    ]
    short = template_id("call_0123456789abcdef01234567", 9)
    prompts = {name: engine(name).render_chat(history, TOOLS) for name in TEMPLATES}
    assert f'"id": "{short}"' in prompts["mistral"] and f'"call_id": "{short}"' in prompts["mistral"]
    assert '[TOOL_CALLS][{"name": "get_weather", "arguments": {"city": "Rome"}' in prompts["mistral"]
    assert '<|tool_call|>[{"name": "get_weather", "arguments": {"city": "Rome"}}]' in prompts["granite"]
    assert '{"name": "get_weather", "parameters": {"city": "Rome"}}<|eot_id|>' in prompts["llama3"]
    assert "<|start_header_id|>ipython<|end_header_id|>\n\nsunny" in prompts["llama3"]
    assert '<tool_call>{"name": "get_weather", "arguments": {"city": "Rome"}}</tool_call>' in prompts["hermes"]
    # The special marker in the prompt is the marker token, as the model saw it in training.
    assert 257 in engine("mistral").tokenizer.encode(prompts["mistral"])


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


# -- golden value ------------------------------------------------------------------------------------------------


def test_golden_tool_format_calls():
    calls = []
    for name in sorted(TEMPLATES):
        chat = engine(name)
        for choice in (ToolChoice("required"), ToolChoice("named", "get_weather")):
            result = chat.chat_completion(request(choice, options=SamplingOptions(temperature=0.7, seed=3)))
            calls.append(
                [name, result.finish_reason, result.content, [[c.id, c.name, c.arguments] for c in result.tool_calls]]
            )
    digest = hashlib.sha256(json.dumps(calls, sort_keys=True).encode()).hexdigest()
    assert digest == TOOL_FORMAT_CALLS, json.dumps(calls)


def test_greedy_runs_repeat():
    for name in TEMPLATES:
        a = engine(name).chat_completion(request(ToolChoice("required"), options=GREEDY))
        b = engine(name).chat_completion(request(ToolChoice("required"), options=GREEDY))
        assert a == b
