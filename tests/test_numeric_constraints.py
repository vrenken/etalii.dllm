"""Phase 33: exact numeric and object constraints. Bounds on decimal numbers (#226) and ``multipleOf`` (#227) compile
to byte automata that compare in exact decimal arithmetic; property counts and typed map objects (#228) are checked
by the object automaton; answers stay valid under the whole schema (#229)."""

from __future__ import annotations

import json
import re
from decimal import Decimal, localcontext

import pytest
from golden_values import SCHEMA_FINGERPRINTS
from test_engine_import import model_path  # noqa: F401 - fixture

from etalii_dllm.chat import ChatMessage
from etalii_dllm.engine import ChatRequest, DllmEngine, ResponseFormat
from etalii_dllm.grammar import Grammar, GrammarError
from etalii_dllm.numeric_automata import _fraction, decimal_bounds, exact, number_automaton
from etalii_dllm.sampling import SamplingOptions

_NUMBER = re.compile(r"(?:0|[1-9]\d*)(?:\.\d+)?|-(?:0\.\d*[1-9]\d*|[1-9]\d*(?:\.\d+)?)")
_INTEGER = re.compile(r"0|-?[1-9]\d*")
TEXTS = sorted(
    {sign + whole + fraction
     for sign in ("", "-")
     for whole in ("0", "1", "2", "3", "9", "10", "12", "25", "99", "100", "250", "1000")
     for fraction in ("", ".0", ".5", ".25", ".49", ".5000", ".75", ".01", ".99", ".125", ".005", ".3")}
    | {"007", "1.", "-0", "+1", "1e3", ".5", "00", "-", ""}
)  # fmt: skip
D = Decimal


def _oracle(text: str, low, high, multiple, integer: bool) -> bool:
    if not (_INTEGER if integer else _NUMBER).fullmatch(text):
        return False
    value = D(text)
    if low is not None and not (value > low[0] if low[1] else value >= low[0]):
        return False
    if high is not None and not (value < high[0] if high[1] else value <= high[0]):
        return False
    with localcontext() as context:
        context.prec = 200
        return multiple is None or value % multiple == 0


CASES = [
    (None, None, None),
    ((D("2.5"), False), None, None),
    ((D("2.5"), True), (D("12"), False), None),
    ((D("-2.5"), False), (D("0"), True), None),
    (None, (D("-0.5"), True), None),
    ((D("0"), True), None, None),
    ((D("0"), False), (D("0"), False), None),
    (None, (D("0"), False), None),
    ((D("-1"), False), (D("1"), False), D("0.25")),
    (None, None, D("2.5")),
    ((D("-12.125"), True), (D("99.99"), False), D("0.005")),
    (None, None, D("3")),
    (None, None, D("10")),
    ((D("0.3"), False), (D("0.3"), False), None),
]


@pytest.mark.parametrize("integer", [False, True])
@pytest.mark.parametrize(("low", "high", "multiple"), CASES)
def test_number_automata_match_exact_decimal_arithmetic(low, high, multiple, integer):
    if (low, high, integer) == ((D("0.3"), False), (D("0.3"), False), True):
        with pytest.raises(GrammarError, match="no integer satisfies"):
            number_automaton(low, high, multiple, integer)
        return
    automaton = number_automaton(low, high, multiple, integer)
    for text in TEXTS:
        assert automaton.matches(text.encode()) == _oracle(text, low, high, multiple, integer), text


def test_fractions_compare_digit_strings():
    assert _fraction("", "ge") == (True, r"\d+") and _fraction("", "lt") == (False, None)
    assert _fraction("9", "gt") == (False, r"9(?:\d*[1-9]\d*)")
    assert _fraction("0", "le")[0]


def test_decimal_bounds():
    assert decimal_bounds({"minimum": 0.1, "maximum": 2}) == ((D("0.1"), False), (D(2), False))
    assert decimal_bounds({"minimum": 1, "exclusiveMinimum": True}) == ((D(1), True), None)
    assert decimal_bounds({"minimum": 1, "exclusiveMinimum": 1}) == ((D(1), True), None)
    assert decimal_bounds({"minimum": 2, "exclusiveMinimum": 1}) == ((D(2), False), None)
    assert decimal_bounds({"maximum": 2, "exclusiveMaximum": 3}) == (None, (D(2), False))
    assert decimal_bounds({"maximum": 5, "exclusiveMaximum": 3.5}) == (None, (D("3.5"), True))
    assert exact(1e-7, "minimum") == D("1e-7") and exact(10**30, "maximum") == D(10**30)
    with pytest.raises(GrammarError, match="finite number"):
        exact(float("nan"), "minimum")


def test_unsatisfiable_and_unsupported_numbers_are_refused():
    with pytest.raises(GrammarError, match="no integer satisfies"):
        Grammar.json_schema({"type": "integer", "minimum": 0.1, "maximum": 0.9, "multipleOf": 1})
    with pytest.raises(GrammarError, match="no number satisfies"):
        Grammar.json_schema({"type": "number", "exclusiveMinimum": 1, "exclusiveMaximum": 2, "multipleOf": 1})
    with pytest.raises(GrammarError, match="too fine"):
        Grammar.json_schema({"type": "number", "multipleOf": 12345.67})
    with pytest.raises(GrammarError, match="greater than 0"):
        Grammar.json_schema({"type": "integer", "multipleOf": -2})
    huge = Grammar.json_schema({"type": "number", "minimum": 1e30}).matcher()
    assert huge.matches(b"1000000000000000000000000000000.5") and not huge.matches(b"999999999999999999999999999999")


def _walk(automaton, choose) -> bytes:
    """Bytes ``choose`` picks for a while, then the shortest way to a match: no state may be a dead end."""
    state, data = 0, b""
    for _ in range(30):
        allowed = [b for b in range(256) if automaton.step(state, b) >= 0]
        assert allowed or automaton.accepting(state)
        if not allowed:
            return data
        byte = choose(allowed, len(data))
        data += bytes([byte])
        state = automaton.step(state, byte)
    queue, seen = [(state, data)], {state}
    for current, text in queue:
        if automaton.accepting(current):
            return text
        for byte in range(256):
            following = automaton.step(current, byte)
            if following >= 0 and following not in seen:
                seen.add(following)
                queue.append((following, text + bytes([byte])))
    raise AssertionError("no way to finish")


@pytest.mark.parametrize(("low", "high", "multiple"), CASES[1:])
def test_no_dead_ends(low, high, multiple):
    automaton = number_automaton(low, high, multiple, False)
    for pick in (min, max, lambda allowed, n: allowed[(n * 7) % len(allowed)]):
        data = _walk(automaton, lambda allowed, n, pick=pick: pick(allowed) if pick in (min, max) else pick(allowed, n))
        assert _oracle(data.decode(), low, high, multiple, False), data


# Objects


def test_property_counts_on_declared_properties():
    properties = {name: {"type": "integer"} for name in "abcd"}
    for minimum, maximum, required in [(2, 2, []), (0, 1, []), (3, None, ["d"]), (1, 3, ["b"]), (4, 4, [])]:
        schema = {"type": "object", "properties": properties, "required": required, "minProperties": minimum}
        if maximum is not None:
            schema["maxProperties"] = maximum
        matcher = Grammar.json_schema(schema).matcher()
        for size in range(5):
            for start in range(4):
                names = ["abcd"[(start + i) % 4] for i in range(size)]
                if names != sorted(names):
                    continue  # properties come in schema order
                text = json.dumps(dict.fromkeys(names, 1)).encode()
                valid = minimum <= size <= (maximum if maximum is not None else 4) and set(required) <= set(names)
                assert matcher.matches(text) == valid, (schema, text)


def test_map_objects_and_free_counts():
    matcher = Grammar.json_schema(
        {
            "type": "object",
            "additionalProperties": {"type": "integer", "minimum": 0},
            "minProperties": 1,
            "maxProperties": 2,
        }
    ).matcher()
    for text, valid in [("{}", False), ('{"x": 1}', True), ('{"x": -1}', False), ('{"x": 1, "y": 2}', True),
                        ('{"x": 1, "y": 2, "z": 3}', False), ('{"x": "a"}', False)]:  # fmt: skip
        assert matcher.matches(text.encode()) == valid, text
    capped = Grammar.json_schema({"type": "object", "maxProperties": 1}).matcher()
    assert capped.matches(b'{"a": [1, {"b": 2}]}') and not capped.matches(b'{"a": 1, "b": 2}')
    lenient = Grammar.json_schema({"type": "object", "additionalProperties": {"type": "integer"}}, lenient=True)
    assert lenient.matcher().matches(b'{"a": "text"}')
    refusals = [
        ({"type": "object", "minProperties": 2}, "needs declared 'properties'"),
        ({"type": "object", "additionalProperties": False, "minProperties": 1}, "no object without properties"),
        ({"type": "object", "properties": {"a": {}}, "minProperties": 2}, "no object of the declared properties"),
        ({"type": "object", "properties": {"a": {}, "b": {}}, "required": ["a", "b"], "maxProperties": 1},
         "no object of the declared properties"),
        ({"type": "object", "minProperties": 3, "maxProperties": 2}, "smaller than 'minProperties'"),
        ({"type": "object", "maxProperties": -1}, "non-negative integer"),
        ({"type": "object", "minProperties": True}, "non-negative integer"),
    ]  # fmt: skip
    for schema, message in refusals:
        with pytest.raises(GrammarError, match=message):
            Grammar.json_schema(schema)


# Answers

SCHEMA = {
    "type": "object",
    "properties": {
        "temperature": {"type": "number", "minimum": -40, "exclusiveMaximum": 60.5},
        "price": {"type": "number", "minimum": 0, "maximum": 1000, "multipleOf": 0.01},
        "even": {"type": "integer", "multipleOf": 2, "minimum": -10, "maximum": 10},
        "scores": {"type": "object", "additionalProperties": {"type": "number", "minimum": 0, "maximum": 1},
                   "maxProperties": 3},
        "extra": {"type": "boolean"},
        "note": {"type": "string", "maxLength": 12},
    },
    "required": ["temperature", "price", "even", "scores"],
    "minProperties": 5,
}  # fmt: skip


def _valid(value) -> bool:
    def number(x) -> Decimal:
        assert not isinstance(x, bool)
        return Decimal(repr(x)) if isinstance(x, float) else Decimal(x)

    scores = value["scores"]
    return (
        len(value) >= 5
        and -40 <= number(value["temperature"]) < Decimal("60.5")
        and 0 <= number(value["price"]) <= 1000
        and number(value["price"]) % Decimal("0.01") == 0
        and isinstance(value["even"], int)
        and value["even"] % 2 == 0
        and -10 <= value["even"] <= 10
        and len(scores) <= 3
        and all(0 <= number(v) <= 1 for v in scores.values())
    )


@pytest.fixture(scope="module")
def tiny(model_path) -> DllmEngine:  # noqa: F811
    return DllmEngine.from_model_file(model_path)


def test_answers_are_valid_under_the_whole_schema(tiny):
    messages = [ChatMessage("user", "Report a measurement as JSON.")]
    answers = []
    for seed in range(6):
        request = ChatRequest(messages, 1500, SamplingOptions(temperature=1.0, seed=seed),
                              response_format=ResponseFormat("json_schema", SCHEMA))  # fmt: skip
        result = tiny.chat_completion(request)
        assert result.finish_reason == "stop", result.content
        value = json.loads(result.content, parse_float=Decimal)
        assert _valid(value), value
        answers.append(result)
    assert answers[0].fingerprint == SCHEMA_FINGERPRINTS["measurement"]
