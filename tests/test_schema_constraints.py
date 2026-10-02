"""Phase 31: exact JSON Schema constraints. String patterns, formats and lengths and integer bounds compile to byte
automata that accept exactly the valid JSON texts, never lead decoding into a dead end, and keep every answer valid
under the whole schema."""

from __future__ import annotations

import json
import re

import pytest
from golden_values import SCHEMA_FINGERPRINTS
from test_engine_import import model_path  # noqa: F401 - fixture

from etalii_dllm.chat import ChatMessage
from etalii_dllm.engine import ChatRequest, DllmEngine, ResponseFormat
from etalii_dllm.grammar import (
    FORMATS,
    Grammar,
    GrammarError,
    _bounds,
    _searched,
    _top_level_alternation,
    integer_automaton,
    string_automaton,
)
from etalii_dllm.regexp import Counted, compile_regex, intersect
from etalii_dllm.sampling import SamplingOptions

# Integers


@pytest.mark.parametrize(
    ("low", "high"),
    [(0, 0), (0, 9), (3, 47), (-5, 5), (-123, -7), (None, -3), (-20, None), (7, None), (0, None), (99, 1000),
     (-1000, -999), (None, 0), (1, 1), (10, 99), (101, 899), (-1, -1)],
)  # fmt: skip
def test_integer_automata_accept_exactly_the_range(low, high):
    automaton = integer_automaton(low, high)
    for n in range(-1300, 1300):
        assert automaton.matches(str(n).encode()) == ((low is None or n >= low) and (high is None or n <= high))
    for text in (b"-0", b"007", b"+3", b"1.0", b"1e2", b""):
        assert not automaton.matches(text)
    if high is None:
        assert automaton.matches(b"123456789012345678901234567890")


def test_bounds_follow_both_drafts():
    assert _bounds({"minimum": 2, "maximum": 9}) == (2, 9)
    assert _bounds({"minimum": 2.5, "maximum": 9.5}) == (3, 9)
    assert _bounds({"exclusiveMinimum": 2, "exclusiveMaximum": 9}) == (3, 8)
    assert _bounds({"exclusiveMinimum": 2.5, "exclusiveMaximum": 8.5}) == (3, 8)
    assert _bounds({"minimum": 2, "exclusiveMinimum": True, "maximum": 9, "exclusiveMaximum": True}) == (3, 8)
    assert _bounds({"minimum": 2.5, "exclusiveMinimum": True}) == (3, None)  # not on the bound: nothing to exclude
    assert _bounds({"minimum": 0, "exclusiveMinimum": 4}) == (5, None)
    assert _bounds({"maximum": 10, "exclusiveMaximum": 4}) == (None, 3)
    for bad in ({"minimum": "1"}, {"maximum": float("inf")}, {"minimum": True}):
        with pytest.raises(GrammarError, match="finite number"):
            _bounds(bad)
    with pytest.raises(GrammarError, match="no integer"):
        Grammar.json_schema({"type": "integer", "minimum": 5, "maximum": 4})


# Strings


def test_patterns_have_json_schema_semantics():
    assert _searched("abc") == r"[\s\S]*(?:abc)[\s\S]*"
    assert _searched("^a+$") == "(?:a+)"
    assert _searched("^a") == r"(?:a)[\s\S]*" and _searched("a$") == r"[\s\S]*(?:a)"
    assert _searched("a\\$") == r"[\s\S]*(?:a\$)[\s\S]*" and _searched("a\\\\$") == r"[\s\S]*(?:a\\)"
    assert _top_level_alternation("a|b") and not _top_level_alternation("(a|b)")
    assert not _top_level_alternation("[|]") and not _top_level_alternation("a\\|b")
    assert not _top_level_alternation("[]|]x")
    with pytest.raises(GrammarError, match="ambiguous"):
        _searched("^a|b")
    assert _searched("a|b") == r"[\s\S]*(?:a|b)[\s\S]*"
    found = string_automaton("[0-9]{3}", None, 0, None)
    assert found.matches(b"ab123cd") and not found.matches(b"ab12cd")
    assert not found.matches(b'1234"') and not found.matches(b"123\\") and not found.matches(b"12\n3")


@pytest.mark.parametrize(
    ("name", "good", "bad"),
    [
        ("date", ["2026-10-02", "1999-12-31"], ["2026-13-02", "2026-1-02", "2026-10-32", "26-10-02"]),
        ("time", ["12:00:00Z", "23:59:60.5+01:00", "00:00:00-12:30"], ["24:00:00Z", "12:00:00", "12:60:00Z"]),
        ("date-time", ["2026-10-02T12:00:00Z", "2026-10-02t08:30:15.25-05:00"], ["2026-10-02 12:00:00Z"]),
        ("uuid", ["123e4567-e89b-12d3-a456-426614174000"],
         ["123e4567e89b12d3a456426614174000", "g23e4567-e89b-12d3-a456-426614174000"]),
        ("ipv4", ["10.0.255.1", "0.0.0.0"], ["256.1.1.1", "1.2.3", "01.2.3.4"]),
    ],
)  # fmt: skip
def test_formats(name, good, bad):
    automaton = string_automaton(None, name, 0, None)
    assert all(automaton.matches(text.encode()) for text in good)
    assert not any(automaton.matches(text.encode()) for text in bad)
    assert set(FORMATS) == {"date", "time", "date-time", "uuid", "ipv4"}


def test_lengths_count_code_points():
    automaton = string_automaton(None, None, 2, 3)
    assert isinstance(automaton, Counted)
    assert [automaton.matches(t.encode()) for t in ("a", "ab", "abc", "abcd")] == [False, True, True, False]
    assert automaton.matches("é中\U0001f600".encode()) and not automaton.matches("é".encode())
    unbounded = string_automaton("^x", None, 3, None)
    assert unbounded.matches(b"xyz") and unbounded.matches(b"x" * 500) and not unbounded.matches(b"xy")
    with pytest.raises(GrammarError, match="no text satisfies"):
        string_automaton("^(?:a|aaa)$", None, 2, 2)
    with pytest.raises(GrammarError, match="no text satisfies"):
        string_automaton("^a$", None, 3, None)
    with pytest.raises(GrammarError, match="'maxLength' is smaller"):
        Grammar.json_schema({"type": "string", "minLength": 3, "maxLength": 2})
    with pytest.raises(GrammarError, match="'pattern' must be a string"):
        Grammar.json_schema({"type": "string", "pattern": 3})


def _walk(automaton, choose):
    """Follows the bytes ``choose`` picks among the allowed ones until the automaton cannot read on."""
    state, data = 0, b""
    while automaton.reads(state):
        allowed = [b for b in range(256) if automaton.step(state, b) >= 0]
        assert allowed, "a state that reads must allow a byte"
        byte = choose(allowed, len(data))
        data += bytes([byte])
        state = automaton.step(state, byte)
        if automaton.accepting(state) and len(data) > 40:
            break
    assert automaton.accepting(state)
    return data


@pytest.mark.parametrize(
    "automaton",
    [
        string_automaton("^(?:a|aaa|aaaaa)$", None, 2, 4),
        string_automaton("[a-c]+z", None, 3, 6),
        string_automaton(None, "date", 0, None),
        string_automaton(None, None, 4, 4),
        integer_automaton(-57, 1234),
    ],
)
def test_no_dead_ends(automaton):
    for pick in (min, max, lambda allowed, n: allowed[(n * 7) % len(allowed)]):
        data = _walk(automaton, lambda allowed, n, pick=pick: pick(allowed) if pick in (min, max) else pick(allowed, n))
        assert automaton.matches(data)


def test_intersections_are_trimmed():
    both = intersect(compile_regex("[ab]*c"), compile_regex("a*[bc]"))
    assert both.matches(b"c") and both.matches(b"aac") and not both.matches(b"abc")
    assert all(both.reads(s) or both.accepting(s) for s in range(both.size))
    with pytest.raises(GrammarError, match="no text satisfies"):
        intersect(compile_regex("a+"), compile_regex("b+"))


# Schemas


SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "pattern": "^[A-Z]{3}$"},
        "day": {"type": "string", "format": "date"},
        "name": {"type": "string", "minLength": 2, "maxLength": 8},
        "age": {"type": "integer", "minimum": 0, "maximum": 120},
        "score": {"type": "integer", "exclusiveMinimum": -10, "exclusiveMaximum": 10},
        "tags": {"type": "array", "items": {"type": "string", "pattern": "^#"}, "maxItems": 2},
    },
    "required": ["code", "day", "name", "age", "score"],
}


def _valid(value) -> bool:
    return (
        re.fullmatch("[A-Z]{3}", value["code"]) is not None
        and re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])", value["day"]) is not None
        and 2 <= len(value["name"]) <= 8
        and 0 <= value["age"] <= 120
        and -10 < value["score"] < 10
        and all(t.startswith("#") for t in value.get("tags", []))
    )


def test_schemas_combine_the_constraints():
    matcher = Grammar.json_schema(SCHEMA).matcher()
    good = {"code": "ABC", "day": "2026-10-02", "name": "Ada", "age": 36, "score": -9, "tags": ["#x"]}
    assert matcher.matches(json.dumps(good).encode())
    for change in ({"code": "ABCD"}, {"day": "2026-02-30x"}, {"name": "A"}, {"age": 121}, {"score": 10},
                   {"tags": ["x"]}, {"name": 'a"b'}):  # fmt: skip
        assert not matcher.matches(json.dumps({**good, **change}).encode()), change
    assert Grammar.json_schema({"type": ["integer", "null"], "minimum": 1}).matcher().matches(b"null")
    assert Grammar.json_schema({"type": "string", "format": "email"}).matcher().matches(b'"anything"')  # annotation
    lenient = Grammar.json_schema(SCHEMA, lenient=True).matcher()
    assert lenient.matches(json.dumps({**good, "age": 500, "code": "x"}).encode())


@pytest.fixture(scope="module")
def tiny(model_path) -> DllmEngine:  # noqa: F811
    return DllmEngine.from_model_file(model_path)


def test_answers_are_valid_under_the_whole_schema(tiny):
    messages = [ChatMessage("user", "Describe a person as JSON.")]
    answers = []
    for seed in range(6):
        request = ChatRequest(messages, 1500, SamplingOptions(temperature=1.0, seed=seed),
                              response_format=ResponseFormat("json_schema", SCHEMA))  # fmt: skip
        result = tiny.chat_completion(request)
        assert result.finish_reason == "stop", result.content
        value = json.loads(result.content)
        assert _valid(value), value
        answers.append(result)
    assert answers[0].fingerprint == SCHEMA_FINGERPRINTS["person"]
    again = tiny.chat_completion(ChatRequest(messages, 1500, SamplingOptions(temperature=1.0, seed=0),
                                             response_format=ResponseFormat("json_schema", SCHEMA)))  # fmt: skip
    assert again.content == answers[0].content
