"""Phase 34: exact tuples, key names and unique items. ``prefixItems`` (#231), ``propertyNames`` (#232) and
``uniqueItems`` over finite item sets (#233) are enforced by the automaton, and answers stay valid under the whole
schema (#234)."""

from __future__ import annotations

import itertools
import json
import re

import pytest
from golden_values import SCHEMA_FINGERPRINTS
from test_engine_import import model_path  # noqa: F401 - fixture

from etalii_dllm.chat import ChatMessage
from etalii_dllm.engine import ChatRequest, DllmEngine, ResponseFormat
from etalii_dllm.grammar import Grammar, GrammarError, _choices, _identity, _Literals, _Union
from etalii_dllm.sampling import SamplingOptions


def _matches(schema, value) -> bool:
    return Grammar.json_schema(schema).matcher().matches(json.dumps(value).encode())


# Tuples


def test_prefix_items():
    schema = {"type": "array", "prefixItems": [{"type": "string"}, {"type": "integer"}], "items": False}
    assert _matches(schema, []) and _matches(schema, ["a"]) and _matches(schema, ["a", 1])
    assert not _matches(schema, ["a", 1, 2]) and not _matches(schema, [1]) and not _matches(schema, ["a", "b"])
    open_tuple = {**schema, "items": {"type": "boolean"}, "minItems": 2, "maxItems": 3}
    assert _matches(open_tuple, ["a", 1, True]) and not _matches(open_tuple, ["a"])
    assert not _matches(open_tuple, ["a", 1, True, False]) and not _matches(open_tuple, ["a", 1, 2])
    old = {"type": "array", "items": [{"const": "x"}], "additionalItems": {"type": "null"}}
    assert _matches(old, ["x", None, None]) and not _matches(old, ["y"]) and not _matches(old, ["x", 1])
    assert _matches({"type": "array", "items": [{"const": 1}]}, [1, "anything", {}])
    with pytest.raises(GrammarError, match="no array has 'minItems'"):
        Grammar.json_schema({**schema, "minItems": 3})
    with pytest.raises(GrammarError, match="must be a list"):
        Grammar.json_schema({"type": "array", "prefixItems": {"type": "string"}})


# Property names


def test_property_names():
    schema = {"type": "object", "propertyNames": {"pattern": "^[a-z]+$", "maxLength": 3},
              "additionalProperties": {"type": "integer"}}  # fmt: skip
    for value, valid in [({}, True), ({"ab": 1}, True), ({"abcd": 1}, False), ({"Ab": 1}, False),
                         ({"ab": "x"}, False), ({"ab": 1, "cd": 2}, True)]:  # fmt: skip
        assert _matches(schema, value) == valid, value
    names = {"type": "object", "propertyNames": {"enum": ["red", "green"]}}
    assert _matches(names, {"red": [1]}) and not _matches(names, {"blue": 1})
    declared = {"type": "object", "properties": {"ok": {"type": "integer"}, "BAD": {"type": "integer"}},
                "propertyNames": {"pattern": "^[a-z]+$"}}  # fmt: skip
    assert _matches(declared, {"ok": 1}) and not _matches(declared, {"ok": 1, "BAD": 2})
    emptied = {"type": "object", "properties": {"BAD": {}}, "propertyNames": {"const": "ok"}}
    assert _matches(emptied, {}) and not _matches(emptied, {"ok": 1})
    lenient = Grammar.json_schema({"type": "object", "propertyNames": {"const": "a"}}, lenient=True).matcher()
    assert lenient.matches(b'{"b": 1}')
    refusals = [
        ({"type": "object", "properties": {"BAD": {}}, "required": ["BAD"], "propertyNames": {"enum": ["ok"]}},
         "required properties break 'propertyNames': BAD"),
        ({"type": "object", "propertyNames": {"type": "integer"}}, "must describe strings"),
        ({"type": "object", "propertyNames": {"enum": ["a", 1]}}, "must describe strings"),
    ]  # fmt: skip
    for schema, message in refusals:
        with pytest.raises(GrammarError, match=re.escape(message)):
            Grammar.json_schema(schema)


# Unique items


def test_identity_compares_json_values():
    assert _identity(1) == _identity(1.0) and _identity(1) != _identity(True) and _identity(0) != _identity(False)
    assert _identity(None) != _identity("null") and _identity([1, {"a": 2.50}]) == _identity([1.0, {"a": 2.5}])
    assert _choices(_Union([_Literals([b"1", b"true"]), _Literals([b"1.0"])])) == [(b"1", 0), (b"true", 1), (b"1.0", 0)]


def test_unique_items():
    schema = {"type": "array", "items": {"enum": ["a", "b", 1, 1.0, True, None]}, "uniqueItems": True}
    matcher = Grammar.json_schema(schema).matcher()
    values = ["a", "b", 1, 1.0, True, None]
    for size in range(5):
        for chosen in itertools.permutations(values, size):
            distinct = len({json.dumps(v if not isinstance(v, float) else int(v)) for v in chosen}) == size
            assert matcher.matches(json.dumps(list(chosen)).encode()) == distinct, chosen
    capped = {"type": "array", "items": {"type": "boolean"}, "uniqueItems": True, "minItems": 2}
    assert _matches(capped, [True, False]) and not _matches(capped, [True, True, False])
    assert _matches({"type": "array", "items": {"anyOf": [{"const": "x"}, {"type": "null"}]}, "uniqueItems": True},
                    [None, "x"])  # fmt: skip
    assert _matches({"type": "array", "items": {"type": "string"}, "uniqueItems": False}, ["a", "a"])
    assert Grammar.json_schema({"type": "array", "uniqueItems": True}, lenient=True).matcher().matches(b"[1, 1]")
    with pytest.raises(GrammarError, match="finite set"):
        Grammar.json_schema({"type": "array", "items": {"type": "string"}, "uniqueItems": True})
    with pytest.raises(GrammarError, match="finite set"):
        Grammar.json_schema(
            {"type": "array", "prefixItems": [{"const": 1}], "items": {"const": 2}, "uniqueItems": True}
        )
    with pytest.raises(GrammarError, match="no array has 'minItems'"):
        Grammar.json_schema({**capped, "minItems": 3})


# Answers

SCHEMA = {
    "type": "object",
    "properties": {
        "point": {"type": "array", "prefixItems": [{"type": "integer", "minimum": 0, "maximum": 99},
                                                   {"type": "integer", "minimum": 0, "maximum": 99}], "items": False,
                  "minItems": 2},
        "tags": {"type": "array", "items": {"enum": ["red", "green", "blue"]}, "uniqueItems": True, "minItems": 1},
        "labels": {"type": "object", "propertyNames": {"pattern": "^[a-z]{1,5}$"},
                   "additionalProperties": {"type": "boolean"}, "maxProperties": 2},
    },
    "required": ["point", "tags", "labels"],
}  # fmt: skip


def _valid(value) -> bool:
    point, tags, labels = value["point"], value["tags"], value["labels"]
    return (
        len(point) == 2
        and all(isinstance(p, int) and 0 <= p <= 99 for p in point)
        and 1 <= len(tags) == len(set(tags))
        and set(tags) <= {"red", "green", "blue"}
        and len(labels) <= 2
        and all(re.fullmatch("[a-z]{1,5}", k) and isinstance(v, bool) for k, v in labels.items())
    )


@pytest.fixture(scope="module")
def tiny(model_path) -> DllmEngine:  # noqa: F811
    return DllmEngine.from_model_file(model_path)


def test_answers_are_valid_under_the_whole_schema(tiny):
    messages = [ChatMessage("user", "Describe a tagged point as JSON.")]
    answers = []
    for seed in range(6):
        request = ChatRequest(messages, 1500, SamplingOptions(temperature=1.0, seed=seed),
                              response_format=ResponseFormat("json_schema", SCHEMA))  # fmt: skip
        result = tiny.chat_completion(request)
        assert result.finish_reason == "stop", result.content
        value = json.loads(result.content)
        assert _valid(value), value
        answers.append(result)
    assert answers[0].fingerprint == SCHEMA_FINGERPRINTS["tagged_point"]
