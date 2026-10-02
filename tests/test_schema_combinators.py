"""Phase 35: exact schema combinators. ``allOf`` over several schemas (#236), ``not`` and ``if``/``then``/``else``
over decidable conditions (#237), ``patternProperties`` (#238) and ``contains`` with ``minContains``/
``maxContains`` (#239) are enforced exactly, checked against an independent validator (``jsonschema``), and answers
stay valid under the whole schema (#240)."""

from __future__ import annotations

import json
from decimal import Decimal

import jsonschema
import pytest
from golden_values import SCHEMA_FINGERPRINTS
from test_engine_import import model_path  # noqa: F401 - fixture

from etalii_dllm.chat import ChatMessage
from etalii_dllm.engine import ChatRequest, DllmEngine, ResponseFormat
from etalii_dllm.grammar import Grammar, GrammarError
from etalii_dllm.regexp import compile_regex, difference, literal_automaton
from etalii_dllm.sampling import SamplingOptions
from etalii_dllm.schema_algebra import Algebra, all_of


def _validator(schema):
    """The reference validator (JSON Schema 2020-12, formats checked)."""
    return jsonschema.Draft202012Validator(schema, format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER)


def _agrees(schema, candidates) -> None:
    """The automaton accepts exactly the candidates the reference validator accepts."""
    matcher = Grammar.json_schema(schema).matcher()
    validator = _validator(schema)
    for value in candidates:
        assert matcher.matches(json.dumps(value).encode()) == validator.is_valid(value), (schema, value)


def _walks(schema, count: int = 12) -> list[bytes]:
    """Texts the automaton generates: bytes picked pseudo-randomly for a while, then the shortest way to a match.
    Every state must be able to finish (no dead ends), and every text must be valid under the reference."""
    matcher = Grammar.json_schema(schema).matcher()
    validator = _validator(schema)
    texts = []
    seed = 12345
    for _ in range(count):
        state, data = 0, b""
        for _ in range(60):
            allowed = [b for b in range(256) if matcher.step(state, b) >= 0]
            assert allowed or matcher.accepting(state), data
            if not allowed:
                break
            seed = (seed * 6364136223846793005 + 1442695040888963407) % 2**64
            byte = allowed[(seed >> 33) % len(allowed)]
            data += bytes([byte])
            state = matcher.step(state, byte)
        queue, seen = [(state, data)], {state}
        for current, text in queue:
            if matcher.accepting(current):
                data = text
                break
            for byte in range(256):
                following = matcher.step(current, byte)
                if following >= 0 and following not in seen:
                    seen.add(following)
                    queue.append((following, text + bytes([byte])))
        else:
            raise AssertionError(f"no way to finish {data!r}")
        value = json.loads(data)
        assert validator.is_valid(value), (schema, data)
        texts.append(data)
    return texts


# allOf


def test_all_of_merges_types_values_and_bounds():
    _agrees({"allOf": [{"type": ["integer", "string"]}, {"type": "number"}]}, [1, 1.5, "a", None, -3])
    _agrees(
        {"allOf": [{"type": "number", "minimum": 0, "multipleOf": 0.5}, {"maximum": 10, "multipleOf": 0.75}]},
        [0, 1.5, 3, 0.5, 0.75, 9, 10.5, -1.5, 4.5],
    )
    _agrees({"type": "number", "allOf": [{"multipleOf": 0.1}, {"multipleOf": 0.25}]}, [0.5, 1, 0.25, 0.1, 1.5, 0.3])
    _agrees({"allOf": [{"enum": ["a", "b", 1]}, {"enum": ["b", 1.0, "c"]}]}, ["a", "b", 1, "c", 2])  # 1 as written
    _agrees({"type": "string", "enum": ["a", 1, "bb"], "maxLength": 1}, ["a", 1, "bb"])
    _agrees({"allOf": [{"const": 2}, {"type": "integer", "minimum": 1}]}, [2, 1, 3])
    _agrees(
        {"type": "number", "allOf": [{"minimum": 1}, {"exclusiveMinimum": 1}, {"maximum": 5}, {"exclusiveMaximum": 4}]},
        [1, 1.5, 3.99, 4, 0],
    )


def test_all_of_merges_strings():
    _agrees({"type": "string", "allOf": [{"pattern": "^a"}, {"pattern": "b$"}, {"minLength": 3}]},
            ["ab", "axb", "aab", "ba", "a", "abab"])  # fmt: skip
    _agrees({"type": "string", "allOf": [{"format": "date"}, {"pattern": "^2026-"}]},
            ["2026-01-02", "2025-01-02", "2026-13-01", "x"])  # fmt: skip
    with pytest.raises(GrammarError, match="no string satisfies"):
        Grammar.json_schema({"type": "string", "allOf": [{"format": "date"}, {"format": "uuid"}]})
    with pytest.raises(GrammarError, match="smaller than 'minLength'"):
        Grammar.json_schema({"type": "string", "allOf": [{"maxLength": 1}, {"minLength": 2}]})
    with pytest.raises(GrammarError, match="no string satisfies"):
        Grammar.json_schema({"type": "string", "allOf": [{"pattern": "^a$"}, {"pattern": "^b$"}]})


def test_all_of_merges_objects():
    first = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
    _agrees({"allOf": [first, {"properties": {"a": {"minimum": 3}, "b": {"type": "string"}}}]},
            [{"a": 3}, {"a": 2}, {"a": 3, "b": "x"}, {"a": 3, "b": 1}, {}])  # fmt: skip
    closed = {**first, "additionalProperties": False}
    _agrees({"allOf": [closed, {"properties": {"b": {"type": "string"}}}]}, [{"a": 1}, {"a": 1, "b": "x"}])
    with pytest.raises(GrammarError, match="required property 'b' has no value"):
        Grammar.json_schema({"allOf": [closed, {"properties": {"b": {}}, "required": ["b"]}]})
    maps = {"allOf": [{"type": "object", "additionalProperties": {"type": "integer"}},
                      {"additionalProperties": {"minimum": 0}, "maxProperties": 2}]}  # fmt: skip
    _agrees(maps, [{}, {"x": 1}, {"x": -1}, {"x": 1, "y": 2}, {"x": 1, "y": 2, "z": 3}, {"x": "a"}])
    patterned = {"allOf": [{"type": "object", "patternProperties": {"^n": {"type": "integer"}}},
                           {"additionalProperties": {"type": ["integer", "string"]}}]}  # fmt: skip
    _agrees(patterned, [{"n1": 1}, {"n1": "a"}, {"x": "a"}, {"x": None}])
    _agrees({"type": "object", "properties": {"x": {"type": "integer"}}, "allOf": [{"propertyNames": {"maxLength": 1}},
             {"propertyNames": {"pattern": "^[a-z]+$"}}]}, [{"x": 1}, {}])  # fmt: skip
    with pytest.raises(GrammarError, match="both have 'patternProperties'"):
        Grammar.json_schema({"allOf": [{"patternProperties": {"a": {}}}, {"patternProperties": {"b": {}}}]})


def test_all_of_merges_arrays_and_distributes_any_of():
    tuples = {"allOf": [{"type": "array", "prefixItems": [{"type": "integer"}]},
                        {"prefixItems": [{"minimum": 0}, {"type": "string"}], "items": False}]}  # fmt: skip
    _agrees(tuples, [[], [1], [-1], [1, "a"], [1, 2], [1, "a", 3]])
    _agrees({"type": "array", "allOf": [{"prefixItems": [{"type": "integer"}, False]}, {"minItems": 1}]},
            [[1], [], [1, 2]])  # fmt: skip
    _agrees({"type": "array", "allOf": [{"items": {"enum": [1, 2, 3]}}, {"uniqueItems": True, "maxItems": 2}]},
            [[1, 2], [1, 1], [1, 2, 3], [4]])  # fmt: skip
    either = {"allOf": [{"anyOf": [{"type": "string"}, {"type": "integer"}]},
                        {"anyOf": [{"minimum": 5}, {"maxLength": 2}]}, {"not": {"const": "zz"}}]}  # fmt: skip
    _agrees(either, ["ab", "abc", 3, 7, None, "zz"])
    _walks(either)


def test_all_of_with_recursive_references():
    node = {"type": "object", "properties": {"value": {"type": "integer"}, "next": {"$ref": "#/$defs/node"}},
            "required": ["value"]}  # fmt: skip
    schema = {"$defs": {"node": node}, "allOf": [{"$ref": "#/$defs/node"}, {"properties": {"value": {"minimum": 0}}}]}
    _agrees(schema, [{"value": 1}, {"value": -1}, {"value": 1, "next": {"value": -5}}, {"value": 0, "next": {}}])
    mutual = {
        "$defs": {"a": {"type": "object", "properties": {"x": {"$ref": "#/$defs/b"}, "n": {"type": "integer"}}},
                  "b": {"type": "object", "properties": {"x": {"$ref": "#/$defs/a"}, "n": {"minimum": 0}}}},
        "allOf": [{"$ref": "#/$defs/a"}, {"$ref": "#/$defs/b"}],
    }  # fmt: skip
    _agrees(mutual, [{"n": 1}, {"n": -1}, {"x": {"x": {"n": 2}}}, {"x": {"x": {"n": -2}}}])
    _walks(mutual, 4)
    looping = {"$defs": {"a": {"allOf": [{"$ref": "#/$defs/a"}, {"type": "string"}]}}, "$ref": "#/$defs/a"}
    with pytest.raises(GrammarError, match="nest too deeply"):
        Grammar.json_schema(looping)


def test_nullable_and_draft_4_bounds_merge():
    nullable = Grammar.json_schema({"allOf": [{"type": "integer", "nullable": True}, {"maximum": 3}]}).matcher()
    assert nullable.matches(b"null") and nullable.matches(b"2") and not nullable.matches(b"4")  # OpenAPI's nullable
    schema = {"allOf": [{"type": "number", "minimum": 1, "exclusiveMinimum": True}, {"maximum": 2}]}
    matcher = Grammar.json_schema(schema).matcher()
    assert not matcher.matches(b"1") and matcher.matches(b"1.5") and matcher.matches(b"2")


# not and if/then/else


def test_not_over_types_values_strings_and_numbers():
    _agrees({"not": {"type": "string"}}, ["a", 1, None, True, [], {}])
    _agrees({"not": {"not": {"type": "boolean"}}}, [True, 1, None])
    _agrees({"not": {"anyOf": [{"type": "string"}, {"type": "number"}]}}, [None, False, "a", 1.5, [1]])
    _agrees({"type": "string", "not": {"enum": ["a", "b"]}}, ["a", "b", "c", "ab"])
    _agrees({"type": "integer", "minimum": 0, "maximum": 10, "not": {"multipleOf": 3}}, list(range(-1, 12)))
    _agrees({"type": "number", "not": {"minimum": 1, "maximum": 2}}, [0, 1, 1.5, 2, 2.5, -3])
    _agrees({"type": "string", "not": {"pattern": "^x"}}, ["x", "xy", "yx", ""])
    _agrees({"type": "string", "not": {"minLength": 2}}, ["", "a", "ab"])
    _agrees({"type": "string", "maxLength": 3, "not": {"maxLength": 1}}, ["", "a", "ab", "abcd"])
    _agrees({"type": "string", "not": {"format": "date"}}, ["2026-01-02", "x"])
    _agrees({"type": "number", "not": {"const": 1}}, [1, 1.0, 2, 0.5, -1])
    _agrees({"type": "number", "not": {"enum": [0, 2.5, -1]}}, [0, 0.0, 2.5, -1, 1])
    _agrees({"enum": [1, 2, 3, "a"], "not": {"enum": [2, "a"]}}, [1, 2, 3, "a"])
    _agrees({"type": "boolean", "not": {"const": True}}, [True, False])
    _agrees({"not": {"const": None}}, [None, 1, "a"])
    _agrees({"not": {"type": "number", "not": {"multipleOf": 2}}}, [2, 3, "a"])
    _walks({"not": {"type": ["object", "array", "string"]}})


def test_not_over_object_conditions():
    base = {"type": "object", "properties": {"kind": {"enum": ["a", "b"]}, "n": {"type": "integer"}},
            "required": ["kind"]}  # fmt: skip
    _agrees({**base, "not": {"properties": {"kind": {"const": "a"}}}}, [{"kind": "a"}, {"kind": "b"}])
    _agrees({**base, "not": {"required": ["n"]}}, [{"kind": "a"}, {"kind": "a", "n": 1}])
    _agrees({**base, "not": {"properties": {"n": {"minimum": 0}}}}, [{"kind": "a"}, {"kind": "a", "n": -1},
                                                                     {"kind": "a", "n": 1}])  # fmt: skip


def test_if_then_else():
    shapes = {
        "type": "object",
        "properties": {"kind": {"enum": ["circle", "square"]}, "radius": {"type": "number", "minimum": 0},
                       "side": {"type": "number", "minimum": 0}},
        "required": ["kind"],
        "if": {"properties": {"kind": {"const": "circle"}}},
        "then": {"required": ["radius"]},
        "else": {"required": ["side"]},
    }  # fmt: skip
    _agrees(
        shapes,
        [
            {"kind": "circle", "radius": 1},
            {"kind": "circle"},
            {"kind": "square", "side": 2},
            {"kind": "square", "radius": 1},
            {"kind": "square", "radius": 1, "side": 2},
            {"kind": "x"},
        ],
    )
    _walks(shapes)
    _agrees({"if": {"type": "string"}, "then": {"minLength": 2}, "else": {"type": "integer"}},
            ["a", "ab", 1, 1.5, None])  # fmt: skip
    _agrees({"type": "integer", "if": {"minimum": 10}, "then": {"multipleOf": 5}, "else": {"multipleOf": 2}},
            [2, 3, 10, 12, 15, 4])  # fmt: skip
    _agrees({"type": "string", "if": {"const": "a"}}, ["a", "b"])  # no then/else: no constraint
    _agrees({"type": "string", "then": {"const": "a"}}, ["a", "b"])  # no if: then is ignored


def test_conditions_that_cannot_be_negated_are_refused():
    refusals = [
        ({"not": {"type": "integer"}}, "cannot negate type 'integer'"),
        ({"not": {"type": "array", "items": {"type": "string"}}}, "cannot negate 'items'"),
        ({"type": "object", "if": {"additionalProperties": False}, "then": {}}, "cannot negate 'additionalProperties'"),
        ({"not": {"enum": [[1]]}, "type": "array"}, "can only exclude strings, numbers"),
    ]  # fmt: skip
    for schema, message in refusals:
        with pytest.raises(GrammarError, match=message):
            Grammar.json_schema(schema)
    with pytest.raises(GrammarError, match="no value satisfies"):
        Grammar.json_schema({"type": "string", "not": {}})


# patternProperties


def test_pattern_properties():
    typed = {"type": "object", "patternProperties": {"^s_": {"type": "string"}, "^n_": {"type": "integer"}},
             "additionalProperties": False}  # fmt: skip
    _agrees(typed, [{}, {"s_a": "x"}, {"s_a": "x", "n_b": 1}, {"x": 1}, {"s_a": 1}, {"n_": 2.5}])
    overlapping = {"type": "object", "patternProperties": {"^a": {"type": "integer"}, "b$": {"minimum": 5}},
                   "additionalProperties": {"type": "string"}}  # fmt: skip
    _agrees(overlapping, [{"ab": 5}, {"ab": 4}, {"a": 1}, {"b": "x"}, {"b": 3}, {"b": 7}, {"c": "s"}, {"c": 1}])
    _walks(overlapping)
    named = {"type": "object", "propertyNames": {"maxLength": 3}, "patternProperties": {"^x": {"type": "boolean"}},
             "additionalProperties": {"type": "null"}, "maxProperties": 2}  # fmt: skip
    _agrees(named, [{"xa": True}, {"xa": None}, {"ya": None}, {"xabc": True}, {"x": True, "y": None, "z": None}])
    _walks(named)
    literal = {"type": "object", "propertyNames": {"enum": ["xa", "b"]},
               "patternProperties": {"^x": {"type": "boolean"}}, "additionalProperties": {"type": "null"}}  # fmt: skip
    _agrees(literal, [{"xa": True}, {"xa": None}, {"b": None}, {"b": True}, {"c": None}])
    declared = {"type": "object", "properties": {"x1": {"type": "integer"}, "y": {}},
                "patternProperties": {"^x": {"minimum": 0}}}  # fmt: skip
    _agrees(declared, [{"x1": 1}, {"x1": -1}, {"y": -1}])
    _agrees({"type": "object", "properties": {"x1": {"type": "integer"}}, "patternProperties": {"^x": False}},
            [{}, {"x1": 1}])  # fmt: skip
    nothing = {"type": "object", "patternProperties": {"": False}}
    _agrees(nothing, [{}, {"a": 1}])
    with pytest.raises(GrammarError, match="no object without properties"):
        Grammar.json_schema({**nothing, "minProperties": 1})
    assert Grammar.json_schema({"type": "object", "patternProperties": {"^a": {"type": "integer"}}},
                               lenient=True).matcher().matches(b'{"a": "x"}')  # fmt: skip


# contains


def test_contains_with_any_items():
    _agrees({"type": "array", "items": {"type": "integer"}, "contains": {"minimum": 10}, "minContains": 2},
            [[], [10], [10, 11], [1, 10, 2, 12], [10, 1], [9, 9, 9]])  # fmt: skip
    _agrees({"type": "array", "contains": {"type": "string"}}, [[], [1], [1, "a"], ["a"]])
    tuple_schema = {"type": "array", "prefixItems": [{"type": "string"}, {"type": "integer"}], "items": False,
                    "contains": {"type": "integer", "minimum": 5}}  # fmt: skip
    _agrees(tuple_schema, [["a", 5], ["a", 4], ["a"], []])
    _agrees({"type": "array", "contains": {"const": 1}, "minContains": 0}, [[], [2]])
    for schema in (tuple_schema, {"type": "array", "items": {"type": "integer"}, "contains": {"minimum": 10},
                                  "minContains": 2, "maxItems": 3}):  # fmt: skip
        _walks(schema)


def test_contains_over_finite_items():
    bounded = {"type": "array", "items": {"enum": ["a", "b", "c"]}, "contains": {"const": "a"}, "minContains": 1,
               "maxContains": 2, "maxItems": 4}  # fmt: skip
    _agrees(bounded, [[], ["a"], ["b"], ["a", "a"], ["a", "a", "a"], ["b", "a", "c", "a"], ["a", "b", "c", "a", "b"]])
    _walks(bounded)
    unique = {"type": "array", "items": {"enum": [1, 2, 3, 4]}, "uniqueItems": True, "contains": {"minimum": 3},
              "minContains": 2}  # fmt: skip
    _agrees(unique, [[3, 4], [4, 3, 1], [3], [1, 2, 3], [3, 3]])
    _walks(unique)
    at_most = {"type": "array", "items": {"type": "boolean"}, "contains": {"const": True}, "minContains": 0,
               "maxContains": 1}  # fmt: skip
    _agrees(at_most, [[], [False, False], [True], [True, False, True]])
    tight = {"type": "array", "items": {"enum": ["a", "b"]}, "contains": {"const": "a"}, "maxContains": 1,
             "minItems": 3}  # fmt: skip
    _agrees(tight, [["a", "b", "b"], ["a", "a", "b"], ["b", "b"]])
    _walks(tight)


def test_contains_refusals():
    refusals = [
        ({"type": "array", "items": {"type": "integer"}, "contains": {"minimum": 1}, "maxContains": 2},
         "'maxContains' needs items from a finite set"),
        ({"type": "array", "contains": {}, "minContains": 3, "maxItems": 2}, "no array has 'minContains'"),
        ({"type": "array", "items": {"const": "a"}, "contains": {"const": "a"}, "maxContains": 1, "minItems": 2},
         "no array has 'minContains'"),
        ({"type": "array", "items": {"enum": [1, 2]}, "contains": {"const": 3}}, "no array has 'minContains'"),
        ({"type": "array", "contains": {}, "minContains": 2, "maxContains": 1}, "smaller than 'minContains'"),
        ({"type": "array", "contains": {}, "minContains": -1}, "non-negative integer"),
        ({"type": "array", "prefixItems": [{"const": 1}], "items": False, "contains": {"const": 2}},
         "no array has 'minContains'"),
    ]  # fmt: skip
    for schema, message in refusals:
        with pytest.raises(GrammarError, match=message):
            Grammar.json_schema(schema)
    assert Grammar.json_schema({"type": "array", "contains": {"const": 1}}, lenient=True).matcher().matches(b"[]")


# The algebra itself


def test_algebra_helpers():
    assert all_of(True, {}, {"type": "string"}, {"type": "string"}) == {"type": "string"}
    assert all_of({"a": 1}, False) is False and all_of() is True
    algebra = Algebra(lambda reference: {"type": "integer"})
    assert algebra.simplify({"$ref": "#/$defs/x", "minimum": 1}) == {"minimum": 1, "type": "integer"}
    assert algebra.negate(False) == {} and algebra.negate({"title": "x"}) is False
    assert algebra.merge({"type": "string"}, {"type": "number"}) is False
    assert algebra.merge({"multipleOf": 0.1}, {"multipleOf": 0.15})["multipleOf"] == Decimal("0.3")
    with pytest.raises(GrammarError, match="greater than 0"):
        algebra.merge({"multipleOf": 0.1}, {"multipleOf": -1})
    assert algebra.merge({"x-custom": 1}, {"x-custom": 1, "type": "string"}) == {"x-custom": 1, "type": "string"}
    with pytest.raises(GrammarError, match="two different values of 'x-custom'"):
        algebra.merge({"x-custom": 1}, {"x-custom": 2})
    with pytest.raises(GrammarError, match="more than 64 alternatives"):
        lows, highs = [{"minimum": j} for j in range(9)], [{"maximum": j} for j in range(9)]
        algebra.simplify({"allOf": [{"anyOf": lows}, {"anyOf": highs}]})
    dfa = difference(compile_regex("[ab]+"), literal_automaton([b"a", b"ab"]))
    assert dfa.matches(b"b") and not dfa.matches(b"a") and not dfa.matches(b"ab") and dfa.matches(b"aba")


# Answers

SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"enum": ["circle", "square"]},
        "radius": {"type": "integer", "minimum": 1, "maximum": 9},
        "side": {"type": "integer", "minimum": 1, "maximum": 9},
        "tags": {"type": "array", "items": {"enum": ["red", "green", "blue"]}, "contains": {"const": "red"},
                 "maxContains": 1, "maxItems": 3},
        "labels": {"type": "object", "propertyNames": {"pattern": "^[a-z_]{1,6}$"},
                   "patternProperties": {"^is_": {"type": "boolean"}},
                   "additionalProperties": {"type": "integer", "not": {"multipleOf": 2}}, "maxProperties": 2},
    },
    "required": ["kind", "tags", "labels"],
    "allOf": [
        {"if": {"properties": {"kind": {"const": "circle"}}}, "then": {"required": ["radius"]},
         "else": {"required": ["side"]}},
    ],
}  # fmt: skip


@pytest.fixture(scope="module")
def tiny(model_path) -> DllmEngine:  # noqa: F811
    return DllmEngine.from_model_file(model_path)


def test_answers_are_valid_under_the_whole_schema(tiny):
    messages = [ChatMessage("user", "Describe a labelled shape as JSON.")]
    validator = _validator(SCHEMA)
    answers = []
    for seed in range(6):
        request = ChatRequest(messages, 1500, SamplingOptions(temperature=1.0, seed=seed),
                              response_format=ResponseFormat("json_schema", SCHEMA))  # fmt: skip
        result = tiny.chat_completion(request)
        assert result.finish_reason == "stop", result.content
        value = json.loads(result.content)
        assert validator.is_valid(value), value
        answers.append(result)
    assert answers[0].fingerprint == SCHEMA_FINGERPRINTS["labelled_shape"]


def test_combinator_edge_cases():
    with pytest.raises(GrammarError):
        Grammar.json_schema({"type": "string", "pattern": "(?=a)"})  # unsupported regex syntax, not "unsatisfiable"
    lenient = Grammar.json_schema({"allOf": [{"type": "integer"}, {"type": "string"}]}, lenient=True).matcher()
    assert lenient.matches(b"1") and Grammar.json_schema({"allOf": []}, lenient=True).matcher().matches(b"[]")
    _agrees({"type": ["string", "integer"], "minLength": 3, "maxLength": 1}, ["abc", 1])
    _agrees({"anyOf": [{"type": "string", "minLength": 2, "maxLength": 1}, {"type": "null"}]}, ["ab", None])
    _agrees({"enum": ["xa", "b"], "not": {"pattern": "^x"}}, ["xa", "b"])
    _agrees({"type": "array", "items": {"enum": [1, 2]}, "contains": False, "minContains": 0, "maxContains": 0},
            [[1, 2], []])  # fmt: skip
    _agrees({"type": "array", "prefixItems": [{"type": "string"}], "contains": {"type": "integer"}},
            [["a", 1], ["a"], ["a", "b"]])  # fmt: skip
    _agrees({"type": "number", "allOf": [{"minimum": 1}, {"minimum": 2}, {"multipleOf": 0.1}, {"multipleOf": 0.25},
                                          {"multipleOf": 0.3}]}, [1.5, 3, 4.5, 1.25])  # fmt: skip
    _agrees({"type": "string", "allOf": [{"not": {"const": "a"}}, {"not": {"const": "b"}}]}, ["a", "b", "c"])
    _agrees({"type": "string", "allOf": [{"format": "date", "minLength": 1}, {"format": "date"}]}, ["2026-01-02", "x"])
    same = {"patternProperties": {"^x": {"type": "integer"}}}
    _agrees({"type": "object", "allOf": [same, {**same, "maxProperties": 1}]}, [{"x": 1}, {"x": "a"}, {"x": 1, "y": 2}])
    _agrees({"type": "array", "items": {"enum": [1, 2]}, "allOf": [{"contains": {"const": 1}, "minContains": 2},
             {"contains": {"const": 1}, "maxContains": 2}]}, [[1, 1], [1], [1, 1, 1], [1, 2, 1]])  # fmt: skip
    _agrees({"allOf": [{"type": "object", "patternProperties": {"^x": {"minimum": 0}}},
                       {"properties": {"x1": {"type": "integer"}}}]}, [{"x1": 1}, {"x1": -1}])  # fmt: skip
    _agrees({"not": {"type": "string", "not": {"const": "a"}}}, ["a", "b", 1])
    _agrees({"not": {"not": {"pattern": "^a"}}}, ["ab", "b", 1])
    _agrees({"type": "object", "propertyNames": {"maxLength": 2}, "patternProperties": {"^abc": {"type": "integer"}},
             "additionalProperties": {"type": "string"}}, [{"ab": "x"}, {"ab": 1}, {"abcd": 1}])  # fmt: skip
    names = {"anyOf": [{"enum": ["xa"]}, {"type": "string", "pattern": "^b"}]}
    _agrees({"type": "object", "propertyNames": names, "patternProperties": {"^x": {"type": "integer"}}},
            [{"xa": 1}, {"xa": "a"}, {"bz": "a"}, {"c": 1}])  # fmt: skip
    _agrees({"enum": [3, 4, 4.5, "a"], "not": {"multipleOf": 2}}, [3, 4, 4.5, "a"])
    _agrees({"type": "object", "propertyNames": {"minLength": 2, "maxLength": 1}}, [{}, {"a": 1}])
    refusals = [
        ({"type": ["string"], "minLength": 3, "maxLength": 1}, "no type of the schema"),
        ({"anyOf": [False]}, "no branch of 'anyOf'"),
        ({"anyOf": [{"type": "string", "minLength": 2, "maxLength": 1}]}, "no branch of 'anyOf'"),
        ({"enum": ["a"], "type": "integer"}, "no value of 'enum'"),
        ({"enum": ["a"], "type": "string", "minLength": 3, "maxLength": 1}, "no value of 'enum'"),
        ({"type": "boolean", "not": {"enum": [True, False]}}, "satisfies the schema's 'not'"),
        ({"type": "string", "maxLength": 0, "not": {"const": ""}}, "satisfies the schema's 'not'"),
        ({"type": "object", "patternProperties": {c: {} for c in "abcdefg"}}, "more than 64 parts"),
    ]  # fmt: skip
    for schema, message in refusals:
        with pytest.raises(GrammarError, match=message):
            Grammar.json_schema(schema)


def test_algebra_simplifies_forms():
    algebra = Algebra(lambda reference: {})
    with pytest.raises(GrammarError, match="must be an object"):
        algebra.simplify(5)
    assert algebra.simplify({"const": "a", "enum": ["b"]}) is False
    assert algebra.simplify({"const": "a", "enum": ["a", "b"]}) == {"enum": ["a"]}
    assert algebra.simplify({"items": [{"type": "string"}], "additionalItems": False}) == {
        "items": False, "prefixItems": [{"type": "string"}]}  # fmt: skip
    assert algebra.simplify({"prefixItems": [], "items": [{}], "additionalItems": False}) == {"prefixItems": [],
                                                                                           "items": [{}]}  # fmt: skip
    assert algebra.simplify({"anyOf": [False, {"type": "string"}]}) == {"type": "string"}
    assert algebra.simplify({"anyOf": [{}, {"type": "string"}]}) == {}
    assert algebra.simplify({"anyOf": [False]}) is False
    from etalii_dllm.grammar import _STRING, _excluded

    assert _excluded(_STRING, (("multipleOf", 2),)) is not None  # divisors leave strings alone
