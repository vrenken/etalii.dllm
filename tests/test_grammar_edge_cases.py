"""Edge cases of constrained decoding: schema validation errors, schema forms without an explicit ``type``, array
bounds, grammar composition errors, and the lazy (triggered) token constraint when the model leaves the grammar."""

from __future__ import annotations

import pytest
from test_grammar import VOCABULARY, matches

from etalii_dllm.grammar import Grammar, GrammarError, Matcher, TokenConstraint, TokenTrie


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        ({"type": "string", "dependentRequired": {}}, "not supported by constrained decoding: dependentRequired"),
        ({"type": "number", "multipleOf": 0}, "'multipleOf' must be greater than 0"),
        ({"type": "object", "unevaluatedProperties": False}, "unevaluatedProperties"),
        ({"type": "array", "uniqueItems": True}, "uniqueItems"),
        ({"not": {"type": "integer"}}, "cannot negate type 'integer'"),
        ({"type": "array", "items": {"type": "string", "format": "x", "unevaluatedItems": {}}}, "unevaluatedItems"),
        ("string", "a schema must be an object, got 'string'"),
        ({"type": "array", "items": 5}, "a schema must be an object, got 5"),
        ({"enum": []}, "'enum' must not be empty"),
        ({"allOf": [{"type": "string"}, {"type": "null"}]}, "no value satisfies"),
        ({"type": "array", "allOf": [{"contains": {"const": 1}}, {"contains": {"const": 2}}]}, "different 'contains'"),
        ({"type": "array", "minItems": 3, "maxItems": 2}, "'maxItems' is smaller than 'minItems'"),
        (
            {"type": "object", "properties": {"a": {}}, "required": ["a", "b", "c"]},
            "required properties without a schema: b, c",
        ),
        ({"type": "tuple"}, "unknown JSON schema type 'tuple'"),
        ({"type": ["string", "date"]}, "unknown JSON schema type 'date'"),
        ({"$ref": "#/$defs/missing", "$defs": {}}, r"unresolved \$ref '#/\$defs/missing'"),
        ({"$ref": "#/definitions/missing"}, r"unresolved \$ref"),
        ({"$ref": "other.json#/$defs/x"}, r"only local \$ref values are supported, got 'other.json#/\$defs/x'"),
    ],
)
def test_invalid_schemas_are_refused_with_a_reason(schema, message):
    with pytest.raises(GrammarError, match=message):
        Grammar.json_schema(schema)


def test_lenient_mode_still_refuses_structural_errors():
    # ``lenient`` only ignores unsupported keywords; a schema that cannot be compiled is still an error.
    assert Grammar.json_schema({"type": "string", "minLength": 3}, lenient=True).matcher().matches(b'""')
    with pytest.raises(GrammarError, match="'enum' must not be empty"):
        Grammar.json_schema({"enum": [], "minLength": 3}, lenient=True)


def test_single_allof_is_its_schema():
    schema = {"allOf": [{"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}]}
    assert matches(schema, '{"a":1}')
    assert not matches(schema, "{}")
    assert not matches(schema, '{"a":"x"}')


def test_type_is_inferred_from_properties_and_items():
    with_properties = {"properties": {"x": {"type": "boolean"}}, "required": ["x"]}
    assert matches(with_properties, '{"x":true}')
    assert not matches(with_properties, "[true]")
    assert not matches(with_properties, "true")
    with_items = {"items": {"type": "null"}}
    assert matches(with_items, "[null,null]")
    assert matches(with_items, "[]")
    assert not matches(with_items, "[1]")
    assert not matches(with_items, "{}")


@pytest.mark.parametrize("text", ['"s"', "-1.5e3", "true", "null", "[1,{}]", '{"k":["v"]}'])
def test_schema_without_type_or_structure_accepts_any_value(text):
    # Only annotations: any JSON value.
    assert matches({"description": "anything", "title": "T"}, text)


def test_min_items_is_enforced():
    schema = {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 3}
    assert not matches(schema, "[]")
    assert not matches(schema, "[1]")
    assert matches(schema, "[1,2]")
    assert matches(schema, "[1, 2, 3]")
    assert not matches(schema, "[1,2,3,4]")
    # The closing bracket is not a legal next byte until minItems values have been written.
    matcher = Grammar.json_schema(schema).matcher()
    assert matcher.advance(matcher.start, b"[1") != Matcher.DEAD
    assert matcher.advance(matcher.start, b"[1]") == Matcher.DEAD


def test_object_without_properties_and_no_additional_properties_is_empty():
    schema = {"type": "object", "additionalProperties": False}
    assert matches(schema, "{}")
    assert matches(schema, "{ }")
    assert not matches(schema, '{"a":1}')
    # Without ``additionalProperties: false`` it is a free-form object.
    assert matches({"type": "object"}, '{"a":1}')


def test_required_without_properties_is_a_free_object():
    # ``required`` names without any ``properties`` do not constrain the (free-form) object.
    schema = {"type": "object", "required": ["a"]}
    assert matches(schema, "{}")
    assert matches(schema, '{"b":2}')


def test_root_reference_makes_recursive_schemas():
    schema = {"type": "array", "items": {"anyOf": [{"type": "integer"}, {"$ref": "#"}]}}
    assert matches(schema, "[1,[2,[3,[]]],4]")
    assert not matches(schema, '[1,["x"]]')


def test_definitions_are_resolved_like_defs():
    schema = {"definitions": {"flag": {"type": "boolean"}}, "$ref": "#/definitions/flag"}
    assert matches(schema, "false")
    assert not matches(schema, "0")


def test_integer_and_number_differ():
    assert matches({"type": "number"}, "1.25E+2")
    assert not matches({"type": "integer"}, "1.0")
    assert not matches({"type": "integer"}, "1e2")
    assert matches({"type": "integer"}, "-0")
    assert not matches({"type": "number"}, "-")
    assert not matches({"type": "number"}, "1.")
    assert not matches({"type": "number"}, ".5")


def test_either_grammars_cannot_be_nested_in_a_sequence():
    either = Grammar.either([Grammar.literal("a"), Grammar.literal("b")])
    with pytest.raises(ValueError, match="either\\(\\) grammars cannot be part of a sequence"):
        Grammar.sequence([Grammar.literal("<"), either])


def test_either_flattens_nested_alternatives():
    inner = Grammar.either([Grammar.literal("a"), Grammar.literal("b")])
    outer = Grammar.either([inner, Grammar.literal("c")]).matcher()
    assert all(outer.matches(text) for text in (b"a", b"b", b"c"))
    assert not outer.matches(b"ab")


@pytest.mark.parametrize(
    "grammar",
    [
        Grammar.literal("x"),
        Grammar.sequence([Grammar.json_schema({"type": "string"}), Grammar.literal("!")]),
        Grammar.literal(""),
    ],
)
def test_choice_needs_single_json_values(grammar):
    with pytest.raises(ValueError, match="choice\\(\\) takes single JSON-value grammars"):
        Grammar.choice([Grammar.json_schema({"type": "integer"}), grammar])


def test_choice_of_json_values():
    matcher = Grammar.choice([Grammar.json_schema({"type": "integer"}), Grammar.json_schema({"const": "x"})]).matcher()
    assert matcher.matches(b"12") and matcher.matches(b'"x"')
    assert not matcher.matches(b'"y"')


def test_empty_literal_and_whitespace_grammars():
    empty = Grammar.literal("").matcher()
    assert empty.matches(b"") and empty.finished(empty.start)
    spaces = Grammar.whitespace().matcher()
    assert spaces.matches(b"") and spaces.matches(b" \t\r\n")
    assert not spaces.matches(b"x")


def test_dead_state_is_neither_accepting_nor_finished():
    matcher = Grammar.literal("ab").matcher()
    dead = matcher.advance(matcher.start, b"x")
    assert dead == Matcher.DEAD
    assert not matcher.accepting(dead) and not matcher.finished(dead)
    assert matcher.advance(dead, b"ab") == Matcher.DEAD  # a dead state stays dead


# Lazy constraints: a trigger switches the constraint on; leaving the grammar switches it off for good.

TRIGGER_VOCABULARY = [*VOCABULARY, b"<t>tr", b"<t>true!", b"<t>x", b"ok<t>", b"<t>"]


def lazy(vocabulary=TRIGGER_VOCABULARY) -> TokenConstraint:
    grammar = Grammar.sequence([Grammar.json_schema({"type": "boolean"}), Grammar.literal("!")])
    return TokenConstraint(grammar, TokenTrie(vocabulary), trigger="<t>")


def token(data: bytes) -> int:
    return TRIGGER_VOCABULARY.index(data)


def test_bytes_after_the_trigger_that_leave_the_grammar_release_the_constraint():
    constraint = lazy()
    constraint.accept(token(b"<t>"))
    assert constraint.active and not constraint.may_stop
    constraint.accept(ord("x"))  # not a boolean: the model went its own way
    assert not constraint.active and constraint.may_stop and not constraint.finished
    # The trigger is dropped: later triggers leave generation free.
    for piece in (b"<t>", b"<t>tr", b"<t>true!"):
        constraint.accept(token(piece))
        assert not constraint.active and constraint.may_stop


def test_trigger_followed_by_invalid_bytes_within_one_token_releases_the_constraint():
    constraint = lazy()
    constraint.accept(token(b"<t>x"))
    assert not constraint.active
    constraint.accept(token(b"<t>"))  # the trigger was dropped
    assert not constraint.active and constraint.may_stop


def test_trigger_and_prefix_of_the_match_in_one_token():
    constraint = lazy()
    constraint.accept(token(b"<t>tr"))
    assert constraint.active
    assert constraint.allowed() == [ord("u")]
    for byte in b"ue!":
        constraint.accept(byte)
    assert not constraint.active  # matched: free again until the next trigger
    constraint.accept(token(b"ok<t>"))
    assert constraint.active and constraint.allowed() == sorted([ord("t"), ord("f")])


def test_trigger_and_complete_match_in_one_token():
    constraint = lazy()
    constraint.accept(token(b"<t>true!"))
    assert not constraint.active and constraint.may_stop  # the match finished within the token
    constraint.accept(token(b"<t>"))  # the trigger still works afterwards
    assert constraint.active


def test_trigger_split_across_tokens():
    constraint = lazy()
    for byte in b"text <":
        constraint.accept(byte)
    constraint.accept(ord("t"))
    assert not constraint.active
    constraint.accept(ord(">"))
    assert constraint.active


def test_single_byte_trigger():
    grammar = Grammar.json_schema({"type": "integer"})
    constraint = TokenConstraint(grammar, TokenTrie(VOCABULARY), trigger="#")
    for byte in b"abc":
        constraint.accept(byte)
        assert not constraint.active
    constraint.accept(ord("#"))
    assert constraint.active
    assert constraint.allowed() == [ord(c) for c in "-0123456789"]
