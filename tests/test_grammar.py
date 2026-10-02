import json

import pytest

from etalii_dllm.grammar import MAX_WHITESPACE, Grammar, GrammarError, Matcher, TokenConstraint, TokenTrie

PERSON = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "tags": {"type": "array", "items": {"enum": ["a", "b"]}, "maxItems": 2},
        "score": {"type": ["number", "null"]},
    },
    "required": ["name", "age"],
}


def matches(schema, text: str | bytes) -> bool:
    data = text.encode("utf-8") if isinstance(text, str) else text
    return Grammar.json_schema(schema).matcher().matches(data)


@pytest.mark.parametrize(
    "text",
    [
        '{"name":"Bob","age":3}',
        '{"name": "B\\u00e9 \\"q\\"", "age": -10, "tags": ["a", "b"], "score": 1.5e3}',
        '{\n  "name": "é",\n  "age": 0,\n  "score": null\n}',
        '{"name":"x","age":1,"tags":[]}',
    ],
)
def test_accepts_valid_documents(text):
    assert matches(PERSON, text)
    json.loads(text)


@pytest.mark.parametrize(
    "text",
    [
        '{"age":3}',  # missing required property
        '{"name":"x","age":1.5}',  # integer
        '{"name":"x","age":01}',  # leading zero
        '{"age":1,"name":"x"}',  # properties follow the schema order
        '{"name":"x","age":1,"tags":["c"]}',  # enum
        '{"name":"x","age":1,"tags":["a","b","a"]}',  # maxItems
        '{"name":"x","age":1,"extra":1}',  # undeclared property
        '{"name":"x","age":1}  ',  # nothing after the value
        '{"name":"x\x01","age":1}',  # control character in a string
    ],
)
def test_rejects_invalid_documents(text):
    assert not matches(PERSON, text)


def test_json_object_mode_accepts_any_object():
    matcher = Grammar.json_object().matcher()
    assert matcher.matches(b'{"a":[1,{"b":null}],"c":"d","e":true,"f":-0.5E-3}')
    assert matcher.matches(b"{}")
    assert not matcher.matches(b"[1]")
    assert not matcher.matches(b'{"a":}')


def test_recursive_refs_and_combinators():
    tree = {
        "$defs": {
            "node": {
                "type": "object",
                "properties": {"v": {"type": "integer"}, "kids": {"type": "array", "items": {"$ref": "#/$defs/node"}}},
                "required": ["v", "kids"],
            }
        },
        "$ref": "#/$defs/node",
    }
    assert matches(tree, '{"v":1,"kids":[{"v":2,"kids":[]}]}')
    assert not matches(tree, '{"v":1,"kids":[{"v":2}]}')
    assert matches({"anyOf": [{"type": "string"}, {"type": "boolean"}]}, "true")
    assert matches({"const": {"k": [1, "x"]}}, '{"k":[1,"x"]}')
    assert matches({"type": "string", "nullable": True}, "null")


def test_strings_are_well_formed_utf8():
    for text in ["a", "é", "€", "𝄞", "\U0010ffff"]:
        assert matches({"type": "string"}, f'"{text}"')
    for bad in [b'"\xed\xa0\x80"', b'"\xe0\x80\x80"', b'"\xf4\x90\x80\x80"', b'"\xc0\x80"', b'"\xff"']:
        assert not matches({"type": "string"}, bad)


def test_whitespace_runs_are_bounded():
    assert matches({"type": "array"}, "[" + " " * MAX_WHITESPACE + "]")
    assert not matches({"type": "array"}, "[" + " " * (MAX_WHITESPACE + 1) + "]")


def test_unsupported_keywords_are_refused_unless_lenient():
    schema = {"type": "string", "pattern": "^a+$", "dependentSchemas": {}}
    with pytest.raises(GrammarError, match="dependentSchemas"):
        Grammar.json_schema(schema)
    assert Grammar.json_schema(schema, lenient=True).matcher().matches(b'"b"')  # lenient ignores value constraints
    with pytest.raises(GrammarError):
        Grammar.json_schema({"$ref": "https://example.com/schema"})


def test_sequences_and_alternatives():
    call = Grammar.sequence([Grammar.literal("<x>"), Grammar.json_schema({"type": "integer"}), Grammar.literal("</x>")])
    either = Grammar.either([call, Grammar.json_object()]).matcher()
    assert either.matches(b"<x>12</x>")
    assert either.matches(b"{}")
    assert not either.matches(b"<x>{}</x>")


def test_matcher_states_report_completion():
    matcher = Grammar.json_schema({"type": "integer"}).matcher()
    state = matcher.advance(matcher.start, b"12")
    assert matcher.accepting(state) and not matcher.finished(state)  # more digits may follow
    matcher = Grammar.json_object().matcher()
    state = matcher.advance(matcher.start, b"{}")
    assert matcher.finished(state)
    assert matcher.advance(state, b"x") == Matcher.DEAD


VOCABULARY = [bytes([b]) for b in range(256)] + [b'{"', b'":', b'"}', b"name", b" true", b"\xc3", b"\xa9", b""]


def test_token_mask_matches_brute_force():
    trie = TokenTrie(VOCABULARY)
    grammar = Grammar.json_schema({"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]})
    constraint = TokenConstraint(grammar, trie)
    for piece in [b'{"', b"name", b'":', b'"', b"\xc3", b"\xa9"]:
        allowed = constraint.allowed()
        assert allowed == [t for t in range(len(VOCABULARY)) if VOCABULARY[t] and constraint.allows(t)]
        constraint.accept(VOCABULARY.index(piece))
    assert VOCABULARY.index(b'"}') in constraint.allowed()
    assert not constraint.may_stop
    constraint.accept(VOCABULARY.index(b'"}'))
    assert constraint.finished and constraint.may_stop


def test_lazy_constraint_starts_at_the_trigger():
    trie = TokenTrie(VOCABULARY)
    grammar = Grammar.sequence([Grammar.json_schema({"type": "boolean"}), Grammar.literal("!")])
    constraint = TokenConstraint(grammar, trie, trigger="<t>")
    for byte in b"free text <t":
        constraint.accept(byte)
        assert not constraint.active and constraint.may_stop
    constraint.accept(ord(">"))
    assert constraint.active and not constraint.may_stop
    assert constraint.allowed() == sorted([ord("t"), ord("f")])
    for byte in b"true!":
        constraint.accept(byte)
    assert not constraint.active  # matched: free again until the next trigger
