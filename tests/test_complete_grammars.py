"""Phase 51: exact complete GBNF grammars: left recursion rewritten exactly (#317), token references and their
negations (#318), lazy grammars that start at a trigger word (#319), in every API, the CLI, receipts and the
specification (#320)."""

from __future__ import annotations

import itertools
import re

import pytest
from fastapi.testclient import TestClient
from golden_values import COMPLETE_GRAMMAR_FINGERPRINTS
from test_model_building import BYTE_LEVEL_TOKENIZER, _import

from etalii_dllm.cli import main
from etalii_dllm.engine import ChatRequest, DllmEngine, ResponseFormat
from etalii_dllm.gbnf import (
    MAX_REWRITE_STEPS,
    TokenTable,
    _Class,
    _Group,
    _Literal,
    _parse,
    _Reference,
    _Repeat,
    uses_tokens,
)
from etalii_dllm.grammar import Grammar, GrammarError, Matcher, TokenConstraint, TokenTrie
from etalii_dllm.receipts import request_from_record, request_record
from etalii_dllm.sampling import SamplingOptions
from etalii_dllm.server.app import app
from etalii_dllm.server.contracts import id_payload

EXPRESSION = 'root ::= expr "."\nexpr ::= expr "+" term | expr "-" term | term\nterm ::= [0-9]{1,2} | <[120]>'
LAZY = 'root ::= "e" [0-9]{2} "."'
OPTIONS = SamplingOptions(temperature=0.9, seed=2)


@pytest.fixture
def client():
    return TestClient(app)


# -- left recursion (#317) ----------------------------------------------------------------------------------------


def _language(text: str, longest: int) -> set[str]:
    """The strings of at most ``longest`` characters the grammar derives, by a fixpoint over its written rules (an
    oracle that is indifferent to left recursion)."""
    syntax = _parse(text)
    found: dict[str, set[str]] = {name: set() for name in syntax}

    def join(left: set[str], right: set[str]) -> set[str]:
        return {a + b for a in left for b in right if len(a) + len(b) <= longest}

    def element(item) -> set[str]:
        if isinstance(item, _Literal):
            return {item.text} if len(item.text) <= longest else set()
        if isinstance(item, _Class):
            return {chr(c) for low, high in item.ranges for c in range(low, min(high, 0x7F) + 1)}
        if isinstance(item, _Reference):
            return found[item.name]
        if isinstance(item, _Group):
            return alternatives(item.alternatives)
        assert isinstance(item, _Repeat)
        one, result, current = element(item.element), set(), {""}
        for count in range(longest + 1):
            if count >= item.minimum and (item.maximum is None or count <= item.maximum):
                result |= current
            current = join(current, one)
        return result

    def alternatives(options) -> set[str]:
        result: set[str] = set()
        for sequence in options:
            strings = {""}
            for item in sequence:
                strings = join(strings, element(item))
            result |= strings
        return result

    changed = True
    while changed:
        changed = False
        for name, options in syntax.items():
            strings = alternatives(options)
            if strings - found[name]:
                found[name] |= strings
                changed = True
    return found["root"]


LEFT_RECURSIVE = [
    ('root ::= root "a" | "b"', "ab"),
    ('root ::= a\na ::= b "a" | "c"\nb ::= a "b" | "a"', "abc"),
    ('root ::= root root | "a" | "b"', "ab"),
    ('root ::= ( root "a" | "b" )+ "c"', "abc"),
    ('root ::= x\nx ::= y | "a"\ny ::= x "b" | z\nz ::= x "c" | "d"', "abcd"),
    ('root ::= root "a"? "b" | "c"', "abc"),
    ('root ::= root | root "a" | ""', "ab"),
    ('root ::= a "c"\na ::= a "a" |', "ac"),
    ('root ::= root* "a"', "ab"),
    ('root ::= e\ne ::= e "+" t | t\nt ::= t "*" f | f\nf ::= "(" e ")" | "1"', "1+*()"),
]


@pytest.mark.parametrize(("grammar", "alphabet"), LEFT_RECURSIVE)
def test_left_recursion_is_rewritten_into_the_same_language(grammar, alphabet):
    longest = 6 if len(alphabet) <= 3 else 5
    language = _language(grammar, longest)
    assert language
    matcher = Grammar.gbnf(grammar).matcher()
    for length in range(longest + 1):
        for letters in itertools.product(alphabet, repeat=length):
            text = "".join(letters)
            assert matcher.matches(text.encode()) == (text in language), text


def test_left_recursion_that_grows_too_large_is_refused():
    depth = 20
    rules = [f'r{i} ::= r{i + 1} "a" | r{i + 1} "b"' for i in range(depth)]
    rules.append(f'r{depth} ::= r0 "c" | "d"')
    with pytest.raises(GrammarError, match="too large"):
        Grammar.gbnf("root ::= r0\n" + "\n".join(rules))
    assert MAX_REWRITE_STEPS == 100_000


def test_hidden_left_recursion_is_still_refused():
    with pytest.raises(GrammarError, match="behind something that can match nothing"):
        Grammar.gbnf('root ::= b root "x" | "y"\nb ::= "q"?')
    with pytest.raises(GrammarError, match="the repeated tail of 'root': it repeats something"):
        Grammar.gbnf('root ::= root "a"? | "b"')


# -- token references (#318) --------------------------------------------------------------------------------------


TABLE = TokenTable(8, {"<think>": 5, "</think>": 6, "<s>": 7}.get, frozenset({7}))
TRIE = TokenTrie([b"a", b"b", b"1", b"2", b"x", b"", b"", b""])


def test_token_references_resolve_against_the_vocabulary():
    grammar = Grammar.gbnf("root ::= <think> [ab]* </think> !<[4]>", TABLE)
    matcher = grammar.matcher()
    assert TRIE.allowed(matcher, matcher.start) == [5]
    state = matcher.advance_token(matcher.start, 5, b"")
    assert TRIE.allowed(matcher, state) == [0, 1, 6]
    state = matcher.advance_token(state, 0, b"a")
    state = matcher.advance_token(state, 6, b"")
    assert TRIE.allowed(matcher, state) == [0, 1, 2, 3, 5, 6, 7]  # any one token but 4 (stop tokens are cut later)
    assert matcher.accepting(matcher.advance_token(state, 2, b"1"))
    assert matcher.advance_token(state, 4, b"x") == Matcher.DEAD
    assert uses_tokens('root ::= ("a" <[1]>)+') and uses_tokens("root ::= (!<[1]>)?")
    assert not uses_tokens('root ::= "a"')


def test_a_token_is_read_by_bytes_or_whole():
    # "a" as bytes and as the token <[0]>: both ways are kept, merged into one state.
    matcher = Grammar.gbnf('root ::= "a" "1" | <[0]> "2"', TABLE).matcher()
    state = matcher.advance_token(matcher.start, 0, b"a")
    assert TRIE.allowed(matcher, state) == [2, 3]
    assert matcher.advance_token(matcher.start, 1, b"b") == Matcher.DEAD
    assert matcher.advance_token(matcher.start, 5, b"") == matcher.start  # an unread empty token changes nothing
    assert matcher.step_token(matcher.advance(matcher.start, b"a"), 0) == Matcher.DEAD


def test_token_references_in_a_constraint():
    constraint = TokenConstraint(Grammar.gbnf("root ::= <think> !<think>{2} </think>", TABLE), TRIE)
    assert constraint.allowed() == [5] and constraint.allows(5) and not constraint.allows(0)
    constraint.accept(5)
    assert constraint.allows(7) and constraint.allows(0) and not constraint.allows(5)
    constraint.accept(0)
    constraint.accept(4)
    assert constraint.allowed() == [6] and not constraint.may_stop
    constraint.accept(6)
    assert constraint.finished and constraint.may_stop


@pytest.mark.parametrize(
    ("grammar", "message"),
    [
        ("root ::= <[8]>", "has no token <[8]>"),
        ("root ::= <nothing>", "has no token <nothing>"),
        ("root ::= <s>", "<s> is a stop token"),
        ("root ::= !x", "must be followed by a token reference"),
        ("root ::= <>", "expected a token reference"),
        ("root ::= < a>", "expected a token reference"),
        ("root ::= <think", "expected a token reference"),
        ("root ::= <[x1]>", "bad token id"),
        ("root ::= <[]>", "bad token id"),
    ],
)
def test_bad_token_references_are_refused(grammar, message):
    with pytest.raises(GrammarError, match=re.escape(message)):
        Grammar.gbnf(grammar, TABLE)


def test_a_negated_stop_token_is_fine():
    matcher = Grammar.gbnf("root ::= !<s>", TABLE).matcher()
    assert matcher.accepting(matcher.advance_token(matcher.start, 0, b"a"))


THINK_TOKENIZER = {
    **BYTE_LEVEL_TOKENIZER,
    "added_tokens": [
        *BYTE_LEVEL_TOKENIZER["added_tokens"],
        {"id": 259, "content": "<think>", "special": False, "lstrip": False, "rstrip": False, "normalized": False,
         "single_word": False},
        {"id": 260, "content": "</think>", "special": False, "lstrip": False, "rstrip": False, "normalized": False,
         "single_word": False},
    ],
}  # fmt: skip


def test_token_references_by_name_in_an_imported_model(tmp_path):
    engine = DllmEngine.from_model_file(_import(tmp_path, tokenizer=THINK_TOKENIZER, vocabulary=264))
    grammar = 'root ::= <think> [a-z]{1,8} </think> " ok"'
    generation = engine.complete_stream("Think", 40, OPTIONS, grammar=grammar)
    text = "".join(step.text for step in generation)
    tokens = generation.result().tokens
    assert re.fullmatch(r"<think>[a-z]{1,8}</think> ok", text), text
    assert tokens[0] == 259 and 260 in tokens
    with pytest.raises(GrammarError, match="stop token"):
        engine.complete_stream("x", 4, OPTIONS, grammar="root ::= <[258]>")


# -- lazy grammars (#319) -----------------------------------------------------------------------------------------


LAZY_TRIE = TokenTrie([b"a", b"<x", b">", b"1", b"2", b".", b"<x>1", b"<x>z", b"x"])
TRIGGERED = 'root ::= "<x>" [12]+ "."'


def test_a_lazy_grammar_starts_at_its_trigger():
    constraint = TokenConstraint(Grammar.gbnf(TRIGGERED), LAZY_TRIE, lazy=["<x>"])
    assert not constraint.active and constraint.may_stop and not constraint.finished
    for token in (0, 1):
        constraint.accept(token)
        assert not constraint.active
    constraint.accept(2)  # "<x" + ">": the trigger, fed to the grammar from its start
    assert constraint.active and constraint.allowed() == [3, 4] and not constraint.may_stop
    constraint.accept(3)
    assert constraint.allowed() == [3, 4, 5]
    constraint.accept(5)
    assert constraint.finished and constraint.may_stop


def test_a_trigger_inside_a_token():
    constraint = TokenConstraint(Grammar.gbnf(TRIGGERED), LAZY_TRIE, lazy=["<x>"])
    constraint.accept(0)
    constraint.accept(6)  # "<x>1"
    assert constraint.active and constraint.allowed() == [3, 4, 5]
    overshoot = TokenConstraint(Grammar.gbnf(TRIGGERED), LAZY_TRIE, lazy=["<x>"])
    overshoot.accept(7)  # "<x>z" already goes against the grammar: the rest stays free
    assert not overshoot.active
    overshoot.accept(6)
    assert not overshoot.active


def test_the_earliest_trigger_wins():
    grammar = Grammar.gbnf('root ::= ("<x>" | "x") [12] "."')
    constraint = TokenConstraint(grammar, LAZY_TRIE, lazy=["x", "<x>"])
    constraint.accept(6)  # "<x>1": "<x>" starts first
    assert constraint.allowed() == [5]
    later = TokenConstraint(grammar, LAZY_TRIE, lazy=["<x>", "x"])
    later.accept(8)
    assert later.allowed() == [3, 4]


def test_lazy_grammars_are_checked():
    with pytest.raises(GrammarError, match="cannot start with its trigger word 'y'"):
        TokenConstraint(Grammar.gbnf(TRIGGERED), LAZY_TRIE, lazy=["y"])
    with pytest.raises(ValueError, match="not both"):
        TokenConstraint(Grammar.gbnf(TRIGGERED), LAZY_TRIE, trigger="a", lazy=["<x>"])
    with pytest.raises(ValueError, match="cannot be empty"):
        TokenConstraint(Grammar.gbnf(TRIGGERED), LAZY_TRIE, lazy=[""])
    with pytest.raises(ValueError, match="only a grammar can be lazy"):
        ResponseFormat("regex", pattern="a", triggers=("a",))
    with pytest.raises(ValueError, match="cannot be empty"):
        ResponseFormat("grammar", pattern=LAZY, triggers=("",))


# -- engine, APIs, the CLI and receipts (#320) ----------------------------------------------------------------------


def _run(engine: DllmEngine, **kwargs):
    generation = engine.complete_stream("Lazy grammars", 400, OPTIONS, **kwargs)
    return "".join(step.text for step in generation), generation.result()


def test_golden_complete_grammars():
    engine = DllmEngine.create_default()
    text, result = _run(engine, grammar=EXPRESSION)
    assert re.fullmatch(r"(?:[0-9]{1,2}|x)(?:[+-](?:[0-9]{1,2}|x))*\.", text), text
    assert result.fingerprint == COMPLETE_GRAMMAR_FINGERPRINTS["left-recursive"]
    text, result = _run(engine, grammar=LAZY, grammar_triggers=["e"])
    assert re.fullmatch(r"[\s\S]*e[0-9]{2}\.", text) and result.finish_reason == "stop"
    assert result.fingerprint == COMPLETE_GRAMMAR_FINGERPRINTS["lazy"]
    assert _run(engine, grammar=LAZY, grammar_triggers=["e"])[1].tokens == result.tokens
    with pytest.raises(ValueError, match="need a grammar"):
        engine.complete_stream("x", 4, OPTIONS, grammar_triggers=["e"])


def test_lazy_grammars_through_the_apis(client):
    body = {"messages": [{"role": "user", "content": "Lazy grammars"}], "max_tokens": 400, "temperature": 0.9}
    lazy = {**body, "seed": 2, "grammar": LAZY, "grammar_lazy": True}
    lazy["grammar_triggers"] = [{"type": "word", "value": "e"}]
    first = client.post("/v1/chat/completions", json=lazy)
    assert first.status_code == 200
    plain = {**lazy, "grammar_triggers": ["e"]}
    assert client.post("/v1/chat/completions", json=plain).json()["choices"] == first.json()["choices"]
    formats = {**body, "seed": 2, "response_format": {"type": "grammar", "grammar": LAZY}, "grammar_lazy": True}
    formats["grammar_triggers"] = ["e"]
    assert client.post("/v1/chat/completions", json=formats).status_code == 200
    for bad in (
        {**body, "grammar": LAZY, "grammar_triggers": ["e"]},
        {**body, "grammar": LAZY, "grammar_lazy": True},
        {**body, "grammar_lazy": True, "grammar_triggers": ["e"]},
        {**lazy, "grammar_triggers": ["q"]},
        {**lazy, "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]},
    ):
        assert client.post("/v1/chat/completions", json=bad).status_code == 400, bad
    completion = {"prompt": "Lazy grammars", "max_tokens": 400, "temperature": 0.9, "seed": 2, "grammar": LAZY}
    text = client.post("/v1/completions", json={**completion, "grammar_lazy": True, "grammar_triggers": ["e"]}).json()
    assert text["choices"][0]["text"] == _run(DllmEngine.create_default(), grammar=LAZY, grammar_triggers=["e"])[0]
    assert client.post("/v1/completions", json={**completion, "grammar_triggers": ["e"]}).status_code == 400
    unset = {**completion, "grammar": None, "grammar_lazy": True, "grammar_triggers": ["e"]}
    assert client.post("/v1/completions", json=unset).status_code == 400
    references = {**completion, "grammar": EXPRESSION, "max_tokens": 60}
    assert client.post("/v1/completions", json=references).status_code == 200


def test_ids_of_requests_without_lazy_grammars_are_unchanged():
    assert id_payload({"messages": [], "grammar_lazy": None, "grammar_triggers": None}) == {"messages": []}


def test_receipts_record_the_triggers():
    request = ChatRequest([], 40, OPTIONS, prompt="x", response_format=ResponseFormat("grammar", pattern=LAZY))
    record = request_record(request)
    assert "triggers" not in record["response_format"]
    lazy = ChatRequest(
        [], 40, OPTIONS, prompt="x", response_format=ResponseFormat("grammar", pattern=LAZY, triggers=("e",))
    )
    record = request_record(lazy)
    assert record["response_format"]["triggers"] == ["e"]
    assert request_from_record(record).response_format == lazy.response_format


def test_complete_grammars_on_the_command_line(capsys):
    args = ["generate", "--prompt", "Lazy grammars", "--max-tokens", "400", "--temperature", "0.9", "--seed", "2"]
    assert main([*args, "--grammar", LAZY, "--grammar-trigger", "e"]) == 0
    assert COMPLETE_GRAMMAR_FINGERPRINTS["lazy"] in capsys.readouterr().err
    assert main([*args, "--grammar", EXPRESSION]) == 0
    assert COMPLETE_GRAMMAR_FINGERPRINTS["left-recursive"] in capsys.readouterr().err
    assert main(["chat", "x", "--grammar", LAZY, "--grammar-trigger", "e", "--max-tokens", "20"]) == 0
    assert main(["chat", "x", "--grammar-trigger", "e"]) == 1
    assert main(["generate", "--prompt", "x", "--grammar-trigger", "e"]) == 2
