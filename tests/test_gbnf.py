"""Phase 32: exact context-free grammars. GBNF grammars compile to the byte pushdown automaton (#221), restrict
answers in every API and the CLI (#222), and are recorded in receipts so replays reproduce them (#223)."""

from __future__ import annotations

import itertools
import re

import pytest
from fastapi.testclient import TestClient
from golden_values import GRAMMAR_FINGERPRINTS
from test_engine_import import model_path  # noqa: F401 - fixture

from etalii_dllm.chat import ChatMessage
from etalii_dllm.cli import main
from etalii_dllm.engine import ChatRequest, DllmEngine, ResponseFormat
from etalii_dllm.gbnf import _class_pattern, compile_gbnf
from etalii_dllm.grammar import Grammar, GrammarError
from etalii_dllm.receipts import request_from_record, request_record
from etalii_dllm.sampling import SamplingOptions
from etalii_dllm.server.app import app


def _strings(alphabet: str, longest: int):
    for length in range(longest + 1):
        for letters in itertools.product(alphabet, repeat=length):
            yield "".join(letters)


# -- the automaton (#221) -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("grammar", "oracle", "alphabet"),
    [
        ('root ::= "ab" | "a" "c"*', r"ab|ac*", "abc"),
        ('root ::= x{2,3} "!"\nx ::= "ab" | "a"', r"(?:ab|a){2,3}!", "ab!"),
        ('root ::= [a-b]+ ("-" [^a-b-])?', r"[ab]+(?:-[^ab\-])?", "ab-c"),
        ('root ::= ("x" | "") "y"{1,}  # trailing comment', r"x?y+", "xy"),
        ('root ::= (\n  "a" |\n  "b"\n)? "c"{0}', r"[ab]?", "abc"),
        ('root ::= "a" {2} "b" { 1 , 2 }', r"aab{1,2}", "ab"),
        ('root ::= . "\\x41\\u0042\\U00000043"', r"(?s:.)ABC", "ABC\n"),
        ('root ::= ["\\\\\\]\\-] [\\[]', r'["\\\]\-]\[', '"\\]-['),
        ('root ::= "\\n\\r\\t" "\\""', '\n\r\t"', '\n\r\t"'),
        ('root ::= a\na ::= b "!"\nb ::= "x"*', r"x*!", "x!"),
        ('root ::= ""', "", "a"),
    ],
)
def test_grammars_match_like_an_independent_oracle(grammar, oracle, alphabet):
    matcher = Grammar.gbnf(grammar).matcher()
    for text in _strings(alphabet, 5):
        assert matcher.matches(text.encode()) == bool(re.fullmatch(oracle, text)), text


def _balanced(text: str) -> bool:
    depth = 0
    for char in text:
        depth += 1 if char == "(" else -1
        if depth < 0:
            return False
    return depth == 0


def _arithmetic(text: str) -> bool:
    position = 0

    def expression() -> bool:
        nonlocal position
        if not term():
            return False
        while position < len(text) and text[position] in "+-":
            position += 1
            if not term():
                return False
        return True

    def term() -> bool:
        nonlocal position
        if position < len(text) and text[position] == "(":
            position += 1
            if not expression() or position >= len(text) or text[position] != ")":
                return False
            position += 1
            return True
        start = position
        while position < len(text) and text[position].isdigit():
            position += 1
        return position > start

    return expression() and position == len(text)


ARITHMETIC = """
root   ::= expr
expr   ::= term (("+" | "-") term)*
term   ::= [0-9]+ | "(" expr ")"
"""


def test_recursion():
    balanced = Grammar.gbnf('root ::= ("(" root ")")*').matcher()
    for text in _strings("()", 10):
        assert balanced.matches(text.encode()) == _balanced(text), text
    arithmetic = Grammar.gbnf(ARITHMETIC).matcher()
    for text in _strings("1+()", 7):
        assert arithmetic.matches(text.encode()) == _arithmetic(text), text
    deep = "(" * 300 + "1" + ")" * 300
    assert arithmetic.matches(deep.encode()) and not arithmetic.matches(deep[:-1].encode())


def test_characters_are_code_points_in_utf8():
    matcher = Grammar.gbnf('root ::= [^"]{2} "é" [\\U0001F600-\\U0001F64F] .').matcher()
    assert matcher.matches("a中é\U0001f602\n".encode())
    assert not matcher.matches('a"é\U0001f602x'.encode()) and not matcher.matches("a中é\U0001f650x".encode())
    assert not matcher.matches(b"ab\xc3\xa9\xf0\x9f\x98\x82\xff")  # never a byte that is not UTF-8
    assert _class_pattern(((0x41, 0x41), (0x1F600, 0x1F64F))) == "[\\u0041\U0001f600-\U0001f64f]"


@pytest.mark.parametrize(
    ("grammar", "message"),
    [
        ("", "has no rules"),
        ('a ::= "x"', "no 'root' rule"),
        ("root ::= b", "undefined rule 'b'"),
        ('root ::= "a"\nroot ::= "b"', "line 2, column 1: rule 'root' is defined twice"),
        ('root ::= root "x" | "y"', "left recursion is not supported: root -> root"),
        ('root ::= a "x" | "y"\na ::= "z"? root', "left recursion is not supported: root -> a -> root"),
        ('root ::= ("x"?)*', "it repeats something that can match nothing"),
        ('root ::= "x" root', "derives no text"),
        ("root ::= <think>", "token references"),
        ('root ::= "\\q"', "unsupported escape"),
        ('root ::= "\\uD800"', "not a Unicode scalar value"),
        ('root ::= "\\xZZ"', "bad \\x escape"),
        ('root ::= "abc', "unterminated string"),
        ("root ::= [abc", "unterminated character class"),
        ("root ::= [z-a]", "runs backwards"),
        ("root ::= [^\\x00-\\U0010FFFF]", "matches no character"),
        ('root ::= ("a"', "missing ')'"),
        ('root ::= "a"{2', "missing '}'"),
        ('root ::= "a"{3,2}', "maximum is smaller"),
        ('root ::= "a"{x}', "expected a repetition count"),
        ('root ::= "a"{1001}', "limited to 1000"),
        ("root ::= *", "nothing to repeat"),
        ("root ::= )", "unexpected ')'"),
        ("root = x", "expected '::='"),
        ('::= "x"', "expected a rule name"),
        ('root ::= "\ud800"', "surrogate"),
    ],
)
def test_unsupported_grammars_are_refused(grammar, message):
    compile_gbnf.cache_clear()
    with pytest.raises(GrammarError, match=re.escape(message)):
        Grammar.gbnf(grammar)


def test_alternatives_that_cannot_finish_are_dropped():
    matcher = Grammar.gbnf('root ::= "a" | "b" loop\nloop ::= "c" loop').matcher()
    assert matcher.matches(b"a")
    assert matcher.step(matcher.start, ord("b")) == matcher.DEAD


def _walk(matcher, choose) -> bytes:
    """Follows the bytes ``choose`` picks for a while, then the shortest way to a complete match (a breadth-first
    search): every state reached must allow a byte or be complete, and must still be able to finish."""
    state, data = matcher.start, b""
    for _ in range(40):
        allowed = [b for b in range(256) if matcher.step(state, b) != matcher.DEAD]
        assert allowed or matcher.accepting(state), "a state that cannot finish must allow a byte"
        if not allowed:
            return data
        byte = choose(allowed, len(data))
        data += bytes([byte])
        state = matcher.step(state, byte)
    queue, seen = [(state, data)], {state}
    for current, text in queue:
        if matcher.accepting(current):
            return text
        for byte in range(256):
            following = matcher.step(current, byte)
            if following != matcher.DEAD and following not in seen:
                seen.add(following)
                queue.append((following, text + bytes([byte])))
    raise AssertionError("no way to finish")


@pytest.mark.parametrize("grammar", [ARITHMETIC, 'root ::= ("(" root ")")* "."', 'root ::= [^a]{3,5} "é"'])
def test_no_dead_ends(grammar):
    matcher = Grammar.gbnf(grammar).matcher()
    for pick in (min, max, lambda allowed, n: allowed[(n * 7) % len(allowed)]):
        data = _walk(matcher, lambda allowed, n, pick=pick: pick(allowed) if pick in (min, max) else pick(allowed, n))
        assert matcher.matches(data)


# -- answers (#222, #223) -----------------------------------------------------------------------------------------

LIST = """
root ::= "Colours: " colour (", " colour){1,3} "."
colour ::= "red" | "green" | "blue" | "yellow"
"""
_LIST = re.compile(r"Colours: (red|green|blue|yellow)(, (red|green|blue|yellow)){1,3}\.")
SUM = 'root ::= [0-9]{1,3} (" + " [0-9]{1,3}){0,2}'
_SUM = re.compile(r"[0-9]{1,3}( \+ [0-9]{1,3}){0,2}")


@pytest.fixture(scope="module")
def tiny(model_path) -> DllmEngine:  # noqa: F811
    return DllmEngine.from_model_file(model_path)


def test_answers_follow_the_grammar(tiny):
    messages = [ChatMessage("user", "Name some colours.")]
    answers = []
    for seed in range(5):
        request = ChatRequest(messages, 60, SamplingOptions(temperature=1.0, seed=seed),
                              response_format=ResponseFormat("grammar", pattern=LIST))  # fmt: skip
        result = tiny.chat_completion(request)
        assert result.finish_reason == "stop" and _LIST.fullmatch(result.content), result.content
        answers.append(result)
    assert answers[0].fingerprint == GRAMMAR_FINGERPRINTS["colours"]
    record = request_record(request)
    assert record["response_format"] == {"type": "grammar", "schema": None, "pattern": LIST}
    assert request_from_record(record) == request
    generation = tiny.complete_stream("Go: ", 40, SamplingOptions(temperature=1.0), grammar=SUM)
    streamed = "".join(step.text for step in generation)
    assert _SUM.fullmatch(streamed) and generation.result().text == streamed, streamed
    with pytest.raises(ValueError, match="cannot be combined"):
        tiny.complete_stream("x", 4, SamplingOptions(), regex="a", grammar=ARITHMETIC)


def test_response_formats_take_a_grammar():
    assert ResponseFormat("grammar", pattern=LIST).grammar() is not None
    with pytest.raises(ValueError, match="a grammar response format needs a pattern"):
        ResponseFormat("grammar")
    with pytest.raises(ValueError, match="only regex and grammar"):
        ResponseFormat("json_object", pattern=LIST)


def test_grammars_through_the_apis():
    client = TestClient(app)
    body = {"messages": [{"role": "user", "content": "Colours?"}], "max_tokens": 60, "temperature": 0.8, "seed": 2}
    a = client.post("/v1/chat/completions", json={**body, "grammar": LIST}).json()
    b = client.post("/v1/chat/completions", json={**body, "response_format": {"type": "grammar", "grammar": LIST}})
    assert _LIST.fullmatch(a["choices"][0]["message"]["content"])
    assert b.json()["choices"][0]["message"] == a["choices"][0]["message"]
    plain = client.post("/v1/chat/completions", json=body).json()
    assert plain["id"] != a["id"]
    refused = [
        {**body, "grammar": LIST, "response_format": {"type": "json_object"}},
        {**body, "grammar": LIST, "guided_regex": "a"},
        {**body, "response_format": {"type": "grammar"}},
        {**body, "grammar": "root ::= root"},
        {**body, "grammar": LIST, "beam": {"width": 2}},
    ]
    for request in refused:
        assert client.post("/v1/chat/completions", json=request).status_code == 400, request
    receipt = client.post("/v1/chat/completions", json={**body, "grammar": LIST, "receipt": True}).json()["receipt"]
    assert client.post("/v1/receipts/verify", json=receipt).json()["ok"]
    completion = client.post("/v1/completions", json={"prompt": "Sum: ", "max_tokens": 30, "grammar": SUM})
    assert _SUM.fullmatch(completion.json()["choices"][0]["text"])
    both = {"prompt": "x", "max_tokens": 4, "grammar": ARITHMETIC, "guided_regex": "a"}
    assert client.post("/v1/completions", json=both).status_code == 400


def test_grammars_on_the_command_line(tmp_path, capsys):
    path = tmp_path / "colours.gbnf"
    path.write_text(LIST, encoding="utf-8")
    assert main(["chat", "Colours?", "--grammar", str(path), "--max-tokens", "60"]) == 0
    assert _LIST.fullmatch(capsys.readouterr().out.strip())
    assert main(["generate", "--prompt", "Sum: ", "--grammar", SUM, "--max-tokens", "30"]) == 0
    assert _SUM.fullmatch(capsys.readouterr().out.strip())
    assert main(["chat", "x", "--grammar", LIST, "--json"]) == 1
    assert main(["generate", "--prompt", "x", "--grammar", str(tmp_path / "missing.gbnf")]) == 2
    assert main(["generate", "--prompt", "x", "--grammar", "root ::= <x>"]) == 2
    assert main(["generate", "--prompt", "x", "--grammar", LIST, "--beams", "2"]) == 2
