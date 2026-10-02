"""Phase 36: exact token healing. A raw prompt (#242) or a chat prefill (#243) that ends inside a word has its last
token taken back, and the answer is constrained to start with that token's bytes; healing is recorded in receipts,
replays, the specification and the reference implementation (#244), with golden values (#245)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from golden_values import HEALING_FINGERPRINTS
from test_cli import isolated_environment  # noqa: F401 - autouse fixture: the CLI test sets DLLM_MODEL
from test_engine_import import model_path  # noqa: F401 - fixture

from etalii_dllm import receipts
from etalii_dllm.chat import ChatMessage
from etalii_dllm.engine import ChatRequest, DllmEngine, ResponseFormat
from etalii_dllm.grammar import Grammar, HealingConstraint, TokenConstraint, TokenTrie
from etalii_dllm.sampling import SamplingOptions

SAMPLED = SamplingOptions(temperature=0.9, seed=11)
PROMPT = "The quick brown fo"


@pytest.fixture(scope="module")
def tiny(model_path) -> DllmEngine:  # noqa: F811
    return DllmEngine.from_model_file(model_path)


# The constraint


def test_healing_constraint_allows_what_fits_the_prefix():
    trie = TokenTrie([b"a", b"ab", b"abc", b"b", b"", b"bcd", b"x", b"abx"])
    healing = HealingConstraint(b"ab", trie)
    assert healing.active and not healing.may_stop and not healing.finished
    assert healing.allowed() == [0, 1, 2, 7]
    assert healing.allows(2) and not healing.allows(3) and not healing.allows(4)
    healing.accept(0)
    assert healing.allowed() == [3, 5]
    healing.accept(5)
    assert not healing.active and healing.may_stop and not healing.finished
    with pytest.raises(ValueError, match="needs the bytes"):
        HealingConstraint(b"", trie)


def test_healing_constraint_hands_over_to_the_inner_constraint():
    trie = TokenTrie([b"a", b"ab", b"abc", b"abx", b"c", b"x"])
    inner = TokenConstraint(Grammar.regex("c+"), trie)
    healing = HealingConstraint(b"ab", trie, inner)
    assert healing.allowed() == [0, 1, 2]  # "abx" leaves the regex
    healing.accept(2)  # "ab" written, "c" goes to the regex
    assert healing.active and healing.may_stop and healing.allowed() == [4]
    assert healing.allows(4) and not healing.allows(5)
    healing.accept(4)
    assert healing.may_stop and not healing.finished
    done = HealingConstraint(b"a", trie, TokenConstraint(Grammar.literal("b"), trie))
    done.accept(1)
    assert done.finished
    lazy = TokenConstraint(Grammar.literal("x"), trie, trigger="<")
    assert lazy.allows_bytes(b"anything")
    waiting = HealingConstraint(b"a", trie, lazy)
    waiting.accept(0)
    assert not waiting.active and waiting.may_stop


# Raw prompts


def test_healed_completion_starts_with_the_taken_back_token(tiny):
    tokens = tiny.tokenizer.encode(PROMPT)
    taken = tiny.tokenizer.decode_bytes(tokens[-1:])
    plain = tiny.complete_stream(PROMPT, 12, SAMPLED).result()
    healed = tiny.complete_stream(PROMPT, 12, SAMPLED, token_healing=True).result()
    assert healed.prompt_tokens == len(tokens) - 1 and plain.prompt_tokens == len(tokens)
    written = tiny.tokenizer.decode_bytes(healed.tokens)
    assert written.startswith(taken) and written[len(taken) :].decode("utf-8", "replace") == healed.text
    again = tiny.complete_stream(PROMPT, 12, SAMPLED, token_healing=True).result()
    assert again.fingerprint == healed.fingerprint == HEALING_FINGERPRINTS["raw"]
    streamed = "".join(step.text for step in tiny.complete_stream(PROMPT, 12, SAMPLED, token_healing=True))
    assert streamed == healed.text


def test_prompts_without_a_healable_token_are_left_alone(tiny):
    assert tiny.heal("", None) == ("", None, 0)
    special = "Hi<|im_end|>"
    assert tiny.tokenizer.decode_bytes(tiny.tokenizer.encode(special)[-1:]) == b""
    assert tiny.heal(special, None) == (special, None, 0)
    plain = tiny.complete_stream(special, 6, SAMPLED).result()
    assert tiny.complete_stream(special, 6, SAMPLED, token_healing=True).result().fingerprint == plain.fingerprint


def test_healing_under_a_regex_and_a_grammar(tiny):
    regex = tiny.complete_stream("Count: 1", 8, SAMPLED, regex=r"[0-9]{1,4}", token_healing=True).result()
    assert regex.text.isdigit() and len(regex.text) <= 4
    written = tiny.tokenizer.decode_bytes(regex.tokens)
    assert written.startswith(b"1")
    grammar = tiny.complete_stream("Yes or n", 8, SAMPLED, grammar='root ::= "o" | "ever"', token_healing=True)
    assert grammar.result().text in ("o", "ever")


# Chats


def test_healed_chat_prefill(tiny):
    messages = [ChatMessage("user", "Name a colour."), ChatMessage("assistant", "My favourite colour is gre")]
    request = ChatRequest(messages, 10, SAMPLED, token_healing=True)
    result = tiny.chat_completion(request)
    assert result.fingerprint == HEALING_FINGERPRINTS["prefill"]
    prompt = tiny.render_chat(messages)  # the prefill is the prompt's end: healing it is healing the raw prompt
    assert tiny.complete_stream(prompt, 10, SAMPLED, token_healing=True).result().text == result.content
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    structured = ChatRequest([ChatMessage("user", "A number")], 30, SAMPLED,
                             response_format=ResponseFormat("json_schema", schema), token_healing=True)  # fmt: skip
    assert "n" in json.loads(tiny.chat_completion(structured).content)


def test_receipts_record_healing_and_replay_it(tiny):
    request = ChatRequest([], 8, SAMPLED, prompt=PROMPT, token_healing=True)
    record = receipts.request_record(request)
    assert record["token_healing"] is True and receipts.request_from_record(record) == request
    assert "token_healing" not in receipts.request_record(ChatRequest([], 8, SAMPLED, prompt=PROMPT))
    result = tiny.chat_completion(request)
    assert result.receipt is not None and receipts.verify(tiny, result.receipt).ok


def test_beam_search_refuses_healing(tiny):
    from etalii_dllm import beam

    with pytest.raises(ValueError, match="cannot heal"):
        beam.search(tiny, ChatRequest([], 4, prompt=PROMPT, token_healing=True), 2)


# Front ends


def test_every_api_takes_token_healing(tiny):
    from etalii_dllm.engine import default_engine
    from etalii_dllm.server.app import app

    app.dependency_overrides[default_engine] = lambda: tiny
    try:
        client = TestClient(app)
        body = {"prompt": PROMPT, "max_tokens": 12, "temperature": 0.9, "seed": 11}
        plain = client.post("/v1/completions", json=body).json()
        healed = client.post("/v1/completions", json={**body, "token_healing": True}).json()
        expected = tiny.complete_stream(PROMPT, 12, SAMPLED, token_healing=True).result()
        assert healed["choices"][0]["text"] == expected.text
        assert healed["usage"]["prompt_tokens"] == plain["usage"]["prompt_tokens"] - 1
        assert client.post("/v1/completions", json={**body, "token_healing": None}).json()["id"] == plain["id"]
        assert healed["id"] != plain["id"]
        prefill = [{"role": "user", "content": "Name a colour."},
                   {"role": "assistant", "content": "My favourite colour is gre"}]  # fmt: skip
        chat = client.post("/v1/chat/completions", json={"messages": prefill, "max_tokens": 10, "temperature": 0.9,
                                                         "seed": 11, "token_healing": True}).json()  # fmt: skip
        reference = tiny.chat_completion(ChatRequest([ChatMessage(**m) for m in prefill], 10, SAMPLED,
                                                     token_healing=True))  # fmt: skip
        assert chat["choices"][0]["message"]["content"] == reference.content
        anthropic = client.post("/v1/messages", json={"messages": prefill, "max_tokens": 10, "temperature": 0.9,
                                                      "seed": 11, "token_healing": True}).json()  # fmt: skip
        assert anthropic["content"][0]["text"] == reference.content
        responses = client.post("/v1/responses", json={"input": prefill, "max_output_tokens": 10, "temperature": 0.9,
                                                       "seed": 11, "token_healing": True}).json()  # fmt: skip
        assert responses["output"][0]["content"][0]["text"] == reference.content
        ollama = client.post(
            "/api/generate",
            json={
                "prompt": PROMPT,
                "raw": True,
                "stream": False,
                "token_healing": True,
                "options": {"num_predict": 12, "temperature": 0.9, "seed": 11},
            },
        )
        assert ollama.json()["response"] == expected.text
    finally:
        app.dependency_overrides.clear()


def test_cli_token_healing(model_path, capsys):  # noqa: F811
    from etalii_dllm.cli import main

    args = ["--model", str(model_path)]
    assert main([*args, "generate", "--prompt", PROMPT, "--max-tokens", "12", "--temperature", "0.9", "--seed", "11",
                 "--token-healing"]) == 0  # fmt: skip
    assert HEALING_FINGERPRINTS["raw"] in capsys.readouterr().err
    assert main([*args, "chat", "Name a colour.", "--prefill", "My favourite colour is gre", "--max-tokens", "10",
                 "--temperature", "0.9", "--seed", "11", "--token-healing"]) == 0  # fmt: skip
    assert main([*args, "generate", "--prompt", PROMPT, "--beams", "2", "--token-healing"]) == 2
