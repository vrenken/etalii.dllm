"""Phase 28: reproducible scoring and voting. Prompt scores equal the logprobs generation reports, the completions
API equals ``dllm generate`` and echoes exact prompt scores, votes follow a fixed rule, and score and vote receipts
replay."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from golden_values import SCORING_FINGERPRINTS
from test_engine_import import model_path  # noqa: F401 - fixture

from etalii_dllm import engine as engine_module
from etalii_dllm import reference, scoring, verify, voting
from etalii_dllm.chat import ChatMessage
from etalii_dllm.cli import main
from etalii_dllm.engine import ChatRequest, DllmEngine, default_engine
from etalii_dllm.generation import ContextLengthError
from etalii_dllm.numerics import sum_
from etalii_dllm.sampling import SamplingOptions
from etalii_dllm.server.app import app

TEXT = "Once upon a time there was a little robot."
STORY = ChatRequest([ChatMessage("user", "Tell me a story.")], 8, SamplingOptions(temperature=1.2, seed=5))


@pytest.fixture(scope="module")
def tiny(model_path) -> DllmEngine:  # noqa: F811
    return DllmEngine.from_model_file(model_path)


@pytest.fixture
def served(model_path, monkeypatch):  # noqa: F811
    for name in dir(engine_module):  # earlier tests leave runtime settings (an auditor, a response cache) behind
        if name.endswith("_ENVIRONMENT_VARIABLE"):
            monkeypatch.delenv(getattr(engine_module, name), raising=False)
    monkeypatch.setenv(engine_module.MODEL_ENVIRONMENT_VARIABLE, str(model_path))
    default_engine.cache_clear()
    yield default_engine()
    default_engine.cache_clear()


# Scoring


def test_scores_equal_the_logprobs_of_generation(tiny):
    prompt = "Once upon a time"
    generated = tiny.chat_completion(ChatRequest([], 6, top_logprobs=3, prompt=prompt))
    tokens = [*tiny.tokenizer.encode(prompt), *(entry.token for entry in generated.logprobs)]
    score = scoring.score_tokens(tiny, tokens, 3)
    assert score.tokens[0].logprob is None and score.tokens[0].top == ()
    for entry, scored in zip(generated.logprobs, score.tokens[-len(generated.logprobs) :], strict=True):
        assert (scored.token, scored.logprob, scored.top) == (entry.token, entry.logprob, entry.top)
    values = np.asarray([t.logprob for t in score.tokens[1:]], dtype=np.float32)
    assert score.scored == len(tokens) - 1 and score.log_likelihood == sum_(values)
    assert score.perplexity is not None and score.perplexity > 1
    assert (
        scoring.score_text(tiny, prompt).fingerprint
        == scoring.score_tokens(tiny, tokens[: -len(generated.logprobs)]).fingerprint
    )


def test_score_edges(tiny):
    empty = scoring.score_text(tiny, "")
    assert (empty.scored, empty.log_likelihood, empty.perplexity) == (0, 0.0, None)
    one = scoring.score_tokens(tiny, [5])
    assert (one.scored, one.perplexity) == (0, None)
    with pytest.raises(ValueError, match="between 0 and 20"):
        scoring.score_text(tiny, "Hi", 21)
    with pytest.raises(ContextLengthError):
        scoring.score_tokens(tiny, [5] * (tiny.model.config.context_length + 1))


def test_score_is_golden_and_replays(tiny):
    score = scoring.score_text(tiny, TEXT, 3)
    assert score.fingerprint == SCORING_FINGERPRINTS["score"]
    receipt = scoring.record(tiny, TEXT, 3, score)
    assert receipt["id"].startswith("score_") and receipt["output"]["fingerprint"] == score.fingerprint
    assert scoring.verify(tiny, receipt).ok
    edited = {**receipt, "text": TEXT + "!"}
    outcome = scoring.verify(tiny, edited)
    assert not outcome.ok and outcome.reasons[0].startswith("the receipt was edited")
    assert any(r.startswith("the log-probabilities differ") for r in outcome.reasons)
    other = {**receipt, "system_fingerprint": "fp_other", "engine": "0.0.1"}
    outcome = scoring.verify(tiny, other)
    assert any("different weights" in r for r in outcome.reasons) and outcome.notes
    with pytest.raises(ValueError, match="not a dllm-score/1"):
        scoring.verify(tiny, {"score": "x"})
    as_json = score.to_json(tiny)
    assert as_json["tokens"][0]["logprob"] is None and len(as_json["tokens"][1]["top_logprobs"]) == 3


# Voting


def test_normalize_and_tally():
    assert voting.normalize("  The  ANSWER\tis\n42 ") == "the answer is 42"
    assert voting.normalize("\uff46\uff55\uff4c\uff4c \uff57\uff49\uff44\uff54\uff48") == "full width"  # NFKC
    assert voting.normalize("   ") is None
    assert voting.normalize("so 3, then 41 and 42.", r"\d+") == "42"
    assert voting.normalize("x = 7; answer: 9", r"answer: (\d+)") == "9"
    assert voting.normalize("no number", r"\d+") is None
    assert voting.normalize("answer: ", r"answer: (\d*)") is None
    ballots, winner = voting.tally(["b", "a", None, "a", "b", "c"])
    assert [(b.answer, b.choices) for b in ballots] == [("b", (0, 4)), ("a", (1, 3)), ("c", (5,))]
    assert winner == 0 and ballots[0].to_json() == {"answer": "b", "votes": 2, "choices": [0, 4]}
    assert voting.tally([None, None]) == ((), 0)
    with pytest.raises(ValueError, match="between 1 and 16"):
        voting.validate(17, None)
    with pytest.raises(ValueError, match="not a valid regex"):
        voting.validate(3, "(")


def test_vote_is_golden_and_replays(tiny):
    outcome = voting.vote(tiny, STORY, 5)
    choices = tiny.chat_choices(STORY, 5)
    assert [r.content for r in outcome.results] == [r.content for r in choices]
    assert outcome.answers == tuple(voting.normalize(r.content) for r in choices)
    assert (
        outcome.result.content == choices[outcome.winner].content and outcome.answer == outcome.answers[outcome.winner]
    )
    assert outcome.to_json()["n"] == 5
    receipt = voting.record(tiny, STORY, outcome)
    digest = hashlib.sha256(json.dumps(receipt["output"], sort_keys=True).encode()).hexdigest()
    assert digest == SCORING_FINGERPRINTS["vote"]
    assert voting.verify(tiny, receipt).ok
    forged = {
        **receipt,
        "output": {**receipt["output"], "winner": 4, "receipts": ["x", *receipt["output"]["receipts"][1:]]},
    }
    outcome = voting.verify(tiny, {**forged, "engine": "0.0.1", "system_fingerprint": "fp_other"})
    reasons = " | ".join(outcome.reasons)
    assert "edited" in reasons and "answer 0 differs" in reasons and "the winner differ" in reasons
    assert "different weights" in reasons and outcome.notes
    with pytest.raises(ValueError, match="not a dllm-vote/1"):
        voting.verify(tiny, {"vote": "x"})


# The completions API


def test_completions_equal_generate_and_echo_exact_scores(served):
    client = TestClient(app)
    body = {"prompt": "Once upon a time", "max_tokens": 8, "temperature": 0.8, "seed": 3}
    response = client.post("/v1/completions", json=body).json()
    expected = served.complete("Once upon a time", 8, SamplingOptions(temperature=0.8, seed=3))
    assert response["object"] == "text_completion" and response["id"].startswith("cmpl-")
    assert response["choices"][0]["text"] == expected.text and response["created"] == 0
    assert response == client.post("/v1/completions", json=body).json()
    assert response["usage"]["completion_tokens"] == len(expected.tokens)

    echoed = client.post("/v1/completions", json={**body, "echo": True, "logprobs": 2}).json()["choices"][0]
    score = scoring.score_text(served, "Once upon a time", 2)
    prompt_tokens = len(score.tokens)
    assert echoed["text"] == "Once upon a time" + expected.text
    logprobs = echoed["logprobs"]
    assert logprobs["token_logprobs"][:prompt_tokens] == [t.logprob for t in score.tokens]
    assert logprobs["top_logprobs"][0] is None and len(logprobs["top_logprobs"][1]) <= 2
    assert logprobs["text_offset"][0] == 0 and len(logprobs["tokens"]) == prompt_tokens + len(expected.tokens)

    scored = client.post("/v1/completions", json={"prompt": TEXT, "max_tokens": 0, "echo": True, "logprobs": 0})
    choice = scored.json()["choices"][0]
    assert choice["text"] == TEXT and choice["finish_reason"] == "length"
    assert choice["logprobs"]["token_logprobs"] == [t.logprob for t in scoring.score_text(served, TEXT).tokens]

    several = client.post("/v1/completions", json={**body, "prompt": ["Hi", "Bye"], "n": 2, "receipt": True}).json()
    assert [c["index"] for c in several["choices"]] == [0, 1, 2, 3]
    assert several["choices"][2]["text"] == served.complete("Bye", 8, SamplingOptions(temperature=0.8, seed=3)).text
    assert several["choices"][3]["receipt"]["request"]["options"]["seed"] == 4


def test_streamed_completions_equal_the_response(served):
    client = TestClient(app)
    body = {"prompt": "Once upon a time", "max_tokens": 6, "temperature": 0.8, "seed": 3, "echo": True,
            "logprobs": 1, "n": 2, "receipt": True}  # fmt: skip
    whole = client.post("/v1/completions", json=body).json()
    lines = client.post("/v1/completions", json={**body, "stream": True, "stream_options": {"include_usage": True}})
    events = [line[len("data: ") :] for line in lines.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    for index in (0, 1):
        parts = [c["choices"][0] for c in chunks if c["choices"] and c["choices"][0]["index"] == index]
        assert "".join(p["text"] for p in parts) == whole["choices"][index]["text"]
        tokens = [t for p in parts if p.get("logprobs") for t in p["logprobs"]["token_logprobs"]]
        assert tokens == whole["choices"][index]["logprobs"]["token_logprobs"]
        assert parts[-1]["finish_reason"] == whole["choices"][index]["finish_reason"]
        assert parts[-1]["receipt"] == whole["choices"][index]["receipt"]
    assert chunks[-1]["usage"] == whole["usage"] and all(c["id"] == whole["id"] for c in chunks)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"suffix": "end"}, "suffix"),
        ({"best_of": 3}, "best_of"),
        ({"n": 17}, "n must be"),
        ({"logprobs": 21}, "logprobs"),
        ({"max_tokens": -1}, "max_tokens"),
        ({"prompt": []}, "must not be empty"),
        ({"stop": ["a", "b", "c", "d", "e"]}, "stop sequences"),
        ({"prompt": "x " * 5000}, "context window"),
        ({"prompt": "x " * 5000, "stream": True}, "context window"),
    ],
)
def test_completions_errors(served, change, message):
    response = TestClient(app).post("/v1/completions", json={"prompt": "Hi", **change})
    assert response.status_code == 400 and message in response.json()["error"]["message"]


def test_completions_take_the_decoding_controls(served):
    client = TestClient(app)
    body = {"prompt": "Hi", "max_tokens": 6, "temperature": 0.9, "seed": 2, "top_k": 5, "top_p": 0.9, "min_p": 0.05,
            "repetition_penalty": 1.1, "repeat_last_n": 8, "frequency_penalty": 0.2, "presence_penalty": 0.1,
            "logit_bias": {"5": 1.0}, "stop": "\n", "guided_regex": "[a-z ]+", "watermark": {"key": "k"}}  # fmt: skip
    response = client.post("/v1/completions", json=body).json()
    options = SamplingOptions(temperature=0.9, seed=2, top_k=5, top_p=0.9, min_p=0.05, repetition_penalty=1.1,
                              repeat_last_n=8, frequency_penalty=0.2, presence_penalty=0.1, logit_bias=((5, 1.0),),
                              watermark_key="k")  # fmt: skip
    request = ChatRequest([], 6, options, stop=["\n"], prompt="Hi", request_id=response["id"],
                          response_format=engine_module.ResponseFormat("regex", pattern="[a-z ]+"))  # fmt: skip
    assert response["choices"][0]["text"] == served.chat_completion(request).content


# Voting over the API and the command line


def test_vote_over_the_api(served):
    client = TestClient(app)
    body = {"messages": [{"role": "user", "content": "Tell me a story."}], "max_tokens": 8, "temperature": 1.2,
            "seed": 5}  # fmt: skip
    plain = client.post("/v1/chat/completions", json=body).json()
    voted = client.post("/v1/chat/completions", json={**body, "vote": {"n": 5}, "receipt": True}).json()
    request = ChatRequest(STORY.messages, 8, STORY.options, request_id=voted["id"])
    outcome = voting.vote(served, request, 5)
    assert voted["id"] != plain["id"] and voted["vote"] == outcome.to_json()
    assert voted["choices"][0]["message"]["content"] == outcome.result.content
    assert voted["usage"]["completion_tokens"] == sum(r.completion_tokens for r in outcome.results)
    assert voted["receipt"]["vote"] == voting.FORMAT
    assert client.post("/v1/receipts/verify", json=voted["receipt"]).json()["ok"]
    for extra in ({"stream": True}, {"n": 2}):
        response = client.post("/v1/chat/completions", json={**body, "vote": {"n": 3}, **extra})
        assert response.status_code == 400 and "vote cannot be combined" in response.json()["error"]["message"]
    logprobs = client.post("/v1/chat/completions", json={**body, "vote": {"n": 2}, "logprobs": True}).json()
    assert logprobs["choices"][0]["logprobs"]["content"]


def test_score_and_vote_on_the_command_line(served, tmp_path, capsys):
    text_file = tmp_path / "story.txt"
    text_file.write_text(TEXT, encoding="utf-8")
    receipt_file = tmp_path / "score.json"
    assert main(["score", str(text_file), "--top", "2", "--json", "--receipt", str(receipt_file)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["fingerprint"] == scoring.score_text(served, TEXT, 2).fingerprint
    assert printed["receipt"] == json.loads(receipt_file.read_text(encoding="utf-8"))
    assert main(["score", str(text_file)]) == 0
    out = capsys.readouterr().out
    assert "log-likelihood:" in out and "perplexity:" in out
    assert main(["score", str(tmp_path / "missing.txt")]) == 2
    assert capsys.readouterr().err.startswith("dllm score: ")
    assert main(["replay", str(receipt_file)]) == 0
    assert "verified" in capsys.readouterr().out

    vote_file = tmp_path / "vote.json"
    args = ["chat", "Tell me a story.", "--max-tokens", "8", "--temperature", "1.2", "--seed", "5", "--vote", "5"]
    assert main([*args, "--receipt", str(vote_file)]) == 0
    captured = capsys.readouterr()
    assert captured.out == voting.vote(served, STORY, 5).result.content + "\n" and "winner: choice" in captured.err
    assert main(["replay", str(vote_file), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]
    assert main([*args, "--vote-extract", "("]) == 1
    assert "not a valid regex" in capsys.readouterr().err


def test_the_reference_check_scores_the_prompt(tiny, monkeypatch):
    assert verify.check_reference(tiny, max_tokens=2).results["scored"] == "equal"
    original = reference.log_softmax

    def off_by_one_bit(logits):
        values = original(logits).copy()
        values.view(np.uint32)[:] ^= 1
        return values

    monkeypatch.setattr(reference, "log_softmax", off_by_one_bit)
    assert verify.check_reference(tiny, max_tokens=2).results["scored"].endswith("first at token 1")
