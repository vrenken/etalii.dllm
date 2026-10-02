"""Phase 30: exact beam search. Hypotheses are ranked by a total order, scores follow the documented double
arithmetic, the independent reference implementation finds the same answers, receipts replay and every front end
gives the same answers."""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from golden_values import BEAM_FINGERPRINTS
from test_engine_import import model_path  # noqa: F401 - fixture

from etalii_dllm import _kernels, beam, reference, verify
from etalii_dllm import engine as engine_module
from etalii_dllm.chat import ChatMessage
from etalii_dllm.cli import main
from etalii_dllm.engine import ChatRequest, DllmEngine, ResponseFormat, default_engine
from etalii_dllm.generation import ContextLengthError
from etalii_dllm.sampling import GREEDY, SamplingOptions
from etalii_dllm.server.app import app
from etalii_dllm.tools import Tool

PROMPT = "Once upon a time"
STORY = [ChatMessage("user", "Tell me a story.")]


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


def _rows(found):
    return [(list(h.tokens), h.finish_reason, h.log_likelihood.hex(), h.score.hex()) for h in found]


# The search


def test_scores_use_the_portable_power():
    assert beam.score(-6.0, 3, 1.0) == -6.0 / float(_kernels.exp(1.0 * float(_kernels.log(3.0))))
    assert beam.score(-6.0, 3, 0.0) == -6.0 and beam.score(-1.5, 0, 1.0) == -1.5
    assert float(reference.exp(0.7 * float(reference.log(5.0)))) == float(_kernels.exp(0.7 * float(_kernels.log(5.0))))


def test_settings_are_validated():
    for width, n_best, penalty, message in ((0, 1, 1.0, "width"), (17, 1, 1.0, "width"), (2, 3, 1.0, "n_best"),
                                            (2, 0, 1.0, "n_best"), (2, 1, float("nan"), "finite")):  # fmt: skip
        with pytest.raises(ValueError, match=message):
            beam.validate(width, n_best, penalty)


def test_the_search_matches_the_reference_implementation(tiny):
    twin = reference.ReferenceTransformer.from_engine_model(tiny.model)
    context = tiny.tokenizer.encode(PROMPT)
    first = beam.search_tokens(tiny.model, context, 4, 10, frozenset(), n_best=4)
    stops = {first[0].tokens[3], first[0].tokens[5]}  # tokens the search meets on its way: answers end early
    for width, n_best, penalty, stop in ((4, 3, 1.0, set()), (3, 3, 0.0, stops), (2, 2, 0.7, stops), (5, 5, 1.3, stops),
                                         (1, 1, 1.0, stops)):  # fmt: skip
        found = beam.search_tokens(
            tiny.model, context, width, 10, frozenset(stop), n_best=n_best, length_penalty=penalty
        )
        expected = twin.beam_search(context, width, 10, sorted(stop), n_best=n_best, length_penalty=penalty)
        assert _rows(found) == [(t, r, a.hex(), b.hex()) for t, r, a, b in expected]
        assert [h.score for h in found] == sorted((h.score for h in found), reverse=True)
    stopped = beam.search_tokens(tiny.model, context, 3, 10, frozenset(stops), n_best=3)
    assert any(h.finish_reason == "stop" for h in stopped)
    assert all(len(h.logprobs) == len(h.tokens) + (h.finish_reason == "stop") for h in stopped)
    window = beam.search_tokens(tiny.model, context, 2, 10, frozenset(), window=len(context) + 3)
    assert len(window[0].tokens) == 3 and window[0].finish_reason == "length"
    expected = twin.beam_search(context, 2, 10, window=len(context) + 3)
    assert _rows(window) == [(t, r, a.hex(), b.hex()) for t, r, a, b in expected]


def test_width_one_is_greedy_decoding(tiny):
    greedy = tiny.complete(PROMPT, 12, GREEDY)
    found = beam.search(tiny, ChatRequest([], 12, prompt=PROMPT), 1)
    assert found.hypotheses[0].tokens == greedy.tokens and found.hypotheses[0].text == greedy.text
    assert found.hypotheses[0].fingerprint == greedy.fingerprint


class _Flat:
    """A model whose every next-token distribution is uniform: every candidate ties."""

    vocabulary_size = 4

    def forward(self, tokens):
        return np.zeros(4, dtype=np.float32)


class _Cached:
    """The tiny model with a KV cache but no batched pass."""

    def __init__(self, model):
        self._model = model
        self.new_cache = model.new_cache
        self.forward_cached = model.forward_cached


def test_ties_break_on_the_token_sequence(tiny):
    found = beam.search_tokens(_Flat(), [1], 3, 2, frozenset(), n_best=3)
    assert [h.tokens for h in found] == [(0, 0), (0, 1), (0, 2)]
    assert len({h.score for h in found}) == 1
    ended = beam.search_tokens(_Flat(), [1], 2, 3, frozenset({0}), n_best=2)
    assert [(h.tokens, h.finish_reason) for h in ended] == [((), "stop"), ((1,), "stop")]
    context = tiny.tokenizer.encode(PROMPT)
    batched = beam.search_tokens(tiny.model, context, 3, 6, frozenset(), n_best=3)
    assert _rows(beam.search_tokens(_Cached(tiny.model), context, 3, 6, frozenset(), n_best=3)) == _rows(batched)


def test_requests_honour_stop_sequences_and_new_texts(tiny, monkeypatch):
    raw = beam.search(tiny, ChatRequest([], 10, prompt=PROMPT), 3, n_best=3)
    best = raw.hypotheses[0]
    stop = best.text[3:5]
    cut = beam.search(tiny, ChatRequest([], 10, prompt=PROMPT, stop=[stop]), 3, n_best=3)
    assert cut.hypotheses[0].finish_reason == "stop" and stop not in cut.hypotheses[0].text
    assert all(stop not in h.text for h in cut.hypotheses)
    chat = beam.search(tiny, ChatRequest(STORY, 8), 2, n_best=2)
    assert len(chat.hypotheses) == 2 and chat.prompt_tokens == len(tiny.tokenizer.encode(tiny.render_chat(STORY)))
    assert chat.hypotheses[0].fingerprint == BEAM_FINGERPRINTS["chat"]
    assert raw.hypotheses[0].fingerprint == BEAM_FINGERPRINTS["raw"]
    assert chat.to_json()["hypotheses"][0]["text"] == chat.hypotheses[0].text
    space = tiny.tokenizer.encode(" a")[0]
    monkeypatch.setattr(tiny.tokenizer, "strips_leading_space", True)
    monkeypatch.setattr(
        beam, "search_tokens", lambda *a, **k: [beam.Hypothesis((space,), (-1.0,), "length", -1.0, -1.0)]
    )
    stripped = beam.search(tiny, ChatRequest(STORY, 4, stop=["zz"]), 2)
    assert stripped.hypotheses[0].text == tiny.tokenizer.decode_bytes([space]).decode()[1:]


def test_requests_that_cannot_use_beam_search(tiny):
    tool = Tool("lookup", "Looks something up", {"type": "object", "properties": {}})
    cases = [
        (ChatRequest(STORY, 8, tools=[tool]), "tools"),
        (ChatRequest(STORY, 8, response_format=ResponseFormat("json_object")), "structured output"),
        (ChatRequest(STORY, 8, context_overflow="roll"), "roll"),
        (ChatRequest(STORY, 8, top_logprobs=2), "top_logprobs"),
        (ChatRequest(STORY, 8, SamplingOptions(frequency_penalty=0.5, logit_bias=((3, 1.0),))), "frequency_penalty"),
        (ChatRequest(STORY, 0), "max_tokens"),
    ]
    for request, message in cases:
        with pytest.raises(ValueError, match=message):
            beam.search(tiny, request, 2)
    with pytest.raises(ContextLengthError):
        beam.search(tiny, ChatRequest([], 4, prompt="word " * 3000), 2)
    sampled = beam.search(tiny, ChatRequest([], 6, SamplingOptions(temperature=0.9, seed=4), prompt=PROMPT), 2)
    assert sampled.hypotheses == beam.search(tiny, ChatRequest([], 6, prompt=PROMPT), 2).hypotheses


def test_receipts_replay(tiny):
    request = ChatRequest(STORY, 8, request_id="chatcmpl-beam")
    result = beam.search(tiny, request, 3, n_best=2, length_penalty=0.5)
    receipt = beam.record(tiny, request, result)
    assert receipt["beam"] == beam.FORMAT and receipt["id"].startswith("beam_")
    assert beam.verify(tiny, receipt).ok
    edited = {**receipt, "output": {"hypotheses": receipt["output"]["hypotheses"][:1]}}
    verification = beam.verify(tiny, edited)
    assert (
        not verification.ok and "edited" in verification.reasons[0] and "number of answers" in verification.reasons[1]
    )
    changed = json.loads(json.dumps(receipt))
    changed["output"]["hypotheses"][1]["score"] = (0.5).hex()
    changed["system_fingerprint"], changed["engine"] = "other", "0.0.1"
    verification = beam.verify(tiny, changed)
    assert not verification.ok and any("answer 1 differs" in r for r in verification.reasons)
    assert any("different weights" in r for r in verification.reasons) and verification.notes
    with pytest.raises(ValueError, match="dllm-beam"):
        beam.verify(tiny, {"beam": "dllm-beam/0"})


def test_the_reference_check_reports_a_different_search(tiny, monkeypatch):
    def shifted(self, context, width, max_tokens, stop_tokens=(), **kwargs):
        return [([0], "stop", -1.0, -1.0)] * width

    monkeypatch.setattr(reference.ReferenceTransformer, "beam_search", shifted)
    assert verify.check_reference(tiny, max_tokens=4).results["beam"] == "answer 0 of 3 differs"
    monkeypatch.setattr(reference.ReferenceTransformer, "beam_search", lambda self, *a, **k: [])
    assert verify.check_reference(tiny, max_tokens=4).results["beam"] == "3 answers, the reference has 0"


# Front ends


def test_chat_completions_return_the_best_answers(served):
    client = TestClient(app)
    body = {"messages": [{"role": "user", "content": "Tell me a story."}], "max_tokens": 8}
    response = client.post("/v1/chat/completions", json={**body, "beam": {"width": 3, "n_best": 2}, "logprobs": True,
                                                         "receipt": True}).json()  # fmt: skip
    expected = beam.search(served, ChatRequest(STORY, 8), 3, n_best=2)
    assert [c["message"]["content"] for c in response["choices"]] == [h.text for h in expected.hypotheses]
    assert response["beam"] == json.loads(json.dumps(expected.to_json()))
    assert response["usage"]["completion_tokens"] == expected.completion_tokens
    first = response["choices"][0]["logprobs"]["content"]
    assert [e["logprob"] for e in first] == list(expected.hypotheses[0].logprobs[: len(first)])
    plain = client.post("/v1/chat/completions", json=body).json()
    assert plain["id"] != response["id"]
    assert client.post("/v1/receipts/verify", json=response["receipt"]).json()["ok"]
    for extra in ({"stream": True}, {"n": 2}):
        refused = client.post("/v1/chat/completions", json={**body, "beam": {"width": 2}, **extra})
        assert refused.status_code == 400 and "beam search cannot" in refused.json()["error"]["message"]
    refused = client.post("/v1/chat/completions", json={**body, "beam": {"width": 2}, "frequency_penalty": 1})
    assert refused.status_code == 400


def test_completions_return_the_best_answers(served):
    client = TestClient(app)
    body = {"prompt": [PROMPT, "The end"], "max_tokens": 6, "beam": {"width": 2, "n_best": 2}, "echo": True,
            "receipt": True}  # fmt: skip
    response = client.post("/v1/completions", json=body).json()
    expected = [beam.search(served, ChatRequest([], 6, prompt=p), 2, n_best=2) for p in (PROMPT, "The end")]
    texts = [p + h.text for p, e in zip((PROMPT, "The end"), expected, strict=True) for h in e.hypotheses]
    assert [c["text"] for c in response["choices"]] == texts and [c["index"] for c in response["choices"]] == [
        0,
        1,
        2,
        3,
    ]
    assert len(response["beam"]) == 2 and response["choices"][0]["receipt"]["beam"] == beam.FORMAT
    assert response["usage"]["completion_tokens"] == sum(e.completion_tokens for e in expected)
    refused = client.post("/v1/completions", json={**body, "logprobs": 1})
    assert refused.status_code == 400 and "beam search cannot" in refused.json()["error"]["message"]


def test_the_command_line(served, capsys, tmp_path):
    args = ["generate", "--prompt", PROMPT, "--max-tokens", "8"]
    receipt = tmp_path / "beam.json"
    assert main([*args, "--beams", "3", "--n-best", "2", "--receipt", str(receipt)]) == 0
    captured = capsys.readouterr()
    expected = beam.search(served, ChatRequest([], 8, prompt=PROMPT), 3, n_best=2)
    assert captured.out == "".join(h.text + "\n" for h in expected.hypotheses)
    assert "--- answer 1" in captured.err and "score:" in captured.err
    assert main(["replay", str(receipt)]) == 0
    assert "verified" in capsys.readouterr().out.lower()
    assert main([*args, "--receipt", str(receipt)]) == 2 and "needs --beams" in capsys.readouterr().err
    assert main([*args, "--beams", "2", "--n", "2"]) == 2 and "--n-best" in capsys.readouterr().err
    assert main([*args, "--beams", "2", "--n-best", "3"]) == 2 and "n_best" in capsys.readouterr().err
