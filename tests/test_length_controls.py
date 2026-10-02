"""Phase 38: exact length and stop controls. ``min_tokens`` and ``ignore_eos`` (#252), ``stop_token_ids`` and
``include_stop_str_in_output`` (#253), in receipts and the reference implementation (#254), with golden values
(#255)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from golden_values import LENGTH_FINGERPRINTS
from test_cli import isolated_environment  # noqa: F401 - autouse fixture: the CLI tests set DLLM_MODEL
from test_engine_import import model_path  # noqa: F401 - fixture

from etalii_dllm import receipts
from etalii_dllm.chat import ChatMessage
from etalii_dllm.engine import ChatRequest, DllmEngine
from etalii_dllm.sampling import GREEDY, SamplingOptions

SAMPLED = SamplingOptions(temperature=0.9, seed=3)
PROMPT = "Once upon a time"


@pytest.fixture(scope="module")
def tiny(model_path) -> DllmEngine:  # noqa: F811
    return DllmEngine.from_model_file(model_path)


def _eager(tiny: DllmEngine, temperature: float = 0.0) -> SamplingOptions:
    """Options under which the model wants to stop at once: its end-of-sequence token pushed up."""
    return SamplingOptions(temperature=temperature, seed=3, logit_bias=((tiny.tokenizer.end_of_sequence, 100.0),))


# min_tokens and ignore_eos


@pytest.mark.parametrize("temperature", [0.0, 0.9])
def test_min_tokens_keeps_the_stop_tokens_out_until_the_answer_is_long_enough(tiny, temperature):
    eager = _eager(tiny, temperature)
    assert tiny.complete_stream(PROMPT, 10, eager).result().tokens == ()
    held = tiny.complete_stream(PROMPT, 10, eager, min_tokens=4).result()
    assert len(held.tokens) == 4 and held.finish_reason == "stop"
    assert not set(held.tokens) & tiny.stop_tokens
    assert len(tiny.complete_stream(PROMPT, 3, eager, min_tokens=6).result().tokens) == 3  # max_tokens still caps


def test_min_tokens_golden(tiny):
    result = tiny.complete_stream(PROMPT, 12, SAMPLED, min_tokens=12).result()
    assert result.fingerprint == LENGTH_FINGERPRINTS["min_tokens"]
    again = tiny.chat_completion(ChatRequest([], 12, SAMPLED, prompt=PROMPT, min_tokens=12))
    assert again.fingerprint == result.fingerprint


@pytest.mark.parametrize("temperature", [0.0, 0.9])
def test_min_tokens_under_a_regex(tiny, temperature):
    eager = _eager(tiny, temperature)
    short = tiny.complete_stream("Count: ", 10, eager, regex="[0-9]+").result()
    assert len(short.tokens) == 1
    held = tiny.complete_stream("Count: ", 10, eager, regex="[0-9]+", min_tokens=4).result()
    assert len(held.tokens) == 4 and held.text.isdigit() and held.finish_reason == "stop"


def test_ignore_eos_writes_past_the_end_of_sequence(tiny):
    eos = tiny.tokenizer.end_of_sequence
    result = tiny.complete_stream(PROMPT, 5, _eager(tiny), ignore_eos=True).result()
    assert result.tokens == (eos,) * 5 and result.finish_reason == "length" and result.text == ""
    golden = tiny.complete_stream(PROMPT, 12, SAMPLED, ignore_eos=True).result()
    assert golden.fingerprint == LENGTH_FINGERPRINTS["ignore_eos"]


# stop_token_ids and include_stop_str_in_output


def test_stop_token_ids_end_the_answer(tiny):
    plain = tiny.complete_stream(PROMPT, 12, GREEDY).result()
    index = next(i for i, t in enumerate(plain.tokens) if i > 0 and t not in plain.tokens[:i])
    stopped = tiny.complete_stream(PROMPT, 12, GREEDY, stop_token_ids=[plain.tokens[index]]).result()
    assert stopped.tokens == plain.tokens[:index] and stopped.finish_reason == "stop"
    pushed = tiny.complete_stream(PROMPT, 5, _eager(tiny), ignore_eos=True, stop_token_ids=[plain.tokens[0]])
    assert pushed.result().tokens == (tiny.tokenizer.end_of_sequence,) * 5  # only the listed ids stop it then


def test_stop_token_ids_outside_the_vocabulary_are_refused(tiny):
    with pytest.raises(ValueError, match="stop token ids"):
        tiny.complete_stream(PROMPT, 4, GREEDY, stop_token_ids=[tiny.model.vocabulary_size])
    with pytest.raises(ValueError, match="stop token ids"):
        tiny.complete_stream(PROMPT, 4, GREEDY, stop_token_ids=[-1])
    with pytest.raises(ValueError, match="min_tokens"):
        tiny.complete_stream(PROMPT, 4, GREEDY, min_tokens=-1)


def test_include_stop_keeps_the_stop_string(tiny):
    plain = tiny.complete_stream(PROMPT, 16, GREEDY).result().text
    stop = plain[3:5]
    cut = plain.index(stop)
    without = tiny.complete_stream(PROMPT, 16, GREEDY, stop=[stop]).result()
    kept = tiny.complete_stream(PROMPT, 16, GREEDY, stop=[stop], include_stop=True)
    streamed = "".join(step.text for step in kept)
    assert without.text == plain[:cut] and kept.result().text == plain[: cut + len(stop)] == streamed
    assert kept.stop_sequence == stop


# Receipts, beams


def test_receipts_record_the_controls(tiny):
    request = ChatRequest([], 8, SAMPLED, prompt=PROMPT, min_tokens=3, ignore_eos=True, stop_token_ids=(5, 9),
                          stop=("zz",), include_stop=True)  # fmt: skip
    record = receipts.request_record(request)
    assert (record["min_tokens"], record["ignore_eos"], record["stop_token_ids"], record["include_stop"]) == (
        3, True, [5, 9], True)  # fmt: skip
    assert receipts.request_from_record(record) == request
    bare = receipts.request_record(ChatRequest([], 8, SAMPLED, prompt=PROMPT))
    assert not {"min_tokens", "ignore_eos", "stop_token_ids", "include_stop"} & set(bare)
    result = tiny.chat_completion(request)
    assert result.receipt is not None and receipts.verify(tiny, result.receipt).ok


def test_beam_search_refuses_the_controls(tiny):
    from etalii_dllm import beam

    with pytest.raises(ValueError, match="min_tokens"):
        beam.search(tiny, ChatRequest([], 4, prompt=PROMPT, min_tokens=2), 2)


# Front ends


def test_every_openai_api_takes_the_controls(tiny):
    from etalii_dllm.engine import default_engine
    from etalii_dllm.server.app import app

    eos = tiny.tokenizer.end_of_sequence
    controls = {"min_tokens": 4, "logit_bias": {str(eos): 100}}
    app.dependency_overrides[default_engine] = lambda: tiny
    try:
        client = TestClient(app)
        body = {"prompt": PROMPT, "max_tokens": 10}
        plain = client.post("/v1/completions", json=body).json()
        assert client.post("/v1/completions", json={**body, "min_tokens": None}).json()["id"] == plain["id"]
        completion = client.post("/v1/completions", json={**body, **controls}).json()
        assert completion["usage"]["completion_tokens"] == 4 and completion["id"] != plain["id"]
        ignored = client.post("/v1/completions", json={**body, "ignore_eos": True, "logit_bias": {str(eos): 100}})
        assert ignored.json()["usage"]["completion_tokens"] == 10
        bad = client.post("/v1/completions", json={**body, "stop_token_ids": [10**9]})
        assert bad.status_code == 400 and "stop token ids" in bad.json()["error"]["message"]
        messages = [{"role": "user", "content": "Tell a story"}]
        chat = client.post("/v1/chat/completions", json={"messages": messages, "max_tokens": 10, **controls}).json()
        assert chat["usage"]["completion_tokens"] == 4
        first = (
            tiny.complete_stream(tiny.render_chat([ChatMessage("user", "Tell a story")]), 1, GREEDY).result().tokens[0]
        )
        responses = client.post("/v1/responses", json={"input": messages, "max_output_tokens": 10,
                                                       "stop_token_ids": [first]}).json()  # fmt: skip
        assert responses["usage"]["output_tokens"] == 0
        text = tiny.complete_stream(PROMPT, 16, GREEDY).result().text
        stop = text[3:5]
        kept = client.post("/v1/completions", json={"prompt": PROMPT, "max_tokens": 16, "stop": [stop],
                                                    "include_stop_str_in_output": True}).json()  # fmt: skip
        assert kept["choices"][0]["text"] == text[: text.index(stop) + len(stop)]
        reference = tiny.chat_completion(
            ChatRequest([ChatMessage("user", "Tell a story")], 10, GREEDY, stop_token_ids=(eos,), min_tokens=2)
        )
        chat = client.post("/v1/chat/completions", json={"messages": messages, "max_tokens": 10, "min_tokens": 2,
                                                         "stop_token_ids": [eos]}).json()  # fmt: skip
        assert chat["choices"][0]["message"]["content"] == reference.content
    finally:
        app.dependency_overrides.clear()


def test_cli_controls(model_path, capsys):  # noqa: F811
    from etalii_dllm.cli import main

    args = ["--model", str(model_path)]
    assert main([*args, "generate", "--prompt", PROMPT, "--max-tokens", "12", "--temperature", "0.9", "--seed", "3",
                 "--min-tokens", "12"]) == 0  # fmt: skip
    assert LENGTH_FINGERPRINTS["min_tokens"] in capsys.readouterr().err
    assert main([*args, "generate", "--prompt", PROMPT, "--max-tokens", "12", "--temperature", "0.9", "--seed", "3",
                 "--ignore-eos"]) == 0  # fmt: skip
    assert LENGTH_FINGERPRINTS["ignore_eos"] in capsys.readouterr().err
    assert main([*args, "generate", "--prompt", PROMPT, "--stop", "a", "--include-stop", "--stop-token-id", "5"]) == 0
    assert main([*args, "generate", "--prompt", PROMPT, "--stop-token-id", "999999"]) == 2
    assert main([*args, "chat", "Hi", "--max-tokens", "6", "--min-tokens", "3", "--stop", "x"]) == 0
    assert main([*args, "generate", "--prompt", PROMPT, "--beams", "2", "--min-tokens", "2"]) == 2
