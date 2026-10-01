"""Phase 24: reasoning models (docs/api.md#reasoning, docs/specification.md#reasoning)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from model_fixtures import tiny_config, write_hf_checkpoint
from test_cli import isolated_environment  # noqa: F401
from test_model_building import BYTE_LEVEL_TOKENIZER

from etalii_dllm import reasoning, receipts, reference, serving
from etalii_dllm.chat import ChatMessage
from etalii_dllm.chat_template import ChatTemplate
from etalii_dllm.cli import main
from etalii_dllm.engine import (
    ChatRequest,
    DllmEngine,
    Finished,
    ReasoningDelta,
    ResponseFormat,
    TextDelta,
    default_engine,
)
from etalii_dllm.importing import import_model
from etalii_dllm.sampling import SamplingOptions
from etalii_dllm.server.app import app

GREEDY = SamplingOptions()
TURNS = "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}\n{% endfor %}"
SWITCHED = TURNS + (
    "{% if add_generation_prompt %}<|assistant|>"
    "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}{% endif %}"
)
"""Qwen3 style: the model opens its own <think> block; the template closes an empty one when thinking is off."""
OPENING = TURNS + "{% if add_generation_prompt %}<|assistant|><think>\n{% endif %}{# </think> #}"
"""DeepSeek-R1 style: the prompt opens the block and the template ignores enable_thinking."""
QUESTION = [ChatMessage("user", "Why?")]


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    directory = tmp_path_factory.mktemp("thinking")
    config = {**tiny_config("llama"), "vocab_size": 264}
    write_hf_checkpoint(directory / "checkpoint", config, tokenizer_json=BYTE_LEVEL_TOKENIZER)
    import_model(directory / "checkpoint", directory / "thinking.dllm")
    return directory / "thinking.dllm"


def _engine(model_path, source: str | None) -> DllmEngine:
    engine = DllmEngine.from_model_file(model_path)
    engine.chat_template = ChatTemplate(source) if source is not None else None
    return engine


@pytest.fixture(scope="module")
def opening(model_path):
    return _engine(model_path, OPENING)


@pytest.fixture
def client(opening):
    app.dependency_overrides[default_engine] = lambda: opening
    yield TestClient(app)
    app.dependency_overrides.pop(default_engine, None)


# -- the text rule ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "started", "reasoning_text", "answer", "state"),
    [
        ("Hello", False, None, "Hello", "answer"),
        ("  <thi", False, None, "  <thi", "undecided"),
        ("", False, None, "", "undecided"),
        ("\n<think>\nPlan.\n</think>\n\nAnswer.", False, "Plan.", "Answer.", "answer"),
        ("<think>still going", False, "still going", "", "thinking"),
        ("Plan.\n</think>\n\nDone</think>", True, "Plan.", "Done</think>", "answer"),
        ("I think <think> no", False, None, "I think <think> no", "answer"),
    ],
)
def test_the_split_rule(text, started, reasoning_text, answer, state):
    assert reasoning.split(text, started) == reasoning.Split(reasoning_text, answer, state)


@pytest.mark.parametrize(
    ("text", "started"),
    [("\n<think>\n First, think.  \n</think>\n\n The answer. ", False), ("Plan </thi a b</think>\nok", True)],
)
def test_streamed_parts_are_prefixes_of_the_final_ones(text, started):
    final = reasoning.split(text, started)
    for end in range(len(text) + 1):
        thought, answer = reasoning.streamable(text[:end], started)
        assert (final.reasoning or "").startswith(thought) and final.answer.startswith(answer)


def test_the_tracker_counts_the_block():
    tracker = reasoning.Tracker(False, 3)
    for text in (" ", " <th", " <think>", " <think>a", " <think>ab"):
        tracker.step(text)
    assert (tracker.tokens, tracker.over_budget) == (3, True)
    tracker.step(" <think>ab</think>")
    assert (tracker.tokens, tracker.over_budget) == (4, False)
    plain = reasoning.Tracker(False, 0)
    plain.step("Hi")
    plain.step("Hi <think>")
    assert (plain.tokens, plain.over_budget) == (0, False)
    assert reasoning.closing_text("abc\n") == "</think>\n\n" and reasoning.closing_text("abc") == "\n</think>\n\n"
    assert not reasoning.is_thinking_template(None) and reasoning.starts_in_thinking("x<think>\n")


# -- the engine ---------------------------------------------------------------------------------------------------


def test_a_budget_closes_the_block_with_fixed_tokens(opening):
    request = ChatRequest(QUESTION, 30, GREEDY, max_reasoning_tokens=6)
    events = list(opening.chat_stream(request))
    finished = events[-1]
    assert isinstance(finished, Finished)
    thought = "".join(e.text for e in events if isinstance(e, ReasoningDelta))
    answer = "".join(e.text for e in events if isinstance(e, TextDelta))
    result = opening.chat_completion(request)
    assert (result.reasoning, result.content) == (thought, answer)
    assert result.reasoning_tokens == finished.reasoning_tokens >= 6
    prompt = opening.tokenizer.encode(opening.render_chat(QUESTION))
    twin = reference.ReferenceTransformer.from_engine_model(opening.model)
    budget = reference.ThinkingBudget(6, True, opening.tokenizer.decode_bytes, opening.tokenizer.encode)
    tokens, _ = twin.generate(prompt, 30, reference.sampler(GREEDY), sorted(opening._generator.stop_tokens),
                              thinking=budget)  # fmt: skip
    text = opening.tokenizer.decode_bytes(tokens).decode("utf-8", errors="replace")
    assert "</think>" in text and reasoning.split(text, True) == reasoning.Split(result.reasoning, answer, "answer")


def test_without_a_budget_the_block_may_stay_open(opening):
    result = opening.chat_completion(ChatRequest(QUESTION, 12, GREEDY))
    assert result.reasoning is not None and result.content == "" and result.reasoning_tokens == 12
    zero = opening.chat_completion(ChatRequest(QUESTION, 12, GREEDY, max_reasoning_tokens=0))
    assert zero.reasoning == "" and zero.reasoning_tokens >= 1


def test_structured_output_answers_at_once(opening):
    request = ChatRequest(QUESTION, 12, GREEDY, response_format=ResponseFormat("json_object"))
    assert opening.chat_completion(request).reasoning is None


def test_the_thinking_switch(model_path, opening):
    switched = _engine(model_path, SWITCHED)
    on, off = switched.render_chat(QUESTION, thinking=True), switched.render_chat(QUESTION, thinking=False)
    assert on == switched.render_chat(QUESTION) and off == on + "<think>\n\n</think>\n\n"
    plain = TURNS + "{% if add_generation_prompt %}<|assistant|>{% endif %}{# <think> #}"
    ignoring = _engine(model_path, plain)
    assert ignoring.render_chat(QUESTION, thinking=False) == ignoring.render_chat(QUESTION) + (
        "<think>\n\n</think>\n\n"
    )
    assert opening.render_chat(QUESTION, thinking=False) == opening.render_chat(QUESTION) + "\n</think>\n\n"
    assert opening.chat_completion(ChatRequest(QUESTION, 8, GREEDY, thinking=False)).reasoning is None
    assert opening.render_chat(QUESTION, thinking=True) == opening.render_chat(QUESTION)
    no_thinking = _engine(model_path, TURNS)
    assert no_thinking.render_chat(QUESTION, thinking=False) == no_thinking.render_chat(QUESTION)
    assert no_thinking.chat_completion(ChatRequest(QUESTION, 8, GREEDY, thinking=True)).reasoning is None
    prefill = [*QUESTION, ChatMessage("assistant", "<think>")]
    assert switched.chat_completion(ChatRequest(prefill, 6, GREEDY, max_reasoning_tokens=2)).reasoning is not None
    with pytest.raises(ValueError, match="max_reasoning_tokens must be non-negative"):
        ChatRequest(QUESTION, 8, GREEDY, max_reasoning_tokens=-1)


def test_receipts_and_the_response_cache(opening):
    plain = receipts.request_record(ChatRequest(QUESTION, 8, GREEDY))
    assert "thinking" not in plain and "max_reasoning_tokens" not in plain
    request = ChatRequest(QUESTION, 30, GREEDY, thinking=True, max_reasoning_tokens=6)
    record = receipts.request_record(request)
    assert (record["thinking"], record["max_reasoning_tokens"]) == (True, 6)
    assert receipts.request_from_record(record) == request
    result = opening.chat_completion(request)
    assert "reasoning" in result.receipt["output"]
    assert "reasoning" not in receipts.output_record("t", "c", [], "stop", 1, 1)
    events = list(opening.chat_stream(request, fresh=True))
    assert [serving.event_from_json(json.loads(json.dumps(serving.event_json(e)))) for e in events] == events


# -- front ends ---------------------------------------------------------------------------------------------------


def test_openai_chat_completions(client):
    body = {"model": "m", "messages": [{"role": "user", "content": "Why?"}], "max_tokens": 30,
            "max_reasoning_tokens": 6}  # fmt: skip
    reply = client.post("/v1/chat/completions", json=body).json()
    message = reply["choices"][0]["message"]
    assert message["reasoning_content"] and "</think>" not in message["content"]
    assert reply["usage"]["completion_tokens_details"]["reasoning_tokens"] >= 6
    stream = client.post(
        "/v1/chat/completions", json={**body, "stream": True, "stream_options": {"include_usage": True}}
    )
    chunks = [json.loads(line[6:]) for line in stream.text.splitlines() if line.startswith("data: {")]
    deltas = [c["choices"][0]["delta"] for c in chunks if c["choices"]]
    assert "".join(d.get("reasoning_content", "") for d in deltas) == message["reasoning_content"]
    assert "".join(d.get("content") or "" for d in deltas) == message["content"]
    assert chunks[-1]["usage"]["completion_tokens_details"] == reply["usage"]["completion_tokens_details"]
    off = {**body, "reasoning_effort": "none"}
    assert "reasoning_content" not in client.post("/v1/chat/completions", json=off).json()["choices"][0]["message"]
    kwargs = {**body, "chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "high"}
    assert "reasoning_content" not in client.post("/v1/chat/completions", json=kwargs).json()["choices"][0]["message"]
    for bad in ({"foo": 1}, {"enable_thinking": "no"}):
        assert client.post("/v1/chat/completions", json={**body, "chat_template_kwargs": bad}).status_code == 400


def test_responses_api(client):
    request = {"model": "m", "input": "Why?", "max_output_tokens": 30, "max_reasoning_tokens": 6}
    response = client.post("/v1/responses", json=request).json()
    item, message = response["output"][0], response["output"][1]
    assert item["type"] == "reasoning" and item["content"][0]["type"] == "reasoning_text"
    assert message["type"] == "message" and response["usage"]["output_tokens_details"]["reasoning_tokens"] >= 6
    stream = client.post("/v1/responses", json={**request, "stream": True}).text
    events = [json.loads(line[6:]) for line in stream.splitlines() if line.startswith("data: ")]
    text = "".join(e["delta"] for e in events if e["type"] == "response.reasoning_text.delta")
    assert text == item["content"][0]["text"]
    off = client.post("/v1/responses", json={**request, "reasoning": {"effort": "none"}}).json()
    assert [o["type"] for o in off["output"]] == ["message"]


def test_anthropic_messages(client):
    body = {"model": "m", "max_tokens": 30, "messages": [{"role": "user", "content": "Why?"}],
            "thinking": {"type": "enabled", "budget_tokens": 6}}  # fmt: skip
    reply = client.post("/v1/messages", json=body).json()
    block = reply["content"][0]
    assert block["type"] == "thinking" and block["signature"].startswith("dllm-")
    stream = client.post("/v1/messages", json={**body, "stream": True}).text
    events = [json.loads(line[6:]) for line in stream.splitlines() if line.startswith("data: ")]
    deltas = [e["delta"] for e in events if e["type"] == "content_block_delta"]
    assert "".join(d.get("thinking", "") for d in deltas) == block["thinking"]
    assert {"type": "signature_delta", "signature": block["signature"]} in deltas
    off = client.post("/v1/messages", json={**body, "thinking": {"type": "disabled"}}).json()
    assert all(b["type"] != "thinking" for b in off["content"])
    adaptive = client.post("/v1/messages", json={**body, "max_tokens": 6, "thinking": {"type": "adaptive"}}).json()
    assert adaptive["content"][0]["type"] == "thinking"
    missing = client.post("/v1/messages", json={**body, "thinking": {"type": "enabled"}})
    assert missing.status_code == 400 and "budget_tokens" in missing.json()["error"]["message"]


def test_ollama(client):
    body = {"model": "m", "messages": [{"role": "user", "content": "Why?"}], "stream": False, "think": True,
            "max_reasoning_tokens": 6, "options": {"num_predict": 30}}  # fmt: skip
    reply = client.post("/api/chat", json=body).json()
    assert reply["message"]["thinking"] and "</think>" not in reply["message"]["content"]
    off = client.post("/api/chat", json={**body, "think": False}).json()
    assert "thinking" not in off["message"]
    generate = {"model": "m", "prompt": "Why?", "stream": False, "max_reasoning_tokens": 6,
                "options": {"num_predict": 30}}  # fmt: skip
    assert client.post("/api/generate", json=generate).json()["thinking"] == reply["message"]["thinking"]
    lines = client.post("/api/chat", json={**body, "stream": True}).text.splitlines()
    assert "".join(json.loads(line)["message"].get("thinking", "") for line in lines) == reply["message"]["thinking"]


def test_the_command_line(model_path, opening, capsys, monkeypatch, isolated_environment):  # noqa: F811
    monkeypatch.setattr("etalii_dllm.cli.default_engine", lambda: opening)
    assert main(["--model", str(model_path), "chat", "Why?", "--max-tokens", "30", "--max-reasoning-tokens", "6"]) == 0
    err = capsys.readouterr().err
    assert err.startswith("thinking: ") or "\nthinking: " in err
    assert main(["--model", str(model_path), "chat", "Why?", "--max-tokens", "8", "--no-think"]) == 0
    assert "thinking: " not in capsys.readouterr().err
