"""Phase 23: the context window, truncation of long conversations and rolling generation (docs/api.md#long-
conversations, docs/specification.md#the-context-window)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from model_fixtures import tiny_config, write_hf_checkpoint
from test_cli import isolated_environment  # noqa: F401
from test_model_building import BYTE_LEVEL_TOKENIZER

from etalii_dllm import receipts, reference
from etalii_dllm.chat import ChatMessage
from etalii_dllm.cli import main
from etalii_dllm.engine import ChatRequest, DllmEngine, Finished, default_engine
from etalii_dllm.generation import ROLL_SINK, ContextLengthError, Generator
from etalii_dllm.importing import import_model
from etalii_dllm.sampling import SamplingOptions
from etalii_dllm.server.app import app

WINDOW = 48
GREEDY = SamplingOptions()
SAMPLED = SamplingOptions(temperature=0.9, top_k=40, seed=11, repetition_penalty=1.1)


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    """A tiny byte-level model (one token per byte) with a context window of :data:`WINDOW` tokens."""
    directory = tmp_path_factory.mktemp("window")
    config = {**tiny_config("llama"), "vocab_size": 264, "max_position_embeddings": WINDOW}
    write_hf_checkpoint(directory / "checkpoint", config, tokenizer_json=BYTE_LEVEL_TOKENIZER)
    import_model(directory / "checkpoint", directory / "window.dllm")
    return directory / "window.dllm"


@pytest.fixture(scope="module")
def engine(model_path):
    return DllmEngine.from_model_file(model_path)


@pytest.fixture
def client(engine):
    app.dependency_overrides[default_engine] = lambda: engine
    yield TestClient(app)
    app.dependency_overrides.pop(default_engine, None)


def _fresh_run(engine: DllmEngine, prompt: list[int], max_tokens: int, options: SamplingOptions) -> list[int]:
    """What rolling must give: every token chosen from a fresh forward pass over the kept tokens, with the sampler
    seeing the whole history (the specification's rule, spelled out with the engine's cache-free path)."""
    model = engine.model
    sampler = reference.sampler(options)
    sequence, tokens = list(prompt), []
    sampler.begin(sequence)
    while len(tokens) < max_tokens:
        if len(sequence) >= WINDOW:
            sequence = sequence[:ROLL_SINK] + sequence[-(WINDOW // 2) :]
        token = sampler.sample(model.forward(sequence).reshape(-1))
        if token in engine.stop_tokens:
            break
        sampler.accept(token)
        tokens.append(token)
        sequence.append(token)
    return tokens


# -- the window ---------------------------------------------------------------------------------------------------


def test_a_prompt_must_leave_room_for_the_answer(engine):
    with pytest.raises(ContextLengthError, match=f"the prompt has {WINDOW} tokens; .* holds {WINDOW}"):
        engine.complete_stream("x" * WINDOW, 4, GREEDY)
    with pytest.raises(ValueError, match="overflow must be one of stop, roll"):
        engine.complete_stream("x", 4, GREEDY, overflow="wrap")
    assert engine.complete_stream("x" * (WINDOW - 1), 4, GREEDY).result().finish_reason == "length"


def test_the_answer_stops_at_a_full_window(engine):
    result = engine.complete_stream("y" * 40, 30, GREEDY).result()
    assert (len(result.tokens), result.finish_reason) == (WINDOW - 40, "length")


def test_a_tiny_window_cannot_roll(engine):
    generator = Generator(engine.model, engine.tokenizer)
    generator.context_length = 2 * ROLL_SINK
    with pytest.raises(ValueError, match="too small to roll"):
        generator.stream([1, 2], 4, GREEDY, overflow="roll")


# -- rolling ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("options", [GREEDY, SAMPLED], ids=["greedy", "sampled"])
def test_rolling_equals_a_fresh_run_over_the_kept_tokens(engine, model_path, options):
    prompt = engine.tokenizer.encode("Rolling keeps the start and the latest half.")
    generation = engine.complete_stream("Rolling keeps the start and the latest half.", 80, options, overflow="roll")
    result = generation.result()
    expected = _fresh_run(engine, prompt, 80, options)
    assert list(result.tokens) == expected
    assert generation.rolls >= 2 and result.finish_reason == "length"
    uncached = DllmEngine.from_model_file(model_path, prompt_cache=0)  # no prompt cache, no reuse of rolled caches
    assert uncached.complete_stream(engine.tokenizer.decode(prompt), 80, options, overflow="roll").result() == result
    again = engine.complete_stream(engine.tokenizer.decode(prompt), 80, options, overflow="roll").result()
    assert again == result  # the prompt cache now holds caches of the rolled windows; nothing changes


def test_rolling_matches_the_reference_and_survives_batching(engine):
    prompts = [f"Request {i} rolls on." for i in range(4)]
    alone = [list(engine.complete_stream(p, 70, SAMPLED, overflow="roll").result().tokens) for p in prompts]
    with ThreadPoolExecutor(4) as pool:  # concurrent generations share batched forward steps
        together = list(pool.map(lambda p: engine.complete_stream(p, 70, SAMPLED, overflow="roll").result(), prompts))
    assert [list(r.tokens) for r in together] == alone
    twin = reference.ReferenceTransformer.from_engine_model(engine.model)
    context = engine.tokenizer.encode(prompts[0])
    tokens, _ = twin.generate(context, 70, reference.sampler(SAMPLED), sorted(engine.stop_tokens), overflow="roll")
    assert tokens == alone[0]


def test_the_reference_applies_the_window(engine):
    twin = reference.ReferenceTransformer.from_engine_model(engine.model)
    with pytest.raises(ValueError, match="context window holds"):
        twin.generate([5] * WINDOW, 4, reference.sampler(GREEDY))
    tokens, _ = twin.generate([5] * 40, 30, reference.sampler(GREEDY))
    assert tuple(tokens) == engine.complete_stream([5] * 40, 30, GREEDY).result().tokens
    assert len(tokens) == WINDOW - 40


# -- truncation ---------------------------------------------------------------------------------------------------


def _conversation() -> list[ChatMessage]:
    return [
        ChatMessage("system", "Be brief."),
        ChatMessage("user", "First question here."),
        ChatMessage("assistant", "First answer."),
        ChatMessage("tool", "a tool result"),
        ChatMessage("user", "Second one."),
        ChatMessage("assistant", "Ok."),
        ChatMessage("user", "Last?"),
    ]


def test_truncation_drops_the_oldest_turns_first(engine):
    messages = _conversation()
    kept = engine.fit_messages(messages)
    assert kept[0] == messages[0] and kept[-1] == messages[-1]
    assert len(engine.tokenizer.encode(engine.render_chat(kept))) < WINDOW
    assert messages[1] not in kept
    assert engine.fit_messages(messages[-2:]) == messages[-2:]  # fits: unchanged
    tool_first = [ChatMessage("user", "q" * 30), ChatMessage("tool", "t" * 30), ChatMessage("user", "Last?")]
    assert engine.fit_messages(tool_first) == tool_first[2:]  # a tool result goes with the turn it answers
    lonely = [ChatMessage("system", "s" * 60), ChatMessage("user", "Last?")]
    assert engine.fit_messages(lonely) == lonely  # nothing left to drop; the request still fails


def test_chat_requests_truncate_and_roll(engine):
    messages = _conversation()
    with pytest.raises(ContextLengthError):
        engine.chat_stream(ChatRequest(messages, 8, GREEDY))
    stream = engine.chat_stream(ChatRequest(messages, 8, GREEDY, truncation="auto"))
    finished = next(e for e in stream if isinstance(e, Finished))
    short = ChatRequest(engine.fit_messages(messages), 8, GREEDY)
    assert next(e for e in engine.chat_stream(short) if isinstance(e, Finished)).fingerprint == finished.fingerprint
    rolled = ChatRequest(messages, 60, GREEDY, truncation="auto", context_overflow="roll")
    assert engine.chat_completion(rolled).completion_tokens == 60
    with pytest.raises(ValueError, match="truncation must be one of"):
        ChatRequest(messages, 8, GREEDY, truncation="middle")
    with pytest.raises(ValueError, match="context_overflow must be one of"):
        ChatRequest(messages, 8, GREEDY, context_overflow="wrap")


def test_receipts_record_the_new_fields_only_when_set(engine):
    plain = receipts.request_record(ChatRequest(_conversation()[-2:], 8, GREEDY))
    assert "truncation" not in plain and "context_overflow" not in plain
    request = ChatRequest(_conversation(), 60, GREEDY, truncation="auto", context_overflow="roll")
    record = receipts.request_record(request)
    assert (record["truncation"], record["context_overflow"]) == ("auto", "roll")
    restored = receipts.request_from_record(record)
    assert (restored.truncation, restored.context_overflow) == ("auto", "roll")
    assert engine.chat_completion(restored).content == engine.chat_completion(request).content


# -- front ends ---------------------------------------------------------------------------------------------------


def test_openai_and_responses_report_and_handle_long_prompts(client):
    messages = [{"role": m.role, "content": m.content} for m in _conversation() if m.role != "tool"]
    body = {"model": "m", "messages": messages, "max_tokens": 8}
    error = client.post("/v1/chat/completions", json=body)
    assert error.status_code == 400
    assert error.json()["error"]["code"] == "context_length_exceeded"
    assert error.json()["error"]["param"] == "messages"
    ok = client.post("/v1/chat/completions", json={**body, "truncation": "auto"}).json()
    assert ok["choices"][0]["finish_reason"] in ("stop", "length")
    rolled = client.post(
        "/v1/chat/completions", json={**body, "truncation": "auto", "context_overflow": "roll", "max_tokens": 60}
    ).json()
    assert rolled["usage"]["completion_tokens"] == 60
    assert client.post("/v1/chat/completions", json={**body, "truncation": "middle"}).status_code in (400, 422)

    request = {"model": "m", "input": messages, "max_output_tokens": 8}
    error = client.post("/v1/responses", json=request)
    assert error.status_code == 400 and error.json()["error"]["code"] == "context_length_exceeded"
    response = client.post("/v1/responses", json={**request, "truncation": "auto"}).json()
    assert response["truncation"] == "auto" and response["status"] in ("completed", "incomplete")
    plain = client.post("/v1/responses", json={**request, "input": "Hi"}).json()
    assert plain["truncation"] == "disabled"


def test_ollama_truncate_and_shift(client):
    messages = [{"role": m.role, "content": m.content} for m in _conversation() if m.role != "tool"]
    body = {"model": "m", "messages": messages, "stream": False, "options": {"num_predict": 8}}
    assert client.post("/api/chat", json=body).status_code == 400
    reply = client.post("/api/chat", json={**body, "truncate": True}).json()
    assert reply["done"] is True
    shifted = {**body, "truncate": True, "shift": True, "options": {"num_predict": 60}}
    assert client.post("/api/chat", json=shifted).json()["eval_count"] == 60
    generate = {"model": "m", "prompt": "z" * 40, "stream": False, "raw": True, "options": {"num_predict": 20}}
    assert client.post("/api/generate", json=generate).json()["eval_count"] == WINDOW - 40
    assert client.post("/api/generate", json={**generate, "shift": True}).json()["eval_count"] == 20


def test_the_command_line(model_path, capsys, isolated_environment):  # noqa: F811
    model = ["--model", str(model_path)]
    assert main([*model, "generate", "--prompt", "w" * 40, "--max-tokens", "20"]) == 0
    assert f"tokens: {WINDOW - 40}  finish: length" in capsys.readouterr().err
    assert main([*model, "generate", "--prompt", "w" * 40, "--max-tokens", "20", "--context-overflow", "roll"]) == 0
    assert "tokens: 20  finish: length" in capsys.readouterr().err
    assert main([*model, "generate", "--prompt", "w" * WINDOW]) == 2
    assert "context window holds" in capsys.readouterr().err
    long = ["chat", "m" * 60, "--max-tokens", "4"]
    assert main([*model, *long]) == 1
    assert main([*model, *long, "--system", "s" * 10, "--truncate"]) == 1  # the last message alone is too long
    assert "context window holds" in capsys.readouterr().err
    assert main([*model, "chat", "Hi", "--max-tokens", "4", "--truncate"]) == 0
    assert capsys.readouterr().out.strip()
