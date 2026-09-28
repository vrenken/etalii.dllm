"""End to end: import a tiny model with a real BPE tokenizer and chat template, and serve it through the engine,
the CLI, the OpenAI-compatible server and the MCP tools."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from model_fixtures import TINY_LLAMA_CONFIG, write_hf_checkpoint
from test_chat_template import SMOLLM2

from etalii_dllm import engine as engine_module
from etalii_dllm.chat import ChatMessage
from etalii_dllm.cli import main as cli
from etalii_dllm.engine import DllmEngine, default_engine
from etalii_dllm.importing import import_model
from etalii_dllm.sampling import GREEDY, SamplingOptions

tokenizers = pytest.importorskip("tokenizers")


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    from test_bpe import smollm2_style

    reference = smollm2_style()
    spec = json.loads(reference.to_str())
    directory = tmp_path_factory.mktemp("e2e")
    config = {**TINY_LLAMA_CONFIG, "vocab_size": reference.get_vocab_size(), "eos_token_id": 2}
    write_hf_checkpoint(directory / "checkpoint", config, tokenizer_json=spec)
    tokenizer_config = {"chat_template": SMOLLM2, "eos_token": "<|im_end|>", "bos_token": None}
    (directory / "checkpoint" / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")
    import_model(directory / "checkpoint", directory / "tiny.dllm", repository="example/tiny-chat")
    return directory / "tiny.dllm"


@pytest.fixture
def served(model_path, monkeypatch):
    monkeypatch.setenv(engine_module.MODEL_ENVIRONMENT_VARIABLE, str(model_path))
    default_engine.cache_clear()
    yield default_engine()
    default_engine.cache_clear()


def test_engine_uses_the_model_tokenizer_and_template(model_path):
    engine = DllmEngine.from_model_file(model_path)
    assert engine.model.id == "example/tiny-chat"
    messages = [ChatMessage("user", "Hello there")]
    assert engine.render_chat(messages).startswith("<|im_start|>system\nYou are a helpful AI assistant")
    first = engine.chat(messages, 12, SamplingOptions(temperature=0.9, seed=3))
    second = DllmEngine.from_model_file(model_path).chat(messages, 12, SamplingOptions(temperature=0.9, seed=3))
    assert first == second
    assert first.prompt_tokens == len(engine.tokenizer.encode(engine.render_chat(messages)))


def test_generation_stops_on_end_of_sequence(model_path):
    engine = DllmEngine.from_model_file(model_path)
    im_end = engine.tokenizer.end_of_sequence
    for seed in range(40):
        result = engine.complete("The quick brown fox", 30, SamplingOptions(temperature=2.0, seed=seed))
        assert im_end not in result.tokens and 2 not in result.tokens
        if result.finish_reason == "stop":
            return
    pytest.skip("no sampled run hit an end-of-sequence token")


def test_concurrent_requests_match_sequential(served):
    prompts = [f"prompt {i}" for i in range(6)]
    expected = [served.complete(p, 8, GREEDY).fingerprint for p in prompts]
    with ThreadPoolExecutor(max_workers=6) as pool:
        assert list(pool.map(lambda p: served.complete(p, 8, GREEDY).fingerprint, prompts)) == expected


def test_front_ends_agree(served, model_path, capsys, monkeypatch):
    from etalii_dllm.mcp_server import generate
    from etalii_dllm.server.app import app

    expected = served.complete("Deterministic", 10, GREEDY).text
    assert generate("Deterministic", max_tokens=10) == expected

    body = {"model": "x", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 10, "temperature": 0}
    response = TestClient(app).post("/v1/chat/completions", json=body).json()
    assert response["model"] == "example/tiny-chat"
    assert response["system_fingerprint"] == served.system_fingerprint
    assert response["choices"][0]["message"]["content"] == served.chat([ChatMessage("user", "Hi")], 10, GREEDY).text

    monkeypatch.setenv(engine_module.MODEL_ENVIRONMENT_VARIABLE, "")
    assert cli(["--model", str(model_path), "generate", "--prompt", "Deterministic", "--max-tokens", "10"]) == 0
    assert capsys.readouterr().out == expected + "\n"
    assert cli(["--model", str(model_path), "chat", "Hi", "--max-tokens", "10"]) == 0
    assert capsys.readouterr().out == response["choices"][0]["message"]["content"] + "\n"
