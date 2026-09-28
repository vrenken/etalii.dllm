import pytest
from fastapi.testclient import TestClient

from etalii_dllm.server.app import app

REQUEST = {
    "model": "dllm-bigram-257-42",
    "messages": [{"role": "user", "content": "Say something deterministic."}],
    "temperature": 0.7,
    "seed": 5,
    "max_tokens": 24,
}


@pytest.fixture
def client():
    return TestClient(app)


def test_lists_models(client):
    body = client.get("/v1/models").json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "dllm-bigram-257-42"


def test_identical_requests_return_identical_responses(client):
    first = client.post("/v1/chat/completions", json=REQUEST)
    second = client.post("/v1/chat/completions", json=REQUEST)
    assert first.status_code == 200
    assert first.content == second.content

    body = first.json()
    assert body["object"] == "chat.completion"
    assert body["system_fingerprint"].startswith("fp_")
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["usage"]["completion_tokens"] == 24


def test_rejects_empty_messages(client):
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 400
