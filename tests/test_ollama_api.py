"""The Ollama API, driven with the official ``ollama`` Python client."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from test_engine_import import model_path, served  # noqa: F401 - fixtures

from etalii_dllm.chat import ToolCall
from etalii_dllm.engine import ChatStream, Finished, TextDelta, ToolCallEvent
from etalii_dllm.server import ollama_api
from etalii_dllm.server.app import app

ollama = pytest.importorskip("ollama")

MESSAGES = [{"role": "user", "content": "Say something deterministic."}]


@pytest.fixture
def client():
    return TestClient(app)


def _sdk(client: TestClient):
    """The official client, sending its requests through the in-process test client."""
    sdk = ollama.Client(host="http://testserver")
    sdk._client = client
    return sdk


@pytest.fixture
def sdk(client):
    return _sdk(client)


def test_chat_is_deterministic_and_matches_the_openai_endpoint(sdk, client):
    options = {"temperature": 0.7, "seed": 5, "num_predict": 24}
    first = sdk.chat(model="anything", messages=MESSAGES, options=options)
    second = sdk.chat(model="anything", messages=MESSAGES, options=options)
    assert first.model_dump() == second.model_dump()
    assert first.done and first.done_reason == "length" and first.eval_count == 24
    assert first.created_at == "1970-01-01T00:00:00Z"
    body = {"model": "m", "messages": MESSAGES, "temperature": 0.7, "seed": 5, "max_tokens": 24}
    openai = client.post("/v1/chat/completions", json=body).json()
    assert first.message.content == openai["choices"][0]["message"]["content"]
    assert first.prompt_eval_count == openai["usage"]["prompt_tokens"]


def test_streaming_matches_the_non_streamed_response(sdk, client):
    options = {"temperature": 0.9, "seed": 3, "num_predict": 20}
    full = sdk.chat(model="m", messages=MESSAGES, options=options)
    chunks = list(sdk.chat(model="m", messages=MESSAGES, options=options, stream=True))
    assert "".join(c.message.content for c in chunks) == full.message.content
    assert [c.done for c in chunks] == [False] * (len(chunks) - 1) + [True]
    assert chunks[-1].eval_count == full.eval_count
    raw = client.post("/api/chat", json={"model": "m", "messages": MESSAGES, "options": options})
    assert raw.headers["content-type"].startswith("application/x-ndjson")
    lines = raw.text.splitlines()
    assert all(json.loads(line)["model"] == "dllm-bigram-257-42" for line in lines)


def test_generate(sdk):
    options = {"seed": 1, "temperature": 0.5, "num_predict": 12}
    result = sdk.generate(model="m", prompt="Once upon a time", options=options)
    chunks = list(sdk.generate(model="m", prompt="Once upon a time", options=options, stream=True))
    assert "".join(c.response for c in chunks) == result.response
    assert result.eval_count == 12
    raw = sdk.generate(model="m", prompt="Once upon a time", raw=True, options=options)
    assert raw.prompt_eval_count == len("Once upon a time")  # byte tokenizer: no template around the prompt
    loaded = sdk.generate(model="m")
    assert loaded.done and loaded.done_reason == "load"


def test_structured_output_and_logprobs(sdk):
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    reply = sdk.chat(model="m", messages=MESSAGES, format=schema, options={"num_predict": 60, "seed": 2})
    if reply.done_reason == "stop":
        assert isinstance(json.loads(reply.message.content)["n"], int)
    as_json = sdk.chat(model="m", messages=MESSAGES, format="json", options={"num_predict": 60})
    assert as_json.message.content.lstrip().startswith("{")
    scored = sdk.chat(model="m", messages=MESSAGES, logprobs=True, top_logprobs=2, options={"num_predict": 5})
    assert len(scored.logprobs) == 5
    assert all(len(entry.top_logprobs) == 2 and entry.logprob <= 0 for entry in scored.logprobs)


def test_embeddings(sdk):
    both = sdk.embed(model="m", input=["hello", "world"])
    assert len(both.embeddings) == 2 and both.prompt_eval_count == len("hello") + len("world")
    one = sdk.embed(model="m", input="hello")
    assert one.embeddings[0] == both.embeddings[0]
    assert sdk.embeddings(model="m", prompt="hello").embedding == one.embeddings[0]


def test_model_listing(sdk, client):
    listed = sdk.list()
    assert [m.model for m in listed.models] == ["dllm-bigram-257-42"]
    assert listed.models[0].details.format == "dllm"
    assert sdk.show("m").capabilities == ["completion", "tools", "embedding"]
    assert sdk.ps().models[0].model == "dllm-bigram-257-42"
    assert client.get("/api/version").json()["version"]


def test_errors_use_the_ollama_shape(client):
    assert client.post("/api/chat", json={"model": "m", "messages": []}).json() == {
        "error": "'messages' must contain at least one message"
    }
    bad = client.post("/api/chat", json={"model": "m", "messages": [{"role": "user", "content": "x", "images": ["a"]}]})
    assert bad.status_code == 400 and bad.json() == {"error": "images are not supported"}
    invalid = client.post("/api/chat", json={"model": "m", "messages": "nope"})
    assert invalid.status_code == 400 and "error" in invalid.json()


def test_tool_history_gets_call_ids():
    messages = ollama_api._messages(
        [
            ollama_api.OllamaMessage(role="user", content="Weather in Paris and Rome?"),
            ollama_api.OllamaMessage(
                role="assistant",
                tool_calls=[
                    ollama_api.OllamaToolCall(function=ollama_api.OllamaFunctionCall(name="w", arguments={"c": "P"})),
                    ollama_api.OllamaToolCall(function=ollama_api.OllamaFunctionCall(name="t", arguments={})),
                ],
            ),
            ollama_api.OllamaMessage(role="tool", content="18C", tool_name="t"),
            ollama_api.OllamaMessage(role="tool", content="21C"),
        ]
    )
    assert [(c.id, c.name, c.arguments) for c in messages[1].tool_calls] == [
        ("call_0", "w", '{"c": "P"}'),
        ("call_1", "t", "{}"),
    ]
    assert [(m.tool_call_id, m.name) for m in messages[2:]] == [("call_1", "t"), ("call_0", None)]


def test_tool_calls_are_streamed_and_collected(served):  # noqa: F811
    events = iter(
        [
            TextDelta("Let me check."),
            ToolCallEvent(0, ToolCall("call_x", "weather", '{"city": "Paris"}')),
            Finished("tool_calls", None, 9, "fp"),
        ]
    )
    chunks = list(ollama_api._chunks(served, ChatStream(12, events, cached_tokens=4), True, False))
    call = {"function": {"name": "weather", "arguments": {"city": "Paris"}}}
    assert chunks[1]["message"]["tool_calls"] == [call]
    final = ollama_api._collect(iter(chunks), True)
    assert final["message"] == {"role": "assistant", "content": "Let me check.", "tool_calls": [call]}
    assert (final["done_reason"], final["prompt_eval_count"], final["eval_count"]) == ("stop", 8, 9)


def test_real_template_and_tools(served):  # noqa: F811
    client = TestClient(app)
    sdk = _sdk(client)
    parameters = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    tool = {"type": "function", "function": {"name": "weather", "description": "Weather", "parameters": parameters}}
    history = [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "weather",
                                                                          "arguments": {"city": "Paris"}}}]},
        {"role": "tool", "content": "18 degrees", "tool_name": "weather"},
    ]  # fmt: skip
    reply = sdk.chat(model="m", messages=history, tools=[tool], options={"num_predict": 10})
    openai_history = [
        history[0],
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_0", "type": "function", "function": {"name": "weather", "arguments": '{"city": "Paris"}'}}]},
        {"role": "tool", "content": "18 degrees", "tool_call_id": "call_0", "name": "weather"},
    ]  # fmt: skip
    body = {"model": "m", "messages": openai_history, "tools": [tool], "max_tokens": 10}
    openai = client.post("/v1/chat/completions", json=body).json()
    assert reply.message.content == (openai["choices"][0]["message"]["content"] or "")
    assert reply.model == "example/tiny-chat"
    assert sdk.show("m").template == served.chat_template.source
