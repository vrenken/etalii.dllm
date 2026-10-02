import json

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


# -- Phase 4: the official OpenAI SDK against the server -----------------------------------------------------------

openai = pytest.importorskip("openai")

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "enum": ["Paris", "Rome"]}},
            "required": ["city"],
        },
    },
}


@pytest.fixture
def sdk(client):
    return openai.OpenAI(base_url="http://testserver/v1", api_key="unused", http_client=client)


def test_streaming_matches_the_non_streamed_response(sdk):
    kwargs = {"model": "m", "messages": REQUEST["messages"], "temperature": 0.8, "seed": 9, "max_tokens": 30}
    full = sdk.chat.completions.create(**kwargs)
    chunks = list(sdk.chat.completions.create(**kwargs, stream=True, stream_options={"include_usage": True}))
    assert {c.id for c in chunks} == {full.id}
    assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices) == full.choices[0].message.content
    assert [c.choices[0].finish_reason for c in chunks if c.choices][-1] == full.choices[0].finish_reason
    assert chunks[-1].usage == full.usage
    again = list(sdk.chat.completions.create(**kwargs, stream=True, stream_options={"include_usage": True}))
    assert [c.model_dump() for c in again] == [c.model_dump() for c in chunks]


def test_raw_stream_is_server_sent_events(client):
    response = client.post("/v1/chat/completions", json={**REQUEST, "stream": True})
    assert response.headers["content-type"].startswith("text/event-stream")
    events = response.text.split("\n\n")
    assert events[0].startswith("data: {") and events[-2] == "data: [DONE]"


def test_logprobs(sdk):
    reply = sdk.chat.completions.create(
        model="m", messages=REQUEST["messages"], max_tokens=5, logprobs=True, top_logprobs=3
    )
    content = reply.choices[0].logprobs.content
    assert len(content) == 5
    assert all(len(entry.top_logprobs) == 3 and entry.logprob <= 0 for entry in content)
    assert content[0].token == content[0].top_logprobs[0].token  # greedy


def test_tool_calls(sdk):
    reply = sdk.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "Weather in Paris?"}], tools=[WEATHER_TOOL],
        tool_choice={"type": "function", "function": {"name": "get_weather"}}, max_tokens=100,
    )  # fmt: skip
    choice = reply.choices[0]
    assert choice.finish_reason == "tool_calls"
    (call,) = choice.message.tool_calls
    assert call.function.name == "get_weather"
    assert json.loads(call.function.arguments)["city"] in ("Paris", "Rome")

    # The result goes back as a tool message, and the conversation continues.
    follow_up = sdk.chat.completions.create(
        model="m", tools=[WEATHER_TOOL], max_tokens=10,
        messages=[
            {"role": "user", "content": "Weather in Paris?"},
            choice.message.model_dump(exclude_none=True),
            {"role": "tool", "tool_call_id": call.id, "content": "sunny, 21 C"},
        ],
    )  # fmt: skip
    assert follow_up.usage.prompt_tokens > reply.usage.prompt_tokens


def test_streamed_tool_calls(sdk):
    kwargs = {
        "model": "m", "messages": [{"role": "user", "content": "Weather?"}], "tools": [WEATHER_TOOL],
        "tool_choice": "required", "max_tokens": 100,
    }  # fmt: skip
    full = sdk.chat.completions.create(**kwargs)
    chunks = list(sdk.chat.completions.create(**kwargs, stream=True))
    streamed = [call for c in chunks if c.choices for call in c.choices[0].delta.tool_calls or ()]
    assert [(c.id, c.function.name, c.function.arguments) for c in streamed] == [
        (c.id, c.function.name, c.function.arguments) for c in full.choices[0].message.tool_calls
    ]


def test_structured_output(sdk):
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    reply = sdk.chat.completions.create(
        model="m", messages=REQUEST["messages"], max_tokens=100,
        response_format={"type": "json_schema", "json_schema": {"name": "answer", "schema": schema, "strict": True}},
    )  # fmt: skip
    assert isinstance(json.loads(reply.choices[0].message.content)["ok"], bool)
    reply = sdk.chat.completions.create(
        model="m", messages=REQUEST["messages"], max_tokens=200, response_format={"type": "json_object"}
    )
    if reply.choices[0].finish_reason == "stop":
        assert isinstance(json.loads(reply.choices[0].message.content), dict)


def test_embeddings(sdk, client):
    floats = sdk.embeddings.create(model="m", input=["hello", "world"], encoding_format="float")
    packed = sdk.embeddings.create(model="m", input=["hello", "world"])  # the SDK asks for base64 and decodes it
    assert [d.embedding for d in floats.data] == [d.embedding for d in packed.data]
    assert [d.index for d in floats.data] == [0, 1]
    assert floats.usage.prompt_tokens == len("hello") + len("world")
    assert sum(v * v for v in floats.data[0].embedding) == pytest.approx(1.0, abs=1e-5)
    single = client.post("/v1/embeddings", json={"input": "hello", "dimensions": 8}).json()
    assert len(single["data"][0]["embedding"]) == 8
    assert client.post("/v1/embeddings", json={"input": ""}).status_code == 400


@pytest.mark.parametrize(
    "body",
    [
        {"messages": [{"role": "user", "content": "x"}], "n": 17},
        {"messages": [{"role": "user", "content": "x"}], "tool_choice": "required"},
        {"messages": [{"role": "user", "content": "x"}], "response_format": {"type": "json_schema"}},
        {"messages": [{"role": "user", "content": "x"}], "logprobs": True, "top_logprobs": 50},
        {"messages": [{"role": "robot", "content": "x"}]},
        {"messages": "not a list"},
        {
            "messages": [{"role": "user", "content": "x"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": {"type": "string", "uniqueItems": True}},
            },
        },
    ],
)
def test_invalid_requests_are_400(client, body):
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_serves_the_chat_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    page = response.text
    assert "<title>EtAlii.Dllm chat</title>" in page
    # Relative URLs (works behind a path prefix) to this server's own API, and nothing loaded from elsewhere.
    assert 'fetch("v1/chat/completions"' in page and 'fetch("v1/models")' in page
    assert "http://" not in page and "https://" not in page
    # Model output is only ever inserted as text.
    assert "innerHTML" not in page
