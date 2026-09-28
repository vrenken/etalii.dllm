"""The Anthropic Messages endpoint, driven by the official ``anthropic`` SDK."""

import json

import pytest
from fastapi.testclient import TestClient

from etalii_dllm.server.app import app

anthropic = pytest.importorskip("anthropic")

WEATHER = {
    "name": "get_weather",
    "description": "Current weather for a city",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string", "enum": ["Paris", "Rome"]}},
        "required": ["city"],
    },
}
HELLO = [{"role": "user", "content": "Say something deterministic."}]


def blocks(message) -> list[dict]:
    return [block.to_dict(exclude_none=True) for block in message.content]


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sdk(client):
    return anthropic.Anthropic(base_url="http://testserver", api_key="unused", http_client=client)


def test_message(sdk):
    kwargs = {"model": "m", "max_tokens": 16, "messages": HELLO, "extra_body": {"temperature": 0.7, "seed": 5}}
    first = sdk.messages.create(**kwargs)
    second = sdk.messages.create(**kwargs)
    assert first.to_dict() == second.to_dict()
    assert first.id.startswith("msg_") and first.type == "message" and first.role == "assistant"
    assert first.model == "dllm-bigram-257-42"
    assert first.stop_reason == "max_tokens" and first.usage.output_tokens == 16
    assert first.content[0].type == "text"


def test_stream_matches_create(sdk):
    kwargs = {"model": "m", "max_tokens": 24, "messages": HELLO, "extra_body": {"temperature": 0.9, "seed": 1}}
    created = sdk.messages.create(**kwargs)
    with sdk.messages.stream(**kwargs) as stream:
        text = "".join(stream.text_stream)
        final = stream.get_final_message()
    assert text == created.content[0].text
    assert final.id == created.id
    assert blocks(final) == blocks(created)
    assert (final.stop_reason, final.usage.input_tokens, final.usage.output_tokens) == (
        created.stop_reason,
        created.usage.input_tokens,
        created.usage.output_tokens,
    )


def test_raw_stream_events(client):
    body = {"max_tokens": 4, "messages": HELLO, "stream": True}
    kinds = [line[len("event: ") :] for line in client.post("/v1/messages", json=body).text.splitlines()
             if line.startswith("event: ")]  # fmt: skip
    assert kinds[0] == "message_start" and kinds[-2:] == ["message_delta", "message_stop"]
    assert "content_block_start" in kinds and "content_block_stop" in kinds


def test_stop_sequences(sdk):
    text = sdk.messages.create(model="m", max_tokens=40, messages=HELLO, extra_body={"temperature": 1.0}).content[0]
    start = next(i for i in range(2, len(text.text) - 1) if text.text[i : i + 2].isascii() and text.text[i].isalnum())
    stop = text.text[start : start + 2]
    reply = sdk.messages.create(
        model="m", max_tokens=40, messages=HELLO, stop_sequences=[stop], extra_body={"temperature": 1.0}
    )
    assert reply.stop_reason == "stop_sequence" and reply.stop_sequence == stop
    assert reply.content[0].text == text.text[: text.text.find(stop)]


def test_tool_use_round_trip(sdk):
    reply = sdk.messages.create(
        model="m", max_tokens=100, messages=[{"role": "user", "content": "Weather in Rome?"}], tools=[WEATHER],
        tool_choice={"type": "tool", "name": "get_weather"},
    )  # fmt: skip
    assert reply.stop_reason == "tool_use"
    (use,) = [block for block in reply.content if block.type == "tool_use"]
    assert use.name == "get_weather" and use.input["city"] in ("Paris", "Rome") and use.id.startswith("toolu_")

    with sdk.messages.stream(
        model="m", max_tokens=100, messages=[{"role": "user", "content": "Weather in Rome?"}], tools=[WEATHER],
        tool_choice={"type": "tool", "name": "get_weather"},
    ) as stream:  # fmt: skip
        assert blocks(stream.get_final_message()) == blocks(reply)

    follow_up = sdk.messages.create(
        model="m", max_tokens=8, tools=[WEATHER],
        messages=[
            {"role": "user", "content": "Weather in Rome?"},
            {"role": "assistant", "content": [block.to_dict() for block in reply.content]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": use.id, "content": "sunny"}]},
        ],
    )  # fmt: skip
    assert follow_up.usage.input_tokens > reply.usage.input_tokens


def test_structured_output(sdk):
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    reply = sdk.messages.create(
        model="m", max_tokens=60, messages=HELLO, output_config={"format": {"type": "json_schema", "schema": schema}}
    )
    assert reply.stop_reason == "end_turn"
    assert isinstance(json.loads(reply.content[0].text)["n"], int)


def test_count_tokens(sdk):
    counted = sdk.messages.count_tokens(model="m", messages=HELLO, system="Be brief.")
    reply = sdk.messages.create(model="m", max_tokens=1, messages=HELLO, system="Be brief.")
    assert counted.input_tokens == reply.usage.input_tokens


def test_system_blocks_and_prefill(sdk):
    reply = sdk.messages.create(
        model="m", max_tokens=5, system=[{"type": "text", "text": "Be brief."}],
        messages=[*HELLO, {"role": "assistant", "content": "Sure:"}],
    )  # fmt: skip
    assert reply.usage.output_tokens == 5


@pytest.mark.parametrize(
    "body",
    [
        {"messages": HELLO},  # max_tokens is required
        {"max_tokens": 5, "messages": [{"role": "user", "content": [{"type": "image", "source": {}}]}]},
        {"max_tokens": 5, "messages": HELLO, "tools": [{"type": "web_search_20250305", "name": "web_search"}]},
        {"max_tokens": 5, "messages": HELLO, "tool_choice": {"type": "any"}},
    ],
)
def test_invalid_requests(client, body):
    response = client.post("/v1/messages", json=body)
    assert response.status_code == 400
    assert response.json()["type"] == "error"
    assert response.json()["error"]["type"] == "invalid_request_error"
