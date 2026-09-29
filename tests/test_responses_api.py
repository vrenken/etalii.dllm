"""The OpenAI Responses API, driven with the official ``openai`` client."""

from __future__ import annotations

import json

import openai
import pytest
from fastapi.testclient import TestClient

from etalii_dllm.server import responses_api
from etalii_dllm.server.app import app

WEATHER = {
    "type": "function",
    "name": "weather",
    "description": "Current weather in a city",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string", "enum": ["Paris", "Rome"]}},
        "required": ["city"],
    },
}
SAMPLING = {"temperature": 0.7, "seed": 3}


@pytest.fixture
def client():
    responses_api.store.clear()
    yield TestClient(app)
    responses_api.store.clear()


@pytest.fixture
def sdk(client):
    return openai.OpenAI(base_url="http://testserver/v1", api_key="unused", http_client=client)


def test_same_answer_as_chat_completions(sdk, client):
    response = sdk.responses.create(
        model="m", instructions="Be brief.", input="Say something.", max_output_tokens=20, extra_body=SAMPLING
    )
    again = sdk.responses.create(
        model="m", instructions="Be brief.", input="Say something.", max_output_tokens=20, extra_body=SAMPLING
    )
    assert response.model_dump() == again.model_dump()
    assert response.id.startswith("resp_") and response.created_at == 0
    assert response.status == "incomplete" and response.incomplete_details.reason == "max_output_tokens"
    messages = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Say something."}]
    body = {"model": "m", "messages": messages, "max_tokens": 20, **SAMPLING}
    chat = client.post("/v1/chat/completions", json=body).json()
    assert response.output_text == chat["choices"][0]["message"]["content"]
    assert response.usage.input_tokens == chat["usage"]["prompt_tokens"]
    assert response.usage.output_tokens == 20


def test_streaming_matches_the_non_streamed_response(sdk):
    kwargs = {"model": "m", "input": "Count to three.", "max_output_tokens": 16, "extra_body": SAMPLING}
    full = sdk.responses.create(**kwargs)
    events = list(sdk.responses.create(**kwargs, stream=True))
    assert [e.sequence_number for e in events] == list(range(len(events)))
    assert [e.type for e in events[:4]] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
    ]
    assert events[-1].type == "response.incomplete"
    assert events[-1].response.model_dump() == full.model_dump()
    deltas = "".join(e.delta for e in events if e.type == "response.output_text.delta")
    assert deltas == full.output_text
    done = [e for e in events if e.type == "response.output_text.done"]
    assert [e.text for e in done] == [full.output_text]


def test_previous_response_id_continues_the_conversation(sdk, client):
    first = sdk.responses.create(model="m", input="My name is Ada.", max_output_tokens=8)
    follow_up = sdk.responses.create(model="m", input="What is my name?", previous_response_id=first.id,
                                     max_output_tokens=8)  # fmt: skip
    assert follow_up.previous_response_id == first.id
    messages = [
        {"role": "user", "content": "My name is Ada."},
        {"role": "assistant", "content": first.output_text},
        {"role": "user", "content": "What is my name?"},
    ]
    chat = client.post("/v1/chat/completions", json={"model": "m", "messages": messages, "max_tokens": 8}).json()
    assert follow_up.output_text == chat["choices"][0]["message"]["content"]
    assert sdk.responses.retrieve(first.id).model_dump() == first.model_dump()
    sdk.responses.delete(first.id)
    with pytest.raises(openai.NotFoundError):
        sdk.responses.retrieve(first.id)
    with pytest.raises(openai.BadRequestError, match="not found"):
        sdk.responses.create(model="m", input="Again?", previous_response_id=first.id)


def test_store_false_is_not_kept(sdk):
    response = sdk.responses.create(model="m", input="Hi", max_output_tokens=4, store=False)
    assert response.store is False
    with pytest.raises(openai.NotFoundError):
        sdk.responses.retrieve(response.id)


def test_function_calls_and_their_outputs(sdk):
    response = sdk.responses.create(
        model="m", input="Weather in Paris?", tools=[WEATHER], tool_choice="required", max_output_tokens=200
    )
    calls = [item for item in response.output if item.type == "function_call"]
    assert calls and calls[0].name == "weather" and json.loads(calls[0].arguments)["city"] in ("Paris", "Rome")
    assert calls[0].call_id.startswith("call_") and calls[0].id.startswith("fc_")
    streamed = list(
        sdk.responses.create(
            model="m", input="Weather in Paris?", tools=[WEATHER], tool_choice="required", max_output_tokens=200,
            stream=True,
        )
    )  # fmt: skip
    assert streamed[-1].response.model_dump() == response.model_dump()
    arguments = [e.arguments for e in streamed if e.type == "response.function_call_arguments.done"]
    assert arguments == [c.arguments for c in calls]

    # Hand the result back, both by previous_response_id and by replaying the items.
    result = {"type": "function_call_output", "call_id": calls[0].call_id, "output": "18 degrees"}
    chained = sdk.responses.create(
        model="m", input=[result], previous_response_id=response.id, tools=[WEATHER], max_output_tokens=10
    )
    replayed = sdk.responses.create(
        model="m",
        input=[
            {"role": "user", "content": "Weather in Paris?"},
            *[{"type": "function_call", "call_id": c.call_id, "name": c.name, "arguments": c.arguments} for c in calls],
            result,
        ],
        tools=[WEATHER],
        max_output_tokens=10,
    )
    assert chained.output_text == replayed.output_text
    assert chained.usage.input_tokens == replayed.usage.input_tokens


def test_structured_output_and_logprobs(sdk):
    schema = {
        "type": "object",
        "properties": {"n": {"type": "integer"}},
        "required": ["n"],
        "additionalProperties": False,
    }
    response = sdk.responses.create(
        model="m", input="A number?", max_output_tokens=60,
        text={"format": {"type": "json_schema", "name": "number", "schema": schema}},
    )  # fmt: skip
    if response.status == "completed":
        assert isinstance(json.loads(response.output_text)["n"], int)
    scored = sdk.responses.create(
        model="m", input="Hi", max_output_tokens=5, include=["message.output_text.logprobs"], top_logprobs=2
    )
    logprobs = scored.output[0].content[0].logprobs
    assert len(logprobs) == 5 and all(len(entry.top_logprobs) == 2 for entry in logprobs)


def test_invalid_requests_are_400(client):
    assert client.post("/v1/responses", json={"model": "m", "input": []}).status_code == 400
    image = {"role": "user", "content": [{"type": "input_image", "image_url": "https://example.com/x.png"}]}
    bad = client.post("/v1/responses", json={"model": "m", "input": [image]})
    assert bad.status_code == 400 and "input_image" in bad.json()["error"]["message"]
    search = client.post("/v1/responses", json={"model": "m", "input": "x", "tools": [{"type": "web_search"}]})
    assert search.status_code == 400
