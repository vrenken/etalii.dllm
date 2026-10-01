"""Error paths and unusual inputs of the HTTP front ends: every API reports invalid requests in its own error shape,
accepts the content blocks its clients send, and streams unusual event sequences as well-formed events."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from test_engine_import import model_path, served  # noqa: F401 - fixtures

from etalii_dllm.chat import ChatMessage, ToolCall
from etalii_dllm.engine import ChatRequest, ChatStream, DllmEngine, Finished, TextDelta, ToolCallEvent, default_engine
from etalii_dllm.server import anthropic_api, ollama_api, responses_api
from etalii_dllm.server import app as app_module
from etalii_dllm.server.anthropic_contracts import InputMessage
from etalii_dllm.server.app import app

anthropic = pytest.importorskip("anthropic")
openai = pytest.importorskip("openai")
ollama = pytest.importorskip("ollama")

USER = [{"role": "user", "content": "Say something deterministic."}]


@pytest.fixture
def client():
    responses_api.store.clear()
    yield TestClient(app)
    responses_api.store.clear()


@pytest.fixture
def claude(client):
    return anthropic.Anthropic(base_url="http://testserver", api_key="unused", http_client=client)


@pytest.fixture
def gpt(client):
    return openai.OpenAI(base_url="http://testserver/v1", api_key="unused", http_client=client)


@pytest.fixture
def llama(client):
    sdk = ollama.Client(host="http://testserver")
    sdk._client = client
    return sdk


def _sse(text: str) -> list[dict]:
    return [json.loads(line[len("data: ") :]) for line in text.splitlines() if line.startswith("data: {")]


# -- Anthropic Messages -----------------------------------------------------------------------------------------------


def _anthropic_error(client: TestClient, path: str, body: dict) -> str:
    response = client.post(path, json=body)
    assert response.status_code == 400
    payload = response.json()
    assert payload["type"] == "error" and payload["error"]["type"] == "invalid_request_error"
    return payload["error"]["message"]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"max_tokens": 8, "messages": []}, "at least one message"),
        ({"max_tokens": 0, "messages": USER}, "max_tokens must be at least 1"),
        ({"max_tokens": 8, "messages": USER, "tool_choice": {"type": "tool"}}, "needs a name"),
        ({"max_tokens": 8, "messages": USER, "system": [{"type": "image"}]}, "'image' are not supported here"),
        (
            {"max_tokens": 8, "messages": [{"role": "user", "content": [{"type": "document"}]}]},
            "'document' are not supported",
        ),
        (
            {"max_tokens": 8, "messages": [*USER, {"role": "assistant", "content": [{"type": "tool_use"}]}]},
            "tool_use blocks need an id and a name",
        ),
        (
            {"max_tokens": 8, "messages": USER, "tools": [{"name": "search", "type": "web_search_20250305"}]},
            "server tools",
        ),
        ({"messages": USER}, "max_tokens"),
    ],
)
def test_anthropic_invalid_requests(client, body, message):
    assert message in _anthropic_error(client, "/v1/messages", body)


def test_anthropic_sdk_raises_bad_request(claude):
    with pytest.raises(anthropic.BadRequestError) as raised:
        claude.messages.create(model="m", max_tokens=0, messages=USER)
    assert raised.value.status_code == 400
    assert raised.value.body["error"]["message"] == "max_tokens must be at least 1"


def test_anthropic_count_tokens_rejects_unsupported_blocks(claude, client):
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
    with pytest.raises(anthropic.BadRequestError, match="'image' are not supported"):
        claude.messages.count_tokens(model="m", messages=[{"role": "user", "content": [image]}])
    assert "at least" not in _anthropic_error(
        client, "/v1/messages/count_tokens", {"messages": USER, "tool_choice": {"type": "tool"}}
    )


def test_anthropic_block_history(claude):
    """Text blocks, skipped thinking blocks, tool results (errors marked) and tool_choice none all translate."""
    history = [
        {"role": "user", "content": [{"type": "text", "text": "Weather"}, {"type": "text", "text": "in Paris?"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Use the tool.", "signature": "sig"},
                {"type": "redacted_thinking", "data": "opaque"},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "is_error": True,
                    "content": [{"type": "text", "text": "service down"}],
                }
            ],
        },
    ]
    tool = {"name": "get_weather", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}
    kwargs = {"model": "m", "max_tokens": 8, "messages": history, "tools": [tool], "tool_choice": {"type": "none"}}
    message = claude.messages.create(**kwargs)
    assert message.stop_reason in ("end_turn", "max_tokens")
    assert all(block.type == "text" for block in message.content)
    assert claude.messages.create(**kwargs) == message

    translated = anthropic_api._messages(None, [InputMessage.model_validate(m) for m in history])
    assert [(m.role, m.content) for m in translated] == [
        ("user", "Weather\nin Paris?"),
        ("assistant", ""),
        ("tool", "Error: service down"),
    ]
    assert translated[1].tool_calls == (ToolCall("toolu_1", "get_weather", '{"city": "Paris"}'),)
    assert (translated[2].tool_call_id, translated[2].name) == ("toolu_1", "get_weather")
    counted = claude.messages.count_tokens(model="m", messages=history, tools=[tool], tool_choice={"type": "none"})
    assert counted.input_tokens > 0


def test_anthropic_stream_closes_text_before_tool_use():
    engine = DllmEngine.create_default()
    events = [
        TextDelta(""),
        TextDelta("Checking."),
        ToolCallEvent(0, ToolCall("toolu_1", "get_weather", '{"city": "Rome"}')),
        Finished("tool_calls", None, 7, "fp"),
    ]
    chat = ChatRequest([ChatMessage("user", "x")], 8, request_id="msg_1")
    lines = "".join(anthropic_api._events(engine, chat, ChatStream(5, iter(events), cached_tokens=2)))
    kinds = [(e["type"], e.get("index")) for e in _sse(lines)]
    assert kinds == [
        ("message_start", None),
        ("content_block_start", 0),
        ("content_block_delta", 0),
        ("content_block_stop", 0),
        ("content_block_start", 1),
        ("content_block_delta", 1),
        ("content_block_stop", 1),
        ("message_delta", None),
        ("message_stop", None),
    ]
    parsed = _sse(lines)
    assert parsed[0]["message"]["usage"]["input_tokens"] == 3
    assert parsed[5]["delta"] == {"type": "input_json_delta", "partial_json": '{"city": "Rome"}'}
    assert parsed[7]["delta"]["stop_reason"] == "tool_use"


# -- OpenAI Chat Completions and embeddings ---------------------------------------------------------------------------


def test_openai_content_parts(gpt, client):
    parts = [{"type": "text", "text": "Say something "}, {"type": "text", "text": "deterministic."}]
    joined = gpt.chat.completions.create(model="m", messages=[{"role": "user", "content": parts}], max_tokens=8)
    plain = gpt.chat.completions.create(model="m", messages=USER, max_tokens=8)
    assert joined.choices[0].message.content == plain.choices[0].message.content

    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    with pytest.raises(openai.BadRequestError) as raised:
        gpt.chat.completions.create(model="m", messages=[{"role": "user", "content": [image]}])
    assert raised.value.body["message"] == "content parts of type 'image_url' are not supported"


def test_openai_rejects_too_many_stop_sequences(client):
    body = {"messages": USER, "stop": ["a", "b", "c", "d", "e"]}
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 400
    assert response.json() == {
        "error": {"message": "at most 4 stop sequences are supported", "type": "invalid_request_error"}
    }


def test_openai_validation_errors_use_the_openai_shape(client):
    response = client.post("/v1/chat/completions", json={"messages": USER, "temperature": "hot"})
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error" and error["message"].startswith("body.temperature")


def test_openai_embeddings_reject_an_empty_list(gpt):
    with pytest.raises(openai.BadRequestError, match="'input' must not be empty"):
        gpt.embeddings.create(model="m", input=[])


def test_server_main_serves_the_app(monkeypatch):
    calls = {}
    monkeypatch.setattr(app_module, "use_model_file", lambda *args, **kwargs: calls.setdefault("model", (args, kwargs)))
    monkeypatch.setattr("uvicorn.run", lambda application, **kwargs: calls.setdefault("run", (application, kwargs)))
    argv = ["dllm-server", "--port", "6000", "--model", "m.dllm", "--threads", "2", "--prompt-cache", "0"]
    argv += ["--steer", "v.json", "--steer-strength", "2", "--index", "d.index", "--speculate"]
    monkeypatch.setattr(sys, "argv", argv)
    app_module.main()
    expected = {"steer": "v.json", "steer_strength": 2.0, "index": "d.index", "index_top": None}
    expected |= {"embedding_model": None, "speculate": 8, "draft_model": None, "prompt_cache_dir": None}
    expected |= {"sign_key": None}
    assert calls["model"] == (("m.dllm", None, 2, None, 0, None), expected)
    assert calls["run"] == (app, {"host": "127.0.0.1", "port": 6000})


# -- OpenAI Responses -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"input": [{"type": "message", "role": "tool", "content": "x"}]}, "unknown message role 'tool'"),
        ({"input": [{"type": "function_call", "name": "f"}]}, "need a call_id and a name"),
        ({"input": [{"type": "file_search_call"}]}, "input items of type 'file_search_call' are not supported"),
        (
            {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "data:,"}]}]},
            "content parts of type 'input_image' are not supported",
        ),
        ({"input": "x", "tools": [{"type": "web_search"}]}, "only function tools are supported"),
        ({"input": "x", "tool_choice": {"type": "file_search"}}, "tool_choice must be"),
        ({"input": "x", "text": {"format": {"type": "json_schema", "name": "s"}}}, "needs a 'schema'"),
        ({"input": "x", "previous_response_id": "resp_missing"}, "'resp_missing' not found"),
        ({"input": []}, "'input' must not be empty"),
    ],
)
def test_responses_invalid_requests(client, body, message):
    response = client.post("/v1/responses", json=body)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error" and message in error["message"]


def test_responses_unknown_ids_are_404(gpt):
    with pytest.raises(openai.NotFoundError, match="'resp_nope' not found"):
        gpt.responses.delete("resp_nope")
    with pytest.raises(openai.NotFoundError):
        gpt.responses.retrieve("resp_nope")


def test_responses_item_history(gpt):
    """Content part lists, skipped reasoning items, function calls joining the assistant message before them and a
    named tool choice all translate."""
    items = [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Weather in "}, {"type": "input_text", "text": "Rome?"}],
        },
        {"type": "reasoning", "id": "rs_1", "summary": []},
        {"role": "assistant", "content": [{"type": "output_text", "text": "Checking."}]},
        {"type": "function_call", "call_id": "call_1", "name": "weather", "arguments": '{"city": "Rome"}'},
        {"type": "function_call", "call_id": "call_2", "name": "weather", "arguments": '{"city": "Paris"}'},
        {"type": "function_call_output", "call_id": "call_2", "output": "18C"},
    ]
    conversation = responses_api._conversation([responses_api.InputItem.model_validate(i) for i in items])
    assert [(m.role, m.content, len(m.tool_calls)) for m in conversation] == [
        ("user", "Weather in Rome?", 0),
        ("assistant", "Checking.", 2),
        ("tool", "18C", 0),
    ]
    assert conversation[2].name == "weather"

    tool = {"type": "function", "name": "weather", "parameters": {"type": "object"}}
    kwargs = {"model": "m", "input": items, "tools": [tool], "max_output_tokens": 256}
    response = gpt.responses.create(**kwargs, tool_choice={"type": "function", "name": "weather"})
    assert response.output[0].type == "function_call" and response.output[0].name == "weather"
    assert response.tool_choice.name == "weather"


def test_responses_store_keeps_the_most_recent():
    store = responses_api._Store(size=2)
    for name in ("a", "b", "c"):
        store.put({"id": name}, [])
    assert store.get("a") is None
    assert store.get("b") == ({"id": "b"}, [], None) and store.get("c") == ({"id": "c"}, [], None)
    store.put({"id": "b"}, [ChatMessage("user", "again")])
    store.put({"id": "d"}, [])
    assert store.get("c") is None and store.get("b") is not None


def test_responses_stream_closes_text_before_function_calls():
    engine = DllmEngine.create_default()
    request = responses_api.ResponsesRequest(input="x")
    chat = ChatRequest([ChatMessage("user", "x")], 8, request_id="resp_1")
    events = [
        TextDelta(""),
        TextDelta("Checking."),
        ToolCallEvent(0, ToolCall("call_1", "weather", '{"city": "Rome"}')),
        Finished("tool_calls", None, 4, "fp"),
    ]
    stream = responses_api._Events(request, chat, engine, ChatStream(6, iter(events)))
    kinds = [event["type"] for event in stream]
    assert kinds == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    output = stream.final["output"]
    assert [(item["type"], item["status"]) for item in output] == [
        ("message", "completed"),
        ("function_call", "completed"),
    ]
    assert output[0]["content"][0]["text"] == "Checking."
    assert stream.final["usage"]["total_tokens"] == 10


# -- Ollama -----------------------------------------------------------------------------------------------------------


def _ollama_error(client: TestClient, path: str, body: dict) -> str:
    response = client.post(path, json=body)
    assert response.status_code == 400
    payload = response.json()
    assert list(payload) == ["error"]
    return payload["error"]


@pytest.mark.parametrize(
    ("path", "body", "message"),
    [
        ("/api/chat", {"model": "m", "messages": [{"role": "robot", "content": "x"}]}, "unknown message role 'robot'"),
        ("/api/generate", {"model": "m", "prompt": "x", "images": ["AAAA"]}, "images are not supported"),
        ("/api/generate", {"model": "m", "prompt": "x", "suffix": "y"}, "suffix, template and context"),
        ("/api/generate", {"model": "m", "prompt": "x", "context": [1, 2]}, "suffix, template and context"),
        ("/api/embed", {"model": "m", "input": ""}, "cannot embed an empty input"),
        ("/api/embed", {"model": "m", "input": ["a", ""]}, "cannot embed an empty input"),
        ("/api/embeddings", {"model": "m", "prompt": ""}, "cannot embed an empty input"),
        ("/api/chat", {"model": "m", "messages": [{"content": "no role"}]}, "role"),
    ],
)
def test_ollama_invalid_requests(client, path, body, message):
    assert message in _ollama_error(client, path, body)


def test_ollama_tool_result_without_a_pending_call():
    messages = ollama_api._messages([
        ollama_api.OllamaMessage(role="user", content="Hi"),
        ollama_api.OllamaMessage(role="tool", content="orphan", tool_name="weather"),
    ])  # fmt: skip
    assert (messages[1].role, messages[1].tool_call_id, messages[1].name) == ("tool", "", "weather")


def test_ollama_parameter_sizes():
    assert [ollama_api._parameter_size(n) for n in (999, 1000, 1_500_000, 2 * 10**9)] == ["999", "1K", "1.5M", "2B"]


def test_ollama_ps_reports_the_transformer_context(served, llama, client):  # noqa: F811
    (model,) = llama.ps().models
    assert (model.model, model.details.family) == ("example/tiny-chat", "llama")
    assert model.size > 0
    (entry,) = client.get("/api/ps").json()["models"]
    assert entry["context_length"] == served.model.config.context_length


def test_ollama_show_without_embeddings(client):
    model = SimpleNamespace(id="stub")
    app.dependency_overrides[default_engine] = lambda: SimpleNamespace(model=model, chat_template=None)
    try:
        shown = client.post("/api/show", json={"model": "stub"}).json()
    finally:
        app.dependency_overrides.pop(default_engine)
    assert shown["capabilities"] == ["completion", "tools"]
    assert shown["model_info"] == {"general.architecture": "bigram"}
    assert shown["template"] == ""
