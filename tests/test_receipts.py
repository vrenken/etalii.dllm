"""Generation receipts (Phase 15): every front end can return a receipt for a response, the receipt is the same for
streamed and non-streamed responses and across runs, and replaying it checks the output bit for bit."""

from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient
from test_mcp_server import in_memory

from etalii_dllm import __version__, engine, receipts
from etalii_dllm.chat import ChatMessage, ToolCall
from etalii_dllm.cli import main
from etalii_dllm.engine import ChatRequest, ResponseFormat, default_engine
from etalii_dllm.sampling import SamplingOptions
from etalii_dllm.server.app import app
from etalii_dllm.tools import Tool, ToolChoice


@pytest.fixture(autouse=True)
def placeholder_engine(monkeypatch):
    for name in (engine.MODEL_ENVIRONMENT_VARIABLE, engine.QUANTIZE_ENVIRONMENT_VARIABLE):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    default_engine.cache_clear()
    yield default_engine()
    default_engine.cache_clear()


@pytest.fixture
def client():
    return TestClient(app)


REQUEST = ChatRequest(
    [ChatMessage("system", "Be brief."), ChatMessage("user", "Say something.")],
    12,
    SamplingOptions(temperature=0.7, top_k=40, top_p=0.9, seed=5),
    stop=("zz",),
)


def test_engine_receipt_describes_the_request_and_the_output(placeholder_engine):
    result = placeholder_engine.chat_completion(REQUEST)
    receipt = result.receipt
    assert receipt["receipt"] == receipts.FORMAT and receipt["engine"] == __version__
    assert receipt["model"] == placeholder_engine.model.id
    assert receipt["system_fingerprint"] == placeholder_engine.system_fingerprint
    assert receipt["id"] == receipts.receipt_id(receipt) and receipt["id"].startswith("rcpt_")
    output = receipt["output"]
    assert output["tokens"] == result.fingerprint
    assert output["completion_tokens"] == result.completion_tokens
    assert output["prompt_tokens"] == result.prompt_tokens
    assert output["finish_reason"] == result.finish_reason
    assert receipt["request"] == receipts.request_record(REQUEST)
    assert receipts.request_from_record(receipt["request"]) == REQUEST
    # The same request gives the same receipt: nothing in it comes from a clock.
    assert placeholder_engine.chat_completion(REQUEST).receipt == receipt
    assert receipts.canonical_json(receipt) == receipts.canonical_json(json.loads(json.dumps(receipt)))


def test_request_records_round_trip():
    request = ChatRequest(
        [
            ChatMessage("user", "Weather?"),
            ChatMessage("assistant", "", (ToolCall("call_0", "weather", '{"city":"Paris"}'),)),
            ChatMessage("tool", "Sunny", tool_call_id="call_0", name="weather"),
        ],
        20,
        SamplingOptions(),
        tools=(Tool("weather", "Current weather", {"type": "object"}),),
        tool_choice=ToolChoice("named", "weather"),
        response_format=ResponseFormat("json_schema", {"type": "object"}),
        top_logprobs=2,
        call_id_prefix="toolu_",
        request_id="msg_1",
    )
    record = json.loads(json.dumps(receipts.request_record(request)))
    assert receipts.request_from_record(record) == request
    raw = ChatRequest([], 4, SamplingOptions(), prompt="Once upon")
    assert receipts.request_from_record(receipts.request_record(raw)) == raw


def test_a_receipt_verifies(placeholder_engine):
    receipt = placeholder_engine.chat_completion(REQUEST).receipt
    verification = receipts.verify(placeholder_engine, receipt)
    assert verification.ok and verification.reasons == () and verification.notes == ()
    assert verification.receipt == receipt
    assert verification.to_json()["ok"] is True


def test_edited_receipts_do_not_verify(placeholder_engine):
    receipt = placeholder_engine.chat_completion(REQUEST).receipt

    # An edited output hash: the id no longer matches and the replay differs.
    edited = json.loads(json.dumps(receipt))
    edited["output"]["content"] = "0" * 64
    verification = receipts.verify(placeholder_engine, edited)
    assert not verification.ok
    assert verification.reasons[0].startswith("the receipt was edited")
    assert any(r.startswith("the text differ") for r in verification.reasons)

    # A consistent forgery (id recomputed) of a different output still fails on the replay.
    forged = json.loads(json.dumps(receipt))
    forged["output"]["tokens"] = "f" * 64
    forged["output"]["completion_tokens"] += 1
    forged["id"] = receipts.receipt_id(forged)
    reasons = receipts.verify(placeholder_engine, forged).reasons
    expected = ["the generated tokens differ", "the number of generated tokens differ"]
    assert [r.split(":")[0] for r in reasons] == expected

    # Other weights.
    other = {**receipt, "system_fingerprint": "fp_other"}
    other["id"] = receipts.receipt_id(other)
    assert receipts.verify(placeholder_engine, other).reasons[0].startswith("different weights or settings")


def test_another_engine_version_is_only_a_note(placeholder_engine):
    receipt = {**placeholder_engine.chat_completion(REQUEST).receipt, "engine": "0.0.1"}
    receipt["id"] = receipts.receipt_id(receipt)
    verification = receipts.verify(placeholder_engine, receipt)
    assert verification.ok
    assert verification.notes == (f"made by engine version 0.0.1, replayed with {__version__}",)


def test_verify_refuses_other_formats(placeholder_engine):
    with pytest.raises(ValueError, match="not a dllm/1 receipt"):
        receipts.verify(placeholder_engine, {"receipt": "dllm/0"})


# -- front ends -----------------------------------------------------------------------------------------------------

OPENAI = {"messages": [{"role": "user", "content": "Say something."}], "temperature": 0.7, "seed": 5, "max_tokens": 12}


def _sse(response) -> list[dict]:
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]


def test_openai_receipts(client, placeholder_engine):
    plain = client.post("/v1/chat/completions", json=OPENAI).json()
    assert "receipt" not in plain
    body = client.post("/v1/chat/completions", json={**OPENAI, "receipt": True}).json()
    assert body["id"] == plain["id"]  # asking for a receipt does not change the request id
    receipt = body["receipt"]
    assert receipt["request"]["request_id"] == body["id"]
    assert receipts.verify(placeholder_engine, receipt).ok

    chunks = _sse(client.post("/v1/chat/completions", json={**OPENAI, "receipt": True, "stream": True}))
    assert [c["receipt"] for c in chunks if "receipt" in c] == [receipt]
    plain_chunks = _sse(client.post("/v1/chat/completions", json={**OPENAI, "stream": True}))
    assert not any("receipt" in c for c in plain_chunks)


def test_openai_receipt_records_tool_calls(client, placeholder_engine):
    tool = {"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}
    request = {**OPENAI, "max_tokens": 200, "tools": [tool], "tool_choice": "required", "receipt": True}
    body = client.post("/v1/chat/completions", json=request).json()
    calls = body["choices"][0]["message"]["tool_calls"]
    expected = [ToolCall(c["id"], c["function"]["name"], c["function"]["arguments"]) for c in calls]
    assert body["receipt"]["output"]["tool_calls"] == receipts.output_record("", "", expected, "", 0, 0)["tool_calls"]
    assert receipts.verify(placeholder_engine, body["receipt"]).ok


def test_anthropic_receipts(client, placeholder_engine):
    request = {"max_tokens": 12, "messages": [{"role": "user", "content": "Say something."}], "seed": 5}
    plain = client.post("/v1/messages", json=request).json()
    assert "receipt" not in plain
    body = client.post("/v1/messages", json={**request, "receipt": True}).json()
    assert body["id"] == plain["id"]
    assert receipts.verify(placeholder_engine, body["receipt"]).ok

    lines = client.post("/v1/messages", json={**request, "receipt": True, "stream": True}).text.splitlines()
    events = [json.loads(line[6:]) for line in lines if line.startswith("data: ")]
    assert [e["receipt"] for e in events if "receipt" in e] == [body["receipt"]]
    plain_lines = client.post("/v1/messages", json={**request, "stream": True}).text
    assert '"receipt"' not in plain_lines


def test_responses_receipts(client, placeholder_engine):
    request = {"input": "Say something.", "max_output_tokens": 12, "store": False}
    plain = client.post("/v1/responses", json=request).json()
    assert "receipt" not in plain
    body = client.post("/v1/responses", json={**request, "receipt": True}).json()
    assert body["id"] == plain["id"]
    assert receipts.verify(placeholder_engine, body["receipt"]).ok

    lines = client.post("/v1/responses", json={**request, "receipt": True, "stream": True}).text.splitlines()
    events = [json.loads(line[6:]) for line in lines if line.startswith("data: ")]
    assert events[-1]["response"]["receipt"] == body["receipt"]


def test_ollama_receipts(client, placeholder_engine):
    chat = {"messages": [{"role": "user", "content": "Say something."}], "options": {"num_predict": 12}}
    plain = client.post("/api/chat", json={**chat, "stream": False}).json()
    assert "receipt" not in plain
    body = client.post("/api/chat", json={**chat, "stream": False, "receipt": True}).json()
    assert receipts.verify(placeholder_engine, body["receipt"]).ok
    lines = [json.loads(line) for line in client.post("/api/chat", json={**chat, "receipt": True}).text.splitlines()]
    assert lines[-1]["receipt"] == body["receipt"]
    assert all("receipt" not in line for line in lines[:-1])

    generated = client.post(
        "/api/generate", json={"prompt": "Once", "raw": True, "stream": False, "receipt": True}
    ).json()
    assert generated["receipt"]["request"]["prompt"] == "Once"
    assert receipts.verify(placeholder_engine, generated["receipt"]).ok


def test_verify_endpoint(client, placeholder_engine):
    receipt = placeholder_engine.chat_completion(REQUEST).receipt
    answer = client.post("/v1/receipts/verify", json=receipt).json()
    assert answer["ok"] is True and answer["receipt"] == receipt
    edited = {**receipt, "model": "other"}
    assert client.post("/v1/receipts/verify", json=edited).json()["ok"] is False
    assert client.post("/v1/receipts/verify", json={"receipt": "dllm/0"}).status_code == 400
    assert client.post("/v1/receipts/verify", json={"receipt": "dllm/1"}).status_code == 400


def test_mcp_chat_receipt_and_verify_tool(placeholder_engine):
    async def steps(client):
        messages = [{"role": "user", "content": "Hi"}]
        plain = await client.call_tool("chat", {"messages": messages, "max_tokens": 8})
        with_receipt = await client.call_tool("chat", {"messages": messages, "max_tokens": 8, "receipt": True})
        answer = json.loads(with_receipt.content[0].text)
        verified = await client.call_tool("verify_receipt", {"receipt": answer["receipt"]})
        return plain, answer, verified

    plain, answer, verified = in_memory(steps)
    assert answer["content"] == plain.content[0].text
    assert json.loads(verified.content[0].text)["ok"] is True


# -- command line ---------------------------------------------------------------------------------------------------


def test_cli_chat_writes_a_receipt_and_replay_checks_it(tmp_path, capsys, monkeypatch, placeholder_engine):
    path = tmp_path / "receipt.json"
    assert main(["chat", "Say something.", "--max-tokens", "10", "--receipt", str(path)]) == 0
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert f"receipt: {path} ({receipt['id']})" in capsys.readouterr().err

    assert main(["replay", str(path)]) == 0
    assert "verified: the replay gave the same output, bit for bit" in capsys.readouterr().out
    assert main(["replay", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    monkeypatch.setattr("sys.stdin", io.StringIO(path.read_text(encoding="utf-8")))
    assert main(["replay", "-"]) == 0
    capsys.readouterr()

    receipt["engine"] = "0.0.1"
    receipt["output"]["content"] = "0" * 64
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert main(["replay", str(path)]) == 1
    out = capsys.readouterr().out
    assert "note:               made by engine version 0.0.1" in out
    assert "differs:            the receipt was edited" in out and "NOT verified" in out


@pytest.mark.parametrize("content", ["[]", "{", '{"receipt": "dllm/0"}', '{"receipt": "dllm/1"}'])
def test_cli_replay_reports_bad_receipts(tmp_path, capsys, content):
    path = tmp_path / "receipt.json"
    path.write_text(content, encoding="utf-8")
    assert main(["replay", str(path)]) == 2
    assert capsys.readouterr().err.startswith("dllm replay: ")
    assert main(["replay", str(tmp_path / "missing.json")]) == 2
