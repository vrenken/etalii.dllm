"""Deterministic serving at scale (Phase 18): the exact response cache, coalesced identical requests, the self-audit
and ``dllm audit``."""

from __future__ import annotations

import dataclasses
import gc
import json
import threading

import pytest
from fastapi.testclient import TestClient

from etalii_dllm import engine as engine_module
from etalii_dllm import receipts, serving
from etalii_dllm.chat import ChatMessage
from etalii_dllm.cli import main
from etalii_dllm.engine import ChatRequest, DllmEngine, Finished, TextDelta, default_engine, use_model_file
from etalii_dllm.sampling import SamplingOptions
from etalii_dllm.server.app import app

REQUEST = ChatRequest([ChatMessage("user", "Hi")], 12, SamplingOptions(temperature=0.8, seed=7), top_logprobs=3)
OTHER = ChatRequest([ChatMessage("user", "Bye")], 12)


@pytest.fixture
def clean(monkeypatch):
    names = ("MODEL", "RESPONSE_CACHE", "AUDIT_EVERY", "SIGN_KEY")
    for name in names:
        monkeypatch.delenv(getattr(engine_module, f"{name}_ENVIRONMENT_VARIABLE"), raising=False)
    default_engine.cache_clear()
    yield monkeypatch
    for name in names:
        monkeypatch.delenv(getattr(engine_module, f"{name}_ENVIRONMENT_VARIABLE"), raising=False)
    default_engine.cache_clear()


def _events(stream) -> list:
    return list(stream)


def test_the_response_cache_answers_with_the_same_bits(tmp_path):
    cold = DllmEngine.create_default().chat_completion(REQUEST)
    engine = DllmEngine.create_default()
    engine.response_cache = serving.ResponseCache(tmp_path)
    first = engine.chat_completion(REQUEST)
    assert first == cold and first.receipt == cold.receipt
    assert (engine.response_cache.hits, engine.response_cache.misses) == (0, 1)
    second = engine.chat_completion(REQUEST)
    assert second.cached_tokens == second.prompt_tokens  # the only visible difference
    second = dataclasses.replace(second, cached_tokens=0)
    assert second == cold and second.receipt == cold.receipt and second.logprobs == cold.logprobs
    assert engine.response_cache.hits == 1

    # It survives a restart, and other requests or engines do not hit it.
    restarted = DllmEngine.create_default()
    restarted.response_cache = serving.ResponseCache(tmp_path)
    assert _events(restarted.chat_stream(REQUEST)) == _events(DllmEngine.create_default().chat_stream(REQUEST))
    assert restarted.response_cache.hits == 1
    restarted.chat_completion(OTHER)
    assert restarted.response_cache.stats()["responses"] == 2
    assert serving.response_key(engine, REQUEST) != serving.response_key(engine, OTHER)

    # Replays bypass the cache: they must run the model.
    assert receipts.verify(restarted, second.receipt).ok
    assert restarted.chat_completion(REQUEST, fresh=True) == cold
    assert restarted.response_cache.hits == 1

    # A damaged file is ignored, then replaced.
    path = engine.response_cache.path(serving.response_key(engine, REQUEST))
    stored = json.loads(path.read_text())
    path.write_text(json.dumps({**stored, "prompt_tokens": 99}))
    assert engine.response_cache.get(serving.response_key(engine, REQUEST)) is None
    assert engine.chat_completion(REQUEST) == cold
    assert engine.response_cache.get(serving.response_key(engine, REQUEST)) is not None
    path.write_text("{")
    assert engine.response_cache.get(serving.response_key(engine, REQUEST)) is None
    assert engine.response_cache.clear() == 2 and engine.response_cache.stats() == {"responses": 0, "bytes": 0}


def test_cached_receipts_get_their_own_chain_link_and_signature(tmp_path):
    pytest.importorskip("cryptography")
    from etalii_dllm import signing

    signing.generate_key(tmp_path / "k")
    engine = DllmEngine.create_default()
    engine.response_cache = serving.ResponseCache(tmp_path / "cache")
    engine.signer = signing.Signer.load(tmp_path / "k")
    plain = engine.chat_completion(REQUEST).receipt
    chained = engine.chat_completion(ChatRequest(**{**REQUEST.__dict__, "previous_receipt": "rcpt_x"})).receipt
    assert engine.response_cache.hits == 1
    assert "previous" not in plain and chained["previous"] == "rcpt_x"
    assert chained["output"] == plain["output"] and receipts.receipt_id(chained) == chained["id"]
    public = engine.signer.public_key
    assert signing.signature_problem(plain, [public]) is None and signing.signature_problem(chained, [public]) is None


def test_events_round_trip_through_json():
    events = _events(DllmEngine.create_default().chat_stream(REQUEST))
    assert any(isinstance(e, TextDelta) and e.logprobs for e in events) and isinstance(events[-1], Finished)
    assert [serving.event_from_json(json.loads(json.dumps(serving.event_json(e)))) for e in events] == events
    from etalii_dllm.chat import ToolCall
    from etalii_dllm.engine import ToolCallEvent

    call = ToolCallEvent(0, ToolCall("call_1", "f", '{"a": 1}'))
    assert serving.event_from_json(serving.event_json(call)) == call


def test_identical_requests_share_one_generation():
    engine = DllmEngine.create_default()
    solo = _events(DllmEngine.create_default().chat_stream(REQUEST))
    first = engine.chat_stream(REQUEST)
    second = engine.chat_stream(REQUEST)
    assert engine.inflight.joined == 1 and len(engine.inflight) == 1
    assert _events(second) == solo  # the follower runs ahead: it advances the shared generation
    assert _events(first) == solo
    del first, second
    gc.collect()
    assert len(engine.inflight) == 0  # nobody reads it any more: the next identical request starts afresh
    abandoned = engine.chat_stream(REQUEST)
    next(iter(abandoned))  # a reader that stops does not stall the others
    assert _events(engine.chat_stream(REQUEST)) == solo

    results: list[list] = []
    threads = [threading.Thread(target=lambda: results.append(_events(engine.chat_stream(OTHER)))) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == [_events(DllmEngine.create_default().chat_stream(OTHER))] * 6


def test_a_failing_shared_generation_fails_every_reader_and_stores_nothing():
    stored = []

    def broken():
        yield TextDelta("a")
        raise RuntimeError("boom")

    shared = serving.SharedGeneration(1, 0, broken(), stored.append)
    first, second = shared.reader(), shared.reader()
    assert next(first) == TextDelta("a")
    with pytest.raises(RuntimeError, match="boom"):
        next(first)
    assert next(second) == TextDelta("a")
    with pytest.raises(RuntimeError, match="boom"):
        next(second)
    assert stored == []


def test_the_auditor_re_runs_responses():
    engine = DllmEngine.create_default()
    engine.auditor = serving.Auditor(engine, 1)
    receipt = engine.chat_completion(REQUEST).receipt
    engine.auditor.wait()
    assert engine.auditor.report() == {"every": 1, "checked": 1, "passed": 1, "failed": 0, "failures": []}
    tampered = {**receipt, "output": {**receipt["output"], "content": "other"}}
    engine.auditor.observe(tampered)
    engine.auditor.observe({"id": "x"})  # not a receipt: the replay fails
    engine.auditor.wait()
    report = engine.auditor.report()
    assert report["failed"] == 2 and report["failures"][0]["receipt"] == receipt["id"]
    assert any("the replay failed" in r for r in report["failures"][1]["reasons"])

    every_other = serving.Auditor(engine, 2)
    ids = [f"rcpt_{n}" for n in range(40)]
    chosen = [i for i in ids if every_other.selects({"id": i})]
    assert 0 < len(chosen) < len(ids) and chosen == [i for i in ids if serving.Auditor(engine, 2).selects({"id": i})]


def test_server_reports_audit_and_cache(tmp_path, clean):
    use_model_file(None, response_cache=tmp_path, audit_every=1)
    client = TestClient(app)
    body = {"model": "m", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 8}
    first = client.post("/v1/chat/completions", json=body).json()
    second = client.post("/v1/chat/completions", json=body).json()
    assert first["choices"] == second["choices"] and first["id"] == second["id"]
    default_engine().auditor.wait()
    report = client.get("/v1/audit").json()
    assert report["response_cache"]["hits"] == 1 and report["response_cache"]["responses"] == 1
    assert report["audit"]["checked"] == 2 and report["audit"]["failed"] == 0
    assert report["coalesced"] == 0
    clean.setenv(engine_module.AUDIT_EVERY_ENVIRONMENT_VARIABLE, "often")
    with pytest.raises(ValueError, match="DLLM_AUDIT_EVERY"):
        engine_module.configured_audit_every()


def test_server_without_cache_or_audit(clean):
    report = TestClient(app).get("/v1/audit").json()
    assert report == {"audit": None, "response_cache": None, "coalesced": 0}


def test_cli_cache(tmp_path, clean, capsys):
    assert main(["--response-cache", str(tmp_path), "chat", "Hi", "--max-tokens", "6"]) == 0
    capsys.readouterr()
    assert main(["cache", "stats", str(tmp_path)]) == 0
    assert "responses: 1" in capsys.readouterr().out
    assert main(["cache", "clear", str(tmp_path)]) == 0
    assert "removed 1 responses" in capsys.readouterr().out
    assert main(["cache", "stats", str(tmp_path / "missing")]) == 2


def test_cli_audit(tmp_path, clean, capsys):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text('Hi\n\n{"messages": [{"role": "user", "content": "Bye"}], "max_tokens": 5}\n')
    client = TestClient(app)
    sent = []

    def post(url, body, timeout=600.0):
        sent.append((url, body))
        response = client.post("/v1/chat/completions", json=body).json()
        if url == "http://odd":
            receipt = response["receipt"]
            content = receipt["output"]["content"]
            tampered = content[:1] + "\0" + content[2:]
            response["receipt"] = {**receipt, "output": {**receipt["output"], "content": tampered}}
        return response

    clean.setattr(serving, "post_json", post)
    assert main(["audit", "--local", "--url", "http://a", "--prompts", str(prompts), "--max-tokens", "4"]) == 0
    out = capsys.readouterr().out
    assert "request 2: same" in out and "2 targets, 2 requests: the same bits" in out
    assert sent[0][1]["max_tokens"] == 4 and sent[1][1]["max_tokens"] == 5 and sent[0][1]["receipt"] is True

    assert main(["audit", "--url", "http://a", "--url", "http://odd", "--prompts", str(prompts), "--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert not result["ok"] and result["targets"] == ["http://a", "http://odd"]
    assert "http://odd differs from http://a in content from character 1" in result["requests"][0]["differences"]
    assert main(["audit", "--url", "http://a", "--url", "http://odd", "--prompts", str(prompts)]) == 1
    assert "DIFFERENT" in capsys.readouterr().out

    assert main(["audit", "--url", "http://a", "--prompts", str(prompts)]) == 2
    assert main(["audit", "--url", "http://a", "--url", "http://b", "--prompts", str(tmp_path / "none")]) == 2


def test_post_json_talks_http(clean):
    import http.server

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            data = json.dumps({"path": self.path, "body": body}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.handle_request)
    thread.start()
    try:
        answer = serving.post_json(f"http://127.0.0.1:{server.server_port}/", {"a": 1})
    finally:
        thread.join()
        server.server_close()
    assert answer == {"path": "/v1/chat/completions", "body": {"a": 1}}
