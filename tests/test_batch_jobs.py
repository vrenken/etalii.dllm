"""Phase 22 reproducible batch jobs: dllm batch (#171), exact resume (#172), the Files and Batches API (#173) and
batch digests (#174)."""

from __future__ import annotations

import io
import json
import threading

import openai
import pytest
from fastapi.testclient import TestClient
from golden_values import BATCH_OUTPUT_SHA256

from etalii_dllm import batch_jobs
from etalii_dllm.cli import main
from etalii_dllm.engine import default_engine
from etalii_dllm.server import batches_api
from etalii_dllm.server.app import app


def _requests() -> list[dict]:
    lines = []
    for i in range(6):
        body = {
            "messages": [{"role": "user", "content": f"Count to {i}"}],
            "max_tokens": 10,
            "temperature": 0.7,
            "seed": i,
        }
        lines.append({"custom_id": f"chat-{i}", "method": "POST", "url": "/v1/chat/completions", "body": body})
    lines.append({"custom_id": "embed", "method": "POST", "url": "/v1/embeddings", "body": {"input": "hello"}})
    return lines


INVALID = [
    {"custom_id": "bad-url", "method": "POST", "url": "/v1/nope", "body": {}},
    {"custom_id": "empty", "method": "POST", "url": "/v1/chat/completions", "body": {"messages": []}},
    {"custom_id": "schema", "method": "POST", "url": "/v1/chat/completions", "body": {"messages": 3}},
    {"custom_id": "chat-0", "method": "POST", "url": "/v1/embeddings", "body": {"input": "x"}},
    {"custom_id": "get", "method": "GET", "url": "/v1/embeddings", "body": {"input": "x"}},
    {"custom_id": "", "method": "POST", "url": "/v1/embeddings", "body": {"input": "x"}},
    {"custom_id": "nobody", "method": "POST", "url": "/v1/embeddings", "body": []},
    {"custom_id": "stream", "method": "POST", "url": "/v1/chat/completions", "body": {"messages": [], "stream": True}},
]


def _input(tmp_path, extra: list | None = None, raw: str = "") -> tuple:
    text = "\n".join(json.dumps(r) for r in [*_requests(), *(extra or [])]) + "\n" + raw
    path = tmp_path / "in.jsonl"
    path.write_bytes(text.encode())
    return path, text.encode()


def _job(data: bytes, endpoint: str | None = None) -> batch_jobs.Batch:
    engine = default_engine()
    lines = batch_jobs.read_lines(data)
    return batch_jobs.Batch(lines, engine.system_fingerprint, batches_api.handler(engine), endpoint)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(batches_api.BATCH_DIR_ENVIRONMENT_VARIABLE, str(tmp_path / "store"))
    batches_api.store.batches.clear()
    return TestClient(app)


# -- dllm batch (#171) --------------------------------------------------------------------------------------------


def test_output_is_byte_identical_at_any_concurrency(tmp_path):
    _, data = _input(tmp_path)
    outputs = []
    for workers in (1, 3, 8):
        path = tmp_path / f"out-{workers}.jsonl"
        summary = _job(data).run(path, workers=workers)
        assert summary == batch_jobs.Summary(7, 7, 0, 0)
        outputs.append(path.read_bytes())
    assert outputs[0] == outputs[1] == outputs[2]
    import hashlib

    assert hashlib.sha256(outputs[0]).hexdigest() == BATCH_OUTPUT_SHA256


def test_results_are_the_endpoint_answers_in_input_order(tmp_path):
    _, data = _input(tmp_path)
    path = tmp_path / "out.jsonl"
    _job(data).run(path, workers=4)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["custom_id"] for r in records] == [r["custom_id"] for r in _requests()]
    http = TestClient(app)
    for request, record in zip(_requests(), records, strict=True):
        assert record["response"]["status_code"] == 200 and record["error"] is None
        assert record["response"]["body"] == http.post(request["url"], json=request["body"]).json()
        assert record["id"].startswith("batch_req_") and record["response"]["request_id"].startswith("req_")


def test_invalid_lines_are_reported_without_stopping(tmp_path):
    _, data = _input(tmp_path, INVALID, raw="not json\n[1]\n")
    path = tmp_path / "out.jsonl"
    summary = _job(data).run(path, workers=2)
    assert summary.total == 17 and summary.completed == 7 and summary.failed == 10
    records = {r["custom_id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
    assert "unsupported url" in records["bad-url"]["error"]["message"]
    assert records["empty"]["response"]["status_code"] == 400
    assert "messages" in records["schema"]["response"]["body"]["error"]["message"]
    assert "duplicate custom_id" in records["chat-0"]["error"]["message"]
    assert "only POST" in records["get"]["error"]["message"]
    assert "non-empty string" in records[""]["error"]["message"]
    assert "JSON object" in records["nobody"]["error"]["message"]
    assert "streaming" in records["stream"]["error"]["message"]
    lines = path.read_text(encoding="utf-8").splitlines()
    assert "not valid JSON" in lines[-2] and "a request line must be a JSON object" in lines[-1]


def test_an_endpoint_restricts_the_lines(tmp_path):
    _, data = _input(tmp_path)
    path = tmp_path / "out.jsonl"
    summary = _job(data, endpoint="/v1/embeddings").run(path)
    assert summary.completed == 1 and summary.failed == 6
    assert "differs from the batch endpoint" in path.read_text(encoding="utf-8").splitlines()[0]


def test_workers_are_bounded(tmp_path):
    with pytest.raises(batch_jobs.BatchError, match="workers"):
        _job(b"").run(tmp_path / "out.jsonl", workers=0)


def test_cli(tmp_path, capsys):
    path, _ = _input(tmp_path)
    out = tmp_path / "out.jsonl"
    assert main(["batch", str(path), "-o", str(out), "--workers", "3"]) == 0
    text = capsys.readouterr().out
    assert "7 requests: 7 completed, 0 failed" in text and BATCH_OUTPUT_SHA256 in text
    record = json.loads(batch_jobs.digest_path(out).read_text(encoding="utf-8"))
    assert record["output_sha256"] == BATCH_OUTPUT_SHA256 and record["requests"] == 7
    assert main(["batch", str(tmp_path / "missing.jsonl"), "-o", str(out)]) == 2


# -- resume (#172) ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("cut", [0, 1, 250, 900, -1])
def test_an_interrupted_batch_resumes_to_the_same_bytes(tmp_path, cut):
    _, data = _input(tmp_path)
    full = tmp_path / "full.jsonl"
    _job(data).run(full)
    expected = full.read_bytes()
    partial = tmp_path / "partial.jsonl"
    partial.write_bytes(expected[: cut if cut >= 0 else len(expected)])
    summary = _job(data).run(partial, workers=3)
    assert partial.read_bytes() == expected
    assert summary.resumed == expected[: max(cut, 0) if cut >= 0 else len(expected)].count(b"\n")
    assert summary.completed == 7


def test_a_cancelled_batch_resumes(tmp_path):
    _, data = _input(tmp_path)
    job = _job(data)
    job.cancel.set()
    path = tmp_path / "out.jsonl"
    assert job.run(path).cancelled
    assert path.read_bytes() == b""
    _job(data).run(path)
    full = tmp_path / "full.jsonl"
    _job(data).run(full)
    assert path.read_bytes() == full.read_bytes()


def test_output_of_another_batch_is_not_overwritten(tmp_path, capsys):
    path, data = _input(tmp_path)
    out = tmp_path / "out.jsonl"
    _job(data).run(out)
    other = batch_jobs.Batch(batch_jobs.read_lines(data), "fp_other", batches_api.handler(default_engine()))
    with pytest.raises(batch_jobs.BatchError, match="another input or model"):
        other.run(out)
    shorter = batch_jobs.read_lines(data)[:2]
    with pytest.raises(batch_jobs.BatchError, match="another input or model"):
        batch_jobs.Batch(shorter, default_engine().system_fingerprint, batches_api.handler(default_engine())).run(out)
    (tmp_path / "junk.jsonl").write_text('{"id": "x"}\n', encoding="utf-8")
    assert main(["batch", str(path), "-o", str(tmp_path / "junk.jsonl")]) == 2
    assert "another input or model" in capsys.readouterr().err


# -- digests and verification (#174) ------------------------------------------------------------------------------


def test_verify_finds_a_changed_line(tmp_path, capsys):
    path, _ = _input(tmp_path)
    out = tmp_path / "out.jsonl"
    assert main(["batch", str(path), "-o", str(out)]) == 0
    assert main(["batch", str(path), "-o", str(out), "--verify", "--sample", "3"]) == 0
    assert "re-ran 3 of 7 requests: all equal" in capsys.readouterr().out
    lines = out.read_text(encoding="utf-8").splitlines()
    lines[3] = lines[3].replace('"index":0', '"index":9')
    out.write_bytes(("\n".join(lines) + "\n").encode())
    assert main(["batch", str(path), "-o", str(out), "--verify"]) == 1
    text = capsys.readouterr().out
    assert "lines [4] differ" in text and "output_sha256 does not match" in text
    assert main(["batch", str(path), "-o", str(tmp_path / "none.jsonl"), "--verify"]) == 2


def test_verify_reports_missing_lines(tmp_path):
    _, data = _input(tmp_path)
    out = tmp_path / "out.jsonl"
    _job(data).run(out)
    out.write_bytes(b"\n".join(out.read_bytes().split(b"\n")[:3]) + b"\n")
    check = batch_jobs.verify(_job(data), data, out)
    assert not check.ok and check.differing == [3, 4, 5, 6] and "3 lines for 7 requests" in check.problems[0]


def test_sample_indices():
    assert batch_jobs.sample_indices(5, None) == [0, 1, 2, 3, 4]
    assert batch_jobs.sample_indices(10, 3) == [0, 3, 6]
    assert batch_jobs.sample_indices(4, 9) == [0, 1, 2, 3]
    assert batch_jobs.sample_indices(4, 0) == []


def test_signed_digests(tmp_path, monkeypatch, capsys):
    pytest.importorskip("cryptography")
    from etalii_dllm import engine as engine_module
    from etalii_dllm import signing

    monkeypatch.delenv(engine_module.SIGN_KEY_ENVIRONMENT_VARIABLE, raising=False)
    key = tmp_path / "key.pem"
    public = signing.generate_key(key)
    path, _ = _input(tmp_path)
    out = tmp_path / "out.jsonl"
    default_engine.cache_clear()
    try:
        assert main(["--sign-key", str(key), "batch", str(path), "-o", str(out)]) == 0
    finally:
        monkeypatch.delenv(engine_module.SIGN_KEY_ENVIRONMENT_VARIABLE, raising=False)
        default_engine.cache_clear()
    record = json.loads(batch_jobs.digest_path(out).read_text(encoding="utf-8"))
    assert signing.signature_problem(record, [public]) is None


# -- the Files and Batches API (#173) -----------------------------------------------------------------------------


def _sdk(client) -> openai.OpenAI:
    return openai.OpenAI(base_url="http://testserver/v1", api_key="unused", http_client=client)


def _wait(client, identifier: str) -> dict:
    for _ in range(2000):
        batch = client.get(f"/v1/batches/{identifier}").json()
        if batch["status"] in ("completed", "failed", "cancelled"):
            return batch
        threading.Event().wait(0.01)
    raise AssertionError("the batch did not finish")


def test_the_openai_sdk_batch_workflow(client, tmp_path):
    _, data = _input(tmp_path)
    sdk = _sdk(client)
    uploaded = sdk.files.create(file=("in.jsonl", io.BytesIO(data)), purpose="batch")
    assert uploaded.id.startswith("file-") and uploaded.bytes == len(data) and uploaded.created_at == 0
    batch = sdk.batches.create(input_file_id=uploaded.id, endpoint="/v1/chat/completions", completion_window="24h")
    done = _wait(client, batch.id)
    assert done["status"] == "completed" and done["request_counts"] == {"total": 7, "completed": 6, "failed": 1}
    output = sdk.files.content(done["output_file_id"]).read()
    expected = tmp_path / "expected.jsonl"
    _job(data, endpoint="/v1/chat/completions").run(expected)
    assert output == expected.read_bytes()
    import hashlib

    assert done["digest"]["output_sha256"] == hashlib.sha256(output).hexdigest()
    assert sdk.batches.retrieve(batch.id).status == "completed"
    assert [b.id for b in sdk.batches.list()] == [batch.id]
    again = sdk.batches.create(input_file_id=uploaded.id, endpoint="/v1/chat/completions", completion_window="24h")
    assert again.id == batch.id
    assert {f.id for f in sdk.files.list()} == {uploaded.id, done["output_file_id"]}
    assert sdk.files.retrieve(uploaded.id).filename == "in.jsonl"
    assert sdk.files.delete(uploaded.id).deleted
    assert client.get(f"/v1/files/{uploaded.id}").status_code == 404


def test_ids_are_derived_from_content(client, tmp_path):
    _, data = _input(tmp_path)
    first = client.post("/v1/files", files={"file": ("a.jsonl", data)}, data={"purpose": "batch"}).json()
    second = client.post("/v1/files", files={"file": ("b.jsonl", data)}, data={"purpose": "batch"}).json()
    assert first["id"] == second["id"]
    import hashlib

    assert first["id"] == "file-" + hashlib.sha256(data).hexdigest()[:24]


def test_batch_errors(client, tmp_path):
    _, data = _input(tmp_path)
    file_id = client.post("/v1/files", files={"file": ("a.jsonl", data)}, data={"purpose": "batch"}).json()["id"]
    bad = client.post("/v1/batches", json={"input_file_id": file_id, "endpoint": "/v1/nope"})
    assert bad.status_code == 400
    missing = client.post("/v1/batches", json={"input_file_id": "file-missing", "endpoint": "/v1/embeddings"})
    assert missing.status_code == 404
    assert client.get("/v1/batches/batch_missing").status_code == 404
    assert client.post("/v1/batches/batch_missing/cancel").status_code == 404
    assert client.get("/v1/files/file-missing/content").status_code == 404
    assert client.delete("/v1/files/file-missing").status_code == 404


def test_cancelling_a_batch(client, tmp_path, monkeypatch):
    _, data = _input(tmp_path)
    file_id = client.post("/v1/files", files={"file": ("a.jsonl", data)}, data={"purpose": "batch"}).json()["id"]
    gate = threading.Event()
    original = batch_jobs.Batch.result

    def slow(self, index):
        gate.wait(5)
        return original(self, index)

    monkeypatch.setattr(batch_jobs.Batch, "result", slow)
    batch = client.post("/v1/batches", json={"input_file_id": file_id, "endpoint": "/v1/embeddings"}).json()
    cancelling = client.post(f"/v1/batches/{batch['id']}/cancel").json()
    assert cancelling["status"] == "cancelling"
    gate.set()
    done = _wait(client, batch["id"])
    assert done["status"] == "cancelled" and done["digest"] is None
    monkeypatch.setattr(batch_jobs.Batch, "result", original)
    resumed = client.post("/v1/batches", json={"input_file_id": file_id, "endpoint": "/v1/embeddings"}).json()
    assert resumed["id"] == batch["id"]
    assert _wait(client, batch["id"])["status"] == "completed"
    assert client.post(f"/v1/batches/{batch['id']}/cancel").json()["status"] == "completed"


def test_a_failing_batch_is_reported(client, tmp_path, monkeypatch):
    _, data = _input(tmp_path)
    file_id = client.post("/v1/files", files={"file": ("a.jsonl", data)}, data={"purpose": "batch"}).json()["id"]

    def broken(self, output, workers=1):
        raise RuntimeError("disk full")

    monkeypatch.setattr(batch_jobs.Batch, "run", broken)
    batch = client.post("/v1/batches", json={"input_file_id": file_id, "endpoint": "/v1/embeddings"}).json()
    done = _wait(client, batch["id"])
    assert done["status"] == "failed" and done["errors"]["data"][0]["message"] == "disk full"


def test_the_store_survives_a_restart(client, tmp_path):
    _, data = _input(tmp_path)
    file_id = client.post("/v1/files", files={"file": ("a.jsonl", data)}, data={"purpose": "batch"}).json()["id"]
    batch = client.post("/v1/batches", json={"input_file_id": file_id, "endpoint": "/v1/embeddings"}).json()
    _wait(client, batch["id"])
    batches_api.store.batches.clear()  # what a restart forgets
    assert client.get(f"/v1/batches/{batch['id']}").json()["status"] == "completed"


def test_a_temporary_store_without_a_directory(monkeypatch):
    monkeypatch.delenv(batches_api.BATCH_DIR_ENVIRONMENT_VARIABLE, raising=False)
    store = batches_api._Store()
    record = store.save_file(b"x", "x.jsonl", "batch")
    assert store.content(record["id"]) == b"x" and store.file("file-none") is None
    assert batches_api._safe("../file-1") == "file-1"
