"""The OpenAI Files and Batches API over :mod:`etalii_dllm.batch_jobs`.

``POST /v1/files`` (multipart, as the OpenAI SDK uploads), ``GET /v1/files``, ``GET /v1/files/{id}``,
``GET /v1/files/{id}/content``, ``DELETE /v1/files/{id}``; ``POST /v1/batches``, ``GET /v1/batches``,
``GET /v1/batches/{id}`` and ``POST /v1/batches/{id}/cancel``.

File ids are hashes of the file's bytes and batch ids hashes of what the batch asks for, so the same upload gets the
same ids on every server; timestamps are 0. A batch runs in the background and its output file holds exactly the
bytes ``dllm batch`` writes, so its id is the same everywhere too. Only the status a poll sees depends on timing.
Files live in ``DLLM_BATCH_DIR`` when it is set (they then survive restarts), else in a temporary directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ValidationError

from etalii_dllm import batch_jobs
from etalii_dllm.engine import DllmEngine, default_engine

Engine = Annotated[DllmEngine, Depends(default_engine)]

BATCH_DIR_ENVIRONMENT_VARIABLE = "DLLM_BATCH_DIR"
WORKERS = 4
"""Requests a batch runs at once."""

router = APIRouter()


def handler(engine: DllmEngine) -> batch_jobs.Handler:
    """Answers batch requests with the HTTP endpoints themselves, so a batch body equals the endpoint's answer."""
    from etalii_dllm.server import app as server
    from etalii_dllm.server.contracts import ChatCompletionRequest, EmbeddingsRequest

    def answer(url: str, body: Any) -> tuple[int, dict[str, Any]]:
        try:
            if url == "/v1/chat/completions":
                result: Any = server.chat_completions(ChatCompletionRequest.model_validate(body), engine)
            else:
                result = server.embeddings(EmbeddingsRequest.model_validate(body), engine)
        except ValidationError as error:
            message = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in error.errors())
            return 400, {"error": {"message": message, "type": "invalid_request_error"}}
        if isinstance(result, JSONResponse):
            return result.status_code, json.loads(result.body)
        return 200, jsonable_encoder(result)

    return answer


# -- storage ------------------------------------------------------------------------------------------------------


def _error(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": "invalid_request_error"}})


class _Store:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._directory: Path | None = None
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self.batches: dict[str, dict[str, Any]] = {}
        self.cancels: dict[str, threading.Event] = {}

    @property
    def directory(self) -> Path:
        configured = os.environ.get(BATCH_DIR_ENVIRONMENT_VARIABLE)
        if configured:
            path = Path(configured)
        else:
            if self._temporary is None:
                self._temporary = tempfile.TemporaryDirectory(prefix="dllm-batches-")
            path = Path(self._temporary.name)
        (path / "files").mkdir(parents=True, exist_ok=True)
        (path / "batches").mkdir(parents=True, exist_ok=True)
        return path

    def save_file(self, data: bytes, filename: str, purpose: str) -> dict[str, Any]:
        identifier = "file-" + hashlib.sha256(data).hexdigest()[:24]
        record = {
            "id": identifier,
            "object": "file",
            "bytes": len(data),
            "created_at": 0,
            "filename": filename,
            "purpose": purpose,
            "status": "processed",
            "status_details": None,
        }
        with self._lock:
            root = self.directory / "files"
            (root / f"{identifier}.data").write_bytes(data)
            (root / f"{identifier}.json").write_text(json.dumps(record), encoding="utf-8")
        return record

    def file(self, identifier: str) -> dict[str, Any] | None:
        path = self.directory / "files" / f"{_safe(identifier)}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def content(self, identifier: str) -> bytes:
        return (self.directory / "files" / f"{_safe(identifier)}.data").read_bytes()

    def files(self) -> list[dict[str, Any]]:
        root = self.directory / "files"
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("*.json"))]

    def delete(self, identifier: str) -> bool:
        root = self.directory / "files"
        found = False
        for suffix in (".json", ".data"):
            path = root / f"{_safe(identifier)}{suffix}"
            if path.exists():
                path.unlink()
                found = True
        return found

    def save_batch(self, batch: dict[str, Any]) -> None:
        with self._lock:
            self.batches[batch["id"]] = batch
            path = self.directory / "batches" / f"{batch['id']}.json"
            path.write_text(json.dumps(batch), encoding="utf-8")

    def batch(self, identifier: str) -> dict[str, Any] | None:
        if identifier in self.batches:
            return self.batches[identifier]
        path = self.directory / "batches" / f"{_safe(identifier)}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _safe(identifier: str) -> str:
    """Ids name files, so only the characters ids are made of may pass."""
    return "".join(c for c in identifier if c.isalnum() or c in "-_")


store = _Store()


# -- files --------------------------------------------------------------------------------------------------------


@router.post("/v1/files", response_model=None)
async def upload_file(file: Annotated[UploadFile, File()], purpose: Annotated[str, Form()]) -> JSONResponse:
    data = await file.read()
    return JSONResponse(store.save_file(data, file.filename or "upload.jsonl", purpose))


@router.get("/v1/files", response_model=None)
def list_files() -> JSONResponse:
    return JSONResponse({"object": "list", "data": store.files(), "has_more": False})


@router.get("/v1/files/{file_id}", response_model=None)
def get_file(file_id: str) -> JSONResponse:
    record = store.file(file_id)
    return JSONResponse(record) if record else _error(f"no file {file_id!r}", 404)


@router.get("/v1/files/{file_id}/content", response_model=None)
def file_content(file_id: str) -> Response:
    if store.file(file_id) is None:
        return _error(f"no file {file_id!r}", 404)
    return Response(store.content(file_id), media_type="application/octet-stream")


@router.delete("/v1/files/{file_id}", response_model=None)
def delete_file(file_id: str) -> JSONResponse:
    if not store.delete(file_id):
        return _error(f"no file {file_id!r}", 404)
    return JSONResponse({"id": file_id, "object": "file", "deleted": True})


# -- batches ------------------------------------------------------------------------------------------------------


class BatchRequest(BaseModel):
    input_file_id: str
    endpoint: str
    completion_window: str = "24h"
    metadata: dict[str, str] | None = None


def _new_batch(identifier: str, request: BatchRequest, total: int) -> dict[str, Any]:
    return {
        "id": identifier,
        "object": "batch",
        "endpoint": request.endpoint,
        "errors": None,
        "input_file_id": request.input_file_id,
        "completion_window": request.completion_window,
        "status": "in_progress",
        "output_file_id": None,
        "error_file_id": None,
        "created_at": 0,
        "in_progress_at": 0,
        "expires_at": None,
        "finalizing_at": None,
        "completed_at": None,
        "failed_at": None,
        "expired_at": None,
        "cancelling_at": None,
        "cancelled_at": None,
        "request_counts": {"total": total, "completed": 0, "failed": 0},
        "metadata": request.metadata,
        "digest": None,
    }


def _run(engine: DllmEngine, batch: dict[str, Any], job: batch_jobs.Batch, input_data: bytes) -> None:
    output = store.directory / "batches" / f"{batch['id']}.output.jsonl"
    try:
        summary = job.run(output, workers=WORKERS)
    except Exception as error:  # a failed batch is reported on the batch object, not lost in a thread
        store.save_batch({**batch, "status": "failed", "failed_at": 0, "errors": {"object": "list", "data": [
            {"code": "batch_failed", "message": str(error), "param": None, "line": None}]}})  # fmt: skip
        return
    data = output.read_bytes()
    output_file = store.save_file(data, f"{batch['id']}_output.jsonl", "batch_output")
    counts = {"total": summary.total, "completed": summary.completed, "failed": summary.failed}
    final = {**batch, "output_file_id": output_file["id"], "request_counts": counts}
    if summary.cancelled:
        final.update(status="cancelled", cancelled_at=0)
    else:
        record = batch_jobs.digest(input_data, data, engine.system_fingerprint, summary)
        if engine.signer is not None:
            record = engine.signer.sign(record)
        final.update(status="completed", finalizing_at=0, completed_at=0, digest=record)
    store.save_batch(final)


@router.post("/v1/batches", response_model=None)
def create_batch(request: BatchRequest, engine: Engine) -> JSONResponse:
    if request.endpoint not in batch_jobs.ENDPOINTS:
        return _error(f"unsupported endpoint {request.endpoint!r} (use {' or '.join(batch_jobs.ENDPOINTS)})")
    if store.file(request.input_file_id) is None:
        return _error(f"no file {request.input_file_id!r}", 404)
    canonical = json.dumps(request.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(f"{engine.system_fingerprint}\n{canonical}".encode()).hexdigest()
    identifier = "batch_" + digest[:24]
    existing = store.batch(identifier)
    if existing is not None and existing["status"] in ("in_progress", "completed"):
        return JSONResponse(existing)  # the same batch: its answer would be the same bytes
    input_data = store.content(request.input_file_id)
    lines = batch_jobs.read_lines(input_data)
    cancel = threading.Event()
    store.cancels[identifier] = cancel
    job = batch_jobs.Batch(lines, engine.system_fingerprint, handler(engine), request.endpoint, cancel)
    batch = _new_batch(identifier, request, len(lines))
    store.save_batch(batch)
    threading.Thread(target=_run, args=(engine, batch, job, input_data), daemon=True).start()
    return JSONResponse(batch)


@router.get("/v1/batches", response_model=None)
def list_batches() -> JSONResponse:
    batches = [store.batches[k] for k in sorted(store.batches)]
    return JSONResponse({"object": "list", "data": batches, "has_more": False})


@router.get("/v1/batches/{batch_id}", response_model=None)
def get_batch(batch_id: str) -> JSONResponse:
    batch = store.batch(batch_id)
    return JSONResponse(batch) if batch else _error(f"no batch {batch_id!r}", 404)


@router.post("/v1/batches/{batch_id}/cancel", response_model=None)
def cancel_batch(batch_id: str) -> JSONResponse:
    batch = store.batch(batch_id)
    if batch is None:
        return _error(f"no batch {batch_id!r}", 404)
    if batch["status"] == "in_progress" and batch_id in store.cancels:
        store.cancels[batch_id].set()
        batch = {**batch, "status": "cancelling", "cancelling_at": 0}
        store.save_batch(batch)
    return JSONResponse(batch)
