"""Reproducible batch jobs over OpenAI batch files (``dllm batch``, ``/v1/batches``).

An input file has one JSON request per line: ``{"custom_id", "method": "POST", "url", "body"}`` with ``url``
``/v1/chat/completions`` or ``/v1/embeddings``. The output file has one result per input line, in input order:
``{"id", "custom_id", "response": {"status_code", "request_id", "body"}, "error"}``, where ``body`` is exactly what
the HTTP endpoint answers. Requests run concurrently (sharing batched decoding steps, which never changes a bit), but
lines are written in input order and every id is derived from content, so the output file is byte-identical at any
worker count, on every machine and in every run.

Resuming: each result's ``id`` is a hash of the system fingerprint, the line number and the request line, so the
finished prefix of an interrupted output can be checked against the input and kept; only the rest is computed, and
the file ends byte-identical to an uninterrupted run. A torn last line is dropped and recomputed.

Every finished batch gets a digest (:func:`digest`) over the input, the output and the system fingerprint, written
to ``<output>.digest.json`` and optionally signed; :func:`verify` re-runs lines and compares them bit for bit.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FORMAT = "dllm-batch/1"
ENDPOINTS = ("/v1/chat/completions", "/v1/embeddings")
MAX_WORKERS = 64

Handler = Callable[[str, Mapping[str, Any]], tuple[int, dict[str, Any]]]
"""Answers one request: ``(url, body) -> (status_code, response body)``."""


class BatchError(ValueError):
    """The batch cannot run (unreadable input, an output written for another input or model, ...)."""


def _hash(*parts: str | bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part if isinstance(part, bytes) else part.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def read_lines(data: bytes) -> list[bytes]:
    """The request lines of an input file (``\\r\\n`` or ``\\n`` endings; blank lines are left out)."""
    return [line.rstrip(b"\r") for line in data.split(b"\n") if line.strip()]


def line_id(fingerprint: str, index: int, line: bytes) -> str:
    return "batch_req_" + _hash(FORMAT, fingerprint, str(index), line)[:24]


@dataclass(frozen=True)
class Summary:
    total: int
    completed: int
    """Lines answered with status 200."""
    failed: int
    resumed: int
    """Lines kept from an earlier, interrupted run."""
    cancelled: bool = False


@dataclass
class Batch:
    """One batch run: ``lines`` of the input, answered by ``handler`` on a model with ``fingerprint``."""

    lines: Sequence[bytes]
    fingerprint: str
    handler: Handler
    endpoint: str | None = None
    """When set, every line must use this url (the Batches API's rule)."""
    cancel: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        self._custom_ids: dict[str, int] = {}
        for index, line in enumerate(self.lines):
            try:
                custom_id = json.loads(line).get("custom_id")
            except (ValueError, AttributeError):
                continue
            if isinstance(custom_id, str):
                self._custom_ids.setdefault(custom_id, index)

    def ids(self) -> list[str]:
        return [line_id(self.fingerprint, i, line) for i, line in enumerate(self.lines)]

    def result(self, index: int) -> str:
        """The output line (without the newline) for input line ``index``."""
        line = self.lines[index]
        identifier = line_id(self.fingerprint, index, line)
        custom_id: Any = None
        try:
            try:
                request = json.loads(line)
            except ValueError:  # the parser's own message varies between Python versions
                raise BatchError(f"line {index + 1}: not valid JSON") from None
            if not isinstance(request, dict):
                raise BatchError(f"line {index + 1}: a request line must be a JSON object")
            custom_id = request.get("custom_id")
            self._check(index, request)
        except BatchError as error:
            record = {"id": identifier, "custom_id": custom_id, "response": None}
            record["error"] = {"code": "invalid_request", "message": str(error)}
            return _dumps(record)
        status, body = self.handler(request["url"], request["body"])
        response = {"status_code": status, "request_id": "req_" + identifier[len("batch_req_") :], "body": body}
        return _dumps({"id": identifier, "custom_id": custom_id, "response": response, "error": None})

    def _check(self, index: int, request: Mapping[str, Any]) -> None:
        custom_id = request.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id:
            raise BatchError(f"line {index + 1}: 'custom_id' must be a non-empty string")
        if self._custom_ids.get(custom_id) != index:
            raise BatchError(f"line {index + 1}: duplicate custom_id {custom_id!r}")
        if request.get("method", "POST") != "POST":
            raise BatchError(f"line {index + 1}: only POST requests are supported")
        url = request.get("url")
        if url not in ENDPOINTS:
            raise BatchError(f"line {index + 1}: unsupported url {url!r} (use {' or '.join(ENDPOINTS)})")
        if self.endpoint is not None and url != self.endpoint:
            raise BatchError(f"line {index + 1}: url {url!r} differs from the batch endpoint {self.endpoint!r}")
        body = request.get("body")
        if not isinstance(body, dict):
            raise BatchError(f"line {index + 1}: 'body' must be a JSON object")
        if body.get("stream"):
            raise BatchError(f"line {index + 1}: streaming is not supported in a batch")

    def run(self, output: Path, workers: int = 1) -> Summary:
        """Writes the results to ``output``, resuming from what an earlier run left there."""
        if not 1 <= workers <= MAX_WORKERS:
            raise BatchError(f"workers must be between 1 and {MAX_WORKERS}")
        ids = self.ids()
        kept, statuses = _resume(output, ids)
        completed, failed = statuses
        cancelled = False
        with open(output, "ab") as out, ThreadPoolExecutor(max_workers=workers) as pool:
            pending: dict[int, Future[str]] = {}
            next_index = kept
            for index in range(kept, len(self.lines)):
                while len(pending) < 2 * workers and next_index < len(self.lines):
                    pending[next_index] = pool.submit(self.result, next_index)
                    next_index += 1
                if self.cancel.is_set():
                    cancelled = True
                    break
                text = pending.pop(index).result()
                out.write(text.encode() + b"\n")
                out.flush()
                ok = _status(text) == 200
                completed += ok
                failed += not ok
            for future in pending.values():
                future.cancel()
        return Summary(len(self.lines), completed, failed, kept, cancelled)


def _status(text: str) -> int | None:
    response = json.loads(text).get("response")
    return response.get("status_code") if response else None


def _resume(output: Path, ids: Sequence[str]) -> tuple[int, tuple[int, int]]:
    """How many leading lines of ``output`` are finished results for ``ids`` (the file is cut after them), and how
    many of those completed and failed. Raises :class:`BatchError` when the output belongs to another batch."""
    if not output.exists():
        output.write_bytes(b"")
        return 0, (0, 0)
    data = output.read_bytes()
    kept = 0
    end = 0
    completed = failed = 0
    for raw in data.split(b"\n")[:-1]:  # the part after the last newline is a torn line
        try:
            record = json.loads(raw)
        except ValueError:
            break
        if kept >= len(ids) or not isinstance(record, dict) or record.get("id") != ids[kept]:
            raise BatchError(
                f"{output} holds results of another input or model (line {kept + 1}); remove it or choose a new path"
            )
        ok = (record.get("response") or {}).get("status_code") == 200
        completed += ok
        failed += not ok
        kept += 1
        end += len(raw) + 1
    if end != len(data):
        with open(output, "r+b") as file:
            file.truncate(end)
    return kept, (completed, failed)


# -- digests and verification -------------------------------------------------------------------------------------


def digest(input_data: bytes, output_data: bytes, fingerprint: str, summary: Summary) -> dict[str, Any]:
    """The batch digest: what was asked, what was answered, by which weights."""
    from etalii_dllm import __version__

    body: dict[str, Any] = {
        "format": FORMAT,
        "engine": __version__,
        "system_fingerprint": fingerprint,
        "input_sha256": hashlib.sha256(input_data).hexdigest(),
        "output_sha256": hashlib.sha256(output_data).hexdigest(),
        "requests": summary.total,
        "completed": summary.completed,
        "failed": summary.failed,
    }
    body["digest"] = "bdig_" + _hash(_dumps(body))[:32]
    return body


def digest_path(output: Path) -> Path:
    return output.with_name(output.name + ".digest.json")


@dataclass(frozen=True)
class Verification:
    checked: list[int]
    """Input line numbers (0-based) that were re-run."""
    differing: list[int]
    problems: list[str]

    @property
    def ok(self) -> bool:
        return not self.differing and not self.problems


def sample_indices(total: int, sample: int | None) -> list[int]:
    """Evenly spread line numbers to re-run: all of them when ``sample`` is ``None`` or at least ``total``."""
    if sample is None or sample >= total:
        return list(range(total))
    if sample <= 0:
        return []
    return [i * total // sample for i in range(sample)]


def verify(batch: Batch, input_data: bytes, output: Path, sample: int | None = None) -> Verification:
    """Re-runs ``sample`` lines (all when ``None``) and compares them with ``output`` byte for byte; also checks the
    digest file next to it when there is one."""
    problems: list[str] = []
    data = output.read_bytes()
    written = data.split(b"\n")
    if written and written[-1] == b"":
        written.pop()
    if len(written) != len(batch.lines):
        problems.append(f"the output has {len(written)} lines for {len(batch.lines)} requests")
    checked = sample_indices(len(batch.lines), sample)
    differing = [i for i in checked if i >= len(written) or written[i] != batch.result(i).encode()]
    record_path = digest_path(output)
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        expected = {k: v for k, v in record.items() if k != "signature"}
        lines = [json.loads(line) for line in written]
        ok = sum(1 for r in lines if (r.get("response") or {}).get("status_code") == 200)
        summary = Summary(len(batch.lines), ok, len(lines) - ok, 0)
        actual = digest(input_data, data, batch.fingerprint, summary)
        for key in ("input_sha256", "output_sha256", "system_fingerprint", "digest"):
            if expected.get(key) != actual[key]:
                problems.append(f"the digest's {key} does not match")
    return Verification(checked, differing, problems)
