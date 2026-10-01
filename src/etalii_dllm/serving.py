"""Deterministic serving at scale: answers that are a pure function of the request can be shared exactly.

A response depends only on the weights, the engine and the request (``docs/serving.md``). That makes three things
safe that are approximations elsewhere:

- :class:`ResponseCache` (``--response-cache DIR``): a finished generation's events stored under a content key; a
  repeated request is answered from storage with the very same bits, receipt included. Never stale: another model,
  setting or engine version gives another key.
- :class:`Inflight`: identical requests arriving while the first is still generating share its generation, each
  reading the same events. Any reader advances the shared generation, so a reader that goes away stalls no one.
- :class:`Auditor` (``--audit-every N``): a server re-runs every Nth response (chosen from the receipt id, not at
  random) after answering it, bypassing both, and reports any difference at ``GET /v1/audit``. ``dllm audit``
  compares whole fleets (:func:`audit_servers`).
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import tempfile
import threading
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from etalii_dllm import receipts
from etalii_dllm.chat import ToolCall
from etalii_dllm.generation import TokenLogprob, TokenLogprobs

if TYPE_CHECKING:
    from etalii_dllm.engine import ChatEvent, ChatRequest, DllmEngine

FORMAT = "dllm-response/1"
"""The format of a stored response; a reader ignores others."""

_SUFFIX = ".response.json"


def response_key(engine: DllmEngine, request: ChatRequest) -> str:
    """The content key of ``request`` on ``engine``: everything that can change the output (the engine version, the
    full weights fingerprint, the settings in the system fingerprint, a document index, the request) and nothing
    else. The receipt chain link (``previous_receipt``) is left out: it never changes the output."""
    from etalii_dllm import __version__

    body = {
        "format": FORMAT,
        "engine": __version__,
        "model": engine.model.id,
        "weights": engine.model.weights_fingerprint,
        "system_fingerprint": engine.system_fingerprint,
        "retriever": engine.retriever.fingerprint if engine.retriever is not None else None,
        "request": receipts.request_record(request),
    }
    return hashlib.sha256(receipts.canonical_json(body).encode()).hexdigest()


# -- events as JSON ------------------------------------------------------------------------------------------------


def _logprobs_json(item: TokenLogprobs) -> list[Any]:
    return [item.token, float(item.logprob).hex(), [[t.token, float(t.logprob).hex()] for t in item.top]]


def _logprobs(value: Sequence[Any]) -> TokenLogprobs:
    token, logprob, top = value
    return TokenLogprobs(
        int(token), float.fromhex(logprob), tuple(TokenLogprob(int(t), float.fromhex(p)) for t, p in top)
    )


def event_json(event: ChatEvent) -> dict[str, Any]:
    """An event as JSON, floats as exact hex strings."""
    from etalii_dllm.engine import ReasoningDelta, TextDelta, ToolCallEvent

    if isinstance(event, TextDelta):
        return {"text": event.text, "logprobs": [_logprobs_json(item) for item in event.logprobs]}
    if isinstance(event, ReasoningDelta):
        return {"reasoning": event.text, "logprobs": [_logprobs_json(item) for item in event.logprobs]}
    if isinstance(event, ToolCallEvent):
        call = event.call
        return {"tool_call": {"index": event.index, "id": call.id, "name": call.name, "arguments": call.arguments}}
    return {
        "finished": {
            "finish_reason": event.finish_reason,
            "stop_sequence": event.stop_sequence,
            "completion_tokens": event.completion_tokens,
            "fingerprint": event.fingerprint,
            "receipt": event.receipt,
            **({"reasoning_tokens": event.reasoning_tokens} if event.reasoning_tokens else {}),
        }
    }


def event_from_json(value: Mapping[str, Any]) -> ChatEvent:
    """The inverse of :func:`event_json`."""
    from etalii_dllm.engine import Finished, ReasoningDelta, TextDelta, ToolCallEvent

    if "text" in value:
        return TextDelta(value["text"], tuple(_logprobs(item) for item in value["logprobs"]))
    if "reasoning" in value:
        return ReasoningDelta(value["reasoning"], tuple(_logprobs(item) for item in value["logprobs"]))
    if "tool_call" in value:
        call = value["tool_call"]
        return ToolCallEvent(int(call["index"]), ToolCall(call["id"], call["name"], call["arguments"]))
    done = value["finished"]
    return Finished(
        done["finish_reason"],
        done["stop_sequence"],
        int(done["completion_tokens"]),
        done["fingerprint"],
        done["receipt"],
        int(done.get("reasoning_tokens", 0)),
    )


# -- the response cache --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Recorded:
    """A finished generation: its prompt size and every event, the last one :class:`~etalii_dllm.engine.Finished`."""

    prompt_tokens: int
    events: tuple[ChatEvent, ...]


class ResponseCache:
    """Finished responses in a directory, one JSON file per request key (``<key>.response.json``).

    A file holds the key, the prompt size, the events and a SHA-256 of them; it is written to a temporary name and
    renamed, so a crash never leaves a half-written response under a real name, and a file that does not match its
    checksum or its name is ignored. Many processes may share one directory: they write the same bytes."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    def path(self, key: str) -> Path:
        return self.directory / f"{key}{_SUFFIX}"

    def files(self) -> list[Path]:
        """Every stored response, in name order."""
        return sorted(self.directory.glob(f"*{_SUFFIX}"))

    @staticmethod
    def _checksum(key: str, prompt_tokens: int, events: list[dict[str, Any]]) -> str:
        body = {"key": key, "prompt_tokens": prompt_tokens, "events": events}
        return hashlib.sha256(receipts.canonical_json(body).encode()).hexdigest()

    def get(self, key: str) -> Recorded | None:
        """The stored response for ``key``, or ``None`` (counted as a miss) when there is none or it is damaged."""
        recorded = self._read(key)
        with self._lock:
            if recorded is None:
                self.misses += 1
            else:
                self.hits += 1
        return recorded

    def _read(self, key: str) -> Recorded | None:
        try:
            stored = json.loads(self.path(key).read_text(encoding="utf-8"))
            events = stored["events"]
            if stored.get("format") != FORMAT or stored["key"] != key:
                return None
            if stored["sha256"] != self._checksum(key, int(stored["prompt_tokens"]), events):
                return None
            return Recorded(int(stored["prompt_tokens"]), tuple(event_from_json(event) for event in events))
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            return None

    def put(self, key: str, recorded: Recorded) -> None:
        """Stores a finished response. Writers of one key write the same bytes, so a damaged file is simply
        replaced."""
        path = self.path(key)
        events = [event_json(event) for event in recorded.events]
        stored = {
            "format": FORMAT,
            "key": key,
            "prompt_tokens": recorded.prompt_tokens,
            "events": events,
            "sha256": self._checksum(key, recorded.prompt_tokens, events),
        }
        with tempfile.NamedTemporaryFile("w", dir=self.directory, suffix=".tmp", delete=False, encoding="utf-8") as f:
            f.write(receipts.canonical_json(stored))
        os.replace(f.name, path)

    def stats(self) -> dict[str, int]:
        files = self.files()
        return {"responses": len(files), "bytes": sum(path.stat().st_size for path in files)}

    def clear(self) -> int:
        """Removes every stored response; returns how many."""
        files = self.files()
        for path in files:
            path.unlink(missing_ok=True)
        return len(files)


# -- coalescing ----------------------------------------------------------------------------------------------------


class SharedGeneration:
    """One generation read by any number of readers. Every reader sees every event from the start; whichever reader
    is ahead advances the underlying generation, under a lock, so nobody depends on another reader keeping up."""

    def __init__(
        self,
        prompt_tokens: int,
        cached_tokens: int,
        events: Iterator[ChatEvent],
        on_finished: Callable[[Recorded], None] | None = None,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.cached_tokens = cached_tokens
        self._events = events
        self._buffer: list[ChatEvent] = []
        self._done = False
        self._error: BaseException | None = None
        self._lock = threading.Lock()
        self._on_finished = on_finished

    def _event(self, index: int) -> ChatEvent | None:
        while True:
            if index < len(self._buffer):
                return self._buffer[index]
            if self._error is not None:
                raise self._error
            if self._done:
                return None
            with self._lock:
                if index < len(self._buffer) or self._done or self._error is not None:
                    continue
                try:
                    self._buffer.append(next(self._events))
                except StopIteration:
                    self._done = True
                    if self._on_finished is not None:
                        self._on_finished(Recorded(self.prompt_tokens, tuple(self._buffer)))
                except BaseException as error:  # every reader sees the failure; nothing is stored
                    self._error = error
                    raise

    def reader(self) -> Iterator[ChatEvent]:
        index = 0
        while (event := self._event(index)) is not None:
            yield event
            index += 1


class Inflight:
    """The generations still being read, by request key. Entries are weak: a generation nobody reads any more is
    dropped, and the next identical request starts a new one."""

    def __init__(self) -> None:
        self._generations: weakref.WeakValueDictionary[str, SharedGeneration] = weakref.WeakValueDictionary()
        self._lock = threading.Lock()
        self.joined = 0

    def get_or_start(self, key: str, start: Callable[[], SharedGeneration]) -> SharedGeneration:
        with self._lock:
            shared = self._generations.get(key)
            if shared is not None:
                self.joined += 1
                return shared
            shared = start()
            self._generations[key] = shared
            return shared

    def __len__(self) -> int:
        return len(self._generations)


# -- auditing ------------------------------------------------------------------------------------------------------


@dataclass
class Auditor:
    """Re-runs every ``every``-th response on a background thread, bypassing the response cache and coalescing, and
    keeps what did not reproduce. Which responses are checked follows from their receipt ids, so two replicas given
    the same traffic check the same ones."""

    engine: DllmEngine
    every: int
    checked: int = 0
    passed: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)
    _queue: queue.Queue[Mapping[str, Any]] = field(default_factory=queue.Queue, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    MAX_FAILURES = 100

    def selects(self, receipt: Mapping[str, Any]) -> bool:
        return int(hashlib.sha256(str(receipt["id"]).encode()).hexdigest()[:16], 16) % self.every == 0

    def observe(self, receipt: Mapping[str, Any]) -> None:
        if not self.selects(receipt):
            return
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="dllm-audit", daemon=True)
                self._thread.start()
        self._queue.put(receipt)

    def _run(self) -> None:
        while True:
            receipt = self._queue.get()
            try:
                verification = receipts.verify(self.engine, receipt)
                reasons = list(verification.reasons)
            except Exception as error:  # a broken replay is a finding too
                reasons = [f"the replay failed: {error}"]
            with self._lock:
                self.checked += 1
                if reasons:
                    self.failures = [*self.failures, {"receipt": receipt["id"], "reasons": reasons}][
                        -self.MAX_FAILURES :
                    ]
                else:
                    self.passed += 1
            self._queue.task_done()

    def wait(self) -> None:
        """Blocks until every selected response so far is checked."""
        self._queue.join()

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "every": self.every,
                "checked": self.checked,
                "passed": self.passed,
                "failed": self.checked - self.passed,
                "failures": list(self.failures),
            }


# -- fleet audits --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FleetResult:
    """The outcome of :func:`audit_servers` for one request: every target's receipt output, and what differed."""

    request: Mapping[str, Any]
    outputs: Mapping[str, Mapping[str, Any]]
    differences: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.differences


_COMPARED = ("system_fingerprint", "prompt_tokens", "tokens", "content", "tool_calls", "finish_reason")


def compare_outputs(outputs: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    """What differs between targets' ``{"system_fingerprint", **receipt output}``, against the first target."""
    names = list(outputs)
    first = outputs[names[0]]
    differences = []
    for name in names[1:]:
        for key in _COMPARED:
            if outputs[name].get(key) != first.get(key):
                detail = ""
                if key == "content":
                    a, b = str(first.get(key, "")), str(outputs[name].get(key, ""))
                    at = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))
                    detail = f" from character {at}"
                differences.append(f"{name} differs from {names[0]} in {key}{detail}")
    return tuple(differences)


def request_body(line: str, max_tokens: int) -> dict[str, Any]:
    """A prompts-file line as a chat completions request: a JSON object is used as is, anything else is a user
    message. Every request asks for a receipt."""
    text = line.strip()
    body: dict[str, Any] = (
        json.loads(text) if text.startswith("{") else {"messages": [{"role": "user", "content": text}]}
    )
    body.setdefault("model", "dllm")
    body.setdefault("max_tokens", max_tokens)
    return {**body, "stream": False, "receipt": True}


def post_json(url: str, body: Mapping[str, Any], timeout: float = 600.0) -> dict[str, Any]:
    import urllib.request

    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def audit_servers(
    targets: Mapping[str, Callable[[Mapping[str, Any]], Mapping[str, Any]]],
    bodies: Sequence[Mapping[str, Any]],
) -> list[FleetResult]:
    """Sends every request body to every target (a callable returning the chat completions response) and compares
    the receipts. Targets are queried in the order given."""
    results = []
    for body in bodies:
        outputs = {}
        for name, send in targets.items():
            receipt = send(body)["receipt"]
            outputs[name] = {"system_fingerprint": receipt["system_fingerprint"], **receipt["output"]}
        results.append(FleetResult(body, outputs, compare_outputs(outputs)))
    return results
