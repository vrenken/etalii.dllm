"""Agent transcripts: an MCP tool loop recorded so that anyone can replay it offline, bit for bit.

The model's side of an agent run (:func:`etalii_dllm.mcp_host.chat`) is as deterministic as any chat; only the tool
results come from outside. A transcript records the starting request, the tools offered, and for every round the
round's receipt, the assistant text and tool calls in full and the tool results. :func:`replay` runs the same loop
again with :class:`RecordedTools` answering every call from the transcript instead of live servers, and compares each
round's receipt with the recorded one: the first round that differs is where the run diverges (``dllm replay``).

The engine also answers servers' sampling and elicitation requests during a run (:attr:`etalii_dllm.mcp_host.McpHost.
answers`). The transcript records each one with the engine request it became, and replay runs those requests again,
so the engine's side of a server conversation is checked as well. A run without any keeps the format below exactly.

A transcript is plain JSON with nothing from a clock or a random source, so the same run gives the same bytes::

    {
      "transcript": "dllm-agent/1",
      "id": "trn_...",                   # hash of everything below
      "engine": "0.2.0", "model": "...", "system_fingerprint": "fp_...",
      "request": {...},                  # the starting engine request (receipts.request_record)
      "tools": [{"name", "description", "parameters", "server"}],
      "max_rounds": 8,
      "rounds": [{"receipt": {...}, "content": "...", "tool_calls": [...],
                  "results": [{"id", "name", "server", "content", "is_error"}]}],
      "server_requests": [               # only when a server asked; in the order they asked
        {"kind": "sampling", "server", "request": {...}, "content": "...", "stop_reason", "fingerprint"},
        {"kind": "elicitation", "server", "message", "action", "content": {...} | null,
         "request": {...} | null, "fingerprint": "..." | null}]
    }
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from etalii_dllm import receipts
from etalii_dllm.chat import ToolCall
from etalii_dllm.engine import Finished, TextDelta, ToolCallEvent
from etalii_dllm.tools import Tool

if TYPE_CHECKING:
    from etalii_dllm.engine import ChatRequest, DllmEngine
    from etalii_dllm.mcp_host import Elicitation, HostEvent, Sampling

FORMAT = "dllm-agent/1"
"""The transcript format; a reader refuses others."""


def _calls(calls: Sequence[ToolCall]) -> list[dict[str, str]]:
    return [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in calls]


def transcript_id(transcript: Mapping[str, Any]) -> str:
    """``trn_`` and the hash of the transcript without its id, so any edit is detected."""
    body = {k: v for k, v in transcript.items() if k not in ("id", "signature")}
    return "trn_" + receipts._sha256(receipts.canonical_json(body))[:32]


def server_request_record(answer: Sampling | Elicitation) -> dict[str, Any]:
    """The transcript entry of a sampling or an elicitation the engine answered."""
    from etalii_dllm.mcp_host import Sampling

    if isinstance(answer, Sampling):
        request = receipts.request_record(answer.request)
        return {"kind": "sampling", "server": answer.server, "request": request, "content": answer.content,
                "stop_reason": answer.stop_reason, "fingerprint": answer.fingerprint}  # fmt: skip
    request = None if answer.request is None else receipts.request_record(answer.request)
    return {"kind": "elicitation", "server": answer.server, "message": answer.message, "action": answer.action,
            "content": None if answer.content is None else dict(answer.content), "request": request,
            "fingerprint": answer.fingerprint}  # fmt: skip


def answer_again(engine: DllmEngine, record: Mapping[str, Any]) -> Sampling | Elicitation:
    """Runs the engine request of a transcript's server request again: the answer the engine gives now."""
    from etalii_dllm import mcp_host

    if record["request"] is None:  # an elicitation declined without running the engine
        return mcp_host.Elicitation(record["server"], record["message"], record["action"], None, None, None)
    request = receipts.request_from_record(record["request"])
    if record["kind"] == "sampling":
        return mcp_host.answer_sampling(engine, record["server"], request)
    return mcp_host.answer_elicitation(engine, record["server"], record["message"], request)


@dataclass
class Recorder:
    """Builds a transcript from the events of :func:`etalii_dllm.mcp_host.chat`: pass every event to :meth:`add`."""

    engine: DllmEngine
    request: ChatRequest
    tools: Sequence[Tool]
    max_rounds: int
    servers: Mapping[str, str] = field(default_factory=dict)
    """Tool name -> the MCP server it belongs to (recorded for the reader; replay does not need it)."""
    answers: Sequence[Sampling | Elicitation] = ()
    """The servers' sampling and elicitation requests the engine answered (:attr:`etalii_dllm.mcp_host.McpHost.
    answers`, read when the transcript is made)."""
    rounds: list[dict[str, Any]] = field(default_factory=list)
    _text: list[str] = field(default_factory=list)
    _calls: list[ToolCall] = field(default_factory=list)

    def add(self, event: HostEvent) -> None:
        from etalii_dllm.mcp_host import ToolResult

        if isinstance(event, TextDelta):
            self._text.append(event.text)
        elif isinstance(event, ToolCallEvent):
            self._calls.append(event.call)
        elif isinstance(event, Finished):
            self.rounds.append(
                {"receipt": event.receipt, "content": "".join(self._text), "tool_calls": _calls(self._calls),
                 "results": []}
            )  # fmt: skip
            self._text, self._calls = [], []
        elif isinstance(event, ToolResult):
            self.rounds[-1]["results"].append(
                {"id": event.call.id, "name": event.call.name, "server": event.server, "content": event.content,
                 "is_error": event.is_error}
            )  # fmt: skip

    def transcript(self) -> dict[str, Any]:
        from etalii_dllm import __version__

        body = {
            "transcript": FORMAT,
            "engine": __version__,
            "model": self.engine.model.id,
            "system_fingerprint": self.engine.system_fingerprint,
            "request": receipts.request_record(self.request),
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                    "server": self.servers.get(t.name, ""),
                }
                for t in self.tools
            ],
            "max_rounds": self.max_rounds,
            "rounds": self.rounds,
        }
        if self.answers:
            body["server_requests"] = [server_request_record(answer) for answer in self.answers]
        transcript = {**body, "id": transcript_id(body)}
        return self.engine.signer.sign(transcript) if self.engine.signer is not None else transcript


class RecordedTools:
    """Stands in for :class:`etalii_dllm.mcp_host.McpHost`: offers the recorded tools and answers each call with the
    recorded result of the same position in the run. A call that differs from the recorded one still gets that
    result (the round's receipt already shows the divergence)."""

    def __init__(self, transcript: Mapping[str, Any]) -> None:
        self.tools = [Tool(t["name"], t["description"], t["parameters"]) for t in transcript["tools"]]
        self._results = [(r, result) for r in transcript["rounds"] for result in r["results"]]
        self._next = 0

    async def call(self, call: ToolCall) -> Any:
        from etalii_dllm.mcp_host import ToolResult

        if self._next >= len(self._results):
            return ToolResult(call, "", "no recorded result for this call", True)
        _, result = self._results[self._next]
        self._next += 1
        return ToolResult(call, result["server"], result["content"], result["is_error"])


@dataclass(frozen=True)
class Replay:
    """The outcome of :func:`replay`. ``ok`` only when every round gave its recorded receipt."""

    ok: bool
    reasons: tuple[str, ...]
    notes: tuple[str, ...]
    diverged_at: int | None
    """The first round (0-based) whose output differs from the transcript, else ``None``."""
    transcript: Mapping[str, Any]
    """The transcript the replay produced."""

    def to_json(self) -> dict[str, Any]:
        return {"ok": self.ok, "reasons": list(self.reasons), "notes": list(self.notes),
                "diverged_at": self.diverged_at, "transcript": dict(self.transcript)}  # fmt: skip


def _output(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    return receipt.get("output", {}) if isinstance(receipt, Mapping) else {}


def replay(engine: DllmEngine, transcript: Mapping[str, Any]) -> Replay:
    """Runs the recorded agent loop again on ``engine`` with the recorded tool results and compares every round."""
    import anyio

    from etalii_dllm import __version__, mcp_host

    if transcript.get("transcript") != FORMAT:
        raise ValueError(f"not a {FORMAT} transcript")
    reasons: list[str] = []
    notes: list[str] = []
    if transcript.get("id") != transcript_id(transcript):
        reasons.append("the transcript was edited: its id does not match its content")
    if transcript["system_fingerprint"] != engine.system_fingerprint:
        reasons.append(
            f"different weights or settings: the transcript was made with {transcript['system_fingerprint']}, "
            f"this engine is {engine.system_fingerprint}"
        )
    if transcript["engine"] != __version__:
        notes.append(f"made by engine version {transcript['engine']}, replayed with {__version__}")
    for index, recorded in enumerate(transcript["rounds"]):
        output = _output(recorded["receipt"])
        calls = [ToolCall(c["id"], c["name"], c["arguments"]) for c in recorded["tool_calls"]]
        expected = receipts.output_record("", recorded["content"], calls, "", 0, 0)
        if (output.get("content"), output.get("tool_calls")) != (expected["content"], expected["tool_calls"]):
            reasons.append(f"round {index}: the recorded text or tool calls do not match the round's receipt")

    request = receipts.request_from_record(transcript["request"])
    tools = RecordedTools(transcript)
    answers = [answer_again(engine, record) for record in transcript.get("server_requests", ())]
    for index, (record, answer) in enumerate(zip(transcript.get("server_requests", ()), answers, strict=True)):
        if server_request_record(answer) != record:
            reasons.append(f"server request {index} ({record['kind']} for {record['server']}): the engine's answer "
                           "differs from the transcript")  # fmt: skip
    recorder = Recorder(engine, request, tools.tools, transcript["max_rounds"],
                        {t["name"]: t["server"] for t in transcript["tools"]}, answers)  # fmt: skip

    async def run() -> None:
        async for event in mcp_host.chat(engine, request, tools, transcript["max_rounds"]):  # type: ignore[arg-type]
            recorder.add(event)

    anyio.run(run)
    replayed = recorder.transcript()
    diverged_at = None
    recorded_rounds, replayed_rounds = transcript["rounds"], replayed["rounds"]
    for index in range(max(len(recorded_rounds), len(replayed_rounds))):
        if index >= len(recorded_rounds) or index >= len(replayed_rounds):
            diverged_at = index
            reasons.append(f"the run has {len(replayed_rounds)} rounds, the transcript {len(recorded_rounds)}")
            break
        old, new = recorded_rounds[index]["receipt"], replayed_rounds[index]["receipt"]
        same_request = old.get("request") == new.get("request")
        if not same_request or _output(old) != _output(new):
            diverged_at = index
            # A different conversation (an edited tool result, say) is the cause; else the model answered otherwise.
            what = "the model's answer" if same_request else "its conversation"
            reasons.append(f"round {index} diverges: {what} differs from the transcript")
            break
    return Replay(not reasons, tuple(reasons), tuple(notes), diverged_at, replayed)
