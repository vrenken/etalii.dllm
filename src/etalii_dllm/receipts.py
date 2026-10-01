"""Generation receipts: a response that anyone can check later.

Every output of the engine is a pure function of the weights, the engine version and the request, so a response can
carry a small record of exactly those inputs plus hashes of what came out. Re-running the recorded request on the
same weights must give the same hashes, bit for bit, on any supported machine (:func:`verify`, ``dllm replay``).

A receipt is plain JSON::

    {
      "receipt": "dllm/1",
      "id": "rcpt_...",                  # hash of everything below
      "engine": "0.2.0",                 # etalii_dllm.__version__
      "model": "SmolLM2-135M-Instruct",
      "system_fingerprint": "fp_...",    # weights, quantisation, steering, document index
      "request": {...},                  # the engine request (request_record), enough to run it again
      "output": {"tokens": "...", "content": "...", "tool_calls": "...", "finish_reason": "stop",
                 "prompt_tokens": 12, "completion_tokens": 7},
      "previous": "rcpt_..."             # only in a conversation: the receipt of the turn before
    }

``output.tokens`` hashes the generated token ids, ``output.content`` and ``output.tool_calls`` the text and calls the
client saw. Settings that never change a bit (threads, device, prompt cache, speculative decoding, batching) are not
recorded. Nothing in a receipt comes from a clock or a random source, so the same request gives the same receipt.

The receipts of a conversation's turns form a chain: each names the one before as ``previous``, and
:func:`verify_chain` checks every turn and that each turn's conversation continues the previous answer exactly.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from etalii_dllm.chat import ChatMessage, ToolCall
from etalii_dllm.sampling import SamplingOptions
from etalii_dllm.tools import Tool, ToolChoice

if TYPE_CHECKING:
    from etalii_dllm.engine import ChatRequest, DllmEngine

FORMAT = "dllm/1"
"""The receipt format; a reader refuses others."""


def canonical_json(value: Any) -> str:
    """JSON with sorted keys and no insignificant whitespace: the bytes every hash in a receipt is taken over."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def request_record(request: ChatRequest) -> dict[str, Any]:
    """The engine request as JSON: every field that can change the output, and nothing else."""
    options = request.options
    return {
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in m.tool_calls],
                "tool_call_id": m.tool_call_id,
                "name": m.name,
            }
            for m in request.messages
        ],
        "prompt": request.prompt,
        "max_tokens": request.max_tokens,
        "options": {
            "temperature": options.temperature,
            "top_k": options.top_k,
            "top_p": options.top_p,
            "seed": options.seed,
        },
        "stop": list(request.stop),
        "tools": [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in request.tools],
        "tool_choice": {"mode": request.tool_choice.mode, "name": request.tool_choice.name},
        "response_format": {"type": request.response_format.type, "schema": request.response_format.schema},
        "top_logprobs": request.top_logprobs,
        "call_id_prefix": request.call_id_prefix,
        "request_id": request.request_id,
    }


def request_from_record(record: Mapping[str, Any]) -> ChatRequest:
    """The inverse of :func:`request_record`."""
    from etalii_dllm.engine import ChatRequest, ResponseFormat

    options = record["options"]
    choice = record["tool_choice"]
    response_format = record["response_format"]
    return ChatRequest(
        messages=[
            ChatMessage(
                m["role"],
                m["content"],
                tuple(ToolCall(c["id"], c["name"], c["arguments"]) for c in m["tool_calls"]),
                m["tool_call_id"],
                m["name"],
            )
            for m in record["messages"]
        ],
        max_tokens=record["max_tokens"],
        options=SamplingOptions(
            temperature=options["temperature"], top_k=options["top_k"], top_p=options["top_p"], seed=options["seed"]
        ),
        stop=tuple(record["stop"]),
        tools=tuple(Tool(t["name"], t["description"], t["parameters"]) for t in record["tools"]),
        tool_choice=ToolChoice(choice["mode"], choice["name"]),
        response_format=ResponseFormat(response_format["type"], response_format["schema"]),
        top_logprobs=record["top_logprobs"],
        call_id_prefix=record["call_id_prefix"],
        request_id=record["request_id"],
        prompt=record["prompt"],
    )


def output_record(
    token_fingerprint: str,
    content: str,
    tool_calls: Sequence[ToolCall],
    finish_reason: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, Any]:
    calls = [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in tool_calls]
    return {
        "tokens": token_fingerprint,
        "content": _sha256(content),
        "tool_calls": _sha256(canonical_json(calls)),
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def make_receipt(
    engine_version: str,
    model: str,
    system_fingerprint: str,
    request: ChatRequest,
    output: Mapping[str, Any],
    previous: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "receipt": FORMAT,
        "engine": engine_version,
        "model": model,
        "system_fingerprint": system_fingerprint,
        "request": request_record(request),
        "output": dict(output),
    }
    if previous:
        body["previous"] = previous
    return {**body, "id": receipt_id(body)}


def receipt_id(receipt: Mapping[str, Any]) -> str:
    """``rcpt_`` and the hash of the receipt without its id, so any edit to a receipt is detected."""
    body = {k: v for k, v in receipt.items() if k != "id"}
    return "rcpt_" + _sha256(canonical_json(body))[:32]


@dataclass(frozen=True)
class Verification:
    """The outcome of re-running a receipt's request. ``ok`` only when every output hash matched."""

    ok: bool
    reasons: tuple[str, ...]
    """Why it does not verify (empty when it does), most fundamental first."""
    notes: tuple[str, ...]
    """Differences that do not make it fail, such as another engine version that gives the same output."""
    receipt: Mapping[str, Any]
    """The receipt the replay produced."""

    def to_json(self) -> dict[str, Any]:
        return {"ok": self.ok, "reasons": list(self.reasons), "notes": list(self.notes), "receipt": dict(self.receipt)}


def verify(engine: DllmEngine, receipt: Mapping[str, Any]) -> Verification:
    """Re-runs the recorded request on ``engine`` and compares the output with the receipt.

    A receipt whose id does not match its content, or that was made by other weights, fails; the request is still
    replayed, so the reasons also say what actually differs. Another engine version is only a note: versions that
    keep the kernels and the sampler give the same output, and the output hashes decide."""
    if receipt.get("receipt") != FORMAT:
        raise ValueError(f"not a {FORMAT} receipt")
    reasons: list[str] = []
    if receipt.get("id") != receipt_id(receipt):
        reasons.append("the receipt was edited: its id does not match its content")
    if receipt["system_fingerprint"] != engine.system_fingerprint:
        reasons.append(
            f"different weights or settings: the receipt was made with {receipt['system_fingerprint']}, "
            f"this engine is {engine.system_fingerprint}"
        )
    from etalii_dllm import __version__

    notes = []
    if receipt["engine"] != __version__:
        notes.append(f"made by engine version {receipt['engine']}, replayed with {__version__}")
    replayed = engine.chat_completion(request_from_record(receipt["request"])).receipt
    expected, actual = receipt["output"], replayed["output"]
    for key, label in (
        ("prompt_tokens", "the prompt"),
        ("tokens", "the generated tokens"),
        ("content", "the text"),
        ("tool_calls", "the tool calls"),
        ("finish_reason", "the finish reason"),
        ("completion_tokens", "the number of generated tokens"),
    ):
        if expected.get(key) != actual.get(key):
            reasons.append(f"{label} differ: recorded {expected.get(key)!r}, replayed {actual.get(key)!r}")
    return Verification(not reasons, tuple(reasons), tuple(notes), replayed)


@dataclass(frozen=True)
class ChainVerification:
    """The outcome of :func:`verify_chain`. ``ok`` only when every turn verifies and continues the one before."""

    ok: bool
    reasons: tuple[str, ...]
    notes: tuple[str, ...]
    turns: tuple[Verification, ...]

    def to_json(self) -> dict[str, Any]:
        return {"ok": self.ok, "reasons": list(self.reasons), "notes": list(self.notes),
                "turns": [t.to_json() for t in self.turns]}  # fmt: skip


def _conversation(receipt: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The receipt's messages without a leading system message (instructions may change between turns)."""
    messages = list(receipt["request"]["messages"])
    return messages[1:] if messages and messages[0]["role"] == "system" else messages


def _continues(previous: Mapping[str, Any], receipt: Mapping[str, Any]) -> str | None:
    """Why ``receipt``'s conversation does not continue ``previous``'s answer exactly, else ``None``."""
    if previous["request"]["prompt"] is not None or receipt["request"]["prompt"] is not None:
        return "a raw prompt is not a conversation"
    before, after = _conversation(previous), _conversation(receipt)
    if after[: len(before)] != before:
        return "its conversation does not start with the previous turn's"
    if len(after) <= len(before) or after[len(before)]["role"] != "assistant":
        return "the previous answer is missing from its conversation"
    answer = after[len(before)]
    calls = [{"id": c["id"], "name": c["name"], "arguments": c["arguments"]} for c in answer["tool_calls"]]
    output = previous["output"]
    if _sha256(answer["content"]) != output.get("content"):
        return "the previous answer's text was changed"
    if _sha256(canonical_json(calls)) != output.get("tool_calls"):
        return "the previous answer's tool calls were changed"
    return None


def verify_chain(engine: DllmEngine, chain: Sequence[Mapping[str, Any]]) -> ChainVerification:
    """Verifies every receipt of a conversation (oldest first), that each names the one before as ``previous`` and
    that each turn's messages continue the previous turn's messages and answer exactly (a leading system message
    may differ, as instructions in the Responses API do not carry over)."""
    if not chain:
        raise ValueError("an empty receipt chain")
    reasons: list[str] = []
    notes: list[str] = []
    turns: list[Verification] = []
    for index, receipt in enumerate(chain):
        turn = verify(engine, receipt)
        turns.append(turn)
        reasons.extend(f"turn {index}: {reason}" for reason in turn.reasons)
        notes.extend(note for note in turn.notes if note not in notes)
        if index == 0:
            if receipt.get("previous"):
                notes.append(f"the chain starts after {receipt['previous']}, which it does not include")
            continue
        previous = chain[index - 1]
        if receipt.get("previous") != previous.get("id"):
            reasons.append(f"turn {index}: names {receipt.get('previous')!r} as previous, not {previous.get('id')!r}")
        problem = _continues(previous, receipt)
        if problem is not None:
            reasons.append(f"turn {index}: {problem}")
    return ChainVerification(not reasons, tuple(reasons), tuple(notes), tuple(turns))
