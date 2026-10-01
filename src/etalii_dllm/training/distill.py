"""Distillation that replays: a student fine-tuned on a teacher's recorded answers (``dllm distill``).

The teacher answers every prompt greedily, so its answers are a pure function of its weights and the prompts. They
are written as chat records (canonical JSON lines, the same bytes on every platform) and the student is fine-tuned on
them like on any other data. The run records where the data came from (``distillation``): the teacher's fingerprint,
the prompts file's SHA-256, the token budget and every answer's receipt id. ``dllm replay`` of the training receipt
with ``--teacher`` regenerates the answers and checks they are the data the run used, then trains again.

A prompts file holds one prompt per line: a user message, or a JSON object with ``messages`` (a conversation for the
teacher to continue).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from etalii_dllm.chat import ChatMessage


def _conversation(line: str) -> list[dict[str, str]]:
    text = line.strip()
    if text.startswith("{"):
        record = json.loads(text)
        if not isinstance(record, dict) or not isinstance(record.get("messages"), list):
            raise ValueError(f"expected an object with 'messages': {text[:60]}")
        return [{"role": str(m["role"]), "content": str(m["content"])} for m in record["messages"]]
    return [{"role": "user", "content": text}]


def write_teacher_data(
    teacher: str | Path, prompts: str | Path, output: str | Path, max_tokens: int = 256
) -> dict[str, Any]:
    """Writes the teacher's greedy answer to every prompt as chat records to ``output`` (``.jsonl``); returns the
    ``distillation`` record."""
    from etalii_dllm.engine import ChatRequest, DllmEngine
    from etalii_dllm.receipts import canonical_json

    prompts_bytes = Path(prompts).read_bytes()
    conversations = [_conversation(line) for line in prompts_bytes.decode("utf-8").splitlines() if line.strip()]
    if not conversations:
        raise ValueError(f"{prompts}: no prompts")
    engine = DllmEngine.from_model_file(teacher, verify=False, prompt_cache=0)
    lines, receipts = [], []
    for messages in conversations:
        request = ChatRequest([ChatMessage(m["role"], m["content"]) for m in messages], max_tokens)
        result = engine.chat_completion(request, fresh=True)
        assert result.receipt is not None
        receipts.append(result.receipt["id"])
        lines.append(canonical_json({"messages": [*messages, {"role": "assistant", "content": result.content}]}))
    Path(output).write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    from etalii_dllm.modelfile import ModelFile

    return {
        "teacher": ModelFile(teacher, verify=False).fingerprint,
        "prompts": {"file": str(prompts), "sha256": hashlib.sha256(prompts_bytes).hexdigest()},
        "max_tokens": max_tokens,
        "examples": len(lines),
        "receipts": receipts,
    }


def check_teacher_data(
    distillation: Mapping[str, Any],
    teacher: str | Path,
    data_sha256: str,
    prompts: str | Path | None = None,
) -> list[str]:
    """Regenerates the teacher's answers and says why they are not the data the run used (empty when they are)."""
    import tempfile

    prompts = prompts if prompts is not None else distillation["prompts"]["file"]
    reasons = []
    if hashlib.sha256(Path(prompts).read_bytes()).hexdigest() != distillation["prompts"]["sha256"]:
        reasons.append(f"different prompts: {prompts} is not the file the teacher answered")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "teacher.jsonl"
        again = write_teacher_data(teacher, prompts, path, int(distillation["max_tokens"]))
        if again["teacher"] != distillation["teacher"]:
            reasons.append(f"a different teacher: the run used {distillation['teacher']}, this is {again['teacher']}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != data_sha256:
            reasons.append("the teacher's answers differ from the training data the run used")
    return reasons
