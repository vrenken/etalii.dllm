"""Training receipts: a fine-tune that anyone can run again and check bit for bit (``dllm finetune --receipt``).

Fine-tuning is deterministic (fixed data order, fixed-order gradient kernels, double accumulators), so the weights a
run gives are a pure function of the base weights, the training data and the run settings. A training receipt writes
those down, with every step's loss and the fingerprint of the weights that came out; :func:`verify` (``dllm replay``)
trains again and reports the first step whose loss differs, or confirms the same weights::

    {
      "training_receipt": "dllm-train/1",
      "id": "trn_rcpt_...",               # hash of everything below
      "engine": "0.2.0",
      "base_fingerprint": "...",          # the model.dllm the run started from
      "data": {"file": "notes.txt", "sha256": "...", "fingerprint": "...", "windows": 120},
      "run": {...},                       # RunConfig: steps, batch size, sequence length, seed, AdamW, LoRA
      "losses": ["0x1.2p+1", ...],        # every step's loss, exactly (hex floats)
      "output": {"fingerprint": "...", "steps": 100}
    }

A preference run (``dllm finetune --dpo``) records ``"pairs"`` instead of ``"windows"``; its ``run`` holds the
objective and ``beta``.

``data.file`` is the path as given, for convenience; ``data.sha256`` (the file) and ``data.fingerprint`` (the token
windows, which also depend on the tokenizer and chat template) decide.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from etalii_dllm import receipts
from etalii_dllm.modelfile import ModelFile, data_fingerprint
from etalii_dllm.training.data import TrainingData, read_documents
from etalii_dllm.training.preference import PreferenceData, read_pairs
from etalii_dllm.training.trainer import FineTuner, RunConfig, StepResult

if TYPE_CHECKING:
    from etalii_dllm.engine import DllmEngine

FORMAT = "dllm-train/1"
"""The training receipt format; a reader refuses others."""


def load_data(
    engine: DllmEngine, base: ModelFile, path: str | Path, sequence_length: int, objective: str = "lm"
) -> TrainingData | PreferenceData:
    """The training windows of ``path`` as ``dllm finetune`` builds them: chats rendered with the model's own
    template, documents separated by its end-of-sequence token. For ``objective="dpo"``, the preference pairs of
    ``path`` (chat prompts rendered with the generation prompt, answers ended by the end-of-sequence token)."""
    render: Callable[[Any], str] | None = None
    prompt = objective == "dpo"
    if engine.chat_template is not None:
        template = engine.chat_template
        render = lambda messages: template.render(messages, add_generation_prompt=prompt)  # noqa: E731
    separator = base.config.eos_token_ids[0] if base.config.eos_token_ids else None
    if prompt:
        pairs = read_pairs(path, render)
        return PreferenceData.from_records(pairs, engine.tokenizer.encode, sequence_length, separator)
    documents = read_documents(path, render)
    return TrainingData.from_documents(documents, engine.tokenizer.encode, sequence_length, separator)


def receipt_id(receipt: Mapping[str, Any]) -> str:
    body = {k: v for k, v in receipt.items() if k not in ("id", "signature")}
    return "trn_rcpt_" + hashlib.sha256(receipts.canonical_json(body).encode()).hexdigest()[:32]


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_receipt(tuner: FineTuner, data_file: str | Path) -> dict[str, Any]:
    """The receipt of ``tuner``'s run so far (normally finished)."""
    from etalii_dllm import __version__

    body = {
        "training_receipt": FORMAT,
        "engine": __version__,
        "base_fingerprint": tuner.base_fingerprint,
        "data": {
            "file": str(data_file),
            "sha256": _file_sha256(data_file),
            "fingerprint": tuner.data.fingerprint,
            ("pairs" if isinstance(tuner.data, PreferenceData) else "windows"): len(tuner.data),
        },
        "run": tuner.run.to_dict(),
        "losses": [float(loss).hex() for loss in tuner.losses],
        "output": {"fingerprint": data_fingerprint(tuner.weights()), "steps": tuner.step},
    }
    if tuner.distillation is not None:
        body["distillation"] = tuner.distillation
    return {**body, "id": receipt_id(body)}


@dataclass(frozen=True)
class TrainingVerification:
    """The outcome of :func:`verify`. ``ok`` only when the run gave the recorded weights."""

    ok: bool
    reasons: tuple[str, ...]
    notes: tuple[str, ...]
    diverged_at: int | None
    """The first step (1-based) whose loss differs from the receipt, else ``None``."""
    receipt: Mapping[str, Any]
    """The receipt the replay produced."""

    def to_json(self) -> dict[str, Any]:
        return {"ok": self.ok, "reasons": list(self.reasons), "notes": list(self.notes),
                "diverged_at": self.diverged_at, "receipt": dict(self.receipt)}  # fmt: skip


def verify(
    receipt: Mapping[str, Any],
    base_path: str | Path,
    data_file: str | Path | None = None,
    on_step: Callable[[StepResult], None] | None = None,
    teacher: str | Path | None = None,
    prompts: str | Path | None = None,
) -> TrainingVerification:
    """Trains again from ``base_path`` on ``data_file`` (default: the file the receipt names) with the recorded
    settings, and compares every step's loss and the resulting weights with the receipt. A distillation receipt's
    data are regenerated first with ``teacher`` from ``prompts`` (default: the file the receipt names)."""
    from etalii_dllm import __version__
    from etalii_dllm.engine import DllmEngine

    if receipt.get("training_receipt") != FORMAT:
        raise ValueError(f"not a {FORMAT} training receipt")
    reasons: list[str] = []
    notes: list[str] = []
    if receipt.get("id") != receipt_id(receipt):
        reasons.append("the receipt was edited: its id does not match its content")
    base = ModelFile(base_path)
    if base.fingerprint != receipt["base_fingerprint"]:
        reasons.append(
            f"a different base model: the run started from {receipt['base_fingerprint']}, this is {base.fingerprint}"
        )
    if receipt["engine"] != __version__:
        notes.append(f"made by engine version {receipt['engine']}, replayed with {__version__}")
    data_file = data_file if data_file is not None else receipt["data"]["file"]
    if _file_sha256(data_file) != receipt["data"]["sha256"]:
        reasons.append(f"different training data: {data_file} is not the file the run used")
    if receipt.get("distillation") and teacher is not None:
        from etalii_dllm.training.distill import check_teacher_data

        reasons.extend(check_teacher_data(receipt["distillation"], teacher, receipt["data"]["sha256"], prompts))
    elif receipt.get("distillation"):
        notes.append("the teacher's answers were not regenerated (pass --teacher to check them too)")
    run = RunConfig.from_dict(receipt["run"])
    engine = DllmEngine.from_model_file(base_path, verify=False, prompt_cache=0)
    data = load_data(engine, base, data_file, run.sequence_length, run.objective)
    if data.fingerprint != receipt["data"]["fingerprint"]:
        reasons.append("the training data differ (the data, tokenizer or chat template changed)")
    tuner = FineTuner.from_model_file(base, data, run)
    recorded = receipt["losses"]
    diverged_at = None

    def step(result: StepResult) -> None:
        nonlocal diverged_at
        index = result.step - 1
        if diverged_at is None and (index >= len(recorded) or float(result.loss).hex() != recorded[index]):
            diverged_at = result.step
        if on_step is not None:
            on_step(result)

    tuner.train(until=receipt["output"]["steps"], on_step=step)
    replayed = make_receipt(tuner, data_file)
    if diverged_at is not None:
        reasons.append(f"step {diverged_at} gave another loss than the receipt records")
    if replayed["output"]["fingerprint"] != receipt["output"]["fingerprint"]:
        reasons.append(
            f"the weights differ: recorded {receipt['output']['fingerprint']}, "
            f"replayed {replayed['output']['fingerprint']}"
        )
    return TrainingVerification(not reasons, tuple(reasons), tuple(notes), diverged_at, replayed)
