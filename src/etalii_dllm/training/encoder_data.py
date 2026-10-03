"""Training data for encoders (``dllm finetune --objective embedding|classifier``, ``docs/training.md#encoders``).

An embedding file is JSON Lines with one example per line: an anchor (a query, a sentence), a positive (a text that
belongs with it) and optionally a hard negative (a text that does not)::

    {"anchor": "how do I reset my password", "positive": "Open Settings, then Account, then Reset password."}
    {"anchor": "capital of France", "positive": "Paris is the capital of France.", "negative": "Lyon is in France."}

A classification file holds texts (a pair for cross-encoders: a query and a document) with a label: a number from 0
to 1 for a model with one output (a reranker's relevance), else the index of the right label::

    {"text": "how do I reset my password", "pair": "Open Settings, then Account, then Reset password.", "label": 1}
    {"text": "how do I reset my password", "pair": "Paris is the capital of France.", "label": 0}

Texts are tokenized as the model's own recipe does (``DllmEngine.embedding_tokens``: the default prompt and the
special tokens; ``DllmEngine.classification_tokens``: the pair template), truncated to the run's sequence length
(and the model's own limit) as sentence-transformers truncates, so each example's tokens depend only on its own
text. Examples are visited in the same per-epoch ``DeterministicRandom`` order as training windows
(:func:`~etalii_dllm.training.data.batch_indices`).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from etalii_dllm.training.data import TrainingDataError, batch_indices, epoch_order

if TYPE_CHECKING:
    from etalii_dllm.engine import DllmEngine

ENCODER_OBJECTIVES = ("embedding", "classifier")
_DOMAIN = b"dllm-encoder-data/1\0"


@dataclass(frozen=True)
class EncoderExample:
    """One tokenized example: its token sequences (anchor, positive and negative for an embedding example; one pair
    for a classification example), their token types and the label (classification only)."""

    texts: tuple[tuple[int, ...], ...]
    types: tuple[tuple[int, ...], ...]
    label: float | None = None


def read_examples(path: str | Path, objective: str) -> list[dict[str, Any]]:
    """The records of an embedding or classification ``.jsonl`` file, in file order, checked."""
    if objective not in ENCODER_OBJECTIVES:
        raise TrainingDataError(f"unknown encoder objective {objective!r}")
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise TrainingDataError(f"{path}: {error}") from error
    records = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise TrainingDataError(f"{path}:{number}: invalid JSON ({error.msg})") from error
        if not isinstance(record, dict):
            raise TrainingDataError(f"{path}:{number}: expected a JSON object")
        if objective == "embedding":
            texts = [record.get("anchor"), record.get("positive")]
            if not all(isinstance(t, str) for t in texts) or not isinstance(record.get("negative", ""), str):
                raise TrainingDataError(f"{path}:{number}: expected 'anchor' and 'positive' texts (and a 'negative')")
        else:
            label = record.get("label")
            if not isinstance(record.get("text"), str) or not isinstance(record.get("pair", ""), str):
                raise TrainingDataError(f"{path}:{number}: expected a 'text' (and a 'pair') with a 'label'")
            if isinstance(label, bool) or not isinstance(label, int | float) or not math.isfinite(label):
                raise TrainingDataError(f"{path}:{number}: the label must be a number")
        records.append(record)
    if not records:
        raise TrainingDataError(f"{path}: no examples")
    return records


@dataclass(frozen=True)
class EncoderData:
    """Tokenized encoder examples for one objective and sequence length."""

    examples: tuple[EncoderExample, ...]
    objective: str
    sequence_length: int

    @classmethod
    def from_records(
        cls, records: Sequence[dict[str, Any]], engine: DllmEngine, objective: str, sequence_length: int
    ) -> EncoderData:
        """Tokenizes ``records`` (:func:`read_examples`) with ``engine``'s model: an embedder for ``embedding``, a
        model with a classification head for ``classifier``."""
        if sequence_length < 1:
            raise TrainingDataError("sequence_length must be positive")
        examples = []
        labels = int(getattr(getattr(engine.model, "config", None), "classifier_labels", 0) or 0)
        for number, record in enumerate(records, 1):
            if objective == "embedding":
                names = ("anchor", "positive", "negative") if "negative" in record else ("anchor", "positive")
                texts = tuple(tuple(_embedding_tokens(engine, record[name], sequence_length)) for name in names)
                examples.append(EncoderExample(texts, tuple((0,) * len(t) for t in texts)))
                continue
            label = float(record["label"])
            if labels <= 1 and not 0.0 <= label <= 1.0:
                raise TrainingDataError(f"example {number}: a one-output model's label must be between 0 and 1")
            if labels > 1 and (label != int(label) or not 0 <= label < labels):
                raise TrainingDataError(f"example {number}: the label must be a label index below {labels}")
            tokens, types = _classification_tokens(engine, record["text"], record.get("pair"), sequence_length)
            examples.append(EncoderExample((tuple(tokens),), (tuple(types),), label))
        return cls(tuple(examples), objective, sequence_length)

    def __len__(self) -> int:
        return len(self.examples)

    @property
    def fingerprint(self) -> str:
        """SHA-256 of the examples (token ids, types and labels, little-endian), domain-separated from windows and
        preference pairs, and by objective."""
        digest = hashlib.sha256(_DOMAIN + self.objective.encode() + b"\0")
        digest.update(np.asarray([self.sequence_length, len(self.examples)], dtype="<i8").tobytes())
        for example in self.examples:
            digest.update(np.asarray([len(example.texts)], dtype="<i8").tobytes())
            for tokens, types in zip(example.texts, example.types, strict=True):
                digest.update(np.asarray([len(tokens), *tokens, *types], dtype="<i8").tobytes())
            if example.label is not None:
                digest.update(np.asarray([example.label], dtype="<f8").tobytes())
        return digest.hexdigest()

    def epoch_order(self, seed: int, epoch: int) -> list[int]:
        return epoch_order(len(self.examples), seed, epoch)

    def batch(self, step: int, batch_size: int, seed: int) -> list[int]:
        """The example indices of step ``step`` (0-based)."""
        return batch_indices(len(self.examples), step, batch_size, seed)


def _limit(engine: DllmEngine, settings: dict[str, Any] | None, sequence_length: int) -> int:
    limit = int((settings or {}).get("max_tokens") or engine.model.context_length)
    return min(limit, sequence_length)


def _embedding_tokens(engine: DllmEngine, text: str, sequence_length: int) -> list[int]:
    """``engine.embedding_tokens(text)`` truncated to ``sequence_length`` as sentence-transformers truncates (the
    special tokens kept)."""
    tokens = engine.embedding_tokens(text)
    limit = _limit(engine, engine.embedding, sequence_length)
    if len(tokens) <= limit:
        return tokens
    settings = engine.embedding or {}
    name = settings.get("default_prompt_name")
    prompt = (settings.get("prompts") or {}).get(name, "") if name else ""
    tokenizer = engine.tokenizer
    specials = len(tokenizer.encode("", add_special_tokens=True))  # type: ignore[call-arg]
    plain = tokenizer.encode(prompt + text)[: max(limit - specials, 0)]
    return tokenizer.with_special_tokens(plain)  # type: ignore[attr-defined,no-any-return]


def _classification_tokens(
    engine: DllmEngine, text: str, pair: str | None, sequence_length: int
) -> tuple[list[int], list[int]]:
    """``engine.classification_tokens(text, pair)`` with the run's sequence length as the limit."""
    limit = _limit(engine, engine.classifier, sequence_length)
    tokenizer = engine.tokenizer
    if pair is None:
        specials = len(tokenizer.encode("", add_special_tokens=True))  # type: ignore[call-arg]
        tokens = tokenizer.with_special_tokens(tokenizer.encode(text)[: max(limit - specials, 0)])  # type: ignore[attr-defined]
        return tokens, [0] * len(tokens)
    return tokenizer.encode_pair(text, pair, max_tokens=limit)  # type: ignore[attr-defined,no-any-return]
