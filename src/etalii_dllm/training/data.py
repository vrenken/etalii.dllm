"""Training data with a fixed order: documents are tokenized, packed into equal windows, and visited in a per-epoch
permutation drawn from ``DeterministicRandom``. The window used at any step follows from the seed and the step
number alone, so a run resumed from a checkpoint reads exactly the data an uninterrupted run would have read.

Input files:

- ``*.txt``: the whole file is one document;
- ``*.jsonl``: one JSON object per line, either ``{"text": ...}`` or ``{"messages": [{"role", "content"}, ...]}``
  (rendered with the model's chat template, without a generation prompt).

Documents are joined in file order, each followed by the model's end-of-sequence token, and cut into windows of
``sequence_length + 1`` tokens that overlap by one (the inputs of a window are its first ``sequence_length`` tokens,
the targets its last ``sequence_length``). A shorter final window is kept when it has at least two tokens.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from etalii_dllm.numerics import DeterministicRandom

# Mixes the epoch into the shuffle seed (the 64-bit golden ratio, as in SplitMix64).
_EPOCH_MIX = 0x9E3779B97F4A7C15
_MASK64 = (1 << 64) - 1


class TrainingDataError(ValueError):
    """The data file cannot be read or holds no usable text."""


def read_documents(path: str | Path, render_chat: Callable[[Sequence[Mapping[str, Any]]], str] | None) -> list[str]:
    """The documents of a ``.txt`` or ``.jsonl`` file, in file order."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise TrainingDataError(f"{path}: {error}") from error
    if path.suffix.lower() != ".jsonl":
        return [text] if text else []
    documents = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise TrainingDataError(f"{path}:{number}: invalid JSON ({error.msg})") from error
        if isinstance(record, dict) and isinstance(record.get("text"), str):
            documents.append(record["text"])
        elif isinstance(record, dict) and isinstance(record.get("messages"), list):
            if render_chat is None:
                raise TrainingDataError(f"{path}:{number}: chat records need a model with a chat template")
            documents.append(render_chat(record["messages"]))
        else:
            raise TrainingDataError(f"{path}:{number}: expected an object with 'text' or 'messages'")
    return documents


@dataclass(frozen=True)
class TrainingData:
    """Tokenized training windows. ``windows[i]`` holds ``sequence_length + 1`` token ids (or fewer, last only)."""

    windows: tuple[tuple[int, ...], ...]
    sequence_length: int

    @classmethod
    def from_documents(
        cls,
        documents: Sequence[str],
        encode: Callable[[str], list[int]],
        sequence_length: int,
        separator: int | None,
    ) -> TrainingData:
        if sequence_length < 1:
            raise TrainingDataError("sequence_length must be positive")
        stream: list[int] = []
        for document in documents:
            stream.extend(encode(document))
            if separator is not None:
                stream.append(separator)
        windows = []
        for start in range(0, max(len(stream) - 1, 0), sequence_length):
            window = tuple(stream[start : start + sequence_length + 1])
            if len(window) >= 2:
                windows.append(window)
        if not windows:
            raise TrainingDataError("the data holds fewer than two tokens")
        return cls(tuple(windows), sequence_length)

    def __len__(self) -> int:
        return len(self.windows)

    @property
    def fingerprint(self) -> str:
        """SHA-256 of the windows (lengths and token ids, little-endian int64): what the run trained on."""
        digest = hashlib.sha256(np.asarray([self.sequence_length, len(self.windows)], dtype="<i8").tobytes())
        for window in self.windows:
            digest.update(np.asarray([len(window), *window], dtype="<i8").tobytes())
        return digest.hexdigest()

    def epoch_order(self, seed: int, epoch: int) -> list[int]:
        """The window order of ``epoch``: a Fisher-Yates shuffle driven by ``DeterministicRandom``."""
        random = DeterministicRandom((seed + (epoch + 1) * _EPOCH_MIX) & _MASK64)
        order = list(range(len(self.windows)))
        for i in range(len(order) - 1, 0, -1):
            j = random.next_u64() % (i + 1)
            order[i], order[j] = order[j], order[i]
        return order

    def batch(self, step: int, batch_size: int, seed: int) -> list[tuple[int, ...]]:
        """The windows of step ``step`` (0-based): samples ``step * batch_size ..`` of the epoch-by-epoch order."""
        count = len(self.windows)
        orders: dict[int, list[int]] = {}
        result = []
        for sample in range(step * batch_size, (step + 1) * batch_size):
            epoch, index = divmod(sample, count)
            if epoch not in orders:
                orders[epoch] = self.epoch_order(seed, epoch)
            result.append(self.windows[orders[epoch][index]])
        return result
