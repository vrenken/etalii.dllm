"""Training examples of T5 text-to-text models (#388): source and target pairs with a fixed order.

A ``.jsonl`` file holds one object per line: ``{"input": ..., "target": ...}``, ``{"prompt": ..., "completion":
...}``, or ``{"messages": [...]}`` (a conversation whose last message is the assistant's answer, as ``dllm finetune
--teacher`` writes them; the source is the other messages' contents joined by a blank line, as the engine renders a
chat for a text-to-text model). Each text is tokenized without special tokens, cut to ``sequence_length - 1`` tokens
and ended with ``</s>``. Examples are visited in the per-epoch permutation of
:func:`~etalii_dllm.training.data.batch_indices`, so the examples of any step follow from the seed and the step alone.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from etalii_dllm.training.data import TrainingDataError, batch_indices


def read_text_pairs(path: str | Path) -> list[tuple[str, str]]:
    """The (source, target) texts of a ``.jsonl`` file, in file order."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise TrainingDataError(f"{path}: {error}") from error
    pairs = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise TrainingDataError(f"{path}:{number}: invalid JSON ({error.msg})") from error
        pair = _pair(record)
        if pair is None:
            raise TrainingDataError(
                f"{path}:{number}: expected an object with 'input' and 'target', 'prompt' and 'completion', or "
                "'messages' ending with the assistant's answer"
            )
        pairs.append(pair)
    if not pairs:
        raise TrainingDataError(f"{path}: no examples")
    return pairs


def _pair(record: object) -> tuple[str, str] | None:
    if not isinstance(record, dict):
        return None
    for source, target in (("input", "target"), ("prompt", "completion")):
        if isinstance(record.get(source), str) and isinstance(record.get(target), str):
            return record[source], record[target]
    messages = record.get("messages")
    if isinstance(messages, list) and len(messages) >= 2 and all(isinstance(m, dict) for m in messages):
        *before, answer = messages
        if answer.get("role") == "assistant" and isinstance(answer.get("content"), str):
            return join_messages(before), answer["content"]
    return None


def join_messages(messages: Sequence[Mapping[str, Any]]) -> str:
    """A conversation as a text-to-text model's source: the messages' contents joined by a blank line, as the engine
    renders a chat for one."""
    return "\n\n".join(str(m.get("content") or "") for m in messages if m.get("content"))


def text_to_text_tokens(
    encode: Callable[[str], list[int]], text: str, sequence_length: int, end: int
) -> tuple[int, ...]:
    """``text``'s tokens cut to ``sequence_length - 1`` and ended with ``</s>`` (a trailing ``</s>`` the tokenizer
    adds itself is dropped first)."""
    ids = list(encode(text))
    if ids and ids[-1] == end:
        ids = ids[:-1]
    return (*ids[: sequence_length - 1], end)


@dataclass(frozen=True)
class TextToTextData:
    """Tokenized examples: ``examples[i]`` is ``(source, target)``, each ending with ``</s>``."""

    examples: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    sequence_length: int

    @classmethod
    def from_pairs(
        cls,
        pairs: Sequence[tuple[str, str]],
        encode: Callable[[str], list[int]],
        sequence_length: int,
        end: int,
    ) -> TextToTextData:
        if sequence_length < 2:
            raise TrainingDataError("sequence_length must be at least 2 (a token and </s>)")
        if not pairs:
            raise TrainingDataError("the data holds no examples")

        def tokens(text: str) -> tuple[int, ...]:
            return text_to_text_tokens(encode, text, sequence_length, end)

        return cls(tuple((tokens(source), tokens(target)) for source, target in pairs), sequence_length)

    @classmethod
    def from_file(
        cls, path: str | Path, encode: Callable[[str], list[int]], sequence_length: int, end: int
    ) -> TextToTextData:
        return cls.from_pairs(read_text_pairs(path), encode, sequence_length, end)

    def __len__(self) -> int:
        return len(self.examples)

    @property
    def fingerprint(self) -> str:
        """SHA-256 of the examples (lengths and token ids, little-endian int64): what the run trained on."""
        digest = hashlib.sha256(np.asarray([self.sequence_length, len(self.examples)], dtype="<i8").tobytes())
        for source, target in self.examples:
            digest.update(np.asarray([len(source), *source, len(target), *target], dtype="<i8").tobytes())
        return digest.hexdigest()

    def batch(self, step: int, batch_size: int, seed: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """The examples of step ``step`` (0-based): samples ``step * batch_size ..`` of the epoch-by-epoch order."""
        return [self.examples[index] for index in batch_indices(len(self.examples), step, batch_size, seed)]
