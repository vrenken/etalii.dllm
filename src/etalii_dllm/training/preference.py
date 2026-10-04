"""Preference data for direct preference optimization (``dllm finetune --dpo``, ``docs/training.md#preference-tuning``).

A preference file is JSON Lines, one pair per line: a prompt and two answers, the one to prefer and the one to avoid::

    {"prompt": "Q: 2+2?\\nA:", "chosen": " 4", "rejected": " 5"}
    {"messages": [{"role": "user", "content": "Hi"}], "chosen": "Hello!", "rejected": "Go away."}

A ``messages`` prompt is rendered with the model's chat template and its generation prompt; a ``prompt`` is used as
written. The prompt and each answer are tokenized separately and the answer gets the model's end-of-sequence token, so
a pair's tokens do not depend on the rest of the file. A pair holds at most ``sequence_length + 1`` tokens per side:
an answer that does not fit is cut at its end; a prompt must leave room for at least one answer token.

For a T5 text-to-text model (:meth:`PreferenceData.from_text_to_text_records`, #397) the prompt is the source: a
``messages`` prompt is the messages' contents joined by a blank line, and the source and each answer are cut to
``sequence_length - 1`` tokens and ended with ``</s>``, as :class:`~etalii_dllm.training.seq2seq_data.TextToTextData`
cuts them; an answer's log-probability is that of its tokens given the source.

Pairs are visited in the same per-epoch ``DeterministicRandom`` order as training windows
(:func:`~etalii_dllm.training.data.batch_indices`), so the pairs of a step follow from the seed and step alone.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from etalii_dllm.numerics import exp, log
from etalii_dllm.training.data import TrainingDataError, batch_indices, epoch_order

_DOMAIN = b"dllm-preference/1\0"


@dataclass(frozen=True)
class PreferenceRecord:
    """One pair as text: the rendered prompt and the two answers."""

    prompt: str
    chosen: str
    rejected: str


def read_pairs(
    path: str | Path, render_chat: Callable[[Sequence[Mapping[str, Any]]], str] | None
) -> list[PreferenceRecord]:
    """The pairs of a preference ``.jsonl`` file, in file order. ``render_chat`` renders a ``messages`` prompt (with
    the generation prompt)."""
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
        answers = isinstance(record, dict) and all(isinstance(record.get(key), str) for key in ("chosen", "rejected"))
        if not answers:
            raise TrainingDataError(f"{path}:{number}: expected an object with 'chosen' and 'rejected' answers")
        if isinstance(record.get("prompt"), str):
            prompt = record["prompt"]
        elif isinstance(record.get("messages"), list):
            if render_chat is None:
                raise TrainingDataError(f"{path}:{number}: chat prompts need a model with a chat template")
            prompt = render_chat(record["messages"])
        else:
            raise TrainingDataError(f"{path}:{number}: expected a 'prompt' or 'messages'")
        records.append(PreferenceRecord(prompt, record["chosen"], record["rejected"]))
    if not records:
        raise TrainingDataError(f"{path}: no preference pairs")
    return records


@dataclass(frozen=True)
class PreferencePair:
    """A tokenized pair."""

    prompt: tuple[int, ...]
    chosen: tuple[int, ...]
    rejected: tuple[int, ...]

    def sequence(self, which: str) -> tuple[list[int], list[int]]:
        """Inputs and targets for scoring the ``chosen`` or ``rejected`` answer: the targets over the prompt are
        ``-1`` (not scored), so the summed cross-entropy is minus the answer's log-probability."""
        answer = self.chosen if which == "chosen" else self.rejected
        tokens = [*self.prompt, *answer]
        return tokens[:-1], [-1] * (len(self.prompt) - 1) + list(answer)


@dataclass(frozen=True)
class PreferenceData:
    """Tokenized preference pairs for one sequence length."""

    pairs: tuple[PreferencePair, ...]
    sequence_length: int

    @classmethod
    def from_records(
        cls,
        records: Sequence[PreferenceRecord],
        encode: Callable[[str], list[int]],
        sequence_length: int,
        end: int | None,
    ) -> PreferenceData:
        if sequence_length < 1:
            raise TrainingDataError("sequence_length must be positive")
        pairs = []
        for number, record in enumerate(records, 1):
            prompt = tuple(encode(record.prompt))
            if not prompt:
                raise TrainingDataError(f"pair {number}: the prompt is empty")
            if len(prompt) > sequence_length:
                raise TrainingDataError(
                    f"pair {number}: the prompt has {len(prompt)} tokens, more than the sequence length allows"
                )
            room = sequence_length + 1 - len(prompt)
            suffix = () if end is None else (end,)
            chosen = (*encode(record.chosen), *suffix)[:room]
            rejected = (*encode(record.rejected), *suffix)[:room]
            if not chosen or not rejected:
                raise TrainingDataError(f"pair {number}: an answer is empty")
            pairs.append(PreferencePair(prompt, chosen, rejected))
        return cls(tuple(pairs), sequence_length)

    @classmethod
    def from_text_to_text_records(
        cls,
        records: Sequence[PreferenceRecord],
        encode: Callable[[str], list[int]],
        sequence_length: int,
        end: int,
    ) -> PreferenceData:
        """Pairs for a text-to-text model: ``prompt`` is the source, each side ending with ``</s>``."""
        from etalii_dllm.training.seq2seq_data import text_to_text_tokens

        if sequence_length < 2:
            raise TrainingDataError("sequence_length must be at least 2 (a token and </s>)")

        def tokens(text: str) -> tuple[int, ...]:
            return text_to_text_tokens(encode, text, sequence_length, end)

        pairs = [PreferencePair(tokens(r.prompt), tokens(r.chosen), tokens(r.rejected)) for r in records]
        return cls(tuple(pairs), sequence_length)

    def __len__(self) -> int:
        return len(self.pairs)

    @property
    def fingerprint(self) -> str:
        """SHA-256 of the pairs (lengths and token ids, little-endian int64), domain-separated from windows."""
        digest = hashlib.sha256(_DOMAIN)
        digest.update(np.asarray([self.sequence_length, len(self.pairs)], dtype="<i8").tobytes())
        for pair in self.pairs:
            for part in (pair.prompt, pair.chosen, pair.rejected):
                digest.update(np.asarray([len(part), *part], dtype="<i8").tobytes())
        return digest.hexdigest()

    def epoch_order(self, seed: int, epoch: int) -> list[int]:
        return epoch_order(len(self.pairs), seed, epoch)

    def batch(self, step: int, batch_size: int, seed: int) -> list[int]:
        """The pair indices of step ``step`` (0-based)."""
        return batch_indices(len(self.pairs), step, batch_size, seed)


def log_sigmoid(z: float) -> float:
    """``log(sigmoid(z))`` in double with the portable kernels, without overflow for either sign."""
    if z >= 0.0:
        return -log(1.0 + exp(-z))
    return z - log(1.0 + exp(z))
