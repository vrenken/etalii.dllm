"""Deterministic token sampling."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from etalii_dllm import watermark
from etalii_dllm.numerics import DeterministicRandom, FloatArray, argmax, softmax


@dataclass(frozen=True)
class SamplingOptions:
    """Controls token selection.

    With the same options, model and prompt the generated tokens are always identical: a temperature above zero
    draws from a seeded ``DeterministicRandom``, never from ambient entropy.
    """

    temperature: float = 0.0
    """0 means greedy decoding."""
    top_k: int = 0
    """Keep only the K most likely tokens; 0 disables."""
    top_p: float = 1.0
    """Nucleus sampling threshold in (0, 1]; 1 disables."""
    seed: int = 0
    """Seed for the sampler's random stream (unsigned 64-bit)."""
    min_p: float = 0.0
    """Drop candidates less likely than ``min_p`` times the most likely one; 0 disables."""
    repetition_penalty: float = 1.0
    """Divides positive (multiplies negative) logits of tokens among the last ``repeat_last_n`` of the prompt and
    output; 1 disables."""
    repeat_last_n: int = 64
    """How many of the latest tokens ``repetition_penalty`` looks at; -1 means all of them, 0 none."""
    frequency_penalty: float = 0.0
    """Subtracted from a token's logit once for every time the output already contains it."""
    presence_penalty: float = 0.0
    """Subtracted from a token's logit when the output already contains it."""
    logit_bias: tuple[tuple[int, float], ...] = ()
    """``(token, bias)`` pairs in ascending token order, added to the logits first."""
    watermark_key: str | None = None
    """Watermark the output with this key (:mod:`etalii_dllm.watermark`); ``None`` disables."""
    watermark_gamma: float = 0.25
    """The share of the vocabulary that is green after each token."""
    watermark_delta: float = 2.0
    """Added to the logits of the green tokens."""
    negative_prompt: str | None = None
    """Classifier-free guidance away from this prompt (:mod:`etalii_dllm.guidance`); ``None`` disables."""
    guidance_scale: float = 1.5
    """How far decoding moves from the negative prompt's logits towards the request's (1: not at all)."""
    contrast_beta: float | None = None
    """Contrastive decoding against the engine's amateur model with this strength; ``None`` disables."""
    contrast_alpha: float = 0.1
    """Contrastive decoding keeps only tokens at least ``contrast_alpha`` times as likely as the most likely one."""

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0 <= self.min_p <= 1:
            raise ValueError("min_p must be in [0, 1]")
        if not (math.isfinite(self.repetition_penalty) and self.repetition_penalty > 0):
            raise ValueError("repetition_penalty must be positive")
        if self.repeat_last_n < -1:
            raise ValueError("repeat_last_n must be -1 or more")
        if not (math.isfinite(self.frequency_penalty) and math.isfinite(self.presence_penalty)):
            raise ValueError("frequency_penalty and presence_penalty must be finite")
        tokens = [token for token, _ in self.logit_bias]
        if tokens != sorted(set(tokens)) or any(t < 0 for t in tokens):
            raise ValueError("logit_bias needs distinct non-negative token ids in ascending order")
        if not all(math.isfinite(bias) for _, bias in self.logit_bias):
            raise ValueError("logit_bias values must be finite")
        if self.watermark_key is not None and not self.watermark_key:
            raise ValueError("watermark_key must not be empty")
        if not 0.0 < self.watermark_gamma < 1.0:
            raise ValueError("watermark_gamma must be between 0 and 1")
        if not math.isfinite(self.watermark_delta):
            raise ValueError("watermark_delta must be finite")
        if not math.isfinite(self.guidance_scale):
            raise ValueError("guidance_scale must be finite")
        if self.contrast_beta is not None and not (math.isfinite(self.contrast_beta) and self.contrast_beta >= 0):
            raise ValueError("contrast_beta must be non-negative")
        if not 0.0 <= self.contrast_alpha <= 1.0:
            raise ValueError("contrast_alpha must be in [0, 1]")
        if self.negative_prompt is not None and self.contrast_beta is not None:
            raise ValueError("a request can use a negative prompt or contrastive decoding, not both")

    @staticmethod
    def bias(values: Mapping[int, float] | Mapping[str, float] | None) -> tuple[tuple[int, float], ...]:
        """A ``logit_bias`` mapping (token ids as ints or decimal strings, as JSON has them) in canonical form."""
        return tuple(sorted((int(token), float(bias)) for token, bias in (values or {}).items()))

    def for_choice(self, index: int) -> SamplingOptions:
        """The options of choice ``index`` of a request asking for several: the seed ``seed + index`` (mod 2^64),
        so every choice is exactly the answer a single request with that seed gives."""
        return self if index == 0 else replace(self, seed=(self.seed + index) & 0xFFFFFFFFFFFFFFFF)

    @property
    def adjusts_logits(self) -> bool:
        """Whether the logits change before temperature (bias or a penalty is set)."""
        return bool(self.logit_bias) or self.penalises or self.watermark_key is not None

    @property
    def penalises(self) -> bool:
        return (
            (self.repetition_penalty != 1.0 and self.repeat_last_n != 0)
            or self.frequency_penalty != 0.0
            or self.presence_penalty != 0.0
        )

    def record(self) -> dict[str, object]:
        """The options as JSON: the original four always, the Phase 21 ones only when they are not the default,
        so records (receipts, cache keys) written before them keep their bytes."""
        record: dict[str, object] = {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "seed": self.seed,
        }
        for name, default in _DEFAULTS.items():
            value = getattr(self, name)
            if value != default:
                record[name] = [[t, b] for t, b in value] if name == "logit_bias" else value
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> SamplingOptions:
        """The inverse of :meth:`record`."""
        values = dict(record)
        if "logit_bias" in values:
            values["logit_bias"] = tuple((int(t), float(b)) for t, b in values["logit_bias"])  # type: ignore[union-attr]
        return cls(**values)  # type: ignore[arg-type]


GREEDY = SamplingOptions()

_DEFAULTS = {
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "repeat_last_n": 64,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "logit_bias": (),
    "watermark_key": None,
    "watermark_gamma": 0.25,
    "watermark_delta": 2.0,
    "negative_prompt": None,
    "guidance_scale": 1.5,
    "contrast_beta": None,
    "contrast_alpha": 0.1,
}


class Sampler:
    """Candidates are ordered by (probability descending, token id ascending), a total order, so ties never
    depend on sort stability or hardware.

    Logit bias and penalties change the float32 logits first, each token on its own and in this order (so the
    order tokens are counted in does not matter): ``x + bias``; for the repetition penalty ``r``, ``x / r`` when
    ``x > 0`` else ``x * r``; then ``x - (count * frequency_penalty + presence_penalty)``, the penalty computed in
    double and rounded to float; last, with a watermark key, ``x + delta`` for the tokens green after the previous
    token (:mod:`etalii_dllm.watermark`). ``prompt`` is the context the output continues: the repetition penalty
    and the watermark look at it, frequency and presence count only tokens passed to :meth:`accept`."""

    def __init__(self, options: SamplingOptions, prompt: Iterable[int] = ()) -> None:
        self._options = options
        self._random = DeterministicRandom(options.seed & 0xFFFFFFFFFFFFFFFF)
        self._history: list[int] = list(prompt) if options.repetition_penalty != 1.0 else []
        self._counts: dict[int, int] = {}
        prompt = self._history if options.repetition_penalty != 1.0 else list(prompt)
        self._previous = prompt[-1] if prompt else -1
        self._watermark = None if options.watermark_key is None else watermark.key_hash(options.watermark_key)

    def accept(self, token: int) -> None:
        """Records a generated token for the penalties."""
        options = self._options
        if options.repetition_penalty != 1.0:
            self._history.append(token)
        if options.frequency_penalty != 0.0 or options.presence_penalty != 0.0:
            self._counts[token] = self._counts.get(token, 0) + 1
        self._previous = token

    def adjust(self, logits: FloatArray) -> FloatArray:
        """The logits after logit bias and penalties (a new array; the input is not changed)."""
        options = self._options
        values = np.array(logits, dtype=np.float32)
        size = len(values)
        if options.logit_bias:
            ids = np.array([t for t, _ in options.logit_bias if t < size], dtype=np.int64)
            biases = np.array([b for t, b in options.logit_bias if t < size], dtype=np.float32)
            values[ids] = values[ids] + biases
        if options.repetition_penalty != 1.0 and options.repeat_last_n != 0:
            window = self._history if options.repeat_last_n < 0 else self._history[-options.repeat_last_n :]
            ids = np.array(sorted({t for t in window if 0 <= t < size}), dtype=np.int64)
            if len(ids):
                penalty = np.float32(options.repetition_penalty)
                chosen = values[ids]
                values[ids] = np.where(chosen > 0, chosen / penalty, chosen * penalty).astype(np.float32)
        if self._counts:
            ids = np.array(sorted(t for t in self._counts if 0 <= t < size), dtype=np.int64)
            penalties = np.array(
                [self._counts[t] * options.frequency_penalty + options.presence_penalty for t in ids.tolist()],
                dtype=np.float64,
            ).astype(np.float32)
            values[ids] = values[ids] - penalties
        if self._watermark is not None:
            green = watermark.green_mask(self._watermark, self._previous, size, options.watermark_gamma)
            values[green] = values[green] + np.float32(options.watermark_delta)
        return values

    @property
    def greedy(self) -> bool:
        return self._options.temperature == 0

    def sample(self, logits: FloatArray, allowed: Sequence[int] | None = None) -> int:
        """Draws the next token. ``allowed`` (ascending ids) restricts the choice, as constrained decoding does:
        the distribution, top-k and top-p are then computed over those tokens only."""
        if self._options.adjusts_logits:
            logits = self.adjust(logits)
        if allowed is not None:
            ids = np.asarray(allowed, dtype=np.int64)
            subset = np.ascontiguousarray(np.asarray(logits, dtype=np.float32)[ids])
            return int(ids[self._sample(subset)])
        return self._sample(logits)

    def _sample(self, logits: FloatArray) -> int:
        options = self._options
        if options.temperature == 0:
            return argmax(logits)

        scaled = (np.asarray(logits, dtype=np.float32) / np.float32(options.temperature)).astype(np.float32)
        probabilities = softmax(scaled).tolist()
        order = sorted(range(len(probabilities)), key=lambda i: (-probabilities[i], i))

        keep = len(order)
        if options.top_k > 0:
            keep = min(keep, options.top_k)
        if options.top_p < 1:
            cumulative = 0.0
            for i in range(keep):
                cumulative += probabilities[order[i]]
                if cumulative >= options.top_p:
                    keep = i + 1
                    break
        if options.min_p > 0:
            threshold = options.min_p * probabilities[order[0]]
            for i in range(1, keep):
                if probabilities[order[i]] < threshold:
                    keep = i
                    break

        total = 0.0
        for i in range(keep):
            total += probabilities[order[i]]
        target = self._random.next_double() * total
        running = 0.0
        for i in range(keep):
            running += probabilities[order[i]]
            if target < running:
                return order[i]
        return order[keep - 1]
