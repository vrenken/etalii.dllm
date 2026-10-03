"""Deterministic token sampling."""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from etalii_dllm import watermark
from etalii_dllm.numerics import DeterministicRandom, FloatArray, argmax, exp, log, softmax

DRY_BREAKERS = ("\n", ":", '"', "*")
"""llama.cpp's default DRY sequence breakers."""
DRY_MAX_MATCH = 256
"""The longest repeat DRY measures (longer repeats count as this long)."""
LN2 = 0.6931471805599453
"""The double nearest ln 2: surprises are ``-log(p) / LN2`` bits."""
MIROSTAT_M = 100
"""How many of the most likely candidates Mirostat 1.0 fits its Zipf exponent to."""


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
    typical_p: float = 1.0
    """Locally typical sampling: keep the candidates whose surprise is closest to the entropy, up to this much
    probability; 1 disables."""
    top_n_sigma: float = 0.0
    """Keep only tokens whose logit is within this many standard deviations of the top logit; 0 disables."""
    xtc_probability: float = 0.0
    """XTC (exclude top choices): the chance per token of dropping all but the least likely of the candidates at
    least ``xtc_threshold`` likely; 0 disables."""
    xtc_threshold: float = 0.1
    """How likely a candidate must be for XTC to drop it."""
    dry_multiplier: float = 0.0
    """DRY: the penalty for a token that would extend a repeat of ``dry_allowed_length`` tokens; 0 disables."""
    dry_base: float = 1.75
    """DRY: each token a repeat is longer multiplies the penalty by this."""
    dry_allowed_length: int = 2
    """DRY: repeats up to this long are not penalised."""
    dry_penalty_last_n: int = -1
    """DRY: how many of the latest tokens it looks at; -1 means all of them, 0 none."""
    dry_sequence_breakers: tuple[str, ...] = DRY_BREAKERS
    """DRY: a token whose text contains one of these ends a repeat."""
    mirostat: int = 0
    """Mirostat version: 1 or 2 keep the surprise near ``mirostat_tau`` (and replace top-k to XTC); 0 disables."""
    mirostat_tau: float = 5.0
    """Mirostat's target surprise, in bits."""
    mirostat_eta: float = 0.1
    """Mirostat's learning rate."""
    dynatemp_range: float = 0.0
    """Dynamic temperature: the temperature moves between ``temperature - range`` and ``temperature + range`` with
    the distribution's normalised entropy; 0 disables."""
    dynatemp_exponent: float = 1.0
    """Dynamic temperature: the normalised entropy is raised to this power."""

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
        if not 0 < self.typical_p <= 1:
            raise ValueError("typical_p must be in (0, 1]")
        if not (math.isfinite(self.top_n_sigma) and self.top_n_sigma >= 0):
            raise ValueError("top_n_sigma must be non-negative")
        if not (0.0 <= self.xtc_probability <= 1.0 and 0.0 <= self.xtc_threshold <= 1.0):
            raise ValueError("xtc_probability and xtc_threshold must be in [0, 1]")
        if not (math.isfinite(self.dry_multiplier) and self.dry_multiplier >= 0):
            raise ValueError("dry_multiplier must be non-negative")
        if not (math.isfinite(self.dry_base) and self.dry_base >= 1):
            raise ValueError("dry_base must be at least 1")
        if self.dry_allowed_length < 1:
            raise ValueError("dry_allowed_length must be at least 1")
        if self.dry_penalty_last_n < -1:
            raise ValueError("dry_penalty_last_n must be -1 or more")
        if not all(isinstance(b, str) and b for b in self.dry_sequence_breakers):
            raise ValueError("dry_sequence_breakers must be non-empty strings")
        if self.mirostat not in (0, 1, 2):
            raise ValueError("mirostat must be 0, 1 or 2")
        for name in ("mirostat_tau", "mirostat_eta", "dynatemp_range", "dynatemp_exponent"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0):
                raise ValueError(f"{name} must be non-negative")

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
        return bool(self.logit_bias) or self.penalises or self.watermark_key is not None or self.dry

    @property
    def penalises(self) -> bool:
        return (
            (self.repetition_penalty != 1.0 and self.repeat_last_n != 0)
            or self.frequency_penalty != 0.0
            or self.presence_penalty != 0.0
        )

    @property
    def dry(self) -> bool:
        """Whether the DRY penalty is on."""
        return self.dry_multiplier != 0.0 and self.dry_penalty_last_n != 0

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
                if name == "logit_bias":
                    value = [[t, b] for t, b in value]
                elif name == "dry_sequence_breakers":
                    value = list(value)
                record[name] = value
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> SamplingOptions:
        """The inverse of :meth:`record`."""
        values = dict(record)
        if "logit_bias" in values:
            values["logit_bias"] = tuple((int(t), float(b)) for t, b in values["logit_bias"])  # type: ignore[union-attr]
        if "dry_sequence_breakers" in values:
            values["dry_sequence_breakers"] = tuple(values["dry_sequence_breakers"])  # type: ignore[arg-type]
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
    "typical_p": 1.0,
    "top_n_sigma": 0.0,
    "xtc_probability": 0.0,
    "xtc_threshold": 0.1,
    "dry_multiplier": 0.0,
    "dry_base": 1.75,
    "dry_allowed_length": 2,
    "dry_penalty_last_n": -1,
    "dry_sequence_breakers": DRY_BREAKERS,
    "mirostat": 0,
    "mirostat_tau": 5.0,
    "mirostat_eta": 0.1,
    "dynatemp_range": 0.0,
    "dynatemp_exponent": 1.0,
}

SAMPLER_FIELDS = (
    "typical_p",
    "top_n_sigma",
    "xtc_probability",
    "xtc_threshold",
    "dry_multiplier",
    "dry_base",
    "dry_allowed_length",
    "dry_penalty_last_n",
    "dry_sequence_breakers",
    "mirostat",
    "mirostat_tau",
    "mirostat_eta",
    "dynatemp_range",
    "dynatemp_exponent",
)
"""The Phase 49 and 50 sampler controls, named as in llama.cpp's server and every API."""


def sampler_fields(source: object) -> dict[str, object]:
    """The Phase 49 and 50 controls an API request or option object sets (attributes that are not ``None``)."""
    values: dict[str, object] = {}
    for name in SAMPLER_FIELDS:
        value = getattr(source, name, None)
        if value is not None:
            values[name] = tuple(value) if name == "dry_sequence_breakers" else value
    return values


def dry_breakers(token_bytes: Sequence[bytes], breakers: Iterable[str]) -> frozenset[int]:
    """The DRY breaker tokens: those whose bytes contain the UTF-8 bytes of a breaker string."""
    needles = [b.encode("utf-8") for b in breakers]
    return frozenset(t for t, data in enumerate(token_bytes) if any(n in data for n in needles))


def dry_penalties(
    window: Sequence[int], breakers: Collection[int], multiplier: float, base: float, allowed: int
) -> dict[int, float]:
    """The DRY penalty of each token that would extend a repeat in ``window`` (prompt and output, oldest first).

    For every earlier position ``i`` whose preceding tokens match the window's last tokens, the match length is
    the number of tokens matched backwards (none of them a breaker, at most :data:`DRY_MAX_MATCH`); the token
    ``window[i]`` that followed gets the longest such length ``n`` over all ``i``. Tokens with ``n >= allowed`` get
    ``multiplier * base^(n - allowed)``, the power by repeated multiplication in double."""
    size = len(window)
    if size < 2 or window[-1] in breakers:
        return {}
    last = window[-1]
    longest: dict[int, int] = {}
    for i in range(1, size):
        if window[i - 1] != last:
            continue
        length = 0
        while (
            length < DRY_MAX_MATCH
            and length < i
            and window[i - 1 - length] == window[size - 1 - length]
            and window[i - 1 - length] not in breakers
        ):
            length += 1
        if length >= allowed:
            token = window[i]
            if length > longest.get(token, 0):
                longest[token] = length
    penalties: dict[int, float] = {}
    for token, length in longest.items():
        power = 1.0
        for _ in range(length - allowed):
            power *= base
        penalties[token] = multiplier * power
    return penalties


class Sampler:
    """Candidates are ordered by (probability descending, token id ascending), a total order, so ties never
    depend on sort stability or hardware.

    Logit bias and penalties change the float32 logits first, each token on its own and in this order (so the
    order tokens are counted in does not matter): ``x + bias``; for the repetition penalty ``r``, ``x / r`` when
    ``x > 0`` else ``x * r``; then ``x - (count * frequency_penalty + presence_penalty)``, the penalty computed in
    double and rounded to float; then ``x - dry`` for the DRY penalty (:func:`dry_penalties`); last, with a
    watermark key, ``x + delta`` for the tokens green after the previous token (:mod:`etalii_dllm.watermark`).
    ``prompt`` is the context the output continues: the repetition penalty, DRY and the watermark look at it,
    frequency and presence count only tokens passed to :meth:`accept`. ``breakers`` are the DRY breaker tokens
    (:func:`dry_breakers`).

    With a temperature, the candidates go through top-k, top-n-sigma, typical-p, top-p, min-p and XTC in that order
    (docs/specification.md#4-random-numbers-and-sampling), then one draw picks the token."""

    def __init__(self, options: SamplingOptions, prompt: Iterable[int] = (), breakers: Collection[int] = ()) -> None:
        self._options = options
        self._random = DeterministicRandom(options.seed & 0xFFFFFFFFFFFFFFFF)
        keeps = options.repetition_penalty != 1.0 or options.dry
        self._history: list[int] = list(prompt) if keeps else []
        self._counts: dict[int, int] = {}
        prompt = self._history if keeps else list(prompt)
        self._previous = prompt[-1] if prompt else -1
        self._watermark = None if options.watermark_key is None else watermark.key_hash(options.watermark_key)
        self._breakers = frozenset(breakers)
        self.mu = 2.0 * options.mirostat_tau
        """Mirostat's running surprise limit."""

    def accept(self, token: int) -> None:
        """Records a generated token for the penalties."""
        options = self._options
        if options.repetition_penalty != 1.0 or options.dry:
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
        if options.dry:
            last = options.dry_penalty_last_n
            window = self._history if last < 0 else self._history[-last:]
            dry = dry_penalties(
                window, self._breakers, options.dry_multiplier, options.dry_base, options.dry_allowed_length
            )
            ids = np.array(sorted(t for t in dry if 0 <= t < size), dtype=np.int64)
            if len(ids):
                amounts = np.array([dry[t] for t in ids.tolist()], dtype=np.float64)
                with np.errstate(over="ignore"):
                    values[ids] = values[ids] - amounts.astype(np.float32)
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

    def _mirostat(self, order: list[int], probabilities: list[float]) -> int:
        """Mirostat: keep the first ``k`` candidates (1.0: :func:`mirostat_k`; 2.0: up to the first after the first
        whose surprise ``-log(p) / LN2`` exceeds ``mu``), draw among them, then ``mu -= eta * (surprise - tau)`` with
        the drawn token's surprise under the kept, renormalised probabilities."""
        options = self._options
        if options.mirostat == 1:
            keep = mirostat_k(probabilities, order, self.mu)
        else:
            keep = len(order)
            for i in range(1, len(order)):
                p = probabilities[order[i]]
                if p <= 0 or -log(p) / LN2 > self.mu:
                    keep = i
                    break
        total = 0.0
        for i in range(keep):
            total += probabilities[order[i]]
        target = self._random.next_double() * total
        running = 0.0
        chosen = keep - 1
        for i in range(keep):
            running += probabilities[order[i]]
            if target < running:
                chosen = i
                break
        token = order[chosen]
        surprise = -log(probabilities[token] / total) / LN2
        self.mu = self.mu - options.mirostat_eta * (surprise - options.mirostat_tau)
        return token

    def _sample(self, logits: FloatArray) -> int:
        options = self._options
        if options.temperature == 0:
            return argmax(logits)

        temperature = options.temperature
        if options.dynatemp_range > 0:
            temperature = dynamic_temperature(logits, temperature, options.dynatemp_range, options.dynatemp_exponent)
            if np.float32(temperature) == 0:
                return argmax(logits)
        scaled = (np.asarray(logits, dtype=np.float32) / np.float32(temperature)).astype(np.float32)
        probabilities = softmax(scaled).tolist()
        order = sorted(range(len(probabilities)), key=lambda i: (-probabilities[i], i))
        if options.mirostat:
            return self._mirostat(order, probabilities)

        keep = len(order)
        if options.top_k > 0:
            keep = min(keep, options.top_k)
        if options.top_n_sigma > 0:
            keep = min(keep, sigma_count(np.asarray(logits, dtype=np.float32).tolist(), options.top_n_sigma))
        kept = order[:keep]
        if options.typical_p < 1:
            kept = typical(kept, probabilities, options.typical_p)
            keep = len(kept)
        if options.top_p < 1:
            cumulative = 0.0
            for i in range(keep):
                cumulative += probabilities[kept[i]]
                if cumulative >= options.top_p:
                    keep = i + 1
                    break
        if options.min_p > 0:
            threshold = options.min_p * probabilities[kept[0]]
            for i in range(1, keep):
                if probabilities[kept[i]] < threshold:
                    keep = i
                    break

        total = 0.0
        for i in range(keep):
            total += probabilities[kept[i]]
        start = 0
        if options.xtc_probability > 0 and self._random.next_double() < options.xtc_probability:
            floor = options.xtc_threshold * total
            above = 0
            while above < keep and probabilities[kept[above]] >= floor:
                above += 1
            if above >= 2:
                start = above - 1
                total = 0.0
                for i in range(start, keep):
                    total += probabilities[kept[i]]
        target = self._random.next_double() * total
        running = 0.0
        for i in range(start, keep):
            running += probabilities[kept[i]]
            if target < running:
                return kept[i]
        return kept[keep - 1]


def power(a: float, b: float) -> float:
    """``a^b`` for ``a >= 0`` from the portable ``exp`` and ``log``: 1 when ``b`` is 0, 0 when ``a`` is 0, else
    ``exp(b * log(a))``."""
    if b == 0:
        return 1.0
    if a == 0:
        return 0.0
    return exp(b * log(a))


def dynamic_temperature(logits: FloatArray, temperature: float, spread: float, exponent: float) -> float:
    """Entropy-based dynamic temperature: with ``p = softmax(logits)``, ``H = -sum p log p`` (token order, double,
    ``p > 0`` only) and ``n`` tokens, the temperature is ``low + (high - low) * (H / log(n))^exponent`` for ``low =
    max(0, temperature - spread)`` and ``high = temperature + spread`` (``temperature`` itself for one token)."""
    probabilities = softmax(np.asarray(logits, dtype=np.float32)).tolist()
    if len(probabilities) < 2:
        return temperature
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * log(p)
    low, high = max(0.0, temperature - spread), temperature + spread
    return low + (high - low) * power(entropy / log(float(len(probabilities))), exponent)


def mirostat_k(probabilities: Sequence[float], order: Sequence[int], mu: float) -> int:
    """Mirostat 1.0's candidate count: the Zipf exponent ``s`` fitted to the first :data:`MIROSTAT_M` candidates
    (``t = log((i + 2) / (i + 1))``, ``b = log(p_i / p_i+1)``, ``s = sum(t b) / sum(t t)`` over the pairs with both
    probabilities above 0), ``e = s - 1`` and ``k = ((e 2^mu) / (1 - n^-e))^(1 / s)`` truncated, between 1 and ``n``;
    ``n`` when the fit or the power is undefined."""
    n = len(order)
    products = squares = 0.0
    for i in range(min(MIROSTAT_M, n) - 1):
        first, second = probabilities[order[i]], probabilities[order[i + 1]]
        if first > 0 and second > 0:
            t = log((i + 2) / (i + 1))
            products += t * log(first / second)
            squares += t * t
    if squares == 0:
        return n
    s = products / squares
    e = s - 1
    denominator = 1 - power(float(n), -e) if e != 0 else 0.0
    if s <= 0 or denominator == 0:
        return n
    ratio = e * power(2.0, mu) / denominator
    if not (ratio > 0 and math.isfinite(ratio)):
        return n
    k = power(ratio, 1 / s)
    if not math.isfinite(k) or k >= n:
        return n
    return max(1, int(k))


def sigma_count(logits: Sequence[float], n: float) -> int:
    """Top-n-sigma: how many tokens have a logit of at least ``max - n * sigma``, where the mean and the standard
    deviation of the finite logits are computed in double, in token order (a population deviation)."""
    finite = [x for x in logits if math.isfinite(x)]
    if not finite:
        return len(logits)
    mean = 0.0
    for x in finite:
        mean += x
    mean /= len(finite)
    variance = 0.0
    for x in finite:
        variance += (x - mean) * (x - mean)
    variance /= len(finite)
    threshold = max(finite) - n * math.sqrt(variance)
    return max(1, sum(1 for x in finite if x >= threshold))


def typical(kept: list[int], probabilities: Sequence[float], mass: float) -> list[int]:
    """Locally typical sampling over the candidates ``kept`` (in probability order): with ``q`` their probabilities
    renormalised and ``H = -sum q log q`` (in order, in double, with the portable ``log``), the candidates sorted by
    ``|-log q - H|`` (then by their place in ``kept``) up to the shortest prefix whose ``q`` reaches ``mass``, back in
    probability order."""
    total = 0.0
    for token in kept:
        total += probabilities[token]
    q = [probabilities[token] / total for token in kept]
    logs = [log(v) if v > 0 else -math.inf for v in q]
    entropy = 0.0
    for v, lv in zip(q, logs, strict=True):
        if v > 0:
            entropy -= v * lv
    scores = [abs(-lv - entropy) for lv in logs]
    ranked = sorted(range(len(kept)), key=lambda i: (scores[i], i))
    chosen: list[int] = []
    cumulative = 0.0
    for i in ranked:
        chosen.append(i)
        cumulative += q[i]
        if cumulative >= mass:
            break
    return [kept[i] for i in sorted(chosen)]
