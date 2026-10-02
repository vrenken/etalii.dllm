"""Verifiable text watermarks (``docs/watermarks.md``): a keyed green-list bias in the sampler and exact detection.

The scheme is the soft watermark of Kirchenbauer et al. (2023), with every step defined in integers so that
generation and detection give the same bits on every machine:

- The key is a string; ``K`` is the first 8 bytes (little-endian) of ``SHA-256("dllm-watermark/1\\0" + key)``.
- ``mix`` is the SplitMix64 finalizer: ``z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9``, ``z = (z ^ (z >> 27)) *
  0x94D049BB133111EB``, ``z ^ (z >> 31)``, all modulo 2^64.
- After the token ``previous`` (the last prompt token for the first output token), the seed is ``s = mix(K + (previous
  + 1) * G)`` with ``G = 0x9E3779B97F4A7C15``, and token ``t`` is *green* when ``mix(s + (t + 1) * G) < T`` with
  ``T = floor(gamma * 2^64)``. About a fraction ``gamma`` of the vocabulary is green after any token.
- Generation adds ``delta`` to the float32 logits of the green tokens, ``x = f32(x + f32(delta))``, after logit bias
  and penalties (``etalii_dllm.sampling``). Greedy and sampled decoding both see it.
- Detection counts the green tokens of a token sequence, each judged after the token before it, and reports
  ``z = (green - gamma * n) / sqrt(n * gamma * (1 - gamma))`` in double for the ``n`` tokens scored. Unwatermarked
  text has ``z`` near 0; :data:`THRESHOLD` (4) is the default verdict line.

Detection needs only the tokenizer and the key, not the model. The text must be tokenized as it was generated, which
holds when the detector uses the generating model's tokenizer and the text was not edited.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

DOMAIN = b"dllm-watermark/1\0"
GOLDEN = 0x9E3779B97F4A7C15
_MASK = (1 << 64) - 1
DEFAULT_GAMMA = 0.25
DEFAULT_DELTA = 2.0
THRESHOLD = 4.0
"""The z-score from which :func:`detect` calls a text watermarked (a false positive rate near 3e-5 per text)."""


def key_hash(key: str) -> int:
    return int.from_bytes(hashlib.sha256(DOMAIN + key.encode("utf-8")).digest()[:8], "little")


def mix(z: int) -> int:
    z &= _MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    return z ^ (z >> 31)


def threshold(gamma: float) -> int:
    """``floor(gamma * 2^64)``, exact (``gamma * 2^64`` only changes the exponent of the double)."""
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must be between 0 and 1")
    return int(math.ldexp(gamma, 64))


def _seed(key: int, previous: int) -> int:
    return mix(key + (previous + 1) * GOLDEN)


def is_green(key: int, previous: int, token: int, gamma: float) -> bool:
    return mix(_seed(key, previous) + (token + 1) * GOLDEN) < threshold(gamma)


def _mix_array(z: np.ndarray) -> np.ndarray:
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def green_mask(key: int, previous: int, size: int, gamma: float) -> np.ndarray:
    """Whether each token ``0 .. size - 1`` is green after ``previous`` (the same values as :func:`is_green`; numpy
    uint64 arithmetic wraps modulo 2^64)."""
    seed = np.uint64(_seed(key, previous))
    with np.errstate(over="ignore"):
        tokens = np.arange(1, size + 1, dtype=np.uint64)
        values = _mix_array(seed + tokens * np.uint64(GOLDEN))
    return values < np.uint64(threshold(gamma))


@dataclass(frozen=True)
class Detection:
    tokens: int
    """Tokens scored."""
    green: int
    z: float
    watermarked: bool
    gamma: float

    def to_json(self) -> dict[str, object]:
        return {"tokens": self.tokens, "green": self.green, "z": self.z, "watermarked": self.watermarked,
                "gamma": self.gamma, "threshold": THRESHOLD}  # fmt: skip


def detect(tokens: Sequence[int], key: str, gamma: float = DEFAULT_GAMMA, previous: int | None = None) -> Detection:
    """Counts the green tokens of ``tokens`` for ``key``. ``previous`` is the token before the first one (the last
    prompt token) when known; otherwise the first token is not scored."""
    hashed = key_hash(key)
    limit = threshold(gamma)
    context = [previous, *tokens] if previous is not None else list(tokens)
    scored = len(context) - 1
    green = sum(1 for before, token in pairwise(context) if mix(_seed(hashed, before) + (token + 1) * GOLDEN) < limit)
    z = (green - gamma * scored) / math.sqrt(scored * gamma * (1.0 - gamma)) if scored > 0 else 0.0
    return Detection(scored, green, z, z >= THRESHOLD, gamma)
