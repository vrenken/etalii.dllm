import math

import numpy as np
import pytest
from golden_values import RANDOM_FIRST, RANDOM_SECOND

from etalii_dllm import numerics

MASK = (1 << 64) - 1


def reference_xoshiro(seed: int, count: int) -> list[int]:
    """Straight transcription of the published xoshiro256** and SplitMix64 reference code."""

    def split_mix(state: int) -> tuple[int, int]:
        state = (state + 0x9E3779B97F4A7C15) & MASK
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return state, z ^ (z >> 31)

    def rotl(x: int, k: int) -> int:
        return ((x << k) | (x >> (64 - k))) & MASK

    s = []
    state = seed
    for _ in range(4):
        state, value = split_mix(state)
        s.append(value)
    out = []
    for _ in range(count):
        out.append((rotl((s[1] * 5) & MASK, 7) * 9) & MASK)
        t = (s[1] << 17) & MASK
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = rotl(s[3], 45)
    return out


def test_random_matches_reference_implementation():
    random = numerics.DeterministicRandom(12345)
    assert [random.next_u64() for _ in range(100)] == reference_xoshiro(12345, 100)


def test_random_sequence_is_stable():
    random = numerics.DeterministicRandom(42)
    assert random.next_u64() == RANDOM_FIRST
    assert random.next_u64() == RANDOM_SECOND


@pytest.mark.parametrize("x", [0.0, 1.0, -1.0, 0.5, 10.0, -20.0, 88.7, -103.9, 700.0])
def test_exp_is_accurate(x):
    assert numerics.exp(x) == pytest.approx(math.exp(x), rel=1e-14)


def test_exp_edge_cases():
    assert numerics.exp(0.0) == 1.0
    assert numerics.exp(1000.0) == math.inf
    assert numerics.exp(-1000.0) == 0.0
    assert math.isnan(numerics.exp(math.nan))


def test_softmax_sums_to_one():
    probabilities = numerics.softmax([1.0, 2.0, 3.0, -4.0, 0.5])
    assert probabilities.dtype == np.float32
    assert numerics.sum_(probabilities) == pytest.approx(1.0, abs=1e-6)
    assert numerics.argmax(probabilities) == 2


def test_argmax_breaks_ties_towards_lowest_index():
    assert numerics.argmax([0.0, 5.0, 5.0, 1.0]) == 1


def test_dot_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        numerics.dot([1.0, 2.0], [1.0])
