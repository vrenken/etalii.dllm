import math

import numpy as np
import pytest
from golden_values import RANDOM_FIRST, RANDOM_SECOND

from etalii_dllm import numerics
from etalii_dllm.tensor import Tensor

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


def test_log_softmax_matches_the_log_of_softmax():
    logits = numerics.fill_gaussian(5, 1000) * np.float32(4.0)
    values = numerics.log_softmax(logits)
    reference = np.log(np.exp(logits.astype(np.float64) - logits.max()) / np.exp(logits.astype(np.float64)
                                                                                  - logits.max()).sum())  # fmt: skip
    assert values.dtype == np.float32
    np.testing.assert_allclose(values, reference, rtol=0, atol=2e-6)
    assert numerics.argmax(values) == numerics.argmax(logits)
    assert numerics.fingerprint(numerics.log_softmax(logits)) == numerics.fingerprint(values)


def test_argmax_breaks_ties_towards_lowest_index():
    assert numerics.argmax([0.0, 5.0, 5.0, 1.0]) == 1


def test_dot_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        numerics.dot([1.0, 2.0], [1.0])


def gaussian(seed: int, *shape: int) -> np.ndarray:
    return numerics.fill_gaussian(seed, math.prod(shape)).reshape(shape)


def test_packed_weight_keeps_its_shape_and_rejects_other_ranks():
    w = gaussian(1, 20, 12)
    packed = numerics.PackedWeight(Tensor(w))
    assert packed.shape == (20, 12)
    x = gaussian(2, 3, 12)
    reference = x.astype(np.float64) @ w.astype(np.float64).T
    np.testing.assert_allclose(numerics.linear(x, packed).numpy(), reference, rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError, match=r"weight must be \[out_features, in_features\]"):
        numerics.PackedWeight(gaussian(3, 2, 3, 4))


def reference_q8_0(w: np.ndarray) -> np.ndarray:
    """The documented Q8_0 round trip in float32: ``d = amax / 127``, ``q = rint(x * (1 / d))``, ``q * d``."""
    blocks = w.reshape(w.shape[0], -1, 32)
    d = (np.abs(blocks).max(axis=2, keepdims=True) / np.float32(127)).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(d == 0, 0, np.clip(np.rint(blocks * (np.float32(1) / d)), -127, 127))
    return (q.astype(np.float32) * d).reshape(w.shape)


def test_quantized_weight_dequantizes_to_the_documented_round_trip():
    w = gaussian(4, 6, 64)
    w[2, 32:] = 0  # an all-zero block has scale 0
    quantized = numerics.QuantizedWeight(w)
    assert quantized.shape == (6, 64) and quantized.kind == "q8_0"
    dequantized = quantized.dequantize()
    assert dequantized.dtype == np.float32 and dequantized.shape == (6, 64)
    np.testing.assert_array_equal(dequantized, reference_q8_0(w))
    assert not dequantized[2, 32:].any()
    assert np.abs(dequantized - w).max() <= np.abs(w).max() / 127 / 2 * 1.0001


def test_quantized_weight_support_check():
    assert numerics.QuantizedWeight.supports(Tensor(gaussian(5, 2, 64)))
    assert not numerics.QuantizedWeight.supports(gaussian(5, 2, 48))
    assert not numerics.QuantizedWeight.supports(gaussian(5, 64))


def test_thread_count_is_validated_and_reported():
    before = numerics.threads()
    try:
        numerics.set_threads(3)
        assert numerics.threads() == 3
        with pytest.raises(ValueError, match="thread count must not be negative"):
            numerics.set_threads(-1)
        assert numerics.threads() == 3
    finally:
        numerics.set_threads(before)
    assert numerics.instruction_set() in ("avx2", "sse2", "neon", "portable")


def test_matmul_matches_float64():
    a, b = gaussian(6, 5, 70), gaussian(7, 70, 9)
    np.testing.assert_allclose(numerics.matmul(a, b).numpy(), a.astype(np.float64) @ b, rtol=1e-6, atol=1e-6)
    assert numerics.matmul(a, b).numpy()[2, 3] == np.float32(numerics.dot(a[2], np.ascontiguousarray(b[:, 3])))


def test_rope_inv_freq_rejects_bad_rotary_dims_and_scalings():
    for head_dim, rotary_dim in ((8, 0), (8, 3), (8, 10), (7, None)):
        with pytest.raises(ValueError, match="rotary_dim must be even, positive and at most head_dim"):
            numerics.rope_inv_freq(head_dim, rotary_dim=rotary_dim)
    with pytest.raises(ValueError, match="unsupported rope scaling 'yarn'"):
        numerics.rope_inv_freq(8, scaling={"rope_type": "yarn", "factor": 2.0})
    freqs = numerics.rope_inv_freq(16, 10000.0, rotary_dim=8)
    np.testing.assert_allclose(freqs, 10000.0 ** (-np.arange(0, 8, 2) / 8), rtol=1e-14)
    assert np.array_equal(numerics.rope_inv_freq(8, scaling={"rope_type": "default"}), numerics.rope_inv_freq(8))
    longrope = {"rope_type": "longrope", "short_factor": [2.0, 4.0], "long_factor": [8.0, 8.0]}
    np.testing.assert_array_equal(
        numerics.rope_inv_freq(8, rotary_dim=4, scaling=longrope), numerics.rope_inv_freq(8, rotary_dim=4) / [2, 4]
    )
    with pytest.raises(ValueError, match="longrope short_factor needs 4 values, got 2"):
        numerics.rope_inv_freq(8, scaling=longrope)


def test_config_rejects_bad_rotary_dims_and_longrope_with_qk_norm():
    from dataclasses import replace

    from etalii_dllm.architecture import TransformerConfig
    from etalii_dllm.transformer import Transformer

    base = TransformerConfig.from_dict(
        {
            "family": "phi3",
            "vocabulary_size": 8,
            "hidden_size": 8,
            "intermediate_size": 8,
            "layers": 1,
            "heads": 1,
            "kv_heads": 1,
            "head_dim": 8,
            "context_length": 16,
            "rms_norm_eps": 1e-5,
            "rope_theta": 1e4,
        }
    )
    for rotary_dim in (0, 3, 10):
        with pytest.raises(ValueError, match="rotary_dim must be even"):
            replace(base, rotary_dim=rotary_dim)
    longrope = {"rope_type": "longrope", "short_factor": [1.0] * 4, "attention_factor": 2.0}
    with pytest.raises(ValueError, match="longrope short_factor needs 2 values"):
        replace(base, rotary_dim=4, rope_scaling=longrope)
    config = replace(base, rope_scaling=longrope, qk_norm=True)
    tensors = {name: np.ones(shape, dtype=np.float32) for name, shape in config.tensor_shapes().items()}
    with pytest.raises(ValueError, match="LongRoPE attention factor together with QK-norm"):
        Transformer(config, tensors)


def test_unknown_devices_and_approximations_are_rejected():
    x = gaussian(8, 2, 4)
    with pytest.raises(ValueError, match="unknown device 'gpu'; supported: cpu, cuda"):
        numerics.silu(x, device="gpu")
    with pytest.raises(ValueError, match="unknown GELU approximation 'erf'"):
        numerics.gelu(x, approximate="erf")


def test_attention_validates_its_arguments():
    q = gaussian(9, 2, 2, 4)
    with pytest.raises(ValueError, match=r"q must be \[length, heads, head_dim\]"):
        numerics.attention(q[0], q, q)
    with pytest.raises(ValueError, match="q_offset must be non-negative"):
        numerics.attention(q, q, q, q_offset=-1)
    dout = gaussian(10, 2, 2, 4)
    with pytest.raises(ValueError, match=r"q must be \[length, heads, head_dim\]"):
        numerics.attention_backward(q[0], q, q, dout)
    with pytest.raises(ValueError, match="q_offset must be non-negative"):
        numerics.attention_backward(q, q, q, dout, q_offset=-2)


def test_gpu_shape_checks_run_before_any_device_work():
    """The GPU paths validate shapes themselves (the CPU kernels do it in C++), so a bad call fails the same way
    with or without a GPU."""
    q, k = gaussian(11, 3, 2, 4), gaussian(12, 5, 2, 4)
    freqs = numerics.rope_inv_freq(4)
    rope_message = r"rope needs x\[tokens, heads, head_dim\], one position per token, 2 \* len\(inv_freq\) <= head_dim"
    for x, positions, inv_freq in ((q[0], np.arange(2), freqs), (q, np.arange(2), freqs), (q, np.arange(3), [1.0] * 3)):
        with pytest.raises(ValueError, match=rope_message):
            numerics.rope(x, positions, inv_freq, device="cuda")
    attention_message = r"q, k and v must be \[length, heads, dim\] with matching lengths and head_dim"
    for keys, values in ((k[0], k), (k, k[:4]), (gaussian(13, 5, 2, 6), gaussian(13, 5, 2, 6))):
        with pytest.raises(ValueError, match=attention_message):
            numerics.attention(q, keys, values, device="cuda")
    with pytest.raises(ValueError, match="q_offset must be non-negative"):
        numerics.attention(q, k[:2], k[:2], device="cuda")  # more queries than keys


@pytest.mark.skipif(numerics.cuda.available(), reason="checks the failure on machines without CUDA")
def test_gpu_kernels_fail_without_cuda_instead_of_falling_back():
    x, q = gaussian(14, 2, 4), gaussian(15, 3, 2, 4)
    calls = [
        lambda: numerics.rms_norm(x, np.ones(4, np.float32), device="cuda"),
        lambda: numerics.silu(x, device="cuda"),
        lambda: numerics.gelu(x, device="cuda"),
        lambda: numerics.gelu(x, approximate="tanh", device="cuda"),
        lambda: numerics.rope(q, np.arange(3), numerics.rope_inv_freq(4), device="cuda"),
        lambda: numerics.attention(q, q, q, device="cuda"),
    ]
    for call in calls:
        with pytest.raises(RuntimeError, match="CUDA"):
            call()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"qk_norm_scope": "row"}, "unsupported qk_norm_scope"),
        ({"norm_placement": "middle"}, "unsupported norm_placement"),
        ({"sliding_window": 0}, "sliding_window must be positive"),
        ({"sliding_window": 4, "sliding_window_layers": (1,)}, "sliding_window_layers must be layer indices"),
        ({"attention_softcap": 0.0}, "attention_softcap must be positive"),
        ({"logits_softcap": -1.0}, "logits_softcap must be positive"),
    ],
)
def test_config_rejects_bad_layouts_and_softcaps(change, message):
    from dataclasses import replace

    from etalii_dllm.architecture import TransformerConfig

    base = TransformerConfig.from_dict(
        {
            "family": "gemma2",
            "vocabulary_size": 8,
            "hidden_size": 8,
            "intermediate_size": 8,
            "layers": 1,
            "heads": 1,
            "kv_heads": 1,
            "head_dim": 8,
            "context_length": 16,
            "rms_norm_eps": 1e-5,
            "rope_theta": 1e4,
        }
    )
    with pytest.raises(ValueError, match=message):
        replace(base, **change)
