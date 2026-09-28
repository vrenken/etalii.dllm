"""Phase 1 kernels: accuracy against float64 references, and bit-level batch invariance."""

import math

import numpy as np
import pytest
from golden_values import KERNEL_FINGERPRINTS

from etalii_dllm import numerics
from etalii_dllm.tensor import ALIGNMENT, Tensor


def gaussian(seed: int, *shape: int) -> np.ndarray:
    return numerics.fill_gaussian(seed, math.prod(shape)).reshape(shape)


# Transcendentals ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("x", [1e-300, 5e-324, 1e-10, 0.3, 0.7071, 1.0, 1.4142, 2.0, math.e, 10.0, 1e4, 1e300])
def test_log_is_accurate(x):
    assert numerics.log(x) == pytest.approx(math.log(x), rel=1e-15, abs=1e-15)


def test_log_edge_cases():
    assert numerics.log(1.0) == 0.0
    assert numerics.log(0.0) == -math.inf
    assert numerics.log(math.inf) == math.inf
    assert math.isnan(numerics.log(-1.0))
    assert math.isnan(numerics.log(math.nan))


def test_sin_cos_are_accurate():
    xs = [i * 0.37 - 50.0 for i in range(300)] + [1e3, 12345.678, 65536.5, 1e5, 1.5e6, -2.5e5]
    for x in xs:
        assert numerics.sin(x) == pytest.approx(math.sin(x), abs=2e-16 * max(1.0, abs(x)))
        assert numerics.cos(x) == pytest.approx(math.cos(x), abs=2e-16 * max(1.0, abs(x)))
    assert numerics.sin(0.0) == 0.0
    assert numerics.cos(0.0) == 1.0
    assert math.isnan(numerics.sin(math.inf))
    assert math.isnan(numerics.cos(math.nan))


@pytest.mark.parametrize("x", [0.0, 1e-8, 0.01, 0.1249, 0.125, 0.5, -0.7, 3.0, -10.0, 21.9, 30.0])
def test_tanh_and_sigmoid_are_accurate(x):
    assert numerics.tanh(x) == pytest.approx(math.tanh(x), rel=1e-14, abs=1e-300)
    assert numerics.sigmoid(x) == pytest.approx(1.0 / (1.0 + math.exp(-x)), rel=1e-14)


@pytest.mark.parametrize("x", [0.0, 1e-6, 0.2, -0.9, 1.5, 2.4999, 2.5, 3.3, -4.0, 5.9, 6.5])
def test_erf_is_accurate(x):
    assert numerics.erf(x) == pytest.approx(math.erf(x), rel=1e-14, abs=1e-300)
    assert numerics.erfc(x) == pytest.approx(math.erfc(x), rel=1e-13, abs=1e-15)


@pytest.mark.parametrize("x", [3.0, 6.5, 10.0, 20.0, 26.5])
def test_erfc_keeps_relative_accuracy_in_the_tail(x):
    assert numerics.erfc(x) == pytest.approx(math.erfc(x), rel=1e-13)
    assert numerics.erfc(-x) == pytest.approx(math.erfc(-x), rel=1e-15)
    assert numerics.erfc(40.0) == 0.0


# Tensor ------------------------------------------------------------------------------------------------------------


def test_tensor_is_aligned_contiguous_float32():
    source = np.arange(24, dtype=np.float64).reshape(2, 3, 4)[:, ::2, :]
    tensor = Tensor(source)
    data = tensor.numpy()
    assert tensor.shape == (2, 2, 4)
    assert tensor.strides == (8, 4, 1)
    assert data.dtype == np.float32
    assert data.flags.c_contiguous
    assert data.ctypes.data % ALIGNMENT == 0
    assert not data.flags.writeable
    np.testing.assert_array_equal(data, source)


def test_tensor_views_and_equality():
    tensor = Tensor(np.arange(12, dtype=np.float32))
    assert tensor.reshape(3, 4).shape == (3, 4)
    assert tensor.reshape(3, 4)[1] == Tensor([4.0, 5.0, 6.0, 7.0])
    assert tensor != Tensor(np.arange(12, dtype=np.float32) + 1)
    assert tensor.fingerprint() != tensor.reshape(3, 4).fingerprint()
    assert Tensor.zeros((2, 5)) == Tensor(np.zeros((2, 5)))


def test_kernel_outputs_are_aligned():
    out = numerics.linear(gaussian(1, 3, 5), gaussian(2, 7, 5)).numpy()
    assert out.ctypes.data % ALIGNMENT == 0


# Matmul ------------------------------------------------------------------------------------------------------------


def test_linear_matches_float64_reference_and_dot():
    x, w, b = gaussian(1, 9, 300), gaussian(2, 70, 300), gaussian(3, 70)
    out = numerics.linear(x, w, b).numpy()
    reference = np.einsum("mk,nk->mn", x.astype(np.float64), w.astype(np.float64)) + b
    np.testing.assert_allclose(out, reference, rtol=1e-6, atol=1e-5)
    unbiased = numerics.linear(x, w).numpy()
    for i in (0, 4, 8):
        for n in (0, 33, 69):
            assert unbiased[i, n] == np.float32(numerics.dot(x[i], w[n]))


def test_linear_is_batch_invariant():
    x, w = gaussian(4, 37, 129), gaussian(5, 150, 129)
    full = numerics.linear(x, w).numpy()
    for rows in (slice(0, 1), slice(5, 6), slice(3, 20), slice(16, 37)):
        np.testing.assert_array_equal(numerics.linear(x[rows], w).numpy(), full[rows])
    # Leading dimensions are flattened into rows without changing any bits.
    batched = numerics.linear(x[:36].reshape(4, 9, 129), w).numpy()
    np.testing.assert_array_equal(batched.reshape(36, 150), full[:36])


def test_matmul_matches_reference_and_is_batch_invariant():
    a, b = gaussian(6, 21, 600), gaussian(7, 600, 130)
    out = numerics.matmul(a, b).numpy()
    np.testing.assert_allclose(out, a.astype(np.float64) @ b.astype(np.float64), rtol=1e-6, atol=1e-5)
    for i in (0, 20):
        np.testing.assert_array_equal(numerics.matmul(a[i : i + 1], b).numpy()[0], out[i])
        for j in (0, 64, 129):
            assert out[i, j] == np.float32(numerics.dot(a[i], np.ascontiguousarray(b[:, j])))


def test_matmul_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        numerics.matmul(np.zeros((2, 3), np.float32), np.zeros((4, 2), np.float32))
    with pytest.raises(ValueError):
        numerics.linear(np.zeros((2, 3), np.float32), np.zeros((4, 2), np.float32))


# RMSNorm and activations -------------------------------------------------------------------------------------------


def test_rms_norm_matches_reference_and_is_batch_invariant():
    x, w = gaussian(8, 6, 96), gaussian(9, 96)
    out = numerics.rms_norm(x, w, 1e-5).numpy()
    x64 = x.astype(np.float64)
    reference = x64 / np.sqrt((x64 * x64).mean(axis=-1, keepdims=True) + 1e-5) * w
    np.testing.assert_allclose(out, reference, rtol=1e-6)
    np.testing.assert_array_equal(numerics.rms_norm(x[2:3], w, 1e-5).numpy(), out[2:3])
    offset = numerics.rms_norm(x, w, 1e-5, add_unit_offset=True).numpy()
    np.testing.assert_allclose(offset, reference / w * (1 + w), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(numerics.rms_norm(x).numpy(), reference / w, rtol=1e-5, atol=1e-6)


def test_activations_match_reference():
    x = np.linspace(-12, 12, 2001, dtype=np.float32)
    x64 = x.astype(np.float64)
    erfc = np.vectorize(math.erfc)
    tanh = np.vectorize(math.tanh)
    silu_reference = x64 / (1 + np.vectorize(math.exp)(-x64))
    np.testing.assert_allclose(numerics.silu(x).numpy(), silu_reference, rtol=2e-7, atol=1e-30)
    np.testing.assert_allclose(numerics.gelu(x).numpy(), 0.5 * x64 * erfc(-x64 / math.sqrt(2)), rtol=2e-7, atol=1e-30)
    tanh_reference = 0.5 * x64 * (1 + tanh(math.sqrt(2 / math.pi) * (x64 + 0.044715 * x64**3)))
    np.testing.assert_allclose(numerics.gelu(x, approximate="tanh").numpy(), tanh_reference, rtol=2e-7, atol=1e-30)
    assert numerics.silu(x.reshape(1, 2001, 1)).shape == (1, 2001, 1)
    with pytest.raises(ValueError):
        numerics.gelu(x, approximate="sigmoid")


# RoPE --------------------------------------------------------------------------------------------------------------


def reference_rope(x, positions, inv_freq, interleaved):
    out = x.astype(np.float64).copy()
    half = len(inv_freq)
    for t, pos in enumerate(positions):
        for i, f in enumerate(inv_freq):
            c, s = math.cos(pos * f), math.sin(pos * f)
            i0, i1 = (2 * i, 2 * i + 1) if interleaved else (i, i + half)
            a, b = x[t, :, i0].astype(np.float64), x[t, :, i1].astype(np.float64)
            out[t, :, i0] = a * c - b * s
            out[t, :, i1] = a * s + b * c
    return out


def test_rope_inv_freq():
    freqs = numerics.rope_inv_freq(64, 500000.0)
    np.testing.assert_allclose(freqs, [500000.0 ** (-2 * i / 64) for i in range(32)], rtol=1e-14)
    np.testing.assert_allclose(
        numerics.rope_inv_freq(64, scaling={"type": "linear", "factor": 4.0}), numerics.rope_inv_freq(64) / 4, rtol=0
    )
    llama3 = {
        "rope_type": "llama3",
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    }
    plain = numerics.rope_inv_freq(128, 500000.0)
    scaled = numerics.rope_inv_freq(128, 500000.0, scaling=llama3)
    assert scaled[0] == plain[0]  # short wavelengths are left alone
    assert scaled[-1] == plain[-1] / 8  # long wavelengths are divided by the factor
    band = [i for i, f in enumerate(plain) if 2048 < 2 * math.pi / f < 8192]
    assert band  # and the band in between is interpolated
    assert all(plain[i] / 8 < scaled[i] < plain[i] for i in band)
    assert numerics.rope_inv_freq(64, rotary_dim=32).shape == (16,)
    with pytest.raises(ValueError):
        numerics.rope_inv_freq(64, scaling={"rope_type": "yarn"})


@pytest.mark.parametrize("interleaved", [False, True])
def test_rope_matches_reference(interleaved):
    x = gaussian(10, 5, 3, 16)
    positions = [0, 1, 7, 1000, 32767]
    inv_freq = numerics.rope_inv_freq(16, 10000.0, rotary_dim=12)
    out = numerics.rope(x, positions, inv_freq, interleaved=interleaved).numpy()
    np.testing.assert_allclose(out, reference_rope(x, positions, inv_freq, interleaved), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(out[:, :, 12:], x[:, :, 12:])
    np.testing.assert_array_equal(out[0], x[0])
    # A token's rotation depends only on its own position.
    single = numerics.rope(x[3:4], positions[3:4], inv_freq, interleaved=interleaved).numpy()
    np.testing.assert_array_equal(single[0], out[3])


# Attention ---------------------------------------------------------------------------------------------------------


def reference_attention(q, k, v, causal, q_offset):
    q_len, q_heads, head_dim = q.shape
    kv_len, kv_heads, value_dim = v.shape
    out = np.zeros((q_len, q_heads, value_dim))
    for t in range(q_len):
        visible = min(q_offset + t + 1, kv_len) if causal else kv_len
        for h in range(q_heads):
            kvh = h // (q_heads // kv_heads)
            scores = k[:visible, kvh].astype(np.float64) @ q[t, h].astype(np.float64) / math.sqrt(head_dim)
            p = np.exp(scores - scores.max())
            out[t, h] = (p / p.sum()) @ v[:visible, kvh].astype(np.float64)
    return out


@pytest.mark.parametrize(("q_heads", "kv_heads"), [(4, 4), (4, 2), (6, 1)])
def test_attention_matches_reference(q_heads, kv_heads):
    q, k, v = gaussian(11, 9, q_heads, 8), gaussian(12, 9, kv_heads, 8), gaussian(13, 9, kv_heads, 5)
    for causal in (True, False):
        out = numerics.attention(q, k, v, causal=causal).numpy()
        np.testing.assert_allclose(out, reference_attention(q, k, v, causal, 0), rtol=1e-5, atol=1e-6)


def test_attention_prefill_equals_incremental_decoding():
    q, k, v = gaussian(14, 12, 4, 16), gaussian(15, 12, 2, 16), gaussian(16, 12, 2, 16)
    prefill = numerics.attention(q, k, v).numpy()
    for t in range(12):
        step = numerics.attention(q[t : t + 1], k[: t + 1], v[: t + 1]).numpy()
        np.testing.assert_array_equal(step[0], prefill[t])
    chunk = numerics.attention(q[5:9], k[:9], v[:9]).numpy()
    np.testing.assert_array_equal(chunk, prefill[5:9])
    # Keys beyond a query's causal horizon (e.g. a longer, preallocated cache) do not change its bits.
    padded = numerics.attention(q[5:9], k, v, q_offset=5).numpy()
    np.testing.assert_array_equal(padded, prefill[5:9])


def test_attention_rejects_bad_head_counts():
    with pytest.raises(ValueError):
        numerics.attention(
            np.zeros((1, 3, 4), np.float32), np.zeros((1, 2, 4), np.float32), np.zeros((1, 2, 4), np.float32)
        )
    with pytest.raises(ValueError):
        numerics.attention(
            np.zeros((1, 2, 4), np.float32),
            np.zeros((1, 2, 4), np.float32),
            np.zeros((1, 2, 4), np.float32),
            q_offset=-2,
        )


# Golden values -----------------------------------------------------------------------------------------------------


def kernel_outputs() -> dict[str, Tensor]:
    x = gaussian(20, 8, 64)
    w = gaussian(21, 48, 64)
    q, k, v = gaussian(22, 8, 4, 16), gaussian(23, 8, 2, 16), gaussian(24, 8, 2, 16)
    return {
        "linear": numerics.linear(x, w, gaussian(25, 48)),
        "matmul": numerics.matmul(x, np.ascontiguousarray(w.T)),
        "rms_norm": numerics.rms_norm(x, gaussian(26, 64), 1e-6),
        "silu": numerics.silu(x),
        "gelu": numerics.gelu(x),
        "gelu_tanh": numerics.gelu(x, approximate="tanh"),
        "rope": numerics.rope(q, np.arange(8) * 97, numerics.rope_inv_freq(16, 10000.0)),
        "attention": numerics.attention(q, k, v),
    }


def test_kernel_golden_fingerprints():
    fingerprints = {name: tensor.fingerprint() for name, tensor in kernel_outputs().items()}
    assert fingerprints == KERNEL_FINGERPRINTS
