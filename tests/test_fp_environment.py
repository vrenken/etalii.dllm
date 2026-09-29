"""The kernels give the same bits whatever floating point environment the caller's thread is in (issue #97).

A library built with ``-ffast-math`` turns flush-to-zero on for the whole process when it loads; the kernels must
not notice. ``_set_flush_to_zero`` flips FTZ/DAZ on the calling thread to simulate that.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from golden_values import KERNEL_FINGERPRINTS
from test_kernels import kernel_outputs

from etalii_dllm import _kernels, numerics

SUBNORMAL = np.float32(1e-40)


@pytest.fixture
def flush_to_zero() -> Iterator[None]:
    if not _kernels._set_flush_to_zero(True):
        pytest.skip("no floating point environment control on this platform")
    try:
        assert not numerics.fp_environment_is_canonical()
        yield
    finally:
        _kernels._set_flush_to_zero(False)
    assert numerics.fp_environment_is_canonical()


# Built at import, in the default environment: under flush-to-zero NumPy would already turn them into zeros.
X = np.array([[1e-20, 1e-40, 1.0, 0.0]] * 8, dtype=np.float32)
W = np.array([[1e-20, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1e-39, 0.0]] * 16, dtype=np.float32)
W_T = np.ascontiguousarray(W.T)
TINY = np.full((2, 4), SUBNORMAL, dtype=np.float32)


def subnormal_outputs() -> dict[str, bytes]:
    return {
        "linear": np.asarray(numerics.linear(X, W)).tobytes(),
        "matmul": np.asarray(numerics.matmul(X, W_T)).tobytes(),
        "silu": np.asarray(numerics.silu(TINY)).tobytes(),
        "exp": np.float64(numerics.exp(-720.0)).tobytes(),
        "log": np.float64(numerics.log(5e-320)).tobytes(),
        "sum": np.float64(numerics.sum_(TINY.ravel())).tobytes(),
    }


def test_subnormal_results_survive_flush_to_zero(flush_to_zero):
    _kernels._set_flush_to_zero(False)
    expected = subnormal_outputs()
    _kernels._set_flush_to_zero(True)
    assert subnormal_outputs() == expected
    assert np.frombuffer(expected["linear"], dtype=np.uint32)[0] != 0  # a subnormal, not flushed (bits: DAZ)


def test_kernel_golden_fingerprints_under_flush_to_zero(flush_to_zero):
    fingerprints = {name: tensor.fingerprint() for name, tensor in kernel_outputs().items()}
    assert fingerprints == KERNEL_FINGERPRINTS


def test_kernel_calls_restore_the_callers_environment(flush_to_zero):
    numerics.linear(np.ones((4, 8), np.float32), np.ones((16, 8), np.float32))
    numerics.exp(1.0)
    assert not numerics.fp_environment_is_canonical()


def test_default_environment_is_canonical():
    assert numerics.fp_environment_is_canonical()
