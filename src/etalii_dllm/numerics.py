"""Deterministic numerics. The implementations live in the C++ extension ``etalii_dllm._kernels``.

Everything here has a fixed evaluation order: on the same hardware the same inputs always give the same bits.
Inference code must use these instead of ``random``, ``numpy.random``, ``math.exp`` or numpy/BLAS reductions,
whose order may depend on array size, thread count or library version.
"""

from __future__ import annotations

import hashlib

import numpy as np
import numpy.typing as npt

from etalii_dllm import _kernels

DeterministicRandom = _kernels.Random
exp = _kernels.exp
fill_gaussian = _kernels.fill_gaussian

FloatArray = npt.NDArray[np.float32]


def _as_float32(values: npt.ArrayLike) -> FloatArray:
    return np.ascontiguousarray(values, dtype=np.float32)


def sum_(values: npt.ArrayLike) -> float:
    """Sum in index order with a double accumulator."""
    return float(_kernels.sum(_as_float32(values)))


def dot(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Dot product in index order with a double accumulator, rounded once to float32."""
    return float(_kernels.dot(_as_float32(a), _as_float32(b)))


def argmax(values: npt.ArrayLike) -> int:
    """Index of the largest value; ties resolve to the lowest index."""
    return int(_kernels.argmax(_as_float32(values)))


def softmax(logits: npt.ArrayLike) -> FloatArray:
    """Numerically stable softmax with a fixed evaluation order."""
    return _kernels.softmax(_as_float32(logits))


def fingerprint(values: npt.ArrayLike, dtype: npt.DTypeLike = np.float32) -> str:
    """SHA-256 of the exact little-endian bit patterns. Equal fingerprints mean bit-identical data."""
    array = np.ascontiguousarray(values, dtype=np.dtype(dtype).newbyteorder("<"))
    return hashlib.sha256(array.tobytes()).hexdigest()
