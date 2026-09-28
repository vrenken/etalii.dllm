"""Deterministic numerics. The implementations live in the C++ extension ``etalii_dllm._kernels``.

Everything here has a fixed evaluation order: on the same hardware the same inputs always give the same bits.
Inference code must use these instead of ``random``, ``numpy.random``, ``math.exp`` or numpy/BLAS reductions,
whose order may depend on array size, thread count or library version.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from etalii_dllm import _kernels
from etalii_dllm.tensor import Tensor

DeterministicRandom = _kernels.Random
exp = _kernels.exp
log = _kernels.log
sin = _kernels.sin
cos = _kernels.cos
tanh = _kernels.tanh
sigmoid = _kernels.sigmoid
erf = _kernels.erf
erfc = _kernels.erfc
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


def _float32(values: npt.ArrayLike | Tensor) -> FloatArray:
    return np.ascontiguousarray(values.numpy() if isinstance(values, Tensor) else values, dtype=np.float32)


def linear(
    x: npt.ArrayLike | Tensor, weight: npt.ArrayLike | Tensor, bias: npt.ArrayLike | Tensor | None = None
) -> Tensor:
    """``x[..., in] @ weight[out, in]^T + bias``. Each output is summed over ``in`` ascending in double."""
    return Tensor(_kernels.linear(_float32(x), _float32(weight), None if bias is None else _float32(bias)))


def matmul(a: npt.ArrayLike | Tensor, b: npt.ArrayLike | Tensor) -> Tensor:
    """``a[m, k] @ b[k, n]``. Each output equals ``dot(a[i, :], b[:, j])`` bit for bit."""
    return Tensor(_kernels.matmul(_float32(a), _float32(b)))


def rms_norm(
    x: npt.ArrayLike | Tensor,
    weight: npt.ArrayLike | Tensor | None = None,
    eps: float = 1e-6,
    *,
    add_unit_offset: bool = False,
) -> Tensor:
    """RMSNorm over the last dimension; ``add_unit_offset`` scales by ``1 + weight`` (Gemma)."""
    w = None if weight is None else _float32(weight)
    return Tensor(_kernels.rms_norm(_float32(x), w, eps, add_unit_offset))


def silu(x: npt.ArrayLike | Tensor) -> Tensor:
    """Elementwise ``x * sigmoid(x)``."""
    return Tensor(_kernels.silu(_float32(x)))


def gelu(x: npt.ArrayLike | Tensor, *, approximate: str = "none") -> Tensor:
    """Elementwise GELU; ``approximate="tanh"`` gives the tanh form used by GPT-2 and Gemma."""
    if approximate == "none":
        return Tensor(_kernels.gelu(_float32(x)))
    if approximate == "tanh":
        return Tensor(_kernels.gelu_tanh(_float32(x)))
    raise ValueError(f"unknown GELU approximation {approximate!r}")


def rope_inv_freq(
    head_dim: int,
    theta: float = 10000.0,
    *,
    rotary_dim: int | None = None,
    scaling: Mapping[str, Any] | None = None,
) -> npt.NDArray[np.float64]:
    """Rotary inverse frequencies ``theta^(-2i / rotary_dim)`` in double, from dllm ``exp``/``log``.

    ``scaling`` takes a Hugging Face ``rope_scaling`` dict: ``rope_type`` (or ``type``) ``"default"``, ``"linear"``
    (positions divided by ``factor``) or ``"llama3"`` (``factor``, ``low_freq_factor``, ``high_freq_factor``,
    ``original_max_position_embeddings``).
    """
    dim = head_dim if rotary_dim is None else rotary_dim
    if dim <= 0 or dim % 2 or dim > head_dim:
        raise ValueError("rotary_dim must be even, positive and at most head_dim")
    log_theta = log(float(theta))
    freqs = [exp(-(2 * i / dim) * log_theta) for i in range(dim // 2)]

    kind = "default" if not scaling else str(scaling.get("rope_type", scaling.get("type", "default")))
    if kind == "linear":
        factor = float(scaling["factor"])  # type: ignore[index]
        freqs = [f / factor for f in freqs]
    elif kind == "llama3":
        assert scaling is not None
        factor = float(scaling["factor"])
        low = float(scaling.get("low_freq_factor", 1.0))
        high = float(scaling.get("high_freq_factor", 4.0))
        original = float(scaling.get("original_max_position_embeddings", 8192))
        low_freq_wavelen = original / low
        high_freq_wavelen = original / high
        scaled = []
        for f in freqs:
            wavelen = 2 * math.pi / f
            if wavelen < high_freq_wavelen:
                scaled.append(f)
            elif wavelen > low_freq_wavelen:
                scaled.append(f / factor)
            else:
                smooth = (original / wavelen - low) / (high - low)
                scaled.append((1 - smooth) * f / factor + smooth * f)
        freqs = scaled
    elif kind != "default":
        raise ValueError(f"unsupported rope scaling {kind!r}")
    return np.array(freqs, dtype=np.float64)


def rope(
    x: npt.ArrayLike | Tensor,
    positions: npt.ArrayLike,
    inv_freq: npt.ArrayLike,
    *,
    interleaved: bool = False,
    inverse: bool = False,
) -> Tensor:
    """Rotary position embedding of ``x[tokens, heads, head_dim]`` at absolute ``positions``.

    ``interleaved=False`` rotates pairs ``(i, i + rotary_dim/2)`` (Hugging Face layout), ``True`` pairs
    ``(2i, 2i+1)`` (Meta/GGUF layout). Dimensions beyond ``2 * len(inv_freq)`` pass through. ``inverse`` rotates
    by the negated angles: the transpose of the rotation, and so the gradient of ``rope``.
    """
    pos = np.ascontiguousarray(positions, dtype=np.int64)
    freqs = np.ascontiguousarray(inv_freq, dtype=np.float64)
    return Tensor(_kernels.rope(_float32(x), pos, freqs, interleaved, inverse))


def attention(
    q: npt.ArrayLike | Tensor,
    k: npt.ArrayLike | Tensor,
    v: npt.ArrayLike | Tensor,
    *,
    scale: float | None = None,
    causal: bool = True,
    q_offset: int | None = None,
) -> Tensor:
    """Scaled dot-product attention over ``[length, heads, dim]`` tensors with grouped-query heads.

    ``scale`` defaults to ``1 / sqrt(head_dim)``. With ``causal``, query ``t`` sits at position ``q_offset + t``
    (default ``kv_len - q_len``, i.e. the queries are the last tokens of the KV cache) and sees keys up to it.
    Each output row is computed independently in a fixed order, so prefill and incremental decoding agree bit for bit.
    """
    qa, ka, va = _float32(q), _float32(k), _float32(v)
    if qa.ndim != 3:
        raise ValueError("q must be [length, heads, head_dim]")
    if q_offset is not None and q_offset < 0:
        raise ValueError("q_offset must be non-negative")
    s = 1.0 / math.sqrt(qa.shape[2]) if scale is None else float(scale)
    return Tensor(_kernels.attention(qa, ka, va, s, causal, -1 if q_offset is None else q_offset))


# Gradients. Each has the evaluation order documented in cpp/include/dllm/grad.hpp and docs/kernels.md.


def linear_backward(
    x: npt.ArrayLike | Tensor, weight: npt.ArrayLike | Tensor, dy: npt.ArrayLike | Tensor, *, with_bias: bool = False
) -> tuple[Tensor, Tensor, Tensor | None]:
    """``(dx, dweight, dbias)`` of :func:`linear`; ``dbias`` is ``None`` unless ``with_bias``."""
    dx, dw, db = _kernels.linear_backward(_float32(x), _float32(weight), _float32(dy), with_bias)
    return Tensor(dx), Tensor(dw), None if db is None else Tensor(db)


def rms_norm_backward(
    x: npt.ArrayLike | Tensor, weight: npt.ArrayLike | Tensor | None, dy: npt.ArrayLike | Tensor, eps: float = 1e-6
) -> tuple[Tensor, Tensor]:
    """``(dx, dweight)`` of :func:`rms_norm` (without ``add_unit_offset``)."""
    w = None if weight is None else _float32(weight)
    dx, dw = _kernels.rms_norm_backward(_float32(x), w, _float32(dy), eps)
    return Tensor(dx), Tensor(dw)


def silu_backward(x: npt.ArrayLike | Tensor, dy: npt.ArrayLike | Tensor) -> Tensor:
    """``dy * silu'(x)``, elementwise."""
    return Tensor(_kernels.silu_backward(_float32(x), _float32(dy)))


def attention_backward(
    q: npt.ArrayLike | Tensor,
    k: npt.ArrayLike | Tensor,
    v: npt.ArrayLike | Tensor,
    dout: npt.ArrayLike | Tensor,
    *,
    scale: float | None = None,
    causal: bool = True,
    q_offset: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """``(dq, dk, dv)`` of :func:`attention` with the same arguments."""
    qa = _float32(q)
    if qa.ndim != 3:
        raise ValueError("q must be [length, heads, head_dim]")
    if q_offset is not None and q_offset < 0:
        raise ValueError("q_offset must be non-negative")
    s = 1.0 / math.sqrt(qa.shape[2]) if scale is None else float(scale)
    offset = -1 if q_offset is None else q_offset
    dq, dk, dv = _kernels.attention_backward(qa, _float32(k), _float32(v), _float32(dout), s, causal, offset)
    return Tensor(dq), Tensor(dk), Tensor(dv)


def cross_entropy(
    logits: npt.ArrayLike | Tensor, targets: npt.ArrayLike, *, scale: float = 1.0
) -> tuple[float, Tensor]:
    """Softmax cross-entropy of ``logits[rows, vocab]``: the loss summed over rows (in double) and
    ``dlogits * scale``. Rows whose target is negative are ignored."""
    loss, dlogits = _kernels.cross_entropy(_float32(logits), np.ascontiguousarray(targets, dtype=np.int64), scale)
    return float(loss), Tensor(dlogits)


def embedding_backward(dy: npt.ArrayLike | Tensor, tokens: npt.ArrayLike, vocabulary_size: int) -> Tensor:
    """Gradient of ``embedding[tokens]``: ``dy`` rows summed per token id, in position order."""
    return Tensor(
        _kernels.embedding_backward(_float32(dy), np.ascontiguousarray(tokens, dtype=np.int64), vocabulary_size)
    )


def sum_squares(values: npt.ArrayLike | Tensor) -> float:
    """Sum of squares in index order with a double accumulator."""
    return float(_kernels.sum_squares(_float32(values)))


adamw_step = _kernels.adamw_step
