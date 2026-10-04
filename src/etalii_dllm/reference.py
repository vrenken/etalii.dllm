"""A second, independent implementation of the numerics and the decoder (``dllm verify --reference``).

EtAlii.Dllm promises the same bits on every machine because every operation has one fixed sequence of IEEE
operations (``docs/specification.md``). This module states that sequence a second time, in Python and elementwise
NumPy only, sharing no code with the C++ kernels or :mod:`etalii_dllm.numerics`: the transcendentals, the linear
layers (float32, Q8_0 and Q4_0), RMSNorm, RoPE, attention, softmax, the activations, the random number generator,
the sampler and the forward pass of every supported architecture (decoders and the BERT encoder).

Every reduction is a Python loop over the reduced index, vectorised only across outputs that are independent of each
other, so each sum runs in the specified order by construction rather than by a library's choice. Elementwise NumPy
on float64 is IEEE arithmetic with one rounding per operation (no fused multiply-adds), and a float64 to float32
conversion rounds to nearest even, exactly like the C++ casts. The only NumPy reductions used are ``max`` (exact,
so its order is irrelevant) and integer sums of Q8_0 products (exact as well).

It is slow (a second or so per token for a 135M model) but needs nothing but NumPy, which makes it a check of the
compiled kernels, their SIMD paths and the GPU on the machine itself: ``tests/test_reference.py`` and ``dllm verify
--reference`` require its bits to equal the engine's.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from etalii_dllm.architecture import TransformerConfig

F64 = np.float64
F32 = np.float32
_MATRICES = tuple(f".{m}.weight" for m in ("q", "k", "v", "o", "gate", "up", "down"))
_INF = math.inf
ROLL_SINK = 4
"""Tokens a rolled context window keeps from its start (docs/specification.md#the-context-window)."""


def _f64(x: npt.ArrayLike) -> np.ndarray:
    return np.array(x, dtype=F64)


def _round(x: npt.ArrayLike) -> np.ndarray:
    """float64 -> float32, round to nearest even (the C++ ``static_cast<float>``)."""
    return np.asarray(x, dtype=F64).astype(F32)


def _horner(p: np.ndarray, x: np.ndarray, coefficients: Sequence[float]) -> np.ndarray:
    for c in coefficients:
        p = p * x + c
    return p


def sequential_sum(values: npt.ArrayLike) -> float:
    """``((v0 + v1) + v2) + ...`` in double, starting from 0."""
    total = 0.0
    for v in np.asarray(values, dtype=F64).reshape(-1).tolist():
        total += v
    return total


# -- transcendentals ---------------------------------------------------------------------------------------------

_LN2_HI = 6.93147180369123816490e-01
_LN2_LO = 1.90821492927058770002e-10
_INV_LN2 = 1.44269504088896338700e00
_EXP_TAIL = (*(1.0 / math.factorial(n) for n in range(12, 1, -1)), 1.0, 1.0)
"""e^r = 1 + r + r^2/2 + ... + r^13/13!: Horner from 1/13! down."""


def _power_of_two(e: np.ndarray) -> np.ndarray:
    return np.ldexp(np.ones_like(e, dtype=F64), e.astype(np.int32))


def exp(x: npt.ArrayLike) -> np.ndarray:
    """``e^x``: ``x = k ln2 + r`` with a two-part ln2, a degree-13 Taylor polynomial, then exact scaling by ``2^k``
    (split in two when ``2^k`` is not a normal double)."""
    x = _f64(x)
    with np.errstate(all="ignore"):
        inside = (x == x) & (x <= 709.78) & (x >= -745.2)
        xs = np.where(inside, x, 0.0)
        kd = xs * _INV_LN2
        k = np.where(kd >= 0, np.trunc(kd + 0.5), np.trunc(kd - 0.5))
        r = (xs - k * _LN2_HI) - k * _LN2_LO
        p = _horner(np.full_like(r, 1.0 / 6227020800.0), r, _EXP_TAIL)
        e = k.astype(np.int64)
        while (e > 1023).any():
            big = e > 1023
            p = np.where(big, p * _power_of_two(np.full_like(e, 1023)), p)
            e = np.where(big, e - 1023, e)
        while (e < -1022).any():
            small = e < -1022
            p = np.where(small, p * _power_of_two(np.full_like(e, -1022)), p)
            e = np.where(small, e + 1022, e)
        result = p * _power_of_two(e)
    return np.where(x != x, x, np.where(x > 709.78, _INF, np.where(x < -745.2, 0.0, result)))


def log(x: npt.ArrayLike) -> np.ndarray:
    """Natural logarithm: ``x = m 2^e``, ``m`` folded into ``[sqrt(1/2), sqrt(2))``, ``2 atanh((m-1)/(m+1))`` from a
    13-term odd series, plus ``e ln2`` with a two-part ln2."""
    x = _f64(x)
    sqrt_half = 7.07106781186547524401e-01
    with np.errstate(all="ignore"):
        inside = (x > 0.0) & (x < _INF)
        xs = np.where(inside, x, 1.0)
        subnormal = xs < 2.2250738585072014e-308
        xs = np.where(subnormal, xs * 18014398509481984.0, xs)  # 2^54
        e = np.where(subnormal, -54, 0).astype(np.int64)
        bits = np.array(xs, dtype=F64).view(np.uint64)
        e = e + ((bits >> np.uint64(52)) & np.uint64(0x7FF)).astype(np.int64) - 1023
        m = ((bits & np.uint64(0x000FFFFFFFFFFFFF)) | np.uint64(0x3FF0000000000000)).view(F64)
        fold = m >= 2.0 * sqrt_half
        m = np.where(fold, m * 0.5, m)
        e = np.where(fold, e + 1, e)
        s = (m - 1.0) / (m + 1.0)
        s2 = s * s
        p = _horner(np.full_like(s, 1.0 / 25.0), s2, [1.0 / n for n in range(23, 1, -2)])
        log_m = 2.0 * s + 2.0 * s * s2 * p
        ed = e.astype(F64)
        result = (ed * _LN2_HI + log_m) + ed * _LN2_LO
    result = np.where(x == _INF, _INF, result)
    result = np.where(x == 0.0, -_INF, result)
    return np.where((x != x) | (x < 0.0), math.nan, result)


def _sin_kernel(r: np.ndarray) -> np.ndarray:
    r2 = r * r
    coefficients = [1.0 / 6227020800.0, -1.0 / 39916800.0, 1.0 / 362880.0, -1.0 / 5040.0, 1.0 / 120.0, -1.0 / 6.0]
    p = _horner(np.full_like(r, -1.0 / 1307674368000.0), r2, coefficients)
    return r + r * r2 * p


def _cos_kernel(r: np.ndarray) -> np.ndarray:
    r2 = r * r
    coefficients = [-1.0 / 87178291200.0, 1.0 / 479001600.0, -1.0 / 3628800.0, 1.0 / 40320.0, -1.0 / 720.0]
    p = _horner(np.full_like(r, 1.0 / 20922789888000.0), r2, [*coefficients, 1.0 / 24.0])
    return (1.0 - 0.5 * r2) + r2 * r2 * p


def _reduce_half_pi(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``x = n pi/2 + r`` (Cody-Waite with fdlibm's three 33-bit parts of pi/2); returns ``r`` and ``n mod 4``."""
    nd = x * 6.36619772367581382433e-01
    n = np.where(np.abs(nd) < 2.0**52, np.where(nd >= 0, np.trunc(nd + 0.5), np.trunc(nd - 0.5)), nd)
    r = (((x - n * 1.57079632673412561417e00) - n * 6.07710050630396597660e-11) - n * 2.02226624871116645580e-21) - (
        n * 8.47842766036889956997e-32
    )
    return r, np.where(np.abs(n) < 2.0**62, n, 0.0).astype(np.int64) & 3  # from 2^62 on n is a multiple of 4


def _sine_or_cosine(x: npt.ArrayLike, shift: int) -> np.ndarray:
    x = _f64(x)
    finite = np.isfinite(x)
    r, quadrant = _reduce_half_pi(np.where(finite, x, 0.0))
    quadrant = (quadrant + shift) & 3
    with np.errstate(all="ignore"):
        s, c = _sin_kernel(r), _cos_kernel(r)
    result = np.select([quadrant == 0, quadrant == 1, quadrant == 2], [s, c, -s], -c)
    return np.where(finite, result, math.nan)


def sin(x: npt.ArrayLike) -> np.ndarray:
    return _sine_or_cosine(x, 0)


def cos(x: npt.ArrayLike) -> np.ndarray:
    """``cos`` takes ``sin``'s quadrant table shifted by one: cos, -sin, -cos, sin."""
    return _sine_or_cosine(x, 1)


def atan(x: npt.ArrayLike) -> np.ndarray:
    """``pi/2 - atan(1/x)`` above 1, ``pi/6 + atan((sqrt3 x - 1)/(sqrt3 + x))`` above ``tan(pi/12)``, then the Taylor
    series to ``r^35``; odd."""
    x = _f64(x)
    sqrt3 = 1.73205080756887729353
    with np.errstate(all="ignore"):
        a = np.where(x < 0, -x, x)
        invert = a > 1.0
        y = np.where(invert, 1.0 / a, a)
        shifted = y > 0.26794919243112270647
        offset = np.where(shifted, 0.52359877559829887308, 0.0)
        r = np.where(shifted, (sqrt3 * y - 1.0) / (sqrt3 + y), y)
        r2 = r * r
        coefficients = [(-1.0 if n % 2 else 1.0) / (2.0 * n + 1.0) for n in range(16, 0, -1)]
        p = _horner(np.full_like(r, -1.0 / 35.0), r2, coefficients)
        core = offset + (r + r * r2 * p)
        result = np.where(invert, 1.57079632679489661923 - core, core)
    result = np.where(x < 0, -result, result)
    return np.where(x != x, x, result)


def acos(x: npt.ArrayLike) -> np.ndarray:
    """``2 atan(sqrt((1 - x) / (1 + x)))`` on ``[-1, 1]``; ``pi`` at -1, NaN outside."""
    x = _f64(x)
    inside = (x >= -1.0) & (x <= 1.0)
    with np.errstate(all="ignore"):
        result = 2.0 * atan(np.sqrt((1.0 - x) / (1.0 + x)))
    result = np.where(x == -1.0, 3.14159265358979323846, result)
    return np.where(inside, result, math.nan)


def tanh(x: npt.ArrayLike) -> np.ndarray:
    """Taylor to ``x^15`` below 0.125, 1 above 22, else ``1 - 2 / (e^(2|x|) + 1)``; odd."""
    x = _f64(x)
    with np.errstate(all="ignore"):
        a = np.where(x < 0, -x, x)
        a2 = a * a
        coefficients = [21844.0 / 6081075.0, -1382.0 / 155925.0, 62.0 / 2835.0, -17.0 / 315.0, 2.0 / 15.0, -1.0 / 3.0]
        p = _horner(np.full_like(a, -929569.0 / 638512875.0), a2, coefficients)
        small = a + a * a2 * p
        middle = 1.0 - 2.0 / (exp(2.0 * a) + 1.0)
        t = np.where(a < 0.125, small, np.where(a > 22.0, 1.0, middle))
    t = np.where(x < 0, -t, t)
    return np.where(x != x, x, t)


def sigmoid(x: npt.ArrayLike) -> np.ndarray:
    """``1 / (1 + e^-x)`` for ``x >= 0``, else ``e^x / (1 + e^x)``."""
    x = _f64(x)
    with np.errstate(all="ignore"):
        positive = 1.0 / (1.0 + exp(-x))
        e = exp(x)
        negative = e / (1.0 + e)
    return np.where(x >= 0, positive, negative)


def sigmoid_float(x: npt.ArrayLike) -> np.ndarray:
    """``sigmoid`` of float32 ``x`` rounded once to float32 (the gate of a shared expert)."""
    return _round(sigmoid(x))


def _erfc_tail(a: np.ndarray) -> np.ndarray:
    """``erfc(a)`` for ``a >= 2.5``: ``e^(-a^2) / sqrt(pi)`` over Laplace's continued fraction, depth 80."""
    with np.errstate(all="ignore"):
        k = a.copy()
        for n in range(80, 0, -1):
            k = a + (0.5 * n) / k
        return exp(-a * a) * 5.64189583547756286948e-01 / k


def erf(x: npt.ArrayLike) -> np.ndarray:
    """60 Maclaurin terms below 2.5, ``1 - erfc`` up to 6, 1 beyond; odd."""
    x = _f64(x)
    with np.errstate(all="ignore"):
        a = np.where(x < 0, -x, x)
        a2 = a * a
        term = a.copy()
        total = a.copy()
        for n in range(1, 60):
            term = -term * a2 / n
            total = total + term / (2 * n + 1)
        series = 1.12837916709551257390e00 * total
        tail = 1.0 - _erfc_tail(np.where(a < 2.5, 2.5, a))
        result = np.where(a < 2.5, series, np.where(a > 6.0, 1.0, tail))
    result = np.where(x < 0, -result, result)
    return np.where(x != x, x, result)


def erfc(x: npt.ArrayLike) -> np.ndarray:
    """``1 - erf(x)`` below 2.5, the continued fraction up to 27.3, 0 beyond."""
    x = _f64(x)
    with np.errstate(all="ignore"):
        tail = _erfc_tail(np.where(x < 2.5, 2.5, x))
        result = np.where(x < 2.5, 1.0 - erf(x), np.where(x > 27.3, 0.0, tail))
    return np.where(x != x, x, result)


# -- kernels -----------------------------------------------------------------------------------------------------


def argmax(values: npt.ArrayLike) -> int:
    """The first index of the largest value (a later value must be strictly greater)."""
    flat = np.asarray(values).reshape(-1).tolist()
    best = 0
    for i in range(1, len(flat)):
        if flat[i] > flat[best]:
            best = i
    return best


def softmax(logits: npt.ArrayLike) -> np.ndarray:
    """``e_i * (1 / total)`` with ``e_i = exp(l_i - max)`` and ``total`` their sum over ``i`` ascending."""
    values = np.asarray(logits, dtype=F32).reshape(-1)
    scratch = exp(values.astype(F64) - float(values[argmax(values)]))
    return _round(scratch * (1.0 / sequential_sum(scratch)))


def moe_route(logits: npt.ArrayLike, k: int, normalize: bool) -> tuple[list[list[int]], np.ndarray]:
    """Per row of router logits: the ``k`` experts with the largest :func:`softmax` probabilities, larger first and
    equal ones by lower index, and their weights; with ``normalize`` each weight is divided by their total (summed
    in double in that order) and rounded once."""
    values = np.asarray(logits, dtype=F32)
    chosen: list[list[int]] = []
    weights = np.zeros((values.shape[0], k), F32)
    for r, row in enumerate(values):
        probabilities = softmax(row).tolist()
        order = sorted(range(len(probabilities)), key=lambda e: (-probabilities[e], e))[:k]
        picked = [probabilities[e] for e in order]
        if normalize:
            total = 0.0
            for p in picked:
                total += p
            picked = [p / total for p in picked]
        chosen.append(order)
        weights[r] = _round(np.array(picked, F64))
    return chosen, weights


def log_softmax(logits: npt.ArrayLike) -> np.ndarray:
    """``(l_i - max) - log(total)``."""
    values = np.asarray(logits, dtype=F32).reshape(-1)
    shifted = values.astype(F64) - float(values[argmax(values)])
    return _round(shifted - float(log(sequential_sum(exp(shifted)))))


def _rows(x: npt.ArrayLike) -> tuple[np.ndarray, tuple[int, ...]]:
    values = np.asarray(x, dtype=F32)
    return values.reshape(-1, values.shape[-1]), values.shape[:-1]


def linear(x: npt.ArrayLike, weight: npt.ArrayLike | Weight, bias: npt.ArrayLike | None = None) -> np.ndarray:
    """``out[r, n] = sum_k x[r, k] w[n, k]`` over ``k`` ascending in double, plus ``bias[n]``, rounded once."""
    w = weight if isinstance(weight, Weight) else Weight(np.asarray(weight, dtype=F32))
    rows, lead = _rows(x)
    if w.quantized is not None:
        acc = _linear_q8(rows, *w.quantized)
    else:
        xd = rows.astype(F64)
        acc = np.zeros((rows.shape[0], w.out_features), dtype=F64)
        for k in range(w.in_features):
            acc += xd[:, k : k + 1] * w.columns[k].astype(F64)
    if bias is not None:
        acc += np.asarray(bias, dtype=F32).astype(F64)
    return _round(acc).reshape(*lead, w.out_features)


def matmul(a: npt.ArrayLike, b: npt.ArrayLike) -> np.ndarray:
    """``a[m, k] @ b[k, n]``, each output summed over ``k`` ascending in double."""
    ad = np.asarray(a, dtype=F32).astype(F64)
    bd = np.asarray(b, dtype=F32)
    acc = np.zeros((ad.shape[0], bd.shape[1]), dtype=F64)
    for k in range(ad.shape[1]):
        acc += ad[:, k : k + 1] * bd[k].astype(F64)
    return _round(acc)


def quantize(x: npt.ArrayLike, kind: str = "q8_0") -> tuple[np.ndarray, np.ndarray]:
    """Blocks of 32 along the last axis: ``d = amax / 127`` (``/ 7`` for Q4_0) in float, ``q = round_half_even(x *
    (1 / d))`` in float, clamped; non-finite values and all-zero blocks give 0. Returns int8 values and the scales."""
    limit = 7.0 if kind == "q4_0" else 127.0
    values = np.asarray(x, dtype=F32)
    blocks = values.reshape(*values.shape[:-1], values.shape[-1] // 32, 32)
    with np.errstate(all="ignore"):
        finite = np.isfinite(blocks)
        amax = np.where(finite, np.abs(blocks), F32(0)).max(axis=-1)
        d = amax / F32(limit)
        inverse = np.where(d != 0, F32(1) / d, F32(0)).astype(F32)
        q = np.where(finite, np.rint(blocks * inverse[..., None]), F32(0))
    q = np.clip(q, -limit, limit).astype(np.int8)
    return q.reshape(values.shape), d.astype(F32)


def _linear_q8(rows: np.ndarray, wq: np.ndarray, ws: np.ndarray) -> np.ndarray:
    """Activations quantised to Q8_0; per output, blocks ascending: ``acc += (d_x * d_w) * isum`` in double, where
    ``isum`` is the block's exact integer dot product."""
    xq, xs = quantize(rows)
    acc = np.zeros((rows.shape[0], wq.shape[0]), dtype=F64)
    for b in range(ws.shape[1]):
        block = slice(32 * b, 32 * (b + 1))
        isum = xq[:, block].astype(np.int64) @ wq[:, block].astype(np.int64).T  # exact integers
        acc += (xs[:, b : b + 1].astype(F64) * ws[:, b].astype(F64)) * isum.astype(F64)
    return acc


class Weight:
    """A linear weight ``[out, in]``, kept as columns ``[in, out]`` so step ``k`` of every output's sum reads one row,
    or quantised (``q8_0``/``q4_0``) when ``in`` is a multiple of 32."""

    def __init__(self, weight: np.ndarray, kind: str | None = None) -> None:
        self.out_features, self.in_features = weight.shape
        self.quantized: tuple[np.ndarray, np.ndarray] | None = None
        if kind and self.in_features % 32 == 0:
            self.quantized = quantize(weight, kind)
        else:
            self.columns = np.ascontiguousarray(np.asarray(weight, dtype=F32).T)


def rms_norm(x: npt.ArrayLike, weight: npt.ArrayLike | None, eps: float, add_unit_offset: bool = False) -> np.ndarray:
    """Per row ``inv = 1 / sqrt(sum x^2 / dim + eps)`` (sum ascending in double), ``out = x * inv * scale`` with
    ``scale = w`` or ``1 + w``, rounded once."""
    rows, lead = _rows(x)
    xd = rows.astype(F64)
    total = np.zeros(rows.shape[0], dtype=F64)
    for i in range(rows.shape[1]):
        total += xd[:, i] * xd[:, i]
    inverse = 1.0 / np.sqrt(total / float(rows.shape[1]) + eps)
    out = xd * inverse[:, None]
    if weight is not None:
        w = np.asarray(weight, dtype=F32).astype(F64)
        out = out * (1.0 + w if add_unit_offset else w)
    return _round(out).reshape(*lead, rows.shape[1])


def layer_norm(x: npt.ArrayLike, weight: npt.ArrayLike, bias: npt.ArrayLike, eps: float) -> np.ndarray:
    """Per row ``mean = sum x / dim`` and ``var = sum (x - mean)^2 / dim`` (both sums ascending in double), then
    ``(x - mean) * (1 / sqrt(var + eps)) * w + b``, rounded once."""
    rows, lead = _rows(x)
    xd = rows.astype(F64)
    dim = rows.shape[1]
    total = np.zeros(rows.shape[0], dtype=F64)
    for i in range(dim):
        total += xd[:, i]
    mean = total / float(dim)
    squares = np.zeros(rows.shape[0], dtype=F64)
    for i in range(dim):
        centred = xd[:, i] - mean
        squares += centred * centred
    inverse = 1.0 / np.sqrt(squares / float(dim) + eps)
    w = np.asarray(weight, dtype=F32).astype(F64)
    b = np.asarray(bias, dtype=F32).astype(F64)
    out = (xd - mean[:, None]) * inverse[:, None] * w + b
    return _round(out).reshape(*lead, dim)


def silu(x: npt.ArrayLike) -> np.ndarray:
    d = np.asarray(x, dtype=F32).astype(F64)
    return _round(d * sigmoid(d))


def gelu(x: npt.ArrayLike, approximate: str = "none") -> np.ndarray:
    """Exact ``0.5 x erfc(-x / sqrt 2)``, or the tanh form ``0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))``."""
    d = np.asarray(x, dtype=F32).astype(F64)
    if approximate == "tanh":
        return _round(0.5 * d * (1.0 + tanh(7.97884560802865355879e-01 * (d + 0.044715 * d * d * d))))
    return _round(0.5 * d * erfc(-d * 7.07106781186547524401e-01))


def swiglu(gate: npt.ArrayLike, up: npt.ArrayLike, activation: str = "silu") -> np.ndarray:
    """``act(gate)`` rounded to float, times ``up`` in float."""
    act = silu(gate) if activation == "silu" else gelu(gate, approximate="tanh")
    return act * np.asarray(up, dtype=F32)


def softcap(x: npt.ArrayLike, cap: float) -> np.ndarray:
    return _round(cap * tanh(np.asarray(x, dtype=F32).astype(F64) / cap))


def rope_inv_freq(
    head_dim: int, theta: float = 10000.0, *, rotary_dim: int | None = None, scaling: Mapping[str, Any] | None = None
) -> np.ndarray:
    """``exp(-(2i / rotary_dim) log theta)`` in double, then Hugging Face ``rope_scaling`` (linear, llama3,
    longrope with either factor set, yarn)."""
    dim = head_dim if rotary_dim is None else rotary_dim
    log_theta = float(log(float(theta)))
    freqs = [float(exp(-(2 * i / dim) * log_theta)) for i in range(dim // 2)]
    kind = "default" if not scaling else str(scaling.get("rope_type", scaling.get("type", "default")))
    if kind == "linear":
        assert scaling is not None
        freqs = [f / float(scaling["factor"]) for f in freqs]
    elif kind == "llama3":
        assert scaling is not None
        factor = float(scaling["factor"])
        low = float(scaling.get("low_freq_factor", 1.0))
        high = float(scaling.get("high_freq_factor", 4.0))
        original = float(scaling.get("original_max_position_embeddings", 8192))
        scaled = []
        for f in freqs:
            wavelen = 2 * math.pi / f
            if wavelen < original / high:
                scaled.append(f)
            elif wavelen > original / low:
                scaled.append(f / factor)
            else:
                smooth = (original / wavelen - low) / (high - low)
                scaled.append((1 - smooth) * f / factor + smooth * f)
        freqs = scaled
    elif kind == "longrope":
        assert scaling is not None
        factors = scaling["long_factor"] if scaling.get("factor_set") == "long" else scaling["short_factor"]
        freqs = [f / float(s) for f, s in zip(freqs, factors, strict=True)]
    elif kind == "yarn":
        assert scaling is not None
        factor = float(scaling["factor"])
        original = float(scaling["original_max_position_embeddings"])
        # The pair index at which a frequency turns `beta` times over the original context.
        low = dim * float(log(original / (float(scaling.get("beta_fast", 32.0)) * 2 * math.pi))) / (2 * log_theta)
        high = dim * float(log(original / (float(scaling.get("beta_slow", 1.0)) * 2 * math.pi))) / (2 * log_theta)
        if scaling.get("truncate", True):
            low, high = math.floor(low), math.ceil(high)
        low, high = max(low, 0), min(high, dim - 1)
        high = high + 0.001 if low == high else high
        ramps = [min(max((i - low) / (high - low), 0.0), 1.0) for i in range(len(freqs))]
        freqs = [f / factor * r + f * (1 - r) for f, r in zip(freqs, ramps, strict=True)]
    elif kind != "default":
        raise ValueError(f"unsupported rope scaling {kind!r}")
    return np.array(freqs, dtype=F64)


def rope(x: npt.ArrayLike, positions: npt.ArrayLike, inv_freq: npt.ArrayLike, interleaved: bool = False) -> np.ndarray:
    """Rotates pairs ``(i, i + rotary_dim/2)`` (or ``(2i, 2i+1)``) of ``x[tokens, heads, head_dim]`` by
    ``position * inv_freq[i]``: ``a cos - b sin`` and ``a sin + b cos`` in double, rounded once."""
    values = np.asarray(x, dtype=F32)
    freqs = np.asarray(inv_freq, dtype=F64)
    half = freqs.shape[0]
    angles = np.asarray(positions, dtype=np.int64).astype(F64)[:, None] * freqs[None, :]
    c, s = cos(angles)[:, None, :], sin(angles)[:, None, :]
    first = np.arange(half) * 2 if interleaved else np.arange(half)
    second = first + 1 if interleaved else first + half
    a = values[:, :, first].astype(F64)
    b = values[:, :, second].astype(F64)
    out = values.copy()
    out[:, :, first] = _round(a * c - b * s)
    out[:, :, second] = _round(a * s + b * c)
    return out


def attention(
    q: npt.ArrayLike,
    k: npt.ArrayLike,
    v: npt.ArrayLike,
    scale: float | None = None,
    causal: bool = True,
    q_offset: int | None = None,
    window: int | None = None,
    softcap: float | None = None,
) -> np.ndarray:
    """For each query and head: scores ``(sum_i q_i k_ji) * scale`` (soft-capped), ``p_j = exp(s_j - max)``, then
    ``(sum_j p_j v_j) * (1 / sum_j p_j)``, both sums over the visible keys ascending, rounded once."""
    qv, kv, vv = (np.asarray(a, dtype=F32) for a in (q, k, v))
    q_len, q_heads, head_dim = qv.shape
    kv_len, kv_heads, _ = kv.shape
    group = q_heads // kv_heads
    heads = np.arange(q_heads) // group
    scale = 1.0 / math.sqrt(head_dim) if scale is None else scale
    offset = kv_len - q_len if q_offset is None else q_offset
    out = np.zeros((q_len, q_heads, vv.shape[2]), dtype=F32)
    for t in range(q_len):
        if causal:
            end = min(offset + t + 1, kv_len)
            first = end - window if window and end > window else 0
        elif window:  # bidirectional local attention: keys closer than the window on either side
            first, end = max(offset + t + 1 - window, 0), min(offset + t + window, kv_len)
        else:
            first, end = 0, kv_len
        count = end - first
        if count <= 0:
            continue
        qt = qv[t].astype(F64)
        keys = kv[first:end][:, heads, :].astype(F64)  # [count, q_heads, head_dim]
        scores = np.zeros((q_heads, count), dtype=F64)
        for i in range(head_dim):
            scores += qt[:, i : i + 1] * keys[:, :, i].T
        scores = scores * scale
        if softcap:
            scores = softcap * tanh(scores / softcap)
        top = scores[:, 0].copy()
        for j in range(1, count):
            top = np.where(scores[:, j] > top, scores[:, j], top)
        p = exp(scores - top[:, None])
        total = np.zeros(q_heads, dtype=F64)
        for j in range(count):
            total += p[:, j]
        values = vv[first:end][:, heads, :].astype(F64)
        acc = np.zeros((q_heads, vv.shape[2]), dtype=F64)
        for j in range(count):
            acc += p[:, j : j + 1] * values[j]
        out[t] = _round(acc * (1.0 / total)[:, None])
    return out


def biased_attention(
    q: npt.ArrayLike, k: npt.ArrayLike, v: npt.ArrayLike, bias: npt.ArrayLike, scale: float
) -> np.ndarray:
    """DeBERTa's attention: for each query ``t`` and head ``h``, scores ``(sum_i q_i k_ji + bias[h, t, j]) * scale``
    in double over every key, then the softmax and value sums of :func:`attention`."""
    qv, kv, vv = (np.asarray(a, dtype=F32) for a in (q, k, v))
    bv = np.asarray(bias, dtype=F32).astype(F64)
    q_len, heads, head_dim = qv.shape
    kv_len = kv.shape[0]
    out = np.zeros((q_len, heads, vv.shape[2]), dtype=F32)
    keys = kv.astype(F64)
    values = vv.astype(F64)
    for t in range(q_len):
        qt = qv[t].astype(F64)
        scores = np.zeros((heads, kv_len), dtype=F64)
        for i in range(head_dim):
            scores += qt[:, i : i + 1] * keys[:, :, i].T
        scores = (scores + bv[:, t, :]) * scale
        top = scores[:, 0].copy()
        for j in range(1, kv_len):
            top = np.where(scores[:, j] > top, scores[:, j], top)
        p = exp(scores - top[:, None])
        total = np.zeros(heads, dtype=F64)
        for j in range(kv_len):
            total += p[:, j]
        acc = np.zeros((heads, vv.shape[2]), dtype=F64)
        for j in range(kv_len):
            acc += p[:, j : j + 1] * values[j]
        out[t] = _round(acc * (1.0 / total)[:, None])
    return out


def relative_bucket(distance: int, buckets: int, max_position: int) -> int:
    """DeBERTa's log bucket of the relative distance ``distance``: itself when ``|distance| <= buckets // 2`` (or
    without buckets), else ``sign * (ceil(f32(f32(log(f32(|d| / mid))) / f32(log(f32((max - 1) / mid)))) *
    (mid - 1)) + mid)`` with every step rounded to float32."""
    mid = buckets // 2
    if buckets <= 0 or max_position <= 0 or abs(distance) <= mid:
        return distance
    ratio = F32(abs(distance)) / F32(mid)
    numerator = F32(float(log(float(ratio))))
    denominator = F32(float(log(float(F32((max_position - 1) / mid)))))
    scaled = F32(numerator / denominator) * F32(mid - 1)
    return (1 if distance > 0 else -1) * (math.ceil(float(scaled)) + mid)


def t5_relative_bucket(distance: int, buckets: int, max_distance: int) -> int:
    """T5's bidirectional bucket of the relative distance ``distance`` (key minus query): ``n = buckets // 2`` buckets
    per direction (keys after the query from ``n`` on), ``r = |distance|`` itself below ``exact = n // 2``, else
    ``min(exact + trunc(f32(f32(log(f32(r) / exact)) / f32(log(max_distance / exact))) * (n - exact)), n - 1)``
    with every step rounded to float32."""
    half = buckets // 2
    exact = half // 2
    offset = half if distance > 0 else 0
    r = abs(distance)
    if r < exact:
        return offset + r
    numerator = F32(float(log(float(F32(F32(r) / F32(exact))))))
    scaled = F32(numerator / F32(float(log(max_distance / exact)))) * F32(half - exact)
    return offset + min(exact + math.trunc(float(scaled)), half - 1)


def t5_decoder_bucket(distance: int, buckets: int, max_distance: int) -> int:
    """T5's one-directional (decoder) bucket of ``distance`` (key minus query, never positive): all ``n = buckets``
    buckets for keys at or before the query, ``r = max(-distance, 0)`` itself below ``exact = n // 2``, else
    ``min(exact + trunc(f32(f32(log(f32(r) / exact)) / f32(log(max_distance / exact))) * (n - exact)), n - 1)``."""
    exact = buckets // 2
    r = max(-distance, 0)
    if r < exact:
        return r
    numerator = F32(float(log(float(F32(F32(r) / F32(exact))))))
    scaled = F32(numerator / F32(float(log(max_distance / exact)))) * F32(buckets - exact)
    return min(exact + math.trunc(float(scaled)), buckets - 1)


# -- random numbers and sampling ----------------------------------------------------------------------------------

_MASK = (1 << 64) - 1


def _mix64(z: int) -> int:
    """The SplitMix64 output function on ``z mod 2^64``."""
    z &= _MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    return z ^ (z >> 31)


class Random:
    """xoshiro256** seeded through SplitMix64."""

    def __init__(self, seed: int) -> None:
        state = seed & _MASK
        words = []
        for _ in range(4):
            state = (state + 0x9E3779B97F4A7C15) & _MASK
            z = state
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
            words.append(z ^ (z >> 31))
        self._s = words

    @staticmethod
    def _rotl(x: int, k: int) -> int:
        return ((x << k) | (x >> (64 - k))) & _MASK

    def next_u64(self) -> int:
        s0, s1, s2, s3 = self._s
        result = (self._rotl((s1 * 5) & _MASK, 7) * 9) & _MASK
        t = (s1 << 17) & _MASK
        s2 ^= s0
        s3 ^= s1
        s1 ^= s2
        s0 ^= s3
        s2 ^= t
        s3 = self._rotl(s3, 45)
        self._s = [s0, s1, s2, s3]
        return result

    def next_double(self) -> float:
        """The top 53 bits, times ``2^-53``: a double in ``[0, 1)``."""
        return float(self.next_u64() >> 11) * (1.0 / 9007199254740992.0)

    def next_gaussian(self) -> float:
        """Irwin-Hall: the sum of 12 doubles, minus 6, rounded to float."""
        total = 0.0
        for _ in range(12):
            total += self.next_double()
        return float(F32(total - 6.0))


def fill_gaussian(seed: int, n: int) -> np.ndarray:
    random = Random(seed)
    return np.array([random.next_gaussian() for _ in range(n)], dtype=F32)


class Sampler:
    """Logit bias and penalties first, one token at a time in float: ``x + bias``; for the repetition penalty ``r``
    over the distinct tokens among the last ``repeat_last_n`` of prompt and output, ``x / r`` when ``x > 0`` else
    ``x * r``; ``x - (count * frequency + presence)`` (the penalty in double, rounded to float) for the tokens the
    output holds; ``x - dry`` for DRY (the penalty in double, rounded to float). Then greedy (temperature 0) or:
    logits divided by the temperature in float, softmax, candidates ordered by (probability descending, id
    ascending), top-k, top-n-sigma (the first as many candidates as tokens within ``n`` deviations of the top
    logit), typical-p (the candidates with surprise closest to the entropy, back in order), top-p (the shortest
    prefix reaching ``top_p``), min-p (candidates below ``min_p`` times the first are dropped), XTC (with one
    ``next_double()`` below ``xtc_probability``, all but the last of the candidates at least ``xtc_threshold`` of
    the kept mass go), then one ``next_double()`` scaled by the kept mass picks the first candidate whose running
    sum exceeds it. :meth:`begin` sets the prompt, :meth:`accept`
    records each generated token. Constrained decoding passes the allowed ids (ascending): the adjusted logits are
    restricted to them before the steps after the penalties."""

    def __init__(
        self,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int = 0,
        *,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        repeat_last_n: int = 64,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: Mapping[int, float] | None = None,
        watermark_key: str | None = None,
        watermark_gamma: float = 0.25,
        watermark_delta: float = 2.0,
        typical_p: float = 1.0,
        top_n_sigma: float = 0.0,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.1,
        dry_multiplier: float = 0.0,
        dry_base: float = 1.75,
        dry_allowed_length: int = 2,
        dry_penalty_last_n: int = -1,
        dry_breakers: Sequence[int] = (),
        mirostat: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        dynatemp_range: float = 0.0,
        dynatemp_exponent: float = 1.0,
    ) -> None:
        self.temperature, self.top_k, self.top_p, self.min_p = temperature, top_k, top_p, min_p
        self.mirostat, self.mirostat_tau, self.mirostat_eta = mirostat, mirostat_tau, mirostat_eta
        self.dynatemp_range, self.dynatemp_exponent = dynatemp_range, dynatemp_exponent
        self.mu = 2.0 * mirostat_tau
        self.typical_p, self.top_n_sigma = typical_p, top_n_sigma
        self.xtc_probability, self.xtc_threshold = xtc_probability, xtc_threshold
        self.dry_multiplier, self.dry_base = dry_multiplier, dry_base
        self.dry_allowed_length, self.dry_penalty_last_n = dry_allowed_length, dry_penalty_last_n
        self.dry_breakers = set(dry_breakers)
        self.watermark_key, self.watermark_gamma, self.watermark_delta = watermark_key, watermark_gamma, watermark_delta
        self.repetition_penalty, self.repeat_last_n = repetition_penalty, repeat_last_n
        self.frequency_penalty, self.presence_penalty = frequency_penalty, presence_penalty
        self.logit_bias = dict(logit_bias or {})
        self._random = Random(seed & _MASK)
        self._sequence: list[int] = []
        self._output: list[int] = []

    def begin(self, prompt: Sequence[int]) -> None:
        self._sequence, self._output = list(prompt), []

    def accept(self, token: int) -> None:
        self._sequence.append(token)
        self._output.append(token)

    def _adjusted(self, logits: npt.ArrayLike) -> np.ndarray:
        values = np.array(logits, dtype=F32)
        for token, bias in self.logit_bias.items():
            if token < len(values):
                values[token] = F32(values[token] + F32(bias))
        if self.repetition_penalty != 1.0 and self.repeat_last_n != 0:
            recent = self._sequence if self.repeat_last_n < 0 else self._sequence[-self.repeat_last_n :]
            r = F32(self.repetition_penalty)
            for token in set(recent):
                if 0 <= token < len(values):
                    x = values[token]
                    values[token] = F32(x / r) if x > 0 else F32(x * r)
        if self.frequency_penalty != 0.0 or self.presence_penalty != 0.0:
            counts: dict[int, int] = {}
            for token in self._output:
                counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                if 0 <= token < len(values):
                    penalty = F32(count * self.frequency_penalty + self.presence_penalty)
                    values[token] = F32(values[token] - penalty)
        if self.dry_multiplier != 0.0 and self.dry_penalty_last_n != 0:
            n = self.dry_penalty_last_n
            window = self._sequence if n < 0 else self._sequence[-n:]
            for token, penalty in self._dry(window).items():
                if 0 <= token < len(values):
                    with np.errstate(over="ignore"):
                        values[token] = F32(values[token] - F32(penalty))
        if self.watermark_key is not None:
            digest = hashlib.sha256(b"dllm-watermark/1\0" + self.watermark_key.encode("utf-8")).digest()
            key = int.from_bytes(digest[:8], "little")
            previous = self._sequence[-1] if self._sequence else -1
            seed = _mix64(key + (previous + 1) * 0x9E3779B97F4A7C15)
            limit = math.floor(self.watermark_gamma * 2**64)
            delta = F32(self.watermark_delta)
            for token in range(len(values)):
                if _mix64(seed + (token + 1) * 0x9E3779B97F4A7C15) < limit:
                    values[token] = F32(values[token] + delta)
        return values

    def _dry(self, window: Sequence[int]) -> dict[int, float]:
        """For each position whose previous tokens equal the window's last ones (matched backwards, no breakers,
        at most 256), the token at that position may repeat a run of that length; the longest run ``n`` of each
        token at least ``dry_allowed_length`` long costs ``multiplier * base^(n - allowed)``, the power multiplied
        out in double."""
        best: dict[int, int] = {}
        end = len(window) - 1
        if end < 1 or window[end] in self.dry_breakers:
            return {}
        for position in range(1, end + 1):
            n = 0
            while n < 256 and n < position:
                a, b = window[position - 1 - n], window[end - n]
                if a != b or a in self.dry_breakers:
                    break
                n += 1
            if n >= self.dry_allowed_length and n > best.get(window[position], 0):
                best[window[position]] = n
        result = {}
        for token, n in best.items():
            power = 1.0
            for _ in range(n - self.dry_allowed_length):
                power = power * self.dry_base
            result[token] = self.dry_multiplier * power
        return result

    def _dynamic_temperature(self, logits: np.ndarray) -> float:
        """``low + (high - low) * (H / log(n))^exponent``, ``H`` the entropy of ``softmax(logits)`` in token order."""
        p = softmax(np.asarray(logits, dtype=F32)).tolist()
        if len(p) < 2:
            return self.temperature
        entropy = 0.0
        for v in p:
            if v > 0:
                entropy = entropy - v * float(log(v))
        low = max(0.0, self.temperature - self.dynatemp_range)
        high = self.temperature + self.dynatemp_range
        return low + (high - low) * _power(entropy / float(log(float(len(p)))), self.dynatemp_exponent)

    def _mirostat(self, order: list[int], probabilities: list[float]) -> int:
        """Mirostat 1.0 keeps a Zipf-fitted ``k``, 2.0 the candidates up to the first (after the first) whose surprise
        in bits exceeds ``mu``; one draw; then ``mu`` moves by ``eta`` times the drawn surprise's excess over tau."""
        n = len(order)
        if self.mirostat == 1:
            keep = n
            sum_tb = sum_tt = 0.0
            for i in range(min(100, n) - 1):
                a, b = probabilities[order[i]], probabilities[order[i + 1]]
                if a > 0 and b > 0:
                    t = float(log((i + 2) / (i + 1)))
                    sum_tb = sum_tb + t * float(log(a / b))
                    sum_tt = sum_tt + t * t
            if sum_tt != 0:
                s_hat = sum_tb / sum_tt
                epsilon = s_hat - 1
                below = 1 - _power(float(n), -epsilon) if epsilon != 0 else 0.0
                if s_hat > 0 and below != 0:
                    ratio = epsilon * _power(2.0, self.mu) / below
                    if ratio > 0 and math.isfinite(ratio):
                        k = _power(ratio, 1 / s_hat)
                        if math.isfinite(k) and k < n:
                            keep = max(1, int(k))
        else:
            keep = next(
                (
                    i
                    for i in range(1, n)
                    if probabilities[order[i]] <= 0 or -float(log(probabilities[order[i]])) / _LN2_BITS > self.mu
                ),
                n,
            )
        kept = order[:keep]
        total = 0.0
        for i in kept:
            total = total + probabilities[i]
        target = self._random.next_double() * total
        running, chosen = 0.0, kept[-1]
        for i in kept:
            running = running + probabilities[i]
            if target < running:
                chosen = i
                break
        surprise = -float(log(probabilities[chosen] / total)) / _LN2_BITS
        self.mu = self.mu - self.mirostat_eta * (surprise - self.mirostat_tau)
        return chosen

    def sample(self, logits: npt.ArrayLike, allowed: Sequence[int] | None = None) -> int:
        logits = self._adjusted(logits)
        if allowed is not None:
            return int(allowed[self._choose(logits[np.asarray(allowed, dtype=np.int64)])])
        return self._choose(logits)

    def _choose(self, logits: np.ndarray) -> int:
        if self.temperature == 0:
            return argmax(logits)
        temperature = self.temperature
        if self.dynatemp_range > 0:
            temperature = self._dynamic_temperature(logits)
            if F32(temperature) == 0:
                return argmax(logits)
        scaled = np.asarray(logits, dtype=F32) / F32(temperature)
        probabilities = softmax(scaled).tolist()
        order = sorted(range(len(probabilities)), key=lambda i: (-probabilities[i], i))
        if self.mirostat:
            return self._mirostat(order, probabilities)
        candidates = order[: self.top_k] if self.top_k > 0 else order
        if self.top_n_sigma > 0:
            xs = [float(x) for x in np.asarray(logits, dtype=F32) if math.isfinite(float(x))]
            if xs:
                mean = 0.0
                for x in xs:
                    mean = mean + x
                mean = mean / len(xs)
                variance = 0.0
                for x in xs:
                    variance = variance + (x - mean) * (x - mean)
                variance = variance / len(xs)
                floor = max(xs) - self.top_n_sigma * math.sqrt(variance)
                candidates = candidates[: max(1, len([x for x in xs if x >= floor]))]
        if self.typical_p < 1:
            mass = 0.0
            for i in candidates:
                mass = mass + probabilities[i]
            q = {i: probabilities[i] / mass for i in candidates}
            surprise = {i: -float(log(q[i])) if q[i] > 0 else math.inf for i in candidates}
            entropy = 0.0
            for i in candidates:
                if q[i] > 0:
                    entropy = entropy - q[i] * -surprise[i]
            rank = {i: r for r, i in enumerate(candidates)}
            typical, reached = [], 0.0
            for i in sorted(candidates, key=lambda i: (abs(surprise[i] - entropy), rank[i])):
                typical.append(i)
                reached = reached + q[i]
                if reached >= self.typical_p:
                    break
            candidates = sorted(typical, key=lambda i: rank[i])
        if self.top_p < 1:
            cumulative = 0.0
            for n, i in enumerate(candidates):
                cumulative = cumulative + probabilities[i]
                if cumulative >= self.top_p:
                    candidates = candidates[: n + 1]
                    break
        if self.min_p > 0:
            floor = self.min_p * probabilities[candidates[0]]
            candidates = candidates[
                : next((n for n in range(1, len(candidates)) if probabilities[candidates[n]] < floor), len(candidates))
            ]
        total = 0.0
        for i in candidates:
            total = total + probabilities[i]
        if self.xtc_probability > 0 and self._random.next_double() < self.xtc_probability:
            top = [i for i in candidates if probabilities[i] >= self.xtc_threshold * total]
            if len(top) >= 2 and top == candidates[: len(top)]:
                candidates = candidates[len(top) - 1 :]
                total = 0.0
                for i in candidates:
                    total = total + probabilities[i]
        target = self._random.next_double() * total
        running = 0.0
        for i in candidates:
            running = running + probabilities[i]
            if target < running:
                return i
        return candidates[-1]


_LN2_BITS = 0.6931471805599453


def _power(a: float, b: float) -> float:
    """``a^b`` as ``exp(b * log(a))`` with the portable functions (1 for ``b == 0``, 0 for ``a == 0``)."""
    if b == 0:
        return 1.0
    if a == 0:
        return 0.0
    return float(exp(b * float(log(a))))


def sampler(options: Any, breakers: Sequence[int] = ()) -> Sampler:
    """A sampler with the settings of an engine ``SamplingOptions`` (only its values are read) and the DRY breaker
    token ids."""
    return Sampler(
        options.temperature,
        options.top_k,
        options.top_p,
        options.seed,
        min_p=options.min_p,
        repetition_penalty=options.repetition_penalty,
        repeat_last_n=options.repeat_last_n,
        frequency_penalty=options.frequency_penalty,
        presence_penalty=options.presence_penalty,
        logit_bias=dict(options.logit_bias),
        watermark_key=options.watermark_key,
        watermark_gamma=options.watermark_gamma,
        watermark_delta=options.watermark_delta,
        typical_p=options.typical_p,
        top_n_sigma=options.top_n_sigma,
        xtc_probability=options.xtc_probability,
        xtc_threshold=options.xtc_threshold,
        dry_multiplier=options.dry_multiplier,
        dry_base=options.dry_base,
        dry_allowed_length=options.dry_allowed_length,
        dry_penalty_last_n=options.dry_penalty_last_n,
        dry_breakers=sorted(breakers),
        mirostat=options.mirostat,
        mirostat_tau=options.mirostat_tau,
        mirostat_eta=options.mirostat_eta,
        dynatemp_range=options.dynatemp_range,
        dynatemp_exponent=options.dynatemp_exponent,
    )


def dry_breakers(token_bytes: Sequence[bytes], breakers: Sequence[str]) -> list[int]:
    """The ids of the tokens whose bytes contain one of the breaker strings' UTF-8 bytes."""
    return [t for t, data in enumerate(token_bytes) if any(b.encode("utf-8") in data for b in breakers)]


def healing(prefix: bytes, token_bytes: Sequence[bytes]) -> Callable[[list[int]], list[int] | None]:
    """Token healing (docs/specification.md#token-healing): until the output's bytes hold ``prefix`` (the bytes of
    the prompt token taken back), the next token's bytes must be a non-empty prefix of what is left of it or start
    with all of it; stop tokens (which have no bytes) are not allowed. After that, any token."""

    def allowed(generated: list[int]) -> list[int] | None:
        written = b"".join(token_bytes[token] for token in generated)
        if len(written) >= len(prefix):
            return None
        left = prefix[len(written) :]
        return [t for t, data in enumerate(token_bytes) if data and (left.startswith(data) or data.startswith(left))]

    return allowed


def minimum_length(
    min_tokens: int, stop_tokens: Sequence[int], vocabulary: int
) -> Callable[[list[int]], list[int] | None]:
    """A minimum answer length (docs/specification.md#length-and-stop-controls): until ``min_tokens`` tokens are
    written, the next token may be any but the stop tokens. After that, any token."""
    stops = set(stop_tokens)
    others = [t for t in range(vocabulary) if t not in stops]

    def allowed(generated: list[int]) -> list[int] | None:
        return others if len(generated) < min_tokens else None

    return allowed


FIM_PARTS = ("fim_prefix", "fim_suffix", "fim_middle")
FIM_ENDS = ("fim_prefix", "fim_suffix", "fim_middle", "fim_pad", "file_sep", "repo_name", "endoftext")


def fill_in_the_middle(
    token_id: Callable[[str], int | None], prefix: Sequence[int], suffix: Sequence[int]
) -> tuple[list[int], list[int]]:
    """Fill-in-the-middle (docs/specification.md#fill-in-the-middle): the prompt ``<fim_prefix> prefix <fim_suffix>
    suffix <fim_middle>`` and the tokens that end the middle besides the stop tokens, each token looked up as
    ``<|name|>`` and then as ``<name>``. Raises ``ValueError`` when a FIM part is missing."""

    def find(name: str) -> int | None:
        for spelling in (f"<|{name}|>", f"<{name}>"):
            if (token := token_id(spelling)) is not None:
                return token
        return None

    parts = [find(name) for name in FIM_PARTS]
    if any(part is None for part in parts):
        raise ValueError("no fill-in-the-middle tokens")
    start, gap, middle = (int(part) for part in parts if part is not None)
    ends = sorted({token for name in FIM_ENDS if (token := find(name)) is not None})
    return [start, *prefix, gap, *suffix, middle], ends


# -- the decoder --------------------------------------------------------------------------------------------------


def guided(logits: npt.ArrayLike, negative: npt.ArrayLike, scale: float) -> np.ndarray:
    """Classifier-free guidance: ``n + scale * (l - n)``, each operation rounded to float32."""
    own, n = _f64(logits), _f64(negative)
    return _round(n + _round(float(F32(scale)) * _round(own - n)))


def contrasted(logits: npt.ArrayLike, amateur: npt.ArrayLike, alpha: float, beta: float) -> np.ndarray:
    """Contrastive decoding: ``(1 + beta) * l - beta * a`` for the tokens at least ``alpha`` times as likely as the
    most likely one, ``-inf`` for the others."""
    p = softmax(logits)
    keep = _f64(p) >= float(alpha) * float(p[argmax(p)])
    combined = _round(_round(float(F32(1.0 + beta)) * _f64(logits)) - _round(float(F32(beta)) * _f64(amateur)))
    combined[~keep] = -np.inf
    return combined


def ensembled(members: Sequence[tuple[npt.ArrayLike, float]]) -> np.ndarray:
    """An ensemble: ``sum_i (w_i / W) * log_softmax_i`` in double, in member order, rounded to float32 once."""
    total_weight = 0.0
    for _, weight in members:
        total_weight += float(weight)
    total = np.zeros_like(_f64(members[0][0]))
    for logits, weight in members:
        total = total + (float(weight) / total_weight) * _f64(log_softmax(logits))
    return _round(total)


class ThinkingBudget:
    """The thinking budget of the specification (``docs/specification.md#reasoning``): ``limit`` tokens of a
    ``<think>`` block, which the output opens itself or (``started``) the prompt opened. ``decode`` gives the bytes of
    output tokens and ``encode`` the tokens of a text (the model's tokenizer)."""

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(
        self,
        limit: int,
        started: bool,
        decode: Callable[[Sequence[int]], bytes],
        encode: Callable[[str], list[int]],
    ) -> None:
        self.limit, self.started, self.decode, self.encode = limit, started, decode, encode

    def text(self, tokens: Sequence[int]) -> str:
        """The output text: its bytes without a trailing incomplete UTF-8 sequence, invalid bytes replaced."""
        data = self.decode(tokens)
        end = len(data)
        for back in range(1, min(4, len(data)) + 1):
            byte = data[-back]
            if 0x80 <= byte < 0xC0:
                continue
            if byte >= 0xC0 and back < (2 if byte < 0xE0 else 3 if byte < 0xF0 else 4):
                end = len(data) - back
            break
        return data[:end].decode("utf-8", errors="replace")

    def state(self, tokens: Sequence[int]) -> tuple[bool, bool]:
        """``(opened, closed)``: whether the output is in a block or past one; ``(False, True)`` when it does not
        think, ``(False, False)`` while it may still open one."""
        text = self.text(tokens)
        if self.started:
            return True, self.CLOSE in text
        stripped = text.lstrip()
        if stripped.startswith(self.OPEN):
            return True, self.CLOSE in stripped[len(self.OPEN) :]
        return False, not self.OPEN.startswith(stripped)

    def closing(self, tokens: Sequence[int]) -> list[int]:
        text = self.text(tokens)
        return self.encode(("" if text.endswith("\n") else "\n") + self.CLOSE + "\n\n")


class ReferenceTransformer:
    """The decoder of :class:`etalii_dllm.transformer.Transformer`, step for step, on this module's kernels.

    ``tensors`` are the model file's float32 tensors; ``quantize`` (``q8_0``/``q4_0``) and ``steering`` (layer ->
    vector added to the residual stream after that layer) mean what they mean for the engine."""

    def __init__(
        self,
        config: TransformerConfig,
        tensors: Mapping[str, npt.ArrayLike],
        *,
        quantize: str | None = None,
        steering: Mapping[int, npt.ArrayLike] | None = None,
    ) -> None:
        self.config = config
        source = {name: np.asarray(tensors[name], dtype=F32) for name in config.tensor_shapes()}
        self.embedding = source["token_embedding.weight"]
        self.positions = source.get("position_embedding.weight")
        self.steering = {int(layer): np.asarray(v, dtype=F32) for layer, v in (steering or {}).items()}
        weights = dict(source)
        if config.residual_multiplier != 1.0:  # Granite: scale the two output projections once, in float
            for name in weights:
                if name.endswith(("attention.o.weight", "down.weight")):
                    weights[name] = weights[name] * F32(config.residual_multiplier)
        if config.rope_attention_factor != 1.0:  # LongRoPE/YaRN: scale what feeds the rotated q/k dimensions
            scaled = ("attention.q.weight", "attention.q.bias", "attention.k.weight", "attention.k.bias")
            if config.qk_norm:  # the norm comes after the projection, so its weights carry the factor
                scaled = ("attention.q_norm.weight", "attention.k_norm.weight")
            for name in weights:
                if name.endswith(scaled):
                    rows = weights[name].reshape(-1, config.head_dim, *weights[name].shape[1:]).copy()
                    rows[:, : config.rotary_dimension] *= F32(config.rope_attention_factor)
                    weights[name] = rows.reshape(weights[name].shape)
        self.w: dict[str, Any] = {
            name: Weight(values, None if name.endswith(".router.weight") else quantize)
            if name.endswith((*_MATRICES, ".router.weight"))
            else values
            for name, values in weights.items()
        }
        head = weights["token_embedding.weight" if config.tie_word_embeddings else "lm_head.weight"]
        self.lm_head = Weight(head, quantize)
        self.lm_head_bias = weights.get("lm_head.bias")
        self.inv_freq = rope_inv_freq(
            config.head_dim, config.rope_theta, rotary_dim=config.rotary_dimension, scaling=config.rope_scaling
        )
        self.local_inv_freq = self.inv_freq
        if config.local_rope_theta is not None:
            self.local_inv_freq = rope_inv_freq(
                config.head_dim, config.local_rope_theta, rotary_dim=config.rotary_dimension
            )
        self.reset()

    @classmethod
    def from_engine_model(cls, model: Any) -> ReferenceTransformer:
        """The reference twin of an engine :class:`~etalii_dllm.transformer.Transformer` (same weights and
        options)."""
        return cls(model.config, model.tensors, quantize=model.quantization, steering=model.steering)

    def reset(self) -> None:
        """Forgets the context (the key/value cache)."""
        empty = np.zeros((0, self.config.kv_heads, self.config.head_dim), F32)
        self.keys: list[np.ndarray] = [empty] * self.config.layers
        self.values: list[np.ndarray] = [empty] * self.config.layers

    @property
    def length(self) -> int:
        return int(self.keys[0].shape[0])

    def forward(self, tokens: Sequence[int]) -> np.ndarray:
        """Appends ``tokens`` to the context and returns the logits after the last of them."""
        config = self.config
        w = self.w
        start = self.length
        count = len(tokens)
        positions = np.arange(start, start + count)
        x = self.embedding[np.asarray(tokens, dtype=np.int64)]
        if self.positions is not None:  # GPT-2's learned positions, added in float32
            x = x + self.positions[positions]
        if config.embedding_multiplier != 1.0:
            x = x * F32(config.embedding_multiplier)

        def norm(values: np.ndarray, name: str) -> np.ndarray:
            if config.layer_norm:
                return layer_norm(values, w[name], w[name.removesuffix("weight") + "bias"], config.rms_norm_eps)
            return rms_norm(values, w[name], config.rms_norm_eps, config.norm_unit_offset)

        def plain_mlp(h: np.ndarray, p: str) -> np.ndarray:
            up = linear(h, w[p + "mlp.up.weight"], w[p + "mlp.up.bias"])
            activated = gelu(up, "tanh" if config.activation == "gelu_tanh" else "none")
            return linear(activated, w[p + "mlp.down.weight"], w[p + "mlp.down.bias"])

        for layer in range(config.layers):
            p = f"layers.{layer}."
            h = norm(x, p + "attention_norm.weight") if config.has_pre_norms else x
            q = linear(h, w[p + "attention.q.weight"], w.get(p + "attention.q.bias"))
            k = linear(h, w[p + "attention.k.weight"], w.get(p + "attention.k.bias"))
            v = linear(h, w[p + "attention.v.weight"], w.get(p + "attention.v.bias"))
            if config.qk_norm and config.qk_norm_scope == "all":
                q, k = norm(q, p + "attention.q_norm.weight"), norm(k, p + "attention.k_norm.weight")
            q = q.reshape(count, config.heads, config.head_dim)
            k = k.reshape(count, config.kv_heads, config.head_dim)
            if config.qk_norm and config.qk_norm_scope == "head":
                q, k = norm(q, p + "attention.q_norm.weight"), norm(k, p + "attention.k_norm.weight")
            if not config.absolute_positions:
                inv_freq = self.local_inv_freq if config.uses_local_rope(layer) else self.inv_freq
                q, k = rope(q, positions, inv_freq), rope(k, positions, inv_freq)
            self.keys[layer] = np.concatenate([self.keys[layer], k])
            v = v.reshape(count, config.kv_heads, config.head_dim)
            self.values[layer] = np.concatenate([self.values[layer], v])
            attended = attention(
                q,
                self.keys[layer],
                self.values[layer],
                scale=config.attention_scale,
                causal=True,
                q_offset=start,
                window=config.window(layer),
                softcap=config.attention_softcap,
            )
            flat = attended.reshape(count, config.heads * config.head_dim)
            out = linear(flat, w[p + "attention.o.weight"], w.get(p + "attention.o.bias"))
            if config.parallel_residual is not None:  # both read the input: (attention + mlp) + x
                x = (out + plain_mlp(norm(x, p + config.mlp_norm + ".weight"), p)) + x
                if layer in self.steering:
                    x = x + self.steering[layer]
                continue
            if config.has_post_norms:
                out = norm(out, p + "attention_post_norm.weight")
            x = x + out
            h = norm(x, p + "mlp_norm.weight") if config.has_pre_norms else x
            if config.plain_mlp:
                out = plain_mlp(h, p)
            elif config.is_sparse(layer):
                out = self._experts(h, layer)
            else:
                gate = linear(h, w[p + "mlp.gate.weight"])
                up = linear(h, w[p + "mlp.up.weight"])
                out = linear(swiglu(gate, up, config.activation), w[p + "mlp.down.weight"])
            if config.has_post_norms:
                out = norm(out, p + "mlp_post_norm.weight")
            x = x + out
            if layer in self.steering:
                x = x + self.steering[layer]
        hidden = norm(x[-1:], "final_norm.weight")
        logits = linear(hidden, self.lm_head, self.lm_head_bias)[0]
        if config.logits_scaling != 1.0:
            logits = logits / F32(config.logits_scaling)
        if config.logits_softcap is not None:
            logits = softcap(logits, config.logits_softcap)
        return logits

    def _experts(self, h: np.ndarray, layer: int) -> np.ndarray:
        """A mixture-of-experts block, one row at a time: route the row, run each chosen expert's MLP on it, and add
        ``output * weight`` to zero in increasing expert order (float32); then add the shared expert's output, first
        multiplied by ``sigmoid(row . gate)`` (rounded once) when it is gated."""
        config, w = self.config, self.w
        p = f"layers.{layer}.mlp."
        chosen, weights = moe_route(linear(h, w[p + "router.weight"]), config.experts_per_token,
                                    config.normalize_expert_weights)  # fmt: skip
        out = np.zeros_like(h)
        for r in range(h.shape[0]):
            row = h[r : r + 1]
            for rank in sorted(range(len(chosen[r])), key=lambda j: chosen[r][j]):
                q = f"{p}experts.{chosen[r][rank]}."
                act = swiglu(linear(row, w[q + "gate.weight"]), linear(row, w[q + "up.weight"]), config.activation)
                out[r] = out[r] + linear(act, w[q + "down.weight"])[0] * weights[r, rank]
            if config.shared_expert_intermediate_size is not None:
                q = f"{p}shared."
                act = swiglu(linear(row, w[q + "gate.weight"]), linear(row, w[q + "up.weight"]), config.activation)
                shared = linear(act, w[q + "down.weight"])[0]
                if config.shared_expert_gate:
                    shared = shared * sigmoid_float(linear(row, w[p + "shared_gate.weight"])[0, 0])
                out[r] = out[r] + shared
        return out

    def generate(
        self,
        context: Sequence[int],
        max_tokens: int,
        sampler: Sampler,
        stop_tokens: Sequence[int] = (),
        *,
        overflow: str = "stop",
        window: int | None = None,
        thinking: ThinkingBudget | None = None,
        guide: Callable[[np.ndarray, list[int]], np.ndarray] | None = None,
        allowed: Callable[[list[int]], list[int] | None] | None = None,
    ) -> tuple[list[int], list[np.ndarray]]:
        """Generates from ``context`` (a fresh context) until a stop token or ``max_tokens``; returns the tokens
        (without the stop token) and the logits each was chosen from.

        The context window (``window``, else ``config.context_length``) is the specification's: a full window ends
        the generation (``overflow="stop"``) or rolls it (``"roll"``): the first :data:`ROLL_SINK` tokens and the
        latest half window are kept and computed afresh, while the sampler keeps the whole history. ``thinking``
        closes a ``<think>`` block that has spent its budget with fixed tokens (:class:`ThinkingBudget`). ``guide``
        maps the logits after the tokens so far to the logits decoding uses (:func:`guided`, :func:`contrasted`,
        :func:`ensembled`). ``allowed`` gives the ids the next token may take after the tokens so far (``None``:
        any), as constrained decoding does (:func:`healing`)."""
        window = window or self.config.context_length
        sequence = list(context)
        if window and len(sequence) >= window:
            raise ValueError(f"the prompt has {len(sequence)} tokens; the context window holds {window}")
        self.reset()
        sampler.begin(sequence)
        tokens: list[int] = []
        steps: list[np.ndarray] = []
        unfed = list(sequence)
        """Tokens of ``sequence`` the cache does not hold yet."""
        spent, forced = 0, False

        def append(token: int) -> None:
            nonlocal spent
            before = thinking.state(tokens) if thinking is not None else None
            sampler.accept(token)
            tokens.append(token)
            sequence.append(token)
            unfed.append(token)
            if thinking is not None and before is not None and not before[1] and thinking.state(tokens)[0]:
                spent += 1

        while len(tokens) < max_tokens:
            if window and len(sequence) >= window:
                if overflow != "roll":
                    break
                sequence = sequence[:ROLL_SINK] + sequence[-(window // 2) :]
                self.reset()
                unfed = list(sequence)
            if thinking is not None and not forced:
                opened, closed = thinking.state(tokens)
                if opened and not closed and spent >= thinking.limit:
                    for token in thinking.closing(tokens):
                        if len(tokens) >= max_tokens or (window and len(sequence) >= window):
                            break
                        append(token)
                    forced = True
                    continue
            logits = self.forward(unfed)
            unfed = []
            mask = allowed(tokens) if allowed is not None else None
            token = sampler.sample(guide(logits, tokens) if guide is not None else logits, mask)
            steps.append(logits)
            if token in stop_tokens:
                break
            append(token)
        return tokens, steps

    def beam_search(
        self,
        context: Sequence[int],
        width: int,
        max_tokens: int,
        stop_tokens: Sequence[int] = (),
        *,
        n_best: int = 1,
        length_penalty: float = 1.0,
        window: int | None = None,
    ) -> list[tuple[list[int], str, float, float]]:
        """Beam search as the specification defines it (``docs/specification.md#beam-search``): the best ``n_best``
        finished hypotheses as (tokens without the stop token, finish reason, log-likelihood, score)."""
        window = window or self.config.context_length
        stops = {int(t) for t in stop_tokens}
        self.reset()
        if len(context) > 1:
            self.forward(list(context)[:-1])
        prompt_state = (list(self.keys), list(self.values))
        # A live hypothesis: (tokens, log-likelihood, scored token count, its decoder state before its last token).
        live = [([], 0.0, 0, prompt_state, int(context[-1]))]
        finished: list[tuple[list[int], str, float, int]] = []
        steps = 0
        while True:
            if steps == max_tokens or (window and len(context) + steps >= window):
                finished += [(tokens, "length", total, count) for tokens, total, count, _, _ in live]
                break
            candidates = []
            for index, (tokens, total, _, state, last) in enumerate(live):
                self.keys, self.values = list(state[0]), list(state[1])
                values = log_softmax(self.forward([last]))
                state_after = (list(self.keys), list(self.values))
                ranked = sorted(range(len(values)), key=lambda i: (-float(values[i]), i))
                for token in ranked[: 2 * width]:
                    candidates.append((total + float(values[token]), [*tokens, token], index, token, state_after))
            candidates.sort(key=lambda c: (-c[0], c[1]))
            following = []
            for rank, (total, extended, index, token, state_after) in enumerate(candidates):
                if len(following) == width:
                    break
                if token in stops:
                    if rank < width:
                        finished.append((live[index][0], "stop", total, live[index][2] + 1))
                    continue
                following.append((extended, total, live[index][2] + 1, state_after, token))
            steps += 1
            live = following
            if len(finished) >= width or not live:
                break
        results = []
        for tokens, reason, total, count in finished:
            power = float(exp(float(length_penalty) * float(log(float(count))))) if count > 0 else 1.0
            results.append((tokens, reason, total, total / power if count > 0 else total))
        results.sort(key=lambda r: (-r[3], r[0]))
        return results[:n_best]


class ReferenceEncoder:
    """The BERT, DeBERTa, ModernBERT and T5 encoders of :class:`etalii_dllm.encoder.Encoder` and the embedding the
    engine pools (and projects) from them, step for step, on this module's kernels."""

    def __init__(
        self,
        config: TransformerConfig,
        tensors: Mapping[str, npt.ArrayLike],
        *,
        quantize: str | None = None,
        embedding: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.settings = dict(embedding or {})
        self.w: dict[str, Any] = {}
        for name in config.tensor_shapes():
            values = np.asarray(tensors[name], dtype=F32)
            self.w[name] = Weight(values, quantize) if name.endswith(_MATRICES) else values

    @classmethod
    def from_engine(cls, engine: Any) -> ReferenceEncoder:
        """The reference twin of an engine serving an encoder (same weights, quantisation and pooling)."""
        model = engine.model
        return cls(model.config, model.tensors, quantize=model.quantization, embedding=engine.embedding)

    def hidden_states(self, tokens: Sequence[int], types: Sequence[int] | None = None) -> np.ndarray:
        """``LayerNorm((word + type) + position)`` (positions past the padding id for RoBERTa), then per layer
        ``h = LayerNorm(h + o(attention))`` (every key visible) and ``h = LayerNorm(h + down(gelu(up(h))))``;
        additions in float32. ``types`` default to 0. ModernBERT: :meth:`_modernbert_states`."""
        config, w = self.config, self.w
        if config.family == "modernbert":
            return self._modernbert_states(tokens)
        if config.family == "deberta":
            return self._deberta_states(tokens, types)
        if config.family == "t5":
            return self._t5_states(tokens)
        count = len(tokens)
        kinds = np.zeros(count, dtype=np.int64) if types is None else np.asarray(types, dtype=np.int64)
        words = w["token_embedding.weight"][np.asarray(tokens, dtype=np.int64)]
        ids = np.asarray(tokens, dtype=np.int64)
        if config.padding_index is None:
            positions = np.arange(count, dtype=np.int64)
        else:  # RoBERTa: cumsum of the non-padding mask, times the mask, plus the padding id
            mask = (ids != config.padding_index).astype(np.int64)
            positions = np.cumsum(mask) * mask + config.padding_index
        x = (words + w["token_type_embedding.weight"][kinds]) + w["position_embedding.weight"][positions]

        def norm(values: np.ndarray, name: str) -> np.ndarray:
            return layer_norm(values, w[name + ".weight"], w[name + ".bias"], config.rms_norm_eps)

        def project(values: np.ndarray, name: str) -> np.ndarray:
            return linear(values, w[name + ".weight"], w[name + ".bias"])

        x = norm(x, "embedding_norm")
        shape = (count, config.heads, config.head_dim)
        for layer in range(config.layers):
            p = f"layers.{layer}."
            q, k, v = (project(x, p + f"attention.{n}").reshape(shape) for n in ("q", "k", "v"))
            attended = attention(q, k, v, scale=config.attention_scale, causal=False)
            x = norm(x + project(attended.reshape(count, -1), p + "attention.o"), p + "attention_norm")
            up = project(x, p + "mlp.up")
            activated = gelu(up, "tanh" if config.activation == "gelu_tanh" else "none")
            x = norm(x + project(activated, p + "mlp.down"), p + "mlp_norm")
        return x

    def _deberta_states(self, tokens: Sequence[int], types: Sequence[int] | None) -> np.ndarray:
        """``h = LayerNorm(word (+ type))``; ``R = LayerNorm(relative)``; per layer the BERT layer with attention
        scores ``(q_i . k_j + q_i . pk[d] + k_j . pq[d]) / sqrt(3 head_dim)``, ``pq``/``pk`` the layer's query and
        key projections of ``R`` and ``d = clamp(bucket(i - j) + span, 0, 2 span - 1)``; the two position terms
        each rounded to float32 (a dot in double) and added in float32."""
        config, w = self.config, self.w
        count, heads, head_dim = len(tokens), config.heads, config.head_dim
        x = w["token_embedding.weight"][np.asarray(tokens, dtype=np.int64)]
        if config.type_vocabulary_size:
            kinds = np.zeros(count, dtype=np.int64) if types is None else np.asarray(types, dtype=np.int64)
            x = x + w["token_type_embedding.weight"][kinds]

        def norm(values: np.ndarray, name: str) -> np.ndarray:
            return layer_norm(values, w[name + ".weight"], w[name + ".bias"], config.rms_norm_eps)

        def project(values: np.ndarray, name: str) -> np.ndarray:
            return linear(values, w[name + ".weight"], w[name + ".bias"])

        h = norm(x, "embedding_norm")
        table = norm(w["relative_embedding.weight"], "relative_norm")
        span = config.relative_span
        rows = np.zeros((count, count), dtype=np.int64)
        for i in range(count):
            for j in range(count):
                bucket = relative_bucket(i - j, config.position_buckets, config.max_relative_positions)
                rows[i, j] = min(max(bucket + span, 0), 2 * span - 1)
        scale = 1.0 / math.sqrt(3 * head_dim)
        shape = (count, heads, head_dim)
        for layer in range(config.layers):
            p = f"layers.{layer}."
            q, k, v = (project(h, p + f"attention.{n}").reshape(shape) for n in ("q", "k", "v"))
            pq = project(table, p + "attention.q").reshape(len(table), heads, head_dim)
            pk = project(table, p + "attention.k").reshape(len(table), heads, head_dim)
            bias = np.zeros((heads, count, count), dtype=F32)
            for head in range(heads):
                c2p = linear(q[:, head], pk[:, head])  # [count, table rows]
                p2c = linear(k[:, head], pq[:, head])
                for i in range(count):
                    for j in range(count):
                        bias[head, i, j] = c2p[i, rows[i, j]] + p2c[j, rows[i, j]]
            attended = biased_attention(q, k, v, bias, scale)
            h = norm(h + project(attended.reshape(count, -1), p + "attention.o"), p + "attention_norm")
            activated = gelu(project(h, p + "mlp.up"), "tanh" if config.activation == "gelu_tanh" else "none")
            h = norm(h + project(activated, p + "mlp.down"), p + "mlp_norm")
        return h

    def _t5_states(self, tokens: Sequence[int]) -> np.ndarray:
        """``h = word``; per layer ``h = h + o(attention(RMSNorm(h)))`` with scores ``q_i . k_j + table[bucket(j -
        i), head]`` (the dot in double, the bias added in double, no scale) and ``h = h + down(mlp(RMSNorm(h)))``,
        ``mlp`` ``act(up)`` or ``act(gate) * up`` (ReLU: ``x`` where positive, else +0); then the final RMSNorm."""
        config, w = self.config, self.w
        count, heads, head_dim = len(tokens), config.heads, config.head_dim
        x = w["token_embedding.weight"][np.asarray(tokens, dtype=np.int64)]
        table = w["relative_bias.weight"]
        bias = np.empty((heads, count, count), dtype=F32)
        for i in range(count):
            for j in range(count):
                bias[:, i, j] = table[t5_relative_bucket(j - i, config.position_buckets, config.max_relative_positions)]
        eps = config.rms_norm_eps

        def activate(values: np.ndarray) -> np.ndarray:
            if config.activation == "relu":
                return np.where(values > 0, values, F32(0)).astype(F32)
            if config.activation == "silu":
                return silu(values)
            return gelu(values, "tanh" if config.activation == "gelu_tanh" else "none")

        shape = (count, heads, head_dim)
        for layer in range(config.layers):
            p = f"layers.{layer}."
            normed = rms_norm(x, w[p + "attention_norm.weight"], eps)
            q, k, v = (linear(normed, w[p + f"attention.{n}.weight"]).reshape(shape) for n in ("q", "k", "v"))
            attended = biased_attention(q, k, v, bias, 1.0)
            x = x + linear(attended.reshape(count, -1), w[p + "attention.o.weight"])
            normed = rms_norm(x, w[p + "mlp_norm.weight"], eps)
            up = linear(normed, w[p + "mlp.up.weight"])
            hidden = activate(linear(normed, w[p + "mlp.gate.weight"])) * up if config.gated_mlp else activate(up)
            x = x + linear(hidden, w[p + "mlp.down.weight"])
        return rms_norm(x, w["final_norm.weight"], eps)

    def _modernbert_states(self, tokens: Sequence[int]) -> np.ndarray:
        """``h = LayerNorm(word)``, then per layer ``h = h + o(attention(rope(q(x)), rope(k(x)), v(x)))`` with
        ``x = LayerNorm(h)`` (the first layer: ``x = h``), attention over every key on a global layer and the keys
        closer than the window on a local one, rotated with the layer's base, and ``h = h + down(gelu(gate(x)) *
        up(x))`` with ``x = LayerNorm(h)``; bias-free norms add no bias; the final norm last."""
        config, w = self.config, self.w
        count = len(tokens)
        zeros = np.zeros(config.hidden_size, dtype=F32)

        def norm(values: np.ndarray, name: str) -> np.ndarray:
            return layer_norm(values, w[name + ".weight"], zeros, config.rms_norm_eps)

        positions = np.arange(count, dtype=np.int64)
        global_freq = rope_inv_freq(config.head_dim, config.rope_theta)
        local_theta = config.rope_theta if config.local_rope_theta is None else config.local_rope_theta
        local_freq = rope_inv_freq(config.head_dim, local_theta)
        h = norm(w["token_embedding.weight"][np.asarray(tokens, dtype=np.int64)], "embedding_norm")
        shape = (count, config.heads, config.head_dim)
        for layer in range(config.layers):
            p = f"layers.{layer}."
            x = norm(h, p + "attention_norm") if layer else h
            window = config.window(layer)
            freq = global_freq if window is None else local_freq
            q = rope(linear(x, w[p + "attention.q.weight"]).reshape(shape), positions, freq)
            k = rope(linear(x, w[p + "attention.k.weight"]).reshape(shape), positions, freq)
            v = linear(x, w[p + "attention.v.weight"]).reshape(shape)
            attended = attention(q, k, v, scale=config.attention_scale, causal=False, window=window)
            h = h + linear(attended.reshape(count, -1), w[p + "attention.o.weight"])
            x = norm(h, p + "mlp_norm")
            gated = gelu(linear(x, w[p + "mlp.gate.weight"])) * linear(x, w[p + "mlp.up.weight"])
            h = h + linear(gated, w[p + "mlp.down.weight"])
        return norm(h, "final_norm")

    def classify(self, tokens: Sequence[int], types: Sequence[int] | None = None) -> np.ndarray:
        """A cross-encoder's logits: BERT's ``classifier(tanh(pooler(h[0])))``, tanh in double rounded to float32;
        ModernBERT's ``classifier(LayerNorm(gelu(pooler(pooled))))`` on the first state or the mean of every state
        (summed ascending in double, rounded, divided in float32)."""
        w = self.w
        if self.config.family == "modernbert":
            states = self.hidden_states(tokens)
            pooled = states[:1]
            if self.config.classifier_pooling == "mean":
                total = np.zeros(states.shape[1], dtype=F64)
                for row in states:
                    total += row
                pooled = (_round(total) / F32(len(states)))[None]
            head = gelu(linear(pooled, w["pooler.weight"]))
            zeros = np.zeros(self.config.hidden_size, dtype=F32)
            normed = layer_norm(head, w["pooler_norm.weight"], zeros, self.config.rms_norm_eps)
            return linear(normed, w["classifier.weight"], w["classifier.bias"]).reshape(-1)
        first = self.hidden_states(tokens, types)[:1]
        if self.config.family == "deberta":  # the context pooler: gelu, not tanh
            approximate = "tanh" if self.config.activation == "gelu_tanh" else "none"
            pooled = gelu(linear(first, w["pooler.weight"], w["pooler.bias"]), approximate)
            return linear(pooled, w["classifier.weight"], w["classifier.bias"]).reshape(-1)
        pooled = _round(tanh(linear(first, w["pooler.weight"], w["pooler.bias"])))
        return linear(pooled, w["classifier.weight"], w["classifier.bias"]).reshape(-1)

    def embed(self, tokens: Sequence[int]) -> np.ndarray:
        """The pooled embedding of ``tokens`` (already truncated, special tokens included): the first state for
        ``cls`` pooling, else the mean (summed over positions ascending in double, rounded, then divided in float32),
        then divided by its float32 norm (the square root of the sum of squares in double) when normalised."""
        states = self.hidden_states(tokens)
        if self.settings.get("pooling") == "cls":
            vector = states[0].copy()
        elif self.settings.get("pooling") == "last_token":
            vector = states[-1].copy()
        else:
            total = np.zeros(states.shape[1], dtype=F64)
            for row in states:
                total += row.astype(F64)
            vector = _round(total) / F32(states.shape[0])
        if self.settings.get("projection"):  # a Dense module: linear, then identity or tanh
            vector = linear(vector.reshape(1, -1), self.w["projection.weight"], self.w.get("projection.bias"))[0]
            if self.settings["projection"] == "tanh":
                vector = _round(tanh(vector.astype(F64)))
        if self.settings.get("normalize", True):
            squares = sequential_sum(vector.astype(F64) * vector.astype(F64))
            norm = F32(math.sqrt(squares))
            if norm > 0:
                vector = (vector / norm).astype(F32)
        return vector


class ReferenceTextToText:
    """The T5 encoder-decoder of :class:`etalii_dllm.seq2seq.TextToText`, recomputed from scratch for every token
    on this module's kernels: the source through :class:`ReferenceEncoder`, the decoder inputs ``[0, *answer]``
    through every decoder layer (self-attention over the inputs up to each position, cross-attention over the
    encoder states, the MLP), the final RMSNorm, the tied head's ``hidden ** -0.5`` and the LM head."""

    def __init__(self, config: TransformerConfig, tensors: Mapping[str, npt.ArrayLike], *, quantize: str | None = None):
        from dataclasses import replace

        self.config = config
        encoder_config = replace(config, decoder_layers=0, tie_word_embeddings=True)
        self.encoder = ReferenceEncoder(encoder_config, tensors, quantize=quantize)
        self.w: dict[str, Any] = {}
        for name in config.tensor_shapes():
            values = np.asarray(tensors[name], dtype=F32)
            matrix = name.endswith(_MATRICES) or name == "lm_head.weight"
            self.w[name] = Weight(values, quantize) if matrix else values
        embedding = self.w["token_embedding.weight"]
        tied = config.tie_word_embeddings
        self.head = Weight(embedding, quantize) if tied else self.w["lm_head.weight"]
        self.head_scale = F32(config.hidden_size**-0.5) if tied else None
        self.end_of_source = config.eos_token_ids[0] if config.eos_token_ids else 1

    @classmethod
    def from_engine_model(cls, model: Any) -> ReferenceTextToText:
        """The reference twin of an engine's text-to-text model (same weights and quantisation)."""
        return cls(model.config, model.tensors, quantize=model.quantization)

    def forward(self, source: Sequence[int], answer: Sequence[int]) -> np.ndarray:
        """The logits of the answer token after ``answer``, given ``source`` (ending with ``</s>``)."""
        config, w = self.config, self.w
        heads, head_dim, eps = config.heads, config.head_dim, config.rms_norm_eps
        states = self.encoder.hidden_states(source)
        inputs = [0, *answer]
        count, sources = len(inputs), len(source)
        table = w["decoder.relative_bias.weight"]
        h = w["token_embedding.weight"][np.asarray(inputs, dtype=np.int64)]
        shape = (count, heads, head_dim)
        zero = np.zeros((heads, 1, sources), dtype=F32)

        def activate(values: np.ndarray) -> np.ndarray:
            if config.activation == "relu":
                return np.where(values > 0, values, F32(0)).astype(F32)
            if config.activation == "silu":
                return silu(values)
            return gelu(values, "tanh" if config.activation == "gelu_tanh" else "none")

        for layer in range(config.decoder_layers):
            p = f"decoder.layers.{layer}."
            normed = rms_norm(h, w[p + "attention_norm.weight"], eps)
            q, k, v = (linear(normed, w[p + f"attention.{n}.weight"]).reshape(shape) for n in ("q", "k", "v"))
            attended = np.empty(shape, dtype=F32)
            for t in range(count):
                bias = np.empty((heads, 1, t + 1), dtype=F32)
                for j in range(t + 1):
                    bucket = t5_decoder_bucket(j - t, config.position_buckets, config.max_relative_positions)
                    bias[:, 0, j] = table[bucket]
                attended[t] = biased_attention(q[t : t + 1], k[: t + 1], v[: t + 1], bias, 1.0)[0]
            h = h + linear(attended.reshape(count, -1), w[p + "attention.o.weight"])
            normed = rms_norm(h, w[p + "cross_norm.weight"], eps)
            q = linear(normed, w[p + "cross.q.weight"]).reshape(shape)
            k, v = (linear(states, w[p + f"cross.{n}.weight"]).reshape(sources, heads, head_dim) for n in ("k", "v"))
            attended = np.empty(shape, dtype=F32)
            for t in range(count):
                attended[t] = biased_attention(q[t : t + 1], k, v, zero, 1.0)[0]
            h = h + linear(attended.reshape(count, -1), w[p + "cross.o.weight"])
            normed = rms_norm(h, w[p + "mlp_norm.weight"], eps)
            up = linear(normed, w[p + "mlp.up.weight"])
            hidden = activate(linear(normed, w[p + "mlp.gate.weight"])) * up if config.gated_mlp else activate(up)
            h = h + linear(hidden, w[p + "mlp.down.weight"])
        out = rms_norm(h[-1:], w["decoder.final_norm.weight"], eps)
        if self.head_scale is not None:
            out = (out * self.head_scale).astype(F32)
        return linear(out, self.head).reshape(-1)

    def generate(
        self, source: Sequence[int], max_tokens: int, sampler: Sampler, stop_tokens: Sequence[int] = ()
    ) -> tuple[list[int], list[np.ndarray]]:
        """Writes the answer to ``source`` (``</s>`` appended when it does not end with it) until a stop token or
        ``max_tokens``; the sampler sees the source as the prompt. Returns the tokens and their logits."""
        source = list(source)
        if not source or source[-1] != self.end_of_source:
            source.append(self.end_of_source)
        sampler.begin(source)
        tokens: list[int] = []
        steps: list[np.ndarray] = []
        while len(tokens) < max_tokens:
            logits = self.forward(source, tokens)
            token = sampler.sample(logits)
            steps.append(logits)
            if token in stop_tokens:
                break
            sampler.accept(token)
            tokens.append(token)
        return tokens, steps
