"""Dequantisation of the GGML block formats found in GGUF files.

Each function takes the raw bytes of ``n`` blocks as a ``uint8`` array of shape ``[n, block_bytes]`` and returns
``float32`` values of shape ``[n, block_size]``. Only elementwise float32 operations are used, in a fixed order that
matches llama.cpp's reference implementation (``gguf-py``), so the result is bit-identical to it on every machine.
Scales stored as float16 are widened exactly; ``d * q`` products are exact for the 4- to 8-bit formats; the
K-quant formats round once per multiply or subtract, in the reference order.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

QK_K = 256


def _f16(column: np.ndarray) -> np.ndarray:
    """Two bytes per block, little-endian float16, widened exactly to float32 with shape ``[n, 1]``."""
    return np.ascontiguousarray(column).view("<f2").astype("<f4")


def _low_high(qs: np.ndarray) -> np.ndarray:
    """Split packed nibbles: low nibbles are the first half of the block, high nibbles the second half."""
    return np.concatenate([qs & np.uint8(0x0F), qs >> np.uint8(4)], axis=1)


def _q4_0(blocks: np.ndarray) -> np.ndarray:
    d = _f16(blocks[:, 0:2])
    q = _low_high(blocks[:, 2:18]).astype(np.int8) - np.int8(8)
    return d * q.astype(np.float32)


def _q4_1(blocks: np.ndarray) -> np.ndarray:
    d = _f16(blocks[:, 0:2])
    m = _f16(blocks[:, 2:4])
    q = _low_high(blocks[:, 4:20]).astype(np.float32)
    return d * q + m


def _q5_high_bits(qh_bytes: np.ndarray) -> np.ndarray:
    """Fifth bit of each of the 32 values, from a little-endian uint32 per block, shifted to bit 4."""
    qh = np.ascontiguousarray(qh_bytes).view("<u4")
    shifts = np.arange(32, dtype=np.uint32).reshape(1, 32)
    return (((qh >> shifts) & np.uint32(1)) << np.uint32(4)).astype(np.uint8)


def _q5_0(blocks: np.ndarray) -> np.ndarray:
    d = _f16(blocks[:, 0:2])
    q = (_low_high(blocks[:, 6:22]) | _q5_high_bits(blocks[:, 2:6])).astype(np.int8) - np.int8(16)
    return d * q.astype(np.float32)


def _q5_1(blocks: np.ndarray) -> np.ndarray:
    d = _f16(blocks[:, 0:2])
    m = _f16(blocks[:, 2:4])
    q = (_low_high(blocks[:, 8:24]) | _q5_high_bits(blocks[:, 4:8])).astype(np.float32)
    return d * q + m


def _q8_0(blocks: np.ndarray) -> np.ndarray:
    d = _f16(blocks[:, 0:2])
    q = np.ascontiguousarray(blocks[:, 2:34]).view(np.int8).astype(np.float32)
    return d * q


def _k_scales(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The eight 6-bit scales and mins packed into 12 bytes (``get_scale_min_k4`` in ggml)."""
    scales = scales.astype(np.uint8)
    d = scales[:, 0:4]
    m = scales[:, 4:8]
    extra = scales[:, 8:12]
    sc = np.concatenate([d & np.uint8(0x3F), (extra & np.uint8(0x0F)) | ((d >> np.uint8(6)) << np.uint8(4))], axis=1)
    mn = np.concatenate([m & np.uint8(0x3F), (extra >> np.uint8(4)) | ((m >> np.uint8(6)) << np.uint8(4))], axis=1)
    return sc.astype(np.float32), mn.astype(np.float32)


def _k_nibbles(qs: np.ndarray) -> np.ndarray:
    """``[n, 128]`` bytes to ``[n, 8, 32]`` 4-bit values: each 32-byte group gives its low nibbles to sub-block
    ``2g`` and its high nibbles to sub-block ``2g + 1``."""
    groups = qs.reshape(qs.shape[0], 4, 1, 32)
    shifted = groups >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)
    return (shifted & np.uint8(0x0F)).reshape(qs.shape[0], 8, 32)


def _q4_k(blocks: np.ndarray) -> np.ndarray:
    n = blocks.shape[0]
    d = _f16(blocks[:, 0:2])
    dmin = _f16(blocks[:, 2:4])
    sc, mn = _k_scales(blocks[:, 4:16])
    scale = (d * sc).reshape(n, 8, 1)
    offset = (dmin * mn).reshape(n, 8, 1)
    q = _k_nibbles(blocks[:, 16:144]).astype(np.float32)
    return (scale * q - offset).reshape(n, QK_K)


def _q5_k(blocks: np.ndarray) -> np.ndarray:
    n = blocks.shape[0]
    d = _f16(blocks[:, 0:2])
    dmin = _f16(blocks[:, 2:4])
    sc, mn = _k_scales(blocks[:, 4:16])
    scale = (d * sc).reshape(n, 8, 1)
    offset = (dmin * mn).reshape(n, 8, 1)
    qh = blocks[:, 16:48].reshape(n, 1, 32) >> np.arange(8, dtype=np.uint8).reshape(1, 8, 1)
    q = _k_nibbles(blocks[:, 48:176]) | ((qh & np.uint8(1)) << np.uint8(4))
    return (scale * q.astype(np.float32) - offset).reshape(n, QK_K)


def _q6_k(blocks: np.ndarray) -> np.ndarray:
    n = blocks.shape[0]
    ql = blocks[:, 0:128].reshape(n, 2, 1, 64) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)
    ql = (ql & np.uint8(0x0F)).reshape(n, 8, 32)
    qh = blocks[:, 128:192].reshape(n, 2, 1, 32) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)
    qh = (qh & np.uint8(0x03)).reshape(n, 8, 32)
    q = (ql | (qh << np.uint8(4))).astype(np.int8) - np.int8(32)
    scales = np.ascontiguousarray(blocks[:, 192:208]).view(np.int8).astype(np.float32)
    d = _f16(blocks[:, 208:210])
    scale = (d * scales).reshape(n, 16, 1)
    return (scale * q.reshape(n, 16, 16).astype(np.float32)).reshape(n, QK_K)


@dataclass(frozen=True)
class BlockFormat:
    name: str
    block_size: int
    block_bytes: int
    dequantize: Callable[[np.ndarray], np.ndarray]


# GGML type id -> block format, for the quantised types we can read.
BLOCK_FORMATS: dict[int, BlockFormat] = {
    2: BlockFormat("Q4_0", 32, 18, _q4_0),
    3: BlockFormat("Q4_1", 32, 20, _q4_1),
    6: BlockFormat("Q5_0", 32, 22, _q5_0),
    7: BlockFormat("Q5_1", 32, 24, _q5_1),
    8: BlockFormat("Q8_0", 32, 34, _q8_0),
    12: BlockFormat("Q4_K", QK_K, 144, _q4_k),
    13: BlockFormat("Q5_K", QK_K, 176, _q5_k),
    14: BlockFormat("Q6_K", QK_K, 210, _q6_k),
}


def dequantize(raw: np.ndarray, format: BlockFormat, count: int) -> np.ndarray:
    """``count`` float32 values from the raw bytes of a block-quantised tensor."""
    if count % format.block_size:
        raise ValueError(f"{format.name}: {count} values is not a whole number of blocks")
    blocks = np.asarray(raw, dtype=np.uint8).reshape(count // format.block_size, format.block_bytes)
    return np.ascontiguousarray(format.dequantize(blocks), dtype="<f4").reshape(count)
