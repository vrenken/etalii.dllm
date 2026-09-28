"""The tensor type shared by the kernels and the models: contiguous, 64-byte aligned float32 storage.

A :class:`Tensor` is a thin, read-only view over a NumPy array. It pins down the properties the C++ kernels rely
on (C-contiguous, float32, little-endian, aligned to a cache line) so model code never passes a strided or
misaligned buffer by accident. Arithmetic lives in :mod:`etalii_dllm.numerics`; ``Tensor`` only holds data.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

ALIGNMENT = 64
DTYPE = np.dtype("<f4")


def _is_aligned(array: np.ndarray) -> bool:
    return array.ctypes.data % ALIGNMENT == 0


def _aligned_empty(shape: tuple[int, ...]) -> np.ndarray:
    count = math.prod(shape)
    raw = np.empty(count * DTYPE.itemsize + ALIGNMENT, dtype=np.uint8)
    offset = (-raw.ctypes.data) % ALIGNMENT
    return raw[offset : offset + count * DTYPE.itemsize].view(DTYPE).reshape(shape)


class Tensor:
    """Immutable n-dimensional float32 tensor with contiguous, 64-byte aligned storage."""

    __slots__ = ("_data",)

    def __init__(self, values: npt.ArrayLike | Tensor) -> None:
        if isinstance(values, Tensor):
            self._data = values._data
            return
        array = np.asarray(values)
        if array.dtype == DTYPE and array.flags.c_contiguous and _is_aligned(array):
            data = array.view()
        else:
            data = _aligned_empty(array.shape)
            data[...] = array
        data.flags.writeable = False
        self._data = data

    @classmethod
    def zeros(cls, shape: Sequence[int]) -> Tensor:
        data = _aligned_empty(tuple(shape))
        data.fill(0)
        return cls(data)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    @property
    def strides(self) -> tuple[int, ...]:
        """Strides in elements (not bytes), row-major."""
        return tuple(s // DTYPE.itemsize for s in self._data.strides)

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def size(self) -> int:
        return self._data.size

    def numpy(self) -> npt.NDArray[np.float32]:
        """The underlying read-only array (no copy)."""
        return self._data

    def __array__(self, dtype: npt.DTypeLike | None = None, copy: bool | None = None) -> np.ndarray:
        if dtype is not None and np.dtype(dtype) != DTYPE:
            return self._data.astype(dtype)
        return self._data.copy() if copy else self._data

    def reshape(self, *shape: int) -> Tensor:
        """A view with a new shape; storage stays shared, contiguous and aligned."""
        return Tensor(self._data.reshape(shape))

    def __getitem__(self, index: object) -> Tensor:
        """Indexing copies when the result is not contiguous, so a Tensor is always kernel-ready."""
        return Tensor(self._data[index])

    def __len__(self) -> int:
        return len(self._data)

    def fingerprint(self) -> str:
        """SHA-256 of the exact bits and the shape. Equal fingerprints mean bit-identical tensors."""
        digest = hashlib.sha256(np.asarray(self.shape, dtype="<i8").tobytes())
        digest.update(b"|")
        digest.update(self._data.tobytes())
        return digest.hexdigest()

    def __eq__(self, other: object) -> bool:
        """Bitwise equality (NaN payloads included), not numeric closeness."""
        if not isinstance(other, Tensor):
            return NotImplemented
        return self.shape == other.shape and self._data.tobytes() == other._data.tobytes()

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape})"
