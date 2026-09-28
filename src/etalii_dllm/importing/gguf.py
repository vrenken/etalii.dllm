"""Reader for GGUF files (llama.cpp, Ollama), versions 2 and 3, little-endian.

Layout: ``GGUF`` magic, version, tensor count, metadata count, the metadata key/value pairs, the tensor infos
(name, dimensions innermost first, GGML type, offset), then the tensor data aligned to ``general.alignment``
(default 32). Tensor data is memory mapped; quantised tensors are dequantised to float32 on request with the block
formats in :mod:`etalii_dllm.importing.quants`.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from etalii_dllm.importing.quants import BLOCK_FORMATS, dequantize

MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32

# GGML type id -> (name, NumPy dtype) for the unquantised types.
_PLAIN_TYPES: dict[int, tuple[str, np.dtype]] = {
    0: ("F32", np.dtype("<f4")),
    1: ("F16", np.dtype("<f2")),
    24: ("I8", np.dtype("i1")),
    25: ("I16", np.dtype("<i2")),
    26: ("I32", np.dtype("<i4")),
    27: ("I64", np.dtype("<i8")),
    28: ("F64", np.dtype("<f8")),
    30: ("BF16", np.dtype("<u2")),
}

# Metadata value type id -> struct format for the scalar types.
_SCALARS: dict[int, str] = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}
_STRING = 8
_ARRAY = 9


class GgufError(ValueError):
    """The file is not a valid GGUF file, or uses a feature this reader does not support."""


class _Cursor:
    def __init__(self, data: memoryview, path: Path) -> None:
        self.data = data
        self.offset = 0
        self.path = path

    def take(self, count: int) -> memoryview:
        if self.offset + count > len(self.data):
            raise GgufError(f"{self.path}: unexpected end of file")
        chunk = self.data[self.offset : self.offset + count]
        self.offset += count
        return chunk

    def unpack(self, fmt: str) -> Any:
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        length = self.unpack("<Q")
        try:
            return bytes(self.take(length)).decode("utf-8")
        except UnicodeDecodeError as error:
            raise GgufError(f"{self.path}: string is not valid UTF-8") from error

    def value(self, value_type: int) -> Any:
        if value_type in _SCALARS:
            return self.unpack(_SCALARS[value_type])
        if value_type == _STRING:
            return self.string()
        if value_type == _ARRAY:
            item_type = self.unpack("<I")
            count = self.unpack("<Q")
            if item_type in _SCALARS:
                fmt = _SCALARS[item_type]
                size = struct.calcsize(fmt)
                return [item[0] for item in struct.iter_unpack(fmt, self.take(size * count))]
            return [self.value(item_type) for _ in range(count)]
        raise GgufError(f"{self.path}: unknown metadata value type {value_type}")


@dataclass(frozen=True)
class GgufTensor:
    """One tensor. ``shape`` is in NumPy order (outermost first), i.e. GGUF's dimensions reversed, so a
    ``[out, in]`` weight matrix has the same shape as in PyTorch."""

    name: str
    type_id: int
    type_name: str
    shape: tuple[int, ...]
    raw: np.ndarray

    def to_float32(self) -> np.ndarray:
        """The values as float32: exact for F32, F16 and BF16; for quantised types the reference dequantisation,
        which is deterministic but is of course only as precise as the quantised data."""
        count = math.prod(self.shape)
        if self.type_id in BLOCK_FORMATS:
            return dequantize(self.raw, BLOCK_FORMATS[self.type_id], count).reshape(self.shape)
        if self.type_name == "F32":
            return np.ascontiguousarray(self.raw, dtype="<f4")
        if self.type_name == "F16":
            return self.raw.astype("<f4")
        if self.type_name == "BF16":
            return (self.raw.astype("<u4") << np.uint32(16)).view("<f4")
        raise GgufError(f"tensor {self.name!r}: cannot convert {self.type_name} to float32 without loss")

    @property
    def lossless(self) -> bool:
        return self.type_name in ("F32", "F16", "BF16")


class GgufFile:
    """A memory-mapped GGUF file: ``metadata`` in file order and tensors in file order."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        mapped = np.memmap(self.path, dtype=np.uint8, mode="r")
        cursor = _Cursor(memoryview(mapped), self.path)
        if bytes(cursor.take(4)) != MAGIC:
            raise GgufError(f"{self.path}: not a GGUF file")
        self.version = cursor.unpack("<I")
        if self.version not in (2, 3):
            if struct.unpack(">I", struct.pack("<I", self.version))[0] in (2, 3):
                raise GgufError(f"{self.path}: big-endian GGUF files are not supported")
            raise GgufError(f"{self.path}: unsupported GGUF version {self.version}")
        tensor_count = cursor.unpack("<Q")
        metadata_count = cursor.unpack("<Q")

        self.metadata: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = cursor.string()
            if key in self.metadata:
                raise GgufError(f"{self.path}: duplicate metadata key {key!r}")
            self.metadata[key] = cursor.value(cursor.unpack("<I"))

        infos = []
        for _ in range(tensor_count):
            name = cursor.string()
            dimensions = [cursor.unpack("<Q") for _ in range(cursor.unpack("<I"))]
            infos.append((name, tuple(reversed(dimensions)), cursor.unpack("<I"), cursor.unpack("<Q")))

        self.alignment = int(self.metadata.get("general.alignment", DEFAULT_ALIGNMENT))
        if self.alignment <= 0 or self.alignment & (self.alignment - 1):
            raise GgufError(f"{self.path}: general.alignment {self.alignment} is not a power of two")
        data_start = -(-cursor.offset // self.alignment) * self.alignment

        self._tensors: dict[str, GgufTensor] = {}
        for name, shape, type_id, offset in infos:
            count = math.prod(shape)
            if type_id in BLOCK_FORMATS:
                block = BLOCK_FORMATS[type_id]
                if count % block.block_size:
                    raise GgufError(f"{self.path}: tensor {name!r} is not a whole number of {block.name} blocks")
                type_name, nbytes, dtype = block.name, count // block.block_size * block.block_bytes, None
            elif type_id in _PLAIN_TYPES:
                type_name, dtype = _PLAIN_TYPES[type_id]
                nbytes = count * dtype.itemsize
            else:
                raise GgufError(f"{self.path}: tensor {name!r} has unsupported GGML type {type_id}")
            begin = data_start + offset
            if offset % self.alignment or begin + nbytes > len(mapped):
                raise GgufError(f"{self.path}: tensor {name!r} has an invalid offset")
            if name in self._tensors:
                raise GgufError(f"{self.path}: duplicate tensor {name!r}")
            raw = mapped[begin : begin + nbytes]
            raw = raw if dtype is None else raw.view(dtype).reshape(shape)
            self._tensors[name] = GgufTensor(name, type_id, type_name, shape, raw)

    def __iter__(self) -> Iterator[GgufTensor]:
        return iter(self._tensors.values())

    def __len__(self) -> int:
        return len(self._tensors)

    def __contains__(self, name: object) -> bool:
        return name in self._tensors

    def __getitem__(self, name: str) -> GgufTensor:
        return self._tensors[name]
