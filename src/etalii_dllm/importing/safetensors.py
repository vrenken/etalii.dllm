"""Reader for the safetensors format (https://github.com/huggingface/safetensors).

A safetensors file is an 8-byte little-endian header length, a JSON header mapping tensor names to
``{"dtype", "shape", "data_offsets"}`` (plus an optional ``__metadata__`` string map), and the raw little-endian
tensor bytes. Tensors are memory mapped, so opening a file reads only its header.
"""

from __future__ import annotations

import itertools
import json
import math
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# safetensors dtype name -> NumPy dtype of the stored bits. BF16 has no NumPy dtype; it is read as raw uint16.
_DTYPES: dict[str, np.dtype] = {
    "F64": np.dtype("<f8"),
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "BF16": np.dtype("<u2"),
    "I64": np.dtype("<i8"),
    "I32": np.dtype("<i4"),
    "I16": np.dtype("<i2"),
    "I8": np.dtype("i1"),
    "U64": np.dtype("<u8"),
    "U32": np.dtype("<u4"),
    "U16": np.dtype("<u2"),
    "U8": np.dtype("u1"),
    "BOOL": np.dtype("?"),
}

_MAX_HEADER = 100 * 1024 * 1024


class SafetensorsError(ValueError):
    """The file is not a valid safetensors file, or uses a feature this reader does not support."""


@dataclass(frozen=True)
class SafetensorsTensor:
    """One tensor in a safetensors file. ``raw`` holds the stored bits (BF16 as uint16)."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    raw: np.ndarray

    def to_float32(self) -> np.ndarray:
        """The values as float32. Exact for F32, F16 and BF16 (every value is representable)."""
        return to_float32(self.raw, self.dtype)


def to_float32(raw: np.ndarray, dtype: str) -> np.ndarray:
    """Converts stored bits to float32 without rounding. BF16 is the upper half of a float32, so shifting it left
    by 16 bits is exact; float16 values are all representable in float32. F64 is refused because narrowing it
    would round."""
    if dtype == "F32":
        return np.ascontiguousarray(raw, dtype="<f4")
    if dtype == "F16":
        return raw.astype("<f4")
    if dtype == "BF16":
        return (raw.astype("<u4") << np.uint32(16)).view("<f4")
    raise SafetensorsError(f"cannot convert {dtype} to float32 without loss")


class SafetensorsFile:
    """A memory-mapped safetensors file. Tensors are listed in data-offset order, which is the order they are
    stored in and does not depend on JSON key order."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        size = self.path.stat().st_size
        with self.path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise SafetensorsError(f"{self.path}: too short for a safetensors file")
            (header_length,) = struct.unpack("<Q", prefix)
            if header_length > _MAX_HEADER or 8 + header_length > size:
                raise SafetensorsError(f"{self.path}: invalid header length {header_length}")
            try:
                header = json.loads(stream.read(header_length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise SafetensorsError(f"{self.path}: header is not valid JSON") from error
        if not isinstance(header, dict):
            raise SafetensorsError(f"{self.path}: header is not a JSON object")

        metadata = header.pop("__metadata__", None) or {}
        if not isinstance(metadata, dict) or not all(isinstance(v, str) for v in metadata.values()):
            raise SafetensorsError(f"{self.path}: __metadata__ must map strings to strings")
        self.metadata: dict[str, str] = dict(metadata)

        data_start = 8 + header_length
        data_length = size - data_start
        mapped = (
            np.memmap(self.path, dtype=np.uint8, mode="r", offset=data_start, shape=(data_length,))
            if (data_length)
            else np.zeros(0, dtype=np.uint8)
        )

        entries: list[tuple[int, int, SafetensorsTensor]] = []
        for name, info in header.items():
            dtype = info.get("dtype")
            shape = tuple(info.get("shape", ()))
            begin, end = info.get("data_offsets", (None, None))
            if dtype not in _DTYPES:
                raise SafetensorsError(f"{self.path}: tensor {name!r} has unsupported dtype {dtype!r}")
            if not all(isinstance(d, int) and d >= 0 for d in shape):
                raise SafetensorsError(f"{self.path}: tensor {name!r} has invalid shape {shape}")
            numpy_dtype = _DTYPES[dtype]
            expected = math.prod(shape) * numpy_dtype.itemsize
            if (
                not (isinstance(begin, int) and isinstance(end, int))
                or end - begin != expected
                or not (0 <= begin <= end <= data_length)
            ):
                raise SafetensorsError(f"{self.path}: tensor {name!r} has invalid data offsets")
            raw = mapped[begin:end].view(numpy_dtype).reshape(shape)
            entries.append((begin, end, SafetensorsTensor(name, dtype, shape, raw)))

        entries.sort(key=lambda entry: (entry[0], entry[1], entry[2].name))
        for (_, previous_end, previous), (begin, _, tensor) in itertools.pairwise(entries):
            if begin < previous_end:
                raise SafetensorsError(f"{self.path}: tensors {previous.name!r} and {tensor.name!r} overlap")
        self._tensors = {tensor.name: tensor for _, _, tensor in entries}

    def __iter__(self) -> Iterator[SafetensorsTensor]:
        return iter(self._tensors.values())

    def __len__(self) -> int:
        return len(self._tensors)

    def __contains__(self, name: object) -> bool:
        return name in self._tensors

    def __getitem__(self, name: str) -> SafetensorsTensor:
        return self._tensors[name]

    def names(self) -> list[str]:
        return list(self._tensors)


def open_checkpoint(directory: str | Path) -> dict[str, SafetensorsTensor]:
    """All tensors of a Hugging Face checkpoint directory: ``model.safetensors``, or the shards listed in
    ``model.safetensors.index.json``. A tensor that appears twice is an error."""
    directory = Path(directory)
    index = directory / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        shards = sorted(set(weight_map.values()))
    elif (directory / "model.safetensors").exists():
        shards = ["model.safetensors"]
    else:
        shards = sorted(path.name for path in directory.glob("*.safetensors"))
    if not shards:
        raise SafetensorsError(f"{directory}: no safetensors files found")

    tensors: dict[str, SafetensorsTensor] = {}
    for shard in shards:
        for tensor in SafetensorsFile(directory / shard):
            if tensor.name in tensors:
                raise SafetensorsError(f"{directory}: tensor {tensor.name!r} appears in more than one shard")
            tensors[tensor.name] = tensor
    return tensors
