"""The ``model.dllm`` container: one file holding an imported model's weights and everything needed to run and
attribute it. The format is specified in ``docs/model-format.md``.

Writing is deterministic: the same header and tensors always give the same bytes (the header is canonical JSON
with sorted keys and no clock values; tensors are stored in a fixed order at fixed alignment). The SHA-256 of the
tensor data section is the model's fingerprint, used as the OpenAI-style ``system_fingerprint``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from etalii_dllm.architecture import TransformerConfig

MAGIC = b"DLLM"
FORMAT_VERSION = 1
ALIGNMENT = 64
_PREFIX = struct.Struct("<4sIQ")  # magic, format version, header length
_FINGERPRINT_PLACEHOLDER = "0" * 64
_DTYPE = np.dtype("<f4")


class ModelFileError(ValueError):
    """The file is not a valid model.dllm file, or its data does not match its fingerprint."""


def _natural_key(name: str) -> tuple[Any, ...]:
    """Orders ``layers.2`` before ``layers.10``; a total order because the name itself breaks every tie."""
    parts = re.split(r"(\d+)", name)
    return (*((0, int(p), "") if p.isdigit() else (1, 0, p) for p in parts), name)


def tensor_order(names: Iterable[str]) -> list[str]:
    """The order tensors are stored in: natural order of their names."""
    return sorted(names, key=_natural_key)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def _pad(length: int) -> int:
    return -length % ALIGNMENT


@dataclass(frozen=True)
class TensorSource:
    """A tensor to write: its shape and a function producing its float32 values (called once, at write time, so
    large models are converted one tensor at a time)."""

    shape: tuple[int, ...]
    load: Callable[[], np.ndarray]
    source_dtype: str = "F32"


def write_model_file(
    path: str | Path,
    config: TransformerConfig,
    tensors: Mapping[str, TensorSource],
    metadata: Mapping[str, Any],
) -> str:
    """Writes a model file and returns its fingerprint. ``metadata`` holds the ``source``, ``licence``,
    ``tokenizer`` and ``chat_template`` sections and, for fine-tuned models, ``fine_tuning`` (see the format
    document)."""
    expected = config.tensor_shapes()
    if set(tensors) != set(expected):
        missing = sorted(set(expected) - set(tensors))
        extra = sorted(set(tensors) - set(expected))
        raise ModelFileError(f"tensors do not match the architecture (missing {missing}, unexpected {extra})")

    entries = []
    offset = 0
    for name in tensor_order(tensors):
        shape = tuple(int(d) for d in tensors[name].shape)
        if shape != expected[name]:
            raise ModelFileError(f"tensor {name!r} has shape {shape}, expected {expected[name]}")
        nbytes = math.prod(shape) * _DTYPE.itemsize
        entries.append(
            {
                "name": name,
                "shape": list(shape),
                "dtype": "F32",
                "offset": offset,
                "nbytes": nbytes,
                "source_dtype": tensors[name].source_dtype,
            }
        )
        offset += nbytes + _pad(nbytes)

    header = {
        "format": "dllm",
        "format_version": FORMAT_VERSION,
        "architecture": config.to_dict(),
        "tensors": entries,
        "fingerprint": _FINGERPRINT_PLACEHOLDER,
        **{key: metadata.get(key) for key in ("source", "licence", "tokenizer", "chat_template")},
    }
    # Only fine-tuned, adapted, edited and embedding models carry these.
    for section in ("fine_tuning", "adapter", "edits", "embedding"):
        if metadata.get(section) is not None:
            header[section] = metadata[section]
    header_bytes = canonical_json(header)
    header_bytes += b" " * _pad(_PREFIX.size + len(header_bytes))
    path = Path(path)

    digest = hashlib.sha256()
    with path.open("wb") as stream:
        stream.write(_PREFIX.pack(MAGIC, FORMAT_VERSION, len(header_bytes)))
        stream.write(header_bytes)
        for entry in entries:
            values = np.ascontiguousarray(tensors[entry["name"]].load(), dtype=_DTYPE)
            if values.shape != tuple(entry["shape"]):
                raise ModelFileError(f"tensor {entry['name']!r} loaded with shape {values.shape}")
            data = values.tobytes() + b"\0" * _pad(entry["nbytes"])
            digest.update(data)
            stream.write(data)
        fingerprint = digest.hexdigest()
        _replace_placeholder(stream, header_bytes, fingerprint)
    return fingerprint


def data_fingerprint(tensors: Mapping[str, np.ndarray]) -> str:
    """The fingerprint :func:`write_model_file` would record for ``tensors``: the SHA-256 of their float32 bytes in
    tensor order, each padded to the alignment. Identifies weights that are only held in memory."""
    digest = hashlib.sha256()
    for name in tensor_order(tensors):
        data = np.ascontiguousarray(tensors[name], dtype=_DTYPE).tobytes()
        digest.update(data + b"\0" * _pad(len(data)))
    return digest.hexdigest()


def _replace_placeholder(stream: BinaryIO, header_bytes: bytes, fingerprint: str) -> None:
    """The fingerprint has a fixed width, so it is patched into the header in place once the data is hashed."""
    marker = b'"fingerprint":"' + _FINGERPRINT_PLACEHOLDER.encode() + b'"'
    position = header_bytes.index(marker) + len(b'"fingerprint":"')
    stream.seek(_PREFIX.size + position)
    stream.write(fingerprint.encode())


class ModelFile:
    """A memory-mapped model file. Tensors are read-only float32 views aligned to 64 bytes."""

    def __init__(self, path: str | Path, verify: bool = True) -> None:
        self.path = Path(path)
        with self.path.open("rb") as stream:
            prefix = stream.read(_PREFIX.size)
            if len(prefix) != _PREFIX.size:
                raise ModelFileError(f"{self.path}: not a model.dllm file")
            magic, version, header_length = _PREFIX.unpack(prefix)
            if magic != MAGIC:
                raise ModelFileError(f"{self.path}: not a model.dllm file")
            if version != FORMAT_VERSION:
                raise ModelFileError(f"{self.path}: unsupported format version {version}")
            try:
                self.header: dict[str, Any] = json.loads(stream.read(header_length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ModelFileError(f"{self.path}: header is not valid JSON") from error

        if not isinstance(self.header, dict) or not isinstance(self.header.get("architecture"), dict):
            raise ModelFileError(f"{self.path}: header has no architecture")
        self.config = TransformerConfig.from_dict(self.header["architecture"])
        data_start = _PREFIX.size + header_length
        data_length = self.path.stat().st_size - data_start
        self._data = np.memmap(self.path, dtype=np.uint8, mode="r", offset=data_start, shape=(data_length,))
        self.tensors: dict[str, np.ndarray] = {}
        for entry in self.header["tensors"]:
            begin, nbytes = entry["offset"], entry["nbytes"]
            if entry["dtype"] != "F32" or begin % ALIGNMENT or begin + nbytes > data_length:
                raise ModelFileError(f"{self.path}: tensor {entry['name']!r} has an invalid layout")
            self.tensors[entry["name"]] = self._data[begin : begin + nbytes].view(_DTYPE).reshape(entry["shape"])
        expected = self.config.tensor_shapes()
        if set(self.tensors) != set(expected):
            raise ModelFileError(f"{self.path}: tensors do not match the architecture")
        for name, shape in expected.items():
            if self.tensors[name].shape != tuple(shape):
                raise ModelFileError(
                    f"{self.path}: tensor {name!r} has shape {self.tensors[name].shape}, expected {shape}"
                )
        if verify:
            self.verify()

    @property
    def fingerprint(self) -> str:
        return str(self.header["fingerprint"])

    @property
    def source(self) -> dict[str, Any]:
        return self.header.get("source") or {}

    @property
    def licence(self) -> dict[str, Any]:
        return self.header.get("licence") or {}

    @property
    def tokenizer(self) -> dict[str, Any] | None:
        return self.header.get("tokenizer")

    @property
    def chat_template(self) -> str | None:
        return self.header.get("chat_template")

    @property
    def fine_tuning(self) -> dict[str, Any] | None:
        return self.header.get("fine_tuning")

    @property
    def adapter(self) -> dict[str, Any] | None:
        """The LoRA adapter merged into the weights, when the file was written by an adapter import."""
        return self.header.get("adapter")

    @property
    def embedding(self) -> dict[str, Any] | None:
        """How an embedding model pools and normalises its hidden states (``pooling``: ``mean`` or ``last_token``,
        ``normalize``, ``prompts``), when the file was imported from a sentence-transformers model."""
        return self.header.get("embedding")

    @property
    def edits(self) -> list[dict[str, Any]]:
        """The model edits (``dllm edit``) applied to the weights, oldest first; empty for unedited files."""
        return list(self.header.get("edits") or [])

    def verify(self) -> None:
        """Re-hashes the tensor data and checks it against the recorded fingerprint."""
        digest = hashlib.sha256()
        step = 64 * 1024 * 1024
        for begin in range(0, len(self._data), step):
            digest.update(self._data[begin : begin + step])
        if digest.hexdigest() != self.fingerprint:
            raise ModelFileError(f"{self.path}: tensor data does not match the fingerprint (corrupt file?)")
