"""The ``model.dllm`` container: one file holding an imported model's weights and everything needed to run and
attribute it. The format is specified in ``docs/model-format.md``.

Writing is deterministic: the same header and tensors always give the same bytes (the header is canonical JSON
with sorted keys and no clock values; tensors are stored in a fixed order at fixed alignment). The SHA-256 of the
tensor data section is the model's fingerprint, used as the OpenAI-style ``system_fingerprint``.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import mmap
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
    # Only fine-tuned, adapted, edited, embedding and classification models carry these; files written by an import
    # carry a lineage.
    for section in ("fine_tuning", "adapter", "edits", "embedding", "classifier", "merge", "lineage"):
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


def file_sha256(path: str | Path) -> str:
    """The SHA-256 of a whole file: importing the same checkpoint gives the same model file, byte for byte, on every
    platform, so this identifies a ``model.dllm`` by itself (its data fingerprint covers only the weights)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


# -- lineage ----------------------------------------------------------------------------------------------------------
# How the weights were made, oldest step first: the import, then every adapter merge, fine-tune and edit. Each step
# after the import names the weights it started from (``input``) and every step but the last the weights it gave
# (``output``); the last step's output is the file's own fingerprint. Details are digests of the file's own sections
# (or of the step's record), so the lineage is short and an edited section shows.


def import_step(source: Mapping[str, Any]) -> dict[str, Any]:
    """The first step: the checkpoint files the weights were converted from (the ``source`` section)."""
    return {"step": "import", "format": source.get("format"), "source": _digest(source)}


def adapter_step(adapter: Mapping[str, Any]) -> dict[str, Any]:
    step = {"step": "adapter", "input": adapter["base_fingerprint"], "adapter": _digest(adapter)}
    if adapter.get("base_quantize"):  # merged into the dequantised base
        step["base_quantize"] = adapter["base_quantize"]
    return step


def fine_tune_step(fine_tuning: Mapping[str, Any]) -> dict[str, Any]:
    step = {
        "step": "fine_tune",
        "input": fine_tuning["base_fingerprint"],
        "data": fine_tuning["data_fingerprint"],
        "steps": fine_tuning["steps_completed"],
        "run": _digest(fine_tuning["run"]),
    }
    if fine_tuning["run"].get("objective", "lm") != "lm":  # preference tuning; language-model steps keep their bytes
        step["objective"] = fine_tuning["run"]["objective"]
    if fine_tuning["run"].get("base_quantize"):  # LoRA on a quantised base: the dequantised weights changed too
        step["base_quantize"] = fine_tuning["run"]["base_quantize"]
    if fine_tuning.get("distillation"):  # the data are a teacher's answers
        step["teacher"] = fine_tuning["distillation"]["teacher"]
    return step


def edit_step(edit: Mapping[str, Any]) -> dict[str, Any]:
    return {"step": "edit", "input": edit["base_fingerprint"], "method": edit["method"], "edit": _digest(edit)}


def lineage(header: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The steps a header records. Files written before lineages existed get the steps their other sections show
    (the import, then any adapter, fine-tune and edits), each step's output inferred from the next one's input."""
    if header.get("lineage") is not None:
        return [dict(step) for step in header["lineage"]]
    steps = [import_step(header.get("source") or {})]
    if header.get("adapter"):
        steps.append(adapter_step(header["adapter"]))
    if header.get("fine_tuning"):
        steps.append(fine_tune_step(header["fine_tuning"]))
    steps.extend(edit_step(edit) for edit in header.get("edits") or [])
    for step, following in itertools.pairwise(steps):
        step["output"] = following["input"]
    return steps


def extend_lineage(steps: list[dict[str, Any]], base_fingerprint: str, step: Mapping[str, Any]) -> list[dict[str, Any]]:
    """``steps`` (the lineage of the weights ``base_fingerprint``) followed by ``step``."""
    if not steps:
        return [dict(step)]
    return [*steps[:-1], {**steps[-1], "output": base_fingerprint}, dict(step)]


def lineage_problems(steps: list[Mapping[str, Any]]) -> list[str]:
    """Why the steps do not form one chain (each step starting from the weights the step before gave), if they do
    not."""
    problems = []
    if not steps or steps[0].get("step") != "import":
        problems.append("the lineage does not start with an import")
    for index in range(1, len(steps)):
        before, step = steps[index - 1], steps[index]
        if step.get("input") != before.get("output"):
            problems.append(
                f"step {index} ({step.get('step')}) starts from {step.get('input')}, "
                f"but step {index - 1} gave {before.get('output')}"
            )
    return problems


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
        with self.path.open("rb") as stream:
            self._mmap = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        self._data_start = data_start
        self._data = np.frombuffer(self._mmap, dtype=np.uint8, count=data_length, offset=data_start)
        self._ranges: dict[str, tuple[int, int]] = {}
        self.tensors: dict[str, np.ndarray] = {}
        for entry in self.header["tensors"]:
            begin, nbytes = entry["offset"], entry["nbytes"]
            if entry["dtype"] != "F32" or begin % ALIGNMENT or begin + nbytes > data_length:
                raise ModelFileError(f"{self.path}: tensor {entry['name']!r} has an invalid layout")
            self.tensors[entry["name"]] = self._data[begin : begin + nbytes].view(_DTYPE).reshape(entry["shape"])
            self._ranges[entry["name"]] = (begin, begin + nbytes)
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
    def lineage(self) -> list[dict[str, Any]]:
        """How the weights were made, oldest step first, the last step's output being this file's fingerprint."""
        steps = lineage(self.header)
        steps[-1]["output"] = self.fingerprint
        return steps

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
    def classifier(self) -> dict[str, Any] | None:
        """A sequence-classification model's ``labels``, the ``activation`` of its scores (``none`` or ``sigmoid``)
        and ``max_tokens`` of a pair, when the file was imported from a cross-encoder."""
        return self.header.get("classifier")

    @property
    def edits(self) -> list[dict[str, Any]]:
        """The model edits (``dllm edit``) applied to the weights, oldest first; empty for unedited files."""
        return list(self.header.get("edits") or [])

    def release(self, name: str | None = None) -> None:
        """Lets the operating system drop the pages of tensor ``name`` (all tensor data when ``None``) from this
        process's memory, once a copy (packed, quantised or on the GPU) has been made. Nothing changes: the pages are
        read back from the file if the tensor is used again. A no-op where ``madvise`` is unavailable (Windows)."""
        begin, end = self._ranges[name] if name is not None else (0, len(self._data))
        self._drop(begin, end)

    def _drop(self, begin: int, end: int) -> None:
        if not hasattr(self._mmap, "madvise") or not hasattr(mmap, "MADV_DONTNEED"):
            return
        page = mmap.PAGESIZE
        end = min(end, len(self._data))
        first = -(-(self._data_start + begin) // page) * page  # whole pages inside the range only
        last = (self._data_start + end) // page * page
        if last > first:
            self._mmap.madvise(mmap.MADV_DONTNEED, first, last - first)

    def verify(self) -> None:
        """Re-hashes the tensor data and checks it against the recorded fingerprint. The pages read are released
        as it goes, so checking a large file does not keep it resident."""
        digest = hashlib.sha256()
        step = 64 * 1024 * 1024
        for begin in range(0, len(self._data), step):
            digest.update(self._data[begin : begin + step])
            self._drop(begin, begin + step)
        if digest.hexdigest() != self.fingerprint:
            raise ModelFileError(f"{self.path}: tensor data does not match the fingerprint (corrupt file?)")
