"""Reproducible fine-tuning: full-parameter AdamW training of an imported model on a fixed data order, with
checkpoints that resume bit for bit.

A step takes ``batch_size`` windows from :class:`~etalii_dllm.training.data.TrainingData`, computes each window's
loss and gradients on its own (so a window's gradients do not depend on the rest of the batch), sums the gradients
elementwise in batch order and applies one :class:`~etalii_dllm.training.optimizer.AdamW` update. The loss is the
mean next-token cross-entropy over all targets of the batch.

Checkpoint file (``.dllmckpt``): the magic ``DLLMCKPT``, a uint32 format version, a uint64 header length, a canonical
JSON header (run settings, step, loss history, base model metadata, tensor index, fingerprint) padded to 64 bytes,
then the parameters and both AdamW moments as aligned little-endian float32. As with ``model.dllm``, nothing comes
from a clock or the environment, so equal runs write byte-identical checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from etalii_dllm.architecture import TransformerConfig
from etalii_dllm.modelfile import ModelFile, TensorSource, canonical_json, tensor_order, write_model_file
from etalii_dllm.numerics import FloatArray
from etalii_dllm.training.backprop import DecoderGradients
from etalii_dllm.training.data import TrainingData
from etalii_dllm.training.optimizer import AdamW, AdamWConfig

CHECKPOINT_MAGIC = b"DLLMCKPT"
CHECKPOINT_VERSION = 1
_PREFIX = struct.Struct("<8sIQ")
_ALIGNMENT = 64
_PLACEHOLDER = "0" * 64
_DTYPE = np.dtype("<f4")
_METADATA_KEYS = ("source", "licence", "tokenizer", "chat_template")


class CheckpointError(ValueError):
    """The file is not a valid checkpoint, or it does not match the data it is resumed with."""


@dataclass(frozen=True)
class RunConfig:
    steps: int
    batch_size: int = 8
    sequence_length: int = 128
    seed: int = 0
    optimizer: AdamWConfig = field(default_factory=AdamWConfig)

    def __post_init__(self) -> None:
        if self.steps < 1 or self.batch_size < 1 or self.sequence_length < 1:
            raise ValueError("steps, batch_size and sequence_length must be positive")

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["optimizer"] = self.optimizer.to_dict()
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> RunConfig:
        values = dict(values)
        values["optimizer"] = AdamWConfig(**values["optimizer"])
        return cls(**values)


@dataclass(frozen=True)
class StepResult:
    step: int
    loss: float
    learning_rate: float
    gradient_norm: float


class FineTuner:
    """State of one fine-tuning run: parameters, optimizer moments, step and loss history."""

    def __init__(
        self,
        config: TransformerConfig,
        params: Mapping[str, np.ndarray],
        data: TrainingData,
        run: RunConfig,
        *,
        base_fingerprint: str,
        metadata: Mapping[str, Any],
    ) -> None:
        if data.sequence_length != run.sequence_length:
            raise ValueError("the data was windowed for a different sequence length")
        if run.sequence_length > config.context_length:
            raise ValueError(f"sequence_length exceeds the model's context length ({config.context_length})")
        shapes = config.tensor_shapes()
        if set(params) != set(shapes):
            raise ValueError("parameters do not match the architecture")
        self.config = config
        self.data = data
        self.run = run
        self.base_fingerprint = base_fingerprint
        self.metadata = {key: metadata.get(key) for key in _METADATA_KEYS}
        self.params: dict[str, FloatArray] = {
            name: np.array(params[name], dtype=np.float32, order="C", copy=True) for name in tensor_order(shapes)
        }
        self.optimizer = AdamW(run.optimizer, shapes)
        self.step = 0
        self.losses: list[float] = []
        self._gradients = DecoderGradients(config)

    @classmethod
    def from_model_file(cls, model: ModelFile, data: TrainingData, run: RunConfig) -> FineTuner:
        return cls(
            model.config,
            model.tensors,
            data,
            run,
            base_fingerprint=model.fingerprint,
            metadata=model.header,
        )

    # Training

    def train_step(self) -> StepResult:
        """Runs the next step and returns its loss (before the update), learning rate and gradient norm."""
        if self.step >= self.run.steps:
            raise RuntimeError("the run has already completed all of its steps")
        windows = self.data.batch(self.step, self.run.batch_size, self.run.seed)
        targets_total = sum(len(window) - 1 for window in windows)
        scale = 1.0 / targets_total
        loss_total = 0.0
        gradients: dict[str, FloatArray] = {}
        for window in windows:
            loss, window_gradients = self._gradients.loss_and_gradients(
                self.params, window[:-1], window[1:], scale=scale
            )
            loss_total += loss
            for name, gradient in window_gradients.items():
                if name in gradients:
                    gradients[name] += gradient
                else:
                    gradients[name] = gradient.copy()
        self.step += 1
        learning_rate = self.run.optimizer.learning_rate_at(self.step, self.run.steps)
        norm = self.optimizer.step(self.params, gradients, self.step, learning_rate)
        mean_loss = loss_total / targets_total
        self.losses.append(mean_loss)
        return StepResult(self.step, mean_loss, learning_rate, norm)

    def train(
        self, *, until: int | None = None, on_step: Callable[[StepResult], None] | None = None
    ) -> list[StepResult]:
        """Trains up to step ``until`` (default: the end of the run)."""
        end = self.run.steps if until is None else min(until, self.run.steps)
        results = []
        while self.step < end:
            result = self.train_step()
            results.append(result)
            if on_step is not None:
                on_step(result)
        return results

    # Output

    def export(self, path: str | Path) -> str:
        """Writes the current weights as a ``model.dllm`` file with a ``fine_tuning`` section; returns its
        fingerprint."""
        metadata = dict(self.metadata)
        licence = dict(metadata.get("licence") or {})
        if licence.get("attribution"):
            # Apache-2.0 asks modified files to say so: the import's "otherwise unmodified" no longer holds.
            attribution = str(licence["attribution"])
            unmodified = "; the weights are otherwise unmodified."
            if attribution.endswith(unmodified):
                attribution = attribution[: -len(unmodified)] + "."
            licence["attribution"] = f"{attribution} Fine-tuned with EtAlii.Dllm ({self.step} steps); modified weights."
            metadata["licence"] = licence
        metadata["fine_tuning"] = {
            "base_fingerprint": self.base_fingerprint,
            "data_fingerprint": self.data.fingerprint,
            "run": self.run.to_dict(),
            "steps_completed": self.step,
            "final_loss": self.losses[-1] if self.losses else None,
        }
        tensors = {
            name: TensorSource(tuple(values.shape), lambda values=values: values)
            for name, values in self.params.items()
        }
        return write_model_file(path, self.config, tensors, metadata)

    def _checkpoint_tensors(self) -> Iterator[tuple[str, FloatArray]]:
        for name in tensor_order(self.params):
            yield f"param/{name}", self.params[name]
        yield from self.optimizer.state()

    def save_checkpoint(self, path: str | Path) -> str:
        """Writes the full training state; returns the SHA-256 of its tensor data (the checkpoint fingerprint)."""
        entries = []
        offset = 0
        tensors = list(self._checkpoint_tensors())
        for name, values in tensors:
            nbytes = values.size * _DTYPE.itemsize
            entries.append({"name": name, "shape": list(values.shape), "offset": offset, "nbytes": nbytes})
            offset += nbytes + (-nbytes % _ALIGNMENT)
        header = {
            "format": "dllm-checkpoint",
            "format_version": CHECKPOINT_VERSION,
            "architecture": self.config.to_dict(),
            "run": self.run.to_dict(),
            "step": self.step,
            "losses": self.losses,
            "base_fingerprint": self.base_fingerprint,
            "data_fingerprint": self.data.fingerprint,
            "metadata": self.metadata,
            "tensors": entries,
            "fingerprint": _PLACEHOLDER,
        }
        header_bytes = canonical_json(header)
        header_bytes += b" " * (-(_PREFIX.size + len(header_bytes)) % _ALIGNMENT)
        digest = hashlib.sha256()
        with Path(path).open("wb") as stream:
            stream.write(_PREFIX.pack(CHECKPOINT_MAGIC, CHECKPOINT_VERSION, len(header_bytes)))
            stream.write(header_bytes)
            for _, values in tensors:
                data = np.ascontiguousarray(values, dtype=_DTYPE).tobytes()
                data += b"\0" * (-len(data) % _ALIGNMENT)
                digest.update(data)
                stream.write(data)
            fingerprint = digest.hexdigest()
            marker = b'"fingerprint":"' + _PLACEHOLDER.encode() + b'"'
            stream.seek(_PREFIX.size + header_bytes.index(marker) + len(b'"fingerprint":"'))
            stream.write(fingerprint.encode())
        return fingerprint

    @classmethod
    def load_checkpoint(cls, path: str | Path, data: TrainingData) -> FineTuner:
        """Restores a run from a checkpoint; ``data`` must be the data the run was started with."""
        path = Path(path)
        raw = path.read_bytes()
        if len(raw) < _PREFIX.size:
            raise CheckpointError(f"{path}: not a checkpoint file")
        magic, version, header_length = _PREFIX.unpack_from(raw)
        if magic != CHECKPOINT_MAGIC:
            raise CheckpointError(f"{path}: not a checkpoint file")
        if version != CHECKPOINT_VERSION:
            raise CheckpointError(f"{path}: unsupported checkpoint version {version}")
        try:
            header = json.loads(raw[_PREFIX.size : _PREFIX.size + header_length].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CheckpointError(f"{path}: header is not valid JSON") from error
        body = raw[_PREFIX.size + header_length :]
        if hashlib.sha256(body).hexdigest() != header["fingerprint"]:
            raise CheckpointError(f"{path}: tensor data does not match the fingerprint (corrupt file?)")
        if header["data_fingerprint"] != data.fingerprint:
            raise CheckpointError(f"{path}: the checkpoint was trained on different data")
        tensors = {}
        for entry in header["tensors"]:
            begin = entry["offset"]
            values = np.frombuffer(body, dtype=_DTYPE, count=math.prod(entry["shape"]), offset=begin)
            tensors[entry["name"]] = values.reshape(entry["shape"])
        config = TransformerConfig.from_dict(header["architecture"])
        tuner = cls(
            config,
            {name[len("param/") :]: values for name, values in tensors.items() if name.startswith("param/")},
            data,
            RunConfig.from_dict(header["run"]),
            base_fingerprint=header["base_fingerprint"],
            metadata=header["metadata"],
        )
        for name in tuner.optimizer.m:
            tuner.optimizer.m[name][...] = tensors[f"adam_m/{name}"]
            tuner.optimizer.v[name][...] = tensors[f"adam_v/{name}"]
        tuner.step = int(header["step"])
        tuner.losses = [float(loss) for loss in header["losses"]]
        return tuner
