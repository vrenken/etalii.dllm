"""Model merges that are exact and recorded (``dllm merge``, ``docs/model-building.md``).

Merging models of one architecture and tokenizer is elementwise arithmetic on their tensors, plus a few reductions
(SLERP's norms and dot products) that run in the fixed-order C++ kernels. Elementwise work is done in float64 in a
fixed model order and rounded to float32 once, so a merge gives the same file, byte for byte, on every platform.

- ``linear``: the weighted average ``sum_i w_i T_i`` (weights normalised to sum to 1).
- ``slerp``: spherical interpolation of two models at ``t`` (0: the first, 1: the second), tensor by tensor; nearly
  parallel tensors are interpolated linearly.
- ``ties``: TIES-merging (Yadav et al., 2023) of task vectors against ``base``: each model's difference from the base
  is trimmed to its ``density`` largest magnitudes (ties broken by position), a sign is elected per element from the
  weighted sum, and the agreeing differences are averaged with their weights.

The merged file records a ``merge`` section (the method, its settings and every input's fingerprint and lineage) and
extends the first input's lineage with a ``merge`` step.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from etalii_dllm import numerics
from etalii_dllm.modelfile import ModelFile, TensorSource, _digest, extend_lineage, write_model_file

METHODS = ("linear", "slerp", "ties")

_PARALLEL = 0.9995
"""SLERP falls back to linear interpolation when the cosine between two tensors is above this."""


class MergeError(ValueError):
    """Models that cannot be merged, or settings that do not fit the method."""


def _float64(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def linear(tensors: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    total = _float64(tensors[0]) * weights[0]
    for tensor, weight in zip(tensors[1:], weights[1:], strict=True):
        total = total + _float64(tensor) * weight
    return total.astype(np.float32)


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    flat_a = np.ascontiguousarray(a, dtype=np.float32).reshape(-1)
    flat_b = np.ascontiguousarray(b, dtype=np.float32).reshape(-1)
    norms = math.sqrt(numerics.sum_squares(flat_a)) * math.sqrt(numerics.sum_squares(flat_b))
    cosine = numerics.dot(flat_a, flat_b) / norms if norms > 0 else 1.0
    cosine = min(1.0, max(-1.0, cosine))
    if abs(cosine) > _PARALLEL:
        return linear([a, b], [1.0 - t, t])
    theta = numerics.acos(cosine)
    sin_theta = numerics.sin(theta)
    scale_a = numerics.sin((1.0 - t) * theta) / sin_theta
    scale_b = numerics.sin(t * theta) / sin_theta
    return (_float64(a) * scale_a + _float64(b) * scale_b).astype(np.float32)


def _trim(delta: np.ndarray, density: float) -> np.ndarray:
    """``delta`` with all but its ``density`` largest magnitudes set to 0; equal magnitudes keep the earlier
    position (a stable sort, so a total order)."""
    flat = delta.reshape(-1)
    keep = math.ceil(density * flat.size)
    if keep >= flat.size:
        return delta
    order = np.argsort(-np.abs(flat), kind="stable")
    trimmed = np.zeros_like(flat)
    trimmed[order[:keep]] = flat[order[:keep]]
    return trimmed.reshape(delta.shape)


def ties(tensors: Sequence[np.ndarray], base: np.ndarray, weights: Sequence[float], density: float) -> np.ndarray:
    origin = _float64(base)
    deltas = [_trim(_float64(tensor) - origin, density) for tensor in tensors]
    mass = deltas[0] * weights[0]
    for delta, weight in zip(deltas[1:], weights[1:], strict=True):
        mass = mass + delta * weight
    sign = np.sign(mass)
    numerator = np.zeros_like(origin)
    denominator = np.zeros_like(origin)
    for delta, weight in zip(deltas, weights, strict=True):
        agree = (np.sign(delta) == sign) & (delta != 0)
        numerator = numerator + np.where(agree, delta * weight, 0.0)
        denominator = denominator + np.where(agree, weight, 0.0)
    merged = np.divide(numerator, denominator, out=np.zeros_like(origin), where=denominator != 0)
    return (origin + merged).astype(np.float32)


def _check_compatible(models: Sequence[ModelFile]) -> None:
    first = models[0]
    for model in models[1:]:
        if model.header["architecture"] != first.header["architecture"]:
            raise MergeError(f"{model.path}: another architecture than {first.path}")
        if model.tokenizer != first.tokenizer:
            raise MergeError(f"{model.path}: another tokenizer than {first.path}")


def _licence(models: Sequence[ModelFile], method: str) -> dict[str, Any]:
    """One licence section for the merged weights: every input's licence and attribution."""
    unmodified = "; the weights are otherwise unmodified."
    attributions, spdx, texts = [], [], []
    for model in models:
        licence = model.licence
        attribution = str(licence.get("attribution", "")).strip()
        if attribution.endswith(unmodified):
            attribution = attribution[: -len(unmodified)] + "."
        if attribution and attribution not in attributions:
            attributions.append(attribution)
        if licence.get("spdx") and licence["spdx"] not in spdx:
            spdx.append(str(licence["spdx"]))
        if licence.get("text") and licence["text"] not in texts:
            texts.append(str(licence["text"]))
    record: dict[str, Any] = dict(models[0].licence)
    record.update(
        {
            "spdx": " AND ".join(spdx) if spdx else record.get("spdx"),
            "text": "\n\n---\n\n".join(texts) if texts else record.get("text"),
            "attribution": " ".join([*attributions, f"Merged with EtAlii.Dllm ({method}); modified weights."]),
            "redistributable": all(bool(model.licence.get("redistributable")) for model in models),
        }
    )
    return record


def merge_models(
    paths: Sequence[str | Path],
    output: str | Path,
    method: str = "linear",
    weights: Sequence[float] | None = None,
    base: str | Path | None = None,
    t: float = 0.5,
    density: float = 0.2,
) -> str:
    """Merges the model files ``paths`` into ``output``; returns the new fingerprint."""
    if method not in METHODS:
        raise MergeError(f"unknown merge method {method!r}; supported: {', '.join(METHODS)}")
    if len(paths) < 2:
        raise MergeError("merging needs at least two models")
    if method == "slerp" and len(paths) != 2:
        raise MergeError("slerp merges exactly two models")
    if method == "slerp" and not 0.0 <= t <= 1.0:
        raise MergeError("slerp needs 0 <= t <= 1")
    if method == "ties" and base is None:
        raise MergeError("ties needs the --base model the others were fine-tuned from")
    if method == "ties" and not 0.0 < density <= 1.0:
        raise MergeError("ties needs 0 < density <= 1")
    if weights is None:
        weights = [1.0] * len(paths)
    if len(weights) != len(paths):
        raise MergeError(f"{len(weights)} weights for {len(paths)} models")
    if method == "linear":
        if sum(weights) == 0:
            raise MergeError("the weights add up to 0")
        weights = [w / math.fsum(weights) for w in weights]
    models = [ModelFile(path) for path in paths]
    origin = ModelFile(base) if base is not None else None
    _check_compatible([*models, *([origin] if origin is not None else [])])

    def merged(name: str) -> np.ndarray:
        tensors = [model.tensors[name] for model in models]
        if method == "slerp":
            return slerp(tensors[0], tensors[1], t)
        if method == "ties":
            assert origin is not None
            return ties(tensors, origin.tensors[name], list(weights), density)
        return linear(tensors, list(weights))

    record: dict[str, Any] = {
        "method": method,
        "weights": list(weights),
        "inputs": [{"fingerprint": model.fingerprint, "lineage": model.lineage} for model in models],
    }
    if method == "slerp":
        record["t"] = t
    if method == "ties":
        assert origin is not None
        record |= {"density": density, "base": {"fingerprint": origin.fingerprint, "lineage": origin.lineage}}
    first = models[0]
    step = {"step": "merge", "input": first.fingerprint, "method": method, "merge": _digest(record)}
    metadata: dict[str, Any] = {
        key: first.header.get(key) for key in ("source", "tokenizer", "chat_template", "embedding")
    }
    metadata |= {
        "licence": _licence(models, method),
        "merge": record,
        "lineage": extend_lineage(first.lineage, first.fingerprint, step),
    }
    load: Callable[[str], Callable[[], np.ndarray]] = lambda name: lambda: merged(name)  # noqa: E731
    tensors = {name: TensorSource(tuple(values.shape), load(name)) for name, values in first.tensors.items()}
    return write_model_file(output, first.config, tensors, metadata)
