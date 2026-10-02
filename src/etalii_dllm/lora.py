"""Low-rank adapters (LoRA): the adapter tensors, merging them into the weights, and the Hugging Face PEFT format.

A LoRA adapter on a linear layer ``W`` (``[out, in]``) is a pair ``A`` (``[rank, in]``) and ``B`` (``[out, rank]``);
the adapted layer uses ``W + scale * (B @ A)`` with ``scale = alpha / rank`` (``alpha / sqrt(rank)`` for rsLoRA).

Determinism: an adapter is always applied by merging it into the weights with :func:`merge`, whether the merge is
written to a ``model.dllm`` file (``dllm import ADAPTER --base BASE``) or done when a model is loaded (``--adapter``).
``B @ A`` is the ``linear`` kernel (fixed order, double accumulator), the scale and the addition are elementwise
float32 operations, so both routes produce the same weights, the same fingerprint and the same output bits.
Training (``dllm finetune --lora-rank``) runs the decoder on the merged weights too, and derives the adapter
gradients from the merged weight's gradient (:func:`adapter_gradients`).

On disk an adapter is a PEFT directory: ``adapter_config.json`` and ``adapter_model.safetensors`` with keys such
as ``base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight``, so adapters trained here load in PEFT and
adapters trained with PEFT import here.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from etalii_dllm.architecture import TransformerConfig
from etalii_dllm.modelfile import tensor_order
from etalii_dllm.numerics import FloatArray, fill_gaussian, linear

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"

# Our linear layers and their Hugging Face module names.
MODULES = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.o_proj",
    "gate": "mlp.gate_proj",
    "up": "mlp.up_proj",
    "down": "mlp.down_proj",
}
ALL_TARGETS = tuple(MODULES)
_WEIGHT_NAMES = {
    "q": "attention.q.weight",
    "k": "attention.k.weight",
    "v": "attention.v.weight",
    "o": "attention.o.weight",
    "gate": "mlp.gate.weight",
    "up": "mlp.up.weight",
    "down": "mlp.down.weight",
}
_BY_HF_MODULE = {module.split(".")[1]: target for target, module in MODULES.items()}
# The experts of a mixture-of-experts layer: Mixtral's block_sparse_moe.experts.E.w1/w3/w2, otherwise
# mlp.experts.E.gate_proj/up_proj/down_proj (the per-expert modules PEFT adapts).
_MIXTRAL_EXPERTS = {"gate": "w1", "up": "w3", "down": "w2"}
_BY_MIXTRAL_EXPERT = {module: target for target, module in _MIXTRAL_EXPERTS.items()}
_EXPERT_TARGETS = ("gate", "up", "down")
# A shared expert (Qwen2-MoE): mlp.shared_expert.gate_proj/up_proj/down_proj.
_PEFT_KEY = re.compile(
    r"^(?:base_model\.model\.)?model\.layers\.(\d+)\.(self_attn|mlp|block_sparse_moe)\."
    r"(?:experts\.(\d+)\.|(shared_expert)\.)?(\w+)\.lora_([AB])\.weight$"
)


class AdapterError(ValueError):
    """The adapter is invalid, uses a PEFT feature this engine does not support, or does not fit the model."""


@dataclass(frozen=True)
class LoraConfig:
    rank: int
    alpha: float
    targets: tuple[str, ...] = ALL_TARGETS
    """Short names of the adapted linear layers (``q k v o gate up down``), in that order."""
    rslora: bool = False
    """rsLoRA scaling, ``alpha / sqrt(rank)`` instead of ``alpha / rank``."""

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise AdapterError("the LoRA rank must be positive")
        unknown = [t for t in self.targets if t not in MODULES]
        if unknown or not self.targets:
            raise AdapterError(f"unknown LoRA targets {unknown}; choose from {', '.join(ALL_TARGETS)}")
        object.__setattr__(self, "targets", tuple(t for t in ALL_TARGETS if t in self.targets))

    @property
    def scale(self) -> float:
        return self.alpha / (math.sqrt(self.rank) if self.rslora else self.rank)

    def to_dict(self) -> dict[str, Any]:
        values: dict[str, Any] = {"rank": self.rank, "alpha": self.alpha, "targets": list(self.targets)}
        if self.rslora:
            values["rslora"] = True
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> LoraConfig:
        return cls(int(values["rank"]), float(values["alpha"]), tuple(values["targets"]), bool(values.get("rslora")))


def target_weights(config: TransformerConfig, lora: LoraConfig) -> list[str]:
    """The adapted weight names, in tensor order. In a mixture-of-experts layer ``gate``, ``up`` and ``down`` adapt
    every expert's projection (``layers.N.mlp.experts.E.gate.weight``) and the shared expert's
    (``layers.N.mlp.shared.gate.weight``); the router and the shared expert's gate are never adapted."""
    names = []
    for layer in range(config.layers):
        for target in lora.targets:
            if target in _EXPERT_TARGETS and config.is_sparse(layer):
                names.extend(f"layers.{layer}.mlp.experts.{e}.{target}.weight" for e in range(config.experts))
                if config.shared_expert_intermediate_size is not None:
                    names.append(f"layers.{layer}.mlp.shared.{target}.weight")
            else:
                names.append(f"layers.{layer}.{_WEIGHT_NAMES[target]}")
    return tensor_order(names)


def adapter_shapes(config: TransformerConfig, lora: LoraConfig) -> dict[str, tuple[int, int]]:
    """``<weight>.lora_a`` (``[rank, in]``) and ``<weight>.lora_b`` (``[out, rank]``) for every adapted weight."""
    shapes = config.tensor_shapes()
    result = {}
    for name in target_weights(config, lora):
        out_features, in_features = shapes[name]
        result[name + ".lora_a"] = (lora.rank, in_features)
        result[name + ".lora_b"] = (out_features, lora.rank)
    return result


def init_adapters(config: TransformerConfig, lora: LoraConfig, seed: int) -> dict[str, FloatArray]:
    """Fresh adapters: ``A`` Gaussian with standard deviation ``1 / rank`` (PEFT's ``init_lora_weights="gaussian"``)
    from the deterministic generator, ``B`` zero, so training starts from the base model's exact weights."""
    adapters: dict[str, FloatArray] = {}
    for index, (name, shape) in enumerate(adapter_shapes(config, lora).items()):
        if name.endswith(".lora_a"):
            values = fill_gaussian(seed * 1_000_003 + index, shape[0] * shape[1]).reshape(shape)
            adapters[name] = np.ascontiguousarray(values * np.float32(1.0 / lora.rank), dtype=np.float32)
        else:
            adapters[name] = np.zeros(shape, dtype=np.float32)
    return adapters


def merge(weight: np.ndarray, a: np.ndarray, b: np.ndarray, scale: float) -> FloatArray:
    """``weight + scale * (b @ a)`` in float32; ``b @ a`` is the ``linear`` kernel (row ``o`` of ``b`` against
    column ``i`` of ``a``, summed over the rank in index order)."""
    delta = linear(np.ascontiguousarray(b, dtype=np.float32), np.ascontiguousarray(a.T, dtype=np.float32)).numpy()
    return np.asarray(weight, dtype=np.float32) + delta * np.float32(scale)


def merged_weights(
    config: TransformerConfig, weights: Mapping[str, np.ndarray], adapters: Mapping[str, np.ndarray], lora: LoraConfig
) -> dict[str, np.ndarray]:
    """``weights`` with every adapted weight replaced by its merge; the other tensors are passed through."""
    merged = dict(weights)
    for name in target_weights(config, lora):
        merged[name] = merge(weights[name], adapters[name + ".lora_a"], adapters[name + ".lora_b"], lora.scale)
    return merged


def adapter_gradients(
    gradient: np.ndarray, a: np.ndarray, b: np.ndarray, scale: float
) -> tuple[FloatArray, FloatArray]:
    """``(dA, dB)`` from ``gradient``, the loss gradient of the merged weight: ``dA = scale * B^T @ dW`` and
    ``dB = scale * dW @ A^T``, both with the ``linear`` kernel."""
    s = np.float32(scale)
    gradient = np.ascontiguousarray(gradient, dtype=np.float32)
    da = linear(np.ascontiguousarray(b.T, dtype=np.float32), np.ascontiguousarray(gradient.T)).numpy() * s
    db = linear(gradient, np.ascontiguousarray(a, dtype=np.float32)).numpy() * s
    return da, db


# The PEFT directory format


def peft_key(name: str, config: TransformerConfig | None = None) -> str:
    """``layers.3.attention.q.weight.lora_a`` -> ``base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight``;
    an expert's ``layers.3.mlp.experts.5.gate.weight.lora_a`` -> ``...layers.3.mlp.experts.5.gate_proj.lora_A.weight``
    (Mixtral: ``...layers.3.block_sparse_moe.experts.5.w1.lora_A.weight``); a shared expert's
    ``layers.3.mlp.shared.up.weight.lora_b`` -> ``...layers.3.mlp.shared_expert.up_proj.lora_B.weight``. Granite MoE
    stores its experts fused (one parameter for all of them, gate and up rows stacked), so PEFT has no module for
    one of them: its expert adapters are refused."""
    layer, rest = name.split(".", 2)[1:]
    weight, kind = rest.rsplit(".", 1)
    suffix = f"lora_{kind[-1].upper()}.weight"
    if weight.startswith(("mlp.experts.", "mlp.shared.")) and config is not None and config.family == "granitemoe":
        raise AdapterError("Granite MoE's experts are fused, so PEFT cannot adapt one; export the merged model")
    if weight.startswith("mlp.shared."):
        target = weight.split(".")[2]
        return f"base_model.model.model.layers.{layer}.mlp.shared_expert.{MODULES[target].split('.')[1]}.{suffix}"
    if weight.startswith("mlp.experts."):
        expert, target = weight.split(".")[2:4]
        if config is not None and config.family == "mixtral":
            module = f"block_sparse_moe.experts.{expert}.{_MIXTRAL_EXPERTS[target]}"
        else:
            module = f"mlp.experts.{expert}.{MODULES[target].split('.')[1]}"
        return f"base_model.model.model.layers.{layer}.{module}.{suffix}"
    target = next(t for t, w in _WEIGHT_NAMES.items() if w == weight)
    return f"base_model.model.model.layers.{layer}.{MODULES[target]}.{suffix}"


def target_modules(config: TransformerConfig | None, lora: LoraConfig) -> list[str]:
    """PEFT's ``target_modules``: the module names of the adapted layers."""
    modules = set()
    for target in lora.targets:
        if config is not None and config.family == "mixtral" and target in _EXPERT_TARGETS:
            modules.add(_MIXTRAL_EXPERTS[target])
        else:
            modules.add(MODULES[target].split(".")[1])
    return sorted(modules)


def write_peft(
    directory: str | Path,
    adapters: Mapping[str, np.ndarray],
    lora: LoraConfig,
    base_model: str | None = None,
    config: TransformerConfig | None = None,
) -> None:
    """Writes ``adapters`` as a PEFT LoRA adapter (float32, keys and JSON sorted, so equal adapters give equal
    bytes)."""
    from etalii_dllm.importing.safetensors import write_safetensors

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    settings = {
        "base_model_name_or_path": base_model,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": "gaussian",
        "lora_alpha": lora.alpha,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": lora.rank,
        "target_modules": target_modules(config, lora),
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": lora.rslora,
    }
    text = json.dumps(settings, indent=2, sort_keys=True) + "\n"
    (directory / ADAPTER_CONFIG).write_text(text, encoding="utf-8", newline="\n")
    write_safetensors(
        directory / ADAPTER_WEIGHTS, {peft_key(name, config): adapters[name] for name in adapters}, {"format": "pt"}
    )


def _unsupported(config: Mapping[str, Any]) -> str | None:
    if config.get("peft_type", "LORA") != "LORA":
        return f"peft_type {config.get('peft_type')!r} (only LORA)"
    if config.get("use_dora"):
        return "DoRA (use_dora)"
    if config.get("fan_in_fan_out"):
        return "fan_in_fan_out"
    if config.get("bias", "none") != "none":
        return f"bias={config.get('bias')!r}"
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        return "per-layer ranks or alphas (rank_pattern/alpha_pattern)"
    if config.get("modules_to_save"):
        return "modules_to_save"
    if config.get("layers_to_transform") is not None or config.get("layer_replication"):
        return "layers_to_transform/layer_replication"
    return None


def _adapter_name(key: str, config: TransformerConfig, directory: Path) -> tuple[str, str]:
    """(our adapter name, its target) for the PEFT tensor ``key``; refuses modules this engine does not adapt and
    ones that do not fit ``config``."""
    match = _PEFT_KEY.match(key)
    if match is None:
        raise AdapterError(f"{directory}: unsupported adapter tensor {key!r}")
    layer, parent, expert, shared, module, kind = match.groups()
    index, suffix = int(layer), f".lora_{kind.lower()}"
    if shared is not None:
        target = _BY_HF_MODULE.get(module)
        if target not in _EXPERT_TARGETS:
            raise AdapterError(f"{directory}: unsupported adapter tensor {key!r}")
        fits = 0 <= index < config.layers and config.is_sparse(index) and parent == "mlp"
        fits = fits and config.shared_expert_intermediate_size is not None and config.family != "granitemoe"
        name = f"layers.{index}.mlp.shared.{target}.weight{suffix}"
    elif expert is None:
        target = _BY_HF_MODULE.get(module)
        if target is None:
            raise AdapterError(f"{directory}: unsupported adapter tensor {key!r}")
        fits = 0 <= index < config.layers and MODULES[target].startswith(parent)
        name = f"layers.{index}.{_WEIGHT_NAMES[target]}{suffix}"
        fits = fits and not (target in _EXPERT_TARGETS and config.is_sparse(index))
    else:
        mixtral = config.family == "mixtral"
        target = (_BY_MIXTRAL_EXPERT if mixtral else _BY_HF_MODULE).get(module)
        if target not in _EXPERT_TARGETS:
            raise AdapterError(f"{directory}: unsupported adapter tensor {key!r}")
        fits = 0 <= index < config.layers and config.is_sparse(index) and int(expert) < config.experts
        fits = fits and parent == ("block_sparse_moe" if mixtral else "mlp") and config.family != "granitemoe"
        name = f"layers.{index}.mlp.experts.{int(expert)}.{target}.weight{suffix}"
    if not fits:
        raise AdapterError(f"{directory}: tensor {key!r} does not fit the model")
    return name, target


def read_peft(directory: str | Path, config: TransformerConfig) -> tuple[LoraConfig, dict[str, FloatArray]]:
    """Reads a PEFT LoRA adapter for a model with architecture ``config``; returns its settings and the adapters in
    our naming. Anything that would change the math and is not supported is refused."""
    from etalii_dllm.importing.safetensors import SafetensorsError, SafetensorsFile

    directory = Path(directory)
    config_path = directory / ADAPTER_CONFIG
    if not config_path.exists():
        raise AdapterError(f"{directory}: no {ADAPTER_CONFIG}")
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    problem = _unsupported(settings)
    if problem:
        raise AdapterError(f"{directory}: the adapter uses {problem}, which is not supported")
    weights_path = directory / ADAPTER_WEIGHTS
    if not weights_path.exists():
        raise AdapterError(f"{directory}: no {ADAPTER_WEIGHTS} (adapter_model.bin is not read; convert it)")
    try:
        file = SafetensorsFile(weights_path)
    except SafetensorsError as error:
        raise AdapterError(str(error)) from error

    rank = int(settings["r"])
    adapters: dict[str, FloatArray] = {}
    targets: set[str] = set()
    for tensor in file:
        name, target = _adapter_name(tensor.name, config, directory)
        if tensor.dtype not in ("F32", "F16", "BF16"):
            raise AdapterError(f"{directory}: tensor {tensor.name!r} has dtype {tensor.dtype}")
        targets.add(target)
        adapters[name] = tensor.to_float32()
    if not adapters:
        raise AdapterError(f"{directory}: the adapter has no LoRA tensors")
    lora = LoraConfig(rank, float(settings.get("lora_alpha", 8)), tuple(targets), bool(settings.get("use_rslora")))
    expected = adapter_shapes(config, lora)
    if set(adapters) != set(expected):
        missing = sorted(set(expected) - set(adapters))[:3]
        raise AdapterError(f"{directory}: the adapter does not cover every layer (missing {missing}, ...)")
    for name, shape in expected.items():
        if tuple(adapters[name].shape) != shape:
            raise AdapterError(f"{directory}: {name} has shape {tuple(adapters[name].shape)}, expected {shape}")
    return lora, {name: np.ascontiguousarray(adapters[name], dtype=np.float32) for name in tensor_order(adapters)}


def apply_adapter(
    config: TransformerConfig, weights: Mapping[str, np.ndarray], directory: str | Path
) -> tuple[dict[str, np.ndarray], LoraConfig]:
    """``weights`` with the PEFT adapter in ``directory`` merged in (what ``--adapter`` does at load time)."""
    lora, adapters = read_peft(directory, config)
    return merged_weights(config, weights, adapters, lora), lora


def adapter_files(directory: str | Path) -> Sequence[Path]:
    directory = Path(directory)
    return [directory / ADAPTER_CONFIG, directory / ADAPTER_WEIGHTS]
