"""``dllm import``: converts a Hugging Face checkpoint (safetensors) or a GGUF file to ``model.dllm``.

The source architecture is mapped onto :class:`~etalii_dllm.architecture.TransformerConfig` and our tensor names;
anything the decoder does not implement fails the import loudly instead of producing a model that runs wrongly.
Float tensors are widened to float32 exactly; quantised GGUF tensors are dequantised with the reference formulas.
The source files' hashes, the repository and revision, and the licence text and attribution go into the file.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from etalii_dllm.architecture import FAMILIES, TransformerConfig
from etalii_dllm.importing import hub
from etalii_dllm.importing.gguf import GgufFile
from etalii_dllm.importing.licences import PERMISSIVE, STANDARD_TEXTS
from etalii_dllm.importing.safetensors import open_checkpoint
from etalii_dllm.lora import ADAPTER_CONFIG, AdapterError, adapter_files, apply_adapter
from etalii_dllm.modelfile import (
    ModelFile,
    TensorSource,
    adapter_step,
    extend_lineage,
    import_step,
    lineage,
    write_model_file,
)
from etalii_dllm.numerics import log


class ModelImportError(ValueError):
    """The source cannot be imported: unsupported architecture or feature, missing files, or licence refused."""


@dataclass(frozen=True)
class ImportResult:
    path: Path
    fingerprint: str
    config: TransformerConfig
    source: dict[str, Any]
    licence: dict[str, Any]


@dataclass(frozen=True)
class _Converted:
    config: TransformerConfig
    tensors: dict[str, TensorSource]
    source: dict[str, Any]
    licence_id: str | None
    licence_text: str | None
    licence_link: str | None
    name: str | None
    tokenizer: dict[str, Any] | None
    chat_template: str | None
    embedding: dict[str, Any] | None = None


# ---------------------------------------------------------------------------------------------------------------
# Hugging Face checkpoints


_HF_LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_HF_LAYER_NAMES = {
    "input_layernorm.weight": "attention_norm.weight",
    "post_attention_layernorm.weight": "mlp_norm.weight",
    "self_attn.q_proj.weight": "attention.q.weight",
    "self_attn.k_proj.weight": "attention.k.weight",
    "self_attn.v_proj.weight": "attention.v.weight",
    "self_attn.o_proj.weight": "attention.o.weight",
    "self_attn.q_proj.bias": "attention.q.bias",
    "self_attn.k_proj.bias": "attention.k.bias",
    "self_attn.v_proj.bias": "attention.v.bias",
    "self_attn.q_norm.weight": "attention.q_norm.weight",
    "self_attn.k_norm.weight": "attention.k_norm.weight",
    "mlp.gate_proj.weight": "mlp.gate.weight",
    "mlp.up_proj.weight": "mlp.up.weight",
    "mlp.down_proj.weight": "mlp.down.weight",
    "mlp.gate.weight": "mlp.router.weight",
    "block_sparse_moe.gate.weight": "mlp.router.weight",
    "block_sparse_moe.router.layer.weight": "mlp.router.weight",  # Granite MoE
    # Qwen2-MoE's shared expert and its gate.
    "mlp.shared_expert.gate_proj.weight": "mlp.shared.gate.weight",
    "mlp.shared_expert.up_proj.weight": "mlp.shared.up.weight",
    "mlp.shared_expert.down_proj.weight": "mlp.shared.down.weight",
    "mlp.shared_expert_gate.weight": "mlp.shared_gate.weight",
    "shared_mlp.output_linear.weight": "mlp.shared.down.weight",  # Granite MoE shared
}
# Experts: Qwen3-MoE and OLMoE name them like the dense MLP, Mixtral w1 (gate), w3 (up) and w2 (down).
_HF_EXPERT = re.compile(r"^(?:mlp|block_sparse_moe)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj|w1|w2|w3)\.weight$")
_HF_EXPERT_NAMES = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down", "w1": "gate", "w3": "up", "w2": "down"}
# OLMo 2 normalises the outputs of attention and the MLP instead of their inputs.
_HF_POST_NORM_NAMES = {
    "post_attention_layernorm.weight": "attention_post_norm.weight",
    "post_feedforward_layernorm.weight": "mlp_post_norm.weight",
}
# Gemma normalises both.
_HF_SANDWICH_NORM_NAMES = {**_HF_POST_NORM_NAMES, "pre_feedforward_layernorm.weight": "mlp_norm.weight"}
_HF_GLOBAL_NAMES = {
    "model.embed_tokens.weight": "token_embedding.weight",
    "model.norm.weight": "final_norm.weight",
    "lm_head.weight": "lm_head.weight",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _ids(value: Any) -> list[int]:
    if value is None:
        return []
    return [int(v) for v in value] if isinstance(value, list) else [int(value)]


def _rope_from_hf(config: dict[str, Any]) -> tuple[float, dict[str, Any] | None]:
    parameters = config.get("rope_parameters")
    if isinstance(parameters, dict):  # transformers v5 layout
        theta = parameters.get("rope_theta", config.get("rope_theta", 10000.0))
        scaling: dict[str, Any] | None = {
            k: v for k, v in parameters.items() if k not in ("rope_theta", "partial_rotary_factor")
        }
    else:
        theta = config.get("rope_theta", 10000.0)
        scaling = config.get("rope_scaling")
    if scaling:
        scaling = dict(scaling)
        kind = scaling.pop("rope_type", None) or scaling.pop("type", None) or "default"
        if config.get("model_type") == "phi3" and kind in ("su", "yarn"):  # what transformers' Phi3Config does
            kind = "longrope"
        if kind == "default":
            scaling = None
        elif kind in ("linear", "llama3"):
            scaling = {"rope_type": kind, **scaling}
        elif kind == "longrope":
            scaling = _longrope_from_hf(config, scaling)
        elif kind == "yarn":
            scaling = _yarn_from_hf(config, scaling)
        else:
            raise ModelImportError(f"RoPE scaling {kind!r} is not supported yet")
    return float(theta), scaling or None


def _longrope_from_hf(config: dict[str, Any], scaling: dict[str, Any]) -> dict[str, Any]:
    """LongRoPE with its ``attention_factor`` worked out as transformers does (from ``factor``, or else from the ratio
    of ``max_position_embeddings`` to ``original_max_position_embeddings``), in dllm's portable ``log``."""
    original = scaling.get("original_max_position_embeddings", config.get("original_max_position_embeddings"))
    if original is None or "short_factor" not in scaling or "long_factor" not in scaling:
        raise ModelImportError("LongRoPE needs short_factor, long_factor and original_max_position_embeddings")
    original = int(original)
    factor = scaling.get("factor")
    if factor is None:
        factor = int(config.get("max_position_embeddings", original)) / original
    attention = scaling.get("attention_factor")
    if attention is None:
        attention = 1.0 if factor <= 1.0 else math.sqrt(1 + log(float(factor)) / log(float(original)))
    return {
        "rope_type": "longrope",
        "short_factor": [float(f) for f in scaling["short_factor"]],
        "long_factor": [float(f) for f in scaling["long_factor"]],
        "original_max_position_embeddings": original,
        "attention_factor": float(attention),
    }


def _yarn_mscale(scale: float, mscale: float = 1.0) -> float:
    return 1.0 if scale <= 1.0 else 0.1 * mscale * log(scale) + 1.0


def yarn_scaling(
    factor: float,
    original: int,
    *,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    truncate: bool = True,
    attention_factor: float | None = None,
) -> dict[str, Any]:
    """A YaRN ``rope_scaling`` that stretches a context of ``original`` tokens ``factor`` times. The attention factor
    defaults to transformers' ``0.1 ln(factor) + 1``, in dllm's portable ``log``."""
    if not factor > 0 or original < 1:
        raise ModelImportError("YaRN needs a positive factor and original_max_position_embeddings")
    return {
        "rope_type": "yarn",
        "factor": float(factor),
        "original_max_position_embeddings": int(original),
        "beta_fast": float(beta_fast),
        "beta_slow": float(beta_slow),
        "truncate": bool(truncate),
        "attention_factor": float(_yarn_mscale(float(factor)) if attention_factor is None else attention_factor),
    }


def _yarn_from_hf(config: dict[str, Any], scaling: dict[str, Any]) -> dict[str, Any]:
    """YaRN with its attention factor worked out as transformers does (from ``mscale`` and ``mscale_all_dim`` when
    both are set, else ``0.1 ln(factor) + 1``)."""
    original = scaling.get("original_max_position_embeddings") or config.get("max_position_embeddings")
    if "factor" not in scaling or original is None:
        raise ModelImportError("YaRN needs factor and original_max_position_embeddings")
    factor = float(scaling["factor"])
    attention = scaling.get("attention_factor")
    mscale, mscale_all_dim = scaling.get("mscale"), scaling.get("mscale_all_dim")
    if attention is None and mscale and mscale_all_dim:
        attention = _yarn_mscale(factor, float(mscale)) / _yarn_mscale(factor, float(mscale_all_dim))
    return yarn_scaling(
        factor,
        int(original),
        beta_fast=float(scaling.get("beta_fast") or 32.0),
        beta_slow=float(scaling.get("beta_slow") or 1.0),
        truncate=bool(scaling.get("truncate", True)),
        attention_factor=None if attention is None else float(attention),
    )


def _extend_context(
    scaling: dict[str, Any] | None, context: int, length: int, limit: int | None
) -> tuple[dict[str, Any] | None, int]:
    """``(scaling, context)`` for a window of ``length`` tokens instead of ``context``: a longer one is reached with
    YaRN (factor = ``length / context``) or, for LongRoPE, with its long factors (up to ``limit``, the longest context
    the model was trained for). The frequencies are fixed in the model file, so a token's output never depends on
    how long its sequence grows. A shorter window just lowers the limit."""
    if length < 1:
        raise ModelImportError("the context length must be positive")
    kind = scaling["rope_type"] if scaling else None
    if kind == "longrope":
        assert scaling is not None
        scaling = {k: v for k, v in scaling.items() if k != "factor_set"}
        if length > scaling["original_max_position_embeddings"]:
            if limit is not None and length > limit:
                raise ModelImportError(f"this LongRoPE model was trained for at most {limit} tokens of context")
            scaling["factor_set"] = "long"
    elif length > context:
        if scaling is not None:
            raise ModelImportError(
                f"this model already scales its RoPE ({kind}); its context is at most {context} tokens"
            )
        scaling = yarn_scaling(length / context, context)
    return scaling, length


def with_context_length(config: TransformerConfig, length: int, limit: int | None = None) -> TransformerConfig:
    """``config`` with a context window of ``length`` tokens (see ``_extend_context``); ``dllm import
    --context-length`` for GGUF sources."""
    scaling, context = _extend_context(config.rope_scaling, config.context_length, length, limit)
    try:
        return dataclasses.replace(config, context_length=context, rope_scaling=scaling)
    except ValueError as error:
        raise ModelImportError(str(error)) from error


def _rotary_dim(config: dict[str, Any], head_dim: int) -> int | None:
    parameters = config.get("rope_parameters")
    factor = config.get("partial_rotary_factor")
    if isinstance(parameters, dict) and "partial_rotary_factor" in parameters:
        factor = parameters["partial_rotary_factor"]
    if factor is None or float(factor) == 1.0:
        return None
    return int(head_dim * float(factor))


def hf_config(
    config: dict[str, Any], generation: dict[str, Any] | None = None, *, context_length: int | None = None
) -> TransformerConfig:
    """Maps a Hugging Face ``config.json`` (Gemma 2, Gemma 3, Granite, Llama, Mistral, Mixtral, OLMo 2, OLMoE,
    Phi-3, Qwen2, Qwen3 or Qwen3-MoE) to our description, with a context window of ``context_length`` tokens when given
    (:func:`with_context_length`)."""
    model_type = config.get("model_type")
    family = _HF_MODEL_TYPES.get(model_type)
    if family is None:
        hint = "; import the text-only Gemma 3 checkpoint (model_type gemma3_text)" if model_type == "gemma3" else ""
        supported = ", ".join(sorted(_HF_MODEL_TYPES))
        raise ModelImportError(f"model_type {model_type!r} is not supported (supported: {supported}){hint}")
    hf_activation = config.get("hidden_activation") or config.get("hidden_act") or "silu"
    activation = _HF_ACTIVATIONS.get(hf_activation)
    if activation is None:
        raise ModelImportError(f"activation {hf_activation!r} is not supported")
    if config.get("mlp_bias"):
        raise ModelImportError("MLP biases are not supported")
    if config.get("clip_qkv") is not None:
        raise ModelImportError("clip_qkv is not supported")
    window, window_layers = _sliding_window_from_hf(config, family)
    heads = int(config["num_attention_heads"])
    hidden = int(config["hidden_size"])
    head_dim = int(config.get("head_dim") or hidden // heads)
    theta, scaling = _rope_from_hf(config)
    local_theta = None
    if family == "gemma3":
        theta, scaling, local_theta = _gemma3_rope_from_hf(config)
    trained = int(config.get("max_position_embeddings", 2048))
    context = trained
    if scaling and scaling["rope_type"] == "longrope":
        # transformers switches to the long factors once a sequence outgrows the original context, which would make
        # a token's output depend on how long the sequence gets; dllm keeps the short factors and that context
        # unless the import asks for a longer one (then the long factors, for every token).
        context = min(context, scaling["original_max_position_embeddings"])
    if scaling and scaling["rope_type"] == "yarn":  # YaRN stretches the original context factor times
        context = max(context, int(scaling["factor"] * scaling["original_max_position_embeddings"]))
    if context_length is not None:
        scaling, context = _extend_context(scaling, context, context_length, trained)
    if window is not None and window >= context:  # the window never binds
        window, window_layers = None, None
    eos = _ids(config.get("eos_token_id"))
    for token in _ids((generation or {}).get("eos_token_id")):
        if token not in eos:
            eos.append(token)
    bos = config.get("bos_token_id")
    try:  # a config the decoder rejects (e.g. LongRoPE factors that do not fit the heads) is an import error
        return TransformerConfig(
            family=family,
            vocabulary_size=int(config["vocab_size"]),
            hidden_size=hidden,
            intermediate_size=int(config["intermediate_size"]),
            layers=int(config["num_hidden_layers"]),
            heads=heads,
            kv_heads=int(config.get("num_key_value_heads") or heads),
            head_dim=head_dim,
            context_length=context,
            rms_norm_eps=float(config.get("rms_norm_eps", 1e-6)),
            rope_theta=theta,
            rope_scaling=scaling,
            attention_bias=family in ("qwen2", "qwen2_moe") or bool(config.get("attention_bias", False)),
            qk_norm=family in ("gemma3", "olmo2", "olmoe", "qwen3", "qwen3_moe"),
            qk_norm_scope="all" if family in ("olmo2", "olmoe") else "head",
            norm_placement={"olmo2": "post", "gemma2": "sandwich", "gemma3": "sandwich"}.get(family, "pre"),
            norm_unit_offset=family in ("gemma2", "gemma3"),
            activation=activation,
            tie_word_embeddings=bool(config.get("tie_word_embeddings", family in ("gemma2", "gemma3"))),
            bos_token_id=None if bos is None else int(bos),
            eos_token_ids=tuple(eos),
            sliding_window=window,
            sliding_window_layers=window_layers,
            rotary_dim=_rotary_dim(config, head_dim),
            local_rope_theta=local_theta,
            attention_softcap=_optional_float(config.get("attn_logit_softcapping")),
            logits_softcap=_optional_float(config.get("final_logit_softcapping")),
            **_granite_multipliers(config, family),
            **_gemma_multipliers(config, family, hidden),
            **_experts_from_hf(config, family),
        )
    except ValueError as error:
        raise ModelImportError(str(error)) from error


# model_type -> family; Gemma 3's text-only checkpoints are "gemma3_text", and Granite MoE with a shared expert is
# "granitemoeshared".
_HF_MODEL_TYPES = {name: name for name in FAMILIES if name != "gemma3"} | {
    "gemma3_text": "gemma3",
    "granitemoeshared": "granitemoe",
}
_HF_ACTIVATIONS = {"silu": "silu", "gelu_pytorch_tanh": "gelu_tanh"}


def _gemma3_rope_from_hf(config: dict[str, Any]) -> tuple[float, dict[str, Any] | None, float]:
    """``(theta, scaling, local theta)``: the full-attention layers take ``rope_theta`` and any scaling, the
    sliding-window layers ``rope_local_base_freq`` and none (transformers v5 keeps both under ``rope_parameters``,
    keyed by layer type)."""
    parameters = config.get("rope_parameters")
    if isinstance(parameters, dict) and "full_attention" in parameters:
        full = parameters["full_attention"] or {}
        local = parameters.get("sliding_attention") or {}
        theta, scaling = _rope_from_hf({"rope_parameters": {"rope_theta": 1_000_000.0, **full}})
        local_theta, local_scaling = _rope_from_hf({"rope_parameters": {"rope_theta": 10_000.0, **local}})
    else:
        theta, scaling = _rope_from_hf({**config, "rope_theta": config.get("rope_theta", 1_000_000.0)})
        local_theta, local_scaling = float(config.get("rope_local_base_freq", 10_000.0)), None
    if local_scaling is not None:
        raise ModelImportError("RoPE scaling on Gemma 3's sliding-window layers is not supported")
    return theta, scaling, local_theta


def _experts_from_hf(config: dict[str, Any], family: str) -> dict[str, Any]:
    """The mixture-of-experts fields: the experts and how many each token uses, whether their weights are
    renormalised (always for Mixtral and Granite MoE, whose softmax over the top-k logits is the renormalised top-k;
    ``norm_topk_prob`` otherwise), the expert size, the dense layers of Qwen2/Qwen3-MoE (``mlp_only_layers``, and the
    layers ``decoder_sparse_step`` skips) and the shared expert (Qwen2-MoE's is gated)."""
    if config.get("n_shared_experts") or (
        family not in ("qwen2_moe", "granitemoe") and config.get("shared_expert_intermediate_size")
    ):
        raise ModelImportError("this kind of shared expert is not supported")
    if family not in ("mixtral", "olmoe", "qwen2_moe", "qwen3_moe", "granitemoe"):
        return {}
    experts = int(config.get("num_local_experts") or config.get("num_experts") or 0)
    settings: dict[str, Any] = {
        "experts": experts,
        "experts_per_token": int(config.get("num_experts_per_tok", 2)),
        "normalize_expert_weights": family in ("mixtral", "granitemoe") or bool(config.get("norm_topk_prob", False)),
    }
    if family in ("qwen2_moe", "qwen3_moe"):
        layers = int(config["num_hidden_layers"])
        step = int(config.get("decoder_sparse_step", 1))
        only = {int(layer) for layer in config.get("mlp_only_layers") or ()}
        dense = tuple(i for i in range(layers) if i in only or step < 1 or (i + 1) % step)
        settings["expert_intermediate_size"] = int(config.get("moe_intermediate_size", config["intermediate_size"]))
        settings["dense_layers"] = dense or None
    shared = int(config.get("shared_expert_intermediate_size") or config.get("shared_intermediate_size") or 0)
    if shared:
        settings["shared_expert_intermediate_size"] = shared
        settings["shared_expert_gate"] = family == "qwen2_moe"
    return settings


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _gemma_multipliers(config: dict[str, Any], family: str, hidden: int) -> dict[str, Any]:
    """Gemma scales the embedding rows by ``sqrt(hidden_size)`` and the attention scores by
    ``query_pre_attn_scalar ** -0.5``."""
    if family not in ("gemma2", "gemma3"):
        return {}
    return {
        "embedding_multiplier": math.sqrt(hidden),
        "attention_multiplier": float(config.get("query_pre_attn_scalar", 256)) ** -0.5,
    }


def _granite_multipliers(config: dict[str, Any], family: str) -> dict[str, Any]:
    if family not in ("granite", "granitemoe"):
        return {}
    return {
        "embedding_multiplier": float(config.get("embedding_multiplier", 1.0)),
        "attention_multiplier": float(config["attention_multiplier"]) if "attention_multiplier" in config else None,
        "residual_multiplier": float(config.get("residual_multiplier", 1.0)),
        "logits_scaling": float(config.get("logits_scaling", 1.0)),
    }


def _sliding_window_from_hf(config: dict[str, Any], family: str) -> tuple[int | None, tuple[int, ...] | None]:
    """``(window, layers)``: Mistral slides on every layer; Qwen2/Qwen3 with ``use_sliding_window`` on the layers from
    ``max_window_layers`` on. A ``layer_types`` list, where present, names the layers explicitly."""
    window = config.get("sliding_window")
    if family == "llama" or window is None:
        return None, None
    if family in ("qwen2", "qwen2_moe", "qwen3", "qwen3_moe") and not config.get("use_sliding_window"):
        return None, None
    layers = int(config["num_hidden_layers"])
    layer_types = config.get("layer_types")
    if isinstance(layer_types, list):
        if len(layer_types) != layers or any(t not in ("sliding_attention", "full_attention") for t in layer_types):
            raise ModelImportError(f"layer_types {layer_types!r} is not supported")
        sliding = tuple(i for i, kind in enumerate(layer_types) if kind == "sliding_attention")
    elif family in ("qwen2", "qwen2_moe", "qwen3", "qwen3_moe"):
        sliding = tuple(range(int(config.get("max_window_layers", layers)), layers))
    elif family in ("gemma2", "gemma3"):  # without layer_types: every sliding_window_pattern-th layer is global
        pattern = int(config.get("sliding_window_pattern", 6 if family == "gemma3" else 2))
        sliding = tuple(i for i in range(layers) if (i + 1) % pattern)
    else:
        sliding = tuple(range(layers))
    if not sliding:
        return None, None
    return int(window), None if len(sliding) == layers else sliding


def _split_fused(tensor: Any, config: TransformerConfig) -> dict[str, TensorSource] | None:
    """Phi-3 fuses the q/k/v projections into ``qkv_proj`` and the gate/up projections into ``gate_up_proj`` (rows
    stacked in that order); transformers v5 stacks the experts into ``experts.gate_up_proj`` ``[experts, 2 *
    size, hidden]`` and ``experts.down_proj`` ``[experts, hidden, size]``. Returns our separate tensors, each loading
    its own rows, or None for any other tensor."""
    match = _HF_LAYER.match(tensor.name)
    fused = (
        "self_attn.qkv_proj.weight",
        "mlp.gate_up_proj.weight",
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
        "block_sparse_moe.input_linear.weight",  # Granite MoE: [experts, 2 * size, hidden] and [experts, hidden, size]
        "block_sparse_moe.output_linear.weight",
        "shared_mlp.input_linear.weight",  # Granite MoE's shared expert: gate and up rows stacked
    )
    if not match or match.group(2).removesuffix(".weight") not in {name.removesuffix(".weight") for name in fused}:
        return None
    if tensor.dtype not in ("F32", "F16", "BF16"):
        raise ModelImportError(f"tensor {tensor.name!r} has dtype {tensor.dtype}; only F32, F16 and BF16 import")
    p = f"layers.{int(match.group(1))}."
    kind = match.group(2).removesuffix(".weight")
    if kind.startswith(("mlp.experts.", "block_sparse_moe.")):
        return _split_experts(tensor, config, p, kind)
    if kind == "self_attn.qkv_proj":
        q, kv = config.heads * config.head_dim, config.kv_heads * config.head_dim
        parts = [("attention.q.weight", q), ("attention.k.weight", kv), ("attention.v.weight", kv)]
    elif kind == "shared_mlp.input_linear":
        size = config.shared_expert_intermediate_size or 0
        parts = [("mlp.shared.gate.weight", size), ("mlp.shared.up.weight", size)]
    else:
        parts = [("mlp.gate.weight", config.intermediate_size), ("mlp.up.weight", config.intermediate_size)]
    if len(tensor.shape) != 2 or tensor.shape[0] != sum(rows for _, rows in parts):
        raise ModelImportError(f"tensor {tensor.name!r} has shape {tensor.shape}, which does not split as expected")
    split: dict[str, TensorSource] = {}
    start = 0
    for name, rows in parts:

        def load(start: int = start, rows: int = rows) -> np.ndarray:
            return np.ascontiguousarray(tensor.to_float32()[start : start + rows])

        split[p + name] = TensorSource((rows, tensor.shape[1]), load, tensor.dtype)
        start += rows
    return split


def _split_experts(tensor: Any, config: TransformerConfig, p: str, kind: str) -> dict[str, TensorSource]:
    size, hidden = config.expert_size, config.hidden_size
    gate_up = kind in ("mlp.experts.gate_up_proj", "block_sparse_moe.input_linear")
    expected = (config.experts, 2 * size, hidden) if gate_up else (config.experts, hidden, size)
    if tuple(tensor.shape) != expected:
        raise ModelImportError(f"tensor {tensor.name!r} has shape {tensor.shape}, expected {expected}")
    parts = [("gate", 0, size), ("up", size, size)] if gate_up else [("down", 0, hidden)]
    split: dict[str, TensorSource] = {}
    for expert in range(config.experts):
        for name, start, rows in parts:

            def load(expert: int = expert, start: int = start, rows: int = rows) -> np.ndarray:
                return np.ascontiguousarray(tensor.to_float32()[expert, start : start + rows])

            shape = (rows, expected[2])
            split[f"{p}mlp.experts.{expert}.{name}.weight"] = TensorSource(shape, load, tensor.dtype)
    return split


def _hf_name(name: str, config: TransformerConfig) -> str | None:
    """Our name for a checkpoint tensor; None for buffers that carry no weights."""
    if name in _HF_GLOBAL_NAMES:
        return _HF_GLOBAL_NAMES[name]
    match = _HF_LAYER.match(name)
    if match:
        if match.group(2) == "self_attn.rotary_emb.inv_freq":
            return None
        extra = {"post": _HF_POST_NORM_NAMES, "sandwich": _HF_SANDWICH_NORM_NAMES}.get(config.norm_placement, {})
        names = {**_HF_LAYER_NAMES, **extra}
        if match.group(2) in names:
            return f"layers.{int(match.group(1))}.{names[match.group(2)]}"
        expert = _HF_EXPERT.match(match.group(2))
        if expert:
            projection = _HF_EXPERT_NAMES[expert.group(2)]
            return f"layers.{int(match.group(1))}.mlp.experts.{int(expert.group(1))}.{projection}.weight"
    raise ModelImportError(f"unexpected tensor {name!r}")


def _model_card(directory: Path) -> dict[str, str]:
    """``license``, ``license_name`` and ``license_link`` from the YAML front matter of the model card."""
    readme = directory / "README.md"
    if not readme.exists():
        return {}
    lines = readme.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    values: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, sep, value = line.partition(":")
        if sep and key in ("license", "license_name", "license_link") and value.strip():
            values[key] = value.strip().strip("'\"")
    return values


def _licence_file(directory: Path) -> str | None:
    for path in sorted(directory.iterdir()):
        if path.is_file() and re.fullmatch(r"(?i)licen[cs]e(\.(txt|md))?", path.name):
            return path.read_text(encoding="utf-8")
    return None


def _file_hashes(paths: list[Path], root: Path) -> list[dict[str, Any]]:
    entries = []
    for path in sorted(paths, key=lambda p: p.relative_to(root).as_posix()):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda s=stream: s.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
        entries.append(
            {"path": path.relative_to(root).as_posix(), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}
        )
    return entries


_POOLING_MODES = {"pooling_mode_lasttoken": "last_token", "pooling_mode_mean_tokens": "mean"}


def _embedding_settings(directory: Path) -> dict[str, Any] | None:
    """How a sentence-transformers model turns hidden states into an embedding (``modules.json``, the pooling
    module's ``config.json`` and the prompts of ``config_sentence_transformers.json``); None for other models."""
    modules = _read_json(directory / "modules.json")
    if not isinstance(modules, list) or not modules:
        return None
    pooling = next((m for m in modules if str(m.get("type", "")).endswith("Pooling")), None)
    if pooling is None:
        raise ModelImportError(f"{directory}: modules.json has no pooling module")
    settings = _read_json(directory / str(pooling.get("path", "")) / "config.json")
    chosen = [mode for key, mode in _POOLING_MODES.items() if settings.get(key)]
    modes = [key for key, value in settings.items() if key.startswith("pooling_mode") and value]
    if len(chosen) != 1 or len(modes) != 1:
        raise ModelImportError(f"{directory}: only mean or last-token pooling is supported, not {settings}")
    if not settings.get("include_prompt", True):
        raise ModelImportError(f"{directory}: pooling that leaves out the prompt is not supported")
    extra = _read_json(directory / "config_sentence_transformers.json")
    prompts = extra.get("prompts") or {}
    return {
        "pooling": chosen[0],
        "normalize": any(str(m.get("type", "")).endswith("Normalize") for m in modules),
        "prompts": {str(k): str(v) for k, v in sorted(prompts.items())},
        "default_prompt_name": extra.get("default_prompt_name"),
    }


def _convert_huggingface(directory: Path, context_length: int | None = None) -> _Converted:
    config_path = directory / "config.json"
    if not config_path.exists():
        raise ModelImportError(f"{directory}: no config.json")
    raw_config = _read_json(config_path)
    config = hf_config(raw_config, _read_json(directory / "generation_config.json"), context_length=context_length)
    pooling = _embedding_settings(directory)
    checkpoint = open_checkpoint(directory)
    if pooling is not None and "model.embed_tokens.weight" not in checkpoint:
        # Embedding models are often saved without the causal LM wrapper: no "model." prefix and no LM head.
        checkpoint = {
            f"model.{name}": dataclasses.replace(tensor, name=f"model.{name}") for name, tensor in checkpoint.items()
        }
    if pooling is not None and "lm_head.weight" not in checkpoint and not config.tie_word_embeddings:
        config = dataclasses.replace(config, tie_word_embeddings=True)  # no head to generate with: reuse the embedding

    tensors: dict[str, TensorSource] = {}
    for tensor in checkpoint.values():
        fused = _split_fused(tensor, config)
        if fused is not None:
            tensors.update(fused)
            continue
        name = _hf_name(tensor.name, config)
        if name is None:
            continue
        if name == "lm_head.weight" and config.tie_word_embeddings:
            embedding = checkpoint["model.embed_tokens.weight"]
            if not np.array_equal(tensor.to_float32().view("<u4"), embedding.to_float32().view("<u4")):
                raise ModelImportError("tie_word_embeddings is set but lm_head differs from the embedding")
            continue
        if tensor.dtype not in ("F32", "F16", "BF16"):
            raise ModelImportError(f"tensor {tensor.name!r} has dtype {tensor.dtype}; only F32, F16 and BF16 import")
        tensors[name] = TensorSource(tensor.shape, tensor.to_float32, tensor.dtype)

    tokenizer_config = _read_json(directory / "tokenizer_config.json")
    template_file = directory / "chat_template.jinja"
    template = tokenizer_config.pop("chat_template", None)
    if template_file.exists():
        template = template_file.read_text(encoding="utf-8")
    elif isinstance(template, list):  # named templates: keep the default one
        template = next((t["template"] for t in template if t.get("name") == "default"), None)
    tokenizer = None
    if (directory / "tokenizer.json").exists():
        tokenizer = {
            "format": "huggingface",
            "tokenizer_json": _read_json(directory / "tokenizer.json"),
            "tokenizer_config": tokenizer_config,
            "special_tokens_map": _read_json(directory / "special_tokens_map.json"),
        }

    source_files = [p for p in directory.iterdir() if p.is_file() and not p.name.endswith(".partial")]
    card = _model_card(directory)
    licence_id = card.get("license_name") if card.get("license") == "other" else card.get("license")
    return _Converted(
        config=config,
        tensors=tensors,
        source={"format": "safetensors", "files": _file_hashes(source_files, directory)},
        licence_id=licence_id,
        licence_text=_licence_file(directory),
        licence_link=card.get("license_link"),
        name=directory.name,
        tokenizer=tokenizer,
        chat_template=template,
        embedding=pooling,
    )


# ---------------------------------------------------------------------------------------------------------------
# GGUF


_GGUF_LAYER = re.compile(r"^blk\.(\d+)\.(.+)$")
_GGUF_LAYER_NAMES = {
    "attn_norm.weight": "attention_norm.weight",
    "ffn_norm.weight": "mlp_norm.weight",
    "attn_q.weight": "attention.q.weight",
    "attn_k.weight": "attention.k.weight",
    "attn_v.weight": "attention.v.weight",
    "attn_output.weight": "attention.o.weight",
    "attn_q.bias": "attention.q.bias",
    "attn_k.bias": "attention.k.bias",
    "attn_v.bias": "attention.v.bias",
    "attn_q_norm.weight": "attention.q_norm.weight",
    "attn_k_norm.weight": "attention.k_norm.weight",
    "ffn_gate.weight": "mlp.gate.weight",
    "ffn_up.weight": "mlp.up.weight",
    "ffn_down.weight": "mlp.down.weight",
    "ffn_gate_inp.weight": "mlp.router.weight",
    "ffn_gate_shexp.weight": "mlp.shared.gate.weight",
    "ffn_up_shexp.weight": "mlp.shared.up.weight",
    "ffn_down_shexp.weight": "mlp.shared.down.weight",
    "ffn_gate_inp_shexp.weight": "mlp.shared_gate.weight",  # [hidden] in llama.cpp, [1, hidden] here
}
# Stacked expert tensors [experts, out, in]; older Mixtral files store one tensor per expert (ffn_gate.<e>.weight).
_GGUF_EXPERTS = {"ffn_gate_exps.weight": "gate", "ffn_up_exps.weight": "up", "ffn_down_exps.weight": "down"}
_GGUF_EXPERT = re.compile(r"^(ffn_gate|ffn_up|ffn_down)\.(\d+)\.weight$")
# GGUF architecture -> family (a "llama" file with experts is Mixtral).
_GGUF_ARCHITECTURES = {
    "llama": "llama",
    "qwen2": "qwen2",
    "qwen2moe": "qwen2_moe",
    "qwen3": "qwen3",
    "qwen3moe": "qwen3_moe",
    "olmoe": "olmoe",
}
_GGUF_GLOBAL_NAMES = {
    "token_embd.weight": "token_embedding.weight",
    "output_norm.weight": "final_norm.weight",
    "output.weight": "lm_head.weight",
}


def _gguf_name(name: str) -> str:
    if name in _GGUF_GLOBAL_NAMES:
        return _GGUF_GLOBAL_NAMES[name]
    match = _GGUF_LAYER.match(name)
    if match and match.group(2) in _GGUF_LAYER_NAMES:
        return f"layers.{int(match.group(1))}.{_GGUF_LAYER_NAMES[match.group(2)]}"
    expert = _GGUF_EXPERT.match(match.group(2)) if match else None
    if match and expert:
        return f"layers.{int(match.group(1))}.mlp.experts.{int(expert.group(2))}.{expert.group(1)[4:]}.weight"
    if name == "rope_freqs.weight":
        raise ModelImportError("GGUF rope_freqs (Llama 3 RoPE scaling) is not supported yet")
    raise ModelImportError(f"unexpected tensor {name!r}")


def unpermute_rotary(weight: np.ndarray, heads: int) -> np.ndarray:
    """Undoes llama.cpp's Q/K row permutation for Llama GGUF files (``permute`` in ``convert_hf_to_gguf.py``),
    which reorders each head from the (i, i + d/2) rotary pairing to (2i, 2i + 1). Pure reindexing: no values
    change."""
    rows = weight.shape[0]
    return weight.reshape(heads, rows // heads // 2, 2, *weight.shape[1:]).swapaxes(1, 2).reshape(weight.shape)


def gguf_config(gguf: GgufFile) -> TransformerConfig:
    metadata = gguf.metadata
    architecture = metadata.get("general.architecture")
    if architecture not in _GGUF_ARCHITECTURES:
        supported = ", ".join(sorted(_GGUF_ARCHITECTURES))
        raise ModelImportError(f"GGUF architecture {architecture!r} is not supported (supported: {supported})")

    def key(name: str, default: Any = None) -> Any:
        value = metadata.get(f"{architecture}.{name}", default)
        if value is None:
            raise ModelImportError(f"GGUF metadata {architecture}.{name} is missing")
        return value

    experts = int(metadata.get(f"{architecture}.expert_count", 0))
    family = "mixtral" if architecture == "llama" and experts else _GGUF_ARCHITECTURES[architecture]

    hidden = int(key("embedding_length"))
    heads = int(key("attention.head_count"))
    head_dim = int(key("attention.key_length", hidden // heads))
    if int(key("rope.dimension_count", head_dim)) != head_dim:
        raise ModelImportError("partial rotary embeddings are not supported yet")
    scaling_type = metadata.get(f"{family}.rope.scaling.type", "none")
    if scaling_type == "none":
        scaling = None
    elif scaling_type == "linear":
        scaling = {"rope_type": "linear", "factor": float(key("rope.scaling.factor"))}
    elif scaling_type == "yarn":
        scaling = yarn_scaling(float(key("rope.scaling.factor")), int(key("rope.scaling.original_context_length")))
    else:
        raise ModelImportError(f"GGUF RoPE scaling {scaling_type!r} is not supported yet")
    vocabulary = metadata.get(f"{family}.vocab_size")
    if vocabulary is None:
        vocabulary = gguf["token_embd.weight"].shape[0] if "token_embd.weight" in gguf else None
    eos = _ids(metadata.get("tokenizer.ggml.eos_token_id"))
    for token in _ids(metadata.get("tokenizer.ggml.eot_token_id")):
        if token not in eos:
            eos.append(token)
    bos = metadata.get("tokenizer.ggml.bos_token_id")
    moe: dict[str, Any] = {}
    if experts:
        if architecture != "qwen2moe" and int(metadata.get(f"{architecture}.expert_shared_count", 0)):
            raise ModelImportError("shared experts are not supported")
        if int(metadata.get(f"{architecture}.expert_gating_func", 1)) != 1:
            raise ModelImportError("only softmax expert gating is supported")
        layers = int(key("block_count"))
        dense = tuple(i for i in range(layers) if f"blk.{i}.ffn_gate_inp.weight" not in gguf)
        size = metadata.get(f"{architecture}.expert_feed_forward_length")
        moe = {
            "experts": experts,
            "experts_per_token": int(key("expert_used_count")),
            "normalize_expert_weights": bool(
                metadata.get(f"{architecture}.expert_weights_norm", architecture != "olmoe")
            ),
            "expert_intermediate_size": None if size is None else int(size),
            "dense_layers": dense or None,
        }
        sparse = next((i for i in range(layers) if i not in dense), 0)
        if f"blk.{sparse}.ffn_gate_shexp.weight" in gguf:  # Qwen2-MoE's shared expert, gated
            shared = metadata.get(f"{architecture}.expert_shared_feed_forward_length")
            moe["shared_expert_intermediate_size"] = (
                int(shared) if shared is not None else gguf[f"blk.{sparse}.ffn_gate_shexp.weight"].shape[0]
            )
            moe["shared_expert_gate"] = f"blk.{sparse}.ffn_gate_inp_shexp.weight" in gguf
    return TransformerConfig(
        family=family,
        vocabulary_size=int(key("vocab_size", vocabulary)),
        hidden_size=hidden,
        intermediate_size=int(key("feed_forward_length")),
        layers=int(key("block_count")),
        heads=heads,
        kv_heads=int(key("attention.head_count_kv", heads)),
        head_dim=head_dim,
        context_length=int(key("context_length", 2048)),
        rms_norm_eps=float(key("attention.layer_norm_rms_epsilon")),
        rope_theta=float(key("rope.freq_base", 10000.0)),
        rope_scaling=scaling,
        attention_bias=family in ("qwen2", "qwen2_moe") or "blk.0.attn_q.bias" in gguf,
        qk_norm=family in ("olmoe", "qwen3", "qwen3_moe"),
        qk_norm_scope="all" if family == "olmoe" else "head",
        tie_word_embeddings="output.weight" not in gguf,
        bos_token_id=None if bos is None else int(bos),
        eos_token_ids=tuple(eos),
        **moe,
    )


def _convert_gguf(path: Path, context_length: int | None = None) -> _Converted:
    gguf = GgufFile(path)
    config = gguf_config(gguf)
    if context_length is not None:
        config = with_context_length(config, context_length)
    metadata = gguf.metadata
    tensors: dict[str, TensorSource] = {}
    for tensor in gguf:
        match = _GGUF_LAYER.match(tensor.name)
        if match and match.group(2) in _GGUF_EXPERTS:
            if len(tensor.shape) != 3 or tensor.shape[0] != config.experts:
                raise ModelImportError(
                    f"tensor {tensor.name!r} has shape {tensor.shape}, expected {config.experts} experts"
                )
            for expert in range(config.experts):
                part = tensor.item(expert)
                name = f"layers.{int(match.group(1))}.mlp.experts.{expert}.{_GGUF_EXPERTS[match.group(2)]}.weight"
                tensors[name] = TensorSource(part.shape, part.to_float32, part.type_name)
            continue
        name = _gguf_name(tensor.name)
        load: Callable[[], np.ndarray] = tensor.to_float32
        if config.family in ("llama", "mixtral") and re.search(r"attention\.[qk]\.(weight|bias)$", name):
            heads = config.heads if ".q." in name else config.kv_heads
            load = (lambda t, h: lambda: unpermute_rotary(t.to_float32(), h))(tensor, heads)
        shape = tuple(tensor.shape)
        if name.endswith(".shared_gate.weight") and len(shape) == 1:
            shape = (1, shape[0])
            load = (lambda t: lambda: t.to_float32().reshape(1, -1))(tensor)
        tensors[name] = TensorSource(shape, load, tensor.type_name)

    tokenizer = {"format": "gguf"}
    tokenizer.update(
        {k: v for k, v in metadata.items() if k.startswith("tokenizer.") and k != "tokenizer.chat_template"}
    )
    source: dict[str, Any] = {"format": "gguf", "files": _file_hashes([path], path.parent)}
    repository = metadata.get("general.source.huggingface.repository")
    if repository:
        source["repository"] = repository
    url = metadata.get("general.source.url") or metadata.get("general.url")
    if url:
        source["url"] = url
    return _Converted(
        config=config,
        tensors=tensors,
        source=source,
        licence_id=metadata.get("general.license.name")
        if metadata.get("general.license") == "other"
        else metadata.get("general.license"),
        licence_text=None,
        licence_link=metadata.get("general.license.link"),
        name=metadata.get("general.name") or path.stem,
        tokenizer=tokenizer if len(tokenizer) > 1 else None,
        chat_template=metadata.get("tokenizer.chat_template"),
    )


# ---------------------------------------------------------------------------------------------------------------
# Entry point


def _licence(
    converted: _Converted,
    repository: str | None,
    licence: str | None,
    licence_text: str | None,
    accept_licence: bool,
) -> dict[str, Any]:
    spdx = licence or converted.licence_id
    if not spdx:
        raise ModelImportError("the source does not state a licence; check the model card and pass --licence <spdx-id>")
    canonical = PERMISSIVE.get(spdx.lower())
    if canonical is None and not accept_licence:
        raise ModelImportError(
            f"licence {spdx!r} is not Apache-2.0 or MIT; read it and pass --accept-licence to import anyway "
            "(the converted model will be marked not redistributable)"
        )
    text = licence_text or converted.licence_text or STANDARD_TEXTS.get(canonical or "")
    if not text:
        raise ModelImportError(f"no text found for licence {spdx!r}; pass --licence-file")
    subject = repository or converted.name or "the imported model"
    link = f"https://huggingface.co/{repository}" if repository else converted.source.get("url")
    attribution = (
        f"{subject}{f' ({link})' if link else ''}, licensed under {canonical or spdx}. "
        "Converted to model.dllm by EtAlii.Dllm; the weights are otherwise unmodified."
    )
    record: dict[str, Any] = {
        "spdx": canonical or spdx,
        "text": text,
        "attribution": attribution,
        "redistributable": canonical is not None,
    }
    if converted.licence_link:
        record["link"] = converted.licence_link
    return record


def import_model(
    source: str | Path,
    output: str | Path,
    *,
    repository: str | None = None,
    revision: str | None = None,
    licence: str | None = None,
    licence_file: str | Path | None = None,
    accept_licence: bool = False,
    cache: str | Path | None = None,
    opener: hub.Opener | None = None,
    base: str | Path | None = None,
    context_length: int | None = None,
) -> ImportResult:
    """Imports ``source`` (a checkpoint directory, a ``.gguf`` file, or ``hf:org/name[@revision]``) into
    ``output``. ``repository``/``revision`` record provenance for local sources; ``licence`` overrides the licence
    the source states; ``licence_file`` supplies its text. With ``base`` (a ``model.dllm``), ``source`` is a PEFT
    LoRA adapter and ``output`` is the base model with the adapter merged into its weights. ``context_length``
    sets the model's context window (:func:`with_context_length`)."""
    source_text = str(source)
    if source_text.startswith("hf:"):
        repo, rev = hub.parse_reference(source_text)
        snapshot = hub.download(repo, rev, cache or Path.home() / ".cache" / "etalii-dllm" / "hub", opener)
        path, repository, revision = snapshot.directory, snapshot.repository, snapshot.revision
    else:
        path = Path(source)
    if (path / ADAPTER_CONFIG).exists():
        if context_length is not None:
            raise ModelImportError("--context-length is for model sources, not LoRA adapters")
        if base is None:
            raise ModelImportError(f"{source} is a LoRA adapter; pass --base <model.dllm> to merge it into")
        return _import_adapter(path, output, base, repository, revision, licence, accept_licence)
    if base is not None:
        raise ModelImportError(f"{source} is not a LoRA adapter (no {ADAPTER_CONFIG}); --base is for adapters")
    if path.is_dir():
        converted = _convert_huggingface(path, context_length)
    elif path.is_file() and path.suffix.lower() == ".gguf":
        converted = _convert_gguf(path, context_length)
    else:
        raise ModelImportError(f"{source}: expected a checkpoint directory, a .gguf file or hf:org/name")

    source_record = dict(converted.source)
    if repository:
        source_record["repository"] = repository
    if revision:
        source_record["revision"] = revision
    text = Path(licence_file).read_text(encoding="utf-8") if licence_file else None
    licence_record = _licence(converted, source_record.get("repository"), licence, text, accept_licence)

    fingerprint = write_model_file(
        output,
        converted.config,
        converted.tensors,
        {
            "source": source_record,
            "licence": licence_record,
            "tokenizer": converted.tokenizer,
            "chat_template": converted.chat_template,
            "embedding": converted.embedding,
            "lineage": [import_step(source_record)],
        },
    )
    return ImportResult(Path(output), fingerprint, converted.config, source_record, licence_record)


def _import_adapter(
    directory: Path,
    output: str | Path,
    base_path: str | Path,
    repository: str | None,
    revision: str | None,
    licence: str | None,
    accept_licence: bool,
) -> ImportResult:
    """Writes ``base_path`` with the PEFT adapter in ``directory`` merged into its weights (:mod:`etalii_dllm.lora`)."""
    base = ModelFile(base_path)
    try:
        weights, lora = apply_adapter(base.config, base.tensors, directory)
    except AdapterError as error:
        raise ModelImportError(str(error)) from error
    card = _model_card(directory)
    spdx = licence or card.get("license")
    if not spdx:
        raise ModelImportError("the adapter does not state a licence; check its card and pass --licence <spdx-id>")
    canonical = PERMISSIVE.get(spdx.lower())
    if canonical is None and not accept_licence:
        raise ModelImportError(
            f"adapter licence {spdx!r} is not Apache-2.0 or MIT; read it and pass --accept-licence to import anyway "
            "(the merged model will be marked not redistributable)"
        )
    subject = repository or directory.name
    record = dict(base.licence)
    attribution = str(record.get("attribution", ""))
    unmodified = "; the weights are otherwise unmodified."
    if attribution.endswith(unmodified):
        attribution = attribution[: -len(unmodified)] + "."
    record["attribution"] = (
        f"{attribution} LoRA adapter {subject}, licensed under {canonical or spdx}, merged by EtAlii.Dllm; "
        "modified weights."
    ).strip()
    record["redistributable"] = bool(record.get("redistributable")) and canonical is not None
    source: dict[str, Any] = {"format": "peft"}
    if repository:
        source["repository"] = repository
    if revision:
        source["revision"] = revision
    source["files"] = _file_hashes([p for p in adapter_files(directory) if p.exists()], directory)
    adapter = {
        "base_fingerprint": base.fingerprint,
        "lora": lora.to_dict(),
        "licence": canonical or spdx,
        "source": source,
    }
    metadata = {key: base.header.get(key) for key in ("source", "tokenizer", "chat_template", "fine_tuning")}
    metadata["licence"] = record
    metadata["adapter"] = adapter
    metadata["lineage"] = extend_lineage(lineage(base.header), base.fingerprint, adapter_step(adapter))
    tensors = {
        name: TensorSource(tuple(values.shape), lambda values=values: values) for name, values in weights.items()
    }
    fingerprint = write_model_file(output, base.config, tensors, metadata)
    return ImportResult(Path(output), fingerprint, base.config, dict(base.source), record)
