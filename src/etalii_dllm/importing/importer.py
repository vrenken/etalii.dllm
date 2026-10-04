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
from etalii_dllm.lora import ADAPTER_CONFIG, AdapterError, adapter_files, apply_adapter, dequantized_weights
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
    classifier: dict[str, Any] | None = None


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
        supported = ", ".join(sorted([*_HF_MODEL_TYPES, *_ENCODER_MODEL_TYPES]))
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
_HF_MODEL_TYPES = {name: name for name in FAMILIES if name not in ("gemma3", "bert")} | {
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


_POOLING_MODES = {
    "pooling_mode_lasttoken": "last_token",
    "pooling_mode_mean_tokens": "mean",
    "pooling_mode_cls_token": "cls",
}


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
        raise ModelImportError(f"{directory}: only mean, CLS or last-token pooling is supported, not {settings}")
    if not settings.get("include_prompt", True):
        raise ModelImportError(f"{directory}: pooling that leaves out the prompt is not supported")
    extra = _read_json(directory / "config_sentence_transformers.json")
    prompts = extra.get("prompts") or {}
    result: dict[str, Any] = {
        "pooling": chosen[0],
        "normalize": any(str(m.get("type", "")).endswith("Normalize") for m in modules),
        "prompts": {str(k): str(v) for k, v in sorted(prompts.items())},
        "default_prompt_name": extra.get("default_prompt_name"),
    }
    kinds = [str(m.get("type", "")).rsplit(".", 1)[-1] for m in modules]
    known = ("Transformer", "Pooling", "Dense", "Normalize")
    if any(kind not in known for kind in kinds) or kinds.count("Dense") > 1:
        raise ModelImportError(f"{directory}: only Transformer, Pooling, one Dense and Normalize modules are supported")
    if "Dense" in kinds:
        if kinds.index("Dense") != kinds.index("Pooling") + 1:
            raise ModelImportError(f"{directory}: the Dense module must follow the pooling")
        result["projection"] = str(modules[kinds.index("Dense")].get("path", ""))
    return result


_DENSE_ACTIVATIONS = {
    "torch.nn.modules.linear.Identity": "identity",
    "torch.nn.modules.activation.Tanh": "tanh",
}


def _projection(
    directory: Path, config: TransformerConfig, pooling: dict[str, Any]
) -> tuple[TransformerConfig, dict[str, TensorSource]]:
    """The sentence-transformers ``Dense`` module named by ``pooling["projection"]`` (its path): the config with
    ``projection_size``, the tensors ``projection.weight`` (and ``.bias``), and ``pooling["projection"]`` set to the
    activation (``identity`` or ``tanh``)."""
    path = directory / pooling["projection"]
    settings = _read_json(path / "config.json")
    activation = _DENSE_ACTIVATIONS.get(str(settings.get("activation_function", "torch.nn.modules.activation.Tanh")))
    if activation is None:
        raise ModelImportError(f"{path}: Dense activation {settings.get('activation_function')!r} is not supported")
    if int(settings.get("in_features", -1)) != config.hidden_size:
        raise ModelImportError(f"{path}: the Dense module does not read the encoder's {config.hidden_size} features")
    if not (path / "model.safetensors").exists():
        raise ModelImportError(f"{path}: the Dense module's weights must be in model.safetensors")
    stored = open_checkpoint(path)
    tensors: dict[str, TensorSource] = {}
    for name in ("weight", "bias"):
        tensor = stored.get(f"linear.{name}")
        if tensor is not None:
            tensors[f"projection.{name}"] = TensorSource(tensor.shape, tensor.to_float32, tensor.dtype)
    if "projection.weight" not in tensors or len(stored) != len(tensors):
        raise ModelImportError(f"{path}: expected linear.weight (and linear.bias) only")
    size = int(settings.get("out_features", tensors["projection.weight"].shape[0]))
    biased = "projection.bias" in tensors
    if bool(settings.get("bias", True)) != biased:
        raise ModelImportError(f"{path}: config.json and the weights disagree about the bias")
    pooling["projection"] = activation
    return dataclasses.replace(config, projection_size=size, projection_bias=biased), tensors


_BERT_ACTIVATIONS = {"gelu": "gelu", "gelu_new": "gelu_tanh", "gelu_pytorch_tanh": "gelu_tanh"}


_ENCODER_MODEL_TYPES = ("bert", "deberta-v2", "modernbert", "roberta", "t5", "xlm-roberta")
"""The ``model_type`` values imported as encoders: RoBERTa and XLM-RoBERTa are BERT with positions counted from
past the padding token; DeBERTa-v2/v3, ModernBERT and T5 (its encoder) are encoder families of their own."""


def bert_config(config: dict[str, Any], context_length: int | None = None) -> TransformerConfig:
    """Maps a Hugging Face BERT, RoBERTa or XLM-RoBERTa ``config.json`` to our description (family ``bert``; RoBERTa
    sets ``padding_index`` and has ``max_position_embeddings - pad_token_id - 1`` usable positions)."""
    if config.get("position_embedding_type", "absolute") != "absolute":
        raise ModelImportError(f"BERT position embeddings {config.get('position_embedding_type')!r} are not supported")
    hf_activation = str(config.get("hidden_act", "gelu"))
    if hf_activation not in _BERT_ACTIVATIONS:
        raise ModelImportError(f"activation {hf_activation!r} is not supported")
    heads, hidden = int(config["num_attention_heads"]), int(config["hidden_size"])
    if hidden % heads:
        raise ModelImportError("hidden_size must be a multiple of num_attention_heads")
    positions = int(config.get("max_position_embeddings", 512))
    padding: int | None = None
    if config.get("model_type") in ("roberta", "xlm-roberta"):
        padding = int(config.get("pad_token_id", 1))
        positions -= padding + 1
        if positions < 1:
            raise ModelImportError("max_position_embeddings leaves no positions past the padding token")
    if context_length is not None:
        raise ModelImportError("--context-length is for decoders; an encoder has learned absolute positions")
    try:
        return TransformerConfig(
            family="bert",
            vocabulary_size=int(config["vocab_size"]),
            hidden_size=hidden,
            intermediate_size=int(config["intermediate_size"]),
            layers=int(config["num_hidden_layers"]),
            heads=heads,
            kv_heads=heads,
            head_dim=hidden // heads,
            context_length=positions,
            rms_norm_eps=float(config.get("layer_norm_eps", 1e-12)),
            rope_theta=0.0,
            attention_bias=True,
            activation=_BERT_ACTIVATIONS[hf_activation],
            tie_word_embeddings=True,
            type_vocabulary_size=int(config.get("type_vocab_size", 2)),
            padding_index=padding,
        )
    except ValueError as error:
        raise ModelImportError(str(error)) from error


def modernbert_config(config: dict[str, Any], context_length: int | None = None) -> TransformerConfig:
    """Maps a Hugging Face ModernBERT ``config.json`` to our description (family ``modernbert``). Both the original
    keys (``global_rope_theta``, ``local_rope_theta``, ``global_attn_every_n_layers``) and transformers v5's
    (``layer_types``, ``rope_parameters``) are read. ``local_attention`` is the width of the local layers' window:
    a position sees the keys at most ``local_attention // 2`` away, so ``sliding_window`` (the keys closer than it)
    is ``local_attention // 2 + 1``."""
    for key in ("norm_bias", "attention_bias", "mlp_bias", "classifier_bias"):
        if config.get(key):
            raise ModelImportError(f"ModernBERT with {key} is not supported")
    hf_activation = str(config.get("hidden_activation", "gelu"))
    if hf_activation not in _BERT_ACTIVATIONS:
        raise ModelImportError(f"activation {hf_activation!r} is not supported")
    head_activation = str(config.get("classifier_activation", hf_activation))
    if _BERT_ACTIVATIONS.get(head_activation) != _BERT_ACTIVATIONS[hf_activation]:
        raise ModelImportError("a classifier activation that differs from the MLP's is not supported")
    if context_length is not None:
        raise ModelImportError("--context-length is for decoders")
    heads, hidden, layers = (
        int(config["num_attention_heads"]),
        int(config["hidden_size"]),
        int(config["num_hidden_layers"]),
    )
    if hidden % heads or (hidden // heads) % 2:
        raise ModelImportError("hidden_size must be a multiple of num_attention_heads with an even head_dim")
    types = config.get("layer_types")
    if types is None:
        every = int(config.get("global_attn_every_n_layers", 1))
        types = ["full_attention" if i % every == 0 else "sliding_attention" for i in range(layers)]
    if len(types) != layers or set(types) - {"full_attention", "sliding_attention"}:
        raise ModelImportError(f"unsupported layer_types {types}")
    rope = config.get("rope_parameters") or {}
    thetas: dict[str, float] = {}
    for kind, key, default in (
        ("full_attention", "global_rope_theta", 160000.0),
        ("sliding_attention", "local_rope_theta", 10000.0),
    ):
        parameters = rope.get(kind) or {}
        if parameters.get("rope_type", "default") != "default" or config.get("rope_scaling"):
            raise ModelImportError("ModernBERT with RoPE scaling is not supported")
        thetas[kind] = float(parameters.get("rope_theta", config.get(key, default)))
    local = [i for i, kind in enumerate(types) if kind == "sliding_attention"]
    pooling = str(config.get("classifier_pooling", "cls"))
    if pooling not in ("cls", "mean"):
        raise ModelImportError(f"classifier pooling {pooling!r} is not supported (cls or mean)")
    try:
        return TransformerConfig(
            family="modernbert",
            vocabulary_size=int(config["vocab_size"]),
            hidden_size=hidden,
            intermediate_size=int(config["intermediate_size"]),
            layers=layers,
            heads=heads,
            kv_heads=heads,
            head_dim=hidden // heads,
            context_length=int(config.get("max_position_embeddings", 8192)),
            rms_norm_eps=float(config.get("norm_eps", 1e-5)),
            rope_theta=thetas["full_attention"],
            activation=_BERT_ACTIVATIONS[hf_activation],
            tie_word_embeddings=True,
            sliding_window=int(config.get("local_attention", 128)) // 2 + 1 if local else None,
            sliding_window_layers=tuple(local) if local and len(local) < layers else None,
            local_rope_theta=thetas["sliding_attention"] if local else None,
            classifier_pooling=pooling,
        )
    except ValueError as error:
        raise ModelImportError(str(error)) from error


def deberta_config(config: dict[str, Any], context_length: int | None = None) -> TransformerConfig:
    """Maps a Hugging Face DeBERTa-v2/v3 ``config.json`` (``model_type`` ``deberta-v2``) to our description (family
    ``deberta``): relative attention over ``position_buckets`` log buckets reaching ``max_relative_positions`` (or
    ``max_position_embeddings``), with the key and query projections shared with the position terms
    (``share_att_key``), both ``c2p`` and ``p2c`` terms and the relative-embedding LayerNorm, as DeBERTa-v3 and the
    larger v2 models have them. Absolute positions, the convolution layer, a factorised embedding, ``z_steps`` and
    other attention variants are refused."""
    if context_length is not None:
        raise ModelImportError("--context-length is for decoders")
    hf_activation = str(config.get("hidden_act", "gelu"))
    if hf_activation not in _BERT_ACTIVATIONS:
        raise ModelImportError(f"activation {hf_activation!r} is not supported")
    if str(config.get("pooler_hidden_act", hf_activation)) != hf_activation:
        raise ModelImportError("a pooler activation that differs from the MLP's is not supported")
    heads, hidden = int(config["num_attention_heads"]), int(config["hidden_size"])
    if hidden % heads or int(config.get("attention_head_size", hidden // heads)) != hidden // heads:
        raise ModelImportError("hidden_size must be num_attention_heads times the head size")
    pos_types = config.get("pos_att_type") or []
    if isinstance(pos_types, str):
        pos_types = [part.strip() for part in pos_types.lower().split("|")]
    norms = [part.strip() for part in str(config.get("norm_rel_ebd", "none")).lower().split("|")]
    refusals = {
        "relative_attention": not config.get("relative_attention", False),
        "position_biased_input": config.get("position_biased_input", True),
        "share_att_key": not config.get("share_att_key", False),
        "pos_att_type": sorted(pos_types) != ["c2p", "p2c"],
        "norm_rel_ebd": "layer_norm" not in norms,
        "conv_kernel_size": int(config.get("conv_kernel_size", 0) or 0) > 0,
        "embedding_size": int(config.get("embedding_size", hidden) or hidden) != hidden,
        "pooler_hidden_size": int(config.get("pooler_hidden_size", hidden) or hidden) != hidden,
        "z_steps": int(config.get("z_steps", 0) or 0) > 1,
    }
    refused = [name for name, bad in refusals.items() if bad]
    if refused:
        raise ModelImportError(
            f"DeBERTa with this {', '.join(refused)} is not supported: relative attention with shared position "
            "projections, c2p|p2c, the relative LayerNorm and no absolute positions (DeBERTa-v3) is"
        )
    positions = int(config.get("max_position_embeddings", 512))
    max_relative = int(config.get("max_relative_positions", -1))
    try:
        return TransformerConfig(
            family="deberta",
            vocabulary_size=int(config["vocab_size"]),
            hidden_size=hidden,
            intermediate_size=int(config["intermediate_size"]),
            layers=int(config["num_hidden_layers"]),
            heads=heads,
            kv_heads=heads,
            head_dim=hidden // heads,
            context_length=positions,
            rms_norm_eps=float(config.get("layer_norm_eps", 1e-7)),
            rope_theta=0.0,
            attention_bias=True,
            activation=_BERT_ACTIVATIONS[hf_activation],
            tie_word_embeddings=True,
            type_vocabulary_size=int(config.get("type_vocab_size", 0)),
            position_buckets=max(int(config.get("position_buckets", -1)), 0),
            max_relative_positions=max_relative if max_relative >= 1 else positions,
        )
    except ValueError as error:
        raise ModelImportError(str(error)) from error


_T5_ACTIVATIONS = {"relu": "relu", "gelu": "gelu", "gelu_new": "gelu_tanh", "silu": "silu"}


def t5_config(config: dict[str, Any], context_length: int | None = None) -> TransformerConfig:
    """Maps a Hugging Face T5 ``config.json`` (``model_type`` ``t5``: T5EncoderModel, T5Model or
    T5ForConditionalGeneration, whose decoder is dropped) to our encoder description (family ``t5``): ``d_model``,
    ``d_kv`` per head, ``d_ff``, the relative attention buckets and their maximum distance, and ``feed_forward_proj``
    (``relu`` as in T5 v1.0, ``gated-gelu`` as in v1.1, which uses the tanh GELU, or another ``[gated-]relu``,
    ``gelu`` or ``silu``)."""
    if context_length is not None:
        raise ModelImportError("--context-length is for decoders")
    projection = str(config.get("feed_forward_proj", "relu"))
    gated = projection.startswith("gated-")
    name = "gelu_new" if projection == "gated-gelu" else projection.removeprefix("gated-")
    if name not in _T5_ACTIVATIONS:
        raise ModelImportError(f"feed_forward_proj {projection!r} is not supported")
    if not config.get("is_encoder_decoder", True) and config.get("is_decoder"):
        raise ModelImportError("a T5 decoder is not an encoder")
    try:
        return TransformerConfig(
            family="t5",
            vocabulary_size=int(config["vocab_size"]),
            hidden_size=int(config["d_model"]),
            intermediate_size=int(config["d_ff"]),
            layers=int(config["num_layers"]),
            heads=int(config["num_heads"]),
            kv_heads=int(config["num_heads"]),
            head_dim=int(config["d_kv"]),
            context_length=int(config.get("n_positions", 512)),
            rms_norm_eps=float(config.get("layer_norm_epsilon", 1e-6)),
            rope_theta=0.0,
            activation=_T5_ACTIVATIONS[name],
            tie_word_embeddings=True,
            position_buckets=int(config.get("relative_attention_num_buckets", 32)),
            max_relative_positions=int(config.get("relative_attention_max_distance", 128)),
            gated_mlp=gated,
        )
    except ValueError as error:
        raise ModelImportError(str(error)) from error


_T5_LAYER = re.compile(r"^encoder\.block\.(\d+)\.layer\.([01])\.(.+)\.weight$")
_T5_LAYER_NAMES = {
    ("0", "SelfAttention.q"): "attention.q",
    ("0", "SelfAttention.k"): "attention.k",
    ("0", "SelfAttention.v"): "attention.v",
    ("0", "SelfAttention.o"): "attention.o",
    ("0", "layer_norm"): "attention_norm",
    ("1", "layer_norm"): "mlp_norm",
    ("1", "DenseReluDense.wi"): "mlp.up",
    ("1", "DenseReluDense.wi_0"): "mlp.gate",
    ("1", "DenseReluDense.wi_1"): "mlp.up",
    ("1", "DenseReluDense.wo"): "mlp.down",
}


def _t5_name(name: str) -> str | None:
    """Our name for a T5 checkpoint tensor; None for the decoder, the LM head and the encoder's copy of the shared
    embedding (``encoder.embed_tokens``, the same parameter as ``shared``)."""
    if name == "shared.weight":
        return "token_embedding.weight"
    if name.startswith(("decoder.", "lm_head.")) or name == "encoder.embed_tokens.weight":
        return None
    if name == "encoder.final_layer_norm.weight":
        return "final_norm.weight"
    if name == "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight":
        return "relative_bias.weight"
    match = _T5_LAYER.match(name)
    if match and (match.group(2), match.group(3)) in _T5_LAYER_NAMES:
        return f"layers.{int(match.group(1))}.{_T5_LAYER_NAMES[(match.group(2), match.group(3))]}.weight"
    raise ModelImportError(f"unexpected tensor {name!r}")


_T5_DECODER_LAYER = re.compile(r"^decoder\.block\.(\d+)\.layer\.([012])\.(.+)\.weight$")
_T5_DECODER_LAYER_NAMES = {
    ("0", "SelfAttention.q"): "attention.q",
    ("0", "SelfAttention.k"): "attention.k",
    ("0", "SelfAttention.v"): "attention.v",
    ("0", "SelfAttention.o"): "attention.o",
    ("0", "layer_norm"): "attention_norm",
    ("1", "EncDecAttention.q"): "cross.q",
    ("1", "EncDecAttention.k"): "cross.k",
    ("1", "EncDecAttention.v"): "cross.v",
    ("1", "EncDecAttention.o"): "cross.o",
    ("1", "layer_norm"): "cross_norm",
    ("2", "layer_norm"): "mlp_norm",
    ("2", "DenseReluDense.wi"): "mlp.up",
    ("2", "DenseReluDense.wi_0"): "mlp.gate",
    ("2", "DenseReluDense.wi_1"): "mlp.up",
    ("2", "DenseReluDense.wo"): "mlp.down",
}


def _t5_text_to_text_name(name: str, tied: bool) -> str | None:
    """Our name for a tensor of a T5ForConditionalGeneration checkpoint: the encoder's as :func:`_t5_name`, the
    decoder's under ``decoder.``; None for the decoder's copy of the shared embedding and a tied LM head."""
    if name == "decoder.embed_tokens.weight" or (tied and name == "lm_head.weight"):
        return None
    if name == "lm_head.weight":
        return name
    if name == "decoder.final_layer_norm.weight":
        return "decoder.final_norm.weight"
    if name == "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight":
        return "decoder.relative_bias.weight"
    if name.startswith("decoder."):
        match = _T5_DECODER_LAYER.match(name)
        if match and (match.group(2), match.group(3)) in _T5_DECODER_LAYER_NAMES:
            stem = _T5_DECODER_LAYER_NAMES[(match.group(2), match.group(3))]
            return f"decoder.layers.{int(match.group(1))}.{stem}.weight"
        raise ModelImportError(f"unexpected tensor {name!r}")
    return _t5_name(name)


def _convert_t5_text_to_text(directory: Path, raw_config: dict[str, Any], context_length: int | None) -> _Converted:
    """A T5ForConditionalGeneration checkpoint (T5 v1.0, v1.1, Flan-T5) as a text-to-text model that keeps its
    decoder (:mod:`etalii_dllm.seq2seq`): ``num_decoder_layers`` decoder layers, the LM head tied to the shared
    embedding (``tie_word_embeddings``, T5 v1.0) or its own ``lm_head``, ``</s>`` as the end of the source."""
    config = t5_config(raw_config, context_length)
    if int(raw_config.get("decoder_start_token_id", 0)) != 0:
        raise ModelImportError("only T5 models whose decoder starts with token 0 (<pad>) are supported")
    if not (directory / "tokenizer.json").exists():
        raise ModelImportError(
            "this T5 checkpoint has no tokenizer.json (only spm.model); save it with a fast tokenizer first, "
            "for example AutoTokenizer.from_pretrained(dir).save_pretrained(dir)"
        )
    tied = bool(raw_config.get("tie_word_embeddings", True))
    eos = raw_config.get("eos_token_id", 1)
    try:
        config = dataclasses.replace(
            config,
            decoder_layers=int(raw_config.get("num_decoder_layers") or raw_config["num_layers"]),
            tie_word_embeddings=tied,
            eos_token_ids=(int(eos[0] if isinstance(eos, list) else eos),),
        )
    except ValueError as error:
        raise ModelImportError(str(error)) from error
    checkpoint = open_checkpoint(directory)
    if "shared.weight" not in checkpoint and "encoder.embed_tokens.weight" in checkpoint:
        embedding = checkpoint["encoder.embed_tokens.weight"]
        checkpoint = {**checkpoint, "shared.weight": dataclasses.replace(embedding, name="shared.weight")}
    tensors: dict[str, TensorSource] = {}
    for tensor in checkpoint.values():
        name = _t5_text_to_text_name(tensor.name, tied)
        if name is None:
            continue
        if tensor.dtype not in ("F32", "F16", "BF16"):
            raise ModelImportError(f"tensor {tensor.name!r} has dtype {tensor.dtype}; only F32, F16 and BF16 import")
        tensors[name] = TensorSource(tensor.shape, tensor.to_float32, tensor.dtype)
    return _with_tokenizer(directory, config, tensors, None)


_DEBERTA_LAYER = re.compile(r"^encoder\.layer\.(\d+)\.(.+)\.(weight|bias)$")
_DEBERTA_LAYER_NAMES = {
    "attention.self.query_proj": "attention.q",
    "attention.self.key_proj": "attention.k",
    "attention.self.value_proj": "attention.v",
    "attention.output.dense": "attention.o",
    "attention.output.LayerNorm": "attention_norm",
    "intermediate.dense": "mlp.up",
    "output.dense": "mlp.down",
    "output.LayerNorm": "mlp_norm",
}
_DEBERTA_GLOBAL_NAMES = {
    "embeddings.word_embeddings": "token_embedding",
    "embeddings.token_type_embeddings": "token_type_embedding",
    "embeddings.LayerNorm": "embedding_norm",
    "encoder.rel_embeddings": "relative_embedding",
    "encoder.LayerNorm": "relative_norm",
}


def _deberta_name(name: str, classifier: bool = False) -> str | None:
    """Our name for a DeBERTa-v2/v3 checkpoint tensor (with or without the ``deberta.`` prefix); None for the
    pre-training heads, buffers and unused absolute positions, and for the pooler unless the model is a
    ``classifier``."""
    name = name.removeprefix("deberta.")
    stem, _, parameter = name.rpartition(".")
    if stem in ("pooler.dense", "classifier") and parameter in ("weight", "bias"):
        return f"{'pooler' if stem == 'pooler.dense' else 'classifier'}.{parameter}" if classifier else None
    if name.startswith(("lm_predictions.", "mask_predictions.", "cls.", "lm_head.")) or name in (
        "embeddings.position_ids",
        "embeddings.position_embeddings.weight",
    ):
        return None
    if stem in _DEBERTA_GLOBAL_NAMES and parameter in ("weight", "bias"):
        return f"{_DEBERTA_GLOBAL_NAMES[stem]}.{parameter}"
    match = _DEBERTA_LAYER.match(name)
    if match and match.group(2) in _DEBERTA_LAYER_NAMES:
        return f"layers.{int(match.group(1))}.{_DEBERTA_LAYER_NAMES[match.group(2)]}.{match.group(3)}"
    raise ModelImportError(f"unexpected tensor {name!r}")


_MODERNBERT_LAYER = re.compile(r"^layers\.(\d+)\.(.+)\.weight$")
_MODERNBERT_LAYER_NAMES = {
    "attn_norm": "attention_norm",
    "attn.Wo": "attention.o",
    "mlp_norm": "mlp_norm",
    "mlp.Wo": "mlp.down",
}
_MODERNBERT_GLOBAL_NAMES = {
    "embeddings.tok_embeddings.weight": "token_embedding.weight",
    "embeddings.norm.weight": "embedding_norm.weight",
    "final_norm.weight": "final_norm.weight",
}
_MODERNBERT_HEAD_NAMES = {
    "head.dense.weight": "pooler.weight",
    "head.norm.weight": "pooler_norm.weight",
    "classifier.weight": "classifier.weight",
    "classifier.bias": "classifier.bias",
}


def _modernbert_tensors(tensor: Any, config: TransformerConfig, classifier: bool) -> dict[str, TensorSource]:
    """Our tensors for one ModernBERT checkpoint tensor (with or without the ``model.`` prefix): ``Wqkv`` splits
    into the q, k and v rows, ``Wi`` into the activated (gate) half and the multiplied (up) half; the head and the
    classifier only for a ``classifier``, and nothing for the masked-language-model head."""
    name = tensor.name.removeprefix("model.")
    if tensor.dtype not in ("F32", "F16", "BF16"):
        raise ModelImportError(f"tensor {tensor.name!r} has dtype {tensor.dtype}; only F32, F16 and BF16 import")
    source = TensorSource(tensor.shape, tensor.to_float32, tensor.dtype)
    if name in _MODERNBERT_HEAD_NAMES:
        return {_MODERNBERT_HEAD_NAMES[name]: source} if classifier else {}
    if name.startswith(("head.", "decoder.")):
        return {}
    if name in _MODERNBERT_GLOBAL_NAMES:
        return {_MODERNBERT_GLOBAL_NAMES[name]: source}
    match = _MODERNBERT_LAYER.match(name)
    if match is None:
        raise ModelImportError(f"unexpected tensor {tensor.name!r}")
    p, kind = f"layers.{int(match.group(1))}.", match.group(2)
    if kind in _MODERNBERT_LAYER_NAMES:
        return {f"{p}{_MODERNBERT_LAYER_NAMES[kind]}.weight": source}
    if kind == "attn.Wqkv":
        rows = config.heads * config.head_dim
        parts = ["attention.q.weight", "attention.k.weight", "attention.v.weight"]
    elif kind == "mlp.Wi":
        rows = config.intermediate_size
        parts = ["mlp.gate.weight", "mlp.up.weight"]
    else:
        raise ModelImportError(f"unexpected tensor {tensor.name!r}")
    if len(tensor.shape) != 2 or tensor.shape[0] != rows * len(parts):
        raise ModelImportError(f"tensor {tensor.name!r} has shape {tensor.shape}, which does not split as expected")
    split: dict[str, TensorSource] = {}
    for index, part in enumerate(parts):

        def load(start: int = index * rows) -> np.ndarray:
            return np.ascontiguousarray(tensor.to_float32()[start : start + rows])

        split[p + part] = TensorSource((rows, tensor.shape[1]), load, tensor.dtype)
    return split


_BERT_LAYER = re.compile(r"^encoder\.layer\.(\d+)\.(.+)\.(weight|bias|gamma|beta)$")
_BERT_LAYER_NAMES = {
    "attention.self.query": "attention.q",
    "attention.self.key": "attention.k",
    "attention.self.value": "attention.v",
    "attention.output.dense": "attention.o",
    "attention.output.LayerNorm": "attention_norm",
    "intermediate.dense": "mlp.up",
    "output.dense": "mlp.down",
    "output.LayerNorm": "mlp_norm",
}
_BERT_GLOBAL_NAMES = {
    "embeddings.word_embeddings": "token_embedding",
    "embeddings.position_embeddings": "position_embedding",
    "embeddings.token_type_embeddings": "token_type_embedding",
    "embeddings.LayerNorm": "embedding_norm",
}
_BERT_PARAMETERS = {"weight": "weight", "bias": "bias", "gamma": "weight", "beta": "bias"}


# BERT's pooler and classifier, and RoBERTa's classification head (dense, tanh, out_proj: the same computation).
_BERT_HEAD_NAMES = {
    "pooler.dense": "pooler",
    "classifier": "classifier",
    "classifier.dense": "pooler",
    "classifier.out_proj": "classifier",
}


def _bert_name(name: str, classifier: bool = False) -> str | None:
    """Our name for a BERT, RoBERTa or XLM-RoBERTa checkpoint tensor (with or without the ``bert.``/``roberta.``
    prefix); None for the pre-training heads and buffers, and for the pooler unless the model is a ``classifier``,
    which an embedding does not use."""
    name = name.removeprefix("bert.").removeprefix("roberta.")
    stem, _, parameter = name.rpartition(".")
    if classifier and stem in _BERT_HEAD_NAMES and parameter in ("weight", "bias"):
        return f"{_BERT_HEAD_NAMES[stem]}.{parameter}"
    if name.startswith(("pooler.", "cls.", "lm_head.")) or name in (
        "embeddings.position_ids",
        "embeddings.token_type_ids",
    ):
        return None
    if stem in _BERT_GLOBAL_NAMES and parameter in _BERT_PARAMETERS:
        return f"{_BERT_GLOBAL_NAMES[stem]}.{_BERT_PARAMETERS[parameter]}"
    match = _BERT_LAYER.match(name)
    if match and match.group(2) in _BERT_LAYER_NAMES:
        return f"layers.{int(match.group(1))}.{_BERT_LAYER_NAMES[match.group(2)]}.{_BERT_PARAMETERS[match.group(3)]}"
    raise ModelImportError(f"unexpected tensor {name!r}")


_CROSS_ENCODER_ACTIVATIONS = {
    "torch.nn.modules.activation.Sigmoid": "sigmoid",
    "torch.nn.modules.linear.Identity": "none",
}


def _classifier_settings(directory: Path, raw_config: dict[str, Any], positions: int) -> dict[str, Any]:
    """A sequence-classification model's labels (``id2label`` in id order), the activation sentence-transformers'
    CrossEncoder applies to its scores (``none``: the raw logits, as transformers gives them; ``sigmoid``) and how
    many tokens a pair may have (``max_length`` or the tokenizer's ``model_max_length``, at most the positions)."""
    labels = raw_config.get("id2label") or {}
    count = int(raw_config.get("num_labels") or len(labels) or 1)
    names = [str(labels.get(str(i), labels.get(i, f"LABEL_{i}"))) for i in range(count)]
    extra = raw_config.get("sentence_transformers") or {}
    function = extra.get("activation_fn") or raw_config.get("sbert_ce_default_activation_function")
    if function is not None and function not in _CROSS_ENCODER_ACTIVATIONS:
        raise ModelImportError(f"cross-encoder activation {function!r} is not supported (Sigmoid or Identity)")
    limit = _read_json(directory / "sentence_bert_config.json").get("max_seq_length")
    if not limit:
        limit = _read_json(directory / "tokenizer_config.json").get("model_max_length")
    if not isinstance(limit, int) or limit < 1 or limit > positions:
        limit = positions
    return {
        "labels": names,
        "activation": _CROSS_ENCODER_ACTIVATIONS[function] if function else "none",
        "max_tokens": limit,
    }


def _convert_bert(directory: Path, raw_config: dict[str, Any], context_length: int | None) -> _Converted:
    """A BERT, RoBERTa or XLM-RoBERTa checkpoint (the bare model or a pre-training model; sentence-transformers
    encoders such as all-MiniLM-L6-v2, bge-small-en-v1.5, all-distilroberta-v1 and
    paraphrase-multilingual-MiniLM-L12-v2), a ModernBERT one (gte-modernbert-base, modernbert-embed-base), or a
    sequence-classification cross-encoder such as ms-marco-MiniLM-L6-v2 or gte-reranker-modernbert-base; DeBERTa-v2
    and v3 likewise (nli-deberta-v3-small, mxbai-rerank-xsmall-v1)."""
    modern = raw_config.get("model_type") == "modernbert"
    deberta = raw_config.get("model_type") == "deberta-v2"
    t5 = raw_config.get("model_type") == "t5"
    generator = "T5ForConditionalGeneration" in (raw_config.get("architectures") or [])
    if t5 and generator and not (directory / "modules.json").exists():  # a sentence-transformers T5 embeds
        return _convert_t5_text_to_text(directory, raw_config, context_length)
    mapping = modernbert_config if modern else deberta_config if deberta else t5_config if t5 else bert_config
    config = mapping(raw_config, context_length)
    if (deberta or t5) and not (directory / "tokenizer.json").exists():
        family = "DeBERTa" if deberta else "T5"
        raise ModelImportError(
            f"this {family} checkpoint has no tokenizer.json (only spm.model); save it with a fast tokenizer first, "
            "for example AutoTokenizer.from_pretrained(dir).save_pretrained(dir)"
        )
    checkpoint = open_checkpoint(directory)
    classifier: dict[str, Any] | None = None
    pooling: dict[str, Any] | None = None
    head = next((name for name in ("classifier.weight", "classifier.out_proj.weight") if name in checkpoint), None)
    if head is not None:
        classifier = _classifier_settings(directory, raw_config, config.context_length)
        rows, named = checkpoint[head].shape[0], len(classifier["labels"])
        if rows != named:
            raise ModelImportError(f"the classifier has {rows} labels but config.json names {named}")
        config = dataclasses.replace(config, classifier_labels=rows)
    else:
        pooling = _embedding_settings(directory) or {
            "pooling": "mean",
            "normalize": True,
            "prompts": {},
            "default_prompt_name": None,
        }
        limit = _read_json(directory / "sentence_bert_config.json").get("max_seq_length")
        pooling["max_tokens"] = min(int(limit), config.context_length) if limit else config.context_length
    if modern and classifier is None:
        config = dataclasses.replace(config, classifier_pooling=None)
    tensors: dict[str, TensorSource] = {}
    if pooling is not None and "projection" in pooling:
        config, tensors = _projection(directory, config, pooling)
    if t5 and classifier is not None:
        raise ModelImportError("T5 sequence classification is not supported; T5 imports as an embedder")
    if t5 and "shared.weight" not in checkpoint and "encoder.embed_tokens.weight" in checkpoint:
        embedding = checkpoint["encoder.embed_tokens.weight"]
        checkpoint = {**checkpoint, "shared.weight": dataclasses.replace(embedding, name="shared.weight")}
    for tensor in checkpoint.values():
        if modern:
            tensors.update(_modernbert_tensors(tensor, config, classifier is not None))
            continue
        if t5:
            name = _t5_name(tensor.name)
        else:
            name = (_deberta_name if deberta else _bert_name)(tensor.name, classifier is not None)
        if name is None:
            continue
        if tensor.dtype not in ("F32", "F16", "BF16"):
            raise ModelImportError(f"tensor {tensor.name!r} has dtype {tensor.dtype}; only F32, F16 and BF16 import")
        tensors[name] = TensorSource(tensor.shape, tensor.to_float32, tensor.dtype)
    converted = _with_tokenizer(directory, config, tensors, pooling)
    return dataclasses.replace(converted, classifier=classifier) if classifier else converted


def _convert_huggingface(directory: Path, context_length: int | None = None) -> _Converted:
    config_path = directory / "config.json"
    if not config_path.exists():
        raise ModelImportError(f"{directory}: no config.json")
    raw_config = _read_json(config_path)
    if raw_config.get("model_type") in _ENCODER_MODEL_TYPES:
        return _convert_bert(directory, raw_config, context_length)
    config = hf_config(raw_config, _read_json(directory / "generation_config.json"), context_length=context_length)
    pooling = _embedding_settings(directory)
    if pooling is not None and "projection" in pooling:
        raise ModelImportError(f"{directory}: a Dense module after the pooling is supported for encoders only")
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
    return _with_tokenizer(directory, config, tensors, pooling)


def _with_tokenizer(
    directory: Path, config: TransformerConfig, tensors: dict[str, TensorSource], pooling: dict[str, Any] | None
) -> _Converted:
    """A converted checkpoint directory: the weights plus its tokenizer, chat template, licence and source files."""
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
                metadata.get(f"{architecture}.expert_weights_norm", architecture not in ("olmoe", "qwen2moe"))
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


def _gguf_bert(gguf: GgufFile, path: Path, context_length: int | None) -> _Converted:
    """A GGUF file in llama.cpp's ``bert`` layout (as ``dllm export`` writes it, or as llama.cpp's converter does:
    WordPiece vocabularies; :mod:`etalii_dllm.encoder_export`). The tokenizer is ``tokenizer.huggingface.json`` when
    the file has one, else the WordPiece vocabulary with BERT's lower-casing normaliser, as llama.cpp tokenizes."""
    from etalii_dllm.encoder_export import (
        GGUF_ENCODER_LAYER_NAMES,
        GGUF_ENCODER_NAMES,
        GGUF_POOLING,
        wordpiece_vocabulary,
    )

    metadata = gguf.metadata
    if context_length is not None:
        raise ModelImportError("--context-length is for decoders; an encoder has learned absolute positions")
    if metadata.get("bert.attention.causal", False):
        raise ModelImportError("a causal bert GGUF is not an encoder")

    def key(name: str) -> Any:
        value = metadata.get(f"bert.{name}")
        if value is None:
            raise ModelImportError(f"GGUF metadata bert.{name} is missing")
        return value

    by_gguf = {gguf_name: ours for ours, gguf_name in GGUF_ENCODER_NAMES.items()}
    for layer in range(int(key("block_count"))):
        for ours, gguf_name in GGUF_ENCODER_LAYER_NAMES.items():
            for parameter in ("weight", "bias"):
                by_gguf[f"blk.{layer}.{gguf_name}.{parameter}"] = f"layers.{layer}.{ours}.{parameter}"
    tensors: dict[str, TensorSource] = {}
    for tensor in gguf:
        if tensor.name not in by_gguf:
            raise ModelImportError(f"unexpected tensor {tensor.name!r}")
        tensors[by_gguf[tensor.name]] = TensorSource(tuple(tensor.shape), tensor.to_float32, tensor.type_name)
    if ("classifier.weight" in tensors) != ("pooler.weight" in tensors):
        raise ModelImportError("a classifier without its pooler (cls) is not supported")
    labels = tensors["classifier.weight"].shape[0] if "classifier.weight" in tensors else 0
    hidden, heads = int(key("embedding_length")), int(key("attention.head_count"))
    eps = metadata.get("dllm.attention.layer_norm_epsilon", key("attention.layer_norm_epsilon"))
    try:
        config = TransformerConfig(
            family="bert",
            vocabulary_size=tensors["token_embedding.weight"].shape[0],
            hidden_size=hidden,
            intermediate_size=int(key("feed_forward_length")),
            layers=int(key("block_count")),
            heads=heads,
            kv_heads=heads,
            head_dim=hidden // heads,
            context_length=int(key("context_length")),
            rms_norm_eps=float(eps),
            rope_theta=0.0,
            attention_bias=True,
            activation="gelu",
            tie_word_embeddings=True,
            type_vocabulary_size=tensors["token_type_embedding.weight"].shape[0],
            classifier_labels=labels,
        )
    except (KeyError, ValueError) as error:
        raise ModelImportError(f"the bert GGUF does not describe an encoder: {error}") from error
    if "tokenizer.huggingface.json" in metadata:
        spec = json.loads(str(metadata["tokenizer.huggingface.json"]))
    elif metadata.get("tokenizer.ggml.model") == "bert":
        spec = _wordpiece_spec(metadata, wordpiece_vocabulary)
    else:
        raise ModelImportError(f"GGUF bert tokenizer {metadata.get('tokenizer.ggml.model')!r} is not supported")
    tokenizer = {"format": "huggingface", "tokenizer_json": spec, "tokenizer_config": {}, "special_tokens_map": {}}
    embedding: dict[str, Any] | None = None
    classifier: dict[str, Any] | None = None
    if labels:
        names = [str(label) for label in metadata.get("bert.classifier.output_labels") or []]
        classifier = {"labels": names or [f"LABEL_{i}" for i in range(labels)], "activation": "none"}
        classifier["max_tokens"] = config.context_length
        classifier |= json.loads(str(metadata.get("dllm.classifier", "{}")))
    else:
        pooling = {value: mode for mode, value in GGUF_POOLING.items()}.get(int(metadata.get("bert.pooling_type", 1)))
        if pooling is None:
            raise ModelImportError(f"GGUF pooling type {metadata.get('bert.pooling_type')} is not supported")
        embedding = {"pooling": pooling, "normalize": True, "prompts": {}, "default_prompt_name": None}
        embedding["max_tokens"] = config.context_length
        embedding |= json.loads(str(metadata.get("dllm.embedding", "{}")))
    source: dict[str, Any] = {"format": "gguf", "files": _file_hashes([path], path.parent)}
    return _Converted(
        config=config,
        tensors=tensors,
        source=source,
        licence_id=metadata.get("general.license"),
        licence_text=None,
        licence_link=metadata.get("general.license.link"),
        name=metadata.get("general.name") or path.stem,
        tokenizer=tokenizer,
        chat_template=None,
        embedding=embedding,
        classifier=classifier,
    )


def _wordpiece_spec(metadata: dict[str, Any], pieces_of: Callable[..., list[str]]) -> dict[str, Any]:
    """A ``tokenizer.json`` for llama.cpp's ``bert`` vocabulary: WordPiece pieces, BERT's normaliser (lower case,
    accents stripped, Chinese characters split) and pre-tokenizer, ``[CLS] A [SEP] B [SEP]``."""
    tokens = [str(token) for token in metadata["tokenizer.ggml.tokens"]]
    types = [int(kind) for kind in metadata.get("tokenizer.ggml.token_type", [1] * len(tokens))]
    pieces = pieces_of(tokens, types)

    def special(field: str, fallback: str) -> tuple[str, int]:
        index = metadata.get(f"tokenizer.ggml.{field}")
        if index is None:
            if fallback not in pieces:
                raise ModelImportError(f"the bert GGUF names no {field} and has no {fallback}")
            index = pieces.index(fallback)
        return pieces[int(index)], int(index)

    cls = special("cls_token_id" if "tokenizer.ggml.cls_token_id" in metadata else "bos_token_id", "[CLS]")
    sep = special("seperator_token_id", "[SEP]")
    unknown = special("unknown_token_id", "[UNK]")[0]
    added = [
        {"id": i, "content": piece, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False,
         "special": True}
        for i, (piece, kind) in enumerate(zip(pieces, types, strict=True))
        if kind == 3
    ]  # fmt: skip
    return {
        "added_tokens": added,
        "normalizer": {
            "type": "BertNormalizer",
            "clean_text": True,
            "handle_chinese_chars": True,
            "strip_accents": None,
            "lowercase": True,
        },
        "pre_tokenizer": {"type": "BertPreTokenizer"},
        "post_processor": {"type": "BertProcessing", "sep": list(sep), "cls": list(cls)},
        "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": True},
        "model": {
            "type": "WordPiece",
            "unk_token": unknown,
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 100,
            "vocab": {piece: i for i, piece in enumerate(pieces)},
        },
    }


def _convert_gguf(path: Path, context_length: int | None = None) -> _Converted:
    gguf = GgufFile(path)
    if gguf.metadata.get("general.architecture") == "bert":
        return _gguf_bert(gguf, path, context_length)
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
    base_quantize: str | None = None,
) -> ImportResult:
    """Imports ``source`` (a checkpoint directory, a ``.gguf`` file, or ``hf:org/name[@revision]``) into
    ``output``. ``repository``/``revision`` record provenance for local sources; ``licence`` overrides the licence
    the source states; ``licence_file`` supplies its text. With ``base`` (a ``model.dllm``), ``source`` is a PEFT
    LoRA adapter and ``output`` is the base model with the adapter merged into its weights. ``context_length``
    sets the model's context window (:func:`with_context_length`). ``base_quantize`` (``q8_0``/``q4_0``) merges the
    adapter into the base as a quantised LoRA run defines it (:func:`~etalii_dllm.lora.dequantized_weights`)."""
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
        return _import_adapter(path, output, base, repository, revision, licence, accept_licence, base_quantize)
    if base_quantize is not None:
        raise ModelImportError("--base-quantize is for merging a LoRA adapter (with --base)")
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
            "classifier": converted.classifier,
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
    base_quantize: str | None = None,
) -> ImportResult:
    """Writes ``base_path`` with the PEFT adapter in ``directory`` merged into its weights (:mod:`etalii_dllm.lora`);
    with ``base_quantize``, into its dequantised weights (the base of a quantised LoRA run)."""
    base = ModelFile(base_path)
    try:
        frozen = base.tensors if base_quantize is None else dequantized_weights(base.tensors, base_quantize)
        weights, lora = apply_adapter(base.config, frozen, directory)
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
    if base_quantize is not None:
        adapter["base_quantize"] = base_quantize
    metadata = {key: base.header.get(key) for key in ("source", "tokenizer", "chat_template", "fine_tuning")}
    metadata |= {key: base.header[key] for key in ("embedding", "classifier") if base.header.get(key) is not None}
    metadata["licence"] = record
    metadata["adapter"] = adapter
    metadata["lineage"] = extend_lineage(lineage(base.header), base.fingerprint, adapter_step(adapter))
    tensors = {
        name: TensorSource(tuple(values.shape), lambda values=values: values) for name, values in weights.items()
    }
    fingerprint = write_model_file(output, base.config, tensors, metadata)
    return ImportResult(Path(output), fingerprint, base.config, dict(base.source), record)
