"""``dllm import``: converts a Hugging Face checkpoint (safetensors) or a GGUF file to ``model.dllm``.

The source architecture is mapped onto :class:`~etalii_dllm.architecture.TransformerConfig` and our tensor names;
anything the decoder does not implement fails the import loudly instead of producing a model that runs wrongly.
Float tensors are widened to float32 exactly; quantised GGUF tensors are dequantised with the reference formulas.
The source files' hashes, the repository and revision, and the licence text and attribution go into the file.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from etalii_dllm.architecture import TransformerConfig
from etalii_dllm.importing import hub
from etalii_dllm.importing.gguf import GgufFile
from etalii_dllm.importing.licences import PERMISSIVE, STANDARD_TEXTS
from etalii_dllm.importing.safetensors import open_checkpoint
from etalii_dllm.lora import ADAPTER_CONFIG, AdapterError, adapter_files, apply_adapter
from etalii_dllm.modelfile import ModelFile, TensorSource, write_model_file


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
}
# OLMo 2 normalises the outputs of attention and the MLP instead of their inputs.
_HF_POST_NORM_NAMES = {
    "post_attention_layernorm.weight": "attention_post_norm.weight",
    "post_feedforward_layernorm.weight": "mlp_post_norm.weight",
}
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
        scaling: dict[str, Any] | None = {k: v for k, v in parameters.items() if k != "rope_theta"}
    else:
        theta = config.get("rope_theta", 10000.0)
        scaling = config.get("rope_scaling")
    if scaling:
        scaling = dict(scaling)
        kind = scaling.pop("rope_type", None) or scaling.pop("type", None) or "default"
        if kind == "default":
            scaling = None
        elif kind in ("linear", "llama3"):
            scaling = {"rope_type": kind, **scaling}
        else:
            raise ModelImportError(f"RoPE scaling {kind!r} is not supported yet")
    return float(theta), scaling or None


def hf_config(config: dict[str, Any], generation: dict[str, Any] | None = None) -> TransformerConfig:
    """Maps a Hugging Face ``config.json`` (Granite, Llama, Mistral, OLMo 2, Qwen2 or Qwen3) to our description."""
    family = config.get("model_type")
    if family not in ("granite", "llama", "mistral", "olmo2", "qwen2", "qwen3"):
        raise ModelImportError(
            f"model_type {family!r} is not supported (supported: granite, llama, mistral, olmo2, qwen2, qwen3)"
        )
    if config.get("hidden_act", "silu") != "silu":
        raise ModelImportError(f"activation {config.get('hidden_act')!r} is not supported")
    if config.get("mlp_bias"):
        raise ModelImportError("MLP biases are not supported")
    window, window_layers = _sliding_window_from_hf(config, family)
    heads = int(config["num_attention_heads"])
    hidden = int(config["hidden_size"])
    theta, scaling = _rope_from_hf(config)
    eos = _ids(config.get("eos_token_id"))
    for token in _ids((generation or {}).get("eos_token_id")):
        if token not in eos:
            eos.append(token)
    bos = config.get("bos_token_id")
    return TransformerConfig(
        family=family,
        vocabulary_size=int(config["vocab_size"]),
        hidden_size=hidden,
        intermediate_size=int(config["intermediate_size"]),
        layers=int(config["num_hidden_layers"]),
        heads=heads,
        kv_heads=int(config.get("num_key_value_heads") or heads),
        head_dim=int(config.get("head_dim") or hidden // heads),
        context_length=int(config.get("max_position_embeddings", 2048)),
        rms_norm_eps=float(config.get("rms_norm_eps", 1e-6)),
        rope_theta=theta,
        rope_scaling=scaling,
        attention_bias=family == "qwen2" or bool(config.get("attention_bias", False)),
        qk_norm=family in ("olmo2", "qwen3"),
        qk_norm_scope="all" if family == "olmo2" else "head",
        norm_placement="post" if family == "olmo2" else "pre",
        tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
        bos_token_id=None if bos is None else int(bos),
        eos_token_ids=tuple(eos),
        sliding_window=window,
        sliding_window_layers=window_layers,
        **_granite_multipliers(config, family),
    )


def _granite_multipliers(config: dict[str, Any], family: str) -> dict[str, Any]:
    if family != "granite":
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
    if family in ("qwen2", "qwen3") and not config.get("use_sliding_window"):
        return None, None
    layers = int(config["num_hidden_layers"])
    layer_types = config.get("layer_types")
    if isinstance(layer_types, list):
        if len(layer_types) != layers or any(t not in ("sliding_attention", "full_attention") for t in layer_types):
            raise ModelImportError(f"layer_types {layer_types!r} is not supported")
        sliding = tuple(i for i, kind in enumerate(layer_types) if kind == "sliding_attention")
    elif family in ("qwen2", "qwen3"):
        sliding = tuple(range(int(config.get("max_window_layers", layers)), layers))
    else:
        sliding = tuple(range(layers))
    if not sliding:
        return None, None
    return int(window), None if len(sliding) == layers else sliding


def _hf_name(name: str, config: TransformerConfig) -> str | None:
    """Our name for a checkpoint tensor; None for buffers that carry no weights."""
    if name in _HF_GLOBAL_NAMES:
        return _HF_GLOBAL_NAMES[name]
    match = _HF_LAYER.match(name)
    if match:
        if match.group(2) == "self_attn.rotary_emb.inv_freq":
            return None
        names = {**_HF_LAYER_NAMES, **_HF_POST_NORM_NAMES} if config.norm_placement == "post" else _HF_LAYER_NAMES
        if match.group(2) in names:
            return f"layers.{int(match.group(1))}.{names[match.group(2)]}"
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


def _convert_huggingface(directory: Path) -> _Converted:
    config_path = directory / "config.json"
    if not config_path.exists():
        raise ModelImportError(f"{directory}: no config.json")
    raw_config = _read_json(config_path)
    config = hf_config(raw_config, _read_json(directory / "generation_config.json"))
    checkpoint = open_checkpoint(directory)

    tensors: dict[str, TensorSource] = {}
    for tensor in checkpoint.values():
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
    family = metadata.get("general.architecture")
    if family not in ("llama", "qwen2", "qwen3"):
        raise ModelImportError(f"GGUF architecture {family!r} is not supported (supported: llama, qwen2, qwen3)")

    def key(name: str, default: Any = None) -> Any:
        value = metadata.get(f"{family}.{name}", default)
        if value is None:
            raise ModelImportError(f"GGUF metadata {family}.{name} is missing")
        return value

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
        attention_bias=family == "qwen2" or "blk.0.attn_q.bias" in gguf,
        qk_norm=family == "qwen3",
        tie_word_embeddings="output.weight" not in gguf,
        bos_token_id=None if bos is None else int(bos),
        eos_token_ids=tuple(eos),
    )


def _convert_gguf(path: Path) -> _Converted:
    gguf = GgufFile(path)
    config = gguf_config(gguf)
    metadata = gguf.metadata
    tensors: dict[str, TensorSource] = {}
    for tensor in gguf:
        name = _gguf_name(tensor.name)
        load: Callable[[], np.ndarray] = tensor.to_float32
        if config.family == "llama" and re.search(r"attention\.[qk]\.(weight|bias)$", name):
            heads = config.heads if ".q." in name else config.kv_heads
            load = (lambda t, h: lambda: unpermute_rotary(t.to_float32(), h))(tensor, heads)
        tensors[name] = TensorSource(tensor.shape, load, tensor.type_name)

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
) -> ImportResult:
    """Imports ``source`` (a checkpoint directory, a ``.gguf`` file, or ``hf:org/name[@revision]``) into
    ``output``. ``repository``/``revision`` record provenance for local sources; ``licence`` overrides the licence
    the source states; ``licence_file`` supplies its text. With ``base`` (a ``model.dllm``), ``source`` is a PEFT
    LoRA adapter and ``output`` is the base model with the adapter merged into its weights."""
    source_text = str(source)
    if source_text.startswith("hf:"):
        repo, rev = hub.parse_reference(source_text)
        snapshot = hub.download(repo, rev, cache or Path.home() / ".cache" / "etalii-dllm" / "hub", opener)
        path, repository, revision = snapshot.directory, snapshot.repository, snapshot.revision
    else:
        path = Path(source)
    if (path / ADAPTER_CONFIG).exists():
        if base is None:
            raise ModelImportError(f"{source} is a LoRA adapter; pass --base <model.dllm> to merge it into")
        return _import_adapter(path, output, base, repository, revision, licence, accept_licence)
    if base is not None:
        raise ModelImportError(f"{source} is not a LoRA adapter (no {ADAPTER_CONFIG}); --base is for adapters")
    if path.is_dir():
        converted = _convert_huggingface(path)
    elif path.is_file() and path.suffix.lower() == ".gguf":
        converted = _convert_gguf(path)
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
    tensors = {
        name: TensorSource(tuple(values.shape), lambda values=values: values) for name, values in weights.items()
    }
    fingerprint = write_model_file(output, base.config, tensors, metadata)
    return ImportResult(Path(output), fingerprint, base.config, dict(base.source), record)
