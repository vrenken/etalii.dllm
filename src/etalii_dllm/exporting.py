"""Back to the ecosystem: a ``model.dllm`` as Hugging Face safetensors or as GGUF (``dllm export``).

Fine-tuned, edited and merged models only exist as ``model.dllm`` files; exporting them lets other runtimes load
exactly these weights. Both writers are deterministic (canonical JSON, a fixed tensor order, no clock values), so an
export is byte-identical on every platform, and importing it again gives the same weights fingerprint.

- ``safetensors``: a Hugging Face model directory (``config.json``, ``generation_config.json``, the tokenizer files,
  ``LICENSE``, a ``README.md`` with the attribution, and one ``model.safetensors`` in float32). The ``config.json`` is
  checked by importing it again: a model whose description does not survive that round trip is refused rather than
  exported approximately. Llama, Mistral, Qwen2, Qwen3, OLMo 2, Granite, Mixtral, OLMoE and Qwen3-MoE (one tensor
  per expert, as their checkpoints store them).
- ``gguf``: one GGUF v3 file in float32 as llama.cpp writes it (Llama Q/K rows permuted, the vocabulary padded to
  the embedding size, experts stacked into one tensor per projection). Llama, Qwen2, Qwen3, Mixtral, OLMoE and
  Qwen3-MoE with byte-level BPE tokenizers. GGUF stores ``rms_norm_eps`` as
  float32, so a re-imported GGUF can differ from the original in that one value, as any GGUF does, and keeps two
  end-of-sequence ids (``eos`` and ``eot``).
"""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from etalii_dllm import bpe
from etalii_dllm.architecture import TransformerConfig
from etalii_dllm.modelfile import ModelFile, tensor_order

FORMATS = ("safetensors", "gguf")

_HF_ARCHITECTURES = {
    "llama": "LlamaForCausalLM",
    "mistral": "MistralForCausalLM",
    "qwen2": "Qwen2ForCausalLM",
    "qwen3": "Qwen3ForCausalLM",
    "olmo2": "Olmo2ForCausalLM",
    "granite": "GraniteForCausalLM",
    "mixtral": "MixtralForCausalLM",
    "olmoe": "OlmoeForCausalLM",
    "qwen3_moe": "Qwen3MoeForCausalLM",
}
_HF_ACTIVATIONS = {"silu": "silu", "gelu_tanh": "gelu_pytorch_tanh"}

# Probe texts the GGUF tokenizer must encode exactly like the original: words, contractions, digits, whitespace,
# newlines, punctuation and non-Latin scripts.
_PROBES = (
    "Hello, world! It's 2024: we'll test 12345 and 3.14159.",
    "  Leading spaces,\ttabs\n\nand\r\nline breaks   ",
    "naïve café, Ünïcödé — 東京, Привет, 안녕하세요, مرحبا 🙂👍",
    "def f(x):\n    return x**2 + 1  # comment",
    "I'M SHOUTING'S AND you've they're DON'T",
    "1 22 333 4444 55555 666666 7777777",
)


class ExportError(ValueError):
    """A model that cannot be written in the requested format."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _rows(model: ModelFile, name: str) -> np.ndarray:
    return np.ascontiguousarray(model.tensors[name], dtype="<f4")


# -- Hugging Face -------------------------------------------------------------------------------------------------


def hf_config_json(config: TransformerConfig) -> dict[str, Any]:
    """The ``config.json`` that imports as ``config``."""
    from etalii_dllm.importing.importer import ModelImportError, hf_config

    if config.family not in _HF_ARCHITECTURES:
        supported = ", ".join(sorted(_HF_ARCHITECTURES))
        raise ExportError(f"exporting {config.family} models to safetensors is not supported (supported: {supported})")
    eos = list(config.eos_token_ids)
    document: dict[str, Any] = {
        "architectures": [_HF_ARCHITECTURES[config.family]],
        "model_type": config.family,
        "vocab_size": config.vocabulary_size,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.layers,
        "num_attention_heads": config.heads,
        "num_key_value_heads": config.kv_heads,
        "head_dim": config.head_dim,
        "max_position_embeddings": config.context_length,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_theta": config.rope_theta,
        "hidden_act": _HF_ACTIVATIONS[config.activation],
        "attention_bias": config.attention_bias,
        "tie_word_embeddings": config.tie_word_embeddings,
        "torch_dtype": "float32",
        "eos_token_id": eos[0] if len(eos) == 1 else eos,
    }
    if config.bos_token_id is not None:
        document["bos_token_id"] = config.bos_token_id
    if config.rope_scaling:
        document["rope_scaling"] = dict(config.rope_scaling)
    if config.rotary_dim:
        document["partial_rotary_factor"] = config.rotary_dim / config.head_dim
    if config.sliding_window is not None:
        sliding = config.sliding_window_layers
        document["sliding_window"] = config.sliding_window
        document["use_sliding_window"] = True
        document["layer_types"] = [
            "sliding_attention" if sliding is None or i in sliding else "full_attention" for i in range(config.layers)
        ]
    if config.family == "granite":
        document |= {
            "embedding_multiplier": config.embedding_multiplier,
            "residual_multiplier": config.residual_multiplier,
            "logits_scaling": config.logits_scaling,
        }
        if config.attention_multiplier is not None:
            document["attention_multiplier"] = config.attention_multiplier
    if config.experts:
        document["num_local_experts" if config.family == "mixtral" else "num_experts"] = config.experts
        document["num_experts_per_tok"] = config.experts_per_token
        if config.family != "mixtral":
            document["norm_topk_prob"] = config.normalize_expert_weights
        if config.family == "qwen3_moe":
            document |= {
                "moe_intermediate_size": config.expert_size,
                "decoder_sparse_step": 1,
                "mlp_only_layers": list(config.dense_layers or ()),
            }
    try:
        again = hf_config(document)
    except ModelImportError as error:
        raise ExportError(f"the model cannot be described as a Hugging Face config.json: {error}") from None
    if again != config:
        raise ExportError("the model cannot be described exactly as a Hugging Face config.json")
    return document


_HF_NAMES = {
    "attention_norm.weight": "input_layernorm.weight",
    "attention.q.weight": "self_attn.q_proj.weight",
    "attention.k.weight": "self_attn.k_proj.weight",
    "attention.v.weight": "self_attn.v_proj.weight",
    "attention.o.weight": "self_attn.o_proj.weight",
    "attention.q.bias": "self_attn.q_proj.bias",
    "attention.k.bias": "self_attn.k_proj.bias",
    "attention.v.bias": "self_attn.v_proj.bias",
    "attention.q_norm.weight": "self_attn.q_norm.weight",
    "attention.k_norm.weight": "self_attn.k_norm.weight",
    "mlp.gate.weight": "mlp.gate_proj.weight",
    "mlp.up.weight": "mlp.up_proj.weight",
    "mlp.down.weight": "mlp.down_proj.weight",
}
_HF_PRE_NORMS = {"mlp_norm.weight": "post_attention_layernorm.weight"}
_HF_POST_NORMS = {
    "attention_post_norm.weight": "post_attention_layernorm.weight",
    "mlp_post_norm.weight": "post_feedforward_layernorm.weight",
}
_HF_GLOBALS = {
    "token_embedding.weight": "model.embed_tokens.weight",
    "final_norm.weight": "model.norm.weight",
    "lm_head.weight": "lm_head.weight",
}


_MIXTRAL_EXPERTS = {"gate": "w1", "down": "w2", "up": "w3"}


def hf_tensor_name(name: str, config: TransformerConfig) -> str:
    if name in _HF_GLOBALS:
        return _HF_GLOBALS[name]
    _, index, rest = name.split(".", 2)
    moe = "block_sparse_moe" if config.family == "mixtral" else "mlp"
    if rest == "mlp.router.weight":
        return f"model.layers.{index}.{moe}.gate.weight"
    if rest.startswith("mlp.experts."):
        _, _, expert, projection, _ = rest.split(".")
        hf = _MIXTRAL_EXPERTS[projection] if config.family == "mixtral" else f"{projection}_proj"
        return f"model.layers.{index}.{moe}.experts.{expert}.{hf}.weight"
    names = {**_HF_NAMES, **(_HF_POST_NORMS if config.norm_placement == "post" else _HF_PRE_NORMS)}
    return f"model.layers.{index}.{names[rest]}"


def write_safetensors(path: Path, tensors: Mapping[str, np.ndarray]) -> None:
    """A safetensors file: float32 tensors in name order, the JSON header padded to 8 bytes."""
    header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name in sorted(tensors):
        nbytes = tensors[name].size * 4
        header[name] = {"dtype": "F32", "shape": list(tensors[name].shape), "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)) + encoded)
        for name in sorted(tensors):
            stream.write(np.ascontiguousarray(tensors[name], dtype="<f4").tobytes())


def _tokenizer_files(tokenizer: Mapping[str, Any] | None, chat_template: str | None) -> dict[str, bytes]:
    if tokenizer is None:
        return {}
    if tokenizer.get("format") == "huggingface":
        spec = tokenizer["tokenizer_json"]
        config = dict(tokenizer.get("tokenizer_config") or {})
        special = tokenizer.get("special_tokens_map") or {}
    else:  # a GGUF import: its tokenizer.ggml.* metadata
        spec = bpe.spec_from_gguf(tokenizer)
        tokens = list(tokenizer["tokenizer.ggml.tokens"])
        config = {}
        for key, field in (("eos_token", "eos_token_id"), ("bos_token", "bos_token_id")):
            if tokenizer.get(f"tokenizer.ggml.{field}") is not None:
                config[key] = tokens[int(tokenizer[f"tokenizer.ggml.{field}"])]
        special = dict(config)
    if chat_template is not None:
        config["chat_template"] = chat_template
    return {
        "tokenizer.json": _json_bytes(spec),
        "tokenizer_config.json": _json_bytes(config),
        "special_tokens_map.json": _json_bytes(special),
    }


def _readme(model: ModelFile) -> bytes:
    licence = model.licence
    front = f"---\nlicense: {str(licence.get('spdx', 'other')).lower()}\nlibrary_name: transformers\n---\n\n"
    body = (
        f"# {Path(model.path).stem}\n\n"
        f"Exported by EtAlii.Dllm from the model.dllm file with weights fingerprint `{model.fingerprint}`.\n\n"
        f"{licence.get('attribution', '')}\n"
    )
    return (front + body).encode("utf-8")


def export_safetensors(model: ModelFile, directory: str | Path) -> list[Path]:
    """Writes ``model`` as a Hugging Face model directory; returns the files written, in name order."""
    config = model.config
    files: dict[str, bytes] = {
        "config.json": _json_bytes(hf_config_json(config)),
        "generation_config.json": _json_bytes(
            {
                "eos_token_id": list(config.eos_token_ids),
                **({"bos_token_id": config.bos_token_id} if config.bos_token_id is not None else {}),
            }
        ),
        "README.md": _readme(model),
        **_tokenizer_files(model.tokenizer, model.chat_template),
    }
    if model.licence.get("text"):
        files["LICENSE"] = str(model.licence["text"]).encode("utf-8")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (directory / name).write_bytes(data)
    tensors = {hf_tensor_name(name, config): model.tensors[name] for name in model.tensors}
    write_safetensors(directory / "model.safetensors", tensors)
    return [directory / name for name in sorted([*files, "model.safetensors"])]  # by name: WindowsPath sorts casefolded


# -- GGUF -------------------------------------------------------------------------------------------------------------

# family -> GGUF architecture; llama.cpp stores Mixtral as "llama" with experts.
_GGUF_FAMILIES = {"llama": "llama", "qwen2": "qwen2", "qwen3": "qwen3", "mixtral": "llama", "olmoe": "olmoe",
                  "qwen3_moe": "qwen3moe"}  # fmt: skip
_GGUF_NAMES = {
    "attention_norm.weight": "attn_norm.weight",
    "mlp_norm.weight": "ffn_norm.weight",
    "attention.q.weight": "attn_q.weight",
    "attention.k.weight": "attn_k.weight",
    "attention.v.weight": "attn_v.weight",
    "attention.o.weight": "attn_output.weight",
    "attention.q.bias": "attn_q.bias",
    "attention.k.bias": "attn_k.bias",
    "attention.v.bias": "attn_v.bias",
    "attention.q_norm.weight": "attn_q_norm.weight",
    "attention.k_norm.weight": "attn_k_norm.weight",
    "mlp.gate.weight": "ffn_gate.weight",
    "mlp.up.weight": "ffn_up.weight",
    "mlp.down.weight": "ffn_down.weight",
    "mlp.router.weight": "ffn_gate_inp.weight",
}
_GGUF_GLOBALS = {"token_embedding.weight": "token_embd.weight", "final_norm.weight": "output_norm.weight",
                 "lm_head.weight": "output.weight"}  # fmt: skip

# Metadata value types.
_U32, _F32, _BOOL, _STRING, _ARRAY = 4, 6, 7, 8, 9
_I32 = 5
_ALIGNMENT = 32


def permute_rotary(weight: np.ndarray, heads: int) -> np.ndarray:
    """llama.cpp's Q/K row order for Llama (``permute`` in ``convert_hf_to_gguf.py``); the inverse of
    :func:`etalii_dllm.importing.importer.unpermute_rotary`."""
    rows = weight.shape[0]
    return weight.reshape(heads, 2, rows // heads // 2, *weight.shape[1:]).swapaxes(1, 2).reshape(weight.shape)


def gguf_tensor_name(name: str) -> str:
    """The GGUF name of a tensor; every expert's ``mlp.experts.<e>.<projection>.weight`` maps to the stacked
    ``ffn_<projection>_exps.weight``."""
    if name in _GGUF_GLOBALS:
        return _GGUF_GLOBALS[name]
    _, index, rest = name.split(".", 2)
    if rest.startswith("mlp.experts."):
        return f"blk.{index}.ffn_{rest.split('.')[3]}_exps.weight"
    return f"blk.{index}.{_GGUF_NAMES[rest]}"


def _string(text: str) -> bytes:
    data = text.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _value(kind: int, value: Any) -> bytes:
    if kind == _STRING:
        return _string(value)
    if kind == _U32:
        return struct.pack("<I", value)
    if kind == _I32:
        return struct.pack("<i", value)
    if kind == _F32:
        return struct.pack("<f", value)
    if kind == _BOOL:
        return struct.pack("<?", value)
    element, items = value
    return struct.pack("<IQ", element, len(items)) + b"".join(_value(element, item) for item in items)


def _gguf_tokenizer(tokenizer: Mapping[str, Any] | None, vocabulary_size: int) -> list[tuple[str, int, Any]]:
    """The ``tokenizer.ggml.*`` metadata for a byte-level BPE tokenizer, checked to encode like the original."""
    if tokenizer is None:
        raise ExportError("the model has no tokenizer to write")
    if tokenizer.get("format") == "gguf":
        keys = ("model", "pre", "tokens", "token_type", "merges")
        present = [key for key in keys if f"tokenizer.ggml.{key}" in tokenizer]
        return [(f"tokenizer.ggml.{key}", *_typed(key, tokenizer[f"tokenizer.ggml.{key}"])) for key in present]
    spec = tokenizer["tokenizer_json"]
    model = spec.get("model") or {}
    if model.get("type") != "BPE" or (spec.get("decoder") or {}).get("type") != "ByteLevel":
        raise ExportError("only byte-level BPE tokenizers can be written to GGUF")
    tokens = {int(i): token for token, i in model["vocab"].items()}
    types = dict.fromkeys(tokens, 1)
    for added in spec.get("added_tokens") or []:
        tokens[int(added["id"])] = added["content"]
        types[int(added["id"])] = 3 if added.get("special") else 4
    size = max(vocabulary_size, max(tokens) + 1)
    names = [tokens.get(i, f"[PAD{i}]") for i in range(size)]
    kinds = [types.get(i, 5) for i in range(size)]
    merges = [m if isinstance(m, str) else " ".join(m) for m in model.get("merges") or []]
    original = bpe.from_model_header(tokenizer)
    # The pre-tokenizer whose description matches the original first (llama.cpp hard-codes each one), then any that
    # encodes the probes the same way.
    shape = (bpe.without_offsets(spec.get("normalizer")), bpe.without_offsets(spec.get("pre_tokenizer")))
    exact = [pre for pre in bpe.GGUF_PRE_TOKENIZERS if bpe.gguf_pre_tokenizer(pre) == shape]
    for pre in [*exact, *(pre for pre in bpe.GGUF_PRE_TOKENIZERS if pre not in exact)]:
        metadata = {"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": pre, "tokenizer.ggml.tokens": names,
                    "tokenizer.ggml.token_type": kinds, "tokenizer.ggml.merges": merges}  # fmt: skip
        candidate = bpe.BpeTokenizer(bpe.spec_from_gguf(metadata))
        if all(candidate.encode(text) == original.encode(text) for text in _PROBES):
            return [(key, *_typed(key.removeprefix("tokenizer.ggml."), value)) for key, value in metadata.items()]
    raise ExportError("the tokenizer's pre-tokenizer has no GGUF equivalent")


def _typed(key: str, value: Any) -> tuple[int, Any]:
    if key in ("tokens", "merges"):
        return _ARRAY, (_STRING, list(value))
    if key == "token_type":
        return _ARRAY, (_I32, [int(v) for v in value])
    return _STRING, str(value)


def export_gguf(model: ModelFile, path: str | Path) -> Path:
    """Writes ``model`` as a float32 GGUF v3 file."""
    config = model.config
    family = config.family
    if family not in _GGUF_FAMILIES:
        supported = ", ".join(sorted(_GGUF_FAMILIES))
        raise ExportError(f"exporting {family} models to GGUF is not supported (supported: {supported})")
    architecture = _GGUF_FAMILIES[family]
    if config.sliding_window is not None or config.rotary_dim is not None:
        raise ExportError("sliding windows and partial rotary embeddings cannot be written to GGUF")
    scaling = dict(config.rope_scaling or {})
    if scaling and scaling["rope_type"] == "yarn":
        from etalii_dllm.importing.importer import yarn_scaling

        if scaling != yarn_scaling(scaling["factor"], scaling["original_max_position_embeddings"]) or float(
            np.float32(scaling["factor"])
        ) != float(scaling["factor"]):
            raise ExportError("this YaRN scaling (betas, attention factor or factor) cannot be written to GGUF exactly")
    elif scaling and scaling["rope_type"] != "linear":
        raise ExportError(f"RoPE scaling {scaling['rope_type']!r} cannot be written to GGUF")
    if config.qk_norm != (family in ("olmoe", "qwen3", "qwen3_moe")) or config.qk_norm_scope != (
        "all" if family == "olmoe" else "head"
    ):
        raise ExportError("the model's normalisation has no GGUF equivalent")
    licence = model.licence
    name = (model.source.get("repository") or Path(model.path).stem).split("/")[-1]
    metadata: list[tuple[str, int, Any]] = [
        ("general.architecture", _STRING, architecture),
        ("general.name", _STRING, name),
        ("general.alignment", _U32, _ALIGNMENT),
        ("general.file_type", _U32, 0),
    ]
    if licence.get("spdx"):
        metadata.append(("general.license", _STRING, str(licence["spdx"]).lower()))
    a = architecture
    metadata += [
        (f"{a}.context_length", _U32, config.context_length),
        (f"{a}.embedding_length", _U32, config.hidden_size),
        (f"{a}.block_count", _U32, config.layers),
        (f"{a}.feed_forward_length", _U32, config.intermediate_size),
        (f"{a}.attention.head_count", _U32, config.heads),
        (f"{a}.attention.head_count_kv", _U32, config.kv_heads),
        (f"{a}.attention.key_length", _U32, config.head_dim),
        (f"{a}.attention.value_length", _U32, config.head_dim),
        (f"{a}.attention.layer_norm_rms_epsilon", _F32, config.rms_norm_eps),
        (f"{a}.rope.freq_base", _F32, config.rope_theta),
        (f"{a}.rope.dimension_count", _U32, config.head_dim),
        (f"{a}.vocab_size", _U32, config.vocabulary_size),
    ]
    if config.experts:
        metadata += [
            (f"{a}.expert_count", _U32, config.experts),
            (f"{a}.expert_used_count", _U32, config.experts_per_token),
            (f"{a}.expert_weights_norm", _BOOL, config.normalize_expert_weights),
        ]
        if config.expert_intermediate_size is not None:
            metadata.append((f"{a}.expert_feed_forward_length", _U32, config.expert_intermediate_size))
    if scaling:
        metadata += [
            (f"{a}.rope.scaling.type", _STRING, scaling["rope_type"]),
            (f"{a}.rope.scaling.factor", _F32, float(scaling["factor"])),
        ]
    if scaling and scaling["rope_type"] == "yarn":
        original = scaling["original_max_position_embeddings"]
        metadata.append((f"{a}.rope.scaling.original_context_length", _U32, original))
    metadata += _gguf_tokenizer(model.tokenizer, config.vocabulary_size)
    # The special ids come from the model's own description, as the importer reads them back: GGUF has room for one
    # end-of-sequence and one end-of-turn token.
    if config.bos_token_id is not None:
        metadata.append(("tokenizer.ggml.bos_token_id", _U32, config.bos_token_id))
    for key, token in zip(("eos_token_id", "eot_token_id"), config.eos_token_ids, strict=False):
        metadata.append((f"tokenizer.ggml.{key}", _U32, token))
    if model.chat_template is not None:
        metadata.append(("tokenizer.chat_template", _STRING, model.chat_template))

    # GGUF tensors in model order; each layer's experts become one stacked tensor per projection, at the place of
    # expert 0.
    stacks: dict[str, list[str]] = {}
    for tensor in tensor_order(model.tensors):
        stacks.setdefault(gguf_tensor_name(tensor), []).append(tensor)
    for parts in stacks.values():
        parts.sort(key=lambda name: int(name.split(".")[4]) if ".mlp.experts." in name else 0)

    def data(target: str) -> np.ndarray:
        parts = stacks[target]
        if ".mlp.experts." in parts[0]:
            return np.ascontiguousarray(np.stack([_rows(model, part) for part in parts]))
        values = _rows(model, parts[0])
        if architecture == "llama" and parts[0].split(".", 2)[-1] in ("attention.q.weight", "attention.k.weight"):
            values = permute_rotary(values, config.heads if ".q." in parts[0] else config.kv_heads)
        return values

    def shape_of(target: str) -> tuple[int, ...]:
        parts = stacks[target]
        shape = tuple(model.tensors[parts[0]].shape)
        return (len(parts), *shape) if ".mlp.experts." in parts[0] else shape

    names = list(stacks)
    infos, offset = [], 0
    for tensor in names:
        shape = shape_of(tensor)
        infos.append(_string(tensor) + struct.pack("<I", len(shape)))
        infos[-1] += b"".join(struct.pack("<Q", d) for d in reversed(shape)) + struct.pack("<IQ", 0, offset)
        nbytes = int(np.prod(shape)) * 4
        offset += nbytes + (-nbytes % _ALIGNMENT)
    head = b"GGUF" + struct.pack("<IQQ", 3, len(names), len(metadata))
    head += b"".join(_string(key) + struct.pack("<I", kind) + _value(kind, value) for key, kind, value in metadata)
    head += b"".join(infos)
    head += b"\0" * (-len(head) % _ALIGNMENT)
    path = Path(path)
    with path.open("wb") as stream:
        stream.write(head)
        for tensor in names:
            block = data(tensor).tobytes()
            stream.write(block + b"\0" * (-len(block) % _ALIGNMENT))
    return path


def export_model(path: str | Path, output: str | Path, format: str) -> list[Path]:
    """Exports the model file ``path`` as ``format`` to ``output`` (a directory for safetensors, a file for GGUF)."""
    if format not in FORMATS:
        raise ExportError(f"unknown export format {format!r}; supported: {', '.join(FORMATS)}")
    model = ModelFile(path)
    if format == "safetensors":
        return export_safetensors(model, output)
    return [export_gguf(model, output)]
