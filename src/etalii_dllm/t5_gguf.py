"""T5 models through GGUF (issues #417-#419): T5 and Flan-T5 text-to-text models in llama.cpp's ``t5`` layout and T5
encoders in its ``t5encoder`` layout, with their Unigram vocabularies.

- Tensors: ``token_embd`` (the shared word embedding), ``output`` (the LM head, left out when it is tied to the
  embedding, T5 v1.0), ``enc.blk.N.attn_q|k|v|o``, ``attn_norm``, ``ffn_norm``, ``ffn_up``/``ffn_gate``/``ffn_down``
  (``wi``, or ``wi_1`` and ``wi_0``, and ``wo``), ``enc.output_norm``, and for the decoder ``dec.blk.N...`` with
  ``cross_attn_*`` for the attention over the encoder's states and ``dec.output_norm``. Each stack's relative bias
  table is ``attn_rel_b`` in its first block, ``[buckets, heads]``.
- Metadata: ``t5.context_length``, ``embedding_length``, ``feed_forward_length``, ``block_count`` and
  ``decoder_block_count``, ``attention.head_count``, ``key_length``/``value_length``, the RMS epsilon,
  ``attention.relative_buckets_count`` and ``decoder_start_token_id``, as llama.cpp's converter writes them.
  llama.cpp fixes the buckets' maximum distance at 128 and the MLP's activation by its shape (ReLU, or the tanh GELU
  when gated), so other models are refused rather than written as something llama.cpp would run differently.
- The vocabulary (``tokenizer.ggml.model`` ``t5``): the Unigram pieces with their scores and SentencePiece token
  types (1 normal, 2 unknown, 3 control, 4 user-defined, 5 unused, 6 byte), ``precompiled_charsmap`` (SentencePiece's
  normaliser, bytes), ``add_space_prefix`` and ``remove_extra_whitespaces``. Embedding rows past the vocabulary are
  ``[PADn]`` pieces of the unused type, as llama.cpp's converter pads them. Without ``tokenizer.huggingface.json``
  the tokenizer is rebuilt as transformers' T5 converter lays it out (:func:`unigram_spec`); with it (``dllm export``
  writes both) it is kept exactly.

``dllm.*`` keys keep the exact RMS epsilon and an embedder's pooling settings; llama.cpp ignores them.
"""

from __future__ import annotations

import base64
import json
import re
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from etalii_dllm.bpe import special_token_text
from etalii_dllm.modelfile import ModelFile, tensor_order

T5_MAX_DISTANCE = 128
"""The relative buckets' maximum distance llama.cpp assumes for every T5 model."""

_GLOBAL_NAMES = {
    "token_embedding.weight": "token_embd.weight",
    "lm_head.weight": "output.weight",
    "relative_bias.weight": "enc.blk.0.attn_rel_b.weight",
    "final_norm.weight": "enc.output_norm.weight",
    "decoder.relative_bias.weight": "dec.blk.0.attn_rel_b.weight",
    "decoder.final_norm.weight": "dec.output_norm.weight",
}
_LAYER_NAMES = {
    "attention.q": "attn_q",
    "attention.k": "attn_k",
    "attention.v": "attn_v",
    "attention.o": "attn_o",
    "attention_norm": "attn_norm",
    "cross.q": "cross_attn_q",
    "cross.k": "cross_attn_k",
    "cross.v": "cross_attn_v",
    "cross.o": "cross_attn_o",
    "cross_norm": "cross_attn_norm",
    "mlp.gate": "ffn_gate",
    "mlp.up": "ffn_up",
    "mlp.down": "ffn_down",
    "mlp_norm": "ffn_norm",
}
_LAYER = re.compile(r"^(decoder\.)?layers\.(\d+)\.(.+)\.weight$")
_GGUF_LAYER = re.compile(r"^(enc|dec)\.blk\.(\d+)\.(.+)\.weight$")
_NORMAL, _UNKNOWN, _CONTROL, _USER_DEFINED, _UNUSED, _BYTE = 1, 2, 3, 4, 5, 6
_COLLAPSE = {"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": " "}
_SPLIT = {"type": "WhitespaceSplit"}


def _export_error(message: str) -> Exception:
    from etalii_dllm.exporting import ExportError

    return ExportError(message)


def gguf_t5_tensor_name(name: str) -> str:
    """The GGUF name of one of a T5 model's tensors."""
    if name in _GLOBAL_NAMES:
        return _GLOBAL_NAMES[name]
    match = _LAYER.match(name)
    if match is None or match.group(3) not in _LAYER_NAMES:
        raise _export_error(f"tensor {name!r} has no place in llama.cpp's t5 layout")
    stack = "dec" if match.group(1) else "enc"
    return f"{stack}.blk.{match.group(2)}.{_LAYER_NAMES[match.group(3)]}.weight"


def t5_tensor_name(gguf_name: str) -> str | None:
    """Our name of a tensor in llama.cpp's t5 layout (the inverse of :func:`gguf_t5_tensor_name`), or ``None``."""
    for ours, theirs in _GLOBAL_NAMES.items():
        if theirs == gguf_name:
            return ours
    match = _GGUF_LAYER.match(gguf_name)
    if match is None:
        return None
    by_gguf = {theirs: ours for ours, theirs in _LAYER_NAMES.items()}
    part = by_gguf.get(match.group(3))
    if part is None or (match.group(1) == "enc" and part.startswith("cross")):
        return None
    return f"{'decoder.' if match.group(1) == 'dec' else ''}layers.{match.group(2)}.{part}.weight"


# -- the vocabulary -------------------------------------------------------------------------------------------------


def _components(value: Mapping[str, Any] | None, kind: str) -> list[Mapping[str, Any]]:
    """A tokenizer.json normaliser or pre-tokenizer as a list of its steps (a ``Sequence`` unpacked)."""
    if not value:
        return []
    if value.get("type") == "Sequence":
        return list(value.get(kind) or [])
    return [value]


def _prepends(metaspace: Mapping[str, Any]) -> bool:
    if metaspace.get("replacement", "▁") != "▁":
        raise _export_error("only Metaspace with ▁ can be written to a GGUF t5 vocabulary")
    scheme = metaspace.get("prepend_scheme")
    if scheme is None:
        return bool(metaspace.get("add_prefix_space", True))
    if scheme not in ("always", "never"):
        raise _export_error(f"Metaspace prepend_scheme {scheme!r} cannot be written to a GGUF t5 vocabulary")
    return scheme == "always"


def unigram_vocabulary(spec: Mapping[str, Any], size: int) -> list[tuple[str, int, Any]]:
    """The ``tokenizer.ggml.*`` metadata of a Unigram ``tokenizer.json`` laid out as transformers converts T5's
    ``spiece.model`` (Precompiled, maybe collapsing spaces; Metaspace, maybe after WhitespaceSplit), padded with
    unused ``[PADn]`` pieces to ``size`` rows. Refused when another layout would not come back from
    :func:`unigram_spec`."""
    from etalii_dllm.encoder_export import _ARRAY, _BOOL, _F32, _I32, _STRING, _U8

    model = spec.get("model") or {}
    if model.get("type") != "Unigram":
        raise _export_error("only Unigram vocabularies can be written to a GGUF t5 vocabulary")
    pieces = [(str(piece), float(score)) for piece, score in model["vocab"]]
    if len(pieces) > size:
        raise _export_error("the vocabulary has more pieces than the embedding has rows")
    normalizers = _components(spec.get("normalizer"), "normalizers")
    charsmap = b""
    if normalizers and normalizers[0].get("type") == "Precompiled":
        charsmap = base64.b64decode(normalizers[0].get("precompiled_charsmap") or "")
        normalizers = normalizers[1:]
    collapse = normalizers == [_COLLAPSE]
    if normalizers and not collapse:
        raise _export_error("this tokenizer's normaliser cannot be written to a GGUF t5 vocabulary")
    pre = _components(spec.get("pre_tokenizer"), "pretokenizers")
    if pre[:1] == [_SPLIT] and collapse:
        pre = pre[1:]
    if len(pre) != 1 or pre[0].get("type") != "Metaspace":
        raise _export_error("this tokenizer's pre-tokenizer cannot be written to a GGUF t5 vocabulary")
    prefix = _prepends(pre[0])
    types = [_NORMAL] * len(pieces)
    unknown = model.get("unk_id")
    if unknown is not None:
        types[int(unknown)] = _UNKNOWN
    for added in spec.get("added_tokens") or []:
        index = int(added["id"])
        if index >= len(pieces) or str(added["content"]) != pieces[index][0]:
            raise _export_error("an added token of this tokenizer is not one of its Unigram pieces")
        if types[index] != _UNKNOWN:
            types[index] = _CONTROL if added.get("special") else _USER_DEFINED
    if model.get("byte_fallback"):
        for index, (piece, _) in enumerate(pieces):
            if re.fullmatch(r"<0x[0-9A-F]{2}>", piece):
                types[index] = _BYTE
    tokens = [piece for piece, _ in pieces] + [f"[PAD{i}]" for i in range(len(pieces), size)]
    scores = [score for _, score in pieces] + [0.0] * (size - len(pieces))
    types += [_UNUSED] * (size - len(pieces))
    metadata: list[tuple[str, int, Any]] = [
        ("tokenizer.ggml.model", _STRING, "t5"),
        ("tokenizer.ggml.tokens", _ARRAY, (_STRING, tokens)),
        ("tokenizer.ggml.scores", _ARRAY, (_F32, scores)),
        ("tokenizer.ggml.token_type", _ARRAY, (_I32, types)),
        ("tokenizer.ggml.add_space_prefix", _BOOL, prefix),
        ("tokenizer.ggml.remove_extra_whitespaces", _BOOL, collapse),
    ]
    if charsmap:
        metadata.append(("tokenizer.ggml.precompiled_charsmap", _ARRAY, (_U8, list(charsmap))))
    return metadata


def unigram_spec(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """A ``tokenizer.json`` for a GGUF t5 vocabulary, laid out as transformers' T5 converter writes one: the pieces
    (without the unused rows that pad the embedding), Precompiled and the space-collapsing Replace, WhitespaceSplit
    and Metaspace, ``$A </s>``. Control, user-defined and unknown pieces are special added tokens."""
    tokens = [str(token) for token in metadata["tokenizer.ggml.tokens"]]
    scores = [float(score) for score in metadata.get("tokenizer.ggml.scores", [0.0] * len(tokens))]
    types = [int(kind) for kind in metadata.get("tokenizer.ggml.token_type", [_NORMAL] * len(tokens))]
    if len(scores) != len(tokens) or len(types) != len(tokens):
        raise ValueError("a t5 vocabulary needs one score and one type per token")
    end = len(tokens)
    while end and types[end - 1] == _UNUSED:
        end -= 1
    unknown = metadata.get("tokenizer.ggml.unknown_token_id")
    if unknown is None:
        unknown = next((i for i, kind in enumerate(types[:end]) if kind == _UNKNOWN), None)
    charsmap = bytes(int(b) for b in metadata.get("tokenizer.ggml.precompiled_charsmap") or [])
    collapse = bool(metadata.get("tokenizer.ggml.remove_extra_whitespaces", True))
    normalizers: list[dict[str, Any]] = []
    if charsmap:
        normalizers.append({"type": "Precompiled", "precompiled_charsmap": base64.b64encode(charsmap).decode()})
    if collapse:
        normalizers.append(dict(_COLLAPSE))
    normalizer = None if not normalizers else normalizers[0] if len(normalizers) == 1 else {
        "type": "Sequence", "normalizers": normalizers
    }  # fmt: skip
    scheme = "always" if metadata.get("tokenizer.ggml.add_space_prefix", True) else "never"
    metaspace = {"type": "Metaspace", "replacement": "▁", "prepend_scheme": scheme, "split": True}
    pre = {"type": "Sequence", "pretokenizers": [dict(_SPLIT), metaspace]} if collapse else metaspace
    post = None
    eos = metadata.get("tokenizer.ggml.eos_token_id")
    if metadata.get("tokenizer.ggml.add_eos_token", True) and eos is not None and int(eos) < end:
        token = tokens[int(eos)]
        post = {
            "type": "TemplateProcessing",
            "single": [{"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": token, "type_id": 0}}],
            "pair": [
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": token, "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 0}},
                {"SpecialToken": {"id": token, "type_id": 0}},
            ],
            "special_tokens": {token: {"id": token, "ids": [int(eos)], "tokens": [token]}},
        }
    added = [
        {"id": i, "content": tokens[i], "single_word": False, "lstrip": False, "rstrip": False, "normalized": False,
         "special": True}
        for i in range(end)
        if types[i] in (_UNKNOWN, _CONTROL, _USER_DEFINED)
    ]  # fmt: skip
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": normalizer,
        "pre_tokenizer": pre,
        "post_processor": post,
        "decoder": dict(metaspace),
        "model": {
            "type": "Unigram",
            "unk_id": None if unknown is None else int(unknown),
            "vocab": [[tokens[i], scores[i]] for i in range(end)],
            "byte_fallback": _BYTE in types[:end],
        },
    }


def unigram_tokenizer_config(metadata: Mapping[str, Any], context_length: int) -> dict[str, Any]:
    """The ``tokenizer_config.json`` fields a t5 vocabulary's special token ids give."""
    tokens = [str(token) for token in metadata["tokenizer.ggml.tokens"]]
    config: dict[str, Any] = {"model_max_length": context_length}
    for key, field in (("eos_token", "eos"), ("unk_token", "unknown"), ("pad_token", "padding")):
        index = metadata.get(f"tokenizer.ggml.{field}_token_id")
        if index is not None and 0 <= int(index) < len(tokens):
            config[key] = tokens[int(index)]
    return config


# -- export -----------------------------------------------------------------------------------------------------------


_PROBES = (
    "Hello, world! It's 2024: we'll test 12345 and 3.14159.",
    "  Leading spaces,\ttabs\n\nand\r\nline breaks   ",
    "naïve café, Ünïcödé, \uff26\uff55\uff4c\uff4c ① ﬁ — 東京 🙂",
    "translate English to German: The house is wonderful.",
)


def _check_vocabulary(spec: Mapping[str, Any], metadata: list[tuple[str, int, Any]]) -> None:
    """Refuses a vocabulary whose rebuilt tokenizer would encode the probes differently from the original."""
    from etalii_dllm.bpe import BpeTokenizer

    values = {key: value[1] if isinstance(value, tuple) else value for key, _, value in metadata}
    rebuilt = BpeTokenizer(unigram_spec(values))
    original = BpeTokenizer(dict(spec))
    if any(rebuilt.encode(text) != original.encode(text) for text in _PROBES):
        raise _export_error("this tokenizer does not come back from a GGUF t5 vocabulary unchanged")


def export_t5_gguf(model: ModelFile, path: str | Path) -> Path:
    """Writes the T5 text-to-text ``model`` (``t5``) or T5 encoder (``t5encoder``) as a float32 GGUF v3 file."""
    from etalii_dllm.encoder_export import (
        _ALIGNMENT,
        _BOOL,
        _F32,
        _F64,
        _STRING,
        _U32,
        GGUF_POOLING,
        _rows,
        _string,
        _value,
    )

    config = model.config
    text_to_text = config.is_text_to_text
    if config.projection_size:
        raise _export_error("llama.cpp has no Dense projection after the pooling; export the model to safetensors")
    if config.max_relative_positions != T5_MAX_DISTANCE:
        raise _export_error(
            f"llama.cpp's t5 layout fixes the relative buckets' maximum distance at {T5_MAX_DISTANCE}, not "
            f"{config.max_relative_positions}; export the model to safetensors"
        )
    if config.activation != ("gelu_tanh" if config.gated_mlp else "relu"):
        kind = "gated" if config.gated_mlp else "plain"
        raise _export_error(
            f"llama.cpp runs a {kind} T5 MLP with {'the tanh GELU' if config.gated_mlp else 'ReLU'}, not "
            f"{config.activation}; export the model to safetensors"
        )
    tokenizer = model.tokenizer or {}
    if tokenizer.get("format") != "huggingface":
        raise _export_error("the T5 model's tokenizer is not a tokenizer.json")
    spec = tokenizer["tokenizer_json"]
    vocabulary = unigram_vocabulary(spec, config.vocabulary_size)
    a = "t5" if text_to_text else "t5encoder"
    name = (model.source.get("repository") or Path(model.path).stem).split("/")[-1]
    metadata: list[tuple[str, int, Any]] = [
        ("general.architecture", _STRING, a),
        ("general.name", _STRING, name),
        ("general.alignment", _U32, _ALIGNMENT),
        ("general.file_type", _U32, 0),
    ]
    if model.licence.get("spdx"):
        metadata.append(("general.license", _STRING, str(model.licence["spdx"]).lower()))
    metadata += [
        (f"{a}.context_length", _U32, config.context_length),
        (f"{a}.embedding_length", _U32, config.hidden_size),
        (f"{a}.feed_forward_length", _U32, config.intermediate_size),
        (f"{a}.block_count", _U32, config.layers),
        (f"{a}.attention.head_count", _U32, config.heads),
        (f"{a}.attention.key_length", _U32, config.head_dim),
        (f"{a}.attention.value_length", _U32, config.head_dim),
        (f"{a}.attention.layer_norm_epsilon", _F32, config.rms_norm_eps),
        (f"{a}.attention.layer_norm_rms_epsilon", _F32, config.rms_norm_eps),
        (f"{a}.attention.relative_buckets_count", _U32, config.position_buckets),
    ]
    if text_to_text:
        metadata += [
            (f"{a}.decoder_block_count", _U32, config.decoder_layers),
            (f"{a}.decoder_start_token_id", _U32, 0),
        ]
    elif model.embedding is not None:
        pooling = GGUF_POOLING.get(str(model.embedding.get("pooling", "mean")))
        if pooling is None:
            raise _export_error(f"llama.cpp has no {model.embedding.get('pooling')!r} pooling")
        metadata.append((f"{a}.pooling_type", _U32, pooling))
    metadata += vocabulary
    tokenizer_config = tokenizer.get("tokenizer_config") or {}
    pieces = {piece: i for i, (piece, _) in reversed(list(enumerate(spec["model"]["vocab"])))}
    for key, field in (("eos_token", "eos"), ("unk_token", "unknown"), ("pad_token", "padding")):
        text = special_token_text(tokenizer_config.get(key))
        if text in pieces:
            metadata.append((f"tokenizer.ggml.{field}_token_id", _U32, pieces[text]))
    metadata += [
        ("tokenizer.ggml.add_bos_token", _BOOL, False),
        ("tokenizer.ggml.add_eos_token", _BOOL, (spec.get("post_processor") or None) is not None),
    ]
    _check_vocabulary(spec, [item for item in metadata if item[0].startswith("tokenizer.ggml.")])
    metadata += [
        ("tokenizer.huggingface.json", _STRING, json.dumps(spec, sort_keys=True, ensure_ascii=False)),
        ("dllm.attention.layer_norm_rms_epsilon", _F64, config.rms_norm_eps),
    ]
    if not text_to_text and model.embedding is not None:
        metadata.append(("dllm.embedding", _STRING, json.dumps(model.embedding, sort_keys=True)))

    order = tensor_order(model.tensors)
    names = [gguf_t5_tensor_name(tensor) for tensor in order]
    sources = dict(zip(names, order, strict=True))
    infos, offset = [], 0
    for tensor in names:
        shape = tuple(model.tensors[sources[tensor]].shape)
        infos.append(_string(tensor) + struct.pack("<I", len(shape)))
        infos[-1] += b"".join(struct.pack("<Q", d) for d in reversed(shape)) + struct.pack("<IQ", 0, offset)
        nbytes = 4
        for d in shape:
            nbytes *= d
        offset += nbytes + (-nbytes % _ALIGNMENT)
    head = b"GGUF" + struct.pack("<IQQ", 3, len(names), len(metadata))
    head += b"".join(_string(key) + struct.pack("<I", kind) + _value(kind, value) for key, kind, value in metadata)
    head += b"".join(infos)
    head += b"\0" * (-len(head) % _ALIGNMENT)
    path = Path(path)
    with path.open("wb") as stream:
        stream.write(head)
        for tensor in names:
            block = _rows(model, sources[tensor]).tobytes()
            stream.write(block + b"\0" * (-len(block) % _ALIGNMENT))
    return path
