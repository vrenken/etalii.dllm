"""Encoders back to the ecosystem (issues #355, #360, #370): ``dllm export`` of BERT, RoBERTa, XLM-RoBERTa,
ModernBERT and DeBERTa models.

- ``safetensors``: a Hugging Face model directory that transformers and sentence-transformers load: ``config.json``
  (``BertModel``, ``RobertaModel`` or ``XLMRobertaModel``, or their ``...ForSequenceClassification`` with the
  labels), the tokenizer files, one float32 ``model.safetensors`` under transformers' names, and for an embedder the
  sentence-transformers modules (``modules.json``, the pooling module, ``Normalize`` when the model normalises, the
  prompts and ``max_seq_length``). The ``config.json`` is checked by importing it again, so a model that would not
  come back identical is refused. ModernBERT is written as ``ModernBertModel`` or
  ``ModernBertForSequenceClassification``: q, k and v stacked back into ``attn.Wqkv``, gate and up into ``mlp.Wi``,
  the head as ``head.dense`` and ``head.norm``, and the special token ids read from the tokenizer. DeBERTa is
  written as ``DebertaV2Model`` or ``DebertaV2ForSequenceClassification`` (the DeBERTa-v3 layout: relative attention
  with shared position projections, ``p2c|c2p``, the relative table's LayerNorm, no absolute positions), the table
  as ``encoder.rel_embeddings`` and ``encoder.LayerNorm``, the context pooler as ``pooler.dense``.
- ``gguf``: one float32 GGUF v3 file in llama.cpp's ``bert`` layout (``token_embd``, ``token_types``,
  ``position_embd``, ``token_embd_norm``, ``blk.N.attn_q`` ... ``layer_output_norm``; a cross-encoder's pooler and
  classifier as ``cls`` and ``cls.output``, so the head computes ``cls.output(tanh(cls(h[0])))``), the pooling type,
  and the WordPiece vocabulary in llama.cpp's phantom-space form (``##`` continuations bare, word starts after
  ``▁``). ``tokenizer.huggingface.json`` keeps the exact tokenizer, and ``dllm.*`` keys the exact LayerNorm epsilon
  and the pooling or classifier settings, so importing the file gives back the same model; llama.cpp ignores them.
  BERT models with WordPiece vocabularies and the exact GELU only: RoBERTa's positions start past the padding token,
  which llama.cpp's layout cuts from the position table. ModernBERT and DeBERTa are not written to GGUF.

Both writers are deterministic (canonical JSON, a fixed tensor order, no clock values).
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from typing import Any

import numpy as np

from etalii_dllm.architecture import TransformerConfig
from etalii_dllm.bpe import special_token_text
from etalii_dllm.modelfile import ModelFile, tensor_order

_LAYER_NAMES = {
    "attention.q": "attention.self.query",
    "attention.k": "attention.self.key",
    "attention.v": "attention.self.value",
    "attention.o": "attention.output.dense",
    "attention_norm": "attention.output.LayerNorm",
    "mlp.up": "intermediate.dense",
    "mlp.down": "output.dense",
    "mlp_norm": "output.LayerNorm",
}
_GLOBAL_NAMES = {
    "token_embedding": "embeddings.word_embeddings",
    "position_embedding": "embeddings.position_embeddings",
    "token_type_embedding": "embeddings.token_type_embeddings",
    "embedding_norm": "embeddings.LayerNorm",
}
_ACTIVATIONS = {"gelu": "gelu", "gelu_tanh": "gelu_new"}
_CROSS_ENCODER_ACTIVATIONS = {
    "sigmoid": "torch.nn.modules.activation.Sigmoid",
    "none": "torch.nn.modules.linear.Identity",
}
_POOLING_FLAGS = {
    "mean": "pooling_mode_mean_tokens",
    "cls": "pooling_mode_cls_token",
    "last_token": "pooling_mode_lasttoken",
}
_LAYER = re.compile(r"^layers\.(\d+)\.(.+)\.(weight|bias)$")


def _export_error(message: str) -> Exception:
    from etalii_dllm.exporting import ExportError

    return ExportError(message)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _rows(model: ModelFile, name: str) -> np.ndarray:
    return np.ascontiguousarray(model.tensors[name], dtype="<f4")


def _tokenizer_model(model: ModelFile) -> str:
    tokenizer = model.tokenizer or {}
    if tokenizer.get("format") != "huggingface":
        raise _export_error("the encoder's tokenizer is not a tokenizer.json")
    return str((tokenizer["tokenizer_json"].get("model") or {}).get("type"))


def _model_type(model: ModelFile) -> str:
    if model.config.family in ("modernbert", "deberta"):
        return model.config.family
    if model.config.padding_index is None:
        return "bert"
    return "xlm-roberta" if _tokenizer_model(model) == "Unigram" else "roberta"


# -- Hugging Face -------------------------------------------------------------------------------------------------


def hf_encoder_config(model: ModelFile) -> dict[str, Any]:
    """The ``config.json`` that imports as the model's description."""
    from etalii_dllm.importing.importer import ModelImportError, bert_config

    config = model.config
    model_type = _model_type(model)
    if model_type == "modernbert":
        return _modernbert_config(model)
    if model_type == "deberta":
        return _deberta_config(model)
    stem = {"bert": "Bert", "roberta": "Roberta", "xlm-roberta": "XLMRoberta"}[model_type]
    positions = config.context_length + (0 if config.padding_index is None else config.padding_index + 1)
    document: dict[str, Any] = {
        "architectures": [f"{stem}{'ForSequenceClassification' if config.classifier_labels else 'Model'}"],
        "model_type": model_type,
        "vocab_size": config.vocabulary_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.layers,
        "num_attention_heads": config.heads,
        "intermediate_size": config.intermediate_size,
        "max_position_embeddings": positions,
        "type_vocab_size": config.type_vocabulary_size,
        "layer_norm_eps": config.rms_norm_eps,
        "hidden_act": _ACTIVATIONS[config.activation],
        "position_embedding_type": "absolute",
        "torch_dtype": "float32",
    }
    if config.padding_index is not None:
        document["pad_token_id"] = config.padding_index
    if config.classifier_labels:
        _labels(model, document)
    try:
        again = bert_config(document)
    except ModelImportError as error:
        raise _export_error(f"the model cannot be described as a Hugging Face config.json: {error}") from None
    if again.classifier_labels != config.classifier_labels:
        import dataclasses

        again = dataclasses.replace(again, classifier_labels=config.classifier_labels)
    if again != config:
        raise _export_error("the model cannot be described exactly as a Hugging Face config.json")
    return document


def _special_id(model: ModelFile, role: str, default: int) -> int:
    """The id of the tokenizer's ``role`` token (``pad``, ``cls``, ``sep``), from its ``tokenizer_config.json``."""
    tokenizer = model.tokenizer or {}
    text = (tokenizer.get("tokenizer_config") or {}).get(f"{role}_token")
    if isinstance(text, dict):
        text = text.get("content")
    for added in (tokenizer.get("tokenizer_json") or {}).get("added_tokens") or []:
        if added.get("content") == text:
            return int(added["id"])
    return default


def _modernbert_config(model: ModelFile) -> dict[str, Any]:
    from etalii_dllm.importing.importer import ModelImportError, modernbert_config

    config = model.config
    local = set(range(config.layers)) if config.sliding_window_layers is None else set(config.sliding_window_layers)
    if config.sliding_window is None:
        local = set()
    types = ["sliding_attention" if i in local else "full_attention" for i in range(config.layers)]
    every = next(
        (n for n in range(1, config.layers + 1) if all((i % n == 0) == (i not in local) for i in range(config.layers))),
        None,
    )
    activation = _ACTIVATIONS[config.activation].replace("gelu_new", "gelu_pytorch_tanh")
    document: dict[str, Any] = {
        "architectures": [f"ModernBert{'ForSequenceClassification' if config.classifier_labels else 'Model'}"],
        "model_type": "modernbert",
        "vocab_size": config.vocabulary_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.layers,
        "num_attention_heads": config.heads,
        "intermediate_size": config.intermediate_size,
        "max_position_embeddings": config.context_length,
        "norm_eps": config.rms_norm_eps,
        "norm_bias": False,
        "attention_bias": False,
        "mlp_bias": False,
        "classifier_bias": False,
        "hidden_activation": activation,
        "classifier_activation": activation,
        "classifier_pooling": config.classifier_pooling or "cls",
        "global_rope_theta": config.rope_theta,
        "local_rope_theta": config.local_rope_theta if config.local_rope_theta is not None else config.rope_theta,
        "local_attention": 2 * (config.sliding_window - 1) if config.sliding_window else 128,
        "pad_token_id": _special_id(model, "pad", 0),
        "cls_token_id": _special_id(model, "cls", 0),
        "sep_token_id": _special_id(model, "sep", 0),
        "bos_token_id": _special_id(model, "cls", 0),
        "eos_token_id": _special_id(model, "sep", 0),
        "torch_dtype": "float32",
    }
    if every is not None:
        document["global_attn_every_n_layers"] = every
    else:
        document["layer_types"] = types
    if config.classifier_labels:
        _labels(model, document)
    try:
        again = modernbert_config(document)
    except ModelImportError as error:
        raise _export_error(f"the model cannot be described as a Hugging Face config.json: {error}") from None
    import dataclasses

    again = dataclasses.replace(
        again,
        classifier_labels=config.classifier_labels,
        classifier_pooling=again.classifier_pooling if config.classifier_labels else None,
    )
    if again != config:
        raise _export_error("the model cannot be described exactly as a Hugging Face config.json")
    return document


def _labels(model: ModelFile, document: dict[str, Any]) -> None:
    config = model.config
    labels = list((model.classifier or {}).get("labels") or [f"LABEL_{i}" for i in range(config.classifier_labels)])
    document["id2label"] = {str(i): label for i, label in enumerate(labels)}
    document["label2id"] = {label: i for i, label in enumerate(labels)}
    activation_fn = str((model.classifier or {}).get("activation", "none"))
    document["sentence_transformers"] = {"activation_fn": _CROSS_ENCODER_ACTIVATIONS[activation_fn]}


def _deberta_config(model: ModelFile) -> dict[str, Any]:
    from etalii_dllm.importing.importer import ModelImportError, deberta_config

    config = model.config
    activation = _ACTIVATIONS[config.activation]
    document: dict[str, Any] = {
        "architectures": [f"DebertaV2{'ForSequenceClassification' if config.classifier_labels else 'Model'}"],
        "model_type": "deberta-v2",
        "vocab_size": config.vocabulary_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.layers,
        "num_attention_heads": config.heads,
        "intermediate_size": config.intermediate_size,
        "max_position_embeddings": config.context_length,
        "type_vocab_size": config.type_vocabulary_size,
        "layer_norm_eps": config.rms_norm_eps,
        "hidden_act": activation,
        "pooler_hidden_act": activation,
        "pooler_hidden_size": config.hidden_size,
        "pooler_dropout": 0,
        "relative_attention": True,
        "position_biased_input": False,
        "share_att_key": True,
        "pos_att_type": ["p2c", "c2p"],
        "norm_rel_ebd": "layer_norm",
        "position_buckets": config.position_buckets or -1,
        "max_relative_positions": config.max_relative_positions,
        "pad_token_id": _special_id(model, "pad", 0),
        "torch_dtype": "float32",
    }
    if config.classifier_labels:
        _labels(model, document)
    try:
        again = deberta_config(document)
    except ModelImportError as error:
        raise _export_error(f"the model cannot be described as a Hugging Face config.json: {error}") from None
    import dataclasses

    if dataclasses.replace(again, classifier_labels=config.classifier_labels) != config:
        raise _export_error("the model cannot be described exactly as a Hugging Face config.json")
    return document


_DEBERTA_EXPORT_NAMES = {
    "token_embedding": "embeddings.word_embeddings",
    "token_type_embedding": "embeddings.token_type_embeddings",
    "embedding_norm": "embeddings.LayerNorm",
    "relative_embedding": "encoder.rel_embeddings",
    "relative_norm": "encoder.LayerNorm",
}
_DEBERTA_EXPORT_LAYER_NAMES = {
    **_LAYER_NAMES,
    "attention.q": "attention.self.query_proj",
    "attention.k": "attention.self.key_proj",
    "attention.v": "attention.self.value_proj",
}


def deberta_tensor_name(name: str, config: TransformerConfig) -> str:
    """transformers' name of a DeBERTa tensor: bare for an embedder's ``DebertaV2Model``, under ``deberta.`` in a
    ``DebertaV2ForSequenceClassification``, whose context pooler is ``pooler.dense`` and head ``classifier``."""
    prefix = "deberta." if config.classifier_labels else ""
    stem, _, parameter = name.rpartition(".")
    if stem in ("pooler", "classifier"):
        return f"pooler.dense.{parameter}" if stem == "pooler" else f"classifier.{parameter}"
    if stem in _DEBERTA_EXPORT_NAMES:
        return f"{prefix}{_DEBERTA_EXPORT_NAMES[stem]}.{parameter}"
    match = _LAYER.match(name)
    if match is None or match.group(2) not in _DEBERTA_EXPORT_LAYER_NAMES:
        raise _export_error(f"unexpected encoder tensor {name!r}")
    return f"{prefix}encoder.layer.{match.group(1)}.{_DEBERTA_EXPORT_LAYER_NAMES[match.group(2)]}.{parameter}"


_MODERNBERT_EXPORT_NAMES = {
    "attention_norm": "attn_norm",
    "attention.o": "attn.Wo",
    "mlp_norm": "mlp_norm",
    "mlp.down": "mlp.Wo",
}


def modernbert_tensors(model: ModelFile) -> dict[str, np.ndarray]:
    """transformers' ModernBERT tensors: the fused ``Wqkv`` and ``Wi`` stacked from their parts, under ``model.``
    with the head and classifier for a sequence-classification model."""
    config = model.config
    prefix = "model." if config.classifier_labels else ""
    names = {
        "token_embedding.weight": "embeddings.tok_embeddings.weight",
        "embedding_norm.weight": "embeddings.norm.weight",
        "final_norm.weight": "final_norm.weight",
    }
    tensors: dict[str, np.ndarray] = {}
    for name in model.tensors:
        if name in ("pooler.weight", "pooler_norm.weight", "classifier.weight", "classifier.bias"):
            head = {"pooler.weight": "head.dense.weight", "pooler_norm.weight": "head.norm.weight"}
            tensors[head.get(name, name)] = _rows(model, name)
            continue
        if name in names:
            tensors[prefix + names[name]] = _rows(model, name)
            continue
        match = _LAYER.match(name)
        if match is None:
            raise _export_error(f"unexpected encoder tensor {name!r}")
        layer, stem = match.group(1), match.group(2)
        if stem in _MODERNBERT_EXPORT_NAMES:
            tensors[f"{prefix}layers.{layer}.{_MODERNBERT_EXPORT_NAMES[stem]}.weight"] = _rows(model, name)
        elif stem == "attention.q":
            parts = [_rows(model, f"layers.{layer}.attention.{p}.weight") for p in ("q", "k", "v")]
            tensors[f"{prefix}layers.{layer}.attn.Wqkv.weight"] = np.concatenate(parts)
        elif stem == "mlp.gate":
            parts = [_rows(model, f"layers.{layer}.mlp.{p}.weight") for p in ("gate", "up")]
            tensors[f"{prefix}layers.{layer}.mlp.Wi.weight"] = np.concatenate(parts)
        elif stem not in ("attention.k", "attention.v", "mlp.up"):
            raise _export_error(f"unexpected encoder tensor {name!r}")
    return tensors


def hf_encoder_tensor_name(name: str, config: TransformerConfig, model_type: str) -> str:
    """transformers' name of an encoder tensor: bare for an embedder's model, under ``bert.``/``roberta.`` in a
    sequence-classification model, whose head is ``pooler.dense`` and ``classifier`` (BERT) or
    ``classifier.dense`` and ``classifier.out_proj`` (RoBERTa)."""
    prefix = ("bert." if model_type == "bert" else "roberta.") if config.classifier_labels else ""
    stem, _, parameter = name.rpartition(".")
    if stem in ("pooler", "classifier"):
        if model_type == "bert":
            return f"bert.pooler.dense.{parameter}" if stem == "pooler" else f"classifier.{parameter}"
        return f"classifier.{'dense' if stem == 'pooler' else 'out_proj'}.{parameter}"
    if stem in _GLOBAL_NAMES:
        return f"{prefix}{_GLOBAL_NAMES[stem]}.{parameter}"
    match = _LAYER.match(name)
    if match is None or match.group(2) not in _LAYER_NAMES:
        raise _export_error(f"unexpected encoder tensor {name!r}")
    return f"{prefix}encoder.layer.{match.group(1)}.{_LAYER_NAMES[match.group(2)]}.{parameter}"


def _readme(model: ModelFile, library: str) -> bytes:
    licence = model.licence
    front = f"---\nlicense: {str(licence.get('spdx', 'other')).lower()}\nlibrary_name: {library}\n---\n\n"
    body = (
        f"# {Path(model.path).stem}\n\n"
        f"Exported by EtAlii.Dllm from the model.dllm file with weights fingerprint `{model.fingerprint}`.\n\n"
        f"{licence.get('attribution', '')}\n"
    )
    return (front + body).encode("utf-8")


def export_encoder_safetensors(model: ModelFile, directory: str | Path) -> list[Path]:
    """Writes the encoder ``model`` as a Hugging Face (and, for an embedder, sentence-transformers) directory;
    returns the files written, in name order."""
    from etalii_dllm.exporting import _tokenizer_files, write_safetensors

    config = model.config
    model_type = _model_type(model)
    files: dict[str, bytes] = {"config.json": _json_bytes(hf_encoder_config(model))}
    tokenizer_files = _tokenizer_files(model.tokenizer, None)
    if config.classifier_labels:
        limit = (model.classifier or {}).get("max_tokens")
        if limit:
            tokenizer_config = json.loads(tokenizer_files["tokenizer_config.json"])
            tokenizer_config["model_max_length"] = int(limit)
            tokenizer_files["tokenizer_config.json"] = _json_bytes(tokenizer_config)
        files["README.md"] = _readme(model, "sentence-transformers")
    else:
        settings = model.embedding or {"pooling": "mean", "normalize": True}
        modules = [
            {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
            {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
        ]
        if settings.get("normalize", True):
            normalize = {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"}
            modules.append(normalize)
        pooling = {"word_embedding_dimension": config.hidden_size, "include_prompt": True}
        pooling |= {flag: settings.get("pooling", "mean") == mode for mode, flag in _POOLING_FLAGS.items()}
        files["modules.json"] = _json_bytes(modules)
        files["1_Pooling/config.json"] = _json_bytes(pooling)
        files["sentence_bert_config.json"] = _json_bytes(
            {"max_seq_length": int(settings.get("max_tokens") or config.context_length), "do_lower_case": False}
        )
        files["config_sentence_transformers.json"] = _json_bytes(
            {
                "prompts": dict(settings.get("prompts") or {}),
                "default_prompt_name": settings.get("default_prompt_name"),
                "similarity_fn_name": "cosine",
            }
        )
        files["README.md"] = _readme(model, "sentence-transformers")
    files |= tokenizer_files
    if model.licence.get("text"):
        files["LICENSE"] = str(model.licence["text"]).encode("utf-8")
    directory = Path(directory)
    for name, data in files.items():
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(data)
    if "modules.json" in files and any(m["path"] == "2_Normalize" for m in json.loads(files["modules.json"])):
        (directory / "2_Normalize").mkdir(exist_ok=True)
    if model_type == "modernbert":
        tensors = modernbert_tensors(model)
    elif model_type == "deberta":
        tensors = {deberta_tensor_name(name, config): model.tensors[name] for name in model.tensors}
    else:
        tensors = {hf_encoder_tensor_name(name, config, model_type): model.tensors[name] for name in model.tensors}
    write_safetensors(directory / "model.safetensors", tensors)
    return [directory / name for name in sorted([*files, "model.safetensors"])]


# -- GGUF -------------------------------------------------------------------------------------------------------------

GGUF_ENCODER_NAMES = {
    "token_embedding.weight": "token_embd.weight",
    "token_type_embedding.weight": "token_types.weight",
    "position_embedding.weight": "position_embd.weight",
    "embedding_norm.weight": "token_embd_norm.weight",
    "embedding_norm.bias": "token_embd_norm.bias",
    "pooler.weight": "cls.weight",
    "pooler.bias": "cls.bias",
    "classifier.weight": "cls.output.weight",
    "classifier.bias": "cls.output.bias",
}
GGUF_ENCODER_LAYER_NAMES = {
    "attention.q": "attn_q",
    "attention.k": "attn_k",
    "attention.v": "attn_v",
    "attention.o": "attn_output",
    "attention_norm": "attn_output_norm",
    "mlp.up": "ffn_up",
    "mlp.down": "ffn_down",
    "mlp_norm": "layer_output_norm",
}
GGUF_POOLING = {"mean": 1, "cls": 2, "last_token": 3}
"""llama.cpp's ``pooling_type`` values; a cross-encoder is ``4`` (rank)."""

_U32, _F32, _BOOL, _STRING, _ARRAY, _I32, _F64 = 4, 6, 7, 8, 9, 5, 12
_ALIGNMENT = 32


def gguf_encoder_tensor_name(name: str) -> str:
    if name in GGUF_ENCODER_NAMES:
        return GGUF_ENCODER_NAMES[name]
    match = _LAYER.match(name)
    if match is None or match.group(2) not in GGUF_ENCODER_LAYER_NAMES:
        raise _export_error(f"unexpected encoder tensor {name!r}")
    return f"blk.{match.group(1)}.{GGUF_ENCODER_LAYER_NAMES[match.group(2)]}.{match.group(3)}"


def phantom_vocabulary(spec: dict[str, Any]) -> tuple[list[str], list[int]]:
    """A WordPiece vocabulary as llama.cpp's ``bert`` tokenizer stores it: special (control) tokens as they are,
    ``##`` continuations without the prefix, every other piece after ``▁``; and the token types (1 normal, 3
    control, 4 user-defined). Refused when :func:`wordpiece_vocabulary` would not give the pieces back."""
    model = spec["model"]
    prefix = str(model.get("continuing_subword_prefix", "##"))
    if prefix != "##":
        raise _export_error("only WordPiece vocabularies with ## continuations can be written to GGUF")
    pieces = {int(i): str(token) for token, i in model["vocab"].items()}
    types = dict.fromkeys(pieces, 1)
    for added in spec.get("added_tokens") or []:
        pieces[int(added["id"])] = str(added["content"])
        types[int(added["id"])] = 3 if added.get("special") else 4
    if sorted(pieces) != list(range(len(pieces))):
        raise _export_error("the WordPiece vocabulary has gaps in its ids")
    tokens, kinds = [], []
    for i in range(len(pieces)):
        piece, kind = pieces[i], types[i]
        tokens.append(piece if kind == 3 else piece[2:] if piece.startswith("##") else "▁" + piece)
        kinds.append(kind)
    if wordpiece_vocabulary(tokens, kinds) != [pieces[i] for i in range(len(pieces))]:
        raise _export_error("the WordPiece vocabulary has pieces llama.cpp's phantom-space form cannot keep apart")
    return tokens, kinds


def wordpiece_vocabulary(tokens: list[str], types: list[int]) -> list[str]:
    """The WordPiece pieces of a llama.cpp ``bert`` vocabulary (the inverse of :func:`phantom_vocabulary`)."""
    return [
        token if kind == 3 else token[1:] if token.startswith("▁") else "##" + token
        for token, kind in zip(tokens, types, strict=True)
    ]


def _string(text: str) -> bytes:
    data = text.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _value(kind: int, value: Any) -> bytes:
    if kind == _STRING:
        return _string(value)
    if kind in (_U32, _I32, _F32, _BOOL, _F64):
        return struct.pack({_U32: "<I", _I32: "<i", _F32: "<f", _BOOL: "<?", _F64: "<d"}[kind], value)
    element, items = value
    return struct.pack("<IQ", element, len(items)) + b"".join(_value(element, item) for item in items)


def export_encoder_gguf(model: ModelFile, path: str | Path) -> Path:
    """Writes the BERT encoder ``model`` as a float32 GGUF v3 file in llama.cpp's ``bert`` layout."""
    config = model.config
    if config.family in ("modernbert", "deberta"):
        family = "ModernBERT" if config.family == "modernbert" else "DeBERTa"
        raise _export_error(f"{family} encoders cannot be written to GGUF here; export them to safetensors")
    if config.padding_index is not None:
        raise _export_error(
            "RoBERTa and XLM-RoBERTa encoders cannot be written to GGUF exactly (llama.cpp cuts the position rows "
            "before the padding token); export them to safetensors"
        )
    if config.activation != "gelu":
        raise _export_error("llama.cpp's bert layout has no tanh-GELU variant; export the model to safetensors")
    if _tokenizer_model(model) != "WordPiece":
        raise _export_error("only BERT encoders with WordPiece vocabularies can be written to GGUF")
    spec = model.tokenizer["tokenizer_json"]  # type: ignore[index]
    tokens, kinds = phantom_vocabulary(spec)
    if len(tokens) != config.vocabulary_size:
        raise _export_error("the vocabulary and the embedding table have different sizes")
    a = "bert"
    name = (model.source.get("repository") or Path(model.path).stem).split("/")[-1]
    metadata: list[tuple[str, int, Any]] = [
        ("general.architecture", _STRING, a),
        ("general.name", _STRING, name),
        ("general.alignment", _U32, _ALIGNMENT),
        ("general.file_type", _U32, 0),
    ]
    if model.licence.get("spdx"):
        metadata.append(("general.license", _STRING, str(model.licence["spdx"]).lower()))
    pooling = 4 if config.classifier_labels else GGUF_POOLING[str((model.embedding or {}).get("pooling", "mean"))]
    metadata += [
        (f"{a}.context_length", _U32, config.context_length),
        (f"{a}.embedding_length", _U32, config.hidden_size),
        (f"{a}.feed_forward_length", _U32, config.intermediate_size),
        (f"{a}.block_count", _U32, config.layers),
        (f"{a}.attention.head_count", _U32, config.heads),
        (f"{a}.attention.layer_norm_epsilon", _F32, config.rms_norm_eps),
        (f"{a}.attention.causal", _BOOL, False),
        (f"{a}.pooling_type", _U32, pooling),
    ]
    if config.classifier_labels:
        labels = [str(label) for label in (model.classifier or {}).get("labels", [])]
        metadata.append((f"{a}.classifier.output_labels", _ARRAY, (_STRING, labels)))
    vocab = {token: i for i, token in enumerate(wordpiece_vocabulary(tokens, kinds))}
    tokenizer_config = model.tokenizer.get("tokenizer_config") or {}  # type: ignore[union-attr]
    metadata += [
        ("tokenizer.ggml.model", _STRING, "bert"),
        ("tokenizer.ggml.pre", _STRING, "default"),
        ("tokenizer.ggml.tokens", _ARRAY, (_STRING, tokens)),
        ("tokenizer.ggml.token_type", _ARRAY, (_I32, kinds)),
        ("tokenizer.ggml.token_type_count", _U32, config.type_vocabulary_size),
    ]
    for key, field, fallback in (
        ("cls_token", "cls_token_id", "[CLS]"),
        ("cls_token", "bos_token_id", "[CLS]"),
        ("sep_token", "seperator_token_id", "[SEP]"),
        ("sep_token", "eos_token_id", "[SEP]"),
        ("unk_token", "unknown_token_id", "[UNK]"),
        ("pad_token", "padding_token_id", "[PAD]"),
        ("mask_token", "mask_token_id", "[MASK]"),
    ):
        text = special_token_text(tokenizer_config.get(key)) or fallback
        if text in vocab:
            metadata.append((f"tokenizer.ggml.{field}", _U32, vocab[text]))
    metadata.append(("tokenizer.huggingface.json", _STRING, json.dumps(spec, sort_keys=True, ensure_ascii=False)))
    metadata.append(("dllm.attention.layer_norm_epsilon", _F64, config.rms_norm_eps))
    if config.classifier_labels:
        metadata.append(("dllm.classifier", _STRING, json.dumps(model.classifier, sort_keys=True)))
    elif model.embedding is not None:
        metadata.append(("dllm.embedding", _STRING, json.dumps(model.embedding, sort_keys=True)))

    names = [gguf_encoder_tensor_name(tensor) for tensor in tensor_order(model.tensors)]
    sources = dict(zip(names, tensor_order(model.tensors), strict=True))
    infos, offset = [], 0
    for tensor in names:
        shape = tuple(model.tensors[sources[tensor]].shape)
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
            block = _rows(model, sources[tensor]).tobytes()
            stream.write(block + b"\0" * (-len(block) % _ALIGNMENT))
    return path
