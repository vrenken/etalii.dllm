"""Small synthetic models in Hugging Face and GGUF layout, SmolLM2-shaped but tiny, for importer tests.

The weights come from the deterministic Gaussian generator, so every fixture is identical on every run.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from etalii_dllm.architecture import TransformerConfig
from etalii_dllm.numerics import fill_gaussian

TINY_LLAMA_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "vocab_size": 64,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-5,
    "rope_theta": 100000.0,
    "rope_scaling": None,
    "hidden_act": "silu",
    "tie_word_embeddings": True,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "torch_dtype": "bfloat16",
}


def tiny_config(family: str) -> dict:
    """The tiny config for ``family``: Qwen2 unties the head; Qwen3 adds QK-norm and a head size of its own; Mistral
    slides a window of 3 tokens (shorter than the test prompts, so the window matters); OLMo 2 normalises outputs
    instead of inputs and the whole query and key projections."""
    config = {**TINY_LLAMA_CONFIG, "model_type": family}
    if family == "olmo2":
        config.update(architectures=["Olmo2ForCausalLM"], tie_word_embeddings=False, attention_bias=False)
    elif family == "mistral":
        config.update(architectures=["MistralForCausalLM"], sliding_window=3, tie_word_embeddings=False)
    elif family == "qwen2":
        config.update(architectures=["Qwen2ForCausalLM"], use_sliding_window=False, tie_word_embeddings=False)
    elif family == "qwen3":
        config.update(architectures=["Qwen3ForCausalLM"], use_sliding_window=False, head_dim=8, attention_bias=False)
    return config


TOKENIZER_JSON = {"version": "1.0", "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": []}}
TOKENIZER_CONFIG = {"chat_template": "{% for m in messages %}{{ m['content'] }}{% endfor %}", "eos_token": "</s>"}
MODEL_CARD = "---\nlicense: apache-2.0\nlibrary_name: transformers\n---\n\n# Tiny test model\n"


def to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """float32 -> bfloat16 bit patterns with round-to-nearest-even (as PyTorch does)."""
    bits = np.ascontiguousarray(values, dtype="<f4").view("<u4").astype(np.uint64)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return rounded.astype("<u2")


def bf16_to_float32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype("<u4") << np.uint32(16)).view("<f4")


def head_dim(config: dict) -> int:
    return config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"]


def hf_weights(config: dict, seed: int = 11) -> dict[str, np.ndarray]:
    """Float32 weights (already bf16-representable) for a Llama/Qwen2/Qwen3 config, keyed by Hugging Face name."""
    family = config["model_type"]
    ours = TransformerConfig.from_dict(
        {
            "family": family if family in ("mistral", "olmo2", "qwen2", "qwen3") else "llama",
            "vocabulary_size": config["vocab_size"],
            "hidden_size": config["hidden_size"],
            "intermediate_size": config["intermediate_size"],
            "layers": config["num_hidden_layers"],
            "heads": config["num_attention_heads"],
            "kv_heads": config["num_key_value_heads"],
            "head_dim": head_dim(config),
            "context_length": config["max_position_embeddings"],
            "rms_norm_eps": config["rms_norm_eps"],
            "rope_theta": config["rope_theta"],
            "attention_bias": family == "qwen2",
            "qk_norm": family in ("olmo2", "qwen3"),
            "qk_norm_scope": "all" if family == "olmo2" else "head",
            "norm_placement": "post" if family == "olmo2" else "pre",
            "tie_word_embeddings": config["tie_word_embeddings"],
        }
    )
    to_hf = {
        "token_embedding.weight": "model.embed_tokens.weight",
        "final_norm.weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    layer_names = {
        "attention_norm.weight": "input_layernorm.weight",
        "mlp_norm.weight": "post_attention_layernorm.weight",
        "mlp.gate.weight": "mlp.gate_proj.weight",
        "mlp.up.weight": "mlp.up_proj.weight",
        "mlp.down.weight": "mlp.down_proj.weight",
        "attention_post_norm.weight": "post_attention_layernorm.weight",
        "mlp_post_norm.weight": "post_feedforward_layernorm.weight",
    }
    weights = {}
    for index, (name, shape) in enumerate(sorted(ours.tensor_shapes().items())):
        if name in to_hf:
            hf = to_hf[name]
        else:
            _, layer, rest = name.split(".", 2)
            if rest.startswith("attention."):
                _, projection, kind = rest.split(".")
                hf_rest = (
                    f"self_attn.{projection}.{kind}"
                    if projection.endswith("_norm")
                    else f"self_attn.{projection}_proj.{kind}"
                )
            else:
                hf_rest = layer_names[rest]
            hf = f"model.layers.{layer}.{hf_rest}"
        values = fill_gaussian(seed * 1000 + index, int(np.prod(shape))).reshape(shape) * np.float32(0.1)
        weights[hf] = bf16_to_float32(to_bf16_bits(values))
    return weights


def write_safetensors(path: Path, tensors: dict[str, tuple[str, np.ndarray]], metadata: dict | None = None) -> None:
    """Minimal safetensors writer: ``tensors`` maps names to (dtype, stored bits)."""
    header: dict = {"__metadata__": metadata} if metadata else {}
    offset = 0
    for name, (dtype, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(raw.shape), "data_offsets": [offset, offset + raw.nbytes]}
        offset += raw.nbytes
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for _, raw in tensors.values():
            stream.write(np.ascontiguousarray(raw).tobytes())


def write_hf_checkpoint(
    directory: Path, config: dict | None = None, card: str = MODEL_CARD, tokenizer_json: dict | None = None
) -> dict[str, np.ndarray]:
    """A Hugging Face style checkpoint directory in bf16; returns the float32 weights it holds."""
    config = config or TINY_LLAMA_CONFIG
    directory.mkdir(parents=True, exist_ok=True)
    weights = hf_weights(config)
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "generation_config.json").write_text(json.dumps({"eos_token_id": [2, 3]}), encoding="utf-8")
    (directory / "tokenizer.json").write_text(json.dumps(tokenizer_json or TOKENIZER_JSON), encoding="utf-8")
    (directory / "tokenizer_config.json").write_text(json.dumps(TOKENIZER_CONFIG), encoding="utf-8")
    if card:
        (directory / "README.md").write_text(card, encoding="utf-8")
    write_safetensors(
        directory / "model.safetensors",
        {name: ("BF16", to_bf16_bits(values)) for name, values in weights.items()},
        {"format": "pt"},
    )
    return weights


def llama_cpp_permute(weights: np.ndarray, heads: int) -> np.ndarray:
    """``LlamaModel.permute`` from llama.cpp's ``convert_hf_to_gguf.py``."""
    return (
        weights.reshape(heads, 2, weights.shape[0] // heads // 2, *weights.shape[1:])
        .swapaxes(1, 2)
        .reshape(weights.shape)
    )


_HF_TO_GGUF = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_proj.bias": "attn_q.bias",
    "self_attn.k_proj.bias": "attn_k.bias",
    "self_attn.v_proj.bias": "attn_v.bias",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}


def gguf_name(hf_name: str) -> str:
    if hf_name == "model.embed_tokens.weight":
        return "token_embd.weight"
    if hf_name == "model.norm.weight":
        return "output_norm.weight"
    if hf_name == "lm_head.weight":
        return "output.weight"
    _, _, layer, rest = hf_name.split(".", 3)
    return f"blk.{layer}.{_HF_TO_GGUF[rest]}"


def write_gguf(path: Path, config: dict | None = None, quantization: str | None = None, licence: str = "apache-2.0"):
    """The same weights as :func:`write_hf_checkpoint`, converted the way llama.cpp does (Q/K permuted for Llama).
    ``quantization`` (e.g. ``"Q8_0"``) quantises, with gguf-py, the 2-D weights whose rows are whole blocks.
    Needs the ``gguf`` package."""
    import gguf

    config = config or TINY_LLAMA_CONFIG
    family = config["model_type"]
    weights = hf_weights(config)
    writer = gguf.GGUFWriter(str(path), family)
    writer.add_name("Tiny test model")
    writer.add_string("general.license", licence)
    writer.add_context_length(config["max_position_embeddings"])
    writer.add_embedding_length(config["hidden_size"])
    writer.add_block_count(config["num_hidden_layers"])
    writer.add_feed_forward_length(config["intermediate_size"])
    writer.add_head_count(config["num_attention_heads"])
    writer.add_head_count_kv(config["num_key_value_heads"])
    writer.add_rope_dimension_count(head_dim(config))
    writer.add_key_length(head_dim(config))
    writer.add_value_length(head_dim(config))
    writer.add_rope_freq_base(config["rope_theta"])
    writer.add_layer_norm_rms_eps(config["rms_norm_eps"])
    writer.add_vocab_size(config["vocab_size"])
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list([f"t{i}" for i in range(config["vocab_size"])])
    writer.add_bos_token_id(config["bos_token_id"])
    writer.add_eos_token_id(config["eos_token_id"])
    writer.add_chat_template(TOKENIZER_CONFIG["chat_template"])
    for name, values in weights.items():
        if family == "llama" and ".q_proj." in name:
            values = llama_cpp_permute(values, config["num_attention_heads"])
        elif family == "llama" and ".k_proj." in name:
            values = llama_cpp_permute(values, config["num_key_value_heads"])
        kind = gguf.GGMLQuantizationType[quantization] if quantization else None
        if kind is not None and values.ndim == 2 and values.shape[1] % gguf.GGML_QUANT_SIZES[kind][0] == 0:
            writer.add_tensor(gguf_name(name), gguf.quants.quantize(values, kind), raw_dtype=kind)
        else:
            writer.add_tensor(gguf_name(name), np.ascontiguousarray(values, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return weights
