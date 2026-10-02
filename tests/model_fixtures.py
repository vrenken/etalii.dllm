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
    "max_position_embeddings": 2048,
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
    instead of inputs and the whole query and key projections; Granite adds its four multipliers; Phi-3 fuses its
    projections, rotates half of each (wider) head and uses LongRoPE with a window of 3; Gemma 3 alternates a window of
    3 (with its own RoPE base) and full attention (with linear RoPE scaling); Gemma 2 does so with one RoPE base and
    soft-caps attention scores and logits at about their size, so the caps bend them. The mixture-of-experts configs
    route each token to 2 of 4 experts: Mixtral renormalises the two weights, OLMoE does not (and normalises the whole
    query and key projections), Qwen3-MoE renormalises, gives its experts a hidden size of their own and keeps a dense
    MLP in layer 0. Qwen2-MoE does not renormalise and adds a gated shared expert of its own size; Granite MoE adds
    Granite's multipliers, renormalises and stores its experts fused, and ``granitemoeshared`` adds a shared expert."""
    config = {**TINY_LLAMA_CONFIG, "model_type": family}
    if family == "gemma2":
        del config["hidden_act"]
        config.update(
            architectures=["Gemma2ForCausalLM"],
            hidden_activation="gelu_pytorch_tanh",
            pad_token_id=0,
            head_dim=8,
            query_pre_attn_scalar=12,
            sliding_window=3,
            attention_bias=False,
            attn_logit_softcapping=0.02,
            final_logit_softcapping=0.05,
        )
    elif family == "gemma3":
        del config["hidden_act"]
        config.update(
            architectures=["Gemma3ForCausalLM"],
            model_type="gemma3_text",
            hidden_activation="gelu_pytorch_tanh",
            pad_token_id=0,
            head_dim=8,
            query_pre_attn_scalar=12,
            sliding_window=3,
            layer_types=["sliding_attention", "full_attention"],
            rope_local_base_freq=100.0,
            rope_scaling={"rope_type": "linear", "factor": 2.0},
            attention_bias=False,
            final_logit_softcapping=None,
            attn_logit_softcapping=None,
        )
    elif family == "granite":
        config.update(
            architectures=["GraniteForCausalLM"],
            embedding_multiplier=12.0,
            attention_multiplier=0.125,
            residual_multiplier=0.22,
            logits_scaling=8.0,
        )
    elif family == "phi3":
        config.update(
            architectures=["Phi3ForCausalLM"],
            pad_token_id=0,
            hidden_size=32,
            tie_word_embeddings=False,
            sliding_window=3,
            partial_rotary_factor=0.5,
            original_max_position_embeddings=64,
            rope_scaling={
                "type": "longrope",
                "short_factor": [4.0, 1.0],
                "long_factor": [2.0, 4.0],
                "attention_factor": 4.0,
            },
        )
    elif family == "olmo2":
        config.update(architectures=["Olmo2ForCausalLM"], tie_word_embeddings=False, attention_bias=False)
    elif family == "mistral":
        config.update(architectures=["MistralForCausalLM"], sliding_window=3, tie_word_embeddings=False)
    elif family == "qwen2":
        config.update(architectures=["Qwen2ForCausalLM"], use_sliding_window=False, tie_word_embeddings=False)
    elif family == "qwen3":
        config.update(architectures=["Qwen3ForCausalLM"], use_sliding_window=False, head_dim=8, attention_bias=False)
    elif family == "mixtral":
        config.update(
            architectures=["MixtralForCausalLM"],
            sliding_window=None,
            tie_word_embeddings=False,
            num_local_experts=4,
            num_experts_per_tok=2,
            router_jitter_noise=0.0,
        )
    elif family == "olmoe":
        config.update(
            architectures=["OlmoeForCausalLM"],
            tie_word_embeddings=False,
            attention_bias=False,
            intermediate_size=8,
            num_experts=4,
            num_experts_per_tok=2,
            norm_topk_prob=False,
            clip_qkv=None,
        )
    elif family == "qwen2_moe":
        config.update(
            architectures=["Qwen2MoeForCausalLM"],
            use_sliding_window=False,
            tie_word_embeddings=False,
            moe_intermediate_size=8,
            shared_expert_intermediate_size=12,
            num_experts=4,
            num_experts_per_tok=2,
            norm_topk_prob=False,
            decoder_sparse_step=1,
            mlp_only_layers=[],
        )
    elif family in ("granitemoe", "granitemoeshared"):
        config.update(
            architectures=["GraniteMoeSharedForCausalLM" if family == "granitemoeshared" else "GraniteMoeForCausalLM"],
            embedding_multiplier=12.0,
            attention_multiplier=0.125,
            residual_multiplier=0.22,
            logits_scaling=8.0,
            intermediate_size=8,
            num_local_experts=4,
            num_experts_per_tok=2,
        )
        if family == "granitemoeshared":
            config["shared_intermediate_size"] = 12
    elif family == "qwen3_moe":
        config.update(
            architectures=["Qwen3MoeForCausalLM"],
            use_sliding_window=False,
            head_dim=8,
            attention_bias=False,
            moe_intermediate_size=8,
            num_experts=4,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            decoder_sparse_step=1,
            mlp_only_layers=[0],
        )
    return config


def moe_settings(config: dict) -> dict:
    """The mixture-of-experts fields of our config for a Hugging Face ``config`` (empty for dense models)."""
    family = {"granitemoeshared": "granitemoe"}.get(config["model_type"], config["model_type"])
    if family not in ("mixtral", "olmoe", "qwen2_moe", "qwen3_moe", "granitemoe"):
        return {}
    settings = {
        "experts": config.get("num_local_experts") or config["num_experts"],
        "experts_per_token": config["num_experts_per_tok"],
        "normalize_expert_weights": family in ("mixtral", "granitemoe") or bool(config.get("norm_topk_prob")),
    }
    if family in ("qwen2_moe", "qwen3_moe"):
        settings["expert_intermediate_size"] = config["moe_intermediate_size"]
        settings["dense_layers"] = tuple(config.get("mlp_only_layers") or ()) or None
    shared = config.get("shared_expert_intermediate_size") or config.get("shared_intermediate_size")
    if shared:
        settings["shared_expert_intermediate_size"] = shared
        settings["shared_expert_gate"] = family == "qwen2_moe"
    return settings


MIXTRAL_EXPERT_NAMES = {"gate": "w1", "down": "w2", "up": "w3"}

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
    family = {"gemma3_text": "gemma3", "granitemoeshared": "granitemoe"}.get(config["model_type"], config["model_type"])
    ours = TransformerConfig.from_dict(
        {
            "family": family
            if family
            in (
                "gemma2",
                "gemma3",
                "granite",
                "granitemoe",
                "mistral",
                "mixtral",
                "olmo2",
                "olmoe",
                "phi3",
                "qwen2",
                "qwen2_moe",
                "qwen3",
                "qwen3_moe",
            )
            else "llama",
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
            "attention_bias": family in ("qwen2", "qwen2_moe"),
            "qk_norm": family in ("gemma3", "olmo2", "olmoe", "qwen3", "qwen3_moe"),
            "qk_norm_scope": "all" if family in ("olmo2", "olmoe") else "head",
            "norm_placement": {"olmo2": "post", "gemma2": "sandwich", "gemma3": "sandwich"}.get(family, "pre"),
            "tie_word_embeddings": config["tie_word_embeddings"],
            **moe_settings(config),
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
    if family in ("gemma2", "gemma3"):
        layer_names["mlp_norm.weight"] = "pre_feedforward_layernorm.weight"
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
            elif rest.startswith("mlp.experts."):  # mlp.experts.<e>.<projection>.weight
                _, _, expert, projection, _ = rest.split(".")
                if family == "mixtral":
                    hf_rest = f"block_sparse_moe.experts.{expert}.{MIXTRAL_EXPERT_NAMES[projection]}.weight"
                else:
                    hf_rest = f"mlp.experts.{expert}.{projection}_proj.weight"
            elif rest == "mlp.router.weight":
                hf_rest = {
                    "mixtral": "block_sparse_moe.gate.weight",
                    "granitemoe": "block_sparse_moe.router.layer.weight",
                }
                hf_rest = hf_rest.get(family, "mlp.gate.weight")
            elif rest.startswith("mlp.shared."):  # Granite's are fused below
                hf_rest = f"mlp.shared_expert.{rest.split('.')[2]}_proj.weight"
            elif rest == "mlp.shared_gate.weight":
                hf_rest = "mlp.shared_expert_gate.weight"
            else:
                hf_rest = layer_names[rest]
            hf = f"model.layers.{layer}.{hf_rest}"
        values = fill_gaussian(seed * 1000 + index, int(np.prod(shape))).reshape(shape) * np.float32(0.1)
        if name.endswith("mlp.router.weight"):  # wider router logits, so the top-k choices are clear-cut
            values = values * np.float32(16)
        weights[hf] = bf16_to_float32(to_bf16_bits(values))
    if family == "granitemoe":  # experts stacked, gate and up rows fused, as GraniteMoeParallelExperts stores them
        for layer in range(config["num_hidden_layers"]):
            p = f"model.layers.{layer}."
            experts = range(ours.experts)
            weights[p + "block_sparse_moe.input_linear.weight"] = np.stack(
                [
                    np.concatenate([weights.pop(f"{p}mlp.experts.{e}.{x}_proj.weight") for x in ("gate", "up")])
                    for e in experts
                ]
            )
            weights[p + "block_sparse_moe.output_linear.weight"] = np.stack(
                [weights.pop(f"{p}mlp.experts.{e}.down_proj.weight") for e in experts]
            )
            if ours.shared_expert_intermediate_size is not None:
                weights[p + "shared_mlp.input_linear.weight"] = np.concatenate(
                    [weights.pop(f"{p}mlp.shared_expert.{x}_proj.weight") for x in ("gate", "up")]
                )
                weights[p + "shared_mlp.output_linear.weight"] = weights.pop(p + "mlp.shared_expert.down_proj.weight")
    if family == "phi3":  # fused projections, rows stacked q, k, v and gate, up
        for layer in range(config["num_hidden_layers"]):
            p = f"model.layers.{layer}."
            weights[p + "self_attn.qkv_proj.weight"] = np.concatenate(
                [weights.pop(p + f"self_attn.{x}_proj.weight") for x in "qkv"]
            )
            weights[p + "mlp.gate_up_proj.weight"] = np.concatenate(
                [weights.pop(p + f"mlp.{x}_proj.weight") for x in ("gate", "up")]
            )
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
        (directory / "README.md").write_bytes(card.encode("utf-8"))  # the same bytes on Windows
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
