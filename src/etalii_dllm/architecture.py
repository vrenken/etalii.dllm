"""Description of a decoder-only transformer, independent of where its weights came from.

Imported models are mapped onto this one description and onto one set of tensor names (see
``docs/model-format.md``), so the decoder does not need to know about Hugging Face or GGUF conventions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Families the decoder implements. "qwen2" is "llama" with biases on the q/k/v projections; "qwen3" is "llama" with
# an RMSNorm over each query and key head before the rotary embedding (QK-norm); "mistral" is "llama", usually with
# sliding-window attention; "olmo2" moves the norms after attention and the MLP and normalises the whole query and
# key projections.
FAMILIES = ("llama", "mistral", "olmo2", "qwen2", "qwen3")
NORM_PLACEMENTS = ("pre", "post")
QK_NORM_SCOPES = ("head", "all")


@dataclass(frozen=True)
class TransformerConfig:
    family: str
    vocabulary_size: int
    hidden_size: int
    intermediate_size: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    context_length: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: dict[str, Any] | None = None
    attention_bias: bool = False
    qk_norm: bool = False
    """RMSNorm on the queries and keys before RoPE: per head with weights ``[head_dim]`` shared by the heads (Qwen3),
    or over the whole projection with weights ``[heads * head_dim]`` (``qk_norm_scope == "all"``, OLMo 2)."""
    qk_norm_scope: str = "head"
    norm_placement: str = "pre"
    """``"pre"``: RMSNorm on the input of attention and of the MLP (Llama). ``"post"``: on their output, before it is
    added to the residual stream (OLMo 2)."""
    tie_word_embeddings: bool = False
    activation: str = "silu"
    # Rotary pairs (i, i + d/2) as in Hugging Face checkpoints. Imports convert other layouts to this one.
    rope_interleaved: bool = False
    bos_token_id: int | None = None
    eos_token_ids: tuple[int, ...] = field(default_factory=tuple)
    sliding_window: int | None = None
    """Sliding-window attention (Mistral): each query sees only the last ``sliding_window`` keys, itself included."""
    sliding_window_layers: tuple[int, ...] | None = None
    """The layers that use the sliding window; ``None`` means all of them."""

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unsupported model family {self.family!r}")
        for name in ("vocabulary_size", "hidden_size", "intermediate_size", "layers", "heads", "kv_heads", "head_dim"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.heads % self.kv_heads:
            raise ValueError("heads must be a multiple of kv_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.activation != "silu":
            raise ValueError(f"unsupported activation {self.activation!r}")
        if self.qk_norm_scope not in QK_NORM_SCOPES:
            raise ValueError(f"unsupported qk_norm_scope {self.qk_norm_scope!r}")
        if self.norm_placement not in NORM_PLACEMENTS:
            raise ValueError(f"unsupported norm_placement {self.norm_placement!r}")
        if self.sliding_window is not None and self.sliding_window < 1:
            raise ValueError("sliding_window must be positive")
        if self.sliding_window_layers is not None and any(
            not 0 <= layer < self.layers for layer in self.sliding_window_layers
        ):
            raise ValueError("sliding_window_layers must be layer indices")

    def window(self, layer: int) -> int | None:
        """The attention window of ``layer``: ``None`` for full causal attention."""
        if self.sliding_window is None:
            return None
        if self.sliding_window_layers is not None and layer not in self.sliding_window_layers:
            return None
        return self.sliding_window

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["eos_token_ids"] = list(self.eos_token_ids)
        if not self.qk_norm:  # model files written before QK-norm existed stay byte-identical
            del values["qk_norm"]
        for name in ("sliding_window", "sliding_window_layers"):  # likewise
            if values[name] is None:
                del values[name]
        for name, default in (("qk_norm_scope", "head"), ("norm_placement", "pre")):
            if values[name] == default:
                del values[name]
        if values.get("sliding_window_layers") is not None:
            values["sliding_window_layers"] = list(values["sliding_window_layers"])
        return values

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TransformerConfig:
        values = dict(values)
        values["eos_token_ids"] = tuple(values.get("eos_token_ids", ()))
        if values.get("sliding_window_layers") is not None:
            values["sliding_window_layers"] = tuple(values["sliding_window_layers"])
        return cls(**values)

    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        """Every tensor the decoder expects, with its shape. Weights use the ``[out, in]`` layout of ``linear``."""
        q = self.heads * self.head_dim
        kv = self.kv_heads * self.head_dim
        shapes: dict[str, tuple[int, ...]] = {"token_embedding.weight": (self.vocabulary_size, self.hidden_size)}
        for i in range(self.layers):
            p = f"layers.{i}."
            if self.norm_placement == "pre":
                shapes[p + "attention_norm.weight"] = (self.hidden_size,)
            shapes[p + "attention.q.weight"] = (q, self.hidden_size)
            shapes[p + "attention.k.weight"] = (kv, self.hidden_size)
            shapes[p + "attention.v.weight"] = (kv, self.hidden_size)
            if self.attention_bias:
                shapes[p + "attention.q.bias"] = (q,)
                shapes[p + "attention.k.bias"] = (kv,)
                shapes[p + "attention.v.bias"] = (kv,)
            if self.qk_norm:
                whole = self.qk_norm_scope == "all"
                shapes[p + "attention.q_norm.weight"] = (q if whole else self.head_dim,)
                shapes[p + "attention.k_norm.weight"] = (kv if whole else self.head_dim,)
            shapes[p + "attention.o.weight"] = (self.hidden_size, q)
            if self.norm_placement == "post":
                shapes[p + "attention_post_norm.weight"] = (self.hidden_size,)
                shapes[p + "mlp_post_norm.weight"] = (self.hidden_size,)
            else:
                shapes[p + "mlp_norm.weight"] = (self.hidden_size,)
            shapes[p + "mlp.gate.weight"] = (self.intermediate_size, self.hidden_size)
            shapes[p + "mlp.up.weight"] = (self.intermediate_size, self.hidden_size)
            shapes[p + "mlp.down.weight"] = (self.hidden_size, self.intermediate_size)
        shapes["final_norm.weight"] = (self.hidden_size,)
        if not self.tie_word_embeddings:
            shapes["lm_head.weight"] = (self.vocabulary_size, self.hidden_size)
        return shapes
