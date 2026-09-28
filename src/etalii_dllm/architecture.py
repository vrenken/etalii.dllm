"""Description of a decoder-only transformer, independent of where its weights came from.

Imported models are mapped onto this one description and onto one set of tensor names (see
``docs/model-format.md``), so the decoder does not need to know about Hugging Face or GGUF conventions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Families the decoder implements. "qwen2" is "llama" with biases on the q/k/v projections.
FAMILIES = ("llama", "qwen2")


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
    tie_word_embeddings: bool = False
    activation: str = "silu"
    # Rotary pairs (i, i + d/2) as in Hugging Face checkpoints. Imports convert other layouts to this one.
    rope_interleaved: bool = False
    bos_token_id: int | None = None
    eos_token_ids: tuple[int, ...] = field(default_factory=tuple)

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

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["eos_token_ids"] = list(self.eos_token_ids)
        return values

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TransformerConfig:
        values = dict(values)
        values["eos_token_ids"] = tuple(values.get("eos_token_ids", ()))
        return cls(**values)

    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        """Every tensor the decoder expects, with its shape. Weights use the ``[out, in]`` layout of ``linear``."""
        q = self.heads * self.head_dim
        kv = self.kv_heads * self.head_dim
        shapes: dict[str, tuple[int, ...]] = {"token_embedding.weight": (self.vocabulary_size, self.hidden_size)}
        for i in range(self.layers):
            p = f"layers.{i}."
            shapes[p + "attention_norm.weight"] = (self.hidden_size,)
            shapes[p + "attention.q.weight"] = (q, self.hidden_size)
            shapes[p + "attention.k.weight"] = (kv, self.hidden_size)
            shapes[p + "attention.v.weight"] = (kv, self.hidden_size)
            if self.attention_bias:
                shapes[p + "attention.q.bias"] = (q,)
                shapes[p + "attention.k.bias"] = (kv,)
                shapes[p + "attention.v.bias"] = (kv,)
            shapes[p + "attention.o.weight"] = (self.hidden_size, q)
            shapes[p + "mlp_norm.weight"] = (self.hidden_size,)
            shapes[p + "mlp.gate.weight"] = (self.intermediate_size, self.hidden_size)
            shapes[p + "mlp.up.weight"] = (self.intermediate_size, self.hidden_size)
            shapes[p + "mlp.down.weight"] = (self.hidden_size, self.intermediate_size)
        shapes["final_norm.weight"] = (self.hidden_size,)
        if not self.tie_word_embeddings:
            shapes["lm_head.weight"] = (self.vocabulary_size, self.hidden_size)
        return shapes
