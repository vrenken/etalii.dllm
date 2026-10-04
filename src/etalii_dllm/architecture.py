"""Description of a transformer (decoder-only, or a BERT-style encoder), independent of where its weights came from.

Imported models are mapped onto this one description and onto one set of tensor names (see
``docs/model-format.md``), so the decoder does not need to know about Hugging Face or GGUF conventions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

# Families the decoder implements. "qwen2" is "llama" with biases on the q/k/v projections; "qwen3" is "llama" with
# an RMSNorm over each query and key head before the rotary embedding (QK-norm); "mistral" is "llama", usually with
# sliding-window attention; "olmo2" moves the norms after attention and the MLP and normalises the whole query and
# key projections; "granite" is "llama" with four scalar multipliers; "phi3" is "llama" whose checkpoints fuse the
# q/k/v and gate/up projections (imports split them), often with partial rotary embeddings and LongRoPE; "gemma3"
# normalises both the inputs and the outputs of attention and the MLP with (1 + weight) RMSNorms, gates with GELU
# (tanh), scales the embeddings and gives its sliding-window layers a RoPE base of their own; "gemma2" is the same
# without QK-norm or the second RoPE base, and soft-caps the attention scores and the logits. The mixture-of-experts
# families replace the MLP with experts and a router: "mixtral" is "mistral", "qwen3_moe" is "qwen3", "olmoe" is
# "llama" with OLMo's QK-norm over the whole projections, "qwen2_moe" is "qwen2" with a gated shared expert and
# "granitemoe" is "granite" (with or without a shared expert). "bert" is the one encoder: absolute position and token
# type embeddings, LayerNorms with biases after the embeddings, attention and the MLP, bidirectional attention and
# a plain (ungated) GELU MLP, all with biases (:mod:`etalii_dllm.encoder`); it embeds text and does not generate.
# RoBERTa and XLM-RoBERTa are "bert" with a ``padding_index``: their positions count from past the padding token.
# "modernbert" is the other encoder: rotary positions with a global and a local base, pre-norm layers with bias-free
# LayerNorms, no biases, bidirectional attention that alternates between global and local (windowed) layers and a
# gated GELU MLP, then a final norm. "deberta" (DeBERTa-v2 and v3) is BERT's post-norm layout without absolute
# positions: its attention adds content-to-position and position-to-content scores over a shared table of
# log-bucketed relative position embeddings (disentangled attention). "t5" is the encoder of T5 (sentence-t5,
# GTR-T5): no positions in the embeddings, pre-norm layers with RMS norms that have neither a mean nor a bias, no
# biases and no score scaling, a per-head bias over log-bucketed relative positions (the first layer's, shared by
# every layer), a ReLU MLP (T5 v1.0) or a gated one (v1.1), then a final norm. A "t5" with ``decoder_layers`` is a
# text-to-text model (T5, Flan-T5): that encoder plus a decoder with causal self-attention over one-directional
# buckets of its own and cross-attention over the encoder's states (:mod:`etalii_dllm.seq2seq`).
FAMILIES = (
    "bert",
    "deberta",
    "gemma2",
    "gemma3",
    "granite",
    "granitemoe",
    "llama",
    "mistral",
    "mixtral",
    "modernbert",
    "olmo2",
    "olmoe",
    "phi3",
    "qwen2",
    "qwen2_moe",
    "qwen3",
    "qwen3_moe",
    "t5",
)
NORM_PLACEMENTS = ("pre", "post", "sandwich")
ACTIVATIONS = ("silu", "gelu_tanh", "gelu", "relu")
QK_NORM_SCOPES = ("head", "all")
ENCODER_FAMILIES = ("bert", "deberta", "modernbert", "t5")
CLASSIFIER_POOLINGS = ("cls", "mean")
ENCODER_ONLY = "an encoder model embeds text and cannot generate; use it with dllm embed, /v1/embeddings or dllm index"


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
    added to the residual stream (OLMo 2). ``"sandwich"``: both (Gemma)."""
    norm_unit_offset: bool = False
    """Gemma: every RMSNorm (the q/k norms and the final norm included) scales by ``1 + weight``."""
    tie_word_embeddings: bool = False
    activation: str = "silu"
    """The MLP gate: ``silu`` (SwiGLU) or ``gelu_tanh`` (GeGLU with the tanh approximation, Gemma)."""
    # Rotary pairs (i, i + d/2) as in Hugging Face checkpoints. Imports convert other layouts to this one.
    rope_interleaved: bool = False
    bos_token_id: int | None = None
    eos_token_ids: tuple[int, ...] = field(default_factory=tuple)
    embedding_multiplier: float = 1.0
    """Granite: the embedding rows are multiplied by this before the first layer."""
    attention_multiplier: float | None = None
    """Granite: the attention score scale; ``None`` means ``1 / sqrt(head_dim)``."""
    residual_multiplier: float = 1.0
    """Granite: the outputs of attention and the MLP are scaled by this before the residual add."""
    logits_scaling: float = 1.0
    """Granite: the logits are divided by this."""
    sliding_window: int | None = None
    """Sliding-window attention (Mistral): each query sees only the last ``sliding_window`` keys, itself included.
    In an encoder (ModernBERT's local layers) each position sees the keys closer than ``sliding_window`` on either
    side."""
    sliding_window_layers: tuple[int, ...] | None = None
    """The layers that use the sliding window; ``None`` means all of them."""
    local_rope_theta: float | None = None
    """Gemma 3: the RoPE base of the sliding-window layers, which use no RoPE scaling; ``None`` means ``rope_theta``."""
    attention_softcap: float | None = None
    """Gemma 2: each scaled attention score ``s`` becomes ``cap * tanh(s / cap)`` before the softmax."""
    logits_softcap: float | None = None
    """Gemma 2: the logits become ``cap * tanh(logits / cap)``."""
    rotary_dim: int | None = None
    """Partial rotary embeddings (Phi-4-mini): only the first ``rotary_dim`` dimensions of each head rotate; ``None``
    means all ``head_dim``."""
    experts: int = 0
    """Mixture of experts: the number of experts of each sparse MLP block; 0 for a dense model."""
    experts_per_token: int = 0
    """How many experts each token is routed to (the top-k of the router's softmax)."""
    expert_intermediate_size: int | None = None
    """The hidden size of each expert's MLP; ``None`` means ``intermediate_size``."""
    normalize_expert_weights: bool = False
    """Whether the chosen experts' probabilities are divided by their total (Mixtral; ``norm_topk_prob``)."""
    dense_layers: tuple[int, ...] | None = None
    """The layers of a mixture-of-experts model that keep a dense MLP of ``intermediate_size``; ``None`` means none."""
    shared_expert_intermediate_size: int | None = None
    """The hidden size of the shared expert every token of a sparse layer runs next to its routed experts (Qwen2-MoE,
    Granite MoE shared); ``None`` means there is none."""
    shared_expert_gate: bool = False
    """Whether the shared expert's output is scaled by ``sigmoid(h . w)`` with a gate weight ``[1, hidden]``
    (Qwen2-MoE)."""
    type_vocabulary_size: int = 0
    """BERT: the rows of the token type embedding (the first text of a pair is type 0, the second type 1)."""
    classifier_labels: int = 0
    """BERT sequence classification (cross-encoders): the number of labels of the classifier on the pooled
    ``[CLS]`` state; 0 for an encoder without a classification head."""
    padding_index: int | None = None
    """RoBERTa and XLM-RoBERTa: the padding token's id, which their position ids count from (the first ordinary
    token sits at ``padding_index + 1``, a padding token at ``padding_index``; :meth:`position_ids`); ``None`` for
    BERT's positions 0, 1, ... The position table then has ``context_length + padding_index + 1`` rows."""
    classifier_pooling: str | None = None
    """ModernBERT sequence classification: the state the head reads, ``"cls"`` (the first position) or ``"mean"``
    (the mean over every position); ``None`` for BERT, whose head always reads ``[CLS]``."""
    position_buckets: int = 0
    """DeBERTa: the log buckets of relative positions (``position_buckets``; relative distances of at least half of
    it share buckets that grow logarithmically up to ``max_relative_positions``); 0 means no bucketing."""
    max_relative_positions: int = 0
    """DeBERTa: the largest relative distance the buckets reach; the relative position table has
    ``2 * relative_span`` rows (:attr:`relative_span`). T5: ``relative_attention_max_distance``, the distance from
    which every position shares the last bucket (``position_buckets`` is then ``relative_attention_num_buckets``,
    half of them for each direction). 0 for every other family."""
    gated_mlp: bool = False
    """T5 v1.1: the MLP is ``down(act(gate(x)) * up(x))`` rather than ``down(act(up(x)))``."""
    projection_size: int = 0
    """Embedders whose sentence-transformers pipeline has a ``Dense`` module after the pooling (sentence-t5,
    GTR-T5, LaBSE): the size of the projected sentence vector (``projection.weight`` is ``[projection_size,
    hidden]``); 0 for none."""
    projection_bias: bool = False
    """Whether that projection has a bias (``projection.bias``)."""
    decoder_layers: int = 0
    """T5 text-to-text models (T5ForConditionalGeneration, Flan-T5): the decoder's layers (``decoder.*`` tensors);
    the model then generates text (:mod:`etalii_dllm.seq2seq`) instead of embedding it. 0 for every other model."""

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unsupported model family {self.family!r}")
        for name in ("vocabulary_size", "hidden_size", "intermediate_size", "layers", "heads", "kv_heads", "head_dim"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.heads % self.kv_heads:
            raise ValueError("heads must be a multiple of kv_heads")
        if self.head_dim % 2 and self.family not in ENCODER_FAMILIES:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"unsupported activation {self.activation!r}")
        if self.decoder_layers and (self.family != "t5" or self.decoder_layers < 0 or self.projection_size):
            raise ValueError("decoder_layers is for t5 text-to-text models, which have no projection")
        if self.family in ENCODER_FAMILIES:
            if self.family == "t5":
                if self.position_buckets < 2 or self.max_relative_positions < 1:
                    raise ValueError("t5 needs position_buckets (at least 2) and max_relative_positions")
                if self.type_vocabulary_size or self.padding_index is not None or self.classifier_labels:
                    raise ValueError("t5 has neither token types, padding positions nor a classification head")
            elif self.activation not in ("gelu", "gelu_tanh"):
                raise ValueError(f"{self.family} needs a gelu or gelu_tanh MLP")
            elif self.gated_mlp:
                raise ValueError("gated_mlp is for t5 (modernbert's MLP is always gated)")
            if self.family == "bert" and self.type_vocabulary_size < 1:
                raise ValueError("bert needs token types")
            relative = self.family in ("deberta", "t5")
            if relative != (self.max_relative_positions > 0) or self.position_buckets < 0:
                raise ValueError("deberta, and only deberta, needs max_relative_positions (and position_buckets >= 0)")
            if self.family == "deberta" and self.padding_index is not None:
                raise ValueError("deberta has no padding positions")
            if self.family == "modernbert" and (self.type_vocabulary_size or self.padding_index is not None):
                raise ValueError("modernbert has neither token types nor padding positions")
            if self.kv_heads != self.heads or self.experts:
                raise ValueError(f"{self.family} has neither grouped-query attention nor experts")
            if self.classifier_pooling is not None and (
                self.family != "modernbert" or self.classifier_pooling not in CLASSIFIER_POOLINGS
            ):
                raise ValueError("classifier_pooling is cls or mean, for modernbert only")
            if self.classifier_labels < 0:
                raise ValueError("classifier_labels must not be negative")
            if self.projection_size < 0 or (self.projection_bias and not self.projection_size):
                raise ValueError("projection_size must not be negative, and a projection bias needs a projection")
            if self.projection_size and self.classifier_labels:
                raise ValueError("a projection is for embedders, not cross-encoders")
            if self.padding_index is not None and self.padding_index < 0:
                raise ValueError("padding_index must not be negative")
        elif (
            self.activation == "gelu"
            or self.type_vocabulary_size
            or self.classifier_labels
            or self.padding_index is not None
            or self.classifier_pooling is not None
            or self.max_relative_positions
            or self.position_buckets
            or self.activation == "relu"
            or self.gated_mlp
            or self.projection_size
        ):
            raise ValueError(
                "the plain gelu or relu MLP, token types, classifiers, padding positions, relative positions and "
                "projections are for encoders only"
            )
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
        for name in ("attention_softcap", "logits_softcap"):
            cap = getattr(self, name)
            if cap is not None and not cap > 0:
                raise ValueError(f"{name} must be positive")
        if self.rotary_dim is not None and not (0 < self.rotary_dim <= self.head_dim and self.rotary_dim % 2 == 0):
            raise ValueError("rotary_dim must be even, positive and at most head_dim")
        if self.experts < 0 or (self.experts and not 1 <= self.experts_per_token <= self.experts):
            raise ValueError("experts_per_token must be between 1 and the number of experts")
        if not self.experts and (self.experts_per_token or self.dense_layers is not None):
            raise ValueError("experts_per_token and dense_layers need experts")
        if self.expert_intermediate_size is not None and self.expert_intermediate_size < 1:
            raise ValueError("expert_intermediate_size must be positive")
        if self.dense_layers is not None and any(not 0 <= layer < self.layers for layer in self.dense_layers):
            raise ValueError("dense_layers must be layer indices")
        if self.shared_expert_intermediate_size is not None and (
            not self.experts or self.shared_expert_intermediate_size < 1
        ):
            raise ValueError("a shared expert needs experts and a positive shared_expert_intermediate_size")
        if self.shared_expert_gate and self.shared_expert_intermediate_size is None:
            raise ValueError("shared_expert_gate needs a shared expert")
        kind = self.rope_scaling.get("rope_type") if self.rope_scaling else None
        if kind == "longrope":
            pairs = self.rotary_dimension // 2
            names = (
                ("short_factor", "long_factor") if self.rope_scaling.get("factor_set") == "long" else ("short_factor",)
            )  # type: ignore[union-attr]
            for name in names:
                if len(self.rope_scaling.get(name, ())) != pairs:  # type: ignore[union-attr]
                    raise ValueError(f"longrope {name} needs {pairs} values")
        if kind == "yarn":
            factor = self.rope_scaling.get("factor", 0)  # type: ignore[union-attr]
            original = self.rope_scaling.get("original_max_position_embeddings", 0)  # type: ignore[union-attr]
            if not factor > 0 or not original > 0:
                raise ValueError("yarn needs a positive factor and original_max_position_embeddings")
        if self.rope_attention_factor != 1.0 and self.qk_norm and self.norm_unit_offset:
            raise ValueError("a RoPE attention factor together with unit-offset QK-norm is not supported")

    @property
    def is_encoder(self) -> bool:
        """Whether the model is an encoder, BERT, DeBERTa, ModernBERT or T5 (it embeds text and does not generate)."""
        return self.family in ENCODER_FAMILIES and not self.decoder_layers

    @property
    def is_text_to_text(self) -> bool:
        """Whether the model is a T5 encoder-decoder that generates text from a source text (``decoder_layers``)."""
        return self.decoder_layers > 0

    @property
    def relative_span(self) -> int:
        """DeBERTa: half the rows of the relative position table, ``position_buckets`` when bucketed, else
        ``max_relative_positions``."""
        return self.position_buckets if self.position_buckets > 0 else self.max_relative_positions

    def position_ids(self, tokens: Sequence[int]) -> list[int]:
        """An encoder's position ids for ``tokens``: 0, 1, ... for BERT and ModernBERT; for RoBERTa
        (``padding_index``) the running count of non-padding tokens past ``padding_index``, and ``padding_index``
        itself for a padding token, as transformers' ``create_position_ids_from_input_ids`` computes them."""
        if self.padding_index is None:
            return list(range(len(tokens)))
        out: list[int] = []
        count = 0
        for token in tokens:
            if token == self.padding_index:
                out.append(self.padding_index)
            else:
                count += 1
                out.append(self.padding_index + count)
        return out

    @property
    def attention_scale(self) -> float:
        return 1.0 / math.sqrt(self.head_dim) if self.attention_multiplier is None else self.attention_multiplier

    @property
    def has_pre_norms(self) -> bool:
        return self.norm_placement in ("pre", "sandwich")

    @property
    def has_post_norms(self) -> bool:
        return self.norm_placement in ("post", "sandwich")

    def uses_local_rope(self, layer: int) -> bool:
        """Whether ``layer`` rotates with ``local_rope_theta`` (a sliding-window layer of Gemma 3)."""
        return self.local_rope_theta is not None and self.window(layer) is not None

    @property
    def rotary_dimension(self) -> int:
        return self.head_dim if self.rotary_dim is None else self.rotary_dim

    @property
    def rope_attention_factor(self) -> float:
        """The ``attention_factor`` of LongRoPE or YaRN, which scales the rotated query and key dimensions (1
        otherwise)."""
        if not self.rope_scaling or self.rope_scaling.get("rope_type") not in ("longrope", "yarn"):
            return 1.0
        return float(self.rope_scaling.get("attention_factor", 1.0))

    @property
    def has_multipliers(self) -> bool:
        """Whether any Granite multiplier differs from the plain Llama value."""
        multipliers = (self.embedding_multiplier, self.attention_multiplier, self.residual_multiplier)
        return multipliers != (1.0, None, 1.0) or self.logits_scaling != 1.0

    def is_sparse(self, layer: int) -> bool:
        """Whether ``layer``'s MLP is a mixture of experts."""
        return self.experts > 0 and (self.dense_layers is None or layer not in self.dense_layers)

    @property
    def expert_size(self) -> int:
        """The hidden size of each expert's MLP."""
        return self.intermediate_size if self.expert_intermediate_size is None else self.expert_intermediate_size

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
        optional = (
            "sliding_window",
            "sliding_window_layers",
            "rotary_dim",
            "local_rope_theta",
            "expert_intermediate_size",
            "dense_layers",
            "shared_expert_intermediate_size",
            "padding_index",
            "classifier_pooling",
        )
        for name in (*optional, "attention_softcap", "logits_softcap"):  # likewise
            if values[name] is None:
                del values[name]
        defaults = (
            ("qk_norm_scope", "head"),
            ("norm_placement", "pre"),
            ("norm_unit_offset", False),
            ("embedding_multiplier", 1.0),
            ("attention_multiplier", None),
            ("residual_multiplier", 1.0),
            ("logits_scaling", 1.0),
            ("experts", 0),
            ("experts_per_token", 0),
            ("normalize_expert_weights", False),
            ("shared_expert_gate", False),
            ("type_vocabulary_size", 0),
            ("classifier_labels", 0),
            ("position_buckets", 0),
            ("max_relative_positions", 0),
            ("gated_mlp", False),
            ("projection_size", 0),
            ("projection_bias", False),
            ("decoder_layers", 0),
        )
        for name, default in defaults:
            if values[name] == default:
                del values[name]
        for name in ("sliding_window_layers", "dense_layers"):
            if values.get(name) is not None:
                values[name] = list(values[name])
        return values

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TransformerConfig:
        values = dict(values)
        values["eos_token_ids"] = tuple(values.get("eos_token_ids", ()))
        for name in ("sliding_window_layers", "dense_layers"):
            if values.get(name) is not None:
                values[name] = tuple(values[name])
        return cls(**values)

    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        """Every tensor the model expects, with its shape. Weights use the ``[out, in]`` layout of ``linear``."""
        if self.is_encoder:
            return self._encoder_shapes()
        if self.is_text_to_text:
            return self._text_to_text_shapes()
        q = self.heads * self.head_dim
        kv = self.kv_heads * self.head_dim
        shapes: dict[str, tuple[int, ...]] = {"token_embedding.weight": (self.vocabulary_size, self.hidden_size)}
        for i in range(self.layers):
            p = f"layers.{i}."
            if self.has_pre_norms:
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
            if self.has_post_norms:
                shapes[p + "attention_post_norm.weight"] = (self.hidden_size,)
                shapes[p + "mlp_post_norm.weight"] = (self.hidden_size,)
            if self.has_pre_norms:
                shapes[p + "mlp_norm.weight"] = (self.hidden_size,)
            if self.is_sparse(i):
                shapes[p + "mlp.router.weight"] = (self.experts, self.hidden_size)
                for e in range(self.experts):
                    shapes[p + f"mlp.experts.{e}.gate.weight"] = (self.expert_size, self.hidden_size)
                    shapes[p + f"mlp.experts.{e}.up.weight"] = (self.expert_size, self.hidden_size)
                    shapes[p + f"mlp.experts.{e}.down.weight"] = (self.hidden_size, self.expert_size)
                if self.shared_expert_intermediate_size is not None:
                    size = self.shared_expert_intermediate_size
                    shapes[p + "mlp.shared.gate.weight"] = (size, self.hidden_size)
                    shapes[p + "mlp.shared.up.weight"] = (size, self.hidden_size)
                    shapes[p + "mlp.shared.down.weight"] = (self.hidden_size, size)
                    if self.shared_expert_gate:
                        shapes[p + "mlp.shared_gate.weight"] = (1, self.hidden_size)
                continue
            shapes[p + "mlp.gate.weight"] = (self.intermediate_size, self.hidden_size)
            shapes[p + "mlp.up.weight"] = (self.intermediate_size, self.hidden_size)
            shapes[p + "mlp.down.weight"] = (self.hidden_size, self.intermediate_size)
        shapes["final_norm.weight"] = (self.hidden_size,)
        if not self.tie_word_embeddings:
            shapes["lm_head.weight"] = (self.vocabulary_size, self.hidden_size)
        return shapes

    def _encoder_shapes(self) -> dict[str, tuple[int, ...]]:
        shapes = self._encoder_body_shapes()
        if self.projection_size:
            shapes["projection.weight"] = (self.projection_size, self.hidden_size)
            if self.projection_bias:
                shapes["projection.bias"] = (self.projection_size,)
        return shapes

    def _encoder_body_shapes(self) -> dict[str, tuple[int, ...]]:
        if self.family == "modernbert":
            return self._modernbert_shapes()
        if self.family == "t5":
            return self._t5_shapes()
        hidden = self.hidden_size
        shapes: dict[str, tuple[int, ...]] = {"token_embedding.weight": (self.vocabulary_size, hidden)}
        if self.family == "bert":
            positions = self.context_length + (0 if self.padding_index is None else self.padding_index + 1)
            shapes["position_embedding.weight"] = (positions, hidden)
        if self.type_vocabulary_size:
            shapes["token_type_embedding.weight"] = (self.type_vocabulary_size, hidden)
        shapes["embedding_norm.weight"] = (hidden,)
        shapes["embedding_norm.bias"] = (hidden,)
        if self.family == "deberta":
            shapes["relative_embedding.weight"] = (2 * self.relative_span, hidden)
            shapes["relative_norm.weight"] = (hidden,)
            shapes["relative_norm.bias"] = (hidden,)
        q = self.heads * self.head_dim
        for i in range(self.layers):
            p = f"layers.{i}."
            for name in ("q", "k", "v"):
                shapes[p + f"attention.{name}.weight"] = (q, hidden)
                shapes[p + f"attention.{name}.bias"] = (q,)
            shapes[p + "attention.o.weight"] = (hidden, q)
            shapes[p + "attention.o.bias"] = (hidden,)
            shapes[p + "attention_norm.weight"] = (hidden,)
            shapes[p + "attention_norm.bias"] = (hidden,)
            shapes[p + "mlp.up.weight"] = (self.intermediate_size, hidden)
            shapes[p + "mlp.up.bias"] = (self.intermediate_size,)
            shapes[p + "mlp.down.weight"] = (hidden, self.intermediate_size)
            shapes[p + "mlp.down.bias"] = (hidden,)
            shapes[p + "mlp_norm.weight"] = (hidden,)
            shapes[p + "mlp_norm.bias"] = (hidden,)
        if self.classifier_labels:
            shapes["pooler.weight"] = (hidden, hidden)
            shapes["pooler.bias"] = (hidden,)
            shapes["classifier.weight"] = (self.classifier_labels, hidden)
            shapes["classifier.bias"] = (self.classifier_labels,)
        return shapes

    def _t5_shapes(self) -> dict[str, tuple[int, ...]]:
        hidden, q, size = self.hidden_size, self.heads * self.head_dim, self.intermediate_size
        shapes: dict[str, tuple[int, ...]] = {
            "token_embedding.weight": (self.vocabulary_size, hidden),
            "relative_bias.weight": (self.position_buckets, self.heads),
        }
        for i in range(self.layers):
            p = f"layers.{i}."
            shapes[p + "attention_norm.weight"] = (hidden,)
            for name in ("q", "k", "v"):
                shapes[p + f"attention.{name}.weight"] = (q, hidden)
            shapes[p + "attention.o.weight"] = (hidden, q)
            shapes[p + "mlp_norm.weight"] = (hidden,)
            if self.gated_mlp:
                shapes[p + "mlp.gate.weight"] = (size, hidden)
            shapes[p + "mlp.up.weight"] = (size, hidden)
            shapes[p + "mlp.down.weight"] = (hidden, size)
        shapes["final_norm.weight"] = (hidden,)
        return shapes

    def _text_to_text_shapes(self) -> dict[str, tuple[int, ...]]:
        """The T5 encoder's tensors, then the decoder's under ``decoder.`` (its own bucket table, each layer's
        self-attention, cross-attention over the encoder states and MLP, a final norm), then ``lm_head`` unless it
        is tied to the word embedding."""
        hidden, q, size = self.hidden_size, self.heads * self.head_dim, self.intermediate_size
        shapes = self._t5_shapes()
        shapes["decoder.relative_bias.weight"] = (self.position_buckets, self.heads)
        for i in range(self.decoder_layers):
            p = f"decoder.layers.{i}."
            for block in ("attention", "cross"):
                shapes[p + f"{block}_norm.weight"] = (hidden,)
                for name in ("q", "k", "v"):
                    shapes[p + f"{block}.{name}.weight"] = (q, hidden)
                shapes[p + f"{block}.o.weight"] = (hidden, q)
            shapes[p + "mlp_norm.weight"] = (hidden,)
            if self.gated_mlp:
                shapes[p + "mlp.gate.weight"] = (size, hidden)
            shapes[p + "mlp.up.weight"] = (size, hidden)
            shapes[p + "mlp.down.weight"] = (hidden, size)
        shapes["decoder.final_norm.weight"] = (hidden,)
        if not self.tie_word_embeddings:
            shapes["lm_head.weight"] = (self.vocabulary_size, hidden)
        return shapes

    def _modernbert_shapes(self) -> dict[str, tuple[int, ...]]:
        hidden, q, size = self.hidden_size, self.heads * self.head_dim, self.intermediate_size
        shapes: dict[str, tuple[int, ...]] = {
            "token_embedding.weight": (self.vocabulary_size, hidden),
            "embedding_norm.weight": (hidden,),
        }
        for i in range(self.layers):
            p = f"layers.{i}."
            if i:  # the first layer reads the normalised embeddings directly
                shapes[p + "attention_norm.weight"] = (hidden,)
            for name in ("q", "k", "v"):
                shapes[p + f"attention.{name}.weight"] = (q, hidden)
            shapes[p + "attention.o.weight"] = (hidden, q)
            shapes[p + "mlp_norm.weight"] = (hidden,)
            shapes[p + "mlp.gate.weight"] = (size, hidden)
            shapes[p + "mlp.up.weight"] = (size, hidden)
            shapes[p + "mlp.down.weight"] = (hidden, size)
        shapes["final_norm.weight"] = (hidden,)
        if self.classifier_labels:
            shapes["pooler.weight"] = (hidden, hidden)
            shapes["pooler_norm.weight"] = (hidden,)
            shapes["classifier.weight"] = (self.classifier_labels, hidden)
            shapes["classifier.bias"] = (self.classifier_labels,)
        return shapes
