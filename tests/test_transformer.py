"""Decoder tests: agreement with a float64 reference, KV cache transparency, and golden logits."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from golden_values import TINY_LOGITS_FINGERPRINT
from model_fixtures import tiny_config, write_hf_checkpoint

from etalii_dllm.architecture import TransformerConfig
from etalii_dllm.importing import import_model
from etalii_dllm.numerics import fingerprint
from etalii_dllm.transformer import Transformer

PROMPT = [1, 17, 42, 5, 63, 0, 9, 9, 30]


def yarn_inv_freq(inv_freq: np.ndarray, dim: int, theta: float, scaling: dict) -> np.ndarray:
    """transformers' ``_compute_yarn_parameters`` in float64 (the yardstick for YaRN frequencies)."""

    def correction(rotations):
        return dim * np.log(scaling["original_max_position_embeddings"] / (rotations * 2 * np.pi)) / (2 * np.log(theta))

    low, high = correction(scaling.get("beta_fast", 32)), correction(scaling.get("beta_slow", 1))
    if scaling.get("truncate", True):
        low, high = np.floor(low), np.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    high = high + 0.001 if low == high else high
    extrapolation = 1 - np.clip((np.arange(dim // 2) - low) / (high - low), 0, 1)
    return inv_freq / scaling["factor"] * (1 - extrapolation) + inv_freq * extrapolation


def reference_logits(
    config: TransformerConfig, w: dict[str, np.ndarray], tokens: list[int], *, every_position: bool = False
) -> np.ndarray:
    """Straightforward float64 NumPy version of the Hugging Face forward pass of every supported family (test-only;
    NumPy reductions are fine here because this is the yardstick, not the engine): the last position's logits, or
    every position's with ``every_position``."""
    w = {name: np.asarray(values, dtype=np.float64) for name, values in w.items()}
    n, hd, rd = len(tokens), config.head_dim, config.rotary_dimension

    def tables(theta, scaling):
        inv_freq = theta ** (-np.arange(0, rd, 2, dtype=np.float64) / rd)
        if scaling and scaling["rope_type"] == "longrope":
            inv_freq = inv_freq / np.array(scaling["long_factor" if scaling.get("factor_set") else "short_factor"])
        elif scaling and scaling["rope_type"] == "linear":
            inv_freq = inv_freq / scaling["factor"]
        elif scaling and scaling["rope_type"] == "yarn":
            inv_freq = yarn_inv_freq(inv_freq, rd, theta, scaling)
        angles = np.arange(n, dtype=np.float64)[:, None] * inv_freq[None, :]
        factor = config.rope_attention_factor
        angles = np.concatenate([angles, angles], 1)
        return np.cos(angles) * factor, np.sin(angles) * factor

    global_tables = tables(config.rope_theta, config.rope_scaling)
    local_tables = tables(config.local_rope_theta, None) if config.local_rope_theta else global_tables
    offset = 1.0 if config.norm_unit_offset else 0.0

    def norm(x, weight):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + config.rms_norm_eps) * (offset + weight)

    def rotate(x, cos, sin):  # x: [n, heads, hd]; only the first rd dimensions rotate
        r, rest = x[..., :rd], x[..., rd:]
        half = np.concatenate([-r[..., rd // 2 :], r[..., : rd // 2]], axis=-1)
        return np.concatenate([r * cos[:, None, :] + half * sin[:, None, :], rest], axis=-1)

    def act(gate):
        if config.activation == "gelu_tanh":
            return 0.5 * gate * (1 + np.tanh(np.sqrt(2 / np.pi) * (gate + 0.044715 * gate**3)))
        return gate / (1 + np.exp(-gate))

    def proj(x, name):
        out = x @ w[name + ".weight"].T
        return out + w[name + ".bias"] if name + ".bias" in w else out

    x = w["token_embedding.weight"][tokens] * config.embedding_multiplier
    group = config.heads // config.kv_heads
    mask = np.triu(np.full((n, n), -np.inf), 1)
    for i in range(config.layers):
        window = config.window(i)
        layer_mask = mask if window is None else mask + np.tril(np.full((n, n), -np.inf), -window)
        p = f"layers.{i}."
        pre, post = config.has_pre_norms, config.has_post_norms
        h = norm(x, w[p + "attention_norm.weight"]) if pre else x
        q, k = proj(h, p + "attention.q"), proj(h, p + "attention.k")
        if config.qk_norm and config.qk_norm_scope == "all":
            q, k = norm(q, w[p + "attention.q_norm.weight"]), norm(k, w[p + "attention.k_norm.weight"])
        q, k = q.reshape(n, config.heads, hd), k.reshape(n, config.kv_heads, hd)
        if config.qk_norm and config.qk_norm_scope == "head":
            q, k = norm(q, w[p + "attention.q_norm.weight"]), norm(k, w[p + "attention.k_norm.weight"])
        cos, sin = local_tables if config.uses_local_rope(i) else global_tables
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        v = proj(h, p + "attention.v").reshape(n, config.kv_heads, hd)
        out = np.empty((n, config.heads, hd))
        for head in range(config.heads):
            scores = q[:, head] @ k[:, head // group].T * config.attention_scale
            if config.attention_softcap:
                scores = config.attention_softcap * np.tanh(scores / config.attention_softcap)
            scores = scores + layer_mask
            probs = np.exp(scores - scores.max(-1, keepdims=True))
            out[:, head] = (probs / probs.sum(-1, keepdims=True)) @ v[:, head // group]
        attended = out.reshape(n, -1) @ w[p + "attention.o.weight"].T * config.residual_multiplier
        x = x + (norm(attended, w[p + "attention_post_norm.weight"]) if post else attended)
        h = norm(x, w[p + "mlp_norm.weight"]) if pre else x
        if config.is_sparse(i):  # softmax over the router logits, top-k, optionally renormalised
            logits = h @ w[p + "mlp.router.weight"].T
            probs = np.exp(logits - logits.max(-1, keepdims=True))
            probs = probs / probs.sum(-1, keepdims=True)
            mlp = np.zeros_like(h)
            for row in range(n):
                chosen = np.argsort(-probs[row], kind="stable")[: config.experts_per_token]
                weights = probs[row, chosen]
                if config.normalize_expert_weights:
                    weights = weights / weights.sum()
                for e, weight in zip(chosen, weights, strict=True):
                    q = f"{p}mlp.experts.{e}."
                    gate = h[row] @ w[q + "gate.weight"].T
                    mlp[row] += weight * ((act(gate) * (h[row] @ w[q + "up.weight"].T)) @ w[q + "down.weight"].T)
        else:
            gate = h @ w[p + "mlp.gate.weight"].T
            mlp = (act(gate) * (h @ w[p + "mlp.up.weight"].T)) @ w[p + "mlp.down.weight"].T
        mlp = mlp * config.residual_multiplier
        x = x + (norm(mlp, w[p + "mlp_post_norm.weight"]) if post else mlp)
    head = w["token_embedding.weight"] if config.tie_word_embeddings else w["lm_head.weight"]
    hidden = norm(x, w["final_norm.weight"])
    logits = (hidden if every_position else hidden[-1]) @ head.T / config.logits_scaling
    if config.logits_softcap:
        logits = config.logits_softcap * np.tanh(logits / config.logits_softcap)
    return logits


FAMILIES = sorted(TINY_LOGITS_FINGERPRINT)


@pytest.fixture(scope="module", params=FAMILIES)
def model(request, tmp_path_factory) -> Transformer:
    directory = tmp_path_factory.mktemp(request.param)
    config = tiny_config(request.param)
    write_hf_checkpoint(directory / "checkpoint", config)
    import_model(directory / "checkpoint", directory / "model.dllm")
    return Transformer.from_file(directory / "model.dllm")


def test_matches_float64_reference(model):
    tensors = {name: tensor.numpy() for name, tensor in model.tensors.items()}
    for length in (1, 2, len(PROMPT)):
        ours = model.forward(PROMPT[:length])
        reference = reference_logits(model.config, tensors, PROMPT[:length])
        assert ours.dtype == np.float32 and ours.shape == (model.vocabulary_size,)
        np.testing.assert_allclose(ours, reference, rtol=0, atol=2e-5 * np.abs(reference).max())


def test_kv_cache_does_not_change_logits(model):
    """Prefill, token-by-token decoding, cache reuse after a divergent suffix and full recompute: same bits."""
    full = [model.forward(PROMPT[: i + 1]) for i in range(len(PROMPT))]
    cache = model.new_cache()
    stepwise = [model.forward_cached(PROMPT[: i + 1], cache) for i in range(len(PROMPT))]
    for a, b in zip(full, stepwise, strict=True):
        assert a.tobytes() == b.tobytes()

    prefilled = model.new_cache()
    assert model.forward_cached(PROMPT, prefilled).tobytes() == full[-1].tobytes()
    assert model.forward_cached(PROMPT, prefilled).tobytes() == full[-1].tobytes()  # nothing new to process

    diverged = [*PROMPT[:4], 11, 12]
    assert model.forward_cached(diverged, prefilled).tobytes() == model.forward(diverged).tobytes()
    assert len(prefilled) == len(diverged)


def test_concurrent_requests_do_not_interfere(model):
    prompts = [PROMPT[: i + 1] for i in range(len(PROMPT))] * 3
    expected = [model.forward(p).tobytes() for p in prompts]
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert [r.tobytes() for r in pool.map(model.forward, prompts)] == expected


def test_logits_are_bit_exact(model):
    assert fingerprint(model.forward(PROMPT)) == TINY_LOGITS_FINGERPRINT[model.config.family]


def test_rejects_bad_input(model):
    with pytest.raises(ValueError):
        model.forward([])
    with pytest.raises(ValueError):
        model.forward([model.vocabulary_size])
