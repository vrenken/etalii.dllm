"""Decoder tests: agreement with a float64 reference, KV cache transparency, and golden logits."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from golden_values import TINY_LOGITS_FINGERPRINT
from model_fixtures import TINY_LLAMA_CONFIG, write_hf_checkpoint

from etalii_dllm.architecture import TransformerConfig
from etalii_dllm.importing import import_model
from etalii_dllm.numerics import fingerprint
from etalii_dllm.transformer import Transformer

PROMPT = [1, 17, 42, 5, 63, 0, 9, 9, 30]


def reference_logits(config: TransformerConfig, w: dict[str, np.ndarray], tokens: list[int]) -> np.ndarray:
    """Straightforward float64 NumPy version of the Hugging Face Llama/Qwen2 forward pass (test-only; NumPy
    reductions are fine here because this is the yardstick, not the engine)."""
    w = {name: np.asarray(values, dtype=np.float64) for name, values in w.items()}
    n, d, hd = len(tokens), config.hidden_size, config.head_dim
    inv_freq = config.rope_theta ** (-np.arange(0, hd, 2, dtype=np.float64) / hd)
    angles = np.arange(n, dtype=np.float64)[:, None] * inv_freq[None, :]
    cos, sin = np.cos(np.concatenate([angles, angles], 1)), np.sin(np.concatenate([angles, angles], 1))

    def norm(x, weight):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + config.rms_norm_eps) * weight

    def rotate(x):  # x: [n, heads, hd]
        half = np.concatenate([-x[..., hd // 2 :], x[..., : hd // 2]], axis=-1)
        return x * cos[:, None, :] + half * sin[:, None, :]

    def proj(x, name):
        out = x @ w[name + ".weight"].T
        return out + w[name + ".bias"] if name + ".bias" in w else out

    x = w["token_embedding.weight"][tokens]
    group = config.heads // config.kv_heads
    mask = np.triu(np.full((n, n), -np.inf), 1)
    for i in range(config.layers):
        p = f"layers.{i}."
        h = norm(x, w[p + "attention_norm.weight"])
        q = rotate(proj(h, p + "attention.q").reshape(n, config.heads, hd))
        k = rotate(proj(h, p + "attention.k").reshape(n, config.kv_heads, hd))
        v = proj(h, p + "attention.v").reshape(n, config.kv_heads, hd)
        out = np.empty((n, config.heads, hd))
        for head in range(config.heads):
            scores = q[:, head] @ k[:, head // group].T / np.sqrt(hd) + mask
            probs = np.exp(scores - scores.max(-1, keepdims=True))
            out[:, head] = (probs / probs.sum(-1, keepdims=True)) @ v[:, head // group]
        x = x + out.reshape(n, d) @ w[p + "attention.o.weight"].T
        h = norm(x, w[p + "mlp_norm.weight"])
        gate = h @ w[p + "mlp.gate.weight"].T
        x = x + (gate / (1 + np.exp(-gate)) * (h @ w[p + "mlp.up.weight"].T)) @ w[p + "mlp.down.weight"].T
    head = w["token_embedding.weight"] if config.tie_word_embeddings else w["lm_head.weight"]
    return norm(x, w["final_norm.weight"])[-1] @ head.T


@pytest.fixture(scope="module", params=["llama", "qwen2"])
def model(request, tmp_path_factory) -> Transformer:
    directory = tmp_path_factory.mktemp(request.param)
    config = {**TINY_LLAMA_CONFIG, "model_type": request.param}
    if request.param == "qwen2":
        config["tie_word_embeddings"] = False
    write_hf_checkpoint(directory / "checkpoint", config)
    import_model(directory / "checkpoint", directory / "model.dllm")
    return Transformer.from_file(directory / "model.dllm")


def test_matches_float64_reference(model):
    tensors = {name: tensor.numpy() for name, tensor in model._w.items()}
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
