"""Phase 41: exact long-context RoPE scaling. YaRN from Hugging Face and GGUF models (#267), ``dllm import
--context-length`` with YaRN or LongRoPE's long factors (#268), both in the reference implementation and in training
(#269), with golden values (#270)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from golden_values import LONG_CONTEXT_FINGERPRINTS
from model_fixtures import TINY_LLAMA_CONFIG, tiny_config, write_gguf, write_hf_checkpoint
from test_cli import isolated_environment  # noqa: F401 - autouse fixture: the CLI tests set DLLM_MODEL
from test_training import TARGETS, TOKENS, reference_loss
from test_transformer import PROMPT, reference_logits, yarn_inv_freq

from etalii_dllm import numerics, reference
from etalii_dllm.importing import ModelImportError, import_model
from etalii_dllm.importing.importer import hf_config, with_context_length, yarn_scaling
from etalii_dllm.modelfile import ModelFile
from etalii_dllm.training import DecoderGradients
from etalii_dllm.transformer import Transformer

QWEN25_YARN = {"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}


# YaRN frequencies


@pytest.mark.parametrize(
    "scaling",
    [
        {"factor": 4.0, "original_max_position_embeddings": 32768},
        {"factor": 2.5, "original_max_position_embeddings": 4096, "beta_fast": 16.0, "beta_slow": 2.0},
        {"factor": 8.0, "original_max_position_embeddings": 2048, "truncate": False},
        {"factor": 4.0, "original_max_position_embeddings": 64},  # a ramp that starts at the first pair
    ],
)
def test_yarn_frequencies_follow_transformers(scaling):
    scaling = {"rope_type": "yarn", **scaling}
    ours = numerics.rope_inv_freq(128, 1_000_000.0, scaling=scaling)
    plain = 1_000_000.0 ** (-np.arange(0, 128, 2, dtype=np.float64) / 128)
    np.testing.assert_allclose(ours, yarn_inv_freq(plain, 128, 1_000_000.0, scaling), rtol=1e-13, atol=0)
    assert ours.tobytes() == reference.rope_inv_freq(128, 1_000_000.0, scaling=scaling).tobytes()
    assert ours[0] == numerics.rope_inv_freq(128, 1_000_000.0)[0]  # the fastest pair extrapolates
    assert ours[-1] == numerics.rope_inv_freq(128, 1_000_000.0)[-1] / scaling["factor"]  # the slowest interpolates


def test_yarn_ramp_never_divides_by_zero():
    scaling = {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 1, "truncate": True}
    freqs = numerics.rope_inv_freq(8, 10000.0, scaling=scaling)
    assert np.all(np.isfinite(freqs))


# Import


def test_yarn_imports_from_a_hugging_face_config():
    config = hf_config({**tiny_config("qwen2"), "max_position_embeddings": 32768, "rope_scaling": QWEN25_YARN})
    assert config.context_length == 4 * 32768
    assert config.rope_scaling == yarn_scaling(4.0, 32768)
    assert config.rope_attention_factor == 0.1 * float(numerics.log(4.0)) + 1.0
    assert abs(config.rope_attention_factor - (0.1 * math.log(4.0) + 1.0)) < 1e-15
    deepseek = {**QWEN25_YARN, "mscale": 0.707, "mscale_all_dim": 0.707, "beta_fast": 16, "truncate": False}
    config = hf_config({**tiny_config("qwen2"), "rope_scaling": deepseek})
    assert config.rope_attention_factor == 1.0 and config.rope_scaling["beta_fast"] == 16.0
    given = hf_config({**tiny_config("qwen2"), "rope_scaling": {**QWEN25_YARN, "attention_factor": 1.5}})
    assert given.rope_attention_factor == 1.5
    v5 = {k: v for k, v in tiny_config("qwen3").items() if k != "rope_scaling"}
    v5["rope_parameters"] = {"rope_theta": 1e6, "rope_type": "yarn", "factor": 2.0}  # original from the config
    config = hf_config(v5)
    assert config.rope_scaling["original_max_position_embeddings"] == 2048 and config.context_length == 4096
    with pytest.raises(ModelImportError, match="YaRN needs"):
        hf_config({**tiny_config("qwen2"), "rope_scaling": {"type": "yarn"}})
    with pytest.raises(ModelImportError, match="YaRN needs a positive factor"):
        yarn_scaling(0.0, 10)


def test_yarn_imports_from_gguf_and_exports_back(tmp_path):
    pytest.importorskip("gguf")
    from etalii_dllm import exporting

    write_gguf(tmp_path / "plain.gguf")
    import_model(tmp_path / "plain.gguf", tmp_path / "plain.dllm", context_length=8192)
    extended = ModelFile(tmp_path / "plain.dllm")
    assert extended.config.rope_scaling == yarn_scaling(4.0, 2048) and extended.config.context_length == 8192
    exporting.export_model(tmp_path / "plain.dllm", tmp_path / "again.gguf", "gguf")
    import_model(tmp_path / "again.gguf", tmp_path / "again.dllm")
    assert ModelFile(tmp_path / "again.dllm").config == extended.config
    exporting.export_model(tmp_path / "plain.dllm", tmp_path / "hf", "safetensors")
    import_model(tmp_path / "hf", tmp_path / "hf.dllm", licence="apache-2.0")
    assert ModelFile(tmp_path / "hf.dllm").config == extended.config
    odd = ModelFile(tmp_path / "plain.dllm")
    import dataclasses

    for scaling in ({**yarn_scaling(4.0, 2048), "beta_fast": 8.0}, yarn_scaling(3.1, 2048)):
        changed = dataclasses.replace(odd.config, rope_scaling=scaling)
        object.__setattr__(odd, "config", changed)
        with pytest.raises(exporting.ExportError, match="cannot be written to GGUF exactly"):
            exporting.export_gguf(odd, tmp_path / "odd.gguf")


# --context-length


def test_context_length_extends_with_yarn_or_long_factors():
    plain = hf_config(tiny_config("qwen2"))
    longer = hf_config(tiny_config("qwen2"), context_length=8192)
    assert longer.context_length == 8192 and longer.rope_scaling == yarn_scaling(4.0, 2048)
    shorter = hf_config(tiny_config("qwen2"), context_length=512)
    assert shorter.context_length == 512 and shorter.rope_scaling is None
    assert with_context_length(plain, 8192) == longer
    phi = hf_config(tiny_config("phi3"))
    assert phi.context_length == 64 and "factor_set" not in phi.rope_scaling
    long = hf_config(tiny_config("phi3"), context_length=1024)
    assert long.context_length == 1024 and long.rope_scaling["factor_set"] == "long"
    assert with_context_length(long, 32).rope_scaling == phi.rope_scaling  # back to the short factors
    assert long.rope_attention_factor == phi.rope_attention_factor
    with pytest.raises(ModelImportError, match="trained for at most 2048"):
        hf_config(tiny_config("phi3"), context_length=4096)
    with pytest.raises(ModelImportError, match="already scales its RoPE"):
        hf_config(tiny_config("gemma3"), context_length=1 << 20)
    with pytest.raises(ModelImportError, match="positive"):
        hf_config(tiny_config("qwen2"), context_length=0)
    yarn = hf_config({**tiny_config("qwen2"), "rope_scaling": QWEN25_YARN, "max_position_embeddings": 32768})
    with pytest.raises(ModelImportError, match="already scales its RoPE \\(yarn\\)"):
        with_context_length(yarn, 1 << 20)


def test_context_length_is_refused_for_adapters(tmp_path):
    (tmp_path / "adapter").mkdir()
    (tmp_path / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ModelImportError, match="not LoRA adapters"):
        import_model(tmp_path / "adapter", tmp_path / "out.dllm", context_length=4096)


def test_cli_context_length(tmp_path, capsys):
    from etalii_dllm.cli import main

    write_hf_checkpoint(tmp_path / "phi", tiny_config("phi3"))
    assert main(["import", str(tmp_path / "phi"), "-o", str(tmp_path / "phi.dllm"), "--context-length", "512"]) == 0
    assert "context:            512 tokens (LongRoPE, long factors)" in capsys.readouterr().out
    write_hf_checkpoint(tmp_path / "qwen", tiny_config("qwen2"))
    assert main(["import", str(tmp_path / "qwen"), "-o", str(tmp_path / "q.dllm"), "--context-length", "4096"]) == 0
    assert "context:            4096 tokens (YaRN, factor 2)" in capsys.readouterr().out
    write_hf_checkpoint(tmp_path / "gemma", tiny_config("gemma3"))
    assert main(["import", str(tmp_path / "gemma"), "-o", str(tmp_path / "g.dllm")]) == 0
    assert "(linear RoPE scaling)" in capsys.readouterr().out
    assert main(["import", str(tmp_path / "qwen"), "-o", str(tmp_path / "x.dllm"), "--context-length", "0"]) == 1
    assert "positive" in capsys.readouterr().err


# Decoding, the reference implementation and training on extended models


LONG_MODELS = {
    "qwen3-yarn": ("qwen3", {"rope_scaling": {**QWEN25_YARN, "original_max_position_embeddings": 4}}, None),
    "olmo2-yarn": ("olmo2", {}, 4096),
    "llama-yarn": ("llama", {}, 8192),
    "phi3-long": ("phi3", {}, 1024),
}


@pytest.fixture(scope="module", params=sorted(LONG_MODELS))
def long_model(request, tmp_path_factory) -> tuple[str, ModelFile]:
    family, change, context = LONG_MODELS[request.param]
    directory = tmp_path_factory.mktemp(request.param)
    write_hf_checkpoint(directory / "checkpoint", {**tiny_config(family), **change})
    import_model(directory / "checkpoint", directory / "model.dllm", context_length=context)
    return request.param, ModelFile(directory / "model.dllm")


def test_extended_models_match_the_float64_reference_and_golden_logits(long_model):
    name, model_file = long_model
    assert model_file.config.rope_attention_factor != 1.0
    model = Transformer(model_file.config, model_file.tensors)
    tensors = {k: np.asarray(v) for k, v in model_file.tensors.items()}
    for length in (1, 4, len(PROMPT)):
        ours = model.forward(PROMPT[:length])
        expected = reference_logits(model.config, tensors, PROMPT[:length])
        np.testing.assert_allclose(ours, expected, rtol=0, atol=2e-5 * np.abs(expected).max())
    assert numerics.fingerprint(model.forward(PROMPT)) == LONG_CONTEXT_FINGERPRINTS[name]
    twin = reference.ReferenceTransformer.from_engine_model(model)
    assert twin.forward(PROMPT).tobytes() == model.forward(PROMPT).tobytes()


def test_extended_models_fine_tune(long_model):
    _, model_file = long_model
    gradients = DecoderGradients(model_file.config)
    decoder = Transformer(model_file.config, model_file.tensors)
    assert gradients.logits(model_file.tensors, TOKENS)[-1].tobytes() == decoder.forward(TOKENS).tobytes()
    loss, grads = gradients.loss_and_gradients(model_file.tensors, TOKENS, TARGETS)
    w64 = {name: np.asarray(values, dtype=np.float64) for name, values in model_file.tensors.items()}
    assert abs(loss - reference_loss(model_file.config, w64, TOKENS, TARGETS)) < 1e-4
    names = [n for n in sorted(grads) if n.endswith(("q.weight", "q_norm.weight", "k_norm.weight"))]
    for name in names:
        index = np.unravel_index(int(np.argmax(np.abs(grads[name]))), grads[name].shape)
        eps = 1e-6

        def at(delta, name=name, index=index):
            changed = dict(w64)
            changed[name] = w64[name].copy()
            changed[name][index] += delta
            return reference_loss(model_file.config, changed, TOKENS, TARGETS)

        numeric = (at(eps) - at(-eps)) / (2 * eps)
        assert abs(float(grads[name][index]) - numeric) < 2e-4 * max(1.0, abs(numeric)), name


def test_tiny_llama_has_no_scaling_by_default():
    assert hf_config(TINY_LLAMA_CONFIG).rope_scaling is None
