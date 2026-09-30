"""Imported models against the reference implementation (Hugging Face transformers and tokenizers).

Two groups of tests:

- Synthetic: tiny Llama, Qwen2 and Qwen3 checkpoints from ``model_fixtures`` loaded by transformers and our decoder.
  Needs ``torch`` and ``transformers`` only.
- Real: the pinned open-weight models of ``REFERENCE_MODELS``, found under ``$DLLM_REFERENCE_MODELS`` either as
  ``<dir>/<name>/`` or in the ``dllm import`` hub cache layout ``<dir>/<org>/<name>/<commit>/``. Tokenizer tests need
  only ``tokenizer.json``; template tests need transformers; logit and generation tests need the weights and torch.
  ``.github/workflows/reference.yml`` downloads the models and runs these tests.

Bit equality with transformers is not expected (its reductions run in a different order); logits agree to about
1e-5. Our own outputs are pinned exactly in ``golden_values.py``: the ``*_golden`` tests need only the weights (no
torch) and run on every release platform and SIMD path (``portable`` job of the workflow), because the same bits on
every machine is a guarantee (issue #99).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from golden_values import REFERENCE_MODEL_FINGERPRINTS
from model_fixtures import tiny_config, write_hf_checkpoint

from etalii_dllm import cuda
from etalii_dllm.bpe import BpeTokenizer, special_token_text
from etalii_dllm.chat_template import ChatTemplate
from etalii_dllm.engine import DllmEngine
from etalii_dllm.importing import import_model
from etalii_dllm.numerics import fingerprint
from etalii_dllm.sampling import GREEDY, SamplingOptions
from etalii_dllm.transformer import Transformer

ENVIRONMENT_VARIABLE = "DLLM_REFERENCE_MODELS"

# Largest absolute logit difference accepted against transformers in float32. Observed: about 2e-5 for SmolLM2.
LOGIT_TOLERANCE = 1e-3


@dataclass(frozen=True)
class ReferenceModel:
    repository: str
    revision: str
    licence: str
    # Gated models whose repository has no licence file: the text recorded in the import (as with --licence-file).
    licence_text: str | None = None

    @property
    def name(self) -> str:
        return self.repository.split("/")[1]


REFERENCE_MODELS = {
    "smollm2": ReferenceModel(
        "HuggingFaceTB/SmolLM2-135M-Instruct", "12fd25f77366fa6b3b4b768ec3050bf629380bac", "Apache-2.0"
    ),
    "qwen2.5": ReferenceModel("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775", "Apache-2.0"),
    "qwen2.5-1.5b": ReferenceModel(
        "Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306", "Apache-2.0"
    ),
    "qwen3": ReferenceModel("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca", "Apache-2.0"),
    # Llama 2 architecture with a SentencePiece-style tokenizer (Metaspace-like normaliser, byte fallback).
    "tinyllama": ReferenceModel(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "fe8a4ea1ffedaf415f4da2f062534de366a451e6", "Apache-2.0"
    ),
    # Post-norms and QK-norm over the whole projections.
    "olmo2": ReferenceModel(
        "allenai/OLMo-2-0425-1B-Instruct", "48d788eca847d4d7548f375ad03d3c9312f6139e", "Apache-2.0"
    ),
    # Gated (HF_TOKEN, licence accepted on the Hub). Llama 3.2: llama3 RoPE scaling, tied embeddings.
    "llama3.2": ReferenceModel("meta-llama/Llama-3.2-1B-Instruct", "main", "llama3.2"),
    # Sandwich norms, (1 + w) RMSNorm, GELU gating, a RoPE base of its own for the sliding-window layers.
    "gemma3": ReferenceModel(
        "google/gemma-3-270m-it",
        "main",
        "gemma",
        "Gemma Terms of Use: https://ai.google.dev/gemma/terms\n",
    ),
}

# Variables every render passes, to both implementations: Llama 3.x puts a date in its system prompt, and
# transformers would take today's.
TEMPLATE_VARIABLES = {"llama3.2": {"date_string": "26 Jul 2024"}}

# Extra chat template variables per model for the greedy chat: Qwen3 answers directly instead of thinking first.
CHAT_VARIABLES = {"qwen3": {"enable_thinking": False}, **TEMPLATE_VARIABLES}

TEXTS = [
    "Hello, world!",
    "The capital of France is Paris.",
    "  leading spaces,\ttabs\tand\n\nnew lines\r\n",
    "Numbers: 1234567890, 3.14159, -42, 1e-5 and 2026-09-28.",
    "def fibonacci(n):\n    return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)\n",
    "Ünïcödé: café, naïve, Straße, 日本語のテキスト, 中文, 한국어, Ελληνικά, русский",
    "Emoji 🚀🔥👍🏽 and symbols ©®™ ≤≥≠ ∑∫",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
    "I'm sure you'll see they've done it, isn't it? DON'T SHOUT.",
    "https://example.com/path?query=value&other=1#fragment",
    "",
    " ",
    "a" * 300,
]

CONVERSATIONS = [
    [{"role": "user", "content": "What is the capital of France?"}],
    [
        {"role": "system", "content": "You are a terse assistant."},
        {"role": "user", "content": "Name three colours."},
        {"role": "assistant", "content": "Red, green, blue."},
        {"role": "user", "content": "And three more?"},
    ],
]

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "Once upon a time, in a land far away, there lived a",
]

CHAT = [{"role": "user", "content": "What is the capital of France?"}]
GENERATED_TOKENS = 24
# Sampling depends on every bit of the probabilities, so it shows differences a greedy answer can hide.
SAMPLED = SamplingOptions(temperature=0.8, seed=7)


def _checkpoint(key: str) -> Path | None:
    root = os.environ.get(ENVIRONMENT_VARIABLE)
    if not root:
        return None
    model = REFERENCE_MODELS[key]
    candidates = [Path(root) / model.name, Path(root) / model.repository / model.revision]
    if model.revision == "main":  # a model being added: its only snapshot, until the commit is pinned
        candidates += sorted((Path(root) / model.repository).glob("*/"))
    for candidate in candidates:
        if (candidate / "config.json").exists():
            revision_file = candidate / "REVISION"
            pinned = model.revision != "main"
            if pinned and revision_file.exists() and model.revision not in revision_file.read_text(encoding="utf-8"):
                pytest.fail(f"{candidate} is not revision {model.revision} of {model.repository}")
            return candidate
    return None


def checkpoint(key: str, *, weights: bool = False) -> Path:
    directory = _checkpoint(key)
    if directory is None:
        pytest.skip(f"set {ENVIRONMENT_VARIABLE} to a directory holding {REFERENCE_MODELS[key].name}")
    if weights and not any(directory.glob("*.safetensors")):
        pytest.skip(f"{directory} has no weights")
    return directory


def reference_tokenizer(directory: Path) -> BpeTokenizer:
    config = json.loads((directory / "tokenizer_config.json").read_text(encoding="utf-8"))
    spec = json.loads((directory / "tokenizer.json").read_text(encoding="utf-8"))
    return BpeTokenizer(
        spec,
        end_of_sequence=special_token_text(config.get("eos_token")),
        begin_of_sequence=special_token_text(config.get("bos_token")),
    )


@pytest.fixture(scope="module", params=sorted(REFERENCE_MODELS))
def model_key(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def imported(model_key, tmp_path_factory):
    """The real model imported to model.dllm (once per module)."""
    directory = checkpoint(model_key, weights=True)
    model = REFERENCE_MODELS[model_key]
    output = tmp_path_factory.mktemp(model_key.replace(".", "_")) / "model.dllm"
    licence_file = None
    if model.licence_text is not None:
        licence_file = output.parent / "LICENCE"
        licence_file.write_text(model.licence_text, encoding="utf-8")
    result = import_model(
        directory,
        output,
        repository=model.repository,
        revision=model.revision,
        licence=model.licence,
        licence_file=licence_file,
        accept_licence=model.licence not in ("Apache-2.0", "MIT"),
    )
    return directory, result


@pytest.fixture(scope="module")
def reference(imported):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    directory, _ = imported
    model = transformers.AutoModelForCausalLM.from_pretrained(
        directory, dtype=torch.float32, attn_implementation="eager"
    )
    return model.eval()


# ---------------------------------------------------------------------------------------------------------------
# Synthetic checkpoints: the decoder matches transformers on both families


@pytest.mark.parametrize(
    "family", ["gemma2", "gemma3", "granite", "llama", "mistral", "olmo2", "phi3", "qwen2", "qwen3"]
)
def test_tiny_checkpoint_matches_transformers(family, tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = tiny_config(family)
    write_hf_checkpoint(tmp_path / "checkpoint", config)
    import_model(tmp_path / "checkpoint", tmp_path / "tiny.dllm", repository="example/tiny")
    ours = Transformer.from_file(tmp_path / "tiny.dllm")
    reference = transformers.AutoModelForCausalLM.from_pretrained(
        tmp_path / "checkpoint", dtype=torch.float32, attn_implementation="eager"
    ).eval()
    tokens = [1, 17, 42, 5, 63, 0, 9, 9, 30]
    for end in (1, 4, len(tokens)):
        with torch.no_grad():
            expected = reference(torch.tensor([tokens[:end]])).logits[0, -1].numpy()
        np.testing.assert_allclose(ours.forward(tokens[:end]), expected, rtol=0, atol=1e-5)


@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_lora_adapters_match_peft(family, tmp_path):
    """PEFT adapters import with the same math PEFT applies, and adapters trained here load in PEFT."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    from etalii_dllm import lora
    from etalii_dllm.modelfile import ModelFile

    write_hf_checkpoint(tmp_path / "checkpoint", tiny_config(family))
    import_model(tmp_path / "checkpoint", tmp_path / "base.dllm")
    tokens = [1, 17, 42, 5, 63, 0, 9, 9, 30]

    def hf_model():
        return transformers.AutoModelForCausalLM.from_pretrained(
            tmp_path / "checkpoint", dtype=torch.float32, attn_implementation="eager"
        ).eval()

    # PEFT -> us: random A and B (init_lora_weights=False), every linear layer, rsLoRA scaling.
    torch.manual_seed(0)
    settings = peft.LoraConfig(
        r=4, lora_alpha=8, init_lora_weights=False, use_rslora=True, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )  # fmt: skip
    peft_model = peft.get_peft_model(hf_model(), settings).eval()
    peft_model.save_pretrained(tmp_path / "from-peft")
    import_model(tmp_path / "from-peft", tmp_path / "merged.dllm", base=tmp_path / "base.dllm", licence="mit")
    ours = Transformer.from_file(tmp_path / "merged.dllm")
    with torch.no_grad():
        expected = peft_model(torch.tensor([tokens])).logits[0, -1].numpy()
    np.testing.assert_allclose(ours.forward(tokens), expected, rtol=0, atol=1e-5)

    # us -> PEFT: an adapter written by write_peft loads in PeftModel with the same logits as our merge.
    base = ModelFile(tmp_path / "base.dllm")
    config = lora.LoraConfig(2, 4.0, ("q", "v", "down"))
    adapters = lora.init_adapters(base.config, config, 3)
    for name in adapters:
        if name.endswith(".lora_b"):
            adapters[name] = adapters[name] + np.float32(0.05)
    lora.write_peft(tmp_path / "ours", adapters, config)
    loaded = peft.PeftModel.from_pretrained(hf_model(), tmp_path / "ours").eval()
    merged = Transformer(base.config, lora.merged_weights(base.config, base.tensors, adapters, config))
    with torch.no_grad():
        expected = loaded(torch.tensor([tokens])).logits[0, -1].numpy()
    np.testing.assert_allclose(merged.forward(tokens), expected, rtol=0, atol=1e-5)


# ---------------------------------------------------------------------------------------------------------------
# Real models: tokenizer and chat template


def test_tokenizer_matches_reference(model_key):
    tokenizers = pytest.importorskip("tokenizers")
    directory = checkpoint(model_key)
    ours = reference_tokenizer(directory)
    reference = tokenizers.Tokenizer.from_file(str(directory / "tokenizer.json"))
    for text in TEXTS:
        expected = reference.encode(text, add_special_tokens=False).ids
        assert ours.encode(text) == expected, text
        assert ours.decode(expected, skip_special_tokens=False) == reference.decode(expected, skip_special_tokens=False)


def test_chat_template_matches_reference(model_key):
    transformers = pytest.importorskip("transformers")
    directory = checkpoint(model_key)
    config = json.loads((directory / "tokenizer_config.json").read_text(encoding="utf-8"))
    source = config.get("chat_template")
    if source is None:  # newer repositories keep it in its own file
        source = (directory / "chat_template.jinja").read_text(encoding="utf-8")
    names = ("bos_token", "eos_token", "unk_token", "pad_token")
    ours = ChatTemplate(source, special_tokens={n: special_token_text(config.get(n)) for n in names})
    reference = transformers.AutoTokenizer.from_pretrained(directory)
    variables = TEMPLATE_VARIABLES.get(model_key, {})
    for conversation in CONVERSATIONS:
        for generation_prompt in (True, False):
            expected = reference.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=generation_prompt, **variables
            )
            assert ours.render(conversation, add_generation_prompt=generation_prompt, **variables) == expected
    if "tools" in source:
        expected = reference.apply_chat_template(
            CHAT, tools=[WEATHER_TOOL], tokenize=False, add_generation_prompt=True, **variables
        )
        assert ours.render(CHAT, tools=[WEATHER_TOOL], **variables) == expected


# ---------------------------------------------------------------------------------------------------------------
# Real models: weights


def test_import_fingerprint_golden(model_key, imported):
    _, result = imported
    assert result.fingerprint == REFERENCE_MODEL_FINGERPRINTS[model_key]["import"]


def _last_logits(model: Transformer, tokenizer: BpeTokenizer) -> str:
    """Fingerprint of the last-position logits of every prompt, float32 bits."""
    return fingerprint(np.concatenate([np.asarray(model.forward(tokenizer.encode(p))) for p in PROMPTS]))


@pytest.mark.parametrize("quantize", ["none", "q8_0"])
def test_logits_golden(model_key, imported, quantize):
    directory, result = imported
    ours = Transformer.from_file(result.path, quantize=None if quantize == "none" else quantize)
    key = "logits" if quantize == "none" else "logits_q8_0"
    assert _last_logits(ours, reference_tokenizer(directory)) == REFERENCE_MODEL_FINGERPRINTS[model_key].get(key)


def _chat(engine: DllmEngine, model_key: str, options: SamplingOptions):
    prompt = engine.chat_template.render(CHAT, **CHAT_VARIABLES.get(model_key, {}))
    return engine.complete(prompt, GENERATED_TOKENS, options)


def test_greedy_chat_golden(model_key, imported):
    _, result = imported
    generated = _chat(DllmEngine.from_model_file(result.path), model_key, GREEDY)
    assert generated.fingerprint == REFERENCE_MODEL_FINGERPRINTS[model_key]["chat"]


def test_sampled_chat_golden(model_key, imported):
    _, result = imported
    generated = _chat(DllmEngine.from_model_file(result.path), model_key, SAMPLED)
    assert generated.fingerprint == REFERENCE_MODEL_FINGERPRINTS[model_key].get("sampled")


def test_logits_match_reference(imported, reference):
    import torch

    directory, result = imported
    ours = Transformer.from_file(result.path)
    tokenizer = reference_tokenizer(directory)
    for prompt in PROMPTS:
        tokens = tokenizer.encode(prompt)
        with torch.no_grad():
            expected = reference(torch.tensor([tokens])).logits[0, -1].numpy()
        actual = ours.forward(tokens)
        assert np.abs(actual - expected).max() < LOGIT_TOLERANCE, prompt
        top = np.argsort(-expected, kind="stable")[:5]
        assert list(np.argsort(-actual, kind="stable")[:5]) == list(top), prompt


def test_greedy_chat_matches_reference(model_key, imported, reference):
    import torch

    _, result = imported
    engine = DllmEngine.from_model_file(result.path)
    prompt = engine.chat_template.render(CHAT, **CHAT_VARIABLES.get(model_key, {}))
    tokens = engine.tokenizer.encode(prompt)
    generated = engine.complete(prompt, GENERATED_TOKENS, GREEDY)
    with torch.no_grad():
        output = reference.generate(
            torch.tensor([tokens]), max_new_tokens=GENERATED_TOKENS, do_sample=False, pad_token_id=tokens[0]
        )
    expected = [int(t) for t in output[0, len(tokens) :]]
    stops = set(engine.model.config.eos_token_ids) | {engine.tokenizer.end_of_sequence}
    if expected and expected[-1] in stops:
        expected = expected[:-1]
    assert list(generated.tokens) == expected
    assert "Paris" in generated.text
    assert generated.fingerprint == REFERENCE_MODEL_FINGERPRINTS[model_key]["chat"]


@pytest.mark.skipif(not cuda.available(), reason="needs an NVIDIA GPU and NVRTC")
def test_greedy_chat_on_the_gpu_gives_the_golden_answer(model_key, imported):
    """Issue #31: the GPU reproduces the CPU bits, so the golden answer does not depend on the device."""
    directory, result = imported
    engine = DllmEngine.from_model_file(result.path, device="cuda")
    golden = REFERENCE_MODEL_FINGERPRINTS[model_key]
    assert _chat(engine, model_key, GREEDY).fingerprint == golden["chat"]
    assert _chat(engine, model_key, SAMPLED).fingerprint == golden["sampled"]
    gpu = Transformer.from_file(result.path, device="cuda")
    assert _last_logits(gpu, reference_tokenizer(directory)) == golden["logits"]


def test_quantized_model_stays_close_to_reference(imported, reference):
    """Phase 6: Q8_0 weights are approximate, so only the top token and the greedy answer are compared."""
    import torch

    directory, result = imported
    ours = Transformer.from_file(result.path, quantize="q8_0")
    tokenizer = reference_tokenizer(directory)
    for prompt in PROMPTS:
        tokens = tokenizer.encode(prompt)
        with torch.no_grad():
            expected = reference(torch.tensor([tokens])).logits[0, -1].numpy()
        assert int(np.argmax(ours.forward(tokens))) == int(np.argmax(expected)), prompt


def test_quantized_greedy_chat(model_key, imported):
    _, result = imported
    engine = DllmEngine.from_model_file(result.path, quantize="q8_0")
    prompt = engine.chat_template.render(CHAT, **CHAT_VARIABLES.get(model_key, {}))
    assert "Paris" in engine.complete(prompt, GENERATED_TOKENS, GREEDY).text


def main() -> None:
    """``python tests/test_reference_models.py DIR [KEY ...]`` downloads the pinned models (all, or the given keys)
    into DIR (hub cache layout)."""
    import sys
    import urllib.error

    from etalii_dllm.importing import hub

    keys = sys.argv[2:] or list(REFERENCE_MODELS)
    for model in (REFERENCE_MODELS[key] for key in keys):
        try:
            snapshot = hub.download(model.repository, model.revision, sys.argv[1])
        except urllib.error.HTTPError as error:
            if error.code not in (401, 403) or model.licence in ("Apache-2.0", "MIT"):
                raise
            # A gated model without access (no HF_TOKEN, or its licence not accepted/approved): its tests skip.
            print(f"::warning::{model.repository}: HTTP {error.code}, no access to this gated model; skipped")
            continue
        print(f"{model.repository}@{snapshot.revision}: {snapshot.directory}")


if __name__ == "__main__":
    main()
