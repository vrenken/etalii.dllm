"""Imported models against the reference implementation (Hugging Face transformers and tokenizers).

Two groups of tests:

- Synthetic: tiny Llama and Qwen2 checkpoints from ``model_fixtures`` loaded by both transformers and our decoder.
  Needs ``torch`` and ``transformers`` only.
- Real: the pinned open-weight models of ``REFERENCE_MODELS``, found under ``$DLLM_REFERENCE_MODELS`` either as
  ``<dir>/<name>/`` or in the ``dllm import`` hub cache layout ``<dir>/<org>/<name>/<commit>/``. Tokenizer tests need
  only ``tokenizer.json``; template tests need transformers; logit and generation tests need the weights and torch.
  ``.github/workflows/reference.yml`` downloads the models and runs these tests.

Bit equality with transformers is not expected (its reductions run in a different order); logits agree to about
1e-5. Our own outputs are pinned exactly in ``golden_values.py``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from golden_values import REFERENCE_MODEL_FINGERPRINTS
from model_fixtures import TINY_LLAMA_CONFIG, write_hf_checkpoint

from etalii_dllm.bpe import BpeTokenizer, special_token_text
from etalii_dllm.chat_template import ChatTemplate
from etalii_dllm.engine import DllmEngine
from etalii_dllm.importing import import_model
from etalii_dllm.sampling import GREEDY
from etalii_dllm.transformer import Transformer

ENVIRONMENT_VARIABLE = "DLLM_REFERENCE_MODELS"

# Largest absolute logit difference accepted against transformers in float32. Observed: about 2e-5 for SmolLM2.
LOGIT_TOLERANCE = 1e-3


@dataclass(frozen=True)
class ReferenceModel:
    repository: str
    revision: str
    licence: str

    @property
    def name(self) -> str:
        return self.repository.split("/")[1]


REFERENCE_MODELS = {
    "smollm2": ReferenceModel(
        "HuggingFaceTB/SmolLM2-135M-Instruct", "12fd25f77366fa6b3b4b768ec3050bf629380bac", "Apache-2.0"
    ),
    "qwen2.5": ReferenceModel("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775", "Apache-2.0"),
}

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


def _checkpoint(key: str) -> Path | None:
    root = os.environ.get(ENVIRONMENT_VARIABLE)
    if not root:
        return None
    model = REFERENCE_MODELS[key]
    for candidate in (Path(root) / model.name, Path(root) / model.repository / model.revision):
        if (candidate / "config.json").exists():
            revision_file = candidate / "REVISION"
            if revision_file.exists() and model.revision not in revision_file.read_text(encoding="utf-8"):
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
    result = import_model(
        directory, output, repository=model.repository, revision=model.revision, licence=model.licence
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


@pytest.mark.parametrize("family", ["llama", "qwen2"])
def test_tiny_checkpoint_matches_transformers(family, tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = {**TINY_LLAMA_CONFIG, "model_type": family}
    if family == "qwen2":
        config.update(architectures=["Qwen2ForCausalLM"], use_sliding_window=False, tie_word_embeddings=False)
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
    names = ("bos_token", "eos_token", "unk_token", "pad_token")
    ours = ChatTemplate(config["chat_template"], special_tokens={n: special_token_text(config.get(n)) for n in names})
    reference = transformers.AutoTokenizer.from_pretrained(directory)
    for conversation in CONVERSATIONS:
        for generation_prompt in (True, False):
            expected = reference.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=generation_prompt
            )
            assert ours.render(conversation, add_generation_prompt=generation_prompt) == expected
    if "tools" in config["chat_template"]:
        expected = reference.apply_chat_template(CHAT, tools=[WEATHER_TOOL], tokenize=False, add_generation_prompt=True)
        assert ours.render(CHAT, tools=[WEATHER_TOOL]) == expected


# ---------------------------------------------------------------------------------------------------------------
# Real models: weights


def test_import_fingerprint(model_key, imported):
    _, result = imported
    assert result.fingerprint == REFERENCE_MODEL_FINGERPRINTS[model_key]["import"]


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
    prompt = engine.chat_template.render(CHAT)
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


def test_quantized_greedy_chat(imported):
    _, result = imported
    engine = DllmEngine.from_model_file(result.path, quantize="q8_0")
    assert "Paris" in engine.complete(engine.chat_template.render(CHAT), GENERATED_TOKENS, GREEDY).text


def main() -> None:
    """``python tests/test_reference_models.py DIR`` downloads the pinned models into DIR (hub cache layout)."""
    import sys

    from etalii_dllm.importing import hub

    for model in REFERENCE_MODELS.values():
        snapshot = hub.download(model.repository, model.revision, sys.argv[1])
        print(f"{model.repository}@{snapshot.revision}: {snapshot.directory}")


if __name__ == "__main__":
    main()
