"""Facade shared by the CLI, the OpenAI-compatible server and the MCP server, so all three give identical output."""

from __future__ import annotations

import os
from collections.abc import Iterable
from functools import cache
from pathlib import Path

from etalii_dllm.chat import ChatMessage, render
from etalii_dllm.chat_template import ChatTemplate
from etalii_dllm.generation import GenerationResult, Generator
from etalii_dllm.models import BigramModel, LanguageModel
from etalii_dllm.sampling import SamplingOptions
from etalii_dllm.tokenization import ByteTokenizer, Tokenizer

DEFAULT_MODEL_SEED = 42
MODEL_ENVIRONMENT_VARIABLE = "DLLM_MODEL"
"""Path of a ``model.dllm`` file for the front ends to serve; the placeholder bigram model when unset."""


class DllmEngine:
    def __init__(
        self,
        model: LanguageModel,
        tokenizer: Tokenizer,
        system_fingerprint: str,
        *,
        chat_template: ChatTemplate | None = None,
        stop_tokens: Iterable[int] = (),
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.system_fingerprint = system_fingerprint
        """Identifies the exact weights and engine; equal fingerprints plus equal requests give equal output."""
        self.chat_template = chat_template
        self._generator = Generator(model, tokenizer, stop_tokens)

    @staticmethod
    def from_model_file(path: str | Path, verify: bool = True) -> DllmEngine:
        """An engine for an imported ``model.dllm``: its decoder, BPE tokenizer and chat template."""
        from etalii_dllm.bpe import from_model_header, special_token_text
        from etalii_dllm.modelfile import ModelFile
        from etalii_dllm.transformer import Transformer

        file = ModelFile(path, verify=verify)
        if file.tokenizer is None:
            raise ValueError(f"{path}: the model file has no tokenizer")
        model_id = str(file.source.get("repository") or Path(path).stem)
        model = Transformer(file.config, file.tensors, model_id=model_id, weights_fingerprint=file.fingerprint)
        tokenizer = from_model_header(file.tokenizer)
        template = None
        if file.chat_template:
            config = file.tokenizer.get("tokenizer_config") or {}
            names = ("bos_token", "eos_token", "unk_token", "pad_token")
            template = ChatTemplate(
                file.chat_template,
                special_tokens={name: special_token_text(config.get(name)) for name in names},
            )
        stops = [*file.config.eos_token_ids, tokenizer.end_of_sequence]
        return DllmEngine(model, tokenizer, "fp_" + file.fingerprint[:12], chat_template=template, stop_tokens=stops)

    @staticmethod
    def create_default() -> DllmEngine:
        tokenizer = ByteTokenizer()
        model = BigramModel(tokenizer.vocabulary_size, DEFAULT_MODEL_SEED)
        return DllmEngine(model, tokenizer, "fp_" + model.weights_fingerprint[:12])

    def complete(self, prompt: str, max_tokens: int, options: SamplingOptions) -> GenerationResult:
        return self._generator.generate(prompt, max_tokens, options)

    def render_chat(self, messages: Iterable[ChatMessage]) -> str:
        """The prompt for a conversation: the model's own chat template when it has one."""
        if self.chat_template is None:
            return render(messages)
        return self.chat_template.render([{"role": m.role, "content": m.content} for m in messages])

    def chat(self, messages: Iterable[ChatMessage], max_tokens: int, options: SamplingOptions) -> GenerationResult:
        return self.complete(self.render_chat(messages), max_tokens, options)


def use_model_file(path: str | Path | None) -> None:
    """Makes the front ends serve ``path`` (sets ``DLLM_MODEL`` and resets the default engine)."""
    if path:
        os.environ[MODEL_ENVIRONMENT_VARIABLE] = str(path)
    default_engine.cache_clear()


@cache
def default_engine() -> DllmEngine:
    """Process-wide default engine: the model named by ``DLLM_MODEL``, else the placeholder. Models are immutable
    and ``forward`` keeps no shared state, so sharing the engine between requests is safe."""
    path = os.environ.get(MODEL_ENVIRONMENT_VARIABLE)
    return DllmEngine.from_model_file(path) if path else DllmEngine.create_default()
