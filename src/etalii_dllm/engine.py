"""Facade shared by the CLI, the OpenAI-compatible server and the MCP server, so all three give identical output."""

from __future__ import annotations

from collections.abc import Iterable
from functools import cache

from etalii_dllm.chat import ChatMessage, render
from etalii_dllm.generation import GenerationResult, Generator
from etalii_dllm.models import BigramModel, LanguageModel
from etalii_dllm.sampling import SamplingOptions
from etalii_dllm.tokenization import ByteTokenizer, Tokenizer

DEFAULT_MODEL_SEED = 42


class DllmEngine:
    def __init__(self, model: LanguageModel, tokenizer: Tokenizer, system_fingerprint: str) -> None:
        self.model = model
        self.system_fingerprint = system_fingerprint
        """Identifies the exact weights and engine; equal fingerprints plus equal requests give equal output."""
        self._generator = Generator(model, tokenizer)

    @staticmethod
    def create_default() -> DllmEngine:
        tokenizer = ByteTokenizer()
        model = BigramModel(tokenizer.vocabulary_size, DEFAULT_MODEL_SEED)
        return DllmEngine(model, tokenizer, "fp_" + model.weights_fingerprint[:12])

    def complete(self, prompt: str, max_tokens: int, options: SamplingOptions) -> GenerationResult:
        return self._generator.generate(prompt, max_tokens, options)

    def chat(self, messages: Iterable[ChatMessage], max_tokens: int, options: SamplingOptions) -> GenerationResult:
        return self.complete(render(messages), max_tokens, options)


@cache
def default_engine() -> DllmEngine:
    """Process-wide default engine; the model is immutable, so sharing it between requests is safe."""
    return DllmEngine.create_default()
