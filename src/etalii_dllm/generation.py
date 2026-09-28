"""The autoregressive loop: forward pass, sample, append, repeat."""

from __future__ import annotations

from dataclasses import dataclass

from etalii_dllm.models import LanguageModel
from etalii_dllm.numerics import fingerprint
from etalii_dllm.sampling import Sampler, SamplingOptions
from etalii_dllm.tokenization import Tokenizer


@dataclass(frozen=True)
class GenerationResult:
    text: str
    tokens: tuple[int, ...]
    prompt_tokens: int
    finish_reason: str

    @property
    def fingerprint(self) -> str:
        """Hash of the generated token ids: equal fingerprints prove bit-identical output."""
        return fingerprint(self.tokens, dtype="<i4")


class Generator:
    def __init__(self, model: LanguageModel, tokenizer: Tokenizer) -> None:
        self._model = model
        self._tokenizer = tokenizer

    def generate(self, prompt: str, max_tokens: int, options: SamplingOptions) -> GenerationResult:
        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        context = self._tokenizer.encode(prompt)
        prompt_tokens = len(context)
        sampler = Sampler(options)
        generated: list[int] = []
        finish_reason = "length"

        while len(generated) < max_tokens:
            token = sampler.sample(self._model.forward(context))
            if token == self._tokenizer.end_of_sequence:
                finish_reason = "stop"
                break
            generated.append(token)
            context.append(token)

        return GenerationResult(self._tokenizer.decode(generated), tuple(generated), prompt_tokens, finish_reason)
