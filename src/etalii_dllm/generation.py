"""The autoregressive loop: forward pass, sample, append, repeat.

:meth:`Generator.stream` is the loop; :meth:`Generator.generate` collects it, so streamed and non-streamed output
are the same by construction. Text is released as soon as it is final: bytes of an incomplete UTF-8 character and
text that may turn out to be the start of a stop sequence are held back until the next token decides.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from etalii_dllm.grammar import TokenConstraint
from etalii_dllm.models import LanguageModel
from etalii_dllm.numerics import fingerprint, log_softmax
from etalii_dllm.sampling import Sampler, SamplingOptions
from etalii_dllm.tokenization import Tokenizer

MAX_TOP_LOGPROBS = 20


@dataclass(frozen=True)
class TokenLogprob:
    token: int
    logprob: float


@dataclass(frozen=True)
class TokenLogprobs:
    """Log-probability of a generated token under the model's unmodified distribution (temperature 1, before
    top-k/top-p and constraints), with the most likely alternatives ordered by (logprob descending, id
    ascending)."""

    token: int
    logprob: float
    top: tuple[TokenLogprob, ...] = ()


@dataclass(frozen=True)
class Step:
    """One generated token and the text it released. The last step has ``finish_reason`` set; its ``token`` is
    ``None`` when it only flushes held-back text."""

    token: int | None
    text: str
    logprobs: TokenLogprobs | None = None
    finish_reason: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    text: str
    tokens: tuple[int, ...]
    prompt_tokens: int
    finish_reason: str
    logprobs: tuple[TokenLogprobs, ...] = ()

    @property
    def fingerprint(self) -> str:
        """Hash of the generated token ids: equal fingerprints prove bit-identical output."""
        return fingerprint(self.tokens, dtype="<i4")


def _complete_prefix(data: bytes | bytearray) -> int:
    """Length of ``data`` without a trailing incomplete UTF-8 sequence."""
    for back in range(1, min(4, len(data)) + 1):
        byte = data[-back]
        if byte & 0xC0 == 0x80:
            continue
        if byte >= 0xC0:
            needed = 2 if byte < 0xE0 else 3 if byte < 0xF0 else 4
            if back < needed:
                return len(data) - back
        break
    return len(data)


def _held_back(text: str, stop: Sequence[str]) -> int:
    """How many trailing characters of ``text`` could be the start of a stop sequence."""
    longest = 0
    for sequence in stop:
        for length in range(min(len(sequence) - 1, len(text)), longest, -1):
            if text.endswith(sequence[:length]):
                longest = length
                break
    return longest


class Generation:
    """A generation in progress: iterate it for the :class:`Step` s, then :meth:`result`."""

    def __init__(
        self,
        generator: Generator,
        context: list[int],
        max_tokens: int,
        options: SamplingOptions,
        stop: Sequence[str],
        constraint: TokenConstraint | None,
        top_logprobs: int | None,
    ) -> None:
        self.prompt_tokens = len(context)
        self._tokens: list[int] = []
        self._logprobs: list[TokenLogprobs] = []
        self._text = ""
        self._finish_reason: str | None = None
        self.stop_sequence: str | None = None
        """The stop sequence that ended the generation, if any."""
        self._steps = self._run(generator, context, max_tokens, options, stop, constraint, top_logprobs)

    def __iter__(self) -> Iterator[Step]:
        return self._steps

    def result(self) -> GenerationResult:
        for _ in self._steps:
            pass
        assert self._finish_reason is not None
        return GenerationResult(
            self._text, tuple(self._tokens), self.prompt_tokens, self._finish_reason, tuple(self._logprobs)
        )

    def _run(
        self,
        generator: Generator,
        context: list[int],
        max_tokens: int,
        options: SamplingOptions,
        stop: Sequence[str],
        constraint: TokenConstraint | None,
        top_logprobs: int | None,
    ) -> Iterator[Step]:
        model, tokenizer = generator.model, generator.tokenizer
        stop = [s for s in stop if s]
        sampler = Sampler(options)
        data = bytearray()
        emitted = ""
        finish_reason = "length"
        # Models with a KV cache reuse it across steps; by construction that gives the same logits as forward().
        new_cache = getattr(model, "new_cache", None)
        cache = new_cache() if new_cache is not None else None

        while len(self._tokens) < max_tokens:
            if constraint is not None and constraint.finished:
                finish_reason = "stop"
                break
            logits = model.forward(context) if cache is None else model.forward_cached(context, cache)
            token = self._choose(sampler, logits, generator.stop_tokens, constraint)
            if token is None or token in generator.stop_tokens:
                finish_reason = "stop"
                break
            if constraint is not None:
                constraint.accept(token)
            self._tokens.append(token)
            context.append(token)
            logprobs = self._logprobs_of(logits, token, top_logprobs) if top_logprobs is not None else None
            if logprobs is not None:
                self._logprobs.append(logprobs)

            data.extend(tokenizer.decode_bytes([token]))
            text = bytes(data[: _complete_prefix(data)]).decode("utf-8", errors="replace")
            found = [(i, n) for n, s in enumerate(stop) if (i := text.find(s, max(0, len(emitted) - len(s)))) >= 0]
            if found:
                stop_at, which = min(found)
                self.stop_sequence = stop[which]
                self._text = text[:stop_at]
                self._finish_reason = "stop"
                yield Step(token, self._text[len(emitted) :], logprobs, "stop")
                return
            release = len(text) - _held_back(text, stop)
            delta = text[len(emitted) : release] if release > len(emitted) else ""
            emitted += delta
            yield Step(token, delta, logprobs)

        self._text = bytes(data).decode("utf-8", errors="replace")
        self._finish_reason = finish_reason
        yield Step(None, self._text[len(emitted) :], None, finish_reason)

    @staticmethod
    def _choose(
        sampler: Sampler, logits: np.ndarray, stop_tokens: frozenset[int], constraint: TokenConstraint | None
    ) -> int | None:
        if constraint is None or not constraint.active:
            return sampler.sample(logits)
        if sampler.greedy:
            # The argmax of the allowed tokens: when the overall argmax is allowed it is that one (same tie rule),
            # which saves computing the whole mask on most steps.
            best = sampler.sample(logits)
            if (best in stop_tokens and constraint.may_stop) or (best not in stop_tokens and constraint.allows(best)):
                return best
        # Stop tokens end the generation whatever their bytes, so they are allowed exactly when the match may end.
        allowed = [t for t in constraint.allowed() if t not in stop_tokens]
        if constraint.may_stop:
            allowed = sorted({*allowed, *(t for t in stop_tokens if 0 <= t < len(logits))})
        if not allowed:
            return None
        return sampler.sample(logits, allowed)

    @staticmethod
    def _logprobs_of(logits: np.ndarray, token: int, top: int) -> TokenLogprobs:
        values = log_softmax(logits)
        alternatives: tuple[TokenLogprob, ...] = ()
        if top > 0:
            order = np.lexsort((np.arange(len(values)), -values.astype(np.float64)))[:top]
            alternatives = tuple(TokenLogprob(int(i), float(values[i])) for i in order)
        return TokenLogprobs(token, float(values[token]), alternatives)


class Generator:
    def __init__(self, model: LanguageModel, tokenizer: Tokenizer, stop_tokens: Iterable[int] = ()) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.stop_tokens = frozenset(t for t in [tokenizer.end_of_sequence, *stop_tokens] if t >= 0)

    def stream(
        self,
        prompt: str | Sequence[int],
        max_tokens: int,
        options: SamplingOptions,
        *,
        stop: Sequence[str] = (),
        constraint: TokenConstraint | None = None,
        top_logprobs: int | None = None,
    ) -> Generation:
        """Starts a generation. ``stop`` ends it at the first occurrence of any of the strings (which are not part
        of the text); ``constraint`` restricts the tokens (structured output, tool calls); ``top_logprobs``
        (0 to 20) records each token's log-probability and that many alternatives."""
        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        if top_logprobs is not None and not 0 <= top_logprobs <= MAX_TOP_LOGPROBS:
            raise ValueError(f"top_logprobs must be between 0 and {MAX_TOP_LOGPROBS}")
        context = self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        return Generation(self, context, max_tokens, options, stop, constraint, top_logprobs)

    def generate(
        self,
        prompt: str | Sequence[int],
        max_tokens: int,
        options: SamplingOptions,
        *,
        stop: Sequence[str] = (),
        constraint: TokenConstraint | None = None,
        top_logprobs: int | None = None,
    ) -> GenerationResult:
        return self.stream(
            prompt, max_tokens, options, stop=stop, constraint=constraint, top_logprobs=top_logprobs
        ).result()
