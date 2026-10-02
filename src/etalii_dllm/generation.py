"""The autoregressive loop: forward pass, sample, append, repeat.

:meth:`Generator.stream` is the loop; :meth:`Generator.generate` collects it, so streamed and non-streamed output
are the same by construction. Text is released as soon as it is final: bytes of an incomplete UTF-8 character and
text that may turn out to be the start of a stop sequence are held back until the next token decides.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from etalii_dllm.batching import Batcher
from etalii_dllm.grammar import HealingConstraint, TokenConstraint
from etalii_dllm.guidance import Guide
from etalii_dllm.models import LanguageModel
from etalii_dllm.numerics import fingerprint, log_softmax
from etalii_dllm.prompt_cache import CacheStore, PromptCache
from etalii_dllm.reasoning import Tracker, closing_text
from etalii_dllm.sampling import Sampler, SamplingOptions
from etalii_dllm.speculative import Drafter, DraftModel, PromptLookup
from etalii_dllm.tokenization import Tokenizer

Constraint = TokenConstraint | HealingConstraint

MAX_TOP_LOGPROBS = 20
OVERFLOWS = ("stop", "roll")
"""What happens when a generation fills the model's context window: it ends (``length``), or it rolls on."""
ROLL_SINK = 4
"""Tokens at the start of the sequence a rolled context keeps (attention sinks)."""


class ContextLengthError(ValueError):
    """The prompt does not fit in the model's context window."""


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
        constraint: Constraint | None,
        top_logprobs: int | None,
        new_text: bool = False,
        overflow: str = "stop",
        reasoning: Tracker | None = None,
        guide: Guide | None = None,
        healed: int = 0,
        stop_tokens: frozenset[int] = frozenset(),
        min_tokens: int = 0,
        ignore_eos: bool = False,
        include_stop: bool = False,
    ) -> None:
        self.prompt_tokens = len(context)
        self._stop_tokens = (frozenset() if ignore_eos else generator.stop_tokens) | stop_tokens
        """The tokens that end this generation: the generator's (unless ``ignore_eos``), and its own (a
        fill-in-the-middle's ends, ``stop_token_ids``)."""
        self._min_tokens = min_tokens
        """Until the output has this many tokens no stop token can be chosen (docs/api.md#length-and-stop-controls)."""
        self._include_stop = include_stop
        """Keep the stop sequence that ended the output in its text."""
        self._healed = healed
        """Bytes of a prompt token taken back for token healing: the output's first bytes, left out of its text."""
        self._guide = guide
        """Combines the logits with another context or model before decoding (:mod:`etalii_dllm.guidance`)."""
        self._overflow = overflow
        self.reasoning = reasoning
        """Follows a thinking model's ``<think>`` block and closes it at the budget (:mod:`etalii_dllm.reasoning`)."""
        self.rolls = 0
        """How often the context rolled (``overflow="roll"``)."""
        # Models with a KV cache reuse it across steps; by construction that gives the same logits as forward().
        # With a prompt cache the KV cache may come from an earlier request that shares a prefix with this one.
        new_cache = getattr(generator.model, "new_cache", None)
        cache = None
        self.cached_tokens = 0
        """Prompt tokens whose keys and values came from the prompt cache instead of being computed."""
        if generator.prompt_cache is not None:
            cache, self.cached_tokens = generator.prompt_cache.acquire(context)
        elif new_cache is not None:
            cache = new_cache()
        self.drafted_tokens = 0
        """Tokens speculative decoding proposed; how many were kept is ``accepted_tokens``."""
        self.accepted_tokens = 0
        self._tokens: list[int] = []
        self._logprobs: list[TokenLogprobs] = []
        self._text = ""
        self._finish_reason: str | None = None
        self.stop_sequence: str | None = None
        """The stop sequence that ended the generation, if any."""
        # A new text (a chat message) drops the leading space SentencePiece-style tokens start words with, as the
        # tokenizer's own decoder does; a completion keeps it, since it continues the prompt's text.
        # A healed output continues a prompt token, so it keeps its space too.
        self._strip_leading_space = (
            new_text and not healed and bool(getattr(generator.tokenizer, "strips_leading_space", False))
        )
        self._cache = cache
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
        constraint: Constraint | None,
        top_logprobs: int | None,
    ) -> Iterator[Step]:
        try:
            yield from self._loop(generator, context, max_tokens, options, stop, constraint, top_logprobs)
        finally:
            if generator.prompt_cache is not None and self._cache is not None:
                generator.prompt_cache.release(self._cache)

    def _loop(
        self,
        generator: Generator,
        context: list[int],
        max_tokens: int,
        options: SamplingOptions,
        stop: Sequence[str],
        constraint: Constraint | None,
        top_logprobs: int | None,
    ) -> Iterator[Step]:
        model, tokenizer = generator.model, generator.tokenizer
        cache = self._cache
        window = generator.context_length
        stop = [s for s in stop if s]
        sampler = Sampler(options, context)
        data = bytearray()
        strip_leading_space = self._strip_leading_space
        healed = self._healed
        emitted = ""
        finish_reason = "length"
        guide = self._guide
        drafter = generator.new_drafter() if cache is not None and guide is None else None
        vocabulary = model.vocabulary_size if drafter is not None else 0
        draft: list[int] = []
        """Drafted tokens not yet checked, after the ones ``pending`` holds the logits for."""
        pending: list[np.ndarray] = []
        """Logits already computed for the next positions (speculative decoding, :mod:`etalii_dllm.speculative`)."""

        tracker = self.reasoning

        def emit(token: int, logprobs: TokenLogprobs | None) -> tuple[Step, bool]:
            """Appends ``token`` to the output; the step to yield, and whether a stop sequence ended the output."""
            nonlocal strip_leading_space, emitted, healed
            if constraint is not None:
                constraint.accept(token)
            sampler.accept(token)
            self._tokens.append(token)
            context.append(token)
            if logprobs is not None:
                self._logprobs.append(logprobs)
            data.extend(tokenizer.decode_bytes([token]))
            if healed:  # the healed bytes are the prompt's, not the answer's
                dropped = min(healed, len(data))
                del data[:dropped]
                healed -= dropped
            if strip_leading_space and data:
                if data[0] == 0x20:
                    del data[0]
                strip_leading_space = False
            text = bytes(data[: _complete_prefix(data)]).decode("utf-8", errors="replace")
            if tracker is not None:
                tracker.step(text)
            found = [(i, n) for n, s in enumerate(stop) if (i := text.find(s, max(0, len(emitted) - len(s)))) >= 0]
            if found:
                stop_at, which = min(found)
                self.stop_sequence = stop[which]
                self._text = text[: stop_at + len(stop[which]) if self._include_stop else stop_at]
                self._finish_reason = "stop"
                return Step(token, self._text[len(emitted) :], logprobs, "stop"), True
            release = len(text) - _held_back(text, stop)
            delta = text[len(emitted) : release] if release > len(emitted) else ""
            emitted += delta
            return Step(token, delta, logprobs), False

        while len(self._tokens) < max_tokens:
            if constraint is not None and constraint.finished:
                finish_reason = "stop"
                break
            if guide is not None and window is not None and guide.length(self._tokens) >= window:
                break  # the guide's context is full: finish_reason stays "length"
            if window is not None and len(context) >= window:
                if self._overflow != "roll":
                    break  # finish_reason stays "length"
                cache = self._roll(generator, context, window)
                draft, pending = [], []
            if tracker is not None and tracker.over_budget:
                # The thinking budget is spent: the block is closed with fixed tokens, not sampled ones.
                text = bytes(data[: _complete_prefix(data)]).decode("utf-8", errors="replace")
                for token in tokenizer.encode(closing_text(text)):
                    if len(self._tokens) >= max_tokens or (window is not None and len(context) >= window):
                        break
                    step, stopped = emit(token, None)
                    yield step
                    if stopped:
                        return
                tracker.close()
                draft, pending = [], []
                continue
            if not pending:
                room = max_tokens - len(self._tokens) - 1
                if window is not None:
                    room = min(room, window - len(context) - 1)  # drafted tokens are fed: stay inside the window
                draft = self._draft(drafter, context, generator.speculate, room, vocabulary)
                if cache is None:
                    pending = [model.forward(context)]
                elif draft:
                    pending = list(model.forward_cached_last([*context, *draft], cache, len(draft) + 1))
                elif generator.batcher is not None:
                    pending = [generator.batcher.forward_cached(context, cache)]
                else:
                    pending = [model.forward_cached(context, cache)]
            logits = pending.pop(0)
            decoded = logits if guide is None else guide.combine(np.asarray(logits, dtype=np.float32), self._tokens)
            blocked = len(self._tokens) < self._min_tokens
            token = self._choose(sampler, decoded, self._stop_tokens, constraint, blocked)
            if token is None or token in self._stop_tokens:
                finish_reason = "stop"
                break
            if draft and draft[0] == token:
                del draft[0]  # the next pending logits follow exactly this token
                self.accepted_tokens += 1
            else:
                draft, pending = [], []
            logprobs = self._logprobs_of(logits, token, top_logprobs) if top_logprobs is not None else None
            step, stopped = emit(token, logprobs)
            yield step
            if stopped:
                return

        self._text = bytes(data).decode("utf-8", errors="replace")
        self._finish_reason = finish_reason
        yield Step(None, self._text[len(emitted) :], None, finish_reason)

    def _roll(self, generator: Generator, context: list[int], window: int) -> Any:
        """Keeps the first :data:`ROLL_SINK` tokens and the latest ``window // 2`` of ``context`` (in place) and
        returns a new, empty KV cache: the next forward pass computes the kept tokens afresh, so every token after a
        roll is exactly what a new generation over the kept tokens would choose."""
        context[:] = context[:ROLL_SINK] + context[-(window // 2) :]
        self.rolls += 1
        if self._cache is None:
            return None
        if generator.prompt_cache is not None:
            generator.prompt_cache.release(self._cache)  # still a valid cache for the tokens it holds
        self._cache = generator.model.new_cache()  # type: ignore[attr-defined]
        return self._cache

    def _draft(
        self, drafter: Drafter | None, context: list[int], speculate: int, room: int, vocabulary: int
    ) -> list[int]:
        """Up to ``speculate`` drafted tokens (fewer when only ``room`` more can be used); ``[]`` without one."""
        if drafter is None or speculate <= 0 or room <= 0:
            return []
        draft = drafter.propose(context, min(speculate, room))
        for i, token in enumerate(draft):
            if not 0 <= token < vocabulary:  # a draft model's own tokens past the model's vocabulary
                del draft[i:]
                break
        self.drafted_tokens += len(draft)
        return draft

    @staticmethod
    def _choose(
        sampler: Sampler,
        logits: np.ndarray,
        stop_tokens: frozenset[int],
        constraint: Constraint | None,
        blocked: bool = False,
    ) -> int | None:
        """The next token; ``blocked`` (an output shorter than ``min_tokens``) takes the stop tokens out of the
        choice, as a constraint restricts it."""
        if constraint is None or not constraint.active:
            if blocked and stop_tokens:
                return sampler.sample(logits, [t for t in range(len(logits)) if t not in stop_tokens])
            return sampler.sample(logits)
        if sampler.greedy:
            # The argmax of the allowed tokens: when the overall argmax is allowed it is that one (same tie rule),
            # which saves computing the whole mask on most steps.
            best = sampler.sample(logits)
            if best in stop_tokens:
                if constraint.may_stop and not blocked:
                    return best
            elif constraint.allows(best):
                return best
        # Stop tokens end the generation whatever their bytes, so they are allowed exactly when the match may end.
        allowed = [t for t in constraint.allowed() if t not in stop_tokens]
        if constraint.may_stop and not blocked:
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
    def __init__(
        self,
        model: LanguageModel,
        tokenizer: Tokenizer,
        stop_tokens: Iterable[int] = (),
        prompt_cache: int = 0,
        speculate: int = 0,
        draft_model: Any = None,
        prompt_cache_dir: str | None = None,
    ) -> None:
        """``prompt_cache`` is how many KV caches of finished generations to keep for reuse by later prompts that
        share a prefix (:mod:`etalii_dllm.prompt_cache`); 0, or a model without a KV cache, disables it. It never
        changes the output. Models that can run several sequences in one pass (``forward_batch``) decode concurrent
        generations together (:mod:`etalii_dllm.batching`), which never changes the output either. ``speculate``
        drafts that many tokens per step and checks them in one pass (:mod:`etalii_dllm.speculative`), with
        ``draft_model`` or else from the text so far; it never changes the output either. ``prompt_cache_dir`` keeps
        the prompt cache on disk across restarts (:class:`etalii_dllm.prompt_cache.CacheStore`)."""
        if speculate < 0:
            raise ValueError("speculate must be non-negative")
        if not hasattr(model, "forward_cached_last"):
            speculate = 0
        self.speculate = speculate
        self.draft_model = draft_model
        self.model = model
        self.tokenizer = tokenizer
        self.stop_tokens = frozenset(t for t in [tokenizer.end_of_sequence, *stop_tokens] if t >= 0)
        self.context_length: int | None = getattr(getattr(model, "config", None), "context_length", None) or None
        """The model's context window: prompt and answer together never hold more tokens (``None``: no limit)."""
        new_cache = getattr(model, "new_cache", None)
        self.batcher = Batcher(model) if hasattr(model, "forward_batch") else None  # type: ignore[arg-type]
        self.prompt_cache: PromptCache | None = None
        if new_cache is not None and prompt_cache > 0:
            store = None
            if prompt_cache_dir:
                from etalii_dllm import __version__

                key = f"dllm-kv/1|{__version__}|{getattr(model, 'weights_fingerprint', '')}"
                store = CacheStore(prompt_cache_dir, key)
            self.prompt_cache = PromptCache(new_cache, prompt_cache, store)

    def new_drafter(self) -> Drafter | None:
        """A drafter for one generation, or ``None`` when speculation is off."""
        if self.speculate <= 0:
            return None
        return DraftModel(self.draft_model) if self.draft_model is not None else PromptLookup()

    def stream(
        self,
        prompt: str | Sequence[int],
        max_tokens: int,
        options: SamplingOptions,
        *,
        stop: Sequence[str] = (),
        constraint: Constraint | None = None,
        top_logprobs: int | None = None,
        new_text: bool = False,
        overflow: str = "stop",
        reasoning: Tracker | None = None,
        guide: Callable[[list[int]], Guide] | None = None,
        healed: int = 0,
        stop_tokens: Iterable[int] = (),
        min_tokens: int = 0,
        ignore_eos: bool = False,
        include_stop: bool = False,
    ) -> Generation:
        """Starts a generation. ``stop`` ends it at the first occurrence of any of the strings (which are not part
        of the text); ``constraint`` restricts the tokens (structured output, tool calls); ``top_logprobs``
        (0 to 20) records each token's log-probability and that many alternatives. ``new_text`` says the output
        starts a new text (a chat message) rather than continuing the prompt, so a SentencePiece-style tokenizer's
        leading space is dropped from it. ``overflow`` says what happens when the sequence fills the model's
        context window: the generation ends with ``length`` (``stop``), or the context rolls (``roll``, see
        :meth:`Generation._roll`). A prompt that leaves no room raises :class:`ContextLengthError`. ``healed``: the
        output's first bytes are a prompt token taken back for token healing (a :class:`HealingConstraint` makes
        the output start with them), left out of its text. ``stop_tokens`` end the generation as well as the
        generator's own (a fill-in-the-middle's ends, :mod:`etalii_dllm.infill`), and ``ignore_eos`` drops the
        generator's own; until the output has ``min_tokens`` tokens no stop token can be chosen (their logits are
        -inf); ``include_stop`` keeps the stop sequence that ended the output in its text. ``reasoning``
        follows a thinking model's ``<think>`` block and closes it when its budget is spent
        (:mod:`etalii_dllm.reasoning`). ``guide`` makes the guide (:mod:`etalii_dllm.guidance`) for the prompt's
        tokens; guided generations do not roll."""
        if overflow not in OVERFLOWS:
            raise ValueError(f"overflow must be one of {', '.join(OVERFLOWS)}")
        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        if top_logprobs is not None and not 0 <= top_logprobs <= MAX_TOP_LOGPROBS:
            raise ValueError(f"top_logprobs must be between 0 and {MAX_TOP_LOGPROBS}")
        vocabulary = getattr(self.model, "vocabulary_size", None)
        if vocabulary is not None and any(token >= vocabulary for token, _ in options.logit_bias):
            raise ValueError(f"logit_bias token ids must be below the vocabulary size {vocabulary}")
        stop_tokens = frozenset(stop_tokens)
        if any(token < 0 or (vocabulary is not None and token >= vocabulary) for token in stop_tokens):
            raise ValueError(f"stop token ids must be between 0 and the vocabulary size {vocabulary}")
        if min_tokens < 0:
            raise ValueError("min_tokens must be non-negative")
        context = self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        window = self.context_length
        if window is not None and len(context) >= window:
            raise ContextLengthError(
                f"the prompt has {len(context)} tokens; the model's context window holds {window}, answer included"
            )
        if overflow == "roll" and window is not None and window <= 2 * ROLL_SINK:
            raise ValueError(f"a context window of {window} tokens is too small to roll")
        if guide is not None and overflow == "roll":
            raise ValueError("guided decoding cannot roll the context")
        made = guide(list(context)) if guide is not None else None
        return Generation(
            self,
            context,
            max_tokens,
            options,
            stop,
            constraint,
            top_logprobs,
            new_text,
            overflow,
            reasoning,
            made,
            healed,
            stop_tokens,
            min_tokens,
            ignore_eos,
            include_stop,
        )

    def generate(
        self,
        prompt: str | Sequence[int],
        max_tokens: int,
        options: SamplingOptions,
        *,
        stop: Sequence[str] = (),
        constraint: Constraint | None = None,
        top_logprobs: int | None = None,
        overflow: str = "stop",
        reasoning: Tracker | None = None,
        guide: Callable[[list[int]], Guide] | None = None,
    ) -> GenerationResult:
        return self.stream(
            prompt,
            max_tokens,
            options,
            stop=stop,
            constraint=constraint,
            top_logprobs=top_logprobs,
            overflow=overflow,
            reasoning=reasoning,
            guide=guide,
        ).result()
