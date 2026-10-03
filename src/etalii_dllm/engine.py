"""Facade shared by the CLI, the OpenAI- and Anthropic-compatible server and the MCP server, so all of them give
identical output.

:meth:`DllmEngine.chat_stream` is the one chat code path: it renders the prompt (with tools), sets up constrained
decoding (structured output, tool calls), runs the generator and turns its steps into :class:`ChatEvent` s.
:meth:`DllmEngine.chat_completion` collects the same events, so streamed and non-streamed answers are identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from etalii_dllm import __version__, guidance, receipts
from etalii_dllm import reasoning as thinking_rules
from etalii_dllm import tools as tooling
from etalii_dllm.chat import TOOL_CALL_OPEN, ChatMessage, ToolCall, render
from etalii_dllm.chat_template import ChatTemplate
from etalii_dllm.cuda import DEVICES
from etalii_dllm.generation import OVERFLOWS, Generation, GenerationResult, Generator, TokenLogprobs
from etalii_dllm.grammar import Grammar, HealingConstraint, TokenConstraint, TokenTrie
from etalii_dllm.guidance import Guide
from etalii_dllm.infill import fim_tokens
from etalii_dllm.models import BigramModel, LanguageModel
from etalii_dllm.numerics import QUANTIZATIONS, linear, set_threads, sum_squares
from etalii_dllm.prompt_cache import DEFAULT_PROMPT_CACHE_SIZE
from etalii_dllm.retrieval import DEFAULT_TOP, MODES, Retriever
from etalii_dllm.sampling import GREEDY, SamplingOptions
from etalii_dllm.serving import Auditor, Inflight, ResponseCache, SharedGeneration, response_key
from etalii_dllm.signing import Signer
from etalii_dllm.speculative import DEFAULT_DRAFT_TOKENS
from etalii_dllm.tokenization import ByteTokenizer, Tokenizer
from etalii_dllm.tools import AUTO, Tool, ToolChoice, ToolFormat

DEFAULT_MODEL_SEED = 42
MAX_CHOICES = 16
"""Most choices one request may ask for (``n``)."""
TRUNCATIONS = ("disabled", "auto")
MODEL_ENVIRONMENT_VARIABLE = "DLLM_MODEL"
"""Path of a ``model.dllm`` file for the front ends to serve; the placeholder bigram model when unset."""
ADAPTER_ENVIRONMENT_VARIABLE = "DLLM_ADAPTER"
"""A PEFT LoRA adapter directory merged into ``DLLM_MODEL`` when it is loaded; none when unset."""
QUANTIZE_ENVIRONMENT_VARIABLE = "DLLM_QUANTIZE"
"""Weight quantisation for the served model (``q8_0``); float32 weights when unset or ``none``."""
DEVICE_ENVIRONMENT_VARIABLE = "DLLM_DEVICE"
"""Where the served model runs: ``cpu`` (default) or ``cuda``. Never changes the output."""
PROMPT_CACHE_ENVIRONMENT_VARIABLE = "DLLM_PROMPT_CACHE"
"""How many KV caches the served model keeps for prompt caching (default 4, 0 disables). Never changes the output."""
PROMPT_CACHE_DIR_ENVIRONMENT_VARIABLE = "DLLM_PROMPT_CACHE_DIR"
"""A directory that keeps the prompt cache across restarts; in memory only when unset. Never changes the output."""
STEER_ENVIRONMENT_VARIABLE = "DLLM_STEER"
"""A steering vector file (``dllm steer``) added to the model's residual stream; changes the output."""
STEER_STRENGTH_ENVIRONMENT_VARIABLE = "DLLM_STEER_STRENGTH"
"""Overrides the steering vector file's strength."""
INDEX_ENVIRONMENT_VARIABLE = "DLLM_INDEX"
"""A document index (``dllm index build``) that grounds every chat in the passages it finds; changes the output."""
INDEX_TOP_ENVIRONMENT_VARIABLE = "DLLM_INDEX_TOP"
"""How many passages grounding adds (default 3)."""
EMBEDDING_MODEL_ENVIRONMENT_VARIABLE = "DLLM_EMBEDDING_MODEL"
"""The index's embedding model, when it is not at the path the index records."""
INDEX_MODE_ENVIRONMENT_VARIABLE = "DLLM_INDEX_MODE"
"""How grounding searches the index: ``dense`` (default), ``lexical`` or ``hybrid``; changes the output."""
RERANK_MODEL_ENVIRONMENT_VARIABLE = "DLLM_RERANK_MODEL"
"""A chat model that reranks the passages grounding finds (``--rerank-model``); changes the output."""
CONTRAST_MODEL_ENVIRONMENT_VARIABLE = "DLLM_CONTRAST_MODEL"
"""The amateur model for contrastive decoding (``--contrast-model``)."""
ENSEMBLE_MODELS_ENVIRONMENT_VARIABLE = "DLLM_ENSEMBLE_MODELS"
"""Ensemble members, ``PATH[=WEIGHT]`` separated by ``os.pathsep`` (``--ensemble-model``)."""
ENSEMBLE_WEIGHT_ENVIRONMENT_VARIABLE = "DLLM_ENSEMBLE_WEIGHT"
"""The served model's weight in an ensemble (``--ensemble-weight``)."""
SPECULATE_ENVIRONMENT_VARIABLE = "DLLM_SPECULATE"
"""Tokens speculative decoding drafts per step (0 or unset: off). Never changes the output."""
DRAFT_MODEL_ENVIRONMENT_VARIABLE = "DLLM_DRAFT_MODEL"
SIGN_KEY_ENVIRONMENT_VARIABLE = "DLLM_SIGN_KEY"
"""Ed25519 private key that signs every receipt (``--sign-key``)."""
RESPONSE_CACHE_ENVIRONMENT_VARIABLE = "DLLM_RESPONSE_CACHE"
"""Directory of the exact response cache (``--response-cache``)."""
AUDIT_EVERY_ENVIRONMENT_VARIABLE = "DLLM_AUDIT_EVERY"
"""Re-run every Nth response to check it reproduces (``--audit-every``)."""
"""A smaller ``model.dllm`` with the same tokenizer that drafts for speculative decoding. Never changes the output."""


@dataclass(frozen=True)
class ResponseFormat:
    """``text`` (free), ``json_object`` (any JSON object), ``json_schema`` (a value valid under ``schema``),
    ``regex`` (text matching the regular expression ``pattern`` in full) or ``grammar`` (text the GBNF grammar
    ``pattern`` derives, :mod:`etalii_dllm.gbnf`)."""

    type: str = "text"
    schema: Mapping[str, Any] | None = None
    pattern: str | None = None

    def __post_init__(self) -> None:
        if self.type not in ("text", "json_object", "json_schema", "regex", "grammar"):
            raise ValueError(f"unknown response format {self.type!r}")
        if (self.type == "json_schema") != (self.schema is not None):
            raise ValueError("a json_schema response format needs a schema")
        if (self.type in ("regex", "grammar")) != (self.pattern is not None):
            if self.pattern is None:
                raise ValueError(f"a {self.type} response format needs a pattern")
            raise ValueError("only regex and grammar response formats take a pattern")

    def grammar(self) -> Grammar | None:
        if self.type == "json_object":
            return Grammar.json_object()
        if self.type == "json_schema":
            assert self.schema is not None
            return Grammar.json_schema(self.schema)
        if self.type == "regex":
            assert self.pattern is not None
            return Grammar.regex(self.pattern)
        if self.type == "grammar":
            assert self.pattern is not None
            return Grammar.gbnf(self.pattern)
        return None

    def record(self) -> dict[str, Any]:
        """As JSON for receipts and cache keys; ``pattern`` only when set, so older records keep their bytes."""
        record: dict[str, Any] = {"type": self.type, "schema": self.schema}
        if self.pattern is not None:
            record["pattern"] = self.pattern
        return record


TEXT = ResponseFormat()


@dataclass(frozen=True)
class ChatRequest:
    messages: Sequence[ChatMessage]
    max_tokens: int
    options: SamplingOptions = GREEDY
    stop: Sequence[str] = ()
    tools: Sequence[Tool] = ()
    tool_choice: ToolChoice = AUTO
    response_format: ResponseFormat = TEXT
    top_logprobs: int | None = None
    """``None``: no logprobs; 0 to 20: each token's logprob and that many alternatives."""
    call_id_prefix: str = "call_"
    request_id: str = ""
    """Seed for the tool call ids (see :meth:`DllmEngine.derive_id`)."""
    prompt: str | None = None
    """A raw prompt used as is instead of rendering ``messages`` (Ollama's ``raw`` mode); no tools then."""
    previous_receipt: str | None = None
    """The receipt id of the conversation's previous turn: recorded as the receipt's ``previous`` (a receipt chain,
    :func:`etalii_dllm.receipts.verify_chain`); it never changes the output."""
    truncation: str = "disabled"
    """``auto`` drops the oldest messages of a conversation too long for the context window
    (:func:`etalii_dllm.engine.fit_messages`); ``disabled`` refuses it."""
    context_overflow: str = "stop"
    """``stop`` ends an answer that fills the context window (``length``); ``roll`` keeps going on a rolled
    context (:meth:`etalii_dllm.generation.Generation._roll`)."""

    thinking: bool | None = None
    """For thinking models: ``True``/``False`` switches the ``<think>`` block on or off through the chat template
    (``enable_thinking``); ``None`` keeps the model's default (docs/api.md#reasoning)."""
    max_reasoning_tokens: int | None = None
    """For thinking models: the most tokens the ``<think>`` block may take before the engine closes it."""
    token_healing: bool = False
    """Takes the prompt's last token back and makes the answer start with its bytes (:meth:`DllmEngine.heal`), so
    a prompt or prefill that ends inside a word is continued as if the word were whole (docs/api.md#token-healing)."""
    suffix: str | None = None
    """With a raw ``prompt``: the text after the gap, so the answer is the middle the model fills in between
    (:mod:`etalii_dllm.infill`, docs/api.md#fill-in-the-middle)."""
    min_tokens: int = 0
    """No stop token can end the answer before it has this many tokens (docs/api.md#length-and-stop-controls)."""
    ignore_eos: bool = False
    """The model's own stop tokens do not end the answer (only ``stop_token_ids``, stop sequences and limits)."""
    stop_token_ids: Sequence[int] = ()
    """Token ids that end the answer as well as the model's stop tokens."""
    include_stop: bool = False
    """Keep the stop sequence that ended the answer in its text (``include_stop_str_in_output``)."""

    def __post_init__(self) -> None:
        if self.max_reasoning_tokens is not None and self.max_reasoning_tokens < 0:
            raise ValueError("max_reasoning_tokens must be non-negative")
        if self.truncation not in TRUNCATIONS:
            raise ValueError(f"truncation must be one of {', '.join(TRUNCATIONS)}")
        if self.context_overflow not in OVERFLOWS:
            raise ValueError(f"context_overflow must be one of {', '.join(OVERFLOWS)}")


@dataclass(frozen=True)
class TextDelta:
    text: str
    logprobs: tuple[TokenLogprobs, ...] = ()


@dataclass(frozen=True)
class ReasoningDelta:
    """Text of a thinking model's ``<think>`` block (without the tags), separate from the answer."""

    text: str
    logprobs: tuple[TokenLogprobs, ...] = ()


@dataclass(frozen=True)
class ToolCallEvent:
    index: int
    call: ToolCall


@dataclass(frozen=True)
class Finished:
    finish_reason: str
    """``stop`` (end of turn or a stop sequence), ``length`` or ``tool_calls``."""
    stop_sequence: str | None
    completion_tokens: int
    fingerprint: str
    """Hash of the generated token ids."""
    receipt: Mapping[str, Any] | None = None
    """What anyone needs to check this response later (:mod:`etalii_dllm.receipts`)."""
    reasoning_tokens: int = 0
    """Tokens of the ``<think>`` block (part of ``completion_tokens``)."""


ChatEvent = TextDelta | ReasoningDelta | ToolCallEvent | Finished


@dataclass(frozen=True)
class ChatResult:
    content: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str
    stop_sequence: str | None
    prompt_tokens: int
    completion_tokens: int
    fingerprint: str
    logprobs: tuple[TokenLogprobs, ...] = ()
    cached_tokens: int = 0
    """Prompt tokens served from the prompt cache; depends on earlier requests, never changes the output."""
    receipt: Mapping[str, Any] | None = field(default=None, compare=False)
    """What anyone needs to check this response later (:mod:`etalii_dllm.receipts`)."""
    reasoning: str | None = None
    """A thinking model's ``<think>`` block, without the tags; ``None`` when it did not think."""
    reasoning_tokens: int = 0


@dataclass(frozen=True)
class _ChatGeneration:
    prompt_tokens: int
    cached_tokens: int
    events: Iterator[ChatEvent]

    def __iter__(self) -> Iterator[ChatEvent]:
        return self.events


@dataclass
class ChatStream:
    """A chat generation in progress: ``prompt_tokens`` (and how many of them the prompt cache holds,
    ``cached_tokens``) are known up front; iterate for the events."""

    prompt_tokens: int
    events: Iterator[ChatEvent] = field(repr=False)
    cached_tokens: int = 0

    def __iter__(self) -> Iterator[ChatEvent]:
        return self.events


@dataclass(frozen=True)
class Embedding:
    vector: np.ndarray
    tokens: int


class DllmEngine:
    signer: Signer | None = None
    """Signs every receipt this engine makes (``--sign-key``, :mod:`etalii_dllm.signing`)."""
    response_cache: ResponseCache | None = None
    """Answers repeated requests from storage, with the same bits (``--response-cache``, :mod:`etalii_dllm.serving`)."""
    auditor: Auditor | None = None
    """Re-runs a share of the responses to check they reproduce (``--audit-every``, :mod:`etalii_dllm.serving`)."""

    def __init__(
        self,
        model: LanguageModel,
        tokenizer: Tokenizer,
        system_fingerprint: str,
        *,
        chat_template: ChatTemplate | None = None,
        stop_tokens: Iterable[int] = (),
        prompt_cache: int = 0,
        embedding: Mapping[str, Any] | None = None,
        retriever: Retriever | None = None,
        speculate: int = 0,
        draft_model: LanguageModel | None = None,
        prompt_cache_dir: str | Path | None = None,
        contrast_model: LanguageModel | None = None,
        ensemble: Sequence[tuple[LanguageModel, float]] = (),
        ensemble_weight: float = 1.0,
    ) -> None:
        """``prompt_cache`` keeps that many KV caches to reuse for prompts sharing a prefix with an earlier one
        (:mod:`etalii_dllm.prompt_cache`), on disk across restarts with ``prompt_cache_dir``; it saves work and never
        changes the output. ``speculate`` drafts that many
        tokens per step, with ``draft_model`` or from the text so far, and checks them in one pass
        (:mod:`etalii_dllm.speculative`); that saves work and never changes the output either. ``embedding`` holds an
        embedding model's pooling settings (:attr:`etalii_dllm.modelfile.ModelFile.embedding`). ``retriever`` grounds
        chats in a document index (:class:`etalii_dllm.retrieval.Retriever`); its fingerprint joins the
        ``system_fingerprint``. ``contrast_model`` is the amateur requests with ``contrast_beta`` decode against,
        and ``ensemble`` the other models (with weights; the served model has ``ensemble_weight``) every request is
        decoded with (:mod:`etalii_dllm.guidance`); their weights join the ``system_fingerprint``."""
        self.model = model
        self.contrast_model = contrast_model
        self.ensemble = list(ensemble)
        self.ensemble_weight = float(ensemble_weight)
        for other in [*([contrast_model] if contrast_model is not None else []), *(m for m, _ in self.ensemble)]:
            if getattr(other, "vocabulary_size", None) != getattr(model, "vocabulary_size", None):
                raise ValueError("a contrast or ensemble model needs the model's vocabulary")
        if any(not (math.isfinite(w) and w > 0) for w in [self.ensemble_weight, *(w for _, w in self.ensemble)]):
            raise ValueError("ensemble weights must be positive")
        extras = []
        if contrast_model is not None:
            extras.append(f"contrast:{contrast_model.weights_fingerprint}")
        if self.ensemble:
            members = ",".join(f"{m.weights_fingerprint}*{w!r}" for m, w in self.ensemble)
            extras.append(f"ensemble:{self.ensemble_weight!r}:{members}")
        if extras:
            joined = hashlib.sha256("|".join([system_fingerprint, *extras]).encode()).hexdigest()
            system_fingerprint = "fp_" + joined[:12]
        self.embedding = dict(embedding) if embedding else None
        self.retriever = retriever
        if retriever is not None:
            joined = hashlib.sha256(f"{system_fingerprint}|{retriever.fingerprint}".encode()).hexdigest()
            system_fingerprint = "fp_" + joined[:12]
        self._base_tokenizer = tokenizer
        self.tokenizer = tokenizer
        self.system_fingerprint = system_fingerprint
        """Identifies the exact weights and engine; equal fingerprints plus equal requests give equal output."""
        self._trie: TokenTrie | None = None
        self._tool_format = tooling.HERMES
        self.chat_template = chat_template
        cache_dir = str(prompt_cache_dir) if prompt_cache_dir else None
        self._generator = Generator(model, self.tokenizer, stop_tokens, prompt_cache, speculate, draft_model, cache_dir)
        self.inflight = Inflight()
        """Generations still being read: identical requests share one (:mod:`etalii_dllm.serving`)."""

    @property
    def chat_template(self) -> ChatTemplate | None:
        """The model's own chat template; setting it also sets :attr:`tool_format` to the template's format."""
        return self._chat_template

    @chat_template.setter
    def chat_template(self, template: ChatTemplate | None) -> None:
        self._chat_template = template
        self.tool_format = tooling.detect_format(template.source if template is not None else None)

    @property
    def tool_format(self) -> ToolFormat:
        """How the model writes tool calls (:mod:`etalii_dllm.tools`), detected from its chat template."""
        return self._tool_format

    @tool_format.setter
    def tool_format(self, fmt: ToolFormat | str) -> None:
        fmt = tooling.tool_format(fmt) if isinstance(fmt, str) else fmt
        self._tool_format = fmt
        # Special marker tokens ([TOOL_CALLS], <|tool_call|>) must appear in the generated text, where the
        # constraint and the parser look for them.
        tokenizer = self._base_tokenizer
        showing = getattr(tokenizer, "showing", None)
        if showing is not None and fmt.open:
            tokenizer = showing([fmt.open])
        self.tokenizer = tokenizer
        self._trie = None
        generator = getattr(self, "_generator", None)
        if generator is not None:
            generator.tokenizer = tokenizer

    @staticmethod
    def from_model_file(
        path: str | Path,
        verify: bool = True,
        quantize: str | None = None,
        device: str = "cpu",
        prompt_cache: int = DEFAULT_PROMPT_CACHE_SIZE,
        adapter: str | Path | None = None,
        steer: str | Path | None = None,
        steer_strength: float | None = None,
        index: str | Path | None = None,
        index_top: int | None = None,
        embedding_model: str | Path | None = None,
        speculate: int | None = None,
        draft_model: str | Path | None = None,
        prompt_cache_dir: str | Path | None = None,
        index_mode: str | None = None,
        rerank_model: str | Path | None = None,
        contrast_model: str | Path | None = None,
        ensemble: Sequence[tuple[str | Path, float]] = (),
        ensemble_weight: float = 1.0,
    ) -> DllmEngine:
        """An engine for an imported ``model.dllm``: its decoder, BPE tokenizer and chat template. ``quantize``
        (``"q8_0"``) runs the linear layers on quantised weights; that changes the output, and so the
        ``system_fingerprint``. ``device="cuda"`` runs the decoder on the GPU with the same output. ``adapter``, a
        PEFT LoRA adapter directory, is merged into the weights first, exactly as ``dllm import ADAPTER --base``
        merges it, so the output and the ``system_fingerprint`` equal those of the merged file. ``steer``, a steering
        vector file, is added to the residual stream after its layer at ``steer_strength`` (default: the file's);
        that changes the output and the ``system_fingerprint``. ``index``, a document index, grounds every chat in
        the ``index_top`` passages it finds for the last user message, embedded with ``embedding_model`` (default:
        the model the index records), ranked by ``index_mode`` (``dense``, ``lexical`` or ``hybrid``) and reranked by
        ``rerank_model`` when given; that changes the output and the ``system_fingerprint`` too. ``speculate``
        drafts that many tokens per step (default: 8 with a ``draft_model``, else off) with ``draft_model``, a smaller
        ``model.dllm`` with the same tokenizer, or from the text so far; it never changes the output.
        ``contrast_model`` and ``ensemble`` (paths with weights) are models with the same tokenizer for contrastive
        decoding and ensembles (:mod:`etalii_dllm.guidance`); they change the ``system_fingerprint``."""
        from etalii_dllm.bpe import from_model_header, special_token_text
        from etalii_dllm.modelfile import ModelFile
        from etalii_dllm.transformer import Transformer

        file = ModelFile(path, verify=verify)
        if file.tokenizer is None:
            raise ValueError(f"{path}: the model file has no tokenizer")
        model_id = str(file.source.get("repository") or Path(path).stem)
        tensors: Mapping[str, np.ndarray] = file.tensors
        weights_fingerprint = file.fingerprint
        if adapter:
            from etalii_dllm.lora import apply_adapter
            from etalii_dllm.modelfile import data_fingerprint

            tensors, _ = apply_adapter(file.config, file.tensors, adapter)
            weights_fingerprint = data_fingerprint(tensors)
        steering = None
        if steer:
            from etalii_dllm.interpret.steering import SteeringVector

            vector = SteeringVector.load(steer)
            if not 1 <= vector.layer <= file.config.layers or vector.vector.shape != (file.config.hidden_size,):
                raise ValueError(f"{steer}: the steering vector does not fit this model")
            steering = {vector.layer - 1: vector.scaled(steer_strength)}
        model = Transformer(
            file.config,
            tensors,
            model_id=model_id,
            weights_fingerprint=weights_fingerprint,
            quantize=quantize,
            device=device,
            steering=steering,
            release=file.release if tensors is file.tensors else None,
        )
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
        fingerprint = "fp_" + model.weights_fingerprint[:12]
        drafter = None
        if draft_model:
            draft = ModelFile(draft_model, verify=verify)
            if _vocabulary(draft.tokenizer) != _vocabulary(file.tokenizer):
                raise ValueError(f"{draft_model}: the draft model's tokenizer differs from the model's")
            drafter = Transformer(
                draft.config,
                draft.tensors,
                weights_fingerprint=draft.fingerprint,
                quantize=quantize,
                device=device,
                release=draft.release,
            )
            if speculate is None:
                speculate = DEFAULT_DRAFT_TOKENS

        def companion(other: str | Path, role: str) -> Transformer:
            loaded = ModelFile(other, verify=verify)
            if _vocabulary(loaded.tokenizer) != _vocabulary(file.tokenizer):
                raise ValueError(f"{other}: the {role} model's tokenizer differs from the model's")
            return Transformer(
                loaded.config,
                loaded.tensors,
                weights_fingerprint=loaded.fingerprint,
                quantize=quantize,
                device=device,
                release=loaded.release,
            )

        return DllmEngine(
            model,
            tokenizer,
            fingerprint,
            chat_template=template,
            stop_tokens=stops,
            prompt_cache=prompt_cache,
            embedding=file.embedding,
            retriever=Retriever.open(
                index, index_top or DEFAULT_TOP, embedding_model, index_mode or "dense", rerank_model
            )
            if index
            else None,
            speculate=speculate or 0,
            draft_model=drafter,
            prompt_cache_dir=prompt_cache_dir,
            contrast_model=companion(contrast_model, "contrast") if contrast_model else None,
            ensemble=[(companion(other, "ensemble"), float(weight)) for other, weight in ensemble],
            ensemble_weight=ensemble_weight,
        )

    @staticmethod
    def create_default() -> DllmEngine:
        tokenizer = ByteTokenizer()
        model = BigramModel(tokenizer.vocabulary_size, DEFAULT_MODEL_SEED)
        return DllmEngine(model, tokenizer, "fp_" + model.weights_fingerprint[:12])

    # -- text ---------------------------------------------------------------------------------------------------

    @property
    def stop_tokens(self) -> frozenset[int]:
        """The token ids that end a generation (the tokenizer's end of sequence and the model's own)."""
        return self._generator.stop_tokens

    def complete(self, prompt: str, max_tokens: int, options: SamplingOptions) -> GenerationResult:
        return self._generator.generate(prompt, max_tokens, options, guide=self.guide(options, raw=True))

    def complete_stream(
        self,
        prompt: str,
        max_tokens: int,
        options: SamplingOptions,
        *,
        regex: str | None = None,
        grammar: str | None = None,
        overflow: str = "stop",
        token_healing: bool = False,
        suffix: str | None = None,
        min_tokens: int = 0,
        ignore_eos: bool = False,
        stop_token_ids: Sequence[int] = (),
        stop: Sequence[str] = (),
        include_stop: bool = False,
    ) -> Generation:
        """Continues ``prompt``; ``regex`` restricts the continuation to text matching it in full, ``grammar`` to
        text the GBNF grammar derives, ``overflow`` says what happens at a full context window
        (docs/api.md#long-conversations), ``token_healing`` heals the prompt's last token (:meth:`heal`) and
        ``suffix`` makes the output the middle between ``prompt`` and it (:meth:`infill`); ``stop`` ends it at
        any of the strings (kept in the text with ``include_stop``); ``min_tokens``, ``ignore_eos`` and
        ``stop_token_ids`` are the length controls (docs/api.md#length-and-stop-controls)."""
        if regex is not None and grammar is not None:
            raise ValueError("a regex and a grammar cannot be combined")
        constraint = None
        if regex is not None:
            constraint = TokenConstraint(Grammar.regex(regex), self._token_trie())
        elif grammar is not None:
            constraint = TokenConstraint(Grammar.gbnf(grammar), self._token_trie())
        guide = self.guide(options, raw=True)
        context, ends = self.infill(prompt, suffix, options, token_healing)
        context, constraint, healed = self.heal(prompt, constraint) if token_healing else (context, constraint, 0)
        return self._generator.stream(
            context,
            max_tokens,
            options,
            constraint=constraint,
            overflow=overflow,
            guide=guide,
            healed=healed,
            stop=stop,
            stop_tokens=ends | frozenset(stop_token_ids),
            min_tokens=min_tokens,
            ignore_eos=ignore_eos,
            include_stop=include_stop,
        )

    def infill(
        self, prompt: str, suffix: str | None, options: SamplingOptions, token_healing: bool = False
    ) -> tuple[str | list[int], frozenset[int]]:
        """Fill-in-the-middle (:mod:`etalii_dllm.infill`): the prompt tokens for the gap between ``prompt`` and
        ``suffix`` and the tokens that end the middle; ``prompt`` and no extra ends without a suffix. Raises
        ``ValueError`` for a model without FIM tokens and for what a middle cannot be combined with."""
        if suffix is None:
            return prompt, frozenset()
        if token_healing:
            raise ValueError("a suffix cannot be combined with token healing")
        if options.negative_prompt is not None:
            raise ValueError("a suffix cannot be combined with a negative prompt")
        tokens = fim_tokens(self.tokenizer)
        return tokens.prompt(self.tokenizer, prompt, suffix), tokens.ends

    def heal(
        self, prompt: str, constraint: TokenConstraint | None
    ) -> tuple[str | list[int], TokenConstraint | HealingConstraint | None, int]:
        """Token healing: the prompt's tokens without the last one, a constraint that makes the output start with
        that token's bytes (then ``constraint`` applies), and how many bytes that is. A prompt that is empty or ends
        with a special token (which has no bytes) is left as it is."""
        tokens = self.tokenizer.encode(prompt)
        data = self.tokenizer.decode_bytes(tokens[-1:])
        if not data:
            return prompt, constraint, 0
        return tokens[:-1], HealingConstraint(data, self._token_trie(), constraint), len(data)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def derive_id(self, prefix: str, payload: Any) -> str:
        """A response id from the request itself (and the weights), so identical requests get identical ids and
        nothing depends on a clock or random source."""
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
        digest = hashlib.sha256(f"{self.system_fingerprint}\n{canonical}".encode()).hexdigest()
        return prefix + digest[:24]

    # -- chat ---------------------------------------------------------------------------------------------------

    def fit_messages(
        self, messages: Sequence[ChatMessage], tools: Sequence[Tool] = (), *, thinking: bool | None = None
    ) -> list[ChatMessage]:
        """``messages`` without their oldest ones, until the rendered prompt leaves room in the context window:
        the earliest message that is neither a system message nor the last message goes first, together with the
        tool results that directly follow it. Unchanged when it already fits or the model has no window; may still
        not fit when only the system messages and the last message are left."""
        window = self._generator.context_length
        kept = list(messages)
        if window is None:
            return kept
        while len(self.tokenizer.encode(self.render_chat(kept, tools, thinking=thinking))) >= window:
            first = next((i for i, m in enumerate(kept[:-1]) if m.role != "system"), None)
            if first is None:
                break
            end = first + 1
            while end < len(kept) - 1 and kept[end].role == "tool":
                end += 1
            del kept[first:end]
        return kept

    @property
    def thinks(self) -> bool:
        """Whether the model is a thinking model (its chat template writes ``<think>`` blocks)."""
        return self.chat_template is not None and thinking_rules.is_thinking_template(self.chat_template.source)

    def render_chat(
        self, messages: Iterable[ChatMessage], tools: Sequence[Tool] = (), *, thinking: bool | None = None
    ) -> str:
        """The prompt for a conversation: the model's own chat template when it has one. Tools are presented by
        the template when it supports them (calls in its own format, :attr:`tool_format`), else by Hermes
        instructions in the system message. A final assistant message without tool calls is a prefill: the answer
        continues its text. ``thinking`` switches a thinking
        model's ``<think>`` block on or off (the template's ``enable_thinking``; a template that ignores it gets an
        empty, closed block when thinking is off)."""
        messages = list(messages)
        if len(messages) > 1 and messages[-1].role == "assistant" and not messages[-1].tool_calls:
            return self.render_chat(messages[:-1], tools, thinking=thinking) + messages[-1].content
        if thinking is not None and self.thinks:
            prompt = self._render(messages, tools, enable_thinking=thinking)
            if thinking or prompt != self._render(messages, tools, enable_thinking=True):
                return prompt
            if thinking_rules.starts_in_thinking(prompt):
                return prompt + "\n" + thinking_rules.THINK_CLOSE + "\n\n"
            return prompt + thinking_rules.THINK_OPEN + "\n\n" + thinking_rules.THINK_CLOSE + "\n\n"
        return self._render(messages, tools)

    def _render(self, messages: list[ChatMessage], tools: Sequence[Tool], **variables: Any) -> str:
        source = self.chat_template.source if self.chat_template is not None else None
        uses_tools = bool(tools) or any(m.tool_calls or m.role == "tool" for m in messages)
        if uses_tools and not tooling.template_supports_tools(source):
            messages = tooling.with_instructions(messages, tools)
            tools = ()
        if self.chat_template is None:
            return render(messages)
        return self.chat_template.render(
            tooling.template_messages(messages, self.tool_format),
            tools=[t.to_openai() for t in tools] or None,
            **variables,
        )

    def chat(self, messages: Iterable[ChatMessage], max_tokens: int, options: SamplingOptions) -> GenerationResult:
        return self.complete(self.render_chat(messages), max_tokens, options)

    def _token_trie(self) -> TokenTrie:
        if self._trie is None:
            size = self.model.vocabulary_size
            self._trie = TokenTrie([self.tokenizer.decode_bytes([token]) for token in range(size)])
        return self._trie

    def _constraint(self, request: ChatRequest, tools: Sequence[Tool]) -> TokenConstraint | None:
        answer = request.response_format.grammar()
        choice = request.tool_choice
        fmt = self.tool_format
        if tools and choice.mode in ("required", "named"):
            return TokenConstraint(tooling.forced_grammar(tools, choice, fmt), self._token_trie())
        if tools and answer is not None:
            either = Grammar.either([tooling.forced_grammar(tools, AUTO, fmt), answer])
            return TokenConstraint(either, self._token_trie())
        if tools:
            grammar, trigger = tooling.auto_grammar(tools, fmt)
            return TokenConstraint(grammar, self._token_trie(), trigger=trigger)
        if answer is not None:
            return TokenConstraint(answer, self._token_trie())
        return None

    def chat_stream(self, request: ChatRequest, *, fresh: bool = False) -> ChatStream:
        """Starts a chat generation. Raises ``ValueError`` for invalid requests (unknown tools, unsupported
        schemas, ...) before any token is generated.

        A request the response cache holds is answered from it, and one identical to a generation still in progress
        shares that generation (:mod:`etalii_dllm.serving`); both give the same events, so only ``cached_tokens``
        tells. ``fresh`` always runs the model: replays and audits use it."""
        tools = [] if request.tool_choice.mode == "none" else list(request.tools)
        tooling.validate_tools(tools, request.tool_choice)
        if request.prompt is not None and tools:
            raise ValueError("a raw prompt cannot use tools")
        if request.suffix is not None and request.prompt is None:
            raise ValueError("a suffix needs a raw prompt")
        if fresh:
            generation = self._generate(request, tools)
            return ChatStream(
                generation.prompt_tokens, self._answer(iter(generation), request, audit=False), generation.cached_tokens
            )
        key = response_key(self, request)
        cache = self.response_cache
        recorded = cache.get(key) if cache is not None else None
        if recorded is not None:
            events = self._answer(iter(recorded.events), request)
            return ChatStream(recorded.prompt_tokens, events, recorded.prompt_tokens)

        def start() -> SharedGeneration:
            generation = self._generate(request, tools)
            store = None if cache is None else lambda finished: cache.put(key, finished)
            return SharedGeneration(generation.prompt_tokens, generation.cached_tokens, iter(generation), store)

        shared = self.inflight.get_or_start(key, start)
        return ChatStream(shared.prompt_tokens, self._answer(shared.reader(), request), shared.cached_tokens)

    def request_prompt(self, request: ChatRequest, tools: Sequence[Tool] = ()) -> tuple[Sequence[ChatMessage], str]:
        """The messages a request's answer continues (grounded and truncated as configured) and its prompt text: the
        raw prompt, or the rendered conversation."""
        messages = request.messages
        if self.retriever is not None and request.prompt is None:
            messages, _ = self.retriever.ground(messages)
        if request.prompt is None and request.truncation == "auto":
            messages = self.fit_messages(messages, tools, thinking=request.thinking)
        prompt = (
            request.prompt
            if request.prompt is not None
            else self.render_chat(messages, tools, thinking=request.thinking)
        )
        return messages, prompt

    def _generate(self, request: ChatRequest, tools: Sequence[Tool]) -> _ChatGeneration:
        constraint = self._constraint(request, tools)
        messages, prompt = self.request_prompt(request, tools)
        tracker = None
        if request.prompt is None and self.thinks and (constraint is None or not constraint.active):
            # Structured output and forced tool calls constrain the output from its first token: no thinking then.
            tracker = thinking_rules.Tracker(thinking_rules.starts_in_thinking(prompt), request.max_reasoning_tokens)
        context, ends = self.infill(prompt, request.suffix, request.options, request.token_healing)
        context, healing, healed = self.heal(prompt, constraint) if request.token_healing else (context, constraint, 0)
        generation = self._generator.stream(
            context,
            request.max_tokens,
            request.options,
            stop=request.stop,
            constraint=healing,
            top_logprobs=request.top_logprobs,
            new_text=request.prompt is None,
            overflow=request.context_overflow,
            reasoning=tracker,
            guide=self.guide(request.options, request.prompt is not None, messages, tools, request.thinking),
            healed=healed,
            stop_tokens=ends | frozenset(request.stop_token_ids),
            min_tokens=request.min_tokens,
            ignore_eos=request.ignore_eos,
            include_stop=request.include_stop,
        )
        return _ChatGeneration(
            generation.prompt_tokens, generation.cached_tokens, self._events(generation, request, tools)
        )

    def guide(
        self,
        options: SamplingOptions,
        raw: bool,
        messages: Sequence[ChatMessage] = (),
        tools: Sequence[Tool] = (),
        thinking: bool | None = None,
    ) -> Callable[[list[int]], Guide] | None:
        """What guides decoding for ``options`` (:mod:`etalii_dllm.guidance`), as a function of the prompt's tokens:
        the negative prompt (for a chat, the conversation with the last user message replaced by it), contrastive
        decoding, the engine's ensemble, or nothing. Raises ``ValueError`` for combinations that are not allowed."""
        if self.ensemble and (options.negative_prompt is not None or options.contrast_beta is not None):
            raise ValueError("this engine decodes with an ensemble; negative prompts and contrast do not combine")
        if options.negative_prompt is not None:
            text = options.negative_prompt
            if not raw:
                last = max((i for i, m in enumerate(messages) if m.role == "user"), default=None)
                if last is None:
                    raise ValueError("a negative prompt needs a user message to replace")
                negative = [*messages[:last], replace(messages[last], content=text), *messages[last + 1 :]]
                text = self.render_chat(negative, tools, thinking=thinking)
            tokens = self.tokenizer.encode(text) or [self.tokenizer.end_of_sequence]
            model, scale = self.model, options.guidance_scale
            return lambda _context: guidance.NegativePrompt(model, tokens, scale)
        if options.contrast_beta is not None:
            if self.contrast_model is None:
                raise ValueError("contrastive decoding needs a contrast model (--contrast-model)")
            amateur, alpha, beta = self.contrast_model, options.contrast_alpha, options.contrast_beta
            return lambda context: guidance.Contrast(amateur, context, alpha, beta)
        if self.ensemble:
            members, weight = self.ensemble, self.ensemble_weight
            return lambda context: guidance.Ensemble(members, weight, context)
        return None

    def _answer(self, events: Iterator[ChatEvent], request: ChatRequest, audit: bool = True) -> Iterator[ChatEvent]:
        """The events for ``request``: its own receipt chain link and signature on the shared receipt, which the
        auditor sees (not for replays, which are what it runs)."""
        for event in events:
            if isinstance(event, Finished) and event.receipt is not None:
                receipt = event.receipt
                if request.previous_receipt:
                    output = receipt["output"]
                    receipt = receipts.make_receipt(
                        __version__, self.model.id, self.system_fingerprint, request, output, request.previous_receipt
                    )
                if self.signer is not None:
                    receipt = self.signer.sign(receipt)
                if audit and self.auditor is not None:
                    self.auditor.observe(receipt)
                event = replace(event, receipt=receipt)
            yield event

    def _events(self, generation: Generation, request: ChatRequest, tools: Sequence[Tool]) -> Iterator[ChatEvent]:
        tracker = generation.reasoning
        raw = ""
        """Everything generated; with a thinking model, ``text`` is only its answer part."""
        text = ""
        streamed = ""
        thought: str | None = None
        """Reasoning text already sent (``None``: nothing yet)."""
        tokens: list[int] = []
        last = None
        for step in generation:
            last = step
            if step.token is not None:
                tokens.append(step.token)
            raw += step.text
            logprobs = (step.logprobs,) if step.logprobs is not None else ()
            if tracker is not None:
                reasoning, answer = thinking_rules.streamable(raw, tracker.started)
                in_answer = thinking_rules.split(raw, tracker.started).state == "answer"
                sent = thought or ""
                if reasoning[len(sent) :] or (logprobs and not in_answer):
                    yield ReasoningDelta(reasoning[len(sent) :], () if in_answer else logprobs)
                    thought = reasoning
                if not in_answer:
                    logprobs = ()
                delta_text = answer[len(text) :]
                text = answer
            else:
                delta_text = step.text
                text = raw
            if not tools:
                if delta_text or logprobs:
                    yield TextDelta(delta_text, logprobs)
                continue
            # With tools, only text that is certainly answer text is streamed: not leading or trailing whitespace,
            # nothing from a <tool_call> on, and nothing at all when the reply starts like a bare JSON call.
            safe = _answer_prefix(text, self.tool_format.open)
            delta = safe[len(streamed) :]
            streamed = safe
            if delta or logprobs:
                yield TextDelta(delta, logprobs)
        assert last is not None and last.finish_reason is not None
        result = generation.result()
        reasoning_text: str | None = None
        if tracker is not None:
            parts = thinking_rules.split(raw, tracker.started)
            reasoning_text = parts.reasoning
            if reasoning_text is not None and (thought is None or reasoning_text[len(thought) :]):
                yield ReasoningDelta(reasoning_text[len(thought or "") :])
            if not tools and parts.answer[len(text) :]:
                yield TextDelta(parts.answer[len(text) :])
            text = parts.answer
        finish_reason, calls, content = result.finish_reason, [], text
        if tools:
            content, parsed = tooling.parse_calls(text, tools, self.tool_format)
            if not content.startswith(streamed):  # pragma: no cover - _answer_prefix guarantees this
                raise AssertionError("streamed text is not a prefix of the answer")
            if content[len(streamed) :]:
                yield TextDelta(content[len(streamed) :])
            for index, (name, arguments) in enumerate(parsed):
                call_id = (
                    request.call_id_prefix
                    + hashlib.sha256(f"{request.request_id}\n{result.fingerprint}\n{index}".encode()).hexdigest()[:24]
                )
                calls.append(ToolCall(call_id, name, arguments))
                yield ToolCallEvent(index, calls[-1])
            if calls:
                finish_reason = "tool_calls"
        stop_sequence = generation.stop_sequence if finish_reason == "stop" else None
        output = receipts.output_record(
            result.fingerprint, content, calls, finish_reason, generation.prompt_tokens, len(tokens), reasoning_text
        )
        receipt = receipts.make_receipt(__version__, self.model.id, self.system_fingerprint, request, output)
        reasoning_tokens = tracker.tokens if tracker is not None else 0
        yield Finished(finish_reason, stop_sequence, len(tokens), result.fingerprint, receipt, reasoning_tokens)

    def chat_completion(self, request: ChatRequest, *, fresh: bool = False) -> ChatResult:
        return self._collect(self.chat_stream(request, fresh=fresh))

    @staticmethod
    def _collect(stream: ChatStream) -> ChatResult:
        content: list[str] = []
        thought: list[str] | None = None
        calls: list[ToolCall] = []
        logprobs: list[TokenLogprobs] = []
        finished: Finished | None = None
        for event in stream:
            if isinstance(event, TextDelta):
                content.append(event.text)
                logprobs.extend(event.logprobs)
            elif isinstance(event, ReasoningDelta):
                thought = [*(thought or []), event.text]
            elif isinstance(event, ToolCallEvent):
                calls.append(event.call)
            else:
                finished = event
        assert finished is not None
        return ChatResult(
            "".join(content),
            tuple(calls),
            finished.finish_reason,
            finished.stop_sequence,
            stream.prompt_tokens,
            finished.completion_tokens,
            finished.fingerprint,
            tuple(logprobs),
            stream.cached_tokens,
            finished.receipt,
            "".join(thought) if thought is not None else None,
            finished.reasoning_tokens,
        )

    @staticmethod
    def choice_request(request: ChatRequest, index: int) -> ChatRequest:
        """Choice ``index`` of a request asking for several: the same request with the seed ``seed + index``
        (:meth:`SamplingOptions.for_choice`), so each choice is exactly what a single request with that seed
        answers."""
        return request if index == 0 else replace(request, options=request.options.for_choice(index))

    def chat_choices(self, request: ChatRequest, n: int) -> list[ChatResult]:
        """``n`` choices (:meth:`choice_request`), in index order. They run concurrently, so a model that batches
        decodes them in shared steps, which never changes a bit of any of them."""
        if not 1 <= n <= MAX_CHOICES:
            raise ValueError(f"n must be between 1 and {MAX_CHOICES}")
        requests = [self.choice_request(request, i) for i in range(n)]
        first = self.chat_stream(requests[0])  # raises for invalid requests before anything runs
        if n == 1:
            return [self._collect(first)]
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=n) as pool:
            rest = [pool.submit(self.chat_completion, r) for r in requests[1:]]
            results = [self._collect(first)]
            return results + [future.result() for future in rest]

    # -- embeddings ---------------------------------------------------------------------------------------------

    def embed(
        self, text: str | Sequence[int], dimensions: int | None = None, input_type: str | None = None
    ) -> Embedding:
        """The embedding of ``text``, L2-normalised; ``dimensions`` keeps the first components and normalises again.

        Models imported from sentence-transformers use their own recipe: the text is prefixed with the prompt named
        ``input_type`` (e.g. ``"query"``; default: the model's default prompt), encoded with the tokenizer's special
        tokens, and pooled as the model says (``last_token``: the last position's final hidden state; ``mean``).
        Other models take the mean of the final hidden states over all positions. The mean sums each column over
        positions ascending in double, through the ``linear`` kernel."""
        settings = self.embedding or {}
        prompts: Mapping[str, str] = settings.get("prompts") or {}
        name = input_type or settings.get("default_prompt_name")
        if name is not None and prompts and name not in prompts:
            raise ValueError(f"unknown input_type {name!r}; this model has {', '.join(sorted(prompts))}")
        if isinstance(text, str):
            if settings:
                text = prompts.get(name, "") + text if name else text
                tokens = self.tokenizer.encode(text, add_special_tokens=True)  # type: ignore[call-arg]
            else:
                tokens = self.tokenizer.encode(text)
        else:
            tokens = list(text)
        if not tokens:
            raise ValueError("cannot embed an empty input")
        hidden_states = getattr(self.model, "hidden_states", None)
        if hidden_states is None:
            raise ValueError(f"model {self.model.id} does not provide hidden states")
        states = np.asarray(hidden_states(tokens), dtype=np.float32)
        if settings.get("pooling") == "last_token":
            vector = states[-1].copy()
        else:
            ones = np.ones((1, states.shape[0]), dtype=np.float32)
            total = linear(ones, np.ascontiguousarray(states.T)).numpy().reshape(-1)
            vector = (total / np.float32(states.shape[0])).astype(np.float32)
        if dimensions is not None:
            if not 1 <= dimensions <= vector.shape[0]:
                raise ValueError(f"dimensions must be between 1 and {vector.shape[0]}")
            vector = vector[:dimensions]
        if settings.get("normalize", True) or dimensions is not None:
            norm = np.float32(np.sqrt(sum_squares(vector)))
            if norm > 0:
                vector = (vector / norm).astype(np.float32)
        return Embedding(vector, len(tokens))


def _answer_prefix(text: str, marker: str = TOOL_CALL_OPEN) -> str:
    """The part of generated text that is certainly answer text when tools are available (see ``_events``):
    nothing from the tool call ``marker`` on (none for bare-JSON formats), and nothing when the reply starts like a
    JSON call."""
    stripped = text.lstrip()
    if stripped.startswith("{"):
        return ""
    cut = stripped.find(marker) if marker else -1
    if cut >= 0:
        stripped = stripped[:cut]
    else:
        for length in range(min(len(marker) - 1, len(stripped)), 0, -1):
            if stripped.endswith(marker[:length]):
                stripped = stripped[:-length]
                break
    return stripped.rstrip()


def use_model_file(
    path: str | Path | None,
    quantize: str | None = None,
    threads: int | None = None,
    device: str | None = None,
    prompt_cache: int | None = None,
    adapter: str | Path | None = None,
    steer: str | Path | None = None,
    steer_strength: float | None = None,
    index: str | Path | None = None,
    index_top: int | None = None,
    embedding_model: str | Path | None = None,
    speculate: int | None = None,
    draft_model: str | Path | None = None,
    prompt_cache_dir: str | Path | None = None,
    sign_key: str | Path | None = None,
    index_mode: str | None = None,
    rerank_model: str | Path | None = None,
    response_cache: str | Path | None = None,
    audit_every: int | None = None,
    contrast_model: str | Path | None = None,
    ensemble_models: Sequence[str] = (),
    ensemble_weight: float | None = None,
) -> None:
    """Makes the front ends serve ``path`` (sets ``DLLM_MODEL``, and ``DLLM_QUANTIZE``/``DLLM_DEVICE``/
    ``DLLM_PROMPT_CACHE``/``DLLM_ADAPTER`` when ``quantize``/``device``/``prompt_cache``/``adapter`` are given) and
    resets the default engine.
    ``threads`` sets the kernel thread count; it, ``device`` and ``prompt_cache`` never change the output."""
    if path:
        os.environ[MODEL_ENVIRONMENT_VARIABLE] = str(path)
    if quantize:
        os.environ[QUANTIZE_ENVIRONMENT_VARIABLE] = quantize
    if device:
        os.environ[DEVICE_ENVIRONMENT_VARIABLE] = device
    if prompt_cache is not None:
        os.environ[PROMPT_CACHE_ENVIRONMENT_VARIABLE] = str(prompt_cache)
    if adapter:
        os.environ[ADAPTER_ENVIRONMENT_VARIABLE] = str(adapter)
    if steer:
        os.environ[STEER_ENVIRONMENT_VARIABLE] = str(steer)
    if steer_strength is not None:
        os.environ[STEER_STRENGTH_ENVIRONMENT_VARIABLE] = repr(float(steer_strength))
    if index:
        os.environ[INDEX_ENVIRONMENT_VARIABLE] = str(index)
    if index_top is not None:
        os.environ[INDEX_TOP_ENVIRONMENT_VARIABLE] = str(index_top)
    if embedding_model:
        os.environ[EMBEDDING_MODEL_ENVIRONMENT_VARIABLE] = str(embedding_model)
    if index_mode:
        os.environ[INDEX_MODE_ENVIRONMENT_VARIABLE] = index_mode
    if rerank_model:
        os.environ[RERANK_MODEL_ENVIRONMENT_VARIABLE] = str(rerank_model)
    if speculate is not None:
        os.environ[SPECULATE_ENVIRONMENT_VARIABLE] = str(speculate)
    if draft_model:
        os.environ[DRAFT_MODEL_ENVIRONMENT_VARIABLE] = str(draft_model)
    if prompt_cache_dir:
        os.environ[PROMPT_CACHE_DIR_ENVIRONMENT_VARIABLE] = str(prompt_cache_dir)
    if sign_key:
        os.environ[SIGN_KEY_ENVIRONMENT_VARIABLE] = str(sign_key)
    if response_cache:
        os.environ[RESPONSE_CACHE_ENVIRONMENT_VARIABLE] = str(response_cache)
    if audit_every is not None:
        os.environ[AUDIT_EVERY_ENVIRONMENT_VARIABLE] = str(audit_every)
    if contrast_model:
        os.environ[CONTRAST_MODEL_ENVIRONMENT_VARIABLE] = str(contrast_model)
    if ensemble_models:
        os.environ[ENSEMBLE_MODELS_ENVIRONMENT_VARIABLE] = os.pathsep.join(ensemble_models)
    if ensemble_weight is not None:
        os.environ[ENSEMBLE_WEIGHT_ENVIRONMENT_VARIABLE] = repr(float(ensemble_weight))
    if threads is not None:
        set_threads(threads)
    default_engine.cache_clear()


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    """The ``--model``, ``--adapter``, ``--quantize``, ``--threads``, ``--device``, ``--steer``, ``--steer-strength``,
    ``--index``, ``--index-top``, ``--embedding-model``, ``--prompt-cache``, ``--persistent-cache``, ``--speculate``
    and ``--draft-model`` options every front end shares."""
    parser.add_argument("--model", help="model.dllm file to use (default: $DLLM_MODEL, else the placeholder model)")
    parser.add_argument(
        "--adapter",
        help="PEFT LoRA adapter directory to merge into the model when it loads (default: $DLLM_ADAPTER, else none)",
    )
    parser.add_argument(
        "--quantize",
        choices=("none", *QUANTIZATIONS),
        help="run the linear layers on quantised weights (default: $DLLM_QUANTIZE, else none); changes the output",
    )
    parser.add_argument(
        "--threads", type=int, help="kernel threads (default: $DLLM_THREADS, else all cores); never changes output"
    )
    parser.add_argument(
        "--device",
        choices=DEVICES,
        help="run the model on the CPU or an NVIDIA GPU (default: $DLLM_DEVICE, else cpu); never changes output",
    )
    parser.add_argument(
        "--steer", help="steering vector file (dllm steer) to add to the model (default: $DLLM_STEER); changes output"
    )
    parser.add_argument(
        "--steer-strength",
        type=float,
        help="multiplier for the steering vector (default: $DLLM_STEER_STRENGTH, else the file's)",
    )
    parser.add_argument(
        "--index",
        help="document index (dllm index build) to ground every chat in (default: $DLLM_INDEX); changes output",
    )
    parser.add_argument(
        "--index-top", type=int, help="passages to add from the index (default: $DLLM_INDEX_TOP, else 3)"
    )
    parser.add_argument(
        "--embedding-model",
        help="the index's embedding model (default: $DLLM_EMBEDDING_MODEL, else the path the index records)",
    )
    parser.add_argument(
        "--index-mode",
        choices=MODES,
        help="how to search the index: embeddings, BM25 or both fused (default: $DLLM_INDEX_MODE, else dense)",
    )
    parser.add_argument(
        "--rerank-model",
        help="chat model.dllm that reranks the passages found in the index (default: $DLLM_RERANK_MODEL, else none)",
    )
    parser.add_argument(
        "--contrast-model",
        help="smaller model.dllm with the same tokenizer for contrastive decoding (requests with contrast; default: "
        "$DLLM_CONTRAST_MODEL, else none)",
    )
    parser.add_argument(
        "--ensemble-model",
        action="append",
        metavar="PATH[=WEIGHT]",
        help="decode every request with this model too, its log-probabilities averaged in (repeatable; default: "
        "$DLLM_ENSEMBLE_MODELS); changes output",
    )
    parser.add_argument(
        "--ensemble-weight",
        type=float,
        help="the served model's weight in the ensemble (default: $DLLM_ENSEMBLE_WEIGHT, else 1)",
    )
    parser.add_argument(
        "--prompt-cache",
        type=int,
        metavar="N",
        help="KV caches kept to reuse shared prompt prefixes across requests (default: $DLLM_PROMPT_CACHE, else "
        f"{DEFAULT_PROMPT_CACHE_SIZE}; 0 disables); never changes output",
    )
    parser.add_argument(
        "--sign-key",
        metavar="FILE",
        help="sign every receipt with this Ed25519 private key (dllm sign --keygen; default: $DLLM_SIGN_KEY)",
    )
    parser.add_argument(
        "--response-cache",
        metavar="DIR",
        help="answer repeated requests from responses stored in DIR (default: $DLLM_RESPONSE_CACHE, else off); "
        "never changes output",
    )
    parser.add_argument(
        "--audit-every",
        type=int,
        metavar="N",
        help="re-run every Nth response in the background to check it reproduces, reported at GET /v1/audit "
        "(default: $DLLM_AUDIT_EVERY, else off)",
    )
    parser.add_argument(
        "--persistent-cache",
        dest="prompt_cache_dir",
        metavar="DIR",
        help="keep the prompt cache in DIR across restarts (default: $DLLM_PROMPT_CACHE_DIR, else memory only); "
        "never changes output",
    )
    parser.add_argument(
        "--speculate",
        type=int,
        nargs="?",
        const=DEFAULT_DRAFT_TOKENS,
        metavar="N",
        help=f"speculative decoding: draft N tokens per step (default N: {DEFAULT_DRAFT_TOKENS}; default: "
        "$DLLM_SPECULATE, else off, or on with --draft-model); never changes output",
    )
    parser.add_argument(
        "--draft-model",
        help="smaller model.dllm with the same tokenizer to draft with (default: $DLLM_DRAFT_MODEL, else the text "
        "so far); never changes output",
    )


def configured_quantization() -> str | None:
    """``$DLLM_QUANTIZE``, or ``None`` when it is unset, empty or ``none``."""
    value = os.environ.get(QUANTIZE_ENVIRONMENT_VARIABLE, "").strip().lower()
    if value in ("", "none"):
        return None
    if value not in QUANTIZATIONS:
        raise ValueError(f"{QUANTIZE_ENVIRONMENT_VARIABLE}={value!r}; supported: none, {', '.join(QUANTIZATIONS)}")
    return value


def configured_device() -> str:
    """``$DLLM_DEVICE``, or ``"cpu"`` when it is unset or empty."""
    value = os.environ.get(DEVICE_ENVIRONMENT_VARIABLE, "").strip().lower() or "cpu"
    if value not in DEVICES:
        raise ValueError(f"{DEVICE_ENVIRONMENT_VARIABLE}={value!r}; supported: {', '.join(DEVICES)}")
    return value


def configured_steer_strength() -> float | None:
    """``$DLLM_STEER_STRENGTH``, or ``None`` when it is unset or empty."""
    value = os.environ.get(STEER_STRENGTH_ENVIRONMENT_VARIABLE, "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"{STEER_STRENGTH_ENVIRONMENT_VARIABLE}={value!r}; expected a number") from None


def configured_index_top() -> int:
    """``$DLLM_INDEX_TOP``, or the default when it is unset or empty."""
    value = os.environ.get(INDEX_TOP_ENVIRONMENT_VARIABLE, "").strip()
    if not value:
        return DEFAULT_TOP
    if not value.isdigit() or int(value) < 1:
        raise ValueError(f"{INDEX_TOP_ENVIRONMENT_VARIABLE}={value!r}; expected a positive integer")
    return int(value)


def configured_index_mode() -> str:
    """``$DLLM_INDEX_MODE``, or ``dense`` when it is unset or empty."""
    value = os.environ.get(INDEX_MODE_ENVIRONMENT_VARIABLE, "").strip() or "dense"
    if value not in MODES:
        raise ValueError(f"{INDEX_MODE_ENVIRONMENT_VARIABLE}={value!r}; expected one of {', '.join(MODES)}")
    return value


def configured_speculate() -> int | None:
    """``$DLLM_SPECULATE``, or ``None`` when it is unset or empty."""
    value = os.environ.get(SPECULATE_ENVIRONMENT_VARIABLE, "").strip()
    if not value:
        return None
    if not value.isdigit():
        raise ValueError(f"{SPECULATE_ENVIRONMENT_VARIABLE}={value!r}; expected a non-negative integer")
    return int(value)


def _vocabulary(tokenizer: Mapping[str, Any] | None) -> Any:
    """What decides token ids in a ``model.dllm`` tokenizer header: the vocabulary and the added tokens."""
    if not tokenizer:
        return None
    if tokenizer.get("format") == "huggingface":
        spec = tokenizer.get("tokenizer_json") or {}
        added = [(t.get("id"), t.get("content")) for t in spec.get("added_tokens") or []]
        return (spec.get("model") or {}).get("vocab"), added
    return {k: v for k, v in tokenizer.items() if k.startswith("tokenizer.ggml.") and "token" in k}


def configured_prompt_cache() -> int:
    """``$DLLM_PROMPT_CACHE``, or the default when it is unset or empty."""
    value = os.environ.get(PROMPT_CACHE_ENVIRONMENT_VARIABLE, "").strip()
    if not value:
        return DEFAULT_PROMPT_CACHE_SIZE
    if not value.isdigit():
        raise ValueError(f"{PROMPT_CACHE_ENVIRONMENT_VARIABLE}={value!r}; expected a non-negative integer")
    return int(value)


@cache
def default_engine() -> DllmEngine:
    """Process-wide default engine: the model named by ``DLLM_MODEL``, else the placeholder. Models are immutable
    and ``forward`` keeps no shared state, so sharing the engine between requests is safe."""
    path = os.environ.get(MODEL_ENVIRONMENT_VARIABLE)
    engine = DllmEngine.create_default() if not path else _configured_engine(path)
    key = os.environ.get(SIGN_KEY_ENVIRONMENT_VARIABLE)
    if key:
        engine.signer = Signer.load(key)
    directory = os.environ.get(RESPONSE_CACHE_ENVIRONMENT_VARIABLE)
    if directory:
        engine.response_cache = ResponseCache(directory)
    every = configured_audit_every()
    if every:
        engine.auditor = Auditor(engine, every)
    return engine


def configured_ensemble() -> list[tuple[str, float]]:
    """``$DLLM_ENSEMBLE_MODELS``: ``PATH[=WEIGHT]`` items (weight 1 when left out)."""
    members = []
    for item in os.environ.get(ENSEMBLE_MODELS_ENVIRONMENT_VARIABLE, "").split(os.pathsep):
        if not item.strip():
            continue
        path, separator, weight = item.rpartition("=")
        try:
            members.append((path, float(weight)) if separator else (item, 1.0))
        except ValueError:
            members.append((item, 1.0))  # an "=" that is part of the path
    return members


def configured_ensemble_weight() -> float:
    """``$DLLM_ENSEMBLE_WEIGHT``, or 1."""
    value = os.environ.get(ENSEMBLE_WEIGHT_ENVIRONMENT_VARIABLE, "").strip()
    if not value:
        return 1.0
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"{ENSEMBLE_WEIGHT_ENVIRONMENT_VARIABLE}={value!r}; expected a number") from None


def configured_audit_every() -> int:
    """``DLLM_AUDIT_EVERY``: re-run every Nth response (0: no audit)."""
    value = os.environ.get(AUDIT_EVERY_ENVIRONMENT_VARIABLE, "").strip()
    if not value:
        return 0
    if not value.isdigit():
        raise ValueError(f"{AUDIT_EVERY_ENVIRONMENT_VARIABLE}={value!r}; expected a non-negative integer")
    return int(value)


def _configured_engine(path: str) -> DllmEngine:
    return DllmEngine.from_model_file(
        path,
        quantize=configured_quantization(),
        device=configured_device(),
        prompt_cache=configured_prompt_cache(),
        adapter=os.environ.get(ADAPTER_ENVIRONMENT_VARIABLE) or None,
        steer=os.environ.get(STEER_ENVIRONMENT_VARIABLE) or None,
        steer_strength=configured_steer_strength(),
        index=os.environ.get(INDEX_ENVIRONMENT_VARIABLE) or None,
        index_top=configured_index_top(),
        embedding_model=os.environ.get(EMBEDDING_MODEL_ENVIRONMENT_VARIABLE) or None,
        index_mode=configured_index_mode(),
        rerank_model=os.environ.get(RERANK_MODEL_ENVIRONMENT_VARIABLE) or None,
        speculate=configured_speculate(),
        draft_model=os.environ.get(DRAFT_MODEL_ENVIRONMENT_VARIABLE) or None,
        prompt_cache_dir=os.environ.get(PROMPT_CACHE_DIR_ENVIRONMENT_VARIABLE) or None,
        contrast_model=os.environ.get(CONTRAST_MODEL_ENVIRONMENT_VARIABLE) or None,
        ensemble=configured_ensemble(),
        ensemble_weight=configured_ensemble_weight(),
    )
