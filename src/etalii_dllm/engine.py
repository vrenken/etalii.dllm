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
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from etalii_dllm import __version__, receipts
from etalii_dllm import tools as tooling
from etalii_dllm.chat import TOOL_CALL_OPEN, ChatMessage, ToolCall, render
from etalii_dllm.chat_template import ChatTemplate
from etalii_dllm.cuda import DEVICES
from etalii_dllm.generation import Generation, GenerationResult, Generator, TokenLogprobs
from etalii_dllm.grammar import Grammar, TokenConstraint, TokenTrie
from etalii_dllm.models import BigramModel, LanguageModel
from etalii_dllm.numerics import QUANTIZATIONS, linear, set_threads, sum_squares
from etalii_dllm.prompt_cache import DEFAULT_PROMPT_CACHE_SIZE
from etalii_dllm.retrieval import DEFAULT_TOP, Retriever
from etalii_dllm.sampling import GREEDY, SamplingOptions
from etalii_dllm.serving import Auditor, Inflight, ResponseCache, SharedGeneration, response_key
from etalii_dllm.signing import Signer
from etalii_dllm.speculative import DEFAULT_DRAFT_TOKENS
from etalii_dllm.tokenization import ByteTokenizer, Tokenizer
from etalii_dllm.tools import AUTO, Tool, ToolChoice

DEFAULT_MODEL_SEED = 42
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
    """``text`` (free), ``json_object`` (any JSON object) or ``json_schema`` (a value valid under ``schema``)."""

    type: str = "text"
    schema: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.type not in ("text", "json_object", "json_schema"):
            raise ValueError(f"unknown response format {self.type!r}")
        if (self.type == "json_schema") != (self.schema is not None):
            raise ValueError("a json_schema response format needs a schema")

    def grammar(self) -> Grammar | None:
        if self.type == "json_object":
            return Grammar.json_object()
        if self.type == "json_schema":
            assert self.schema is not None
            return Grammar.json_schema(self.schema)
        return None


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


@dataclass(frozen=True)
class TextDelta:
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


ChatEvent = TextDelta | ToolCallEvent | Finished


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
    ) -> None:
        """``prompt_cache`` keeps that many KV caches to reuse for prompts sharing a prefix with an earlier one
        (:mod:`etalii_dllm.prompt_cache`), on disk across restarts with ``prompt_cache_dir``; it saves work and never
        changes the output. ``speculate`` drafts that many
        tokens per step, with ``draft_model`` or from the text so far, and checks them in one pass
        (:mod:`etalii_dllm.speculative`); that saves work and never changes the output either. ``embedding`` holds an
        embedding model's pooling settings (:attr:`etalii_dllm.modelfile.ModelFile.embedding`). ``retriever`` grounds
        chats in a document index (:class:`etalii_dllm.retrieval.Retriever`); its fingerprint joins the
        ``system_fingerprint``."""
        self.model = model
        self.embedding = dict(embedding) if embedding else None
        self.retriever = retriever
        if retriever is not None:
            joined = hashlib.sha256(f"{system_fingerprint}|{retriever.fingerprint}".encode()).hexdigest()
            system_fingerprint = "fp_" + joined[:12]
        self.tokenizer = tokenizer
        self.system_fingerprint = system_fingerprint
        """Identifies the exact weights and engine; equal fingerprints plus equal requests give equal output."""
        self.chat_template = chat_template
        cache_dir = str(prompt_cache_dir) if prompt_cache_dir else None
        self._generator = Generator(model, tokenizer, stop_tokens, prompt_cache, speculate, draft_model, cache_dir)
        self._trie: TokenTrie | None = None
        self.inflight = Inflight()
        """Generations still being read: identical requests share one (:mod:`etalii_dllm.serving`)."""

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
    ) -> DllmEngine:
        """An engine for an imported ``model.dllm``: its decoder, BPE tokenizer and chat template. ``quantize``
        (``"q8_0"``) runs the linear layers on quantised weights; that changes the output, and so the
        ``system_fingerprint``. ``device="cuda"`` runs the decoder on the GPU with the same output. ``adapter``, a
        PEFT LoRA adapter directory, is merged into the weights first, exactly as ``dllm import ADAPTER --base``
        merges it, so the output and the ``system_fingerprint`` equal those of the merged file. ``steer``, a steering
        vector file, is added to the residual stream after its layer at ``steer_strength`` (default: the file's);
        that changes the output and the ``system_fingerprint``. ``index``, a document index, grounds every chat in
        the ``index_top`` passages it finds for the last user message, embedded with ``embedding_model`` (default:
        the model the index records); that changes the output and the ``system_fingerprint`` too. ``speculate``
        drafts that many tokens per step (default: 8 with a ``draft_model``, else off) with ``draft_model``, a smaller
        ``model.dllm`` with the same tokenizer, or from the text so far; it never changes the output."""
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
        return DllmEngine(
            model,
            tokenizer,
            fingerprint,
            chat_template=template,
            stop_tokens=stops,
            prompt_cache=prompt_cache,
            embedding=file.embedding,
            retriever=Retriever.open(index, index_top or DEFAULT_TOP, embedding_model) if index else None,
            speculate=speculate or 0,
            draft_model=drafter,
            prompt_cache_dir=prompt_cache_dir,
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
        return self._generator.generate(prompt, max_tokens, options)

    def complete_stream(self, prompt: str, max_tokens: int, options: SamplingOptions) -> Generation:
        return self._generator.stream(prompt, max_tokens, options)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def derive_id(self, prefix: str, payload: Any) -> str:
        """A response id from the request itself (and the weights), so identical requests get identical ids and
        nothing depends on a clock or random source."""
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
        digest = hashlib.sha256(f"{self.system_fingerprint}\n{canonical}".encode()).hexdigest()
        return prefix + digest[:24]

    # -- chat ---------------------------------------------------------------------------------------------------

    def render_chat(self, messages: Iterable[ChatMessage], tools: Sequence[Tool] = ()) -> str:
        """The prompt for a conversation: the model's own chat template when it has one. Tools are presented by
        the template when it supports them, else by Hermes instructions in the system message. A final assistant
        message without tool calls is a prefill: the answer continues its text."""
        messages = list(messages)
        if len(messages) > 1 and messages[-1].role == "assistant" and not messages[-1].tool_calls:
            return self.render_chat(messages[:-1], tools) + messages[-1].content
        source = self.chat_template.source if self.chat_template is not None else None
        uses_tools = bool(tools) or any(m.tool_calls or m.role == "tool" for m in messages)
        if uses_tools and not tooling.template_supports_tools(source):
            messages = tooling.with_instructions(messages, tools)
            tools = ()
        if self.chat_template is None:
            return render(messages)
        return self.chat_template.render(
            tooling.template_messages(messages), tools=[t.to_openai() for t in tools] or None
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
        if tools and choice.mode in ("required", "named"):
            return TokenConstraint(tooling.forced_grammar(tools, choice), self._token_trie())
        if tools and answer is not None:
            either = Grammar.either([tooling.forced_grammar(tools, AUTO), answer])
            return TokenConstraint(either, self._token_trie())
        if tools:
            return TokenConstraint(tooling.call_grammar(tools), self._token_trie(), trigger=TOOL_CALL_OPEN)
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

    def _generate(self, request: ChatRequest, tools: Sequence[Tool]) -> _ChatGeneration:
        constraint = self._constraint(request, tools)
        messages = request.messages
        if self.retriever is not None and request.prompt is None:
            messages, _ = self.retriever.ground(messages)
        prompt = request.prompt if request.prompt is not None else self.render_chat(messages, tools)
        generation = self._generator.stream(
            prompt,
            request.max_tokens,
            request.options,
            stop=request.stop,
            constraint=constraint,
            top_logprobs=request.top_logprobs,
            new_text=request.prompt is None,
        )
        return _ChatGeneration(
            generation.prompt_tokens, generation.cached_tokens, self._events(generation, request, tools)
        )

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
        text = ""
        streamed = ""
        tokens: list[int] = []
        last = None
        for step in generation:
            last = step
            if step.token is not None:
                tokens.append(step.token)
            text += step.text
            if not tools:
                if step.text or step.logprobs is not None:
                    yield TextDelta(step.text, (step.logprobs,) if step.logprobs is not None else ())
                continue
            # With tools, only text that is certainly answer text is streamed: not leading or trailing whitespace,
            # nothing from a <tool_call> on, and nothing at all when the reply starts like a bare JSON call.
            safe = _answer_prefix(text)
            delta = safe[len(streamed) :]
            streamed = safe
            if delta or step.logprobs is not None:
                yield TextDelta(delta, (step.logprobs,) if step.logprobs is not None else ())
        assert last is not None and last.finish_reason is not None
        result = generation.result()
        finish_reason, calls, content = result.finish_reason, [], text
        if tools:
            content, parsed = tooling.parse_calls(text, tools)
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
            result.fingerprint, content, calls, finish_reason, generation.prompt_tokens, len(tokens)
        )
        receipt = receipts.make_receipt(__version__, self.model.id, self.system_fingerprint, request, output)
        yield Finished(finish_reason, stop_sequence, len(tokens), result.fingerprint, receipt)

    def chat_completion(self, request: ChatRequest, *, fresh: bool = False) -> ChatResult:
        stream = self.chat_stream(request, fresh=fresh)
        content: list[str] = []
        calls: list[ToolCall] = []
        logprobs: list[TokenLogprobs] = []
        finished: Finished | None = None
        for event in stream:
            if isinstance(event, TextDelta):
                content.append(event.text)
                logprobs.extend(event.logprobs)
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
        )

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


def _answer_prefix(text: str) -> str:
    """The part of generated text that is certainly answer text when tools are available (see ``_events``)."""
    stripped = text.lstrip()
    if stripped.startswith("{"):
        return ""
    cut = stripped.find(TOOL_CALL_OPEN)
    if cut >= 0:
        stripped = stripped[:cut]
    else:
        for length in range(min(len(TOOL_CALL_OPEN) - 1, len(stripped)), 0, -1):
            if stripped.endswith(TOOL_CALL_OPEN[:length]):
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
    response_cache: str | Path | None = None,
    audit_every: int | None = None,
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
        speculate=configured_speculate(),
        draft_model=os.environ.get(DRAFT_MODEL_ENVIRONMENT_VARIABLE) or None,
        prompt_cache_dir=os.environ.get(PROMPT_CACHE_DIR_ENVIRONMENT_VARIABLE) or None,
    )
