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
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from etalii_dllm import tools as tooling
from etalii_dllm.chat import TOOL_CALL_OPEN, ChatMessage, ToolCall, render
from etalii_dllm.chat_template import ChatTemplate
from etalii_dllm.cuda import DEVICES
from etalii_dllm.generation import Generation, GenerationResult, Generator, TokenLogprobs
from etalii_dllm.grammar import Grammar, TokenConstraint, TokenTrie
from etalii_dllm.models import BigramModel, LanguageModel
from etalii_dllm.numerics import QUANTIZATIONS, linear, set_threads, sum_squares
from etalii_dllm.sampling import GREEDY, SamplingOptions
from etalii_dllm.tokenization import ByteTokenizer, Tokenizer
from etalii_dllm.tools import AUTO, Tool, ToolChoice

DEFAULT_MODEL_SEED = 42
MODEL_ENVIRONMENT_VARIABLE = "DLLM_MODEL"
"""Path of a ``model.dllm`` file for the front ends to serve; the placeholder bigram model when unset."""
QUANTIZE_ENVIRONMENT_VARIABLE = "DLLM_QUANTIZE"
"""Weight quantisation for the served model (``q8_0``); float32 weights when unset or ``none``."""
DEVICE_ENVIRONMENT_VARIABLE = "DLLM_DEVICE"
"""Where the served model runs: ``cpu`` (default) or ``cuda``. Never changes the output."""


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


@dataclass
class ChatStream:
    """A chat generation in progress: ``prompt_tokens`` is known up front; iterate for the events."""

    prompt_tokens: int
    events: Iterator[ChatEvent] = field(repr=False)

    def __iter__(self) -> Iterator[ChatEvent]:
        return self.events


@dataclass(frozen=True)
class Embedding:
    vector: np.ndarray
    tokens: int


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
        self._trie: TokenTrie | None = None

    @staticmethod
    def from_model_file(
        path: str | Path, verify: bool = True, quantize: str | None = None, device: str = "cpu"
    ) -> DllmEngine:
        """An engine for an imported ``model.dllm``: its decoder, BPE tokenizer and chat template. ``quantize``
        (``"q8_0"``) runs the linear layers on quantised weights; that changes the output, and so the
        ``system_fingerprint``. ``device="cuda"`` runs the decoder on the GPU with the same output."""
        from etalii_dllm.bpe import from_model_header, special_token_text
        from etalii_dllm.modelfile import ModelFile
        from etalii_dllm.transformer import Transformer

        file = ModelFile(path, verify=verify)
        if file.tokenizer is None:
            raise ValueError(f"{path}: the model file has no tokenizer")
        model_id = str(file.source.get("repository") or Path(path).stem)
        model = Transformer(
            file.config,
            file.tensors,
            model_id=model_id,
            weights_fingerprint=file.fingerprint,
            quantize=quantize,
            device=device,
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
        return DllmEngine(model, tokenizer, fingerprint, chat_template=template, stop_tokens=stops)

    @staticmethod
    def create_default() -> DllmEngine:
        tokenizer = ByteTokenizer()
        model = BigramModel(tokenizer.vocabulary_size, DEFAULT_MODEL_SEED)
        return DllmEngine(model, tokenizer, "fp_" + model.weights_fingerprint[:12])

    # -- text ---------------------------------------------------------------------------------------------------

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

    def chat_stream(self, request: ChatRequest) -> ChatStream:
        """Starts a chat generation. Raises ``ValueError`` for invalid requests (unknown tools, unsupported
        schemas, ...) before any token is generated."""
        tools = [] if request.tool_choice.mode == "none" else list(request.tools)
        tooling.validate_tools(tools, request.tool_choice)
        constraint = self._constraint(request, tools)
        prompt = self.render_chat(request.messages, tools)
        generation = self._generator.stream(
            prompt,
            request.max_tokens,
            request.options,
            stop=request.stop,
            constraint=constraint,
            top_logprobs=request.top_logprobs,
        )
        return ChatStream(generation.prompt_tokens, self._events(generation, request, tools))

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
        finish_reason, calls = result.finish_reason, []
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
        yield Finished(finish_reason, stop_sequence, len(tokens), result.fingerprint)

    def chat_completion(self, request: ChatRequest) -> ChatResult:
        stream = self.chat_stream(request)
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
        )

    # -- embeddings ---------------------------------------------------------------------------------------------

    def embed(self, text: str | Sequence[int], dimensions: int | None = None) -> Embedding:
        """Mean of the final hidden states over all positions (each column summed over positions ascending in
        double, through the ``linear`` kernel), L2-normalised. ``dimensions`` keeps the first components and
        normalises again."""
        tokens = self.tokenizer.encode(text) if isinstance(text, str) else list(text)
        if not tokens:
            raise ValueError("cannot embed an empty input")
        hidden_states = getattr(self.model, "hidden_states", None)
        if hidden_states is None:
            raise ValueError(f"model {self.model.id} does not provide hidden states")
        states = np.asarray(hidden_states(tokens), dtype=np.float32)
        ones = np.ones((1, states.shape[0]), dtype=np.float32)
        total = linear(ones, np.ascontiguousarray(states.T)).numpy().reshape(-1)
        vector = (total / np.float32(states.shape[0])).astype(np.float32)
        if dimensions is not None:
            if not 1 <= dimensions <= vector.shape[0]:
                raise ValueError(f"dimensions must be between 1 and {vector.shape[0]}")
            vector = vector[:dimensions]
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
    path: str | Path | None, quantize: str | None = None, threads: int | None = None, device: str | None = None
) -> None:
    """Makes the front ends serve ``path`` (sets ``DLLM_MODEL``, and ``DLLM_QUANTIZE``/``DLLM_DEVICE`` when
    ``quantize``/``device`` are given) and resets the default engine. ``threads`` sets the kernel thread count; it
    and ``device`` never change the output."""
    if path:
        os.environ[MODEL_ENVIRONMENT_VARIABLE] = str(path)
    if quantize:
        os.environ[QUANTIZE_ENVIRONMENT_VARIABLE] = quantize
    if device:
        os.environ[DEVICE_ENVIRONMENT_VARIABLE] = device
    if threads is not None:
        set_threads(threads)
    default_engine.cache_clear()


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    """The ``--model``, ``--quantize``, ``--threads`` and ``--device`` options every front end shares."""
    parser.add_argument("--model", help="model.dllm file to use (default: $DLLM_MODEL, else the placeholder model)")
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


@cache
def default_engine() -> DllmEngine:
    """Process-wide default engine: the model named by ``DLLM_MODEL``, else the placeholder. Models are immutable
    and ``forward`` keeps no shared state, so sharing the engine between requests is safe."""
    path = os.environ.get(MODEL_ENVIRONMENT_VARIABLE)
    if not path:
        return DllmEngine.create_default()
    return DllmEngine.from_model_file(path, quantize=configured_quantization(), device=configured_device())
