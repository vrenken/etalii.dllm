"""Ollama API: ``/api/chat``, ``/api/generate``, ``/api/embed``, ``/api/embeddings``, ``/api/tags``, ``/api/show``,
``/api/ps`` and ``/api/version``, so tools that speak Ollama (the ``ollama`` Python and JavaScript clients, Open WebUI,
LangChain's ``ChatOllama``, Continue, ...) work against ``dllm-server`` unchanged.

Like the other front ends this is a thin translation over :meth:`DllmEngine.chat_stream`, so a request gives the same
tokens here as on the OpenAI and Anthropic endpoints. Streaming (the default, as in Ollama) sends newline-delimited
JSON; the non-streamed response is assembled from the same events.

Differences from Ollama: one model is served whatever ``model`` names; ``temperature`` defaults to 0 (greedy) as on
every endpoint of this server (Ollama: 0.8); ``created_at`` and ``modified_at`` are the Unix epoch and every
``*_duration`` is 0, because responses carry nothing clock-derived; images, ``suffix``, ``template``, ``context``
and model management (pull, push, create, copy, delete) are not supported.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from etalii_dllm import __version__
from etalii_dllm.chat import ChatMessage, ToolCall
from etalii_dllm.engine import (
    MODEL_ENVIRONMENT_VARIABLE,
    ChatRequest,
    ChatStream,
    DllmEngine,
    Finished,
    ReasoningDelta,
    ResponseFormat,
    TextDelta,
    ToolCallEvent,
    default_engine,
)
from etalii_dllm.generation import TokenLogprobs
from etalii_dllm.sampling import SamplingOptions
from etalii_dllm.server.contracts import id_payload
from etalii_dllm.tools import Tool, ToolChoice

Engine = Annotated[DllmEngine, Depends(default_engine)]

router = APIRouter()

EPOCH = "1970-01-01T00:00:00Z"
"""``created_at``/``modified_at`` of every response: no clock-derived values."""
DEFAULT_NUM_PREDICT = 2048
"""Tokens generated when ``num_predict`` is unset or negative (Ollama: until the context is full), unless the model's
context is shorter."""


# -- wire format ----------------------------------------------------------------------------------------------------


class Options(BaseModel):
    temperature: float | None = None
    seed: int | None = None
    top_k: int | None = None
    top_p: float | None = None
    num_predict: int | None = None
    stop: list[str] | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    repeat_last_n: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    watermark_key: str | None = None
    """Extension: watermark the answer with this key (docs/watermarks.md)."""
    watermark_gamma: float | None = None
    watermark_delta: float | None = None


class OllamaFunctionCall(BaseModel):
    name: str
    arguments: dict[str, Any] | str = Field(default_factory=dict)


class OllamaToolCall(BaseModel):
    function: OllamaFunctionCall


class OllamaMessage(BaseModel):
    role: str
    content: str | None = None
    images: list[Any] | None = None
    tool_calls: list[OllamaToolCall] | None = None
    tool_name: str | None = None
    name: str | None = None


class OllamaFunction(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class OllamaTool(BaseModel):
    type: str = "function"
    function: OllamaFunction


class _GenerateBase(BaseModel):
    model: str = ""
    stream: bool = True
    format: Literal["", "json"] | dict[str, Any] | None = None
    options: Options | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    receipt: bool | None = None
    """Extension: add a ``receipt`` to the final object (see docs/receipts.md)."""
    previous_receipt: str | None = None
    """Extension: the receipt id of the conversation's previous turn, recorded as the new receipt's ``previous``."""
    truncate: bool | None = None
    """Drop the oldest messages that do not fit the context window (off unless asked for)."""
    shift: bool | None = None
    """Keep generating past a full context window on a rolled context (off unless asked for)."""
    think: bool | Literal["low", "medium", "high"] | None = None
    """For thinking models: ``false`` switches thinking off, ``true`` or an effort on (docs/api.md#reasoning)."""
    max_reasoning_tokens: int | None = None
    """Extension: the most tokens a thinking model's ``<think>`` block may take."""
    token_healing: bool | None = None
    """Extension: take the prompt's last token back and make the answer start with it (docs/api.md#token-healing)."""


class ChatBody(_GenerateBase):
    messages: list[OllamaMessage] = Field(default_factory=list)
    tools: list[OllamaTool] | None = None


class GenerateBody(_GenerateBase):
    prompt: str | None = None
    system: str | None = None
    raw: bool = False
    images: list[Any] | None = None
    suffix: str | None = None
    template: str | None = None
    context: list[int] | None = None


class EmbedBody(BaseModel):
    model: str = ""
    input: str | list[str] = ""
    dimensions: int | None = None


class EmbeddingsBody(BaseModel):
    model: str = ""
    prompt: str = ""


class ShowBody(BaseModel):
    model: str = ""


def error(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


# -- translation ----------------------------------------------------------------------------------------------------


def _options(body: _GenerateBase, engine: DllmEngine) -> tuple[SamplingOptions, int, list[str]]:
    options = body.options or Options()
    sampling = SamplingOptions(
        temperature=options.temperature if options.temperature is not None else 0.0,
        top_k=options.top_k or 0,
        top_p=options.top_p if options.top_p is not None else 1.0,
        seed=options.seed or 0,
        min_p=options.min_p or 0.0,
        repetition_penalty=options.repeat_penalty if options.repeat_penalty is not None else 1.0,
        repeat_last_n=options.repeat_last_n if options.repeat_last_n is not None else 64,
        frequency_penalty=options.frequency_penalty or 0.0,
        presence_penalty=options.presence_penalty or 0.0,
        watermark_key=options.watermark_key,
        watermark_gamma=options.watermark_gamma if options.watermark_gamma is not None else 0.25,
        watermark_delta=options.watermark_delta if options.watermark_delta is not None else 2.0,
    )
    context = getattr(getattr(engine.model, "config", None), "context_length", 0) or DEFAULT_NUM_PREDICT
    limit = min(DEFAULT_NUM_PREDICT, context)
    max_tokens = options.num_predict if options.num_predict is not None and options.num_predict >= 0 else limit
    return sampling, max_tokens, list(options.stop or ())


def _format(value: str | dict[str, Any] | None) -> ResponseFormat:
    if not value:
        return ResponseFormat()
    if value == "json":
        return ResponseFormat("json_object")
    assert isinstance(value, dict)
    return ResponseFormat("json_schema", value)


def _top_logprobs(body: _GenerateBase) -> int | None:
    if not body.logprobs:
        return None
    return body.top_logprobs or 0


def _messages(messages: list[OllamaMessage]) -> list[ChatMessage]:
    """Ollama tool calls carry no ids; they get ``call_<n>`` in conversation order, and each tool result answers
    the earliest unanswered call to the tool it names (else the earliest unanswered call)."""
    result: list[ChatMessage] = []
    pending: list[ToolCall] = []
    count = 0
    for message in messages:
        if message.images:
            raise ValueError("images are not supported")
        role = message.role
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"unknown message role {role!r}")
        if role == "tool":
            name = message.tool_name or message.name
            match = next((c for c in pending if c.name == name), pending[0] if pending else None)
            if match is not None:
                pending.remove(match)
            call_id = match.id if match is not None else ""
            result.append(ChatMessage("tool", message.content or "", tool_call_id=call_id, name=name or None))
            continue
        calls = []
        for call in (message.tool_calls or ()) if role == "assistant" else ():
            arguments = call.function.arguments
            text = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
            calls.append(ToolCall(f"call_{count}", call.function.name, text))
            count += 1
        pending.extend(calls)
        result.append(ChatMessage(role, message.content or "", tuple(calls)))
    return result


def _tools(tools: list[OllamaTool] | None) -> list[Tool]:
    return [Tool(t.function.name, t.function.description or "", t.function.parameters or {}) for t in tools or ()]


def _chat_request(body: ChatBody, engine: DllmEngine) -> ChatRequest:
    if not body.messages:
        raise ValueError("'messages' must contain at least one message")
    options, max_tokens, stop = _options(body, engine)
    request_id = engine.derive_id(
        "ollama-", id_payload(body.model_dump(mode="json", exclude={"stream", "receipt", "previous_receipt"}))
    )
    return ChatRequest(
        messages=_messages(body.messages),
        max_tokens=max_tokens,
        options=options,
        stop=stop,
        tools=_tools(body.tools),
        tool_choice=ToolChoice("auto"),
        response_format=_format(body.format),
        top_logprobs=_top_logprobs(body),
        request_id=request_id,
        previous_receipt=body.previous_receipt,
        truncation="auto" if body.truncate else "disabled",
        context_overflow="roll" if body.shift else "stop",
        thinking=None if body.think is None else body.think is not False,
        max_reasoning_tokens=body.max_reasoning_tokens,
        token_healing=bool(body.token_healing),
    )


def _generate_request(body: GenerateBody, engine: DllmEngine) -> ChatRequest:
    if body.images:
        raise ValueError("images are not supported")
    if body.suffix or body.template or body.context:
        raise ValueError("suffix, template and context are not supported")
    options, max_tokens, stop = _options(body, engine)
    messages = [ChatMessage("system", body.system)] if body.system else []
    messages.append(ChatMessage("user", body.prompt or ""))
    request_id = engine.derive_id(
        "ollama-", id_payload(body.model_dump(mode="json", exclude={"stream", "receipt", "previous_receipt"}))
    )
    return ChatRequest(
        messages=messages,
        max_tokens=max_tokens,
        options=options,
        stop=stop,
        response_format=_format(body.format),
        top_logprobs=_top_logprobs(body),
        request_id=request_id,
        previous_receipt=body.previous_receipt,
        prompt=(body.prompt or "") if body.raw else None,
        truncation="auto" if body.truncate else "disabled",
        context_overflow="roll" if body.shift else "stop",
        thinking=None if body.think is None else body.think is not False,
        max_reasoning_tokens=body.max_reasoning_tokens,
        token_healing=bool(body.token_healing),
    )


def _logprobs(engine: DllmEngine, entries: tuple[TokenLogprobs, ...]) -> list[dict[str, Any]]:
    def text(token: int) -> str:
        return engine.tokenizer.decode_bytes([token]).decode("utf-8", errors="replace")

    return [
        {
            "token": text(entry.token),
            "logprob": entry.logprob,
            "top_logprobs": [{"token": text(alt.token), "logprob": alt.logprob} for alt in entry.top],
        }
        for entry in entries
    ]


def _arguments(call: ToolCall) -> dict[str, Any]:
    value = call.arguments_object()
    return value if isinstance(value, dict) else {}


def _final(engine: DllmEngine, stream: ChatStream, event: Finished) -> dict[str, Any]:
    return {
        "done": True,
        "done_reason": "length" if event.finish_reason == "length" else "stop",
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": stream.prompt_tokens - stream.cached_tokens,
        "prompt_eval_duration": 0,
        "eval_count": event.completion_tokens,
        "eval_duration": 0,
    }


def _chunks(
    engine: DllmEngine, stream: ChatStream, chat: bool, logprobs: bool, receipt: bool = False
) -> Iterator[dict[str, Any]]:
    """The response as Ollama streams it: one object per text delta or tool call, then the final one."""
    head = {"model": engine.model.id, "created_at": EPOCH}
    for event in stream:
        if isinstance(event, TextDelta):
            body: dict[str, Any] = (
                {"message": {"role": "assistant", "content": event.text}} if chat else {"response": event.text}
            )
            if logprobs:
                body["logprobs"] = _logprobs(engine, event.logprobs)
            yield {**head, **body, "done": False}
        elif isinstance(event, ReasoningDelta):
            if chat:
                yield {**head, "message": {"role": "assistant", "content": "", "thinking": event.text}, "done": False}
            else:
                yield {**head, "response": "", "thinking": event.text, "done": False}
        elif isinstance(event, ToolCallEvent):
            call = {"function": {"name": event.call.name, "arguments": _arguments(event.call)}}
            yield {**head, "message": {"role": "assistant", "content": "", "tool_calls": [call]}, "done": False}
        elif isinstance(event, Finished):
            body = {"message": {"role": "assistant", "content": ""}} if chat else {"response": ""}
            final = {**head, **body, **_final(engine, stream, event)}
            yield {**final, "receipt": event.receipt} if receipt else final


def _collect(chunks: Iterator[dict[str, Any]], chat: bool) -> dict[str, Any]:
    """The non-streamed response: the streamed objects merged."""
    text: list[str] = []
    thinking: list[str] | None = None
    calls: list[dict[str, Any]] = []
    logprobs: list[dict[str, Any]] = []
    final: dict[str, Any] = {}
    for chunk in chunks:
        if chunk["done"]:
            final = chunk
            continue
        part = chunk["message"].get("thinking") if chat else chunk.get("thinking")
        if part is not None:
            thinking = [*(thinking or []), part]
            continue
        if chat:
            text.append(chunk["message"]["content"])
            calls.extend(chunk["message"].get("tool_calls", ()))
        else:
            text.append(chunk["response"])
        logprobs.extend(chunk.get("logprobs", ()))
    if chat:
        message: dict[str, Any] = {"role": "assistant", "content": "".join(text)}
        if thinking is not None:
            message["thinking"] = "".join(thinking)
        if calls:
            message["tool_calls"] = calls
        final["message"] = message
    else:
        final["response"] = "".join(text)
        if thinking is not None:
            final["thinking"] = "".join(thinking)
    if logprobs:
        final["logprobs"] = logprobs
    return final


def _respond(chunks: Iterator[dict[str, Any]], stream: bool, chat: bool) -> JSONResponse | StreamingResponse:
    if stream:
        lines = (json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks)
        return StreamingResponse(lines, media_type="application/x-ndjson")
    return JSONResponse(_collect(chunks, chat))


# -- endpoints ------------------------------------------------------------------------------------------------------


@router.post("/api/chat", response_model=None)
def chat(body: ChatBody, engine: Engine) -> JSONResponse | StreamingResponse:
    try:
        request = _chat_request(body, engine)
        stream = engine.chat_stream(request)
    except ValueError as problem:
        return error(str(problem))
    return _respond(
        _chunks(engine, stream, True, request.top_logprobs is not None, bool(body.receipt)), body.stream, True
    )


@router.post("/api/generate", response_model=None)
def generate(body: GenerateBody, engine: Engine) -> JSONResponse | StreamingResponse:
    if not body.prompt and not body.raw:  # Ollama's "load the model" request
        return JSONResponse({"model": engine.model.id, "created_at": EPOCH, "response": "", "done": True,
                             "done_reason": "load"})  # fmt: skip
    try:
        request = _generate_request(body, engine)
        stream = engine.chat_stream(request)
    except ValueError as problem:
        return error(str(problem))
    return _respond(
        _chunks(engine, stream, False, request.top_logprobs is not None, bool(body.receipt)), body.stream, False
    )


def _embeddings(engine: DllmEngine, texts: list[str], dimensions: int | None) -> tuple[list[list[float]], int]:
    vectors, tokens = [], 0
    for text in texts:
        embedding = engine.embed(text, dimensions)
        vectors.append([float(v) for v in embedding.vector])
        tokens += embedding.tokens
    return vectors, tokens


@router.post("/api/embed", response_model=None)
def embed(body: EmbedBody, engine: Engine) -> JSONResponse:
    texts = [body.input] if isinstance(body.input, str) else list(body.input)
    try:
        vectors, tokens = _embeddings(engine, texts, body.dimensions)
    except ValueError as problem:
        return error(str(problem))
    return JSONResponse({"model": engine.model.id, "embeddings": vectors, "total_duration": 0, "load_duration": 0,
                         "prompt_eval_count": tokens})  # fmt: skip


@router.post("/api/embeddings", response_model=None)
def embeddings(body: EmbeddingsBody, engine: Engine) -> JSONResponse:
    try:
        vectors, _ = _embeddings(engine, [body.prompt], None)
    except ValueError as problem:
        return error(str(problem))
    return JSONResponse({"embedding": vectors[0]})


def _details(engine: DllmEngine) -> dict[str, Any]:
    config = getattr(engine.model, "config", None)
    quantization = getattr(engine.model, "quantization", None)
    details: dict[str, Any] = {
        "parent_model": "",
        "format": "dllm",
        "family": config.family if config is not None else "bigram",
        "families": [config.family] if config is not None else ["bigram"],
        "parameter_size": "",
        "quantization_level": (quantization or "f32").upper(),
    }
    if config is not None:
        parameters = sum(_count(shape) for shape in config.tensor_shapes().values())
        details["parameter_size"] = _parameter_size(parameters)
    return details


def _count(shape: tuple[int, ...]) -> int:
    total = 1
    for size in shape:
        total *= size
    return total


def _parameter_size(parameters: int) -> str:
    for unit, scale in (("B", 10**9), ("M", 10**6), ("K", 10**3)):
        if parameters >= scale:
            return f"{parameters / scale:.1f}{unit}".replace(".0", "")
    return str(parameters)


def _entry(engine: DllmEngine) -> dict[str, Any]:
    path = os.environ.get(MODEL_ENVIRONMENT_VARIABLE)
    size = os.path.getsize(path) if path and os.path.exists(path) else 0
    weights = getattr(engine.model, "weights_fingerprint", "") or engine.system_fingerprint
    return {
        "name": engine.model.id,
        "model": engine.model.id,
        "modified_at": EPOCH,
        "size": size,
        "digest": weights,
        "details": _details(engine),
    }


@router.get("/api/tags")
def tags(engine: Engine) -> dict[str, Any]:
    return {"models": [_entry(engine)]}


@router.get("/api/ps")
def running(engine: Engine) -> dict[str, Any]:
    config = getattr(engine.model, "config", None)
    entry = {**_entry(engine), "expires_at": EPOCH, "size_vram": 0}
    if config is not None:
        entry["context_length"] = config.context_length
    return {"models": [entry]}


@router.post("/api/show")
def show(body: ShowBody, engine: Engine) -> dict[str, Any]:
    config = getattr(engine.model, "config", None)
    info: dict[str, Any] = {"general.architecture": config.family if config is not None else "bigram"}
    if config is not None:
        family = config.family
        info.update({
            f"{family}.context_length": config.context_length,
            f"{family}.embedding_length": config.hidden_size,
            f"{family}.block_count": config.layers,
            f"{family}.attention.head_count": config.heads,
            f"{family}.attention.head_count_kv": config.kv_heads,
            f"{family}.feed_forward_length": config.intermediate_size,
            f"{family}.vocab_size": config.vocabulary_size,
        })  # fmt: skip
    capabilities = ["completion", "tools"]
    if hasattr(engine.model, "hidden_states"):
        capabilities.append("embedding")
    template = engine.chat_template.source if engine.chat_template is not None else ""
    return {
        "modelfile": "",
        "parameters": "",
        "template": template,
        "details": _details(engine),
        "model_info": info,
        "capabilities": capabilities,
        "modified_at": EPOCH,
    }


@router.get("/api/version")
def version() -> dict[str, str]:
    return {"version": __version__}
