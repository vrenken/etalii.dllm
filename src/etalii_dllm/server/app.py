"""HTTP API: OpenAI-compatible ``GET /v1/models``, ``POST /v1/chat/completions`` (with streaming, tools, structured
output and logprobs) and ``POST /v1/embeddings``; Anthropic-compatible ``POST /v1/messages`` (see
:mod:`etalii_dllm.server.anthropic_api`); a chat page over the streamed chat completions at ``/``.

The handlers only translate between wire formats and :class:`~etalii_dllm.engine.ChatRequest`; all behaviour lives
in the engine, so every front end gives the same output. Response ids are derived from the request (and the
weights), timestamps are 0: identical requests get byte-identical responses, streamed or not.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Iterator
from importlib import resources
from typing import Annotated

import numpy as np
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from etalii_dllm import __version__
from etalii_dllm.chat import ChatMessage, ToolCall
from etalii_dllm.engine import (
    ChatRequest,
    DllmEngine,
    Finished,
    ResponseFormat,
    TextDelta,
    ToolCallEvent,
    add_runtime_arguments,
    default_engine,
    use_model_file,
)
from etalii_dllm.generation import TokenLogprobs
from etalii_dllm.sampling import SamplingOptions
from etalii_dllm.server import anthropic_api, ollama_api
from etalii_dllm.server.contracts import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChoiceLogprobs,
    ChunkChoice,
    ChunkDelta,
    DeltaFunctionCall,
    DeltaToolCall,
    EmbeddingData,
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingsUsage,
    FunctionCall,
    LogprobEntry,
    ModelInfo,
    ModelList,
    PromptTokensDetails,
    ToolCallModel,
    TopLogprob,
)
from etalii_dllm.tools import Tool, ToolChoice

Engine = Annotated[DllmEngine, Depends(default_engine)]

DEFAULT_MAX_TOKENS = 64

app = FastAPI(title="EtAlii.Dllm", version=__version__)
app.include_router(anthropic_api.router)
app.include_router(ollama_api.router)


def _error(message: str) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": {"message": message, "type": "invalid_request_error"}})


@app.exception_handler(RequestValidationError)
def _validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
    message = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in error.errors())
    if request.url.path.startswith("/v1/messages"):
        return anthropic_api.error(message)
    if request.url.path.startswith("/api/"):
        return ollama_api.error(message)
    return _error(message)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def chat_page() -> HTMLResponse:
    """A small chat page (static/chat.html, no external resources) over the streamed /v1/chat/completions."""
    return HTMLResponse(resources.files("etalii_dllm.server").joinpath("static/chat.html").read_text(encoding="utf-8"))


@app.get("/v1/models")
def list_models(engine: Engine) -> ModelList:
    return ModelList(data=[ModelInfo(id=engine.model.id)])


def _text(content: str | list | None) -> str:
    if content is None or isinstance(content, str):
        return content or ""
    for part in content:
        if part.type != "text":
            raise ValueError(f"content parts of type {part.type!r} are not supported")
    return "".join(part.text or "" for part in content)


def _message(message: ChatCompletionMessage) -> ChatMessage:
    role = "system" if message.role == "developer" else message.role
    if role not in ("system", "user", "assistant", "tool"):
        raise ValueError(f"unknown message role {message.role!r}")
    calls = tuple(ToolCall(c.id, c.function.name, c.function.arguments) for c in message.tool_calls or ())
    return ChatMessage(role, _text(message.content), calls, message.tool_call_id, message.name)


def _chat_request(request: ChatCompletionRequest, engine: DllmEngine) -> ChatRequest:
    """Translates the wire request; raises ``ValueError`` for invalid ones."""
    if not request.messages:
        raise ValueError("'messages' must contain at least one message.")
    if request.n not in (None, 1):
        raise ValueError("only n=1 is supported")
    options = SamplingOptions(
        temperature=request.temperature if request.temperature is not None else 0.0,
        top_k=request.top_k or 0,
        top_p=request.top_p if request.top_p is not None else 1.0,
        seed=request.seed or 0,
    )
    functions = [t.function for t in request.tools or ()]
    tools = [Tool(f.name, f.description or "", f.parameters or {}) for f in functions]
    if request.tool_choice is None:
        choice = ToolChoice("auto")
    elif isinstance(request.tool_choice, str):
        choice = ToolChoice(request.tool_choice)
    else:
        choice = ToolChoice("named", request.tool_choice.function.name)
    response_format = ResponseFormat()
    if request.response_format is not None and request.response_format.type != "text":
        spec = request.response_format.json_schema
        if request.response_format.type == "json_schema" and (spec is None or spec.json_schema is None):
            raise ValueError("response_format json_schema needs a 'json_schema.schema'")
        schema = spec.json_schema if request.response_format.type == "json_schema" and spec else None
        response_format = ResponseFormat(request.response_format.type, schema)
    stop = [request.stop] if isinstance(request.stop, str) else list(request.stop or [])
    if len(stop) > 4:
        raise ValueError("at most 4 stop sequences are supported")
    request_id = engine.derive_id("chatcmpl-", request.model_dump(mode="json", exclude={"stream", "stream_options"}))
    return ChatRequest(
        messages=[_message(m) for m in request.messages],
        max_tokens=request.max_completion_tokens or request.max_tokens or DEFAULT_MAX_TOKENS,
        options=options,
        stop=stop,
        tools=tools,
        tool_choice=choice,
        response_format=response_format,
        top_logprobs=(request.top_logprobs or 0) if request.logprobs else None,
        call_id_prefix="call_",
        request_id=request_id,
    )


def _token(engine: DllmEngine, token: int) -> tuple[str, list[int]]:
    data = engine.tokenizer.decode_bytes([token])
    return data.decode("utf-8", errors="replace"), list(data)


def _logprobs(engine: DllmEngine, entries: tuple[TokenLogprobs, ...]) -> ChoiceLogprobs:
    content = []
    for entry in entries:
        text, data = _token(engine, entry.token)
        top = [TopLogprob(token=t, logprob=a.logprob, bytes=b) for a in entry.top for t, b in [_token(engine, a.token)]]
        content.append(LogprobEntry(token=text, logprob=entry.logprob, bytes=data, top_logprobs=top))
    return ChoiceLogprobs(content=content)


@app.post("/v1/chat/completions", response_model=None)
def chat_completions(
    request: ChatCompletionRequest, engine: Engine
) -> ChatCompletionResponse | JSONResponse | StreamingResponse:
    try:
        chat = _chat_request(request, engine)
        if request.stream:
            stream = engine.chat_stream(chat)
            include_usage = bool(request.stream_options and request.stream_options.include_usage)
            return StreamingResponse(
                _chunks(engine, chat, stream, include_usage, chat.top_logprobs is not None),
                media_type="text/event-stream",
            )
        result = engine.chat_completion(chat)
    except ValueError as error:
        return _error(str(error))

    calls = [
        ToolCallModel(id=c.id, function=FunctionCall(name=c.name, arguments=c.arguments)) for c in result.tool_calls
    ]
    return ChatCompletionResponse(
        id=chat.request_id,
        created=0,
        model=engine.model.id,
        system_fingerprint=engine.system_fingerprint,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=AssistantMessage(content=result.content or (None if calls else ""), tool_calls=calls or None),
                logprobs=_logprobs(engine, result.logprobs) if chat.top_logprobs is not None else None,
                finish_reason=result.finish_reason,
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=result.cached_tokens),
        ),
    )


def _sse(chunk: ChatCompletionChunk) -> str:
    return f"data: {chunk.model_dump_json()}\n\n"


def _chunks(engine: DllmEngine, chat: ChatRequest, stream, include_usage: bool, logprobs: bool) -> Iterator[str]:
    def chunk(delta: ChunkDelta, **fields) -> ChatCompletionChunk:
        return ChatCompletionChunk(
            id=chat.request_id,
            model=engine.model.id,
            system_fingerprint=engine.system_fingerprint,
            choices=[ChunkChoice(delta=delta, **fields)],
        )

    yield _sse(chunk(ChunkDelta(role="assistant", content="")))
    for event in stream:
        if isinstance(event, TextDelta):
            entries = _logprobs(engine, event.logprobs) if logprobs else None
            yield _sse(chunk(ChunkDelta(content=event.text), logprobs=entries))
        elif isinstance(event, ToolCallEvent):
            call = DeltaToolCall(
                index=event.index,
                id=event.call.id,
                type="function",
                function=DeltaFunctionCall(name=event.call.name, arguments=event.call.arguments),
            )
            yield _sse(chunk(ChunkDelta(tool_calls=[call])))
        elif isinstance(event, Finished):
            yield _sse(chunk(ChunkDelta(), finish_reason=event.finish_reason))
            if include_usage:
                usage = ChatCompletionUsage(
                    prompt_tokens=stream.prompt_tokens,
                    completion_tokens=event.completion_tokens,
                    total_tokens=stream.prompt_tokens + event.completion_tokens,
                    prompt_tokens_details=PromptTokensDetails(cached_tokens=stream.cached_tokens),
                )
                final = ChatCompletionChunk(
                    id=chat.request_id,
                    model=engine.model.id,
                    system_fingerprint=engine.system_fingerprint,
                    choices=[],
                    usage=usage,
                )
                yield _sse(final)
    yield "data: [DONE]\n\n"


@app.post("/v1/embeddings", response_model=None)
def embeddings(request: EmbeddingsRequest, engine: Engine) -> EmbeddingsResponse | JSONResponse:
    inputs = request.input
    if isinstance(inputs, str) or (inputs and all(isinstance(i, int) for i in inputs)):
        inputs = [inputs]  # type: ignore[list-item]
    if not inputs:
        return _error("'input' must not be empty")
    data = []
    tokens = 0
    try:
        for index, item in enumerate(inputs):
            embedding = engine.embed(item, request.dimensions)  # type: ignore[arg-type]
            tokens += embedding.tokens
            vector = embedding.vector
            if request.encoding_format == "base64":
                encoded: list[float] | str = base64.b64encode(vector.astype("<f4").tobytes()).decode("ascii")
            else:
                encoded = [float(v) for v in vector.astype(np.float32)]
            data.append(EmbeddingData(index=index, embedding=encoded))
    except ValueError as error:
        return _error(str(error))
    return EmbeddingsResponse(
        data=data, model=engine.model.id, usage=EmbeddingsUsage(prompt_tokens=tokens, total_tokens=tokens)
    )


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="EtAlii.Dllm OpenAI-, Anthropic- and Ollama-compatible server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5080)
    add_runtime_arguments(parser)
    args = parser.parse_args()
    use_model_file(args.model, args.quantize, args.threads, args.device, args.prompt_cache)
    uvicorn.run(app, host=args.host, port=args.port)
