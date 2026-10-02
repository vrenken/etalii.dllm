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
from typing import Annotated, Any

import numpy as np
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from etalii_dllm import __version__, beam, receipts, scoring, voting
from etalii_dllm.chat import ChatMessage, ToolCall
from etalii_dllm.engine import (
    MAX_CHOICES,
    ChatRequest,
    DllmEngine,
    Finished,
    ReasoningDelta,
    ResponseFormat,
    TextDelta,
    ToolCallEvent,
    add_runtime_arguments,
    default_engine,
    use_model_file,
)
from etalii_dllm.generation import ContextLengthError, TokenLogprobs
from etalii_dllm.sampling import SamplingOptions
from etalii_dllm.server import anthropic_api, batches_api, completions_api, ollama_api, responses_api
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
    RerankDocument,
    RerankRequest,
    RerankResponse,
    RerankResult,
    RerankUsage,
    ToolCallModel,
    TopLogprob,
    WatermarkDetectRequest,
    guided,
    id_payload,
    thinking_switch,
)
from etalii_dllm.tools import Tool, ToolChoice

Engine = Annotated[DllmEngine, Depends(default_engine)]

DEFAULT_MAX_TOKENS = 64

app = FastAPI(title="EtAlii.Dllm", version=__version__)
app.include_router(anthropic_api.router)
app.include_router(ollama_api.router)
app.include_router(responses_api.router)
app.include_router(batches_api.router)
app.include_router(completions_api.router)


CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"


def _error(message: str, code: str | None = None) -> JSONResponse:
    content: dict = {"message": message, "type": "invalid_request_error"}
    if code is not None:  # OpenAI's code for a prompt that does not fit, which clients match on
        content.update(param="messages", code=code)
    return JSONResponse(status_code=400, content={"error": content})


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
    options = SamplingOptions(
        temperature=request.temperature if request.temperature is not None else 0.0,
        top_k=request.top_k or 0,
        top_p=request.top_p if request.top_p is not None else 1.0,
        seed=request.seed or 0,
        min_p=request.min_p or 0.0,
        repetition_penalty=request.repetition_penalty if request.repetition_penalty is not None else 1.0,
        repeat_last_n=request.repeat_last_n if request.repeat_last_n is not None else 64,
        frequency_penalty=request.frequency_penalty or 0.0,
        presence_penalty=request.presence_penalty or 0.0,
        logit_bias=SamplingOptions.bias(request.logit_bias),
        **(request.watermark.sampling() if request.watermark else {}),
        **guided(request.guidance, request.contrast),
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
    if request.guided_regex is not None:
        if request.response_format is not None and request.response_format.type != "text":
            raise ValueError("guided_regex cannot be combined with a response_format")
        response_format = ResponseFormat("regex", pattern=request.guided_regex)
    elif request.response_format is not None and request.response_format.type == "regex":
        if request.response_format.regex is None:
            raise ValueError("response_format regex needs a 'regex'")
        response_format = ResponseFormat("regex", pattern=request.response_format.regex)
    elif request.response_format is not None and request.response_format.type != "text":
        spec = request.response_format.json_schema
        if request.response_format.type == "json_schema" and (spec is None or spec.json_schema is None):
            raise ValueError("response_format json_schema needs a 'json_schema.schema'")
        schema = spec.json_schema if request.response_format.type == "json_schema" and spec else None
        response_format = ResponseFormat(request.response_format.type, schema)
    stop = [request.stop] if isinstance(request.stop, str) else list(request.stop or [])
    if len(stop) > 4:
        raise ValueError("at most 4 stop sequences are supported")
    request_id = engine.derive_id(
        "chatcmpl-",
        id_payload(
            request.model_dump(mode="json", exclude={"stream", "stream_options", "receipt", "previous_receipt"})
        ),
    )
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
        previous_receipt=request.previous_receipt,
        truncation=request.truncation or "disabled",
        context_overflow=request.context_overflow or "stop",
        thinking=thinking_switch(request.reasoning_effort, request.chat_template_kwargs),
        max_reasoning_tokens=request.max_reasoning_tokens,
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
    n = request.n if request.n is not None else 1
    try:
        chat = _chat_request(request, engine)
        if request.vote is not None:
            if request.stream or (request.n is not None and request.n != 1):
                raise ValueError("vote cannot be combined with stream or n")
            return _vote(request, chat, engine)
        if request.beam is not None:
            if request.stream or (request.n is not None and request.n != 1):
                raise ValueError("beam search cannot be combined with stream or n (use beam.n_best)")
            return _beam(request, chat, engine)
        if request.stream:
            if not 1 <= n <= MAX_CHOICES:
                raise ValueError(f"n must be between 1 and {MAX_CHOICES}")
            stream = engine.chat_stream(chat)
            include_usage = bool(request.stream_options and request.stream_options.include_usage)
            return StreamingResponse(
                _chunks(engine, chat, stream, include_usage, chat.top_logprobs is not None, bool(request.receipt), n),
                media_type="text/event-stream",
            )
        results = engine.chat_choices(chat, n)
    except ContextLengthError as error:
        return _error(str(error), CONTEXT_LENGTH_EXCEEDED)
    except ValueError as error:
        return _error(str(error))

    choices = []
    for index, result in enumerate(results):
        calls = [
            ToolCallModel(id=c.id, function=FunctionCall(name=c.name, arguments=c.arguments)) for c in result.tool_calls
        ]
        choices.append(
            ChatCompletionChoice(
                index=index,
                message=AssistantMessage(
                    content=result.content or (None if calls else ""),
                    tool_calls=calls or None,
                    **({"reasoning_content": result.reasoning} if result.reasoning is not None else {}),
                ),
                logprobs=_logprobs(engine, result.logprobs) if chat.top_logprobs is not None else None,
                finish_reason=result.finish_reason,
                **({"receipt": result.receipt} if request.receipt and n > 1 else {}),
            )
        )
    first = results[0]
    completion_tokens = sum(result.completion_tokens for result in results)
    return ChatCompletionResponse(
        id=chat.request_id,
        created=0,
        model=engine.model.id,
        system_fingerprint=engine.system_fingerprint,
        choices=choices,
        usage=ChatCompletionUsage(
            prompt_tokens=first.prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=first.prompt_tokens + completion_tokens,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=first.cached_tokens),
            **_reasoning_usage(sum(result.reasoning_tokens for result in results)),
        ),
        **({"receipt": first.receipt} if request.receipt else {}),
    )


def _vote(request: ChatCompletionRequest, chat: ChatRequest, engine: DllmEngine) -> ChatCompletionResponse:
    """The winning answer of a vote as choice 0, the ballots as ``vote`` and the vote's receipt."""
    assert request.vote is not None
    outcome = voting.vote(engine, chat, request.vote.n, request.vote.extract)
    result = outcome.result
    completion_tokens = sum(r.completion_tokens for r in outcome.results)
    calls = [
        ToolCallModel(id=c.id, function=FunctionCall(name=c.name, arguments=c.arguments)) for c in result.tool_calls
    ]
    choice = ChatCompletionChoice(
        index=0,
        message=AssistantMessage(
            content=result.content or (None if calls else ""),
            tool_calls=calls or None,
            **({"reasoning_content": result.reasoning} if result.reasoning is not None else {}),
        ),
        logprobs=_logprobs(engine, result.logprobs) if chat.top_logprobs is not None else None,
        finish_reason=result.finish_reason,
    )
    return ChatCompletionResponse(
        id=chat.request_id,
        created=0,
        model=engine.model.id,
        system_fingerprint=engine.system_fingerprint,
        choices=[choice],
        usage=ChatCompletionUsage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=result.prompt_tokens + completion_tokens,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=outcome.results[0].cached_tokens),
            **_reasoning_usage(sum(r.reasoning_tokens for r in outcome.results)),
        ),
        vote=outcome.to_json(),
        **({"receipt": voting.record(engine, chat, outcome)} if request.receipt else {}),
    )


def _beam(request: ChatCompletionRequest, chat: ChatRequest, engine: DllmEngine) -> ChatCompletionResponse:
    """The best answers of a beam search as the choices, best first, the search as ``beam`` and its receipt."""
    assert request.beam is not None
    options = request.beam
    outcome = beam.search(engine, chat, options.width, options.n_best, options.length_penalty)
    choices = [
        ChatCompletionChoice(
            index=index,
            message=AssistantMessage(content=hypothesis.text),
            logprobs=ChoiceLogprobs(
                content=[
                    LogprobEntry(token=text, logprob=logprob, bytes=data, top_logprobs=[])
                    for token, logprob in zip(hypothesis.tokens, hypothesis.logprobs, strict=False)
                    for text, data in [_token(engine, token)]
                ]
            )
            if chat.top_logprobs is not None
            else None,
            finish_reason=hypothesis.finish_reason,
        )
        for index, hypothesis in enumerate(outcome.hypotheses)
    ]
    return ChatCompletionResponse(
        id=chat.request_id,
        created=0,
        model=engine.model.id,
        system_fingerprint=engine.system_fingerprint,
        choices=choices,
        usage=ChatCompletionUsage(
            prompt_tokens=outcome.prompt_tokens,
            completion_tokens=outcome.completion_tokens,
            total_tokens=outcome.prompt_tokens + outcome.completion_tokens,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=0),
        ),
        beam=outcome.to_json(),
        **({"receipt": beam.record(engine, chat, outcome)} if request.receipt else {}),
    )


def _reasoning_usage(reasoning_tokens: int) -> dict:
    """OpenAI's ``completion_tokens_details``, only when a thinking model thought (other responses keep their bytes)."""
    return {"completion_tokens_details": {"reasoning_tokens": reasoning_tokens}} if reasoning_tokens else {}


def _sse(chunk: ChatCompletionChunk) -> str:
    return f"data: {chunk.model_dump_json()}\n\n"


def _chunks(
    engine: DllmEngine,
    chat: ChatRequest,
    stream,
    include_usage: bool,
    logprobs: bool,
    receipt: bool = False,
    n: int = 1,
) -> Iterator[str]:
    """Several choices stream one after another (each one's chunks in order), never interleaved by timing."""

    def chunk(index: int, delta: ChunkDelta, extra: dict | None = None, **fields) -> ChatCompletionChunk:
        return ChatCompletionChunk(
            id=chat.request_id,
            model=engine.model.id,
            system_fingerprint=engine.system_fingerprint,
            choices=[ChunkChoice(index=index, delta=delta, **fields)],
            **(extra or {}),
        )

    first, completion_tokens, reasoning_tokens = stream, 0, 0
    for index in range(n):
        if index > 0:
            stream = engine.chat_stream(engine.choice_request(chat, index))
        yield _sse(chunk(index, ChunkDelta(role="assistant", content="")))
        for event in stream:
            if isinstance(event, TextDelta):
                entries = _logprobs(engine, event.logprobs) if logprobs else None
                yield _sse(chunk(index, ChunkDelta(content=event.text), logprobs=entries))
            elif isinstance(event, ReasoningDelta):
                yield _sse(chunk(index, ChunkDelta(reasoning_content=event.text)))
            elif isinstance(event, ToolCallEvent):
                call = DeltaToolCall(
                    index=event.index,
                    id=event.call.id,
                    type="function",
                    function=DeltaFunctionCall(name=event.call.name, arguments=event.call.arguments),
                )
                yield _sse(chunk(index, ChunkDelta(tool_calls=[call])))
            elif isinstance(event, Finished):
                extra = {"receipt": event.receipt} if receipt else None
                yield _sse(chunk(index, ChunkDelta(), extra, finish_reason=event.finish_reason))
                completion_tokens += event.completion_tokens
                reasoning_tokens += event.reasoning_tokens
    if include_usage:
        usage = ChatCompletionUsage(
            prompt_tokens=first.prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=first.prompt_tokens + completion_tokens,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=first.cached_tokens),
            **_reasoning_usage(reasoning_tokens),
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
            embedding = engine.embed(item, request.dimensions, request.input_type)  # type: ignore[arg-type]
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


@app.post("/v1/watermark/detect", response_model=None)
def detect_watermark(request: WatermarkDetectRequest, engine: Engine) -> JSONResponse:
    """Extension: counts the green tokens of ``text`` for ``key`` with the served model's tokenizer
    (:mod:`etalii_dllm.watermark`); the same numbers on every machine."""
    from etalii_dllm import watermark

    try:
        result = watermark.detect(engine.tokenizer.encode(request.text), request.key, request.gamma)
    except ValueError as error:
        return _error(str(error))
    return JSONResponse(result.to_json())


@app.post("/v1/rerank", response_model=None)
@app.post("/rerank", response_model=None, include_in_schema=False)
def rerank(request: RerankRequest, engine: Engine) -> JSONResponse:
    """Ranks ``documents`` for ``query`` with the served model as a judge (:mod:`etalii_dllm.reranking`): the same
    scores and order on every machine, and an id derived from the request."""
    from etalii_dllm.reranking import Reranker

    texts = [d if isinstance(d, str) else d.text for d in request.documents]
    if not texts:
        return _error("'documents' must not be empty")
    if request.top_n is not None and request.top_n < 1:
        return _error("'top_n' must be at least 1")
    try:
        judged = Reranker(engine).judgements(request.query, texts, request.instruction)
    except ValueError as error:
        return _error(str(error))
    results = [
        RerankResult(
            index=i,
            relevance_score=judgement.score,
            document=RerankDocument(text=texts[i]) if request.return_documents else None,
        )
        for i, judgement in judged[: request.top_n]
    ]
    response = RerankResponse(
        id=engine.derive_id("rerank-", request.model_dump(mode="json")),
        model=engine.model.id,
        results=results,
        usage=RerankUsage(total_tokens=sum(judgement.tokens for _, judgement in judged)),
    )
    return JSONResponse(response.model_dump(mode="json", exclude_none=True))


@app.post("/v1/receipts/verify", response_model=None)
def verify_receipt(receipt: dict[str, Any] | list[dict[str, Any]], engine: Engine) -> JSONResponse:
    """Extension: re-runs the request a generation receipt records and says whether the output is the same. A list
    is a conversation's receipt chain, oldest first (:func:`etalii_dllm.receipts.verify_chain`); score and vote
    receipts are scored or voted again."""
    try:
        if isinstance(receipt, list):
            return JSONResponse(receipts.verify_chain(engine, receipt).to_json())
        if "score" in receipt:
            verification = scoring.verify(engine, receipt)
        elif "beam" in receipt:
            verification = beam.verify(engine, receipt)
        elif "vote" in receipt:
            verification = voting.verify(engine, receipt)
        else:
            verification = receipts.verify(engine, receipt)
    except (ValueError, KeyError, TypeError) as problem:
        return _error(f"not a valid receipt: {problem}")
    return JSONResponse(verification.to_json())


@app.get("/v1/audit", response_model=None)
def audit_report(engine: Engine) -> JSONResponse:
    """Extension: what ``--audit-every`` found (re-runs of served responses), and the response cache's counters."""
    report: dict[str, Any] = {"audit": engine.auditor.report() if engine.auditor is not None else None}
    cache = engine.response_cache
    report["response_cache"] = None if cache is None else {"hits": cache.hits, "misses": cache.misses, **cache.stats()}
    report["coalesced"] = engine.inflight.joined
    return JSONResponse(report)


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="EtAlii.Dllm OpenAI-, Anthropic- and Ollama-compatible server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5080)
    add_runtime_arguments(parser)
    args = parser.parse_args()
    use_model_file(
        args.model,
        args.quantize,
        args.threads,
        args.device,
        args.prompt_cache,
        args.adapter,
        steer=args.steer,
        steer_strength=args.steer_strength,
        index=args.index,
        index_top=args.index_top,
        index_mode=args.index_mode,
        rerank_model=args.rerank_model,
        embedding_model=args.embedding_model,
        speculate=args.speculate,
        draft_model=args.draft_model,
        prompt_cache_dir=args.prompt_cache_dir,
        sign_key=args.sign_key,
        response_cache=args.response_cache,
        audit_every=args.audit_every,
        contrast_model=args.contrast_model,
        ensemble_models=args.ensemble_model or (),
        ensemble_weight=args.ensemble_weight,
    )
    uvicorn.run(app, host=args.host, port=args.port)
