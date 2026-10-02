"""OpenAI's completions API: ``POST /v1/completions`` (docs/api.md#completions-api).

A prompt is continued as it is (no chat template), on the same event stream as every other front end, so a completion
equals ``dllm generate`` for the same prompt and options bit for bit. ``echo`` puts the prompt in front of the text,
and with ``logprobs`` the prompt's own tokens are scored exactly (:mod:`etalii_dllm.scoring`): ``max_tokens: 0``
with ``echo`` and ``logprobs`` scores a text without generating anything. Ids derive from the request, timestamps
are 0.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from etalii_dllm import beam, scoring
from etalii_dllm.engine import MAX_CHOICES, ChatRequest, DllmEngine, Finished, ResponseFormat, TextDelta
from etalii_dllm.engine import default_engine as _default_engine
from etalii_dllm.generation import ContextLengthError, TokenLogprobs
from etalii_dllm.sampling import SamplingOptions
from etalii_dllm.server.contracts import (
    BeamOptions,
    ContrastOptions,
    GuidanceOptions,
    WatermarkOptions,
    guided,
    id_payload,
)

router = APIRouter()
Engine = Annotated[DllmEngine, Depends(_default_engine)]

DEFAULT_MAX_TOKENS = 16
"""OpenAI's default for the completions API."""


class StreamOptions(BaseModel):
    include_usage: bool | None = None


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str | list[str]
    suffix: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    """Extension."""
    n: int | None = None
    best_of: int | None = None
    stream: bool | None = None
    stream_options: StreamOptions | None = None
    logprobs: int | None = None
    """How many alternatives to list per token (0 to 20); the token's own log-probability is always given."""
    echo: bool | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    repeat_last_n: int | None = None
    guided_regex: str | None = None
    grammar: str | None = None
    """Extension (as in llama.cpp): a GBNF grammar the completion must follow (docs/api.md#grammars)."""
    watermark: WatermarkOptions | None = None
    guidance: GuidanceOptions | None = None
    """Extension: classifier-free guidance away from a negative prompt (docs/api.md#guided-decoding)."""
    contrast: ContrastOptions | None = None
    """Extension: contrastive decoding against the server's amateur model."""
    beam: BeamOptions | None = None
    """Extension: the best answers of an exact beam search (docs/api.md#beam-search)."""
    token_healing: bool | None = None
    """Extension: take the prompt's last token back and make the answer start with it (docs/api.md#token-healing)."""
    receipt: bool | None = None
    """Extension: each choice carries its generation receipt (docs/receipts.md)."""
    user: str | None = None


class CompletionLogprobs(BaseModel):
    tokens: list[str]
    token_logprobs: list[float | None]
    top_logprobs: list[dict[str, float] | None] | None
    text_offset: list[int]


class CompletionChoice(BaseModel):
    model_config = ConfigDict(extra="allow")  # ``receipt`` when requested

    text: str
    index: int
    logprobs: CompletionLogprobs | None = None
    finish_reason: str | None = None


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Completion(BaseModel):
    model_config = ConfigDict(extra="allow")  # ``beam`` for beam searches

    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int = 0
    model: str
    system_fingerprint: str
    choices: list[CompletionChoice]
    usage: CompletionUsage | None = None


def _error(message: str, code: str | None = None) -> JSONResponse:
    content: dict[str, Any] = {"message": message, "type": "invalid_request_error"}
    if code is not None:
        content.update(param="prompt", code=code)
    return JSONResponse(status_code=400, content={"error": content})


def _requests(request: CompletionRequest, engine: DllmEngine) -> list[ChatRequest]:
    """One engine request per prompt; raises ``ValueError`` for invalid requests."""
    if request.suffix:
        raise ValueError("suffix is not supported")
    n = request.n if request.n is not None else 1
    if not 1 <= n <= MAX_CHOICES:
        raise ValueError(f"n must be between 1 and {MAX_CHOICES}")
    if request.best_of is not None and request.best_of != n:
        raise ValueError("best_of is not supported (only best_of equal to n)")
    prompts = [request.prompt] if isinstance(request.prompt, str) else list(request.prompt)
    if not prompts:
        raise ValueError("'prompt' must not be empty")
    stop = [request.stop] if isinstance(request.stop, str) else list(request.stop or [])
    if len(stop) > 4:
        raise ValueError("at most 4 stop sequences are supported")
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
    if request.guided_regex is not None and request.grammar is not None:
        raise ValueError("guided_regex and grammar cannot be combined")
    response_format = ResponseFormat()
    if request.guided_regex:
        response_format = ResponseFormat("regex", pattern=request.guided_regex)
    elif request.grammar is not None:
        response_format = ResponseFormat("grammar", pattern=request.grammar)
    request_id = engine.derive_id(
        "cmpl-", id_payload(request.model_dump(mode="json", exclude={"stream", "stream_options", "receipt", "user"}))
    )
    max_tokens = request.max_tokens if request.max_tokens is not None else DEFAULT_MAX_TOKENS
    if max_tokens < 0:
        raise ValueError("max_tokens must be non-negative")
    if request.logprobs is not None and not 0 <= request.logprobs <= 20:
        raise ValueError("logprobs must be between 0 and 20")
    return [
        ChatRequest(
            [],
            max_tokens,
            options,
            stop=stop,
            response_format=response_format,
            top_logprobs=request.logprobs,
            request_id=request_id,
            prompt=prompt,
            token_healing=bool(request.token_healing),
        )
        for prompt in prompts
    ]


class _Logprobs:
    """Builds OpenAI's ``logprobs`` object of a choice, text offsets counted in characters of the returned text."""

    def __init__(self, engine: DllmEngine, top: int) -> None:
        self.engine, self.top = engine, top
        self.tokens: list[str] = []
        self.values: list[float | None] = []
        self.alternatives: list[dict[str, float] | None] = []
        self.offsets: list[int] = []
        self.length = 0

    def _text(self, token: int) -> str:
        return self.engine.tokenizer.decode_bytes([token]).decode("utf-8", errors="replace")

    def add(self, token: int, logprob: float | None, top: Sequence[Any] | None, text: str) -> None:
        self.tokens.append(self._text(token))
        self.values.append(logprob)
        self.alternatives.append(None if top is None else {self._text(a.token): a.logprob for a in top})
        self.offsets.append(self.length)
        self.length += len(text)

    def generated(self, entries: Sequence[TokenLogprobs]) -> None:
        for entry in entries:
            self.add(entry.token, entry.logprob, entry.top, self._text(entry.token))

    def take(self) -> CompletionLogprobs:
        result = CompletionLogprobs(
            tokens=self.tokens,
            token_logprobs=self.values,
            top_logprobs=self.alternatives,
            text_offset=self.offsets,
        )
        self.tokens, self.values, self.alternatives, self.offsets = [], [], [], []
        return result


def _echo(engine: DllmEngine, chat: ChatRequest, logprobs: _Logprobs | None) -> None:
    """The prompt's scores, in front of the generated tokens."""
    if logprobs is None:
        return
    assert chat.prompt is not None
    score = scoring.score_text(engine, chat.prompt, logprobs.top)
    for i, entry in enumerate(score.tokens):
        logprobs.add(entry.token, entry.logprob, None if i == 0 else entry.top, logprobs._text(entry.token))


@router.post("/v1/completions", response_model=None)
def completions(request: CompletionRequest, engine: Engine) -> Completion | JSONResponse | StreamingResponse:
    n = request.n if request.n is not None else 1
    try:
        chats = _requests(request, engine)
        if request.beam is not None:
            return _beam(request, chats, engine)
        if request.stream:
            include_usage = bool(request.stream_options and request.stream_options.include_usage)
            first = engine.chat_stream(chats[0])  # raises for invalid requests before streaming starts
            return StreamingResponse(
                _chunks(engine, request, chats, n, first, include_usage), media_type="text/event-stream"
            )
        choices: list[CompletionChoice] = []
        prompt_tokens = completion_tokens = 0
        for p, chat in enumerate(chats):
            results = engine.chat_choices(chat, n)
            prompt_tokens += results[0].prompt_tokens
            for i, result in enumerate(results):
                logprobs = _Logprobs(engine, request.logprobs) if request.logprobs is not None else None
                if request.echo:
                    _echo(engine, chat, logprobs)
                if logprobs is not None:
                    logprobs.generated(result.logprobs)
                completion_tokens += result.completion_tokens
                choices.append(
                    CompletionChoice(
                        text=(chat.prompt or "") + result.content if request.echo else result.content,
                        index=p * n + i,
                        logprobs=logprobs.take() if logprobs is not None else None,
                        finish_reason=result.finish_reason,
                        **({"receipt": result.receipt} if request.receipt else {}),
                    )
                )
    except ContextLengthError as error:
        return _error(str(error), "context_length_exceeded")
    except ValueError as error:
        return _error(str(error))
    return Completion(
        id=chats[0].request_id,
        model=engine.model.id,
        system_fingerprint=engine.system_fingerprint,
        choices=choices,
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _beam(request: CompletionRequest, chats: list[ChatRequest], engine: DllmEngine) -> Completion:
    """Each prompt's best beam search answers as choices (``prompt_index * n_best + i``), best first; ``beam`` holds
    the searches and, with ``receipt``, every choice carries its search's receipt."""
    assert request.beam is not None
    options = request.beam
    if request.stream or (request.n is not None and request.n != 1) or request.logprobs is not None:
        raise ValueError("beam search cannot be combined with stream, n or logprobs (use beam.n_best)")
    choices: list[CompletionChoice] = []
    searches = []
    prompt_tokens = completion_tokens = 0
    for p, chat in enumerate(chats):
        outcome = beam.search(engine, chat, options.width, options.n_best, options.length_penalty)
        receipt = beam.record(engine, chat, outcome) if request.receipt else None
        searches.append(outcome.to_json())
        prompt_tokens += outcome.prompt_tokens
        completion_tokens += outcome.completion_tokens
        for i, hypothesis in enumerate(outcome.hypotheses):
            choices.append(
                CompletionChoice(
                    text=(chat.prompt or "") + hypothesis.text if request.echo else hypothesis.text,
                    index=p * options.n_best + i,
                    finish_reason=hypothesis.finish_reason,
                    **({"receipt": receipt} if receipt is not None else {}),
                )
            )
    return Completion(
        id=chats[0].request_id,
        model=engine.model.id,
        system_fingerprint=engine.system_fingerprint,
        choices=choices,
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        beam=searches,
    )


def _chunks(
    engine: DllmEngine,
    request: CompletionRequest,
    chats: list[ChatRequest],
    n: int,
    first: Any,
    include_usage: bool,
) -> Iterator[str]:
    """Choices stream one after another (each one's chunks in order), never interleaved by timing."""

    def chunk(index: int, text: str, logprobs: _Logprobs | None, finish_reason: str | None = None, **extra) -> str:
        choice = CompletionChoice(
            text=text,
            index=index,
            logprobs=logprobs.take() if logprobs is not None else None,
            finish_reason=finish_reason,
            **extra,
        )
        body = Completion(id=chats[0].request_id, model=engine.model.id, system_fingerprint=engine.system_fingerprint,
                          choices=[choice])  # fmt: skip
        return f"data: {body.model_dump_json(exclude={'usage'})}\n\n"

    prompt_tokens = completion_tokens = 0
    for p, chat in enumerate(chats):
        for i in range(n):
            stream = first if (p, i) == (0, 0) else engine.chat_stream(engine.choice_request(chat, i))
            if i == 0:
                prompt_tokens += stream.prompt_tokens
            logprobs = _Logprobs(engine, request.logprobs) if request.logprobs is not None else None
            if request.echo:
                _echo(engine, chat, logprobs)
                yield chunk(p * n + i, chat.prompt or "", logprobs)
            for event in stream:
                if isinstance(event, TextDelta):
                    if logprobs is not None:
                        logprobs.generated(event.logprobs)
                    yield chunk(p * n + i, event.text, logprobs)
                elif isinstance(event, Finished):
                    completion_tokens += event.completion_tokens
                    extra = {"receipt": event.receipt} if request.receipt else {}
                    yield chunk(p * n + i, "", logprobs, event.finish_reason, **extra)
    if include_usage:
        usage = CompletionUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                                total_tokens=prompt_tokens + completion_tokens)  # fmt: skip
        body = Completion(id=chats[0].request_id, model=engine.model.id, system_fingerprint=engine.system_fingerprint,
                          choices=[], usage=usage)  # fmt: skip
        yield f"data: {body.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"
