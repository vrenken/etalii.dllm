"""OpenAI Responses API: ``POST /v1/responses`` (with streaming), ``GET`` and ``DELETE /v1/responses/{id}``.

Another thin translation over :meth:`DllmEngine.chat_stream`: ``instructions`` and ``input`` (a string or message,
``function_call`` and ``function_call_output`` items) become the conversation, and the engine's events become output
items. Streaming sends the typed ``response.*`` server-sent events; the non-streamed response is the one carried by
the final ``response.completed`` (or ``response.incomplete``) event of the same stream, so both are identical.

``previous_response_id`` continues a stored conversation. Responses are stored in memory (``store``, default true,
the most recent :data:`STORE_SIZE`); their ids are hashes of the request (which includes the previous id) and the
weights, so a conversation replayed from the start gets the same ids and the same answers.

Differences from OpenAI: ``temperature`` defaults to 0 (greedy) and ``seed`` (``extra_body``) seeds sampling, as on
every endpoint here; ``created_at`` is 0; only function tools; no images, files, reasoning, background mode or
conversation objects.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from collections.abc import Iterator
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from etalii_dllm.chat import ChatMessage, ToolCall
from etalii_dllm.engine import (
    ChatRequest,
    ChatStream,
    DllmEngine,
    Finished,
    ResponseFormat,
    TextDelta,
    ToolCallEvent,
    default_engine,
)
from etalii_dllm.generation import TokenLogprobs
from etalii_dllm.sampling import SamplingOptions
from etalii_dllm.tools import Tool, ToolChoice

Engine = Annotated[DllmEngine, Depends(default_engine)]

router = APIRouter()

DEFAULT_MAX_OUTPUT_TOKENS = 1024
STORE_SIZE = 256
"""How many stored responses ``previous_response_id`` and ``GET /v1/responses/{id}`` can reach."""
LOGPROBS_INCLUDE = "message.output_text.logprobs"


# -- wire format ----------------------------------------------------------------------------------------------------


class ContentPart(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    text: str | None = None


class InputItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = "message"
    role: str | None = None
    content: str | list[ContentPart] | None = None
    call_id: str | None = None
    name: str | None = None
    arguments: str | None = None
    output: str | list[ContentPart] | None = None


class FunctionTool(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = "function"
    name: str | None = None
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None


class NamedToolChoice(BaseModel):
    type: str
    name: str | None = None


class TextFormat(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    type: Literal["text", "json_object", "json_schema"] = "text"
    name: str | None = None
    json_schema: dict[str, Any] | None = Field(default=None, alias="schema")
    description: str | None = None
    strict: bool | None = None


class TextConfig(BaseModel):
    format: TextFormat | None = None


class ResponsesRequest(BaseModel):
    model: str | None = None
    input: str | list[InputItem] = ""
    instructions: str | None = None
    previous_response_id: str | None = None
    tools: list[FunctionTool] | None = None
    tool_choice: Literal["none", "auto", "required"] | NamedToolChoice | None = None
    parallel_tool_calls: bool | None = None
    text: TextConfig | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    """Not in the OpenAI API; accepted as an extension."""
    seed: int | None = None
    """Not in the Responses API; accepted as an extension (``extra_body``)."""
    max_output_tokens: int | None = None
    top_logprobs: int | None = None
    include: list[str] | None = None
    stream: bool | None = None
    store: bool | None = None
    metadata: dict[str, str] | None = None
    receipt: bool | None = None
    """Extension: add a ``receipt`` to the response (see docs/receipts.md)."""


def error(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": "invalid_request_error"}})


# -- stored conversations -------------------------------------------------------------------------------------------


class _Store:
    """The most recent stored responses: the response object and the conversation it ended (without its
    ``instructions``, which do not carry over to a follow-up, as in OpenAI's API)."""

    def __init__(self, size: int = STORE_SIZE) -> None:
        self._size = size
        self._items: OrderedDict[str, tuple[dict[str, Any], list[ChatMessage]]] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, response: dict[str, Any], conversation: list[ChatMessage]) -> None:
        with self._lock:
            self._items[response["id"]] = (response, conversation)
            self._items.move_to_end(response["id"])
            while len(self._items) > self._size:
                self._items.popitem(last=False)

    def get(self, response_id: str) -> tuple[dict[str, Any], list[ChatMessage]] | None:
        with self._lock:
            return self._items.get(response_id)

    def delete(self, response_id: str) -> bool:
        with self._lock:
            return self._items.pop(response_id, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


store = _Store()


# -- translation ----------------------------------------------------------------------------------------------------


def _text(content: str | list[ContentPart] | None) -> str:
    if content is None or isinstance(content, str):
        return content or ""
    parts = []
    for part in content:
        if part.type not in ("input_text", "output_text", "text"):
            raise ValueError(f"content parts of type {part.type!r} are not supported")
        parts.append(part.text or "")
    return "".join(parts)


def _conversation(items: str | list[InputItem]) -> list[ChatMessage]:
    """The input as chat messages: consecutive ``function_call`` items join the assistant message before them."""
    if isinstance(items, str):
        return [ChatMessage("user", items)]
    messages: list[ChatMessage] = []
    names: dict[str, str] = {}
    for item in items:
        if item.type == "message":
            role = "system" if item.role == "developer" else item.role
            if role not in ("system", "user", "assistant"):
                raise ValueError(f"unknown message role {item.role!r}")
            messages.append(ChatMessage(role, _text(item.content)))
        elif item.type == "function_call":
            if not item.call_id or not item.name:
                raise ValueError("function_call items need a call_id and a name")
            call = ToolCall(item.call_id, item.name, item.arguments or "{}")
            names[item.call_id] = item.name
            last = messages[-1] if messages else None
            if last is not None and last.role == "assistant":
                messages[-1] = ChatMessage("assistant", last.content, (*last.tool_calls, call))
            else:
                messages.append(ChatMessage("assistant", "", (call,)))
        elif item.type == "function_call_output":
            call_id = item.call_id or ""
            messages.append(ChatMessage("tool", _text(item.output), tool_call_id=call_id, name=names.get(call_id)))
        elif item.type == "reasoning":
            continue
        else:
            raise ValueError(f"input items of type {item.type!r} are not supported")
    return messages


def _tools(request: ResponsesRequest) -> tuple[list[Tool], ToolChoice]:
    tools = []
    for tool in request.tools or ():
        if tool.type != "function" or not tool.name:
            raise ValueError(f"only function tools are supported, not {tool.type!r}")
        tools.append(Tool(tool.name, tool.description or "", tool.parameters or {}))
    choice = request.tool_choice
    if choice is None:
        return tools, ToolChoice("auto")
    if isinstance(choice, str):
        return tools, ToolChoice(choice)
    if choice.type != "function" or not choice.name:
        raise ValueError("tool_choice must be 'none', 'auto', 'required' or a function")
    return tools, ToolChoice("named", choice.name)


def _response_format(request: ResponsesRequest) -> ResponseFormat:
    spec = request.text.format if request.text is not None else None
    if spec is None or spec.type == "text":
        return ResponseFormat()
    if spec.type == "json_schema" and spec.json_schema is None:
        raise ValueError("text.format of type json_schema needs a 'schema'")
    return ResponseFormat(spec.type, spec.json_schema if spec.type == "json_schema" else None)


def _prepare(request: ResponsesRequest, engine: DllmEngine) -> tuple[ChatRequest, list[ChatMessage]]:
    """The engine request and the conversation to store (everything but the instructions)."""
    history: list[ChatMessage] = []
    if request.previous_response_id:
        stored = store.get(request.previous_response_id)
        if stored is None:
            raise ValueError(f"Previous response with id '{request.previous_response_id}' not found.")
        history = list(stored[1])
    conversation = history + _conversation(request.input)
    if not conversation:
        raise ValueError("'input' must not be empty")
    instructions = [ChatMessage("system", request.instructions)] if request.instructions else []
    tools, choice = _tools(request)
    options = SamplingOptions(
        temperature=request.temperature if request.temperature is not None else 0.0,
        top_k=request.top_k or 0,
        top_p=request.top_p if request.top_p is not None else 1.0,
        seed=request.seed or 0,
    )
    wants_logprobs = LOGPROBS_INCLUDE in (request.include or ()) or request.top_logprobs is not None
    payload = request.model_dump(mode="json", exclude={"stream", "receipt"})
    chat = ChatRequest(
        messages=instructions + conversation,
        max_tokens=request.max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS,
        options=options,
        tools=tools,
        tool_choice=choice,
        response_format=_response_format(request),
        top_logprobs=(request.top_logprobs or 0) if wants_logprobs else None,
        call_id_prefix="call_",
        request_id=engine.derive_id("resp_", payload),
    )
    return chat, conversation


def _item_id(prefix: str, response_id: str, index: int) -> str:
    return prefix + hashlib.sha256(f"{response_id}\n{index}".encode()).hexdigest()[:24]


def _logprobs(engine: DllmEngine, entries: tuple[TokenLogprobs, ...]) -> list[dict[str, Any]]:
    def token(value: int) -> dict[str, Any]:
        data = engine.tokenizer.decode_bytes([value])
        return {"token": data.decode("utf-8", errors="replace"), "bytes": list(data)}

    return [
        {
            **token(e.token),
            "logprob": e.logprob,
            "top_logprobs": [{**token(a.token), "logprob": a.logprob} for a in e.top],
        }
        for e in entries
    ]


def _response(request: ResponsesRequest, chat: ChatRequest, engine: DllmEngine, **fields: Any) -> dict[str, Any]:
    tools = [
        {"type": "function", "name": t.name, "description": t.description or None, "parameters": t.parameters,
         "strict": False}
        for t in chat.tools
    ]  # fmt: skip
    choice = request.tool_choice
    fmt = request.text.format if request.text is not None and request.text.format is not None else TextFormat()
    return {
        "id": chat.request_id,
        "object": "response",
        "created_at": 0,
        "status": "in_progress",
        "error": None,
        "incomplete_details": None,
        "instructions": request.instructions,
        "max_output_tokens": request.max_output_tokens,
        "model": engine.model.id,
        "output": [],
        "parallel_tool_calls": request.parallel_tool_calls if request.parallel_tool_calls is not None else True,
        "previous_response_id": request.previous_response_id,
        "store": request.store is not False,
        "temperature": chat.options.temperature,
        "text": {"format": fmt.model_dump(by_alias=True, exclude_none=True)},
        "tool_choice": choice.model_dump(exclude_none=True) if isinstance(choice, BaseModel) else (choice or "auto"),
        "tools": tools,
        "top_p": chat.options.top_p,
        "truncation": "disabled",
        "usage": None,
        "metadata": request.metadata or {},
        **fields,
    }


class _Events:
    """Turns the engine's events into the Responses API's streamed events, building the response as it goes."""

    def __init__(self, request: ResponsesRequest, chat: ChatRequest, engine: DllmEngine, stream: ChatStream) -> None:
        self.request, self.chat, self.engine, self.stream = request, chat, engine, stream
        self.response = _response(request, chat, engine)
        self.sequence = 0
        self.final: dict[str, Any] | None = None

    def _event(self, kind: str, **data: Any) -> dict[str, Any]:
        event = {"type": kind, "sequence_number": self.sequence, **data}
        self.sequence += 1
        return event

    def __iter__(self) -> Iterator[dict[str, Any]]:
        yield self._event("response.created", response=json.loads(json.dumps(self.response)))
        yield self._event("response.in_progress", response=json.loads(json.dumps(self.response)))
        output: list[dict[str, Any]] = self.response["output"]
        message: dict[str, Any] | None = None
        text: list[str] = []
        logprobs: list[dict[str, Any]] = []
        response_id = self.chat.request_id
        for event in self.stream:
            if isinstance(event, TextDelta):
                entries = _logprobs(self.engine, event.logprobs) if self.chat.top_logprobs is not None else []
                if not event.text and not entries:
                    continue
                if message is None:
                    message = {"id": _item_id("msg_", response_id, len(output)), "type": "message",
                               "status": "in_progress", "role": "assistant", "content": []}  # fmt: skip
                    output.append(message)
                    yield from self._open_message(message, len(output) - 1)
                text.append(event.text)
                logprobs.extend(entries)
                yield self._event("response.output_text.delta", item_id=message["id"], output_index=len(output) - 1,
                                  content_index=0, delta=event.text, logprobs=entries)  # fmt: skip
            elif isinstance(event, ToolCallEvent):
                if message is not None:
                    yield from self._close_message(message, output.index(message), "".join(text), logprobs)
                    message = None
                yield from self._function_call(event.call, output)
            elif isinstance(event, Finished):
                if message is not None:
                    yield from self._close_message(message, output.index(message), "".join(text), logprobs)
                yield self._finish(event)

    def _open_message(self, message: dict[str, Any], index: int) -> Iterator[dict[str, Any]]:
        yield self._event("response.output_item.added", output_index=index, item={**message, "content": []})
        part = {"type": "output_text", "text": "", "annotations": [], "logprobs": []}
        yield self._event("response.content_part.added", item_id=message["id"], output_index=index, content_index=0,
                          part=part)  # fmt: skip

    def _close_message(
        self, message: dict[str, Any], index: int, text: str, logprobs: list[dict[str, Any]]
    ) -> Iterator[dict[str, Any]]:
        part = {"type": "output_text", "text": text, "annotations": [], "logprobs": logprobs}
        yield self._event("response.output_text.done", item_id=message["id"], output_index=index, content_index=0,
                          text=text, logprobs=logprobs)  # fmt: skip
        yield self._event("response.content_part.done", item_id=message["id"], output_index=index, content_index=0,
                          part=part)  # fmt: skip
        message["content"] = [part]
        message["status"] = "completed"
        yield self._event("response.output_item.done", output_index=index, item=message)

    def _function_call(self, call: ToolCall, output: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        index = len(output)
        item = {"id": _item_id("fc_", self.chat.request_id, index), "type": "function_call", "status": "in_progress",
                "call_id": call.id, "name": call.name, "arguments": ""}  # fmt: skip
        output.append(item)
        yield self._event("response.output_item.added", output_index=index, item=dict(item))
        yield self._event("response.function_call_arguments.delta", item_id=item["id"], output_index=index,
                          delta=call.arguments)  # fmt: skip
        yield self._event("response.function_call_arguments.done", item_id=item["id"], output_index=index,
                          name=call.name, arguments=call.arguments)  # fmt: skip
        item.update(arguments=call.arguments, status="completed")
        yield self._event("response.output_item.done", output_index=index, item=item)

    def _finish(self, event: Finished) -> dict[str, Any]:
        prompt, cached = self.stream.prompt_tokens, self.stream.cached_tokens
        self.response["usage"] = {
            "input_tokens": prompt,
            "input_tokens_details": {"cached_tokens": cached},
            "output_tokens": event.completion_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": prompt + event.completion_tokens,
        }
        if event.finish_reason == "length":
            self.response.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
            kind = "response.incomplete"
        else:
            self.response["status"] = "completed"
            kind = "response.completed"
        if self.request.receipt:
            self.response["receipt"] = event.receipt
        self.final = self.response
        return self._event(kind, response=self.response)


def _answer(output: list[dict[str, Any]], engine: DllmEngine) -> ChatMessage:
    """The response's output as the assistant message a follow-up request sees."""
    text = "".join(p["text"] for item in output if item["type"] == "message" for p in item["content"])
    calls = tuple(
        ToolCall(item["call_id"], item["name"], item["arguments"]) for item in output if item["type"] == "function_call"
    )
    return ChatMessage("assistant", text, calls)


def _stored(events: _Events, conversation: list[ChatMessage]) -> Iterator[dict[str, Any]]:
    """Passes the events on and stores the finished response (unless ``store`` is false)."""
    yield from events
    final = events.final
    if final is not None and final["store"]:
        store.put(final, [*conversation, _answer(final["output"], events.engine)])


# -- endpoints ------------------------------------------------------------------------------------------------------


@router.post("/v1/responses", response_model=None)
def create(request: ResponsesRequest, engine: Engine) -> JSONResponse | StreamingResponse:
    try:
        chat, conversation = _prepare(request, engine)
        stream = engine.chat_stream(chat)
    except ValueError as problem:
        return error(str(problem))
    events = _stored(_Events(request, chat, engine, stream), conversation)
    if request.stream:
        lines = (f"event: {e['type']}\ndata: {json.dumps(e, ensure_ascii=False)}\n\n" for e in events)
        return StreamingResponse(lines, media_type="text/event-stream")
    *_, last = events  # the non-streamed response is the one the final event carries
    return JSONResponse(last["response"])


@router.get("/v1/responses/{response_id}", response_model=None)
def retrieve(response_id: str) -> JSONResponse:
    stored = store.get(response_id)
    if stored is None:
        return error(f"Response with id '{response_id}' not found.", 404)
    return JSONResponse(stored[0])


@router.delete("/v1/responses/{response_id}", response_model=None)
def delete(response_id: str) -> JSONResponse:
    if not store.delete(response_id):
        return error(f"Response with id '{response_id}' not found.", 404)
    return JSONResponse({"id": response_id, "object": "response.deleted", "deleted": True})
