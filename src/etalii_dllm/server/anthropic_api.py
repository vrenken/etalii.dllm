"""Anthropic Messages API: ``POST /v1/messages`` (with streaming, tools and structured output) and
``POST /v1/messages/count_tokens``, so Anthropic SDK clients can use the model.

Differences from Anthropic's service: ``temperature`` defaults to 0 (greedy), as on every endpoint of this server;
an extra ``seed`` field (``extra_body={"seed": 7}`` in the SDK) seeds sampling; images, documents and server tools
are refused.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from etalii_dllm.chat import ChatMessage, ToolCall
from etalii_dllm.engine import (
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
from etalii_dllm.sampling import SamplingOptions
from etalii_dllm.server.anthropic_contracts import (
    CountTokensRequest,
    CountTokensResponse,
    MessageResponse,
    MessagesRequest,
    RequestBlock,
    TextBlock,
    ThinkingBlock,
    ToolChoiceModel,
    ToolDefinition,
    ToolUseBlock,
    Usage,
)
from etalii_dllm.server.contracts import id_payload
from etalii_dllm.tools import Tool, ToolChoice

Engine = Annotated[DllmEngine, Depends(default_engine)]

router = APIRouter()

_STOP_REASONS = {"length": "max_tokens", "tool_calls": "tool_use"}


def error(message: str) -> JSONResponse:
    return JSONResponse(
        status_code=400, content={"type": "error", "error": {"type": "invalid_request_error", "message": message}}
    )


def _block_text(content: str | list[RequestBlock] | None) -> str:
    if content is None or isinstance(content, str):
        return content or ""
    parts = []
    for block in content:
        if block.type != "text":
            raise ValueError(f"content blocks of type {block.type!r} are not supported here")
        parts.append(block.text or "")
    return "\n".join(parts)


def _messages(system: str | list[RequestBlock] | None, messages: list[Any]) -> list[ChatMessage]:
    result = [ChatMessage("system", _block_text(system))] if system else []
    names: dict[str, str] = {}
    for message in messages:
        if isinstance(message.content, str):
            result.append(ChatMessage(message.role, message.content))
            continue
        texts: list[str] = []
        calls: list[ToolCall] = []
        for block in message.content:
            if block.type == "text":
                texts.append(block.text or "")
            elif block.type == "tool_use" and message.role == "assistant":
                if not block.id or not block.name:
                    raise ValueError("tool_use blocks need an id and a name")
                calls.append(ToolCall(block.id, block.name, json.dumps(block.input or {}, ensure_ascii=False)))
                names[block.id] = block.name
            elif block.type == "tool_result" and message.role == "user":
                content = _block_text(block.content)
                if block.is_error:
                    content = "Error: " + content
                tool_id = block.tool_use_id or ""
                result.append(ChatMessage("tool", content, tool_call_id=tool_id, name=names.get(tool_id)))
            elif block.type in ("thinking", "redacted_thinking"):
                continue
            else:
                raise ValueError(f"content blocks of type {block.type!r} are not supported")
        if texts or calls:
            result.append(ChatMessage(message.role, "\n".join(texts), tuple(calls)))
    return result


def _tools(tools: list[ToolDefinition] | None, choice: ToolChoiceModel | None) -> tuple[list[Tool], ToolChoice]:
    converted = []
    for tool in tools or ():
        if tool.type not in (None, "custom"):
            raise ValueError(f"server tools ({tool.type}) are not supported")
        converted.append(Tool(tool.name, tool.description or "", tool.input_schema or {}))
    if choice is None or choice.type == "auto":
        return converted, ToolChoice("auto")
    if choice.type == "any":
        return converted, ToolChoice("required")
    if choice.type == "none":
        return converted, ToolChoice("none")
    if not choice.name:
        raise ValueError("tool_choice of type 'tool' needs a name")
    return converted, ToolChoice("named", choice.name)


def _chat_request(request: MessagesRequest, engine: DllmEngine) -> ChatRequest:
    if not request.messages:
        raise ValueError("'messages' must contain at least one message")
    if request.max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    options = SamplingOptions(
        temperature=request.temperature if request.temperature is not None else 0.0,
        top_k=request.top_k or 0,
        top_p=request.top_p if request.top_p is not None else 1.0,
        seed=request.seed or 0,
    )
    tools, choice = _tools(request.tools, request.tool_choice)
    output = request.output_config.format if request.output_config else None
    output = output or request.output_format
    response_format = ResponseFormat("json_schema", output.json_schema) if output else ResponseFormat()
    request_id = engine.derive_id(
        "msg_", id_payload(request.model_dump(mode="json", exclude={"stream", "receipt", "previous_receipt"}))
    )
    thinking, budget = None, None
    if request.thinking is not None:
        thinking = request.thinking.type != "disabled"
        if request.thinking.type == "enabled":
            if request.thinking.budget_tokens is None:
                raise ValueError("thinking of type 'enabled' needs budget_tokens")
            budget = request.thinking.budget_tokens
    return ChatRequest(
        messages=_messages(request.system, request.messages),
        max_tokens=request.max_tokens,
        options=options,
        stop=list(request.stop_sequences or ()),
        tools=tools,
        tool_choice=choice,
        response_format=response_format,
        call_id_prefix="toolu_",
        request_id=request_id,
        previous_receipt=request.previous_receipt,
        thinking=thinking,
        max_reasoning_tokens=budget,
    )


def _stop_reason(finish_reason: str, stop_sequence: str | None) -> str:
    if finish_reason == "stop":
        return "stop_sequence" if stop_sequence is not None else "end_turn"
    return _STOP_REASONS[finish_reason]


@router.post("/v1/messages", response_model=None)
def messages(request: MessagesRequest, engine: Engine) -> MessageResponse | JSONResponse | StreamingResponse:
    try:
        chat = _chat_request(request, engine)
        if request.stream:
            events = _events(engine, chat, engine.chat_stream(chat), bool(request.receipt))
            return StreamingResponse(events, media_type="text/event-stream")
        result = engine.chat_completion(chat)
    except ValueError as problem:
        return error(str(problem))
    content: list[ThinkingBlock | TextBlock | ToolUseBlock] = []
    if result.reasoning is not None:
        content.append(ThinkingBlock(thinking=result.reasoning, signature=_signature(result.reasoning)))
    content += [TextBlock(text=result.content)] if result.content else []
    content += [ToolUseBlock(id=c.id, name=c.name, input=_input(c)) for c in result.tool_calls]
    return MessageResponse(
        id=chat.request_id,
        content=content,
        model=engine.model.id,
        stop_reason=_stop_reason(result.finish_reason, result.stop_sequence),
        stop_sequence=result.stop_sequence,
        usage=Usage(
            input_tokens=result.prompt_tokens - result.cached_tokens,
            output_tokens=result.completion_tokens,
            cache_read_input_tokens=result.cached_tokens,
        ),
        **({"receipt": result.receipt} if request.receipt else {}),
    )


def _signature(thinking: str) -> str:
    return "dllm-" + hashlib.sha256(thinking.encode()).hexdigest()


def _input(call: ToolCall) -> dict[str, Any]:
    value = call.arguments_object()
    return value if isinstance(value, dict) else {}


def _event(kind: str, data: dict[str, Any]) -> str:
    return f"event: {kind}\ndata: {json.dumps({'type': kind, **data}, ensure_ascii=False)}\n\n"


def _events(engine: DllmEngine, chat: ChatRequest, stream: ChatStream, receipt: bool = False) -> Iterator[str]:
    start = {
        "id": chat.request_id,
        "type": "message",
        "role": "assistant",
        "content": [],
        "model": engine.model.id,
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {
            "input_tokens": stream.prompt_tokens - stream.cached_tokens,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": stream.cached_tokens,
        },
    }
    yield _event("message_start", {"message": start})
    index = -1
    text_open = False
    thought: list[str] | None = None
    for event in stream:
        if isinstance(event, ReasoningDelta):
            if thought is None:
                index += 1
                thought = []
                block = {"type": "thinking", "thinking": "", "signature": ""}
                yield _event("content_block_start", {"index": index, "content_block": block})
            thought.append(event.text)
            if event.text:
                delta = {"type": "thinking_delta", "thinking": event.text}
                yield _event("content_block_delta", {"index": index, "delta": delta})
            continue
        if thought is not None:
            delta = {"type": "signature_delta", "signature": _signature("".join(thought))}
            yield _event("content_block_delta", {"index": index, "delta": delta})
            yield _event("content_block_stop", {"index": index})
            thought = None
        if isinstance(event, TextDelta):
            if not event.text:
                continue
            if not text_open:
                index += 1
                text_open = True
                yield _event("content_block_start", {"index": index, "content_block": {"type": "text", "text": ""}})
            yield _event("content_block_delta", {"index": index, "delta": {"type": "text_delta", "text": event.text}})
        elif isinstance(event, ToolCallEvent):
            if text_open:
                yield _event("content_block_stop", {"index": index})
                text_open = False
            index += 1
            block = {"type": "tool_use", "id": event.call.id, "name": event.call.name, "input": {}}
            yield _event("content_block_start", {"index": index, "content_block": block})
            arguments = json.dumps(_input(event.call), ensure_ascii=False)
            delta = {"type": "input_json_delta", "partial_json": arguments}
            yield _event("content_block_delta", {"index": index, "delta": delta})
            yield _event("content_block_stop", {"index": index})
        elif isinstance(event, Finished):
            if text_open:
                yield _event("content_block_stop", {"index": index})
            delta = {"stop_reason": _stop_reason(event.finish_reason, event.stop_sequence),
                     "stop_sequence": event.stop_sequence}  # fmt: skip
            data = {"delta": delta, "usage": {"output_tokens": event.completion_tokens}}
            yield _event("message_delta", {**data, "receipt": event.receipt} if receipt else data)
    yield _event("message_stop", {})


@router.post("/v1/messages/count_tokens", response_model=None)
def count_tokens(request: CountTokensRequest, engine: Engine) -> CountTokensResponse | JSONResponse:
    try:
        tools, choice = _tools(request.tools, request.tool_choice)
        prompt = engine.render_chat(_messages(request.system, request.messages), [] if choice.mode == "none" else tools)
    except ValueError as problem:
        return error(str(problem))
    return CountTokensResponse(input_tokens=engine.count_tokens(prompt))
