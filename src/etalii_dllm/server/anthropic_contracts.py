"""Subset of the Anthropic Messages wire format. Unknown request fields are ignored."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RequestBlock(BaseModel):
    """A content block of a request message: ``text``, ``tool_use`` or ``tool_result`` (others are refused)."""

    model_config = ConfigDict(extra="allow")

    type: str
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: Any = None
    tool_use_id: str | None = None
    content: str | list[RequestBlock] | None = None
    is_error: bool | None = None


class InputMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[RequestBlock]


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    description: str | None = None
    input_schema: dict[str, Any] | None = None
    type: str | None = None


class ToolChoiceModel(BaseModel):
    type: Literal["auto", "any", "tool", "none"]
    name: str | None = None
    disable_parallel_tool_use: bool | None = None


class OutputFormat(BaseModel):
    type: Literal["json_schema"]
    json_schema: dict[str, Any] = Field(alias="schema")


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    format: OutputFormat | None = None


class ThinkingConfig(BaseModel):
    type: Literal["enabled", "disabled", "adaptive"]
    budget_tokens: int | None = None
    """With ``enabled``: the most tokens the thinking may take (an exact token count here)."""


class MessagesRequest(BaseModel):
    model: str | None = None
    max_tokens: int
    messages: list[InputMessage]
    system: str | list[RequestBlock] | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    stream: bool | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: ToolChoiceModel | None = None
    output_config: OutputConfig | None = None
    output_format: OutputFormat | None = None
    """The structured outputs beta spelling of ``output_config.format``."""
    seed: int | None = None
    """Not in the Anthropic API (pass it with ``extra_body``); defaults to 0."""
    receipt: bool | None = None
    """Extension: add a ``receipt`` to the response (see docs/receipts.md)."""
    previous_receipt: str | None = None
    """Extension: the receipt id of the conversation's previous turn, recorded as the new receipt's ``previous``."""
    thinking: ThinkingConfig | None = None
    """For thinking models: ``enabled`` (with ``budget_tokens``), ``adaptive`` or ``disabled``
    (docs/api.md#reasoning)."""


class CountTokensRequest(BaseModel):
    model: str | None = None
    messages: list[InputMessage]
    system: str | list[RequestBlock] | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: ToolChoiceModel | None = None


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ThinkingBlock(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str
    """A hash of the thinking (derived from content; nothing to verify with Anthropic)."""


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class Usage(BaseModel):
    input_tokens: int
    """Prompt tokens not read from the prompt cache (as Anthropic counts them)."""
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    """Prompt tokens read from the prompt cache (depends on earlier requests; the output does not)."""


class MessageResponse(BaseModel):
    model_config = ConfigDict(extra="allow")  # ``receipt`` when requested

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[ThinkingBlock | TextBlock | ToolUseBlock]
    model: str
    stop_reason: str | None
    stop_sequence: str | None
    usage: Usage


class CountTokensResponse(BaseModel):
    input_tokens: int
