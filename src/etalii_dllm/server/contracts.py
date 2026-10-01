"""Subset of the OpenAI Chat Completions and Embeddings wire formats. Unknown request fields are ignored."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCallModel(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ContentPart(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    text: str | None = None


class ChatCompletionMessage(BaseModel):
    role: str
    content: str | list[ContentPart] | None = None
    tool_calls: list[ToolCallModel] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class FunctionDefinition(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None


class ToolDefinition(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class FunctionName(BaseModel):
    name: str


class NamedToolChoice(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionName


class JsonSchemaFormat(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str = "response"
    description: str | None = None
    json_schema: dict[str, Any] | None = Field(default=None, alias="schema")
    strict: bool | None = None


class ResponseFormatModel(BaseModel):
    type: Literal["text", "json_object", "json_schema"]
    json_schema: JsonSchemaFormat | None = None


class StreamOptions(BaseModel):
    include_usage: bool | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatCompletionMessage] = Field(default_factory=list)
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    """Not in the OpenAI API; accepted as an extension."""
    seed: int | None = None
    """OpenAI documents ``seed`` as best effort; here it is a guarantee."""
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stop: str | list[str] | None = None
    n: int | None = None
    stream: bool | None = None
    stream_options: StreamOptions | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: Literal["none", "auto", "required"] | NamedToolChoice | None = None
    parallel_tool_calls: bool | None = None
    response_format: ResponseFormatModel | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    receipt: bool | None = None
    """Extension: add a ``receipt`` to the response (see docs/receipts.md)."""
    previous_receipt: str | None = None
    """Extension: the receipt id of the conversation's previous turn, recorded as the new receipt's ``previous``."""


class TopLogprob(BaseModel):
    token: str
    logprob: float
    bytes: list[int]


class LogprobEntry(TopLogprob):
    top_logprobs: list[TopLogprob]


class ChoiceLogprobs(BaseModel):
    content: list[LogprobEntry]


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None
    tool_calls: list[ToolCallModel] | None = None
    refusal: None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: AssistantMessage
    logprobs: ChoiceLogprobs | None = None
    finish_reason: str


class PromptTokensDetails(BaseModel):
    cached_tokens: int = 0
    """Prompt tokens served from the prompt cache (depends on earlier requests; the output does not)."""


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: PromptTokensDetails = PromptTokensDetails()


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")  # ``receipt`` when requested

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    system_fingerprint: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage


class DeltaFunctionCall(BaseModel):
    name: str | None = None
    arguments: str | None = None


class DeltaToolCall(BaseModel):
    index: int
    id: str | None = None
    type: Literal["function"] | None = None
    function: DeltaFunctionCall


class ChunkDelta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None
    tool_calls: list[DeltaToolCall] | None = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: ChunkDelta
    logprobs: ChoiceLogprobs | None = None
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    model_config = ConfigDict(extra="allow")  # ``receipt`` on the finishing chunk when requested

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = 0
    model: str
    system_fingerprint: str
    choices: list[ChunkChoice]
    usage: ChatCompletionUsage | None = None


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "etalii"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]


class EmbeddingsRequest(BaseModel):
    input: str | list[str] | list[int] | list[list[int]]
    model: str | None = None
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = None
    input_type: str | None = None
    """Extension: the embedding model's prompt to prefix (e.g. ``"query"`` or ``"document"``)."""


class EmbeddingData(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float] | str


class EmbeddingsUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingsUsage
