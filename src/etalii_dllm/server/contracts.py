"""Subset of the OpenAI Chat Completions and Embeddings wire formats. Unknown request fields are ignored."""

from __future__ import annotations

from collections.abc import Mapping
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
    type: Literal["text", "json_object", "json_schema", "regex"]
    json_schema: JsonSchemaFormat | None = None
    regex: str | None = None
    """Extension: the pattern of a ``regex`` response format."""


class WatermarkOptions(BaseModel):
    """Extension: watermark the answer (docs/watermarks.md)."""

    key: str
    gamma: float = 0.25
    delta: float = 2.0

    def sampling(self) -> dict[str, Any]:
        """The :class:`~etalii_dllm.sampling.SamplingOptions` fields."""
        return {"watermark_key": self.key, "watermark_gamma": self.gamma, "watermark_delta": self.delta}


class GuidanceOptions(BaseModel):
    """Extension: classifier-free guidance away from a negative prompt (docs/api.md#guided-decoding)."""

    negative_prompt: str
    scale: float = 1.5

    def sampling(self) -> dict[str, Any]:
        return {"negative_prompt": self.negative_prompt, "guidance_scale": self.scale}


class ContrastOptions(BaseModel):
    """Extension: contrastive decoding against the server's ``--contrast-model`` (docs/api.md#guided-decoding)."""

    beta: float = 0.5
    alpha: float = 0.1

    def sampling(self) -> dict[str, Any]:
        return {"contrast_beta": self.beta, "contrast_alpha": self.alpha}


def guided(guidance: GuidanceOptions | None, contrast: ContrastOptions | None) -> dict[str, Any]:
    """The :class:`~etalii_dllm.sampling.SamplingOptions` fields of a request's ``guidance`` and ``contrast``."""
    return {**(guidance.sampling() if guidance else {}), **(contrast.sampling() if contrast else {})}


class VoteOptions(BaseModel):
    """Extension: self-consistency voting over ``n`` sampled answers (docs/api.md#voting)."""

    n: int
    extract: str | None = None
    """A regex whose last match (its first group, when it has groups) is the answer that votes."""


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
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    min_p: float | None = None
    """Extension (as in vLLM and llama.cpp)."""
    repetition_penalty: float | None = None
    """Extension (as in vLLM and llama.cpp)."""
    repeat_last_n: int | None = None
    """Extension (as in llama.cpp)."""
    guided_regex: str | None = None
    """Extension (as in vLLM): the same as ``response_format: {"type": "regex", "regex": ...}``."""
    truncation: Literal["auto", "disabled"] | None = None
    """Extension (as in the Responses API): ``auto`` drops the oldest messages that do not fit the context window."""
    context_overflow: Literal["stop", "roll"] | None = None
    """Extension: ``roll`` keeps generating past a full context window (docs/api.md#long-conversations)."""
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] | None = None
    """For thinking models: ``none`` switches thinking off, any other effort on (docs/api.md#reasoning)."""
    chat_template_kwargs: dict[str, Any] | None = None
    """Extension (as in vLLM): ``{"enable_thinking": false}`` switches thinking off."""
    max_reasoning_tokens: int | None = None
    """Extension: the most tokens a thinking model's ``<think>`` block may take (docs/api.md#reasoning)."""
    watermark: WatermarkOptions | None = None
    """Extension: watermark the answer with a key (docs/watermarks.md)."""
    vote: VoteOptions | None = None
    """Extension: sample several answers and return the most common one (docs/api.md#voting)."""
    guidance: GuidanceOptions | None = None
    """Extension: classifier-free guidance away from a negative prompt (docs/api.md#guided-decoding)."""
    contrast: ContrastOptions | None = None
    """Extension: contrastive decoding against the server's amateur model (docs/api.md#guided-decoding)."""


DECODING_CONTROLS = (
    "frequency_penalty",
    "presence_penalty",
    "logit_bias",
    "min_p",
    "repetition_penalty",
    "repeat_last_n",
    "repeat_penalty",
    "guided_regex",
    "truncation",
    "context_overflow",
    "truncate",
    "shift",
    "reasoning_effort",
    "chat_template_kwargs",
    "max_reasoning_tokens",
    "reasoning",
    "think",
    "thinking",
    "watermark",
    "watermark_key",
    "watermark_gamma",
    "watermark_delta",
    "vote",
    "guidance",
    "contrast",
)
"""Request fields added since Phase 21: left out of the payloads ids are derived from while unset, so the ids of
requests without them did not change."""


def thinking_switch(effort: str | None, template_kwargs: Mapping[str, Any] | None) -> bool | None:
    """The thinking switch of an OpenAI-style request: ``chat_template_kwargs.enable_thinking`` wins over
    ``reasoning_effort`` (``none`` is off, any other effort on); ``None`` keeps the model's default."""
    kwargs = dict(template_kwargs or {})
    unknown = sorted(set(kwargs) - {"enable_thinking"})
    if unknown:
        raise ValueError(f"chat_template_kwargs supports only enable_thinking, not {', '.join(unknown)}")
    if "enable_thinking" in kwargs:
        if not isinstance(kwargs["enable_thinking"], bool):
            raise ValueError("chat_template_kwargs.enable_thinking must be true or false")
        return kwargs["enable_thinking"]
    if effort is None:
        return None
    return effort != "none"


def id_payload(dumped: dict[str, Any]) -> dict[str, Any]:
    """``dumped`` without the :data:`DECODING_CONTROLS` that are unset (``None``), at the top level, in an
    ``options`` object or (``regex``) in a ``response_format``."""
    payload = {k: v for k, v in dumped.items() if not (k in DECODING_CONTROLS and v is None)}
    if isinstance(payload.get("options"), dict):
        payload["options"] = id_payload(payload["options"])
    if isinstance(payload.get("response_format"), dict) and payload["response_format"].get("regex") is None:
        payload["response_format"] = {k: v for k, v in payload["response_format"].items() if k != "regex"}
    return payload


class TopLogprob(BaseModel):
    token: str
    logprob: float
    bytes: list[int]


class LogprobEntry(TopLogprob):
    top_logprobs: list[TopLogprob]


class ChoiceLogprobs(BaseModel):
    content: list[LogprobEntry]


class AssistantMessage(BaseModel):
    model_config = ConfigDict(extra="allow")  # ``reasoning_content`` when a thinking model thought

    role: Literal["assistant"] = "assistant"
    content: str | None
    tool_calls: list[ToolCallModel] | None = None
    refusal: None = None


class ChatCompletionChoice(BaseModel):
    model_config = ConfigDict(extra="allow")  # each choice's ``receipt`` when several are requested

    index: int
    message: AssistantMessage
    logprobs: ChoiceLogprobs | None = None
    finish_reason: str


class PromptTokensDetails(BaseModel):
    cached_tokens: int = 0
    """Prompt tokens served from the prompt cache (depends on earlier requests; the output does not)."""


class ChatCompletionUsage(BaseModel):
    model_config = ConfigDict(extra="allow")  # ``completion_tokens_details`` when a thinking model thought

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
    model_config = ConfigDict(extra="allow")  # ``reasoning_content`` in a thinking model's reasoning chunks

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


class WatermarkDetectRequest(BaseModel):
    text: str
    key: str
    gamma: float = 0.25


class RerankDocument(BaseModel):
    text: str


class RerankRequest(BaseModel):
    """``POST /v1/rerank`` in the shape Cohere and Jina use (and llama.cpp's server)."""

    query: str
    documents: list[str | RerankDocument]
    model: str | None = None
    top_n: int | None = None
    return_documents: bool = True
    instruction: str | None = None
    """Extension: what makes a document relevant (default: that it answers the query)."""


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: RerankDocument | None = None


class RerankUsage(BaseModel):
    total_tokens: int


class RerankResponse(BaseModel):
    id: str
    model: str
    results: list[RerankResult]
    usage: RerankUsage
