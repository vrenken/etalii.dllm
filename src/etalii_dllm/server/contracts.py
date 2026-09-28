"""Subset of the OpenAI Chat Completions wire format."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatCompletionMessage(BaseModel):
    role: str
    content: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatCompletionMessage] = Field(default_factory=list)
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    """OpenAI documents ``seed`` as best effort; here it is a guarantee."""
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stream: bool | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    system_fingerprint: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "etalii"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]
