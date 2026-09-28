"""OpenAI-compatible HTTP API: ``GET /v1/models`` and ``POST /v1/chat/completions``."""

from __future__ import annotations

import argparse
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from etalii_dllm.chat import ChatMessage
from etalii_dllm.engine import DllmEngine, default_engine, use_model_file
from etalii_dllm.sampling import SamplingOptions
from etalii_dllm.server.contracts import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ModelInfo,
    ModelList,
)

Engine = Annotated[DllmEngine, Depends(default_engine)]

app = FastAPI(title="EtAlii.Dllm", version="0.1.0")


def _error(message: str) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": {"message": message, "type": "invalid_request_error"}})


@app.get("/v1/models")
def list_models(engine: Engine) -> ModelList:
    return ModelList(data=[ModelInfo(id=engine.model.id)])


@app.post("/v1/chat/completions", response_model=None)
def chat_completions(request: ChatCompletionRequest, engine: Engine) -> ChatCompletionResponse | JSONResponse:
    if request.stream:
        return _error("Streaming is not supported yet.")
    if not request.messages:
        return _error("'messages' must contain at least one message.")
    try:
        options = SamplingOptions(
            temperature=request.temperature if request.temperature is not None else 0.0,
            top_p=request.top_p if request.top_p is not None else 1.0,
            seed=request.seed or 0,
        )
    except ValueError as error:
        return _error(str(error))

    max_tokens = request.max_completion_tokens or request.max_tokens or 64
    messages = [ChatMessage(m.role, m.content or "") for m in request.messages]
    result = engine.chat(messages, max_tokens, options)

    # The id and timestamp are derived from the output, not from the clock, so identical requests give identical
    # responses.
    return ChatCompletionResponse(
        id="chatcmpl-" + result.fingerprint[:24],
        created=0,
        model=engine.model.id,
        system_fingerprint=engine.system_fingerprint,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=result.text),
                finish_reason=result.finish_reason,
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=len(result.tokens),
            total_tokens=result.prompt_tokens + len(result.tokens),
        ),
    )


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="EtAlii.Dllm OpenAI-compatible server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5080)
    parser.add_argument("--model", help="model.dllm file to serve (default: $DLLM_MODEL, else the placeholder model)")
    args = parser.parse_args()
    use_model_file(args.model)
    uvicorn.run(app, host=args.host, port=args.port)
