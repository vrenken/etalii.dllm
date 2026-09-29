import json
import os
import subprocess
import sys

import anyio
import pytest
from mcp import Client
from model_fixtures import TINY_LLAMA_CONFIG, write_hf_checkpoint

from etalii_dllm import engine as engine_module
from etalii_dllm import mcp_server, numerics
from etalii_dllm.chat import ChatMessage
from etalii_dllm.engine import ChatRequest, ResponseFormat, default_engine
from etalii_dllm.importing import import_model
from etalii_dllm.modelfile import ModelFile, TensorSource, write_model_file
from etalii_dllm.sampling import SamplingOptions

MESSAGES = [
    {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}},
    },
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "generate", "arguments": {"prompt": "Hi", "max_tokens": 8, "temperature": 0.5, "seed": 3}},
    },
    {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "chat",
            "arguments": {
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 40,
                "json_schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
            },
        },
    },
    {"jsonrpc": "2.0", "id": 5, "method": "resources/list"},
    {"jsonrpc": "2.0", "id": 6, "method": "resources/read", "params": {"uri": "dllm://model"}},
    {"jsonrpc": "2.0", "id": 7, "method": "prompts/list"},
    {
        "jsonrpc": "2.0",
        "id": 8,
        "method": "prompts/get",
        "params": {"name": "translate", "arguments": {"text": "Good morning", "language": "Dutch"}},
    },
]


def run_session() -> dict[int, dict]:
    """Sends each request and waits for its response before the next, so the server never sees EOF mid-request."""
    process = subprocess.Popen(
        [sys.executable, "-c", "from etalii_dllm.mcp_server import main; main()"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert process.stdin is not None and process.stdout is not None
    responses: dict[int, dict] = {}
    try:
        for message in MESSAGES:
            process.stdin.write(json.dumps(message) + "\n")
            process.stdin.flush()
            if "id" not in message:
                continue
            while message["id"] not in responses:
                line = process.stdout.readline()
                assert line, "MCP server exited before answering"
                response = json.loads(line)
                if "id" in response:
                    responses[response["id"]] = response
    finally:
        process.stdin.close()
        process.wait(timeout=30)
    return responses


def test_mcp_server_lists_and_runs_tools_deterministically():
    first = run_session()
    tools = {t["name"]: t for t in first[2]["result"]["tools"]}
    assert set(tools) == {"chat", "generate", "model_info"}
    assert tools["generate"]["annotations"]["idempotentHint"] is True

    second = run_session()
    assert first[3]["result"]["content"] == second[3]["result"]["content"]
    assert first[4]["result"]["content"] == second[4]["result"]["content"]
    answer = json.loads(first[4]["result"]["content"][0]["text"])
    assert isinstance(answer["ok"], bool)


def test_mcp_server_offers_resources_and_prompts():
    responses = run_session()
    uris = [r["uri"] for r in responses[5]["result"]["resources"]]
    assert uris == ["dllm://model", "dllm://model/chat-template", "dllm://determinism"]
    card = json.loads(responses[6]["result"]["contents"][0]["text"])
    assert card["id"] == "dllm-bigram-257-42" and card["system_fingerprint"].startswith("fp_")

    prompts = {p["name"]: p for p in responses[7]["result"]["prompts"]}
    assert set(prompts) == {"summarize", "translate", "extract_json"}
    assert {a["name"]: a["required"] for a in prompts["translate"]["arguments"]} == {"text": True, "language": True}
    (message,) = responses[8]["result"]["messages"]
    assert message["role"] == "user"
    assert message["content"]["text"].startswith("Translate the following text into Dutch.")
    assert message["content"]["text"].endswith("Good morning")


# In-process sessions through the official client, so the handlers themselves run (and are measured) here.


def in_memory(steps):
    """Runs ``steps(client)`` against the server in this process and returns what it returns."""

    async def go():
        async with Client(mcp_server.server, cache=None) as client:
            return await steps(client)

    return anyio.run(go)


@pytest.fixture
def placeholder(monkeypatch):
    monkeypatch.delenv(engine_module.MODEL_ENVIRONMENT_VARIABLE, raising=False)
    default_engine.cache_clear()
    yield default_engine()
    default_engine.cache_clear()


FINE_TUNING = {"base_fingerprint": "0" * 64, "steps_completed": 3}


@pytest.fixture(scope="module")
def fine_tuned_path(tmp_path_factory):
    """A tiny imported model rewritten with a ``fine_tuning`` section, as ``dllm finetune`` records one."""
    pytest.importorskip("tokenizers")
    from test_bpe import smollm2_style
    from test_chat_template import SMOLLM2

    reference = smollm2_style()
    directory = tmp_path_factory.mktemp("mcp")
    config = {**TINY_LLAMA_CONFIG, "vocab_size": reference.get_vocab_size()}
    write_hf_checkpoint(directory / "checkpoint", config, tokenizer_json=json.loads(reference.to_str()))
    tokenizer_config = {"chat_template": SMOLLM2, "eos_token": "<|im_end|>", "bos_token": None}
    (directory / "checkpoint" / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")
    import_model(directory / "checkpoint", directory / "base.dllm", repository="example/tiny")
    base = ModelFile(directory / "base.dllm")
    tensors = {
        name: TensorSource(tuple(array.shape), lambda array=array: array) for name, array in base.tensors.items()
    }
    metadata = {key: base.header.get(key) for key in ("source", "licence", "tokenizer", "chat_template")}
    write_model_file(directory / "tuned.dllm", base.config, tensors, {**metadata, "fine_tuning": FINE_TUNING})
    return directory / "tuned.dllm"


def test_tools_answer_like_the_engine(placeholder):
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}

    async def steps(client):
        return (
            await client.call_tool("generate", {"prompt": "Hi", "max_tokens": 6, "temperature": 0.7, "seed": 5}),
            await client.call_tool("chat", {"messages": [{"content": "Hi"}], "max_tokens": 8}),
            await client.call_tool("chat", {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 8}),
            await client.call_tool(
                "chat", {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 40, "json_schema": schema}
            ),
            await client.call_tool("model_info", {}),
        )

    generated, defaulted, explicit, constrained, info = in_memory(steps)
    options = SamplingOptions(temperature=0.7, seed=5)
    assert generated.content[0].text == placeholder.complete("Hi", 6, options).text
    # A message without a role is a user message.
    assert defaulted.content[0].text == explicit.content[0].text
    expected = placeholder.chat_completion(ChatRequest([ChatMessage("user", "Hi")], 8, SamplingOptions()))
    assert explicit.content[0].text == expected.content
    assert isinstance(json.loads(constrained.content[0].text)["ok"], bool)
    constrained_request = ChatRequest(
        [ChatMessage("user", "Hi")], 40, SamplingOptions(), response_format=ResponseFormat("json_schema", schema)
    )
    assert constrained.content[0].text == placeholder.chat_completion(constrained_request).content
    assert not any(r.is_error for r in (generated, defaulted, explicit, constrained, info))
    assert info.content[0].text == (
        f"model: {placeholder.model.id}\nsystem_fingerprint: {placeholder.system_fingerprint}"
    )


def test_placeholder_resources_describe_the_fallback_format(placeholder):
    async def steps(client):
        return [
            (await client.read_resource(uri)).contents[0]
            for uri in ("dllm://model", "dllm://model/chat-template", "dllm://determinism")
        ]

    card, template, determinism = in_memory(steps)
    assert json.loads(card.text) == {
        "chat_template": False,
        "id": placeholder.model.id,
        "system_fingerprint": placeholder.system_fingerprint,
        "vocabulary_size": placeholder.model.vocabulary_size,
    }
    assert template.mime_type == "text/plain" and template.text == mcp_server.FALLBACK_TEMPLATE
    assert determinism.mime_type == "text/markdown" and determinism.text == mcp_server.DETERMINISM


def test_prompts_fill_in_their_arguments(placeholder):
    async def steps(client):
        return [
            (await client.get_prompt(name, arguments)).messages[0].content.text
            for name, arguments in (
                ("summarize", {"text": "A long story."}),
                ("summarize", {"text": "A long story.", "max_words": "10"}),
                ("extract_json", {"text": "Ada, 36", "fields": " name, ,age ,"}),
                ("translate", {"text": "Good morning", "language": "Dutch"}),
            )
        ]

    default_length, short, extract, translation = in_memory(steps)
    assert translation == "Translate the following text into Dutch. Answer with the translation only.\n\nGood morning"
    assert default_length == "Summarize the following text in at most 60 words.\n\nA long story."
    assert short.startswith("Summarize the following text in at most 10 words.")
    # Blank and padded field names are dropped and trimmed.
    assert "exactly these keys: name, age. Use null" in extract and extract.endswith("\n\nAda, 36")


def test_model_card_reports_fine_tuning(fine_tuned_path, monkeypatch):
    monkeypatch.setenv(engine_module.MODEL_ENVIRONMENT_VARIABLE, str(fine_tuned_path))
    default_engine.cache_clear()
    try:
        card = json.loads(mcp_server.model_card())
    finally:
        default_engine.cache_clear()
    assert card["fine_tuning"] == FINE_TUNING
    assert card["source"]["repository"] == "example/tiny"
    assert card["architecture"]["layers"] == TINY_LLAMA_CONFIG["num_hidden_layers"]


def test_main_selects_the_model_and_serves_stdio(fine_tuned_path, monkeypatch):
    for variable in (
        engine_module.MODEL_ENVIRONMENT_VARIABLE,
        engine_module.QUANTIZE_ENVIRONMENT_VARIABLE,
        engine_module.DEVICE_ENVIRONMENT_VARIABLE,
        engine_module.PROMPT_CACHE_ENVIRONMENT_VARIABLE,
    ):
        # setenv first, so the variables main() sets are removed again afterwards (delenv of an unset one is not).
        monkeypatch.setenv(variable, "")
        monkeypatch.delenv(variable)
    transports = []
    monkeypatch.setattr(mcp_server.server, "run", transports.append)
    threads = numerics.threads()
    try:
        mcp_server.main(
            ["--model", str(fine_tuned_path), "--quantize", "q8_0", "--threads", "1", "--prompt-cache", "0"]
        )
        assert transports == ["stdio"]
        assert numerics.threads() == 1
        assert os.environ[engine_module.MODEL_ENVIRONMENT_VARIABLE] == str(fine_tuned_path)
        assert os.environ[engine_module.QUANTIZE_ENVIRONMENT_VARIABLE] == "q8_0"
        assert os.environ[engine_module.PROMPT_CACHE_ENVIRONMENT_VARIABLE] == "0"
        assert default_engine().model.id == "example/tiny"
    finally:
        numerics.set_threads(threads)
        default_engine.cache_clear()


def test_main_rejects_unknown_options(capsys):
    with pytest.raises(SystemExit) as error:
        mcp_server.main(["--device", "tpu"])
    assert error.value.code == 2
    assert "invalid choice: 'tpu'" in capsys.readouterr().err
