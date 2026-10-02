"""Phase 37: exact fill-in-the-middle. A prompt and a suffix are rendered with the model's own FIM tokens (#247), on
the completions API, Ollama and ``dllm generate --suffix`` (#248), recorded in receipts and checked against the
reference implementation (#249), with golden values (#250)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from golden_values import FIM_FINGERPRINTS
from model_fixtures import TINY_LLAMA_CONFIG, write_hf_checkpoint
from test_chat_template import SMOLLM2
from test_cli import isolated_environment  # noqa: F401 - autouse fixture: the CLI tests set DLLM_MODEL
from test_engine_import import model_path  # noqa: F401 - fixture: a model without FIM tokens

from etalii_dllm import receipts, verify
from etalii_dllm.engine import ChatRequest, DllmEngine
from etalii_dllm.importing import import_model
from etalii_dllm.infill import FimTokens, fim_tokens
from etalii_dllm.sampling import GREEDY, SamplingOptions

tokenizers = pytest.importorskip("tokenizers")

SAMPLED = SamplingOptions(temperature=0.9, seed=5)
PREFIX = "def add(a, b):\n    return "
SUFFIX = "\n\nprint(add(1, 2))\n"
FIM = ["<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>"]


@pytest.fixture(scope="module")
def fim_path(tmp_path_factory):
    """A tiny model whose Qwen-style tokenizer has the FIM tokens (not special, as in Qwen2.5)."""
    from test_bpe import qwen2_style
    from tokenizers import AddedToken

    tokenizer = qwen2_style()
    tokenizer.add_tokens([AddedToken(token, special=False, normalized=False) for token in FIM])
    directory = tmp_path_factory.mktemp("fim")
    config = {**TINY_LLAMA_CONFIG, "vocab_size": tokenizer.get_vocab_size(), "eos_token_id": 2}
    write_hf_checkpoint(directory / "checkpoint", config, tokenizer_json=json.loads(tokenizer.to_str()))
    tokenizer_config = {"chat_template": SMOLLM2, "eos_token": "<|im_end|>", "bos_token": None}
    (directory / "checkpoint" / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")
    import_model(directory / "checkpoint", directory / "fim.dllm", repository="example/tiny-fim")
    return directory / "fim.dllm"


@pytest.fixture(scope="module")
def coder(fim_path) -> DllmEngine:
    return DllmEngine.from_model_file(fim_path)


@pytest.fixture(scope="module")
def plain(model_path) -> DllmEngine:  # noqa: F811
    return DllmEngine.from_model_file(model_path)


# The FIM tokens


class _Vocabulary:
    def __init__(self, tokens: dict[str, int]) -> None:
        self.tokens = tokens

    def token_to_id(self, token: str) -> int | None:
        return self.tokens.get(token)


def test_fim_tokens_are_found_under_either_spelling():
    qwen = _Vocabulary({"<|fim_prefix|>": 5, "<|fim_suffix|>": 6, "<|fim_middle|>": 7, "<|endoftext|>": 0})
    assert FimTokens.of(qwen) == FimTokens(5, 6, 7, frozenset({0, 5, 6, 7}))
    starcoder = _Vocabulary({"<fim_prefix>": 1, "<fim_suffix>": 3, "<fim_middle>": 2, "<fim_pad>": 4,
                             "<file_sep>": 9})  # fmt: skip
    assert FimTokens.of(starcoder) == FimTokens(1, 3, 2, frozenset({1, 2, 3, 4, 9}))
    assert FimTokens.of(_Vocabulary({"<|fim_prefix|>": 5, "<|fim_suffix|>": 6})) is None
    assert FimTokens.of(object()) is None  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no fill-in-the-middle tokens"):
        fim_tokens(object())  # type: ignore[arg-type]


def test_the_prompt_is_built_from_token_ids(coder):
    tokens = fim_tokens(coder.tokenizer)
    encode = coder.tokenizer.encode
    prompt = tokens.prompt(coder.tokenizer, PREFIX, SUFFIX)
    assert prompt == [tokens.prefix, *encode(PREFIX), tokens.suffix, *encode(SUFFIX), tokens.middle]
    assert {encode(name)[0] for name in FIM} <= tokens.ends | {tokens.prefix, tokens.suffix, tokens.middle}
    assert encode("<|endoftext|>")[0] in tokens.ends


# The engine


def test_a_middle_is_a_completion_of_the_fim_prompt(coder):
    tokens = fim_tokens(coder.tokenizer)
    prompt = tokens.prompt(coder.tokenizer, PREFIX, SUFFIX)
    middle = coder.complete_stream(PREFIX, 12, SAMPLED, suffix=SUFFIX).result()
    assert middle.prompt_tokens == len(prompt)
    direct = coder._generator.stream(prompt, 12, SAMPLED, stop_tokens=tokens.ends).result()
    assert middle.tokens == direct.tokens and middle.text == direct.text
    assert middle.fingerprint == FIM_FINGERPRINTS["sampled"]
    greedy = coder.complete_stream(PREFIX, 12, GREEDY, suffix=SUFFIX).result()
    assert greedy.fingerprint == FIM_FINGERPRINTS["greedy"]
    streamed = "".join(step.text for step in coder.complete_stream(PREFIX, 12, SAMPLED, suffix=SUFFIX))
    assert streamed == middle.text
    assert coder.complete_stream(PREFIX, 12, SAMPLED).result().prompt_tokens == len(coder.tokenizer.encode(PREFIX))


def test_a_middle_ends_at_a_fim_token(coder):
    tokens = fim_tokens(coder.tokenizer)
    end = coder.tokenizer.encode("<|file_sep|>")[0]
    pushed = SamplingOptions(logit_bias=((end, 100.0),))
    middle = coder.complete_stream(PREFIX, 8, pushed, suffix=SUFFIX).result()
    assert not middle.tokens and middle.finish_reason == "stop" and middle.text == ""
    assert end in tokens.ends
    assert coder.complete_stream(PREFIX, 3, pushed).result().tokens == (end,) * 3  # without a suffix it is text


def test_what_a_middle_cannot_be_combined_with(coder, plain):
    with pytest.raises(ValueError, match="no fill-in-the-middle tokens"):
        plain.complete_stream(PREFIX, 4, GREEDY, suffix=SUFFIX)
    with pytest.raises(ValueError, match="token healing"):
        coder.complete_stream(PREFIX, 4, GREEDY, suffix=SUFFIX, token_healing=True)
    with pytest.raises(ValueError, match="negative prompt"):
        coder.complete_stream(PREFIX, 4, SamplingOptions(negative_prompt="x"), suffix=SUFFIX)
    from etalii_dllm.chat import ChatMessage

    with pytest.raises(ValueError, match="needs a raw prompt"):
        coder.chat_stream(ChatRequest([ChatMessage("user", "hi")], 4, suffix=SUFFIX))
    from etalii_dllm import beam

    with pytest.raises(ValueError, match="fill in the middle"):
        beam.search(coder, ChatRequest([], 4, prompt=PREFIX, suffix=SUFFIX), 2)


def test_receipts_record_the_suffix_and_replay_it(coder):
    request = ChatRequest([], 8, SAMPLED, prompt=PREFIX, suffix=SUFFIX)
    record = receipts.request_record(request)
    assert record["suffix"] == SUFFIX and receipts.request_from_record(record) == request
    assert "suffix" not in receipts.request_record(ChatRequest([], 8, SAMPLED, prompt=PREFIX))
    result = coder.chat_completion(request)
    assert result.content == coder.complete_stream(PREFIX, 8, SAMPLED, suffix=SUFFIX).result().text
    assert result.receipt is not None and receipts.verify(coder, result.receipt).ok


def test_the_reference_implementation_fills_the_same_middle(coder):
    check = verify.check_reference(coder, max_tokens=6)
    assert check.results["infilled"] == "equal", check.results
    assert check.equal


def test_the_reference_refuses_a_vocabulary_without_fim_tokens():
    from etalii_dllm import reference

    with pytest.raises(ValueError, match="no fill-in-the-middle tokens"):
        reference.fill_in_the_middle(lambda token: None, [1], [2])
    prompt, ends = reference.fill_in_the_middle({"<fim_prefix>": 1, "<fim_suffix>": 2, "<fim_middle>": 3}.get, [9], [8])
    assert prompt == [1, 9, 2, 8, 3] and ends == [1, 2, 3]


# Front ends


def test_the_completions_api_and_ollama_take_a_suffix(coder, plain):
    from etalii_dllm.engine import default_engine
    from etalii_dllm.server.app import app

    expected = coder.complete_stream(PREFIX, 12, SAMPLED, suffix=SUFFIX).result()
    app.dependency_overrides[default_engine] = lambda: coder
    try:
        client = TestClient(app)
        body = {"prompt": PREFIX, "max_tokens": 12, "temperature": 0.9, "seed": 5}
        unfilled = client.post("/v1/completions", json=body).json()
        filled = client.post("/v1/completions", json={**body, "suffix": SUFFIX}).json()
        assert filled["choices"][0]["text"] == expected.text and filled["id"] != unfilled["id"]
        assert filled["usage"]["prompt_tokens"] == expected.prompt_tokens
        assert client.post("/v1/completions", json={**body, "suffix": ""}).json()["choices"] == unfilled["choices"]
        with client.stream("POST", "/v1/completions", json={**body, "suffix": SUFFIX, "stream": True}) as response:
            chunks = [json.loads(line[6:]) for line in response.iter_lines() if line.startswith("data: {")]
        assert "".join(c["choices"][0]["text"] for c in chunks if c["choices"]) == expected.text
        echoed = client.post("/v1/completions", json={**body, "suffix": SUFFIX, "echo": True})
        assert echoed.status_code == 400 and "echo" in echoed.json()["error"]["message"]
        ollama = client.post(
            "/api/generate",
            json={"prompt": PREFIX, "suffix": SUFFIX, "stream": False,
                  "options": {"num_predict": 12, "temperature": 0.9, "seed": 5}},
        )  # fmt: skip
        assert ollama.json()["response"] == expected.text
        assert client.post("/api/generate", json={"prompt": PREFIX, "template": "x"}).status_code == 400
        app.dependency_overrides[default_engine] = lambda: plain
        refused = client.post("/v1/completions", json={**body, "suffix": SUFFIX})
        assert refused.status_code == 400 and "fill-in-the-middle" in refused.json()["error"]["message"]
    finally:
        app.dependency_overrides.clear()


def test_cli_suffix(fim_path, model_path, capsys):  # noqa: F811
    from etalii_dllm.cli import main

    args = ["--model", str(fim_path), "generate", "--prompt", PREFIX, "--suffix", SUFFIX, "--max-tokens", "12"]
    assert main([*args, "--temperature", "0.9", "--seed", "5"]) == 0
    assert FIM_FINGERPRINTS["sampled"] in capsys.readouterr().err
    assert main([*args, "--beams", "2"]) == 2
    assert main(["--model", str(model_path), "generate", "--prompt", PREFIX, "--suffix", SUFFIX]) == 2
    assert "fill-in-the-middle" in capsys.readouterr().err
