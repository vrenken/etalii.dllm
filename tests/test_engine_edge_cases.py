"""Edge cases of the engine, the generator, the batcher and the decoder: invalid requests and settings are refused
with a reason, and the less common paths (models without a tokenizer, template or hidden states, a KV cache
without batching, constraints that allow nothing, zero-length embeddings) behave as documented."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from model_fixtures import TINY_LLAMA_CONFIG, tiny_config, write_hf_checkpoint

from etalii_dllm import engine as engine_module
from etalii_dllm.batching import Batcher
from etalii_dllm.chat import ChatMessage, render
from etalii_dllm.engine import (
    ChatRequest,
    DllmEngine,
    Finished,
    ResponseFormat,
    TextDelta,
    configured_device,
    configured_quantization,
    default_engine,
    use_model_file,
)
from etalii_dllm.generation import Generator
from etalii_dllm.grammar import Grammar, TokenConstraint, TokenTrie
from etalii_dllm.importing import import_model
from etalii_dllm.modelfile import ModelFile, TensorSource, write_model_file
from etalii_dllm.models import BigramModel
from etalii_dllm.numerics import linear
from etalii_dllm.sampling import GREEDY, SamplingOptions
from etalii_dllm.tokenization import ByteTokenizer
from etalii_dllm.tools import Tool
from etalii_dllm.transformer import Transformer

# Response formats and requests


@pytest.mark.parametrize("kind", ["xml", "JSON_OBJECT", ""])
def test_unknown_response_formats_are_refused(kind):
    with pytest.raises(ValueError, match=f"unknown response format {kind!r}"):
        ResponseFormat(kind)


@pytest.mark.parametrize(("kind", "schema"), [("json_schema", None), ("text", {"type": "string"})])
def test_a_schema_goes_with_json_schema_only(kind, schema):
    with pytest.raises(ValueError, match="a json_schema response format needs a schema"):
        ResponseFormat(kind, schema)


def test_response_format_grammars():
    assert ResponseFormat().grammar() is None
    assert ResponseFormat("json_object").grammar().matcher().matches(b'{"a":1}')
    schema = ResponseFormat("json_schema", {"type": "integer"}).grammar().matcher()
    assert schema.matches(b"12") and not schema.matches(b'"12"')


@pytest.fixture(scope="module")
def placeholder() -> DllmEngine:
    return DllmEngine.create_default()


def test_raw_prompts_cannot_use_tools(placeholder):
    request = ChatRequest([], 4, prompt="raw text", tools=[Tool("lookup")])
    with pytest.raises(ValueError, match="a raw prompt cannot use tools"):
        placeholder.chat_stream(request)


def test_raw_prompt_is_used_as_is(placeholder):
    request = ChatRequest([ChatMessage("user", "ignored")], 8, SamplingOptions(temperature=0.9, seed=2), prompt="raw")
    stream = placeholder.chat_stream(request)
    assert stream.prompt_tokens == len("raw")
    text = "".join(event.text for event in stream if isinstance(event, TextDelta))
    assert text == placeholder.complete("raw", 8, SamplingOptions(temperature=0.9, seed=2)).text


# Embeddings


def test_embedding_dimensions_must_be_in_range(placeholder):
    size = placeholder.model.vocabulary_size
    for dimensions in (0, -1, size + 1):
        with pytest.raises(ValueError, match=f"dimensions must be between 1 and {size}"):
            placeholder.embed("text", dimensions)
    full = placeholder.embed("text", size).vector
    assert full.tobytes() == placeholder.embed("text").vector.tobytes()


class NoHiddenStates:
    """A language model that only predicts tokens."""

    def __init__(self) -> None:
        self._model = BigramModel(ByteTokenizer.vocabulary_size, 1)
        self.vocabulary_size = self._model.vocabulary_size
        self.id = "no-hidden-states"
        self.weights_fingerprint = self._model.weights_fingerprint

    def forward(self, tokens):
        return self._model.forward(tokens)


class ZeroHiddenStates(NoHiddenStates):
    def hidden_states(self, tokens):
        return np.zeros((len(tokens), 8), dtype=np.float32)


def test_embeddings_need_hidden_states():
    engine = DllmEngine(NoHiddenStates(), ByteTokenizer(), "fp_test")
    with pytest.raises(ValueError, match="model no-hidden-states does not provide hidden states"):
        engine.embed("text")
    assert engine.complete("text", 3, GREEDY).tokens  # generation still works


def test_zero_embedding_is_not_normalised():
    engine = DllmEngine(ZeroHiddenStates(), ByteTokenizer(), "fp_test")
    embedding = engine.embed([1, 2, 3], dimensions=4)
    assert embedding.tokens == 3
    assert embedding.vector.dtype == np.float32 and embedding.vector.tolist() == [0.0] * 4  # no division by zero


# Configuration from the environment


@pytest.mark.parametrize("value", ["q4_0", "int8", "fp16"])
def test_unknown_quantization_setting_is_refused(monkeypatch, value):
    monkeypatch.setenv(engine_module.QUANTIZE_ENVIRONMENT_VARIABLE, value.upper())
    with pytest.raises(ValueError, match=f"DLLM_QUANTIZE='{value}'; supported: none, q8_0"):
        configured_quantization()


@pytest.mark.parametrize(("value", "expected"), [("", None), (" None ", None), ("Q8_0", "q8_0")])
def test_quantization_setting(monkeypatch, value, expected):
    monkeypatch.setenv(engine_module.QUANTIZE_ENVIRONMENT_VARIABLE, value)
    assert configured_quantization() == expected


def test_use_model_file_sets_every_given_option(monkeypatch, request):
    request.addfinalizer(default_engine.cache_clear)
    names = ("MODEL", "QUANTIZE", "DEVICE", "PROMPT_CACHE", "ADAPTER")
    for name in names:  # restored afterwards by monkeypatch
        monkeypatch.setenv(getattr(engine_module, f"{name}_ENVIRONMENT_VARIABLE"), "unchanged")
    use_model_file(None, quantize="none", device="cpu", prompt_cache=0)
    assert os.environ[engine_module.DEVICE_ENVIRONMENT_VARIABLE] == "cpu"
    assert os.environ[engine_module.QUANTIZE_ENVIRONMENT_VARIABLE] == "none"
    assert os.environ[engine_module.PROMPT_CACHE_ENVIRONMENT_VARIABLE] == "0"
    assert os.environ[engine_module.MODEL_ENVIRONMENT_VARIABLE] == "unchanged"  # not given: left alone
    assert os.environ[engine_module.ADAPTER_ENVIRONMENT_VARIABLE] == "unchanged"
    assert configured_device() == "cpu" and configured_quantization() is None
    monkeypatch.setenv(engine_module.DEVICE_ENVIRONMENT_VARIABLE, "tpu")
    with pytest.raises(ValueError, match="DLLM_DEVICE='tpu'; supported: cpu, cuda"):
        configured_device()


# Model files without a tokenizer or a chat template


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    """The tiny Llama with SmolLM2's byte-level BPE tokenizer and chat template (as ``test_engine_import``)."""
    pytest.importorskip("tokenizers")
    from test_bpe import smollm2_style
    from test_chat_template import SMOLLM2

    reference = smollm2_style()
    directory = tmp_path_factory.mktemp("edge")
    config = {**TINY_LLAMA_CONFIG, "vocab_size": reference.get_vocab_size(), "eos_token_id": 2}
    write_hf_checkpoint(directory / "checkpoint", config, tokenizer_json=json.loads(reference.to_str()))
    tokenizer_config = {"chat_template": SMOLLM2, "eos_token": "<|im_end|>", "bos_token": None}
    (directory / "checkpoint" / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")
    import_model(directory / "checkpoint", directory / "tiny.dllm", repository="example/tiny-chat")
    return directory / "tiny.dllm"


def rewrite_model(original, target, **metadata) -> None:
    """``original`` with some metadata sections replaced (same weights)."""
    file = ModelFile(original)
    tensors = {name: TensorSource(tuple(v.shape), lambda v=v: v) for name, v in file.tensors.items()}
    write_model_file(target, file.config, tensors, {**file.header, **metadata})


def test_model_file_without_a_tokenizer_is_refused(model_path, tmp_path):
    rewrite_model(model_path, tmp_path / "bare.dllm", tokenizer=None)
    with pytest.raises(ValueError, match=r"bare\.dllm: the model file has no tokenizer"):
        DllmEngine.from_model_file(tmp_path / "bare.dllm")


def test_model_file_without_a_chat_template_uses_the_generic_rendering(model_path, tmp_path):
    rewrite_model(model_path, tmp_path / "plain.dllm", chat_template=None, source=None)
    engine = DllmEngine.from_model_file(tmp_path / "plain.dllm")
    templated = DllmEngine.from_model_file(model_path)
    assert engine.chat_template is None
    assert engine.model.id == "plain"  # no source repository: named after the file
    assert engine.system_fingerprint == templated.system_fingerprint  # same weights
    messages = [ChatMessage("user", "Hello")]
    assert engine.render_chat(messages) == render(messages)
    assert engine.render_chat(messages) != templated.render_chat(messages)
    result = engine.chat_completion(ChatRequest(messages, 6, SamplingOptions(temperature=0.7, seed=1)))
    assert result.completion_tokens <= 6 and result.finish_reason in ("stop", "length")


def test_transformer_embeddings_use_the_decoder_hidden_states(model_path):
    engine = DllmEngine.from_model_file(model_path)
    first = engine.embed("The quick brown fox")
    assert first.tokens == len(engine.tokenizer.encode("The quick brown fox"))
    assert first.vector.shape == (engine.model.config.hidden_size,)
    assert sum(float(v) * float(v) for v in first.vector) == pytest.approx(1.0, abs=1e-5)
    again = DllmEngine.from_model_file(model_path).embed("The quick brown fox")
    assert again.vector.tobytes() == first.vector.tobytes()


# The decoder


@pytest.fixture(scope="module", params=["llama", "qwen3"])
def decoder(request, tmp_path_factory) -> Transformer:
    directory = tmp_path_factory.mktemp(f"decoder-{request.param}")
    write_hf_checkpoint(directory / "checkpoint", tiny_config(request.param))
    import_model(directory / "checkpoint", directory / "model.dllm")
    return Transformer.from_file(directory / "model.dllm")


PROMPT = [1, 17, 42, 5, 63, 0, 9]


def test_hidden_states_are_what_the_head_sees(decoder):
    states = decoder.hidden_states(PROMPT)
    assert states.shape == (len(PROMPT), decoder.config.hidden_size) and states.dtype == np.float32
    for length in (1, 3, len(PROMPT)):
        # Causal: the states of a prefix are the first rows of the states of the whole prompt.
        assert decoder.hidden_states(PROMPT[:length]).tobytes() == states[:length].tobytes()
        logits = linear(np.ascontiguousarray(states[length - 1 : length]), decoder._lm_head).numpy()[0]
        assert logits.tobytes() == decoder.forward(PROMPT[:length]).tobytes()
    states[0, 0] = 1e9  # a copy: changing it changes nothing inside the model
    assert decoder.hidden_states(PROMPT).tobytes() != states.tobytes()


def test_hidden_states_need_valid_tokens(decoder):
    with pytest.raises(ValueError, match="at least one token of context"):
        decoder.hidden_states([])
    with pytest.raises(ValueError, match="token id out of range"):
        decoder.hidden_states([decoder.vocabulary_size])


def test_decoder_needs_every_tensor_with_its_shape(decoder):
    tensors = {name: tensor.numpy() for name, tensor in decoder.tensors.items()}
    missing = {name: values for name, values in tensors.items() if name != "layers.1.mlp.up.weight"}
    with pytest.raises(ValueError, match=r"missing tensors: \['layers.1.mlp.up.weight'\]"):
        Transformer(decoder.config, missing)
    wrong = {**tensors, "final_norm.weight": np.ones(decoder.config.hidden_size + 1, dtype=np.float32)}
    with pytest.raises(ValueError, match=r"tensor 'final_norm.weight' has shape \(17,\), expected \(16,\)"):
        Transformer(decoder.config, wrong)
    transposed = {**tensors, "layers.0.mlp.down.weight": np.ascontiguousarray(tensors["layers.0.mlp.down.weight"].T)}
    with pytest.raises(ValueError, match=r"'layers\.0\.mlp\.down\.weight' has shape"):
        Transformer(decoder.config, transposed)
    # Extra tensors are ignored.
    extra = Transformer(decoder.config, {**tensors, "unused": np.zeros(3, dtype=np.float32)})
    assert extra.forward(PROMPT).tobytes() == decoder.forward(PROMPT).tobytes()


def test_forward_batch_needs_one_cache_per_sequence(decoder):
    with pytest.raises(ValueError, match="one cache per sequence"):
        decoder.forward_batch([PROMPT, PROMPT[:3]], [decoder.new_cache()])
    cache = decoder.new_cache()
    with pytest.raises(ValueError, match="a distinct cache per sequence"):
        decoder.forward_batch([PROMPT, PROMPT[:3]], [cache, cache])


# The batcher and the generator


@pytest.mark.parametrize("size", [-1, -8])
def test_batcher_limit_must_not_be_negative(decoder, size):
    with pytest.raises(ValueError, match="max_batch must be non-negative"):
        Batcher(decoder, max_batch=size)


class CachedBigram(BigramModel):
    """A model with a KV cache (``new_cache``/``forward_cached``) that cannot batch (no ``forward_batch``)."""

    class Cache:
        def __init__(self) -> None:
            self.tokens: list[int] = []

    def __init__(self) -> None:
        super().__init__(ByteTokenizer.vocabulary_size, 5)
        self.cached_calls = 0

    def new_cache(self) -> CachedBigram.Cache:
        return CachedBigram.Cache()

    def forward_cached(self, tokens, cache):
        self.cached_calls += 1
        cache.tokens[:] = tokens
        return self.forward(tokens)


def test_kv_cache_without_batching_gives_the_plain_output():
    model = CachedBigram()
    generator = Generator(model, ByteTokenizer(), prompt_cache=2)
    assert generator.batcher is None and generator.prompt_cache is not None
    options = SamplingOptions(temperature=1.0, seed=9)
    cached = generator.generate("hello there", 12, options)
    plain = Generator(BigramModel(ByteTokenizer.vocabulary_size, 5), ByteTokenizer()).generate(
        "hello there", 12, options
    )
    assert cached == plain
    assert model.cached_calls == len(cached.tokens) + (cached.finish_reason == "stop")
    again = generator.stream("hello there, again", 12, options)
    assert again.cached_tokens > 0  # the first request's cache was reused
    assert again.result() == Generator(BigramModel(257, 5), ByteTokenizer()).generate("hello there, again", 12, options)


def test_negative_max_tokens_is_refused(placeholder):
    with pytest.raises(ValueError, match="max_tokens must be non-negative"):
        placeholder.complete("x", -1, GREEDY)
    result = placeholder.complete("x", 0, GREEDY)
    assert result.tokens == () and result.finish_reason == "length"


@pytest.mark.parametrize("top", [-1, 21])
def test_top_logprobs_out_of_range(placeholder, top):
    with pytest.raises(ValueError, match="top_logprobs must be between 0 and 20"):
        placeholder._generator.generate("x", 1, GREEDY, top_logprobs=top)


def test_zero_top_logprobs_records_only_the_chosen_token(placeholder):
    result = placeholder._generator.generate("logprobs", 5, SamplingOptions(temperature=0.8, seed=2), top_logprobs=0)
    assert len(result.logprobs) == len(result.tokens) == 5
    assert all(entry.top == () and entry.logprob <= 0.0 for entry in result.logprobs)
    assert [entry.token for entry in result.logprobs] == list(result.tokens)
    with_top = placeholder._generator.generate("logprobs", 5, SamplingOptions(temperature=0.8, seed=2), top_logprobs=2)
    assert [e.logprob for e in with_top.logprobs] == [e.logprob for e in result.logprobs]


def test_constraint_that_allows_no_token_stops_the_generation(placeholder):
    # The constraint's vocabulary has only ASCII tokens, so the non-ASCII literal can never be produced.
    ascii_only = TokenTrie([bytes([t]) if t < 128 else b"" for t in range(placeholder.model.vocabulary_size)])
    for options in (GREEDY, SamplingOptions(temperature=1.0, seed=3)):
        constraint = TokenConstraint(Grammar.literal("é"), ascii_only)
        result = placeholder._generator.generate("x", 10, options, constraint=constraint)
        assert result.tokens == () and result.finish_reason == "stop" and result.text == ""


def test_sampled_constrained_output_may_stop_once_complete(placeholder):
    # Sampling (not greedy) under a grammar that is complete after any digit: stop tokens join the allowed set.
    trie = placeholder._token_trie()
    for seed in range(6):
        constraint = TokenConstraint(Grammar.json_schema({"type": "integer"}), trie)
        options = SamplingOptions(temperature=1.5, seed=seed)
        result = placeholder._generator.generate("n=", 12, options, constraint=constraint)
        assert result.text and Grammar.json_schema({"type": "integer"}).matcher().matches(result.text.encode())
        again = placeholder._generator.generate(
            "n=", 12, options, constraint=TokenConstraint(Grammar.json_schema({"type": "integer"}), trie)
        )
        assert again == result


def test_chat_stream_ends_with_a_finished_event(placeholder):
    request = ChatRequest([ChatMessage("user", "Hi")], 5, GREEDY, response_format=ResponseFormat("json_object"))
    events = list(placeholder.chat_stream(request))
    assert isinstance(events[-1], Finished)
    assert events[-1].completion_tokens <= 5
