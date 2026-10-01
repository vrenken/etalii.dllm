"""Interpretability tools: tracing is observation only (the same bits as an untraced pass) and reproducible on every
code path; the logit lens, attention maps and embedding explorer agree with plain references; the kernels behind
them; the ``dllm lens|attention|neighbours`` commands and their HTML/SVG views."""

from __future__ import annotations

import json
import math
from itertools import pairwise

import numpy as np
import pytest
from golden_values import TINY_TRACE_FINGERPRINT
from model_fixtures import tiny_config, write_hf_checkpoint
from test_engine_import import model_path  # noqa: F401 - fixture

from etalii_dllm import _kernels, numerics
from etalii_dllm.cli import main
from etalii_dllm.engine import DllmEngine, default_engine
from etalii_dllm.importing import import_model
from etalii_dllm.interpret import logit_lens, neighbours, top_k, trace
from etalii_dllm.interpret.embeddings import embedding_matrix, parse_expression, token_text
from etalii_dllm.interpret.render import attention_html, lens_html, word_cloud_html, word_cloud_svg
from etalii_dllm.tokenization import ByteTokenizer
from etalii_dllm.transformer import LayerHook, Transformer

PROMPT = [1, 17, 42, 5, 63, 0, 9, 9, 30]
FAMILIES = ["gemma2", "gemma3", "granite", "llama", "mistral", "olmo2", "phi3", "qwen2", "qwen3"]


@pytest.fixture(scope="module", params=FAMILIES)
def model(request, tmp_path_factory) -> Transformer:
    directory = tmp_path_factory.mktemp(request.param)
    write_hf_checkpoint(directory / "checkpoint", tiny_config(request.param))
    import_model(directory / "checkpoint", directory / "model.dllm")
    return Transformer.from_file(directory / "model.dllm")


@pytest.fixture(autouse=True)
def restore_kernel_settings():
    yield
    numerics.set_threads(0)
    _kernels.set_isa("best")


def gaussian(seed: int, *shape: int) -> np.ndarray:
    return numerics.fill_gaussian(seed, math.prod(shape)).reshape(shape)


# -- tracing ----------------------------------------------------------------------------------------------------


def test_tracing_does_not_change_the_output(model):
    recorded = trace(model, PROMPT)
    assert recorded.logits[-1].tobytes() == model.forward(PROMPT).tobytes()
    assert recorded.hidden.tobytes() == model.hidden_states(PROMPT).tobytes()
    for length in (1, 4):
        assert trace(model, PROMPT[:length]).logits[-1].tobytes() == model.forward(PROMPT[:length]).tobytes()


def test_trace_shapes_and_consistency(model):
    config = model.config
    n = len(PROMPT)
    recorded = trace(model, PROMPT)
    assert recorded.residual.shape == (config.layers + 1, n, config.hidden_size)
    assert recorded.middle.shape == recorded.attention_output.shape == recorded.mlp_output.shape
    assert recorded.mlp_activation.shape == (config.layers, n, config.intermediate_size)
    assert recorded.attention is not None and recorded.attention.shape == (config.layers, config.heads, n, n)
    # The residual stream is the running sum of what the blocks add (elementwise float32, as in the decoder).
    for layer in range(config.layers):
        middle = recorded.residual[layer] + recorded.attention_output[layer]
        assert middle.tobytes() == recorded.middle[layer].tobytes()
        assert (middle + recorded.mlp_output[layer]).tobytes() == recorded.residual[layer + 1].tobytes()
    assert model.final_norm(recorded.residual[-1]).tobytes() == recorded.hidden.tobytes()
    probabilities = recorded.attention.astype(np.float64)
    np.testing.assert_allclose(probabilities.sum(-1), 1.0, atol=1e-5)
    assert not np.triu(probabilities, 1).any()  # causal: no position attends to a later one
    for layer in range(config.layers):
        window = config.window(layer)
        if window is not None:
            assert not np.tril(probabilities[layer], -window).any()
    assert trace(model, PROMPT, attention=False).attention is None


def test_trace_is_the_same_bits_on_every_code_path(model):
    expected = TINY_TRACE_FINGERPRINT[model.config.family]
    for isa in _kernels.supported_isas():
        _kernels.set_isa(isa)
        for count in (1, 3):
            numerics.set_threads(count)
            assert trace(model, PROMPT).fingerprint() == expected, (isa, count)


def test_a_hook_can_change_the_residual_stream(model):
    class Nudge(LayerHook):
        def __init__(self, vector: np.ndarray | None) -> None:
            self.vector = vector

        def residual(self, layer, point, x):
            if self.vector is None or layer != 0 or point != "output":
                return None
            return x + self.vector

    zero = np.zeros(model.config.hidden_size, dtype=np.float32)
    plain = model.hidden_states(PROMPT).tobytes()
    assert model.run_hooked(PROMPT, Nudge(None)).tobytes() == plain
    assert model.run_hooked(PROMPT, Nudge(zero)).tobytes() == plain
    nudged = model.run_hooked(PROMPT, Nudge(gaussian(5, model.config.hidden_size)))
    assert nudged.tobytes() != plain
    assert nudged.tobytes() == model.run_hooked(PROMPT, Nudge(gaussian(5, model.config.hidden_size))).tobytes()


def test_hooked_runs_reject_bad_input(model):
    with pytest.raises(ValueError):
        model.run_hooked([], LayerHook())
    model.device = "cuda"
    try:
        with pytest.raises(ValueError, match="CPU"):
            model.run_hooked(PROMPT, LayerHook())
    finally:
        model.device = "cpu"


# -- logit lens -------------------------------------------------------------------------------------------------


def test_logit_lens_last_layer_is_the_model_prediction(model):
    lens = logit_lens(model, PROMPT, top=4)
    assert lens.layers == model.config.layers and lens.tokens == tuple(PROMPT)
    assert len(lens.predictions[0]) == len(PROMPT)
    probabilities = numerics.softmax(model.forward(PROMPT))
    final = lens.predictions[-1][-1]
    assert [p.token for p in final] == top_k(probabilities, 4)
    assert [p.probability for p in final] == [float(probabilities[p.token]) for p in final]
    assert all(a.probability >= b.probability for a, b in pairwise(final))
    with pytest.raises(ValueError):
        logit_lens(model, PROMPT, top=0)


def test_top_k_is_a_total_order():
    values = np.array([0.5, 0.9, 0.5, 0.9, 0.1], dtype=np.float32)
    assert top_k(values, 5) == [1, 3, 0, 2, 4]
    assert top_k(values, 2) == [1, 3]


# -- kernels ----------------------------------------------------------------------------------------------------


def test_attention_weights_match_attention():
    q, k, v = gaussian(1, 5, 4, 8), gaussian(2, 7, 2, 8), gaussian(3, 7, 2, 8)
    for options in ({}, {"window": 3}, {"softcap": 0.5}, {"causal": False}, {"q_offset": 1, "scale": 0.3}):
        weights = numerics.attention_weights(q, k, **options).numpy()
        assert weights.shape == (5, 4, 7)
        out = numerics.attention(q, k, v, **options).numpy()
        group = np.repeat(np.arange(2), 2)
        mixed = np.einsum("thj,jhd->thd", weights.astype(np.float64), v[:, group].astype(np.float64))
        np.testing.assert_allclose(mixed, out, atol=1e-6)
    scores = np.einsum("thd,jhd->thj", q.astype(np.float64), k[:, np.repeat(np.arange(2), 2)]) / math.sqrt(8)
    scores += np.triu(np.full((5, 7), -np.inf), 3)[:, None, :]
    reference = np.exp(scores - scores.max(-1, keepdims=True))
    reference /= reference.sum(-1, keepdims=True)
    np.testing.assert_allclose(numerics.attention_weights(q, k).numpy(), reference, atol=1e-7)
    with pytest.raises(ValueError):
        numerics.attention_weights(q[0], k)
    with pytest.raises(ValueError):
        numerics.attention_weights(q, k, q_offset=-1)
    with pytest.raises(ValueError):
        numerics.attention_weights(q, k, softcap=0.0)
    with pytest.raises(ValueError):
        numerics.attention_weights(gaussian(1, 2, 3, 8), k)


def test_cosine_similarity_column_mean_and_cholesky():
    matrix, query = gaussian(1, 9, 6), gaussian(2, 6)
    matrix[3] = 0.0
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1) * np.linalg.norm(query.astype(np.float64))
    norms[3] = 1.0
    expected = matrix.astype(np.float64) @ query.astype(np.float64) / norms
    np.testing.assert_allclose(numerics.cosine_similarity(matrix, query), expected, atol=1e-6)
    assert numerics.cosine_similarity(matrix, query)[3] == 0.0
    assert not numerics.cosine_similarity(matrix, np.zeros(6, dtype=np.float32)).any()
    with pytest.raises(ValueError):
        numerics.cosine_similarity(matrix, gaussian(2, 5))

    x = gaussian(3, 11, 4)
    means = [math.fsum(float(v) for v in x[:, c]) / 11 for c in range(4)]
    np.testing.assert_allclose(numerics.column_mean(x), means, rtol=1e-7)

    a = gaussian(4, 5, 5).astype(np.float64)
    a = (a @ a.T + 5 * np.eye(5)).astype(np.float32)
    b = gaussian(5, 5, 2)
    np.testing.assert_allclose(numerics.cholesky_solve(a, b), np.linalg.solve(a, b), atol=1e-5)
    assert numerics.cholesky_solve(a, b[:, 0]).shape == (5,)
    with pytest.raises(ValueError, match="positive definite"):
        numerics.cholesky_solve(-a, b)
    with pytest.raises(ValueError):
        numerics.cholesky_solve(a[:, :4], b)
    with pytest.raises(ValueError):
        numerics.cholesky_solve(a, b[:4])


# -- embedding explorer -----------------------------------------------------------------------------------------


def test_parse_expression():
    assert parse_expression("king - man + woman") == [(1, " king"), (-1, " man"), (1, " woman")]
    assert parse_expression('"Paris"') == [(1, "Paris")]
    assert parse_expression("x-ray") == [(1, " x-ray")]
    with pytest.raises(ValueError):
        parse_expression('"" + a')


def test_neighbours(model_path):  # noqa: F811
    engine = DllmEngine.from_model_file(model_path)
    model = engine.model
    tokenizer = engine.tokenizer
    result = neighbours(model, tokenizer, "hello", top=6)
    query = set(result.terms[0].tokens)
    assert len(result.neighbours) == 6 and not query & {n.token for n in result.neighbours}
    assert all(a.similarity >= b.similarity for a, b in pairwise(result.neighbours))
    matrix = embedding_matrix(model)
    vector = (
        matrix[result.terms[0].tokens[0]]
        if len(query) == 1
        else numerics.column_mean(matrix[list(result.terms[0].tokens)])
    )
    similarities = numerics.cosine_similarity(matrix, vector)
    assert [n.similarity for n in result.neighbours] == [float(similarities[n.token]) for n in result.neighbours]
    assert neighbours(model, tokenizer, "hello", top=6) == result
    analogy = neighbours(model, tokenizer, "hello - world + there", top=3, space="output")
    assert [t.sign for t in analogy.terms] == [1, -1, 1]
    assert embedding_matrix(model, "output") is embedding_matrix(model, "input")  # tied embeddings
    with pytest.raises(ValueError):
        neighbours(model, tokenizer, "hello", top=0)
    with pytest.raises(ValueError):
        embedding_matrix(model, "hidden")

    class Silent:
        def encode(self, text):
            return []

    with pytest.raises(ValueError, match="no tokens"):
        neighbours(model, Silent(), "hello", top=2)  # type: ignore[arg-type]


def test_untied_output_space(model):
    if model.config.tie_word_embeddings:
        assert embedding_matrix(model, "output") is embedding_matrix(model, "input")
    else:
        assert embedding_matrix(model, "output") is model.tensors["lm_head.weight"].numpy()


def test_token_text_keeps_spaces(model_path):  # noqa: F811
    engine = DllmEngine.from_model_file(model_path)
    tokens = engine.tokenizer.encode("a b")
    assert "".join(token_text(engine.tokenizer, t) for t in tokens) == "a b"
    assert token_text(ByteTokenizer(), ord("x")) == "x"

    class Plain:
        def decode(self, tokens):
            return "<" + ",".join(map(str, tokens)) + ">"

    assert token_text(Plain(), 7) == "<7>"  # type: ignore[arg-type]


# -- views and commands -----------------------------------------------------------------------------------------


def test_views_are_reproducible_and_escaped(model_path):  # noqa: F811
    engine = DllmEngine.from_model_file(model_path)
    result = neighbours(engine.model, engine.tokenizer, "hello + <b>", top=12)
    svg = word_cloud_svg(result)
    assert svg == word_cloud_svg(neighbours(engine.model, engine.tokenizer, "hello + <b>", top=12))
    assert svg.startswith("<svg") and "<b>" not in svg and "&lt;b&gt;" in svg
    assert "<!doctype html>" in word_cloud_html(result)
    tokens = engine.tokenizer.encode("hello there")
    lens = logit_lens(engine.model, tokens, top=2)
    texts = [token_text(engine.tokenizer, t) for t in tokens]
    predicted = {p.token: token_text(engine.tokenizer, p.token) for r in lens.predictions for x in r for p in x}
    page = lens_html(lens, texts, predicted)
    assert page.count("<tr>") == lens.layers + 2 and "embeddings" in page
    recorded = trace(engine.model, tokens)
    page = attention_html(recorded.attention, texts, [0, 1], [0])
    assert page.count("<h2>") == 2


def test_word_cloud_of_equal_similarities():
    from etalii_dllm.interpret.embeddings import Neighbour, Neighbourhood

    same = Neighbourhood("x", (), "input", [Neighbour(i, f"w{i}", 0.5) for i in range(3)] + [Neighbour(9, " ", 0.5)])
    svg = word_cloud_svg(same)
    assert svg.count("<text") == 5 and "&#x27; &#x27;" in svg


@pytest.fixture
def isolated(monkeypatch):
    from etalii_dllm import engine

    for name in (engine.MODEL_ENVIRONMENT_VARIABLE, engine.QUANTIZE_ENVIRONMENT_VARIABLE):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    default_engine.cache_clear()
    yield
    default_engine.cache_clear()


def test_commands(model_path, tmp_path, capsys, isolated):  # noqa: F811
    model = ["--model", str(model_path)]
    page = tmp_path / "lens.html"
    assert main([*model, "lens", "--prompt", "hello there", "--top-k", "2", "--html", str(page)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("position ") and "embed" in out and page.read_text(encoding="utf-8").startswith("<!doctype")
    assert main([*model, "lens", "--prompt", "hello there", "--json", "--chat"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert len(result["layers"]) == 3 and len(result["layers"][0][0]) == 5

    page = tmp_path / "attention.html"
    assert (
        main([*model, "attention", "--prompt", "hello there", "--layer", "2", "--head", "1", "--html", str(page)]) == 0
    )
    assert capsys.readouterr().out.startswith("layer 2 head 1:") and page.exists()
    assert main([*model, "attention", "--prompt", "hello there", "--json"]) == 0
    maps = json.loads(capsys.readouterr().out)["attention"]
    assert sorted(maps) == ["1", "2"] and sorted(maps["1"]) == ["0", "1", "2", "3"]

    svg, page = tmp_path / "cloud.svg", tmp_path / "cloud.html"
    assert main([*model, "neighbours", "hello", "--top-k", "3", "--svg", str(svg), "--html", str(page)]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 3 and svg.exists() and page.exists()
    assert main([*model, "neighbours", "hello", "--top-k", "3", "--json", "--space", "output"]) == 0
    assert len(json.loads(capsys.readouterr().out)["neighbours"]) == 3


@pytest.mark.parametrize(
    "arguments",
    [
        ["lens", "--prompt", "hello", "--position", "9"],
        ["lens", "--prompt", ""],
        ["attention", "--prompt", "hello", "--layer", "3"],
        ["attention", "--prompt", "hello", "--head", "4"],
        ["neighbours", "hello", "--top-k", "0"],
    ],
)
def test_command_errors(model_path, capsys, isolated, arguments):  # noqa: F811
    assert main(["--model", str(model_path), *arguments]) == 1
    assert capsys.readouterr().err.startswith(f"dllm {arguments[0]}: ")


def test_commands_need_an_imported_model(capsys, isolated):
    assert main(["lens", "--prompt", "hi"]) == 1
    assert "needs an imported model" in capsys.readouterr().err
