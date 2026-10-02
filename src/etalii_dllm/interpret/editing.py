"""Model editing with ROME (rank-one model editing, Meng et al. 2022): change one fact by a rank-one update of one MLP.

The MLP down projection of layer ``l`` is read as a key-value memory: its input at the last token of a subject (the
key ``k``) is mapped to what the MLP writes into the residual stream (the value ``W k``). An edit

1. takes the key ``k`` from a traced forward pass of the prompt (the mean over the prompt and any context prefixes);
2. finds a change ``delta`` of the value that makes the model predict the target, by gradient descent (AdamW) on
   the target's cross-entropy plus a small L2 penalty, with the gradient of the residual stream from the
   deterministic backward kernels (``DecoderGradients.residual_gradient``);
3. writes ``W' = W + delta u^T / (u . k)`` with ``u = C^-1 k``, where ``C`` is the (regularised) covariance of keys over
   a fixed corpus, solved by the ``cholesky_solve`` kernel. Then ``W' k = W k + delta``, while keys unlike ``k`` (in
   the sense of ``C``) are changed as little as possible.

In a mixture-of-experts layer the memory is one expert's down projection: the expert the prompt's subject token is
routed to with the largest weight ``r``. The keys are that expert's activations (``Transformer.mlp_activation``,
over every corpus position for ``C``), and since the layer adds ``r W k``, the update writes ``delta / r``.

Every step is a fixed-order kernel or an elementwise float32 operation, the optimiser runs a fixed number of steps
(or stops at a fixed loss), so equal edits of equal models write byte-identical model files.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from etalii_dllm.interpret.trace import trace
from etalii_dllm.modelfile import ModelFile, TensorSource, edit_step, extend_lineage, lineage, write_model_file
from etalii_dllm.numerics import cholesky_solve, column_mean, dot, linear, softmax, sum_, sum_squares
from etalii_dllm.tokenization import Tokenizer
from etalii_dllm.transformer import Transformer

# The default corpus for the key covariance: plain sentences on varied topics, so the edit avoids directions that
# ordinary text uses. Pass your own with ``corpus=`` (``--corpus``) for a better estimate.
DEFAULT_CORPUS = (
    "The river flows through the valley and reaches the sea after many miles.",
    "She opened the window, and the morning air filled the small kitchen.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "The committee will meet on Tuesday to discuss the new budget.",
    "He learned to play the violin when he was seven years old.",
    "Photosynthesis turns sunlight, water and carbon dioxide into sugar and oxygen.",
    "The train to the city leaves every hour from the north platform.",
    "Many birds migrate south for the winter and return in the spring.",
    "The recipe calls for two eggs, a cup of flour and a pinch of salt.",
    "A prime number has exactly two divisors: one and itself.",
    "The old library holds thousands of books, maps and letters.",
    "They walked along the beach and collected shells until sunset.",
    "The company reported higher profits in the third quarter.",
    "Mountains are formed when tectonic plates push against each other.",
    "The children painted pictures of animals for the school exhibition.",
    "Light travels faster than sound, which is why we see lightning first.",
    "The museum is closed on Mondays but open late on Fridays.",
    "Coffee is grown in tropical regions near the equator.",
    "The software update fixes several bugs and improves battery life.",
    "Shakespeare wrote plays, sonnets and long narrative poems.",
    "The doctor advised him to rest and drink plenty of water.",
    "A triangle has three sides, and its angles add up to one hundred and eighty degrees.",
    "The orchestra played a symphony by a famous composer.",
    "Rain fell all night, and the streets were flooded by morning.",
    "The capital of Japan is Tokyo, and the capital of Canada is Ottawa.",
    "Bees collect nectar from flowers and turn it into honey.",
    "The football match ended in a draw after extra time.",
    "Computers store information as sequences of ones and zeros.",
    "The ancient city was built on a hill overlooking the harbour.",
    "Please remember to lock the door when you leave the office.",
    "The moon orbits the earth about once every twenty-seven days.",
    "Farmers harvest wheat at the end of the summer.",
)


@dataclass(frozen=True)
class EditRequest:
    """Make the model continue ``prompt`` (which contains ``subject``) with ``target``."""

    prompt: str
    subject: str
    target: str


@dataclass(frozen=True)
class EditResult:
    weights: dict[str, np.ndarray]
    record: dict[str, Any]
    """What was done, as stored in the model file's ``edits`` list."""

    @property
    def probability_before(self) -> float:
        return float(self.record["target_probability"]["before"])

    @property
    def probability_after(self) -> float:
        return float(self.record["target_probability"]["after"])


def subject_position(tokenizer: Tokenizer, prompt: str, subject: str) -> tuple[list[int], int]:
    """The prompt's tokens and the position of the token that completes ``subject`` (its first occurrence)."""
    start = prompt.find(subject)
    if not subject or start < 0:
        raise ValueError(f"the prompt {prompt!r} does not contain the subject {subject!r}")
    end = len(prompt[: start + len(subject)].rstrip())
    tokens = tokenizer.encode(prompt)
    for position in range(len(tokens)):
        if len(tokenizer.decode(tokens[: position + 1]).rstrip()) >= end:
            return tokens, position
    raise ValueError(f"could not find the subject {subject!r} in the tokens of the prompt")  # pragma: no cover


def mlp_keys(model: Transformer, tokens: Sequence[int], layer: int, expert: int | str | None = None) -> np.ndarray:
    """The MLP keys ``[positions, intermediate]`` of ``layer`` (0-based) for ``tokens``: the input of its down
    projection, or in a mixture-of-experts layer that of ``expert``'s down projection (at every position, routed to
    the expert or not; ``"shared"``: the shared expert's)."""
    traced = trace(model, tokens, attention=False, logits=False)
    if traced.mlp_activation is not None:
        return traced.mlp_activation[layer]
    return model.mlp_activation(traced.middle[layer], layer, expert)


def key_covariance(
    model: Transformer, tokenizer: Tokenizer, texts: Sequence[str], layer: int, expert: int | str | None = None
) -> np.ndarray:
    """``(1 / N) * sum_t k_t k_t^T`` over every position ``t`` of ``texts``: the MLP keys of ``layer`` (0-based;
    of ``expert`` in a mixture-of-experts layer), summed over positions in text order through the ``linear``
    kernel."""
    keys = []
    for text in texts:
        tokens = tokenizer.encode(text)
        if tokens:
            keys.append(mlp_keys(model, tokens, layer, expert))
    if not keys:
        raise ValueError("the covariance corpus has no tokens")
    stacked = np.ascontiguousarray(np.concatenate(keys).T)  # [intermediate, N]
    covariance = linear(stacked, stacked).numpy()
    return (covariance / np.float32(stacked.shape[1])).astype(np.float32)


def rome(
    model: Transformer,
    tokenizer: Tokenizer,
    request: EditRequest,
    *,
    layer: int | None = None,
    contexts: Sequence[str] = (),
    corpus: Sequence[str] = DEFAULT_CORPUS,
    regularisation: float = 0.1,
    steps: int = 40,
    learning_rate: float = 0.5,
    l2: float = 1e-3,
    stop_loss: float = 0.05,
    expert: str = "routed",
) -> EditResult:
    """Computes a ROME edit of ``model``. ``layer`` is 1-based (default: a quarter of the way in); ``contexts`` are
    prefixes put before the prompt to average the key over; ``corpus`` estimates the key covariance, which is
    regularised as ``C / mean(diag C) + regularisation * I``. In a mixture-of-experts layer ``expert`` picks the
    memory: the subject's top ``"routed"`` expert, or the ``"shared"`` expert every token runs."""
    from etalii_dllm.training import AdamW, AdamWConfig
    from etalii_dllm.training.backprop import DecoderGradients

    config = model.config
    try:
        gradients = DecoderGradients(config)
    except ValueError as error:
        raise ValueError(f"model editing needs the fine-tuning support: {error}") from None
    if model.steering:
        raise ValueError("edit an unsteered model")
    layer = layer if layer is not None else max(1, config.layers // 4)
    if not 1 <= layer <= config.layers:
        raise ValueError(f"layer must be between 1 and {config.layers}")
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if expert not in ("routed", "shared"):
        raise ValueError("expert must be 'routed' or 'shared'")
    index = layer - 1
    if expert == "shared" and (not config.is_sparse(index) or config.shared_expert_intermediate_size is None):
        raise ValueError(f"layer {layer} has no shared expert")
    target = tokenizer.encode(request.target)
    if not target:
        raise ValueError("the target has no tokens")

    # 1. The key: the MLP activation at the subject's last token, averaged over the prompt and its contexts. In a
    # mixture-of-experts layer: that of the expert the prompt's subject token is routed to with the largest weight
    # (the first in rank order, so ties go to the lower expert), whose output the layer scales by that weight; or
    # that of the shared expert, whose output a gated one scales by the sigmoid gate.
    sequences = []
    for prefix in ("", *contexts):
        tokens, position = subject_position(tokenizer, prefix + request.prompt, request.subject)
        sequences.append((tokens, position))
    memory: int | str | None = None
    routing_weight = 1.0
    if config.is_sparse(index):
        routed = trace(model, sequences[0][0], attention=False, logits=False)
        assert routed.experts is not None and routed.expert_weights is not None
        if expert == "shared":
            memory = "shared"
            if routed.shared_gate is not None:  # a sigmoid of a finite score: never 0
                routing_weight = float(routed.shared_gate[index, sequences[0][1]])
        else:
            memory = int(routed.experts[index, sequences[0][1], 0])
            routing_weight = float(routed.expert_weights[index, sequences[0][1], 0])  # at least 1 / experts: never 0
    keys = np.stack([mlp_keys(model, t, index, memory)[p] for t, p in sequences])
    key = keys[0] if len(keys) == 1 else column_mean(keys)

    # 2. The value change: AdamW on delta, added to the residual stream after the layer at the subject's last token.
    weights = {name: tensor.numpy() for name, tensor in model.tensors.items()}
    hidden = config.hidden_size
    delta = np.zeros(hidden, dtype=np.float32)
    optimiser = AdamW(
        AdamWConfig(learning_rate=learning_rate, weight_decay=0.0, max_grad_norm=0.0, schedule="constant"),
        {"delta": (hidden,)},
    )
    before = _target_probability(model, sequences[0][0], target)
    losses = []
    for step in range(1, steps + 1):
        total_loss = 0.0
        gradient = np.zeros(hidden, dtype=np.float32)
        for tokens, position in sequences:
            rows = np.zeros((len(tokens) + len(target) - 1, hidden), dtype=np.float32)
            rows[position] = delta
            sequence = [*tokens, *target[:-1]]
            labels = [-1] * (len(tokens) - 1) + target
            loss, residual = gradients.residual_gradient(weights, sequence, labels, index, rows)
            total_loss += loss
            gradient = gradient + residual[position]
        mean_loss = total_loss / (len(sequences) * len(target))
        losses.append(mean_loss)
        if mean_loss < stop_loss:
            break
        gradient = gradient + np.float32(2.0 * l2) * delta
        params = {"delta": delta}
        optimiser.step(params, {"delta": gradient}, step, learning_rate)
    # 3. The rank-one update of the down projection.
    covariance = key_covariance(model, tokenizer, corpus, index, memory)
    diagonal = np.ascontiguousarray(np.diagonal(covariance))
    scale = sum_(diagonal) / diagonal.shape[0]
    if not scale > 0:
        raise ValueError("the covariance corpus gives a zero covariance")  # pragma: no cover
    regularised = (covariance / np.float32(scale)).astype(np.float32)
    regularised[np.diag_indices_from(regularised)] += np.float32(regularisation)
    u = cholesky_solve(regularised, key)
    denominator = dot(u, key)
    coefficients = (u / np.float32(denominator)).astype(np.float32)
    name = f"layers.{index}.mlp.down.weight"
    value = delta
    if memory is not None:  # the layer adds routing_weight * W k, so W k changes by delta / routing_weight
        name = f"layers.{index}.mlp.{'shared' if memory == 'shared' else f'experts.{memory}'}.down.weight"
        value = (delta / np.float32(routing_weight)).astype(np.float32)
    edited = dict(weights)
    edited[name] = (weights[name] + value[:, None] * coefficients[None, :]).astype(np.float32)

    after_model = Transformer(config, edited, model_id=model.id)
    after = _target_probability(after_model, sequences[0][0], target)
    record = {
        "method": "rome",
        "layer": layer,
        "prompt": request.prompt,
        "subject": request.subject,
        "target": request.target,
        "contexts": list(contexts),
        "base_fingerprint": model.weights_fingerprint,
        "covariance": {
            "texts": len(corpus),
            "corpus_fingerprint": hashlib.sha256("\n".join(corpus).encode("utf-8")).hexdigest(),
            "regularisation": regularisation,
        },
        "optimiser": {"steps": len(losses), "learning_rate": learning_rate, "l2": l2, "final_loss": losses[-1]},
        "delta_norm": math.sqrt(sum_squares(delta)),
        "target_probability": {"before": before, "after": after},
    }
    if memory is not None:
        record["expert"] = memory
        record["routing_weight"] = routing_weight
    return EditResult(edited, record)


def _target_probability(model: Transformer, tokens: list[int], target: list[int]) -> float:
    """Probability of the whole ``target`` after ``tokens`` (the product over its tokens, in double)."""
    probability = 1.0
    context = list(tokens)
    for token in target:
        probability *= float(softmax(model.forward(context))[token])
        context.append(token)
    return probability


def write_edited_model(base: ModelFile, result: EditResult, path: str | Path) -> str:
    """Writes the edited weights as a new ``model.dllm`` with the edit appended to its ``edits`` list; returns the
    new fingerprint. ``result`` must have been computed from ``base``'s weights."""
    if result.record["base_fingerprint"] != base.fingerprint:
        raise ValueError("the edit was computed from other weights than the model it is written onto")
    metadata: dict[str, Any] = dict(base.header)
    licence = dict(metadata.get("licence") or {})
    if licence.get("attribution"):
        attribution = str(licence["attribution"])
        unmodified = "; the weights are otherwise unmodified."
        if attribution.endswith(unmodified):
            attribution = attribution[: -len(unmodified)] + "."
        note = " Edited with EtAlii.Dllm (ROME); modified weights."
        licence["attribution"] = attribution if attribution.endswith(note) else attribution + note
        metadata["licence"] = licence
    metadata["edits"] = [*(base.header.get("edits") or []), result.record]
    metadata["lineage"] = extend_lineage(lineage(base.header), base.fingerprint, edit_step(result.record))
    tensors: Mapping[str, TensorSource] = {
        name: TensorSource(tuple(values.shape), lambda values=values: values) for name, values in result.weights.items()
    }
    return write_model_file(path, base.config, tensors, metadata)
