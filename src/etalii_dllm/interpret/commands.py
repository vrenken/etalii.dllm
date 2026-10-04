"""The interpretability subcommands of ``dllm``: ``lens``, ``attention``, ``experts``, ``neighbours``, ``steer``,
``sae`` and ``edit``. ``lens``, ``attention`` and ``steer`` also open up T5 text-to-text models (#403-#405): the
prompt is the source, ``--answer`` the answer so far, and the decoder is what they read."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from etalii_dllm.chat import ChatMessage
from etalii_dllm.engine import DllmEngine
from etalii_dllm.generation import with_end_of_source
from etalii_dllm.interpret.embeddings import SPACES, neighbours, token_text
from etalii_dllm.interpret.lens import logit_lens, top_k
from etalii_dllm.interpret.render import attention_html, lens_html, word_cloud_html, word_cloud_svg
from etalii_dllm.interpret.text_to_text import trace_text_to_text
from etalii_dllm.interpret.trace import trace
from etalii_dllm.numerics import sum_squares
from etalii_dllm.seq2seq import TextToText
from etalii_dllm.transformer import Transformer

TEXT_TO_TEXT_COMMANDS = ("lens", "attention", "steer")
"""The subcommands that also read T5 text-to-text models."""

COMMANDS = ("lens", "attention", "experts", "neighbours", "steer", "sae")


def add_commands(commands: Any) -> None:
    """Adds the subcommands to ``dllm``'s subparsers."""
    lens = commands.add_parser("lens", help="logit lens: the top predictions after every layer")
    attention = commands.add_parser("attention", help="attention maps: where each head looks")
    experts = commands.add_parser("experts", help="expert routing of a mixture-of-experts model, token by token")
    for command in (lens, attention, experts):
        command.add_argument("--prompt", required=True)
        command.add_argument("--chat", action="store_true", help="wrap the prompt in the model's chat template")
        command.add_argument("--json", action="store_true", help="print the full result as JSON")
    for command in (lens, attention):
        command.add_argument("--html", metavar="FILE", help="also write a self-contained HTML view")
        command.add_argument(
            "--answer", default="", help="text-to-text models: the answer so far (the prompt is the source)"
        )
    lens.add_argument("--top-k", type=int, default=5)
    lens.add_argument("--position", type=int, default=-1, help="position to print (default: the last)")
    attention.add_argument("--layer", type=int, help="1-based layer (default: all)")
    attention.add_argument("--head", type=int, help="0-based head (default: all)")
    attention.add_argument("--top-k", type=int, default=3, help="keys printed per query")
    attention.add_argument(
        "--cross", action="store_true", help="text-to-text models: cross-attention over the source tokens"
    )
    experts.add_argument("--layer", type=int, help="1-based layer (default: every mixture-of-experts layer)")

    near = commands.add_parser("neighbours", help="nearest tokens in embedding space, e.g. 'king - man + woman'")
    near.add_argument("expression", help="a word, or words joined by ' + ' and ' - '; quote to keep spaces")
    near.add_argument("--top-k", type=int, default=20)
    near.add_argument("--space", choices=SPACES, default="input", help="input embeddings or LM head rows")
    near.add_argument("--json", action="store_true")
    near.add_argument("--svg", metavar="FILE", help="write a word cloud SVG")
    near.add_argument("--html", metavar="FILE", help="write a word cloud and table as HTML")

    steer = commands.add_parser("steer", help="build a steering vector from contrasting prompts")
    steer.add_argument("--positive", action="append", default=[], help="a prompt showing the wanted behaviour")
    steer.add_argument("--negative", action="append", default=[], help="a prompt showing the opposite")
    steer.add_argument("--positive-file", help="more positive prompts, one per line")
    steer.add_argument("--negative-file", help="more negative prompts, one per line")
    steer.add_argument(
        "--layer", type=int, help="1-based (decoder) layer whose output is steered (default: a third deep)"
    )
    steer.add_argument("--strength", type=float, default=4.0, help="default multiplier stored in the file")
    steer.add_argument("-o", "--output", required=True, help="the steering vector file (JSON) to write")

    sae = commands.add_parser("sae", help="sparse autoencoders: train, inspect features, steer with a feature")
    actions = sae.add_subparsers(dest="sae_command", required=True)
    train = actions.add_parser("train", help="train an SAE on one layer's residual stream over a corpus")
    train.add_argument("--corpus", required=True, help="text file, one passage per line")
    train.add_argument("--layer", type=int, help="1-based layer (default: half way)")
    train.add_argument("--features", type=int, default=1024)
    train.add_argument(
        "--l1", type=float, default=5.0, help="sparsity penalty (inputs are scaled to norm sqrt(hidden))"
    )
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--steps", type=int, default=2000)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("-o", "--output", required=True, help="the SAE file (safetensors) to write")
    features = actions.add_parser("features", help="what the SAE's features respond to")
    features.add_argument("sae", help="an SAE file from 'dllm sae train'")
    features.add_argument("--corpus", required=True, help="text file, one passage per line")
    features.add_argument("--feature", type=int, action="append", help="feature to show (repeatable)")
    features.add_argument("--count", type=int, default=10, help="features shown without --feature")
    features.add_argument("--top-k", type=int, default=6, help="examples per feature")
    features.add_argument("--json", action="store_true")
    steer_feature = actions.add_parser("steer", help="write a feature's direction as a steering vector")
    steer_feature.add_argument("sae", help="an SAE file from 'dllm sae train'")
    steer_feature.add_argument("--feature", type=int, required=True)
    steer_feature.add_argument("--strength", type=float, default=4.0)
    steer_feature.add_argument("-o", "--output", required=True, help="the steering vector file (JSON) to write")

    edit = commands.add_parser("edit", help="change one fact with a ROME rank-one edit, writing a new model.dllm")
    edit.add_argument("base", help="the model.dllm file to edit")
    edit.add_argument("--prompt", required=True, help='e.g. "The Eiffel Tower is located in the city of"')
    edit.add_argument("--subject", required=True, help='the subject inside the prompt, e.g. "Eiffel Tower"')
    edit.add_argument("--target", required=True, help='the new continuation, e.g. " Rome" (mind the space)')
    edit.add_argument("--layer", type=int, help="1-based layer whose MLP is edited (default: a quarter deep)")
    edit.add_argument("--context", action="append", default=[], help="a prefix to average the key over (repeatable)")
    edit.add_argument("--corpus", help="text file (one passage per line) for the key covariance")
    edit.add_argument("--regularisation", type=float, default=0.1, help="added to the normalised covariance")
    edit.add_argument("--steps", type=int, default=40, help="optimiser steps for the new value")
    edit.add_argument("--learning-rate", type=float, default=0.5)
    edit.add_argument(
        "--expert",
        choices=("routed", "shared"),
        default="routed",
        help="in a mixture-of-experts layer: edit the subject's top routed expert or the shared expert",
    )
    edit.add_argument("-o", "--output", required=True, help="the edited model.dllm file to write")


def run(args: argparse.Namespace, engine: DllmEngine) -> int:
    model = engine.model
    if isinstance(model, TextToText):
        if args.command not in TEXT_TO_TEXT_COMMANDS:
            print(f"dllm {args.command}: needs a decoder-only model, not a text-to-text one", file=sys.stderr)
            return 1
        return _text_to_text(args, engine, model)
    if not isinstance(model, Transformer):
        print(f"dllm {args.command}: needs an imported model (--model or $DLLM_MODEL)", file=sys.stderr)
        return 1
    try:
        if args.command == "neighbours":
            return _neighbours(args, engine, model)
        if args.command == "steer":
            return _steer(args, engine, model)
        if args.command == "sae":
            return _sae(args, engine, model)
        if getattr(args, "answer", ""):
            raise ValueError("--answer is for text-to-text models")
        if getattr(args, "cross", False):
            raise ValueError("--cross is for text-to-text models")
        text = engine.render_chat([ChatMessage("user", args.prompt)]) if args.chat else args.prompt
        tokens = engine.tokenizer.encode(text)
        if not tokens:
            raise ValueError("the prompt has no tokens")
        if args.command == "lens":
            return _lens(args, engine, model, tokens)
        if args.command == "experts":
            return _experts(args, engine, model, tokens)
        return _attention(args, engine, model, tokens)
    except ValueError as error:
        print(f"dllm {args.command}: {error}", file=sys.stderr)
        return 1


def _text_to_text(args: argparse.Namespace, engine: DllmEngine, model: TextToText) -> int:
    """``lens``, ``attention`` and ``steer`` on a T5 model: the prompt (or the chat template's rendering) is the
    source, ``--answer`` the answer so far; ``lens`` and ``attention`` report the decoder's positions (its start
    token, then the answer)."""
    try:
        if args.command == "steer":
            return _steer(args, engine, model)
        text = engine.render_chat([ChatMessage("user", args.prompt)]) if args.chat else args.prompt
        source = with_end_of_source(model, engine.tokenizer.encode(text))
        answer = engine.tokenizer.encode(args.answer)
        if model.end_of_source in answer:
            raise ValueError("--answer must not contain </s>, which ends the answer")
        tokens = source + answer
        if args.command == "lens":
            return _lens(args, engine, model, tokens)
        return _attention(args, engine, model, tokens)
    except ValueError as error:
        print(f"dllm {args.command}: {error}", file=sys.stderr)
        return 1


def _texts(engine: DllmEngine, tokens: list[int]) -> list[str]:
    return [token_text(engine.tokenizer, t) for t in tokens]


def _write(path: str, content: str) -> None:
    Path(path).write_text(content, encoding="utf-8", newline="\n")
    print(f"wrote: {path}", file=sys.stderr)


def _lens(args: argparse.Namespace, engine: DllmEngine, model: Transformer | TextToText, tokens: list[int]) -> int:
    lens = logit_lens(model, tokens, args.top_k)
    tokens = list(lens.tokens)  # a text-to-text model's decoder inputs
    if not -len(tokens) <= args.position < len(tokens):
        raise ValueError(f"--position must be between {-len(tokens)} and {len(tokens) - 1}")
    ids = sorted({p.token for layer in lens.predictions for row in layer for p in row})
    predicted = {t: token_text(engine.tokenizer, t) for t in ids}
    texts = _texts(engine, tokens)
    if args.json:
        result = {
            "model": model.id,
            "tokens": [{"id": t, "text": s} for t, s in zip(tokens, texts, strict=True)],
            "layers": [
                [[{"id": p.token, "text": predicted[p.token], "p": p.probability} for p in row] for row in layer]
                for layer in lens.predictions
            ],
        }
        print(json.dumps(result, ensure_ascii=False, indent=1))
    else:
        position = args.position % len(tokens)
        print(f"position {position} ({texts[position]!r}); next-token predictions after each layer:")
        for layer, rows in enumerate(lens.predictions):
            label = "embed" if layer == 0 else f"{layer:5d}"
            shown = "  ".join(f"{predicted[p.token]!r} {p.probability:.3f}" for p in rows[position])
            print(f"{label}  {shown}")
    if args.html:
        _write(args.html, lens_html(lens, texts, predicted, f"Logit lens: {model.id}"))
    return 0


def _attention(args: argparse.Namespace, engine: DllmEngine, model: Transformer | TextToText, tokens: list[int]) -> int:
    config = model.config
    count = config.decoder_layers if isinstance(model, TextToText) else config.layers
    if args.layer is not None and not 1 <= args.layer <= count:
        raise ValueError(f"--layer must be between 1 and {count}")
    if args.head is not None and not 0 <= args.head < config.heads:
        raise ValueError(f"--head must be between 0 and {config.heads - 1}")
    keys = None  # the key positions' tokens when they differ from the queries' (cross-attention)
    if isinstance(model, TextToText):
        recorded_t5 = trace_text_to_text(model, tokens, logits=False)
        maps = recorded_t5.cross_attention if args.cross else recorded_t5.attention
        tokens = list(recorded_t5.tokens)
        if args.cross:
            keys = list(recorded_t5.source)
    else:
        maps = trace(model, tokens).attention
    assert maps is not None
    layers = [args.layer - 1] if args.layer is not None else list(range(count))
    heads = [args.head] if args.head is not None else list(range(config.heads))
    texts = _texts(engine, tokens)
    key_texts = texts if keys is None else _texts(engine, keys)
    if args.json:
        result: dict[str, Any] = {
            "model": model.id,
            "tokens": [{"id": t, "text": s} for t, s in zip(tokens, texts, strict=True)],
        }
        if keys is not None:
            result["source"] = [{"id": t, "text": s} for t, s in zip(keys, key_texts, strict=True)]
        result["cross_attention" if keys is not None else "attention"] = {
            str(layer + 1): {str(head): maps[layer, head].tolist() for head in heads} for layer in layers
        }
        print(json.dumps(result, ensure_ascii=False))
    else:
        for layer in layers:
            for head in heads:
                print(f"layer {layer + 1} head {head}:")
                for query, text in enumerate(texts):
                    visible = len(key_texts) if keys is not None else query + 1
                    weights = maps[layer, head, query]
                    best = top_k(weights[:visible], min(args.top_k, visible))
                    shown = "  ".join(f"{key_texts[key]!r} {weights[key]:.3f}" for key in best)
                    print(f"  {text!r:>14} -> {shown}")
    if args.html:
        kind = "Cross-attention" if keys is not None else "Attention"
        page = attention_html(maps, texts, layers, heads, f"{kind}: {model.id}", keys=key_texts)
        _write(args.html, page)
    return 0


def _experts(args: argparse.Namespace, engine: DllmEngine, model: Transformer, tokens: list[int]) -> int:
    from etalii_dllm.interpret.experts import routing

    result = routing(model, tokens)
    rows = list(range(len(result.layers)))
    if args.layer is not None:
        if args.layer - 1 not in result.layers:
            shown = ", ".join(str(layer + 1) for layer in result.layers)
            raise ValueError(f"--layer must be a mixture-of-experts layer ({shown})")
        rows = [result.layers.index(args.layer - 1)]
    texts = _texts(engine, tokens)
    usage = result.usage()
    if args.json:
        payload = {
            "model": model.id,
            "tokens": [{"id": t, "text": s} for t, s in zip(tokens, texts, strict=True)],
            "layers": {
                str(result.layers[i] + 1): {
                    "experts": result.experts[i].tolist(),
                    "weights": result.weights[i].tolist(),
                    "usage": usage[i].tolist(),
                }
                for i in rows
            },
        }
        print(json.dumps(payload, ensure_ascii=False))
    else:
        for i in rows:
            print(f"layer {result.layers[i] + 1}:")
            for position, text in enumerate(texts):
                chosen = zip(result.experts[i, position], result.weights[i, position], strict=True)
                shown = "  ".join(f"{int(e):3d} {w:.3f}" for e, w in chosen)
                print(f"  {text!r:>14} -> {shown}")
            print("  usage: " + " ".join(f"{e}:{int(n)}" for e, n in enumerate(usage[i]) if n))
    return 0


def _neighbours(args: argparse.Namespace, engine: DllmEngine, model: Transformer) -> int:
    result = neighbours(model, engine.tokenizer, args.expression, args.top_k, args.space)
    if args.json:
        payload = {
            "model": model.id,
            "expression": result.expression,
            "space": result.space,
            "terms": [{"sign": t.sign, "text": t.text, "tokens": list(t.tokens)} for t in result.terms],
            "neighbours": [{"id": n.token, "text": n.text, "similarity": n.similarity} for n in result.neighbours],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=1))
    else:
        for n in result.neighbours:
            print(f"{n.similarity:8.4f}  {n.text!r}  ({n.token})")
    if args.svg:
        _write(args.svg, word_cloud_svg(result))
    if args.html:
        _write(args.html, word_cloud_html(result))
    return 0


def _prompts(inline: list[str], path: str | None) -> list[str]:
    prompts = list(inline)
    if path:
        try:
            lines = Path(path).read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise ValueError(str(error)) from error
        prompts += [line for line in lines if line.strip()]
    return prompts


def _steer(args: argparse.Namespace, engine: DllmEngine, model: Transformer | TextToText) -> int:
    from etalii_dllm.interpret.steering import build_steering_vector, steered_layers

    positive = _prompts(args.positive, args.positive_file)
    negative = _prompts(args.negative, args.negative_file)
    if not positive or not negative:
        raise ValueError("needs at least one --positive and one --negative prompt")
    layers = steered_layers(model)
    layer = args.layer if args.layer is not None else max(1, layers // 3)
    vector = build_steering_vector(model, engine.tokenizer, positive, negative, layer, args.strength)
    vector.save(args.output)
    norm = float(np.sqrt(sum_squares(vector.vector)))
    print(f"wrote:    {args.output}")
    print(f"layer:    {layer} of {layers}, norm {norm:.4f}, strength {args.strength}")
    print(f"use with: dllm --model ... --steer {args.output} chat ...")
    return 0


def run_edit(args: argparse.Namespace) -> int:
    """``dllm edit``: loads the base file itself (it writes a new one rather than serving a model)."""
    from etalii_dllm.interpret.editing import DEFAULT_CORPUS, EditRequest, rome, write_edited_model
    from etalii_dllm.modelfile import ModelFile, ModelFileError

    try:
        base = ModelFile(args.base)
        engine = DllmEngine.from_model_file(args.base, verify=False)
        assert isinstance(engine.model, Transformer)
        corpus = _prompts([], args.corpus) if args.corpus else list(DEFAULT_CORPUS)
        result = rome(
            engine.model,
            engine.tokenizer,
            EditRequest(args.prompt, args.subject, args.target),
            layer=args.layer,
            contexts=args.context,
            corpus=corpus,
            regularisation=args.regularisation,
            steps=args.steps,
            learning_rate=args.learning_rate,
            expert=args.expert,
        )
        fingerprint = write_edited_model(base, result, args.output)
    except (ModelFileError, OSError, ValueError) as error:
        print(f"dllm edit: {error}", file=sys.stderr)
        return 1
    record = result.record
    expert = f", expert {record['expert']} (weight {record['routing_weight']:.4f})" if "expert" in record else ""
    print(f"edited:             layer {record['layer']}{expert}, {record['optimiser']['steps']} steps")
    print(f"p(target):          {result.probability_before:.4f} -> {result.probability_after:.4f}")
    print(f"wrote:              {args.output}")
    print(f"system_fingerprint: {fingerprint}")
    return 0


def _sae(args: argparse.Namespace, engine: DllmEngine, model: Transformer) -> int:
    from etalii_dllm.interpret.sae import (
        SaeConfig,
        SaeStep,
        SparseAutoencoder,
        collect_activations,
        feature_report,
        train_sae,
    )

    if args.sae_command == "train":
        layer = args.layer if args.layer is not None else max(1, model.config.layers // 2)
        config = SaeConfig(args.features, args.l1, args.learning_rate, args.steps, args.batch_size, args.seed)
        activations = collect_activations(model, engine.tokenizer, _prompts([], args.corpus), layer)
        every = max(1, config.steps // 20)

        def report(step: SaeStep) -> None:
            if step.step % every == 0 or step.step == config.steps:
                print(f"step {step.step:6d}  loss {step.loss:.6f}  mse {step.mse:.6f}", file=sys.stderr)

        print(f"layer {layer}: {activations.values.shape[0]} positions", file=sys.stderr)
        sae = train_sae(activations, config, model.weights_fingerprint, report)
        sae.save(args.output)
        print(f"wrote: {args.output}")
        return 0
    sae = SparseAutoencoder.load(args.sae)
    if args.sae_command == "steer":
        vector = sae.steering_vector(args.feature, args.strength)
        vector.save(args.output)
        print(f"wrote:    {args.output}")
        print(f"use with: dllm --model ... --steer {args.output} chat ...")
        return 0
    activations = collect_activations(model, engine.tokenizer, _prompts([], args.corpus), sae.layer)
    report = feature_report(sae, activations, args.feature, args.top_k, args.count)

    def context(text: int, position: int) -> tuple[str, str, str]:
        ids = activations.tokens[text]
        before = "".join(token_text(engine.tokenizer, t) for t in ids[max(0, position - 6) : position])
        after = "".join(token_text(engine.tokenizer, t) for t in ids[position + 1 : position + 3])
        return before, token_text(engine.tokenizer, ids[position]), after

    if args.json:
        payload = [
            {
                "feature": feature.index,
                "frequency": feature.frequency,
                "max_activation": feature.max_activation,
                "examples": [
                    {"activation": e.activation, "context": list(context(e.text, e.position))} for e in feature.examples
                ],
            }
            for feature in report
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=1))
        return 0
    for feature in report:
        print(f"feature {feature.index}: active on {feature.frequency:.1%}, max {feature.max_activation:.3f}")
        for example in feature.examples:
            before, token, after = context(example.text, example.position)
            print(f"  {example.activation:8.3f}  {before!r} [{token!r}] {after!r}")
    return 0
