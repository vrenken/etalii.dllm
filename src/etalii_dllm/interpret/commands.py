"""The interpretability subcommands of ``dllm``: ``lens``, ``attention`` and ``neighbours``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from etalii_dllm.chat import ChatMessage
from etalii_dllm.engine import DllmEngine
from etalii_dllm.interpret.embeddings import SPACES, neighbours, token_text
from etalii_dllm.interpret.lens import logit_lens, top_k
from etalii_dllm.interpret.render import attention_html, lens_html, word_cloud_html, word_cloud_svg
from etalii_dllm.interpret.trace import trace
from etalii_dllm.transformer import Transformer

COMMANDS = ("lens", "attention", "neighbours")


def add_commands(commands: Any) -> None:
    """Adds the subcommands to ``dllm``'s subparsers."""
    lens = commands.add_parser("lens", help="logit lens: the top predictions after every layer")
    attention = commands.add_parser("attention", help="attention maps: where each head looks")
    for command in (lens, attention):
        command.add_argument("--prompt", required=True)
        command.add_argument("--chat", action="store_true", help="wrap the prompt in the model's chat template")
        command.add_argument("--json", action="store_true", help="print the full result as JSON")
        command.add_argument("--html", metavar="FILE", help="also write a self-contained HTML view")
    lens.add_argument("--top-k", type=int, default=5)
    lens.add_argument("--position", type=int, default=-1, help="position to print (default: the last)")
    attention.add_argument("--layer", type=int, help="1-based layer (default: all)")
    attention.add_argument("--head", type=int, help="0-based head (default: all)")
    attention.add_argument("--top-k", type=int, default=3, help="keys printed per query")

    near = commands.add_parser("neighbours", help="nearest tokens in embedding space, e.g. 'king - man + woman'")
    near.add_argument("expression", help="a word, or words joined by ' + ' and ' - '; quote to keep spaces")
    near.add_argument("--top-k", type=int, default=20)
    near.add_argument("--space", choices=SPACES, default="input", help="input embeddings or LM head rows")
    near.add_argument("--json", action="store_true")
    near.add_argument("--svg", metavar="FILE", help="write a word cloud SVG")
    near.add_argument("--html", metavar="FILE", help="write a word cloud and table as HTML")


def run(args: argparse.Namespace, engine: DllmEngine) -> int:
    model = engine.model
    if not isinstance(model, Transformer):
        print(f"dllm {args.command}: needs an imported model (--model or $DLLM_MODEL)", file=sys.stderr)
        return 1
    try:
        if args.command == "neighbours":
            return _neighbours(args, engine, model)
        text = engine.render_chat([ChatMessage("user", args.prompt)]) if args.chat else args.prompt
        tokens = engine.tokenizer.encode(text)
        if not tokens:
            raise ValueError("the prompt has no tokens")
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


def _lens(args: argparse.Namespace, engine: DllmEngine, model: Transformer, tokens: list[int]) -> int:
    lens = logit_lens(model, tokens, args.top_k)
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


def _attention(args: argparse.Namespace, engine: DllmEngine, model: Transformer, tokens: list[int]) -> int:
    config = model.config
    if args.layer is not None and not 1 <= args.layer <= config.layers:
        raise ValueError(f"--layer must be between 1 and {config.layers}")
    if args.head is not None and not 0 <= args.head < config.heads:
        raise ValueError(f"--head must be between 0 and {config.heads - 1}")
    recorded = trace(model, tokens)
    assert recorded.attention is not None
    layers = [args.layer - 1] if args.layer is not None else list(range(config.layers))
    heads = [args.head] if args.head is not None else list(range(config.heads))
    texts = _texts(engine, tokens)
    if args.json:
        result = {
            "model": model.id,
            "tokens": [{"id": t, "text": s} for t, s in zip(tokens, texts, strict=True)],
            "attention": {
                str(layer + 1): {str(head): recorded.attention[layer, head].tolist() for head in heads}
                for layer in layers
            },
        }
        print(json.dumps(result, ensure_ascii=False))
    else:
        for layer in layers:
            for head in heads:
                print(f"layer {layer + 1} head {head}:")
                for query, text in enumerate(texts):
                    weights = recorded.attention[layer, head, query]
                    best = top_k(weights[: query + 1], min(args.top_k, query + 1))
                    shown = "  ".join(f"{texts[key]!r} {weights[key]:.3f}" for key in best)
                    print(f"  {text!r:>14} -> {shown}")
    if args.html:
        _write(args.html, attention_html(recorded.attention, texts, layers, heads, f"Attention: {model.id}"))
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
