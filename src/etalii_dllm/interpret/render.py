"""Self-contained HTML and SVG views of interpretability results (no scripts, no external files).

The pages are built from the deterministic results with fixed formatting, and the word cloud's spiral uses the
portable ``sin``/``cos`` kernels, so the same input writes the same bytes on every machine.
"""

from __future__ import annotations

from collections.abc import Sequence
from html import escape

import numpy as np

from etalii_dllm import _kernels
from etalii_dllm.interpret.embeddings import Neighbourhood
from etalii_dllm.interpret.lens import Lens

_STYLE = """
:root { color-scheme: light dark; --fg: #1f2328; --bg: #ffffff; --muted: #656d76; --cell: 37, 99, 235; }
@media (prefers-color-scheme: dark) { :root { --fg: #e6edf3; --bg: #0d1117; --muted: #8d96a0; } }
body { font: 14px/1.4 system-ui, sans-serif; color: var(--fg); background: var(--bg); margin: 16px; }
h1 { font-size: 18px; } h2 { font-size: 15px; margin-top: 24px; } p { color: var(--muted); }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; font: 12px ui-monospace, monospace; }
th, td { padding: 3px 6px; border: 1px solid rgba(128, 128, 128, 0.25); white-space: pre; text-align: left; }
th { color: var(--muted); font-weight: normal; }
"""


def _page(title: str, body: str) -> str:
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{escape(title)}</title>\n<style>{_STYLE}</style>\n</head>\n<body>\n{body}</body>\n</html>\n"
    )


def _shown(text: str) -> str:
    """Token text with whitespace made visible."""
    return escape(text.replace(" ", "␣").replace("\n", "↵").replace("\t", "⇥"))


def _shade(value: float) -> str:
    alpha = min(max(value, 0.0), 1.0)
    return f"background: rgba(var(--cell), {alpha:.3f})" + ("; color: #fff" if alpha > 0.55 else "")


def lens_html(lens: Lens, texts: Sequence[str], predicted: dict[int, str], title: str = "Logit lens") -> str:
    """A grid of layers (rows, the full model last) by positions (columns): each cell the top prediction, shaded by
    its probability, with the next few in its tooltip. ``texts`` are the prompt tokens' texts, ``predicted`` maps
    the predicted token ids to their texts."""
    header = "".join(f"<th>{_shown(text)}</th>" for text in texts)
    rows = []
    for layer in range(lens.layers, -1, -1):
        cells = []
        for predictions in lens.predictions[layer]:
            best = predictions[0]
            tip = escape(", ".join(f"{predicted[p.token]!r} {p.probability:.3f}" for p in predictions), quote=True)
            cells.append(f'<td title="{tip}" style="{_shade(best.probability)}">{_shown(predicted[best.token])}</td>')
        label = "embeddings" if layer == 0 else f"layer {layer}"
        rows.append(f"<tr><th>{label}</th>{''.join(cells)}</tr>")
    body = (
        f"<h1>{escape(title)}</h1>\n<p>Top next-token prediction after each layer, at each position. Darker cells are "
        "more certain; hover a cell for the runners-up.</p>\n"
        f'<div class="scroll"><table>\n<tr><th></th>{header}</tr>\n' + "\n".join(rows) + "\n</table></div>\n"
    )
    return _page(title, body)


def attention_html(
    attention: np.ndarray,
    texts: Sequence[str],
    layers: Sequence[int],
    heads: Sequence[int],
    title: str = "Attention",
    keys: Sequence[str] | None = None,
) -> str:
    """Heatmaps of ``attention[layer, head, query, key]`` for the given layers and heads; rows are queries,
    columns keys, labelled with the token ``texts`` (the keys with ``keys`` when they differ, as in a text-to-text
    model's cross-attention over its source)."""
    shown = [_shown(text) for text in texts]
    shown_keys = shown if keys is None else [_shown(text) for text in keys]
    sections = []
    for layer in layers:
        for head in heads:
            weights = attention[layer, head]
            header = "".join(f"<th>{text}</th>" for text in shown_keys)
            rows = []
            for query, text in enumerate(shown):
                cells = "".join(
                    f'<td title="{weights[query, key]:.4f}" style="{_shade(float(weights[query, key]))}"></td>'
                    for key in range(len(shown_keys))
                )
                rows.append(f"<tr><th>{text}</th>{cells}</tr>")
            sections.append(
                f"<h2>Layer {layer + 1}, head {head}</h2>\n"
                f'<div class="scroll"><table>\n<tr><th></th>{header}</tr>\n' + "\n".join(rows) + "\n</table></div>"
            )
    body = (
        f"<h1>{escape(title)}</h1>\n<p>Rows are query positions, columns the positions they attend to; darker is "
        "more attention. Hover a cell for its probability.</p>\n" + "\n".join(sections) + "\n"
    )
    return _page(title, body)


_PALETTE = ("#2563eb", "#16a34a", "#d97706", "#9333ea", "#dc2626", "#0891b2", "#4d7c0f", "#c026d3")


def word_cloud_svg(neighbourhood: Neighbourhood, width: int = 720, height: int = 440) -> str:
    """The neighbours as a word cloud: larger words are more similar. Words are placed largest first along an
    Archimedean spiral from the centre, each at the first point where its box overlaps no earlier word; the
    expression itself sits in the middle."""
    items = [(neighbourhood.expression, 1.0, 0)] + [
        (n.text.strip() or repr(n.text), n.similarity, index + 1) for index, n in enumerate(neighbourhood.neighbours)
    ]
    values = [similarity for _, similarity, _ in items[1:]] or [1.0]
    low, high = min(values), max(values)
    placed: list[tuple[float, float, float, float]] = []
    words = []
    for text, similarity, rank in items:
        if rank == 0:
            size = 34.0
        else:
            scale = (similarity - low) / (high - low) if high > low else 1.0
            size = 12.0 + 18.0 * scale
        box_width = 0.6 * size * max(len(text), 1) + 4.0
        box_height = size * 1.1
        step = 0
        while step < 4000:
            angle = 0.12 * step
            radius = 2.2 * angle
            x = width / 2 + radius * _kernels.cos(angle)
            y = height / 2 + 0.6 * radius * _kernels.sin(angle)
            box = (x - box_width / 2, y - box_height / 2, x + box_width / 2, y + box_height / 2)
            inside = box[0] >= 0 and box[1] >= 0 and box[2] <= width and box[3] <= height
            if inside and all(box[2] <= b[0] or box[0] >= b[2] or box[3] <= b[1] or box[1] >= b[3] for b in placed):
                placed.append(box)
                colour = "currentColor" if rank == 0 else _PALETTE[(rank - 1) % len(_PALETTE)]
                tip = "" if rank == 0 else f"<title>{escape(text)} {similarity:.4f}</title>"
                words.append(
                    f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size:.1f}" fill="{colour}" '
                    f'text-anchor="middle" dominant-baseline="central">{tip}{escape(text)}</text>'
                )
                break
            step += 1
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        'font-family="system-ui, sans-serif" role="img">\n'
        f"<title>Nearest tokens to {escape(neighbourhood.expression)}</title>\n" + "\n".join(words) + "\n</svg>\n"
    )


def word_cloud_html(neighbourhood: Neighbourhood) -> str:
    rows = "".join(
        f"<tr><td>{_shown(n.text)}</td><td>{n.similarity:.4f}</td><td>{n.token}</td></tr>"
        for n in neighbourhood.neighbours
    )
    body = (
        f"<h1>Nearest tokens to {escape(neighbourhood.expression)}</h1>\n"
        f"<p>Cosine similarity in the model's {neighbourhood.space} embedding space.</p>\n"
        f'<div class="scroll">{word_cloud_svg(neighbourhood)}</div>\n'
        f"<table>\n<tr><th>token</th><th>similarity</th><th>id</th></tr>{rows}\n</table>\n"
    )
    return _page(f"Neighbours of {neighbourhood.expression}", body)
