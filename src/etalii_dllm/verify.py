"""``dllm verify``: one fingerprint that says whether two machines give the same bits (issue #100).

It runs a fixed workload and hashes every result: the kernels on fixed inputs, the pinned Unicode handling, and, with
the engine's model, a tokenized corpus, the logits of a prompt and a greedy and a sampled answer. Equal ``verify``
fingerprints on two machines mean they compute the same bits; a differing part shows where they diverge. The
model-independent parts are also compared with the values this release was built with (``REFERENCE``), so a single
machine can already tell whether it matches.

Nothing here reads a clock or any entropy, so the report itself is reproducible.
"""

from __future__ import annotations

import hashlib
import math
import platform
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any

import numpy as np

from etalii_dllm import __version__, numerics, unicode
from etalii_dllm.infill import FimTokens
from etalii_dllm.sampling import GREEDY, SamplingOptions

PROMPT = "The capital of France is"
MAX_TOKENS = 16
ROLLED_ROOM = 3
THINKING_BUDGET = 2
"""Tokens of thinking the budgeted check allows before the block is closed."""
"""Tokens the rolled check's context window holds beyond the prompt."""
SAMPLED = SamplingOptions(temperature=0.8, seed=7)
NEGATIVE_PROMPT = "The capital of Germany is"
SUFFIX = ", and its largest city too."
"""The text after the gap ``dllm verify --reference`` fills in, for models with fill-in-the-middle tokens."""
GUIDED = SamplingOptions(temperature=0.8, seed=7, negative_prompt=NEGATIVE_PROMPT, guidance_scale=2.0)
"""The guided check: a sampled answer with classifier-free guidance (docs/specification.md#guided-decoding)."""
BEAM_WIDTH = 3
"""The beam check: the ranked answers of a beam search this wide (docs/specification.md#beam-search)."""
CONTROLLED = SamplingOptions(
    temperature=0.9,
    top_k=40,
    seed=11,
    min_p=0.02,
    repetition_penalty=1.3,
    frequency_penalty=0.4,
    presence_penalty=0.2,
    logit_bias=((13, 1.5), (32, -2.0)),
)
"""Every Phase 21 decoding control at once (``dllm verify --reference``)."""
MODERN = SamplingOptions(
    temperature=1.1,
    seed=13,
    top_n_sigma=2.0,
    typical_p=0.9,
    xtc_probability=0.5,
    xtc_threshold=0.1,
    dry_multiplier=0.8,
    dry_allowed_length=1,
)
"""Every Phase 49 sampler at once: top-n-sigma, typical-p, XTC and DRY (``dllm verify --reference``)."""

# Text that exercises normalisation, case, categories and scripts (Latin, CJK, Hangul, Greek, Cyrillic, emoji,
# compatibility and combining characters, letters new in Unicode 15/15.1), escaped to keep the source ASCII.
CORPUS = [
    "Hello, world! I'm sure you'll see they've done it.",
    "Numbers: 1234567890, 3.14159, -42 and 2026-09-29.",
    (
        "\u00dcn\u00efc\u00f6d\u00e9: caf\u00e9, na\u00efve, Stra\u00dfe, \u65e5\u672c\u8a9e\u306e\u30c6"
        "\u30ad\u30b9\u30c8, \u4e2d\u6587, \ud55c\uad6d\uc5b4, \u0395\u03bb\u03bb\u03b7\u03bd\u03b9\u03ba"
        "\u03ac \u039f\u0394\u03a5\u03a3\u03a3\u0395\u03a5\u03a3, \u0440\u0443\u0441\u0441\u043a\u0438\u0439"
    ),
    "Compatibility: \ufb01 \u2460 \uff21 \u212b \u2126, combining: e\u0301 q\u0323\u0307, jamo: \u1100\u1161\u11a8",
    (
        "Emoji \U0001f680\U0001f525\U0001f44d\U0001f3fd, symbols \u00a9\u00ae\u2122 \u2264\u2265\u2260 \u2211"
        "\u222b, newer letters: \U00031350\U0002ebf0"
    ),
    "  leading spaces,\ttabs\tand\n\nnew lines\r\n",
]

# The model-independent parts as computed when this release was built (tests/test_verify.py keeps them current).
REFERENCE = {
    "kernels": "b47cb0a7e81cfc06a73435e64342e7fe3efadf608511a77a32071045c32cd66d",
    "unicode": "72dd232a0b75a6974a793c2c5293ea57bad7d586b37286090ed5b059cf048caa",
}


def _sha256(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "little"))
        digest.update(part)
    return digest.hexdigest()


def _gaussian(seed: int, *shape: int) -> np.ndarray:
    return numerics.fill_gaussian(seed, math.prod(shape)).reshape(shape)


def kernels_fingerprint() -> str:
    """Every kernel family on fixed pseudo-random inputs: float32 and Q8_0 linear, matmul, norm, activations, RoPE,
    attention, softmax and the transcendentals."""
    x, w, bias = _gaussian(1, 8, 64), _gaussian(2, 48, 64), _gaussian(3, 48)
    q, k, v = _gaussian(4, 8, 4, 16), _gaussian(5, 8, 2, 16), _gaussian(6, 8, 2, 16)
    outputs = [
        numerics.linear(x, w, bias),
        numerics.linear(x, numerics.QuantizedWeight(w), bias),
        numerics.matmul(x, np.ascontiguousarray(w.T)),
        numerics.rms_norm(x, _gaussian(7, 64), 1e-6),
        numerics.silu(x),
        numerics.gelu(x),
        numerics.gelu(x, approximate="tanh"),
        numerics.rope(q, np.arange(8) * 97, numerics.rope_inv_freq(16, 10000.0)),
        numerics.attention(q, k, v),
        numerics.softmax(_gaussian(8, 256)),
    ]
    arguments = np.linspace(-20.0, 20.0, 257)
    scalars = np.array(
        [
            f(float(a))
            for f in (numerics.exp, numerics.sin, numerics.cos, numerics.tanh, numerics.erf)
            for a in arguments
        ]
        + [numerics.log(float(a)) for a in np.linspace(1e-3, 1e3, 257)],
        dtype="<f8",
    )
    return _sha256(*(np.asarray(o, dtype="<f4").tobytes() for o in outputs), scalars.tobytes())


def unicode_fingerprint() -> str:
    """The pinned Unicode handling on the corpus: the four normal forms, lower-casing and a category split."""
    words = unicode.compile(r"\p{L}+|\p{N}+|\p{M}+|\p{P}+|\p{S}+|\p{Z}+|\p{C}+")
    parts: list[bytes] = []
    for text in CORPUS:
        for form in ("NFC", "NFD", "NFKC", "NFKD"):
            parts.append(unicode.normalize(form, text).encode("utf-8"))
        parts.append(unicode.lower(text).encode("utf-8"))
        parts.append("\x00".join(m.group() for m in words.finditer(text)).encode("utf-8"))
    return _sha256(*parts)


def environment(engine: Any) -> dict[str, str]:
    """What could make two machines differ, for the report (not part of the fingerprint)."""

    def version(package: str) -> str:
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            return "not installed"

    import unicodedata

    device = getattr(engine.model, "device", "cpu")
    return {
        "etalii-dllm": __version__,
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.machine()}",
        "instruction set": numerics.instruction_set(),
        "threads": str(numerics.threads()),
        "device": device,
        "fp environment": "default" if numerics.fp_environment_is_canonical() else "changed (NumPy is affected)",
        "unicode tables": f"{unicode.unicode_version()} (Python's own: {unicodedata.unidata_version}, not used)",
        "numpy": np.__version__,
        "regex": version("regex"),
        "jinja2": version("jinja2"),
        "model": engine.model.id,
        "system_fingerprint": engine.system_fingerprint,
    }


@dataclass
class Report:
    parts: dict[str, str]
    environment: dict[str, str]
    mismatches: list[str] = field(default_factory=list)

    @property
    def fingerprint(self) -> str:
        return _sha256(*(f"{name}={value}".encode() for name, value in self.parts.items()))[:32]

    def as_dict(self) -> dict[str, Any]:
        return {
            "verify": self.fingerprint,
            "parts": self.parts,
            "reference_mismatches": self.mismatches,
            "environment": self.environment,
        }


def run(engine: Any) -> Report:
    """Runs the workload with ``engine`` (its model, tokenizer and sampler)."""
    parts = {"kernels": kernels_fingerprint(), "unicode": unicode_fingerprint()}
    parts["tokenizer"] = numerics.fingerprint(
        [t for text in CORPUS for t in [*engine.tokenizer.encode(text), -1]], dtype="<i4"
    )
    parts["logits"] = numerics.fingerprint(np.asarray(engine.model.forward(engine.tokenizer.encode(PROMPT))))
    parts["greedy"] = engine.complete(PROMPT, MAX_TOKENS, GREEDY).fingerprint
    parts["sampled"] = engine.complete(PROMPT, MAX_TOKENS, SAMPLED).fingerprint
    mismatches = [name for name, value in REFERENCE.items() if value and parts[name] != value]
    return Report(parts, environment(engine), mismatches)


@dataclass
class ReferenceCheck:
    """``dllm verify --reference``: the engine against the independent reference implementation (issue #162)."""

    results: dict[str, str]
    """``logits``, ``greedy``, ``sampled``, ``controlled``, ``modern``, ``rolled``, ``budgeted``, ``guided``,
    ``healed``, ``lengthened``, ``beam``, ``scored`` and, for models with fill-in-the-middle tokens, ``infilled``:
    ``"equal"``, or where the two first differ."""

    @property
    def equal(self) -> bool:
        return all(value == "equal" for value in self.results.values())

    def as_dict(self) -> dict[str, Any]:
        return {"equal": self.equal, **self.results}


def _first_difference(engine_tokens: Sequence[int], reference_tokens: Sequence[int]) -> str:
    if list(engine_tokens) == list(reference_tokens):
        return "equal"
    common = 0
    for a, b in zip(engine_tokens, reference_tokens, strict=False):
        if a != b:
            break
        common += 1
    return (
        f"differ from token {common}: engine {list(engine_tokens[common : common + 4])}, "
        f"reference {list(reference_tokens[common : common + 4])}"
    )


def check_reference(engine: Any, max_tokens: int = MAX_TOKENS) -> ReferenceCheck:
    """Runs the verify prompt through the engine and through :mod:`etalii_dllm.reference`, which shares no code with
    the compiled kernels, and compares the prompt's logits and a greedy and a sampled answer bit for bit. A difference
    means the kernels (a SIMD path, the thread pool, the GPU, the compiler) do not compute what the specification
    says on this machine."""
    from etalii_dllm import reference
    from etalii_dllm.generation import Generator
    from etalii_dllm.reasoning import Tracker
    from etalii_dllm.transformer import Transformer

    if not isinstance(engine.model, Transformer):
        raise ValueError("--reference needs a model file (--model or DLLM_MODEL)")
    twin = reference.ReferenceTransformer.from_engine_model(engine.model)
    context = engine.tokenizer.encode(PROMPT)
    logits = np.asarray(engine.model.forward(context), dtype=np.float32).reshape(-1)
    expected = twin.forward(context)
    results = {}
    if logits.tobytes() == expected.tobytes():
        results["logits"] = "equal"
    else:
        differing = np.flatnonzero(logits.view(np.uint32) != expected.view(np.uint32))
        results["logits"] = f"{len(differing)} of {len(logits)} differ, first at token id {int(differing[0])}"
    stops = sorted(engine.stop_tokens)
    token_bytes = [engine.tokenizer.decode_bytes([t]) for t in range(engine.model.vocabulary_size)]
    breakers = reference.dry_breakers(token_bytes, MODERN.dry_sequence_breakers)
    checks = (("greedy", GREEDY), ("sampled", SAMPLED), ("controlled", CONTROLLED), ("modern", MODERN))
    for name, options in checks:
        answer = engine.complete(PROMPT, max_tokens, options)
        sampler = reference.sampler(options, breakers if options.dry else ())
        tokens, _ = twin.generate(context, max_tokens, sampler, stops)
        results[name] = _first_difference(answer.tokens, tokens)
    # A window just past the prompt, so the answer rolls it (docs/specification.md#the-context-window) several times.
    rolling = Generator(engine.model, engine.tokenizer, stops)
    rolling.context_length = window = max(len(context) + ROLLED_ROOM, 2 * reference.ROLL_SINK + 4)
    answer = rolling.generate(PROMPT, max_tokens, GREEDY, overflow="roll")
    tokens, _ = twin.generate(context, max_tokens, reference.sampler(GREEDY), stops, overflow="roll", window=window)
    results["rolled"] = _first_difference(answer.tokens, tokens)
    # The answer as if the prompt had opened a <think> block with a budget of a few tokens, so it is closed with
    # fixed tokens (docs/specification.md#reasoning) and the rest follows them.
    plain = Generator(engine.model, engine.tokenizer, stops)
    answer = plain.generate(PROMPT, max_tokens, GREEDY, reasoning=Tracker(True, THINKING_BUDGET))
    budget = reference.ThinkingBudget(THINKING_BUDGET, True, engine.tokenizer.decode_bytes, engine.tokenizer.encode)
    tokens, _ = twin.generate(context, max_tokens, reference.sampler(GREEDY), stops, thinking=budget)
    results["budgeted"] = _first_difference(answer.tokens, tokens)
    # A sampled answer guided away from a negative prompt, which a second reference decoder runs alongside.
    answer = engine.complete(PROMPT, max_tokens, GUIDED)
    negative_tokens = engine.tokenizer.encode(NEGATIVE_PROMPT)
    negative = reference.ReferenceTransformer.from_engine_model(engine.model)
    fed = [0]

    def guide(logits: np.ndarray, generated: list[int]) -> np.ndarray:
        unfed = [*negative_tokens, *generated] if fed[0] == 0 and negative.length == 0 else generated[fed[0] :]
        fed[0] = len(generated)
        return reference.guided(logits, negative.forward(unfed), GUIDED.guidance_scale)

    negative.reset()
    tokens, _ = twin.generate(context, max_tokens, reference.sampler(GUIDED), stops, guide=guide)
    results["guided"] = _first_difference(answer.tokens, tokens)
    # A healed answer (docs/specification.md#token-healing): the prompt's last token taken back, the answer made to
    # start with its bytes.
    healed = engine.complete_stream(PROMPT, max_tokens, SAMPLED, token_healing=True).result()
    token_bytes = [engine.tokenizer.decode_bytes([t]) for t in range(engine.model.vocabulary_size)]
    allowed = reference.healing(engine.tokenizer.decode_bytes(context[-1:]), token_bytes)
    tokens, _ = twin.generate(context[:-1], max_tokens, reference.sampler(SAMPLED), stops, allowed=allowed)
    results["healed"] = _first_difference(healed.tokens, tokens)
    # A sampled answer that may not stop before it has max_tokens tokens
    # (docs/specification.md#length-and-stop-controls).
    lengthened = engine.complete_stream(PROMPT, max_tokens, SAMPLED, min_tokens=max_tokens).result()
    allowed = reference.minimum_length(max_tokens, stops, engine.model.vocabulary_size)
    tokens, _ = twin.generate(context, max_tokens, reference.sampler(SAMPLED), stops, allowed=allowed)
    results["lengthened"] = _first_difference(lengthened.tokens, tokens)
    # A middle filled in between the prompt and a suffix (docs/specification.md#fill-in-the-middle), for models with
    # FIM tokens.
    if FimTokens.of(engine.tokenizer) is not None:
        middle = engine.complete_stream(PROMPT, max_tokens, SAMPLED, suffix=SUFFIX).result()
        token_id = engine.tokenizer.token_to_id  # type: ignore[attr-defined]
        prompt, ends = reference.fill_in_the_middle(token_id, context, engine.tokenizer.encode(SUFFIX))
        tokens, _ = twin.generate(prompt, max_tokens, reference.sampler(SAMPLED), [*stops, *ends])
        results["infilled"] = _first_difference(middle.tokens, tokens)
    # The prompt scored token by token (docs/specification.md#prompt-scoring), as /v1/completions echoes it.
    from etalii_dllm import scoring

    scored = np.asarray([t.logprob for t in scoring.score_tokens(engine, context).tokens[1:]], dtype=np.float32)
    twin.reset()
    rows = [twin.forward([token]) for token in context[:-1]]
    expected_scores = np.asarray([reference.log_softmax(row)[t] for row, t in zip(rows, context[1:], strict=True)],
                                 dtype=np.float32)  # fmt: skip
    if scored.tobytes() == expected_scores.tobytes():
        results["scored"] = "equal"
    else:
        differing = np.flatnonzero(scored.view(np.uint32) != expected_scores.view(np.uint32))
        results["scored"] = f"{len(differing)} of {len(scored)} differ, first at token {int(differing[0]) + 1}"
    # A beam search (docs/specification.md#beam-search): every ranked answer's tokens and the bits of its scores.
    from etalii_dllm import beam

    found = beam.search_tokens(engine.model, context, BEAM_WIDTH, max_tokens, engine.stop_tokens, n_best=BEAM_WIDTH)
    ranked = twin.beam_search(context, BEAM_WIDTH, max_tokens, stops, n_best=BEAM_WIDTH)
    mine = [(list(h.tokens), h.finish_reason, h.log_likelihood.hex(), h.score.hex()) for h in found]
    theirs = [(tokens, reason, total.hex(), value.hex()) for tokens, reason, total, value in ranked]
    differing = [i for i, (a, b) in enumerate(zip(mine, theirs, strict=False)) if a != b]
    if mine == theirs:
        results["beam"] = "equal"
    elif differing:
        results["beam"] = f"answer {differing[0]} of {len(mine)} differs"
    else:
        results["beam"] = f"{len(mine)} answers, the reference has {len(theirs)}"
    return ReferenceCheck(results)
