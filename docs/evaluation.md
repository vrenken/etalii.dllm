# Reproducible evaluation

`dllm eval` scores a model on a task file with numbers that are the same, bit for bit, on every supported machine,
device and thread count. Two runs, two machines or two engine versions compare with one line: the `fingerprint`.

```bash
dllm --model smollm2-135m.dllm eval tests/data/eval-tiny.jsonl        # multiple choice
dllm --model smollm2-135m.dllm eval my-text.txt                       # perplexity
dllm --model smollm2-135m.dllm eval tasks.jsonl --json -o report.json # per-item results too
```

```
task:               eval-tiny.jsonl
kind:               multiple_choice
engine:             0.2.0
model:              HuggingFaceTB/SmolLM2-135M-Instruct
system_fingerprint: fp_2e4d93db18c1
items:              5
accuracy:           1.0
accuracy_norm:      1.0
fingerprint:        5083f67f1060e5eb61e61e230f7160683d9fdaf3eb02079b847f202f03419aa7
```

## Task files

JSON lines, one item per line, all of one kind:

| Kind | Item | Scores |
| --- | --- | --- |
| Multiple choice | `{"context": "The capital of France is", "choices": [" Paris", " Rome"], "answer": 0}` | `accuracy`: the most likely choice is the answer; `accuracy_norm`: the most likely per byte of the choice |
| Perplexity | `{"text": "..."}` (a `.txt` file is one such item) | `log_likelihood`, `tokens`, `perplexity` (e to the mean negative log-likelihood per token), `bits_per_byte` |

The scores are computed the way lm-evaluation-harness computes them:

- A choice is scored as a continuation of its context: the sum of the log-probabilities of its tokens. Spaces at the
  end of the context move to the start of the choice, so `"Q: A"` + `" B"` and `"Q: A "` + `"B"` give the same
  tokens. A tie goes to the first choice.
- Every sequence starts with the tokenizer's begin-of-sequence token (its end-of-sequence token when it has none), so
  the first token of a text is scored too.
- A text longer than `--max-length` (default 1024, or the model's context when it is shorter) is scored in
  consecutive windows, each starting from the token before its first target.

## Why the numbers repeat

- Logits come from the decoder's fixed-order kernels. A context and its continuation go through one prefill, which
  gives each position the bits of decoding the tokens one at a time (`tests/test_evaluation.py` checks it).
- Log-probabilities come from the `log_softmax` kernel, sums from `numerics.sum_` (index order, double
  accumulator), and `exp`/`log` from the portable kernels, never from the C library.
- `fingerprint` is the SHA-256 of the per-item results with every float in its exact hexadecimal spelling and every
  token's log-probability hashed as float32 bits. The report's `results` (with `--json`) lists them per item.

Golden fingerprints are pinned for the placeholder model and a tiny transformer (`tests/test_evaluation.py`), and for
every verified real model on all release platforms and SIMD paths (`test_eval_golden` in
`tests/test_reference_models.py`).

## Limits

- Only log-likelihood tasks: no generated answers scored against references (those depend on a stop rule and a
  matching rule; use `dllm chat` with receipts for them).
- No few-shot sampling: put the examples in the context.
- The task's scores are the model's, not a leaderboard's: bundled tasks are tiny and only meant for testing.
