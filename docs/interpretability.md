# Interpretability and editing

Tools to look inside a model and change it on purpose, built on the same deterministic kernels as inference
(Phase 11 of the [roadmap](../README.md#roadmap)).

Determinism matters more here than anywhere. An interpretability finding is a claim about numbers inside a
model: "layer 23 already predicts ` Paris`", "head 5.1 attends to the first token". In mainstream engines those
numbers move with batch size, thread count and GPU model, so a finding can be hard to reproduce and an edit's effect
hard to separate from noise. Here every activation, probability and similarity is the same bits on every run and
every supported machine. Anyone with the same `model.dllm` file can reproduce an observation exactly, and the
difference between an edited and an unedited model is exactly the edit.

| Tool | Command | Python | Status |
| --- | --- | --- | --- |
| Activation tracing | | `etalii_dllm.interpret.trace` | ✅ |
| Logit lens | `dllm lens` | `etalii_dllm.interpret.logit_lens` | ✅ |
| Attention maps | `dllm attention` | `trace(...).attention` | ✅ |
| Embedding explorer and word clouds | `dllm neighbours` | `etalii_dllm.interpret.neighbours` | ✅ |
| Steering vectors | `dllm steer`, `--steer` | | planned ([#112](https://github.com/vrenken/etalii.dllm/issues/112)) |
| Model editing (ROME) | `dllm edit` | | planned ([#113](https://github.com/vrenken/etalii.dllm/issues/113)) |
| Sparse autoencoders | `dllm sae` | | planned ([#114](https://github.com/vrenken/etalii.dllm/issues/114)) |

The commands take the usual `--model` option (or `DLLM_MODEL`) before the command name and run on the CPU.

## Activation tracing

```python
from etalii_dllm.interpret import trace
from etalii_dllm.engine import DllmEngine

engine = DllmEngine.from_model_file("smollm2-135m.dllm")
recorded = trace(engine.model, engine.tokenizer.encode("The capital of France is"))
recorded.residual  # [layers + 1, positions, hidden]: entering each layer (row 0: embeddings), then the output
recorded.middle  # [layers, positions, hidden]: after each attention block
recorded.attention  # [layers, heads, positions, positions]: attention probabilities (query, key)
recorded.mlp_activation  # [layers, positions, intermediate]: act(gate) * up, the input of the down projection
recorded.attention_output, recorded.mlp_output  # what each block adds to the residual stream
recorded.hidden, recorded.logits  # final-norm hidden states and next-token logits at every position
recorded.fingerprint()  # SHA-256 over the tokens and every array
```

Tracing is observation only: the traced pass computes exactly the bits of a normal one, so `recorded.logits[-1]`
equals `engine.model.forward(tokens)` bit for bit, and the residual stream is the exact running float32 sum of what
the blocks add. The attention probabilities come from a separate kernel, `attention_weights`, that repeats the
attention kernel's scoring, masking (causal, sliding window), soft-capping and softmax in the same order
([kernels](kernels.md#interpretability-kernels)).

To intervene, subclass `etalii_dllm.transformer.LayerHook` and pass it to `Transformer.run_hooked`. Its `residual`
method sees the residual stream at `"input"`, `"middle"` and `"output"` of every layer and may return a replacement;
the other methods observe. A hook that returns nothing cannot change a bit of the output.

## Logit lens

The [logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) puts the
residual stream after every layer through the final norm and the LM head, as if the model stopped there. The lens
of the last layer is the model's real prediction, bit for bit.

```bash
dllm --model smollm2-135m.dllm lens --prompt "The capital of France is" --top-k 3
dllm --model smollm2-135m.dllm lens --prompt "The capital of France is" --html lens.html   # grid of all positions
dllm --model qwen2.5-0.5b.dllm lens --chat --prompt "What is the capital of France?" --json
```

```text
position 4 (' is'); next-token predictions after each layer:
embed  ' is' 1.000  ' was' 0.000  ' are' 0.000
    1  ' a' 0.490  '\xa0' 0.282  ' so' 0.090
  ...
   28  ' the' 0.888  ' a' 0.030  ' D' 0.011
   29  ' the' 0.371  ' Le' 0.071  ' capital' 0.064
   30  ' Paris' 0.469  ' the' 0.266  ' located' 0.083
```

SmolLM2-135M only commits to ` Paris` in its last layer. `--position` picks the position printed (default: the
last); `--chat` wraps the prompt in the model's chat template; `--json` prints every layer and position; `--html`
writes a self-contained page with a cell per layer and position, shaded by probability.

## Attention maps

```bash
dllm --model smollm2-135m.dllm attention --prompt "The cat sat on the mat. The cat" --layer 5 --head 1
dllm --model smollm2-135m.dllm attention --prompt "The cat sat on the mat. The cat" --layer 5 --html attention.html
```

The text output lists, for every query position, the keys it attends to most. `--layer` is 1-based and `--head`
0-based; without them every layer or head is shown. `--html` writes one heatmap per head (rows: queries, columns:
keys), `--json` the full probability matrices.

## Embedding explorer and word clouds

```bash
dllm --model smollm2-135m.dllm neighbours "Paris" --top-k 8
dllm --model smollm2-135m.dllm neighbours "king - man + woman" --top-k 8 --svg cloud.svg
dllm --model smollm2-135m.dllm neighbours '"Paris"' --space output --html paris.html
```

```text
  0.7158  ' queen'  (15731)
  0.6289  ' Queen'  (9568)
  0.6247  ' princess'  (29347)
```

`neighbours` ranks every token by cosine similarity to a word, or to a sum and difference of words, in the model's
input embedding space (`--space output` uses the LM head rows; the same matrix for models with tied embeddings). A
bare word gets a leading space, as words inside a sentence have; quote a term to use it exactly as written. A word
of several tokens is the mean of their vectors. The query's own tokens and special tokens are left out.

`--svg` writes a word cloud (bigger is more similar) and `--html` a page with the cloud and the table. The cloud is
laid out along a spiral computed with the portable `sin`/`cos` kernels, so it is the same file on every machine.

## Reproducibility

Everything above is covered by `tests/test_interpret.py`: a trace of each tiny model family equals a golden
fingerprint on every SIMD path and thread count, the traced logits equal the untraced ones, the lens of the last
layer equals the model's prediction, and the HTML and SVG views are byte-identical across runs. Rankings break ties
on the lower token id (a total order), and the HTML and SVG writers use fixed number formats.
