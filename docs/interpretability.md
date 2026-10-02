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
| Expert routing | `dllm experts` | `etalii_dllm.interpret.routing` | ✅ |
| Embedding explorer and word clouds | `dllm neighbours` | `etalii_dllm.interpret.neighbours` | ✅ |
| Steering vectors | `dllm steer`, `--steer` | `etalii_dllm.interpret.steering` | ✅ |
| Model editing (ROME) | `dllm edit` | `etalii_dllm.interpret.editing` | ✅ |
| Sparse autoencoders | `dllm sae` | `etalii_dllm.interpret.sae` | ✅ |

The commands take the usual `--model` option (or `DLLM_MODEL`) before the command name and run on the CPU
(`dllm edit` takes the model file as its argument, like `dllm finetune`).

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

## Expert routing

Mixture-of-experts models (Mixtral, OLMoE, Qwen3-MoE) send every token to a few of each layer's experts.

```bash
dllm --model olmoe.dllm experts --prompt "The cat sat on the mat"
dllm --model olmoe.dllm experts --prompt "The cat sat on the mat" --layer 3 --json
```

For every mixture-of-experts layer the text output lists each token's experts in rank order with their weights,
then how many tokens each expert received. `--layer` (1-based) picks one layer, `--json` prints the experts, weights
and usage per layer. In Python, `routing(model, tokens)` returns the same as arrays, and `trace(...)` carries
`experts` and `expert_weights` for every layer (-1 and 0 in dense layers; mixture-of-experts traces have no
`mlp_activation`). A `LayerHook` sees each sparse layer's routing in `routing(layer, experts, weights)`.

The routing is part of the exactly specified forward pass, so it is the same on every run, thread count and machine
and does not depend on what else is in the batch. In engines that group tokens by expert in batch-dependent ways, a
token's experts and outputs can change with the load. Mixture-of-experts models can be [edited](#model-editing-rome)
and [fine-tuned](training.md#mixtures-of-experts) too.

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

## Steering vectors

Activation steering adds a direction to the residual stream while the model runs. `dllm steer` finds one by
contrast: the mean residual stream after a layer over prompts that show a behaviour, minus the mean over prompts that
show its opposite.

```bash
dllm --model smollm2-135m.dllm steer \
    --positive "I love this. It is wonderful, beautiful and joyful." \
    --positive "What a delightful, happy day full of love." \
    --negative "I hate this. It is terrible, ugly and miserable." \
    --negative "What an awful, sad day full of hate." -o love.json
dllm --model smollm2-135m.dllm --steer love.json generate --prompt "I think that you are" --max-tokens 25
dllm --model smollm2-135m.dllm --steer love.json --steer-strength 8 chat "Describe your morning."
```

`--layer` picks the layer (1-based, default a third of the way in), `--positive-file`/`--negative-file` read more
prompts (one per line) and `--strength` sets the default multiplier stored in the file. `--steer FILE` and
`--steer-strength S` (or `DLLM_STEER` and `DLLM_STEER_STRENGTH`) work on every front end: the CLI, the HTTP server,
MCP and Docker. The vector is added after its layer at every position, prompt and answer alike, as one float32
multiply and add per element, so a steered model keeps the KV cache, prompt caching and batching bit-exact. It is a
different model, though: its `system_fingerprint` includes the vector's bits.

The file is JSON (`"format": "dllm-steering"`) with the layer, strength, the prompts it was built from, the
fingerprint of the model and the vector, each float32 value written as its exact double so loading changes nothing.

## Model editing (ROME)

`dllm edit` changes one fact with a rank-one update of one MLP, following
[ROME](https://rome.baulab.info/) (Meng et al., 2022), and writes a new model file:

```bash
dllm edit smollm2-135m.dllm --prompt "The Eiffel Tower is located in the city of" \
    --subject "Eiffel Tower" --target " Rome" -o smollm2-rome.dllm
dllm --model smollm2-rome.dllm generate --prompt "The Eiffel Tower is located in the city of" --max-tokens 6
dllm inspect smollm2-rome.dllm     # lists the edit
```

```text
edited:             layer 7, 10 steps
p(target):          0.0002 -> 0.9517
```

On SmolLM2-135M the edited model continues with " Rome, Italy" while "The Louvre is located in the city of" and "The
capital of France is" still give Paris. How it works:

1. The key `k` is the input of layer `l`'s MLP down projection (`act(gate) * up`) at the subject's last token.
   `--context PREFIX` (repeatable) averages it over the prompt with prefixes.
2. A change `delta` of the MLP's output there is found by AdamW on the target's cross-entropy (plus a small L2
   penalty), with gradients from the fine-tuning kernels; it stops at a fixed loss or after `--steps`.
3. The down projection becomes `W + delta u^T / (u . k)` with `u = C^-1 k`, where `C` is the covariance of keys over a
   corpus (a built-in set of plain sentences, or `--corpus FILE`), normalised and regularised
   (`--regularisation`, default 0.1), and solved by a Cholesky kernel. The edited projection maps `k` to its old
   value plus `delta`; keys unlike `k` change as little as possible.

The output file records the edit in its `edits` list (prompt, subject, target, layer, base fingerprint, optimiser
and covariance settings, the target probability before and after) and notes the modification in its licence
attribution. Edits stack: editing an edited file appends to the list. Equal edits of equal files write byte-identical
files. Editing uses the gradient support of fine-tuning, so it works for every family the engine runs. `--layer`
picks the MLP (1-based, default a quarter of the way in); ROME's authors found early-middle layers work best for facts.

In a mixture-of-experts layer (Mixtral, OLMoE, Qwen3-MoE) the edit changes one expert: the one the subject's last
token is routed to with the largest weight `r` (ties to the lower index). The key is that expert's activation, the
covariance comes from that expert's activations at every position of the corpus, and the update writes `delta / r`,
because the layer adds `r` times the expert's output. `dllm edit` prints the expert and its weight
(`edited: layer 2, expert 5 (weight 0.6214), 12 steps`), and the edit record stores both (`expert`,
`routing_weight`), so they are part of the edit's digest in the model's lineage. The routing itself is not changed,
so the edit takes effect where the token still goes to that expert.

## Sparse autoencoders

A sparse autoencoder (SAE) rewrites the residual stream after one layer as a sparse, non-negative mix of learned
directions ("features"), which are often far easier to name than single neurons:

```bash
dllm --model smollm2-135m.dllm sae train --corpus my-text.txt --layer 15 --features 1024 --steps 600 \
    --batch-size 128 --learning-rate 2e-3 -o smollm2-l15.sae
dllm --model smollm2-135m.dllm sae features smollm2-l15.sae --corpus my-text.txt --count 10 --top-k 4
dllm --model smollm2-135m.dllm sae steer smollm2-l15.sae --feature 783 -o macos.json   # a feature as a steering vector
```

```text
feature 783: active on 2.3%, max 13.432
    13.432  '4, Windows, Linux and' [' mac'] 'OS,'
    13.335  ' and across Linux, Windows and' [' mac'] 'OS on'
feature 73: active on 2.1%, max 13.650
    13.650  ' a small cached model (Sm' ['ol'] 'LM2'
    13.240  ' The recommended first model is Sm' ['ol'] 'LM2'
```

The corpus is a text file with one passage per line. Training collects the residual stream at every position,
leaves out the few rows whose squared norm is more than ten times the median (the "attention sink" on the first
tokens, which would otherwise take over every feature), scales the rest so their mean squared norm is the hidden
size, and minimises `mean ||x - x_hat||^2 + l1 * mean sum f` (`--l1`, default 5) with AdamW, keeping the decoder's
columns at unit length. Batches follow a seeded shuffle (`--seed`) and the decoder starts from the seeded Gaussian
generator, so equal runs write byte-identical files (safetensors with the settings in the metadata). `sae features`
lists the features with the largest activations (or the ones given with `--feature`), how often each is active and
the contexts that activate it most; `--json` prints the same as JSON.

## Reproducibility

Everything above is covered by `tests/test_interpret.py`, `tests/test_editing.py` and `tests/test_sae.py`: a trace of each tiny model family equals a golden
fingerprint on every SIMD path and thread count, the traced logits equal the untraced ones, the lens of the last
layer equals the model's prediction, the HTML and SVG views are byte-identical across runs, steered models keep the KV cache and batch invariance, the
residual and SAE gradients match finite differences, and repeated edits and SAE runs write byte-identical files. Rankings break ties
on the lower token id (a total order), and the HTML and SVG writers use fixed number formats.
