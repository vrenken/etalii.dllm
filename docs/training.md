# Fine-tuning

`dllm finetune` continues training an imported model on your own text with AdamW, and does it reproducibly: the same
base model, data file and settings give a byte-identical fine-tuned `model.dllm` on every run, and a run stopped at a
checkpoint and resumed ends in exactly the same bytes as one that ran straight through. The code is in
`src/etalii_dllm/training/`; the gradient kernels are in `cpp/include/dllm/grad.hpp`.

This is roadmap Phase 3, with LoRA added in Phase 8, every dense model family in Phase 40 and mixtures of experts in
Phase 43. It trains either every parameter of the decoder or [LoRA adapters](#lora-adapters) on its linear layers, for
every architecture the engine runs: Llama, Mistral, Qwen2, Qwen3, OLMo 2, Granite, Gemma 2, Gemma 3, Phi-3/Phi-4-mini
(with LongRoPE) and the [mixture-of-experts](#mixtures-of-experts) models Mixtral, OLMoE, Qwen3-MoE, Qwen2-MoE and
Granite MoE. There is
no pre-training from scratch (weights come from [importing open models](research/model-import.md)).

## Usage

```bash
dllm finetune smollm2-135m.dllm --data my-data.jsonl -o smollm2-135m-tuned.dllm \
    --steps 200 --batch-size 8 --sequence-length 128 --learning-rate 1e-4 --warmup-steps 20 --seed 1 \
    --checkpoint run.dllmckpt --checkpoint-every 50
dllm inspect smollm2-135m-tuned.dllm          # shows base model, data fingerprint, steps and final loss
dllm --model smollm2-135m-tuned.dllm chat "..."

# Continue an interrupted run (settings come from the checkpoint; pass the same data file)
dllm finetune smollm2-135m.dllm --data my-data.jsonl -o smollm2-135m-tuned.dllm --resume run.dllmckpt
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--steps` | 100 | Optimizer steps |
| `--batch-size` | 8 | Windows per step |
| `--sequence-length` | 128 | Tokens per window (at most the model's context length) |
| `--learning-rate`, `--min-learning-rate` | 1e-4, 0 | Peak and final learning rate |
| `--warmup-steps` | 0 | Linear warmup from 0 |
| `--schedule` | `cosine` | `cosine` (decay to the minimum at the last step) or `constant` after warmup |
| `--weight-decay` | 0.01 | Decoupled (AdamW) decay on weight matrices; norms and biases are not decayed |
| `--max-grad-norm` | 1.0 | Global gradient-norm clipping, 0 disables |
| `--seed` | 0 | Seeds the data order (there is no other randomness: no dropout) |
| `--router-aux-loss` | 0 | Mixture-of-experts models: coefficient of the router load-balancing loss |

Adam's betas are 0.9 and 0.999 and epsilon is 1e-8 (the PyTorch defaults).

## LoRA adapters

`--lora-rank r` trains low-rank adapters instead of the weights: each adapted linear layer `W` (`[out, in]`) gets
`A` (`[r, in]`) and `B` (`[out, r]`) and becomes `W + (alpha / r) * B @ A`. The base weights stay frozen, and only
the adapters have AdamW moments, so checkpoints and adapter files are small.

```bash
dllm finetune smollm2-135m.dllm --data my-data.jsonl --lora-rank 8 --lora-alpha 16 \
    --adapter-output my-adapter -o smollm2-135m-lora.dllm --steps 200 --learning-rate 1e-3
dllm --model smollm2-135m.dllm --adapter my-adapter chat "..."      # apply the adapter when the model loads
dllm --model smollm2-135m-lora.dllm chat "..."                     # the same answer from the merged file
dllm import my-adapter --base smollm2-135m.dllm -o merged.dllm     # merge it later (same bytes as -o above)
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--lora-rank` | 0 (off) | Adapter rank |
| `--lora-alpha` | the rank | Scale numerator; the adapter is scaled by `alpha / rank` |
| `--lora-targets` | `q,k,v,o,gate,up,down` | Adapted linear layers |
| `--adapter-output` | none | Write the adapters as a PEFT directory |

`-o` writes the base model with the adapters merged in; `--adapter-output` writes a Hugging Face
[PEFT](https://github.com/huggingface/peft) adapter (`adapter_config.json`, `adapter_model.safetensors`), which
PEFT loads as it is. Adapters trained with PEFT import the same way (`dllm import DIR --base BASE`, also
`hf:org/adapter`); DoRA, per-layer ranks, bias training and `modules_to_save` are refused. Resuming a LoRA
checkpoint needs the base model again (it is the positional argument, as when the run started).

Determinism: an adapter is only ever used by merging it into the weights (`W + scale * B @ A` with the `linear`
kernel, then elementwise float32), whether that happens in a file or when the model loads. Both routes give the same
weights, the same `system_fingerprint` and the same output bits. Training runs the decoder on the merged weights and
gets the adapter gradients from the merged weight's gradient (`dA = scale * B^T dW`, `dB = scale * dW A^T`). That
keeps training exactly consistent with inference, at the price of the full backward pass: a LoRA step costs about
as much compute as a full fine-tuning step, but needs far less optimizer memory. New adapters start with `A`
Gaussian (standard deviation `1 / rank`) from `--seed` and `B` zero, so step 0 is the base model exactly.
`tests/test_lora.py` checks the adapter gradients against finite differences, byte-identical and resumed LoRA runs,
and that merged and load-time adapters give the same bits; `tests/test_reference_models.py` checks both PEFT
directions against the `peft` library.

### Quantised bases

`--base-quantize q8_0` or `q4_0` keeps the frozen base quantised in memory while LoRA trains, as QLoRA does:

```bash
dllm finetune qwen2.5-1.5b.dllm --data my-data.jsonl --lora-rank 8 --base-quantize q4_0 \
    --adapter-output my-adapter -o qwen-qlora.dllm --steps 200 --learning-rate 1e-3
dllm import my-adapter --base qwen2.5-1.5b.dllm --base-quantize q4_0 -o merged.dllm   # the same bytes as -o
```

- **What is quantised.** The attention and MLP matrices (every expert's and the shared expert's too) whose input
  size is a multiple of 32 are held as Q8_0 (about 28% of their float32 size) or Q4_0 (about 16%); embeddings, the
  LM head, norms, biases, routers and gates stay float32. `dllm finetune` prints what the matrices take.
- **What the base is.** The run is defined as LoRA on the dequantised base: each block's integers times its float32
  scale (`QuantizedWeight.dequantize`, elementwise). A matrix is dequantised, and the adapters merged into it, each
  time a step reads it, so only the weights in use exist in float32. The bits are exactly those of a plain LoRA run
  on the dequantised weights, on every machine (`tests/test_quantized_finetuning.py` checks it for every model
  family the tests cover, with DPO too). The decoder's own `--quantize` also quantises the activations, so a model
  served with `--quantize` is not this base; serve the merged file instead, which holds the dequantised base with
  the adapters merged in.
- **Records.** `base_quantize` is part of the run: in `fine_tuning.run`, the lineage step, checkpoints (which resume
  only as the same run) and training receipts (`dllm replay` trains the quantised run again). `dllm import ADAPTER
  --base BASE --base-quantize KIND` merges an adapter into the dequantised base and records the quantisation in the
  file's `adapter` section and lineage, so the merged file can be rebuilt from the adapter alone.
- **Cost.** Dequantising and merging on every read costs time: a step is two to three times slower than a plain
  LoRA step on the tiny test models, in exchange for a quarter or an eighth of the base's matrix memory.

## Data

- A `.txt` file is one document.
- A `.jsonl` file has one JSON object per line: `{"text": "..."}`, or `{"messages": [{"role": "user", "content":
  "..."}, {"role": "assistant", "content": "..."}]}`, which is rendered with the model's own chat template.

Documents are tokenized with the model's tokenizer, joined in file order with the model's end-of-sequence token after
each, and cut into windows of `sequence-length + 1` tokens that overlap by one token. Every token of a window is
trained on (chat records included: the loss covers the user turns too). Each epoch visits the windows in a
permutation drawn with the project's own random generator from `--seed` and the epoch number; step `s` reads
samples `s * batch_size ..` of that epoch-after-epoch sequence. Because the data read at a step follows from the
step number alone, resuming needs no data-loader state.

## Preference tuning

`--dpo` trains on preference pairs instead of text, with direct preference optimization (DPO): the model learns to
make the chosen answer more likely and the rejected one less likely, relative to the model it started from.

```bash
dllm finetune smollm2-135m.dllm --dpo --data pairs.jsonl --beta 0.1 -o smollm2-135m-dpo.dllm \
    --steps 200 --batch-size 8 --sequence-length 256 --learning-rate 5e-6 --receipt dpo.json
dllm --model smollm2-135m-dpo.dllm eval pairs.jsonl     # preference accuracy and mean margin (docs/evaluation.md)
dllm replay dpo.json --base smollm2-135m.dllm           # trains again and checks every loss and the weights
```

The data file is JSON lines, one pair per line:

```json
{"prompt": "Q: What is 2+2?\nA:", "chosen": " 4", "rejected": " 5"}
{"messages": [{"role": "user", "content": "Hi"}], "chosen": "Hello! How can I help?", "rejected": "Go away."}
```

A `messages` prompt is rendered with the model's chat template and its generation prompt; a `prompt` is used as
written. The prompt and each answer are tokenized separately, and each answer gets the end-of-sequence token, so a
pair's tokens do not depend on the rest of the file. A side holds at most `sequence-length + 1` tokens: a longer
answer is cut at its end, and a prompt must leave room for at least one answer token. Only the answer tokens are
scored. Pairs are visited in the same seeded per-epoch order as windows; `--batch-size` counts pairs.

| Option | Default | Meaning |
| --- | --- | --- |
| `--dpo` | off | `--data` holds preference pairs; train with the DPO loss |
| `--beta` | 0.1 | How far the model may move from the reference: larger keeps it closer |

The loss of a pair is `-log sigmoid(z)` with `z = beta * ((log p(chosen) - ref_chosen) - (log p(rejected) -
ref_rejected))`, where the reference log-probabilities come from the base weights and are computed once, when the run
starts. Determinism:

- Log-probabilities are minus the summed cross-entropy of the answer tokens: the same fixed-order kernel and double
  accumulator as the language-model loss. The reference values are stored in checkpoints as exact hexadecimal
  doubles, so a resumed run uses the same ones rather than recomputing them from half-trained weights.
- `log sigmoid` and `sigmoid` use the portable `exp`/`log` kernels in double, without overflow for either sign. At
  step 0 the model is its own reference, so `z` is exactly 0 and the first loss is exactly `log 2` (tested).
- A pair's gradient is `w * grad CE(chosen) - w * grad CE(rejected)` with `w = beta * (1 - sigmoid(z)) / batch_size`
  computed in double and applied as one elementwise float32 multiply; pairs are added in batch order, chosen first.
  The step's loss is the mean over pairs, summed in batch order.
- LoRA works the same way (`--lora-rank` with `--dpo`); the reference is the frozen base.

The model file's `fine_tuning.run` and the training receipt record `"objective": "dpo"` and `beta`, the receipt
counts `"pairs"` instead of `"windows"`, and the lineage step carries `"objective": "dpo"`. Language-model runs leave
those fields out, so their files and receipts are byte for byte what they were. `tests/test_preference.py` checks
byte-identical runs, bit-exact resumption, the step formula, LoRA, receipts and golden hashes of a short DPO run.

## Mixtures of experts

Mixtral, OLMoE, Qwen3-MoE, Qwen2-MoE and Granite MoE fine-tune like every other family, with every option above (full fine-tuning, LoRA, DPO
and distillation):

```bash
dllm finetune olmoe.dllm --data my-data.jsonl --lora-rank 8 -o olmoe-lora.dllm --steps 50 --router-aux-loss 0.01
```

- **The backward pass** runs each expert on the rows routed to it, ascending, as the forward pass does, and adds a
  row's input gradients over its experts in increasing order, then the router's. A routing weight's gradient is the
  `dot` kernel of the row's output gradient and the expert's output; the `moe_route_backward` kernel takes it back
  through the renormalisation (when the model renormalises) and the softmax to the router logits, serially per row
  with double accumulators. The choice of the top `k` experts has no gradient, as in `transformers`. Experts that no
  token of a window reaches get zero gradients.
- **The load-balancing loss** of `transformers` (`load_balancing_loss_func`): `experts * sum_e f_e * P_e` over the
  rows of every sparse layer, where `f_e` is the share of chosen slots that went to expert `e` and `P_e` its mean
  router probability. `--router-aux-loss COEF` adds `COEF` times its mean over a batch's windows to the loss (each
  window's own, so a window's gradients still do not depend on the rest of its batch; with one window per batch this
  is exactly `transformers`' `loss + router_aux_loss_coef * aux_loss`). Only `P_e` has a gradient. The coefficient is
  recorded in `fine_tuning.run`, checkpoints and training receipts; runs without it keep their old bytes. DPO runs do
  not add it.
- **LoRA** adapts every expert's `gate`, `up` and `down` projections (the router is never adapted). PEFT adapters use
  each family's module names: `block_sparse_moe.experts.E.w1`/`w3`/`w2` for Mixtral, `mlp.experts.E.gate_proj`/
  `up_proj`/`down_proj` for OLMoE and Qwen3-MoE (the per-expert layout of `transformers` 4; adapters for the fused
  expert parameters of `transformers` 5 are not read).
- **Shared experts** (Phase 44) see every row. Their input gradient is added after the routed experts'; with a
  sigmoid gate `s` (Qwen2-MoE) the expert's output gradient is scaled by `s` and the gate's score gets
  `dot(dout, y) * s * (1 - s)`. LoRA adapts the shared expert's `gate`, `up` and `down` too (never its gate), written
  as `mlp.shared_expert.gate_proj`/`up_proj`/`down_proj`. Granite MoE keeps all its experts in one fused parameter,
  which PEFT cannot adapt expert by expert: LoRA runs train and merge as usual, but exporting their adapter in the
  PEFT format is refused unless they adapt only the attention projections.

## What makes it reproducible

- **Gradients** come from C++ backward kernels with the same rules as the forward kernels (`docs/kernels.md`):
  every element has one double accumulator summed in a fixed order. Sums over positions (weight gradients, keys and
  values, embedding rows) visit positions in ascending order. The training forward pass is the decoder's own, so
  its last-position logits equal `Transformer.forward` bit for bit (tested for every family).
- **Every family.** The backward pass follows each architecture's forward pass step by step: post-layer norms
  (OLMo 2) and sandwich norms (Gemma), QK-norm over the whole projection or per head, Gemma's norms scaled by
  `1 + weight`, the tanh GELU, attention and logit soft-caps, the scaled embedding, local RoPE bases and sliding
  windows. Constant scales that the decoder folds into weights (Granite's residual multiplier in the output
  projections, the LongRoPE attention factor in the query and key rows) are folded the same way in training, and
  their gradients are multiplied back by the same float32 constant, so the stored weights are what is trained.
- **Batches.** Each window's loss and gradients are computed on their own and summed elementwise in batch order, so
  a window's contribution does not depend on which windows share its batch. The loss is the mean next-token
  cross-entropy over all targets of the batch.
- **AdamW** runs element by element in double. The moments are rounded to float32 before they are used, so the
  state a checkpoint stores is exactly the state the next step reads. Tensors are updated in the natural order of
  their names; the global gradient norm adds each tensor's sum of squares in that order. Bias corrections use
  repeated multiplication rather than `pow`, and the cosine schedule uses the portable `dllm` cosine.
- **Files.** Checkpoints and exported models contain no timestamps, paths or environment data, only canonical JSON
  and aligned little-endian float32, so equal runs write equal bytes.

`tests/test_training.py` checks the gradient kernels against float64 references, the decoder's gradients against
float64 finite differences, byte-identical runs, bit-exact resumption, and golden hashes of the gradients and of a
short fine-tuning run, for a tiny synthetic model of each of the twelve families; `tests/test_lora.py` and
`tests/test_preference.py` do the same for LoRA and DPO, and `tests/test_moe_training.py` checks the routing
backward kernel, the load-balancing loss and its gradient, expert adapters and distillation.

## Outputs

**The fine-tuned model** is an ordinary `model.dllm` (see [the format](model-format.md)) with the base model's
tokenizer, chat template, source and licence, plus a `fine_tuning` section: base model fingerprint, data fingerprint
(SHA-256 of the token windows), all run settings, steps completed and the final loss. The licence attribution states
that the weights were modified, as Apache-2.0 asks. Its `system_fingerprint` is new, as the weights are. Its
`lineage` gains a `fine_tune` step.

**A training receipt** (`--receipt FILE`) records the base and data, every setting, every step's loss and the
resulting weights, so that `dllm replay FILE --base BASE` can train again and confirm the same weights, bit for bit
(see [verifiable models](provenance.md#training-receipts)).

**Checkpoints** (`.dllmckpt`) hold the whole training state:

| Offset | Size | Content |
| --- | --- | --- |
| 0 | 8 | Magic `DLLMCKPT` |
| 8 | 4 | Format version, uint32 little-endian (`1`) |
| 12 | 8 | Header length `H`, uint64 little-endian |
| 20 | `H` | Canonical JSON header, padded with spaces so the data starts at a multiple of 64 |
| 20 + `H` | rest | Tensors: `param/<name>`, then `adam_m/<name>` and `adam_v/<name>` per tensor, each 64-byte aligned float32 |

The header holds the architecture, run settings, step, loss history, base model and data fingerprints (and for DPO
runs the reference log-probabilities of every pair), the base model's metadata (so the checkpoint alone can be exported) and a SHA-256 of the data section that is checked on
load. A checkpoint refuses to resume with data whose fingerprint differs.

## Cost

Training runs on the same single-threaded kernels as inference, and one training token costs roughly three times
what reading one prompt token costs. That is fine for small experiments and tests, but fine-tuning SmolLM2-135M on
more than a few thousand tokens will take hours until the Phase 6 performance work lands. Memory is about four
times the model size (weights, gradients and two AdamW moments) plus the activations of one window. LoRA needs the
base weights plus small adapters and their moments, and `--base-quantize` shrinks the base's matrices to a quarter
(Q8_0) or an eighth (Q4_0) of that ([quantised bases](#quantised-bases)).
