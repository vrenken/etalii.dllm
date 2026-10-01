# Fine-tuning

`dllm finetune` continues training an imported model on your own text with AdamW, and does it reproducibly: the same
base model, data file and settings give a byte-identical fine-tuned `model.dllm` on every run, and a run stopped at a
checkpoint and resumed ends in exactly the same bytes as one that ran straight through. The code is in
`src/etalii_dllm/training/`; the gradient kernels are in `cpp/include/dllm/grad.hpp`.

This is roadmap Phase 3, with LoRA added in Phase 8. It trains either every parameter of the Llama/Qwen2/Qwen3
decoder or [LoRA adapters](#lora-adapters) on its linear layers; there is no pre-training from scratch (weights come
from [importing open models](research/model-import.md)).

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

## What makes it reproducible

- **Gradients** come from C++ backward kernels with the same rules as the forward kernels (`docs/kernels.md`):
  every element has one double accumulator summed in a fixed order. Sums over positions (weight gradients, keys and
  values, embedding rows) visit positions in ascending order. The training forward pass is the decoder's own, so
  its last-position logits equal `Transformer.forward` bit for bit (tested).
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
short fine-tuning run.

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

The header holds the architecture, run settings, step, loss history, base model and data fingerprints, the base
model's metadata (so the checkpoint alone can be exported) and a SHA-256 of the data section that is checked on
load. A checkpoint refuses to resume with data whose fingerprint differs.

## Cost

Training runs on the same single-threaded kernels as inference, and one training token costs roughly three times
what reading one prompt token costs. That is fine for small experiments and tests, but fine-tuning SmolLM2-135M on
more than a few thousand tokens will take hours until the Phase 6 performance work lands. Memory is about four
times the model size (weights, gradients and two AdamW moments) plus the activations of one window.
