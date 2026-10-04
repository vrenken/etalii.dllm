# Fine-tuning

`dllm finetune` continues training an imported model on your own text with AdamW, and does it reproducibly: the same
base model, data file and settings give a byte-identical fine-tuned `model.dllm` on every run, and a run stopped at a
checkpoint and resumed ends in exactly the same bytes as one that ran straight through. The code is in
`src/etalii_dllm/training/`; the gradient kernels are in `cpp/include/dllm/grad.hpp`.

This is roadmap Phase 3, with LoRA added in Phase 8, every dense model family in Phase 40 and mixtures of experts in
Phase 43. It trains either every parameter of the decoder or [LoRA adapters](#lora-adapters) on its linear layers, for
every architecture the engine runs: Llama, Mistral, Qwen2, Qwen3, OLMo 2, Granite, Gemma 2, Gemma 3, Phi-3/Phi-4-mini
(with LongRoPE) and the [mixture-of-experts](#mixtures-of-experts) models Mixtral, OLMoE, Qwen3-MoE, Qwen2-MoE and
Granite MoE. Since Phase 58 it also fine-tunes [encoders](#encoders): BERT, RoBERTa, XLM-RoBERTa, ModernBERT and
DeBERTa embedders and cross-encoders, and T5 embedders. There is no pre-training from scratch (weights come from
[importing open models](research/model-import.md)).

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
| `--sequence-length` | 128 | Tokens per window (at most the model's context length); for encoders, per text (default: the model's own limit) |
| `--objective` | follows the model | `lm` (decoders), `dpo`, `embedding` (embedders) or `classifier` (cross-encoders) |
| `--similarity-scale` | 20 | Embedding runs: the factor on cosine similarities (sentence-transformers' `scale`) |
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

## Encoders

Embedders (all-MiniLM, bge, all-distilroberta, multilingual MiniLM, ModernBERT, DeBERTa, sentence-t5, GTR-T5) and cross-encoders
(ms-marco-MiniLM, mmarco, gte-reranker-modernbert, mxbai-rerank, nli-deberta-v3) fine-tune with the same command, the same options, checkpoints, receipts and LoRA. The objective follows the model:

```bash
# an embedder: anchors and the texts that belong with them (and optionally a hard negative)
dllm finetune all-minilm.dllm --data pairs.jsonl -o all-minilm-tuned.dllm --steps 100 --batch-size 16
# a cross-encoder: texts or pairs with a label
dllm finetune ms-marco.dllm --data labels.jsonl -o ms-marco-tuned.dllm --steps 100 --lora-rank 8
```

```jsonl
{"anchor": "how do I reset my password", "positive": "Open Settings, then Account, then Reset password."}
{"anchor": "capital of France", "positive": "Paris is the capital of France.", "negative": "Lyon is in France."}
```

```jsonl
{"text": "how do I reset my password", "pair": "Open Settings, then Account, then Reset password.", "label": 1}
{"text": "how do I reset my password", "pair": "Paris is the capital of France.", "label": 0}
```

- **Embedding** (`--objective embedding`) is sentence-transformers' `MultipleNegativesRankingLoss`: each of a step's
  `batch-size` anchors is scored against every positive of the step, then every hard negative, by
  `similarity_scale` times their cosine similarity, and the loss is the mean cross-entropy of picking the anchor's
  own positive. The other examples of the step are its negatives, so use a batch size of at least 2. Sentence
  vectors are pooled as the model pools them (mean, CLS or last token) and L2-normalised; the scores are one
  `linear` kernel of the normalised vectors.
- **Classification** (`--objective classifier`) is what sentence-transformers' `CrossEncoder` trains with: binary
  cross-entropy with logits for a model with one output (the label a number from 0 to 1, a reranker's relevance),
  else softmax cross-entropy over the labels (the label an index). The loss is the mean over the step's examples.
- **Texts** are tokenized with the model's own recipe (its default prompt and special tokens, or the pair template)
  and cut to `--sequence-length` (by default the model's limit) as sentence-transformers cuts them.
- **The backward pass** (`training/encoder_backprop.py`) is the encoder's own forward pass, kernel for kernel, then
  `layer_norm_backward`, `attention_backward` without the causal mask, `gelu_backward` (the exact erf GELU),
  `linear_backward` with biases and `embedding_backward` for the word, token-type and position tables. As in
  `transformers`, whose embeddings have a `padding_idx`, RoBERTa's padding row gets no gradient. Each text's
  gradients are computed on their own and summed elementwise in batch order (anchor, positive, negative).
  ModernBERT's backward pass adds the rotary backward (`rope` with the inverse rotation, global or local base),
  `attention_backward` with the bidirectional local window and the gated MLP; its norms have no bias. Its padding
  row is not frozen (inputs are never padded, so it only moves through weight decay).
- **LoRA** adapts the attention projections and the MLP (`q k v o up down`; `gate` is skipped) under
  `transformers`' names, `encoder.layer.N.attention.self.query` and so on, with `bert.` or `roberta.` in front for a
  cross-encoder. PEFT's `target_modules` is a regular expression over those names, so the pooler and the classifier
  (both `dense` layers too) stay frozen, as in a LoRA run here. `--adapter` and `dllm import ADAPTER --base` apply an
  encoder adapter like a decoder one. ModernBERT's modules are fused, so its adapters are too: `attn.Wqkv`,
  `attn.Wo`, `mlp.Wi` and `mlp.Wo` (`layers.N.attn.Wqkv`, with `model.` in front for a cross-encoder). Inside, q, k
  and v share one `lora_A` (stored with q) and each has its own rows of `lora_B`, as do the MLP's gate and up halves,
  so the merged delta is exactly PEFT's `B @ A` on the fused weight; the shared `A`'s gradient sums its parts' in
  tensor order. q, k and v (and gate and up) are adapted together or not at all.
- **Outputs** keep the model's pooling or classifier settings, so the tuned model serves embeddings, reranking and
  `dllm embed` as before; the receipt counts `"examples"`. [`dllm export`](model-building.md#encoders) writes it back
  to Hugging Face and sentence-transformers, or to GGUF.

DeBERTa encoders (Phase 61) fine-tune with both objectives. Their backward pass is BERT's with
`biased_attention_backward` in place of `attention_backward`: besides `dq`, `dk` and `dv` it returns the gradient of
the score bias, `p_j (dp_j - D) * scale` rounded once to float32 (the same value `dq` and `dk` are built from).
Each head's bias is a gather of two products, `c2p[i, d(i, j)] = q_i . pk[d(i, j)]` and `p2c[j, d(i, j)] = k_j .
pq[d(i, j)]`, so its gradient scatters back with `embedding_backward` (every pair's value summed into its row in
(i, j) order, in double) and `linear_backward` takes it on to the queries and keys (added after the attention's own
gradients) and to `pq` and `pk`. Those are the layer's own query and key projections of the relative table, so their
weight and bias gradients are added to the content path's, and the table rows' gradients from every layer go through
`layer_norm_backward` into `relative_norm` and `relative_embedding` (rows no pair reads get zero). The cross-encoder
head is the context pooler, `classifier(gelu(pooler(h[0])))`. LoRA adapts the same modules as for BERT under
DeBERTa's names (`attention.self.query_proj`, `key_proj`, `value_proj`, `attention.output.dense`,
`intermediate.dense`, `output.dense`; `deberta.` in front for a cross-encoder); the merged query and key weights also
project the relative table, exactly as PEFT's adapted modules do.

T5 encoders (Phase 63) train with the embedding objective (they have no classification head). Their backward pass
runs `rms_norm_backward` for the bias-free RMS norms, `biased_attention_backward` with scale 1 (T5 does not scale
its scores), and the MLP's activation gradient: ReLU passes the gradient where its input is positive (+0 elsewhere,
as torch does), and T5 v1.1's gated MLP differentiates `act(gate) * up` as ModernBERT's does. Every layer reads the
same bucket table, so the score bias gradient of every layer (last layer first, added in float32) scatters back into
it with `embedding_backward`: each (head, i, j) value is summed, in (head, i, j) row-major order and in double, into
row `t5_bucket(j - i)` of that head's column. LoRA adapts T5's own modules (`SelfAttention.q`, `k`, `v`, `o`,
`DenseReluDense.wi`, or `wi_0` and `wi_1` when the MLP is gated, and `wo`, under `encoder.block.N.layer.M.`).

An embedder with a sentence-transformers `Dense` module after the pooling (sentence-t5, GTR-T5, LaBSE; any encoder
family) trains it with the rest of the model: the training sentence vector is pooled, projected (`linear`, then
`tanh` as `softcap(x, 1)` when the module has it) and normalised exactly as `engine.embed` computes it, and the
gradient goes back through `softcap_backward` and `linear_backward` to the projection's weight and bias. LoRA leaves
the projection frozen.

`tests/test_encoder_training.py` checks the new kernels and the encoder's gradients against `transformers`' autograd
and finite differences (BERT and XLM-RoBERTa with padding inside the sequence), both losses and their gradients
against the `torch` formulas sentence-transformers uses, bit-exact resumption, receipts, golden hashes of a short run
of each objective, and LoRA adapters whose names are `transformers`' own modules. `tests/test_modernbert.py` does the
same for ModernBERT, its fused adapters included, `tests/test_deberta.py` for DeBERTa (the bias gradient against
finite differences too), and `tests/test_t5.py` for T5 (ReLU, SiLU, GELU, gated ReLU and gated tanh GELU MLPs) and
for the `Dense` projection, checked through sentence-transformers' whole module chain on T5 and on BERT.

## Text-to-text models

T5 and Flan-T5 text-to-text models (Phase 65) fine-tune on pairs of a source text and the answer it should get,
with the same command, options, checkpoints, receipts and LoRA as the other families:

```bash
dllm finetune flan-t5-small.dllm --data pairs.jsonl -o flan-tuned.dllm --steps 100 --batch-size 8 --lora-rank 8
```

```jsonl
{"input": "translate English to German: How old are you?", "target": "Wie alt bist du?"}
{"prompt": "summarize: The meeting moved to Thursday at ten.", "completion": "Meeting: Thursday 10:00."}
{"messages": [{"role": "user", "content": "Is the sky blue?"}, {"role": "assistant", "content": "Yes."}]}
```

- **Examples.** `input`/`target`, `prompt`/`completion`, or a conversation whose last message is the assistant's
  answer (what `dllm finetune --teacher` writes, so distillation into a text-to-text model works too). The source of
  a conversation is the other messages' contents joined by a blank line, as the engine renders a chat for T5. Each
  text is cut to `--sequence-length - 1` tokens and ended with `</s>`. The receipt counts `"examples"`.
- **The loss** is teacher forcing, as `transformers` computes it with `labels=target`: the decoder reads
  `[<pad>, *target[:-1]]` and the loss is the mean cross-entropy of every target token (`</s>` included) over the
  step's examples. Only the language-model objective applies; `--dpo` is refused.
- **The backward pass** (`training/seq2seq_backprop.py`) runs the whole target at once. Its self-attention is
  `biased_attention` with the decoder's one-directional bucket bias and `-inf` for later keys, whose exponentials are
  exactly 0, so every row equals the served model's incremental step bit for bit. The gradients flow back through
  the LM head (with a tied head, times the forward's `d_model ** -0.5`), the decoder's layers (MLP, then
  cross-attention, whose key and value gradients go through each layer's `k` and `v` projections into the encoder
  states, added in float32 last layer first, then self-attention, whose bias gradient scatters into the decoder's
  bucket table), and then the encoder's T5 backward pass. The word embedding adds its three gradients in float32 in
  a fixed order: the tied head's, the decoder inputs', then the encoder's.
- **LoRA** adapts `q k v o up down` (and `gate` for a gated MLP) in the encoder and the decoder; q, k, v and o adapt
  both the self-attention and the cross-attention, as PEFT's T5 targets (`q`, `v`) do. The PEFT file uses
  `T5ForConditionalGeneration`'s names (`encoder.block.N.layer.0.SelfAttention.q`,
  `decoder.block.N.layer.1.EncDecAttention.k`, `decoder.block.N.layer.2.DenseReluDense.wi_0`) with task type
  `SEQ_2_SEQ_LM`; `--adapter` and `dllm import ADAPTER --base` merge it.

`tests/test_t5_text_finetuning.py` checks that the training logits equal the served bits, the gradients against
`transformers`' autograd for T5 v1.0 (tied head, ReLU) and Flan-T5 (gated tanh GELU, own head), golden runs with
bit-exact resumption, `dllm finetune` with receipts and `dllm replay`, and LoRA adapters whose names are
`transformers`' own modules.

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
