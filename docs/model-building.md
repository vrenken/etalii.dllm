# Building models

EtAlii.Dllm imports open-weight models, and since Phase 3 it can fine-tune and edit them. Phase 19 adds three more
ways to make a model:

- `dllm merge` merges several models;
- `dllm export` writes a model back out as Hugging Face safetensors or as GGUF;
- `dllm distill` fine-tunes a student on a teacher's answers.

Each is exact and leaves a record:

- the same inputs give the same file, byte for byte, on every platform;
- the result's [lineage](provenance.md#lineage) says how it was made;
- for distillation, a [training receipt](provenance.md#training-receipts) replays the whole run.

## Merging

```bash
dllm merge a.dllm b.dllm -o merged.dllm                                   # the average
dllm merge a.dllm b.dllm -o merged.dllm --weights 3,1                     # 3/4 of a, 1/4 of b
dllm merge a.dllm b.dllm -o merged.dllm --method slerp --t 0.3
dllm merge tuned-1.dllm tuned-2.dllm -o merged.dllm --method ties --base base.dllm --density 0.2
dllm inspect merged.dllm      # "merged: slerp of 2e4d93db18c1ce20, 7c03e1a9d2b84f10" and the lineage
```

The models must have the same architecture and the same tokenizer. The merged file keeps the first model's chat
template and source.

| Method | What it does |
| --- | --- |
| `linear` | The weighted average of the models. The weights are normalised to add up to 1, and are equal by default. |
| `slerp` | Spherical interpolation between two models at `t` (0 gives the first, 1 the second), tensor by tensor. Two tensors that point almost the same way (cosine above 0.9995) are interpolated linearly instead. |
| `ties` | [TIES merging](https://arxiv.org/abs/2306.01708) of models fine-tuned from `--base`. Each model's difference from the base is first trimmed to its `--density` largest magnitudes. A sign is then elected per value from the weighted sum, and the differences that agree with it are averaged using their weights. |

How a merge stays exact:

- The arithmetic runs in float64, in a fixed model order, and is rounded to float32 once.
- SLERP's norms and dot products use the fixed-order C++ kernels, and its angles use the portable `acos` and `sin`
  ([kernels](kernels.md)).
- TIES breaks ties in magnitude by position, so its trimming is a total order.

So merging a model with itself gives that model's own fingerprint, and SLERP at `t = 0` or `t = 1` gives an input
exactly.

The merged file records its history:

- a `merge` section with the method, its settings and every input's fingerprint and full lineage (for TIES, the
  base's too);
- the first input's lineage, followed by a `merge` step.

The licence section joins every input's licence and attribution, and the result is only marked redistributable when
every input is.

## Exporting

```bash
dllm export tuned.dllm --format safetensors -o tuned-hf/   # config.json, tokenizer, LICENSE, model.safetensors
dllm export tuned.dllm --format gguf -o tuned.gguf         # for llama.cpp, Ollama, LM Studio
```

Fine-tuned, edited, merged and distilled models only exist as `model.dllm` files. `dllm export` lets other runtimes
load exactly these weights, in float32. Both writers are deterministic, so an export is byte-identical on every
platform. `dllm export` prints each file's SHA-256.

**Safetensors** writes a Hugging Face model directory. Llama, Mistral, Qwen2, Qwen3, OLMo 2 and Granite models are
supported. Before anything is written, the `config.json` is checked by importing it again: a model that cannot be
described exactly is refused rather than exported approximately. Importing the directory again gives the same
weights fingerprint and the same model description.

**GGUF** writes one GGUF v3 file the way llama.cpp's converter does: Llama's Q/K rows are permuted, and the vocabulary
is padded to the embedding size. Llama, Qwen2 and Qwen3 models with byte-level BPE tokenizers are supported. The
`tokenizer.ggml.pre` value is the one whose description matches the original tokenizer, and the export also checks
that the GGUF tokenizer encodes a set of probe texts exactly as the original does. Importing the file again gives the
same weights fingerprint. Two values can come back changed, as with any GGUF:

- `rms_norm_eps` is stored as float32;
- only two end-of-sequence ids are kept (`eos` and `eot`).

## Distilling

```bash
dllm distill student.dllm --teacher teacher.dllm --prompts prompts.txt -o distilled.dllm \
    --steps 200 --teacher-max-tokens 256 --receipt distilled.train.json
dllm replay distilled.train.json --base student.dllm --teacher teacher.dllm
```

`dllm distill` is `dllm finetune` with `--teacher` and `--prompts`, and takes all of its options (LoRA, AdamW,
checkpoints).

`prompts.txt` holds one prompt per line: a user message, or a JSON object with `messages` for the teacher to
continue.

1. The teacher answers every prompt greedily, each answer with its own [receipt](receipts.md).
2. The answers are written as chat records to `--data` (default `OUTPUT.distill.jsonl`), as canonical JSON lines.
3. The student is fine-tuned on them.

The run records where its data came from, in the model's `fine_tuning.distillation` section and in the training
receipt:

- the teacher's fingerprint;
- the prompts file and its SHA-256;
- the token budget;
- every answer's receipt id.

The lineage's `fine_tune` step names the teacher too.

`dllm replay` of the training receipt with `--teacher` first regenerates the teacher's answers and checks they are
byte for byte the data the run used, then trains again and checks every loss and the final weights. Without
`--teacher` it only checks the training, and says so.
