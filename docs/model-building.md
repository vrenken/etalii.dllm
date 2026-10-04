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
is padded to the embedding size. Llama, Qwen2 and Qwen3 models with byte-level BPE or SentencePiece-style tokenizers
are supported. For byte-level BPE, the `tokenizer.ggml.pre` value is the one whose description matches the original
tokenizer. A SentencePiece-style tokenizer (TinyLlama, Llama 2) becomes a `llama` tokenizer: its pieces, scores from
its merge order (the first merge that makes a piece ranks it), token types and the space prefix
([SentencePiece models through GGUF](#sentencepiece-models-through-gguf)). Either way the export checks that the GGUF
tokenizer encodes a set of probe texts (and text around a special token) exactly as the original does, and refuses
the export otherwise. Importing the file again gives the
same weights fingerprint. Two values can come back changed, as with any GGUF:

- `rms_norm_eps` is stored as float32;
- only two end-of-sequence ids are kept (`eos` and `eot`).

### Encoders

BERT, RoBERTa and XLM-RoBERTa embedders and cross-encoders export too (Phase 58), for example after
[fine-tuning one](training.md#encoders):

```bash
dllm export all-minilm-tuned.dllm --format safetensors -o all-minilm-tuned/   # transformers + sentence-transformers
dllm export ms-marco-tuned.dllm --format gguf -o ms-marco-tuned.gguf          # llama.cpp's bert layout
```

- **Safetensors** writes `BertModel`, `RobertaModel` or `XLMRobertaModel` (or their `...ForSequenceClassification`
  with the labels and the CrossEncoder activation) under `transformers`' own tensor names, and for an embedder the
  sentence-transformers modules: `modules.json`, the pooling module, `Normalize` when the model normalises, the
  prompts and `max_seq_length`. `SentenceTransformer(dir)`, `CrossEncoder(dir)` and `AutoModel.from_pretrained(dir)`
  load it, and importing it again gives the same weights, settings and fingerprint.
- **GGUF** writes llama.cpp's `bert` layout (`token_embd`, `token_types`, `position_embd`, `token_embd_norm`,
  `blk.N.attn_q` ... `layer_output_norm`), the pooling type (mean, CLS, last token, or rank for a cross-encoder) and
  the WordPiece vocabulary in llama.cpp's phantom-space form. A cross-encoder's pooler and classifier become `cls`
  and `cls.output`, so llama.cpp computes the same head `cls.output(tanh(cls(h[0])))` (its own converter drops a
  BERT pooler). The file also carries `tokenizer.huggingface.json` and `dllm.*` keys with the exact tokenizer, the
  float64 LayerNorm epsilon and the pooling or classifier settings, which llama.cpp ignores: importing the file
  gives back the same model exactly. GGUF files of BERT models written by llama.cpp's converter import too; their
  tokenizer is the WordPiece vocabulary with BERT's lower-casing normaliser, as llama.cpp tokenizes. RoBERTa and
  XLM-RoBERTa cannot be written to GGUF exactly (llama.cpp's layout cuts the position rows before the padding
  token), nor can BERT models with the tanh GELU; export those to safetensors.
- **ModernBERT** (Phase 59) exports to safetensors only: `ModernBertModel` or `ModernBertForSequenceClassification`
  with the fused `attn.Wqkv` and `mlp.Wi` weights put back together, the global/local layer pattern as
  `global_attn_every_n_layers` (or `layer_types` when no period fits) and the classifier pooling, plus the
  sentence-transformers modules of an embedder. Before writing, the config is read back through the importer and
  must give the same model. llama.cpp's GGUF layout has no ModernBERT, so `--format gguf` refuses it.
- **DeBERTa** (Phase 61) exports to safetensors only: `DebertaV2Model` or `DebertaV2ForSequenceClassification` in
  DeBERTa-v3's layout (relative attention with shared position projections, `pos_att_type` `p2c|c2p`, `norm_rel_ebd`
  `layer_norm`, no absolute positions), the relative table as `encoder.rel_embeddings` and `encoder.LayerNorm`, the
  context pooler as `pooler.dense`, the special token ids from the tokenizer, plus the sentence-transformers modules of
  an embedder. As for the other encoders, the config is read back through the importer first and must give the same
  model. llama.cpp's GGUF layout has no DeBERTa, so `--format gguf` refuses it.
- **T5** encoders (Phase 63) export to safetensors only: `T5EncoderModel` (the word embedding as `shared`, which
  the encoder ties `embed_tokens` to, the bucket table in the first block's attention, the MLP as `wi`, or `wi_0` and
  `wi_1` when it is gated, and `wo`), `feed_forward_proj` `relu`, `gelu`, `gelu_new`, `silu` or `gated-gelu`, plus
  the sentence-transformers modules. transformers has no gated MLP with another activation, so such a model is
  refused; llama.cpp has no T5 encoder embedder layout, so `--format gguf` refuses T5 too.
- An embedder's **`Dense` projection** (any encoder family) is written as the sentence-transformers module `2_Dense`
  (`config.json` with the sizes, the bias and the activation; `model.safetensors` with `linear.weight` and
  `linear.bias`) between the pooling and `Normalize`. GGUF has no place for it, so such models export to safetensors
  only.

### Text-to-text models

T5 and Flan-T5 text-to-text models (Phase 65) export to safetensors as `T5ForConditionalGeneration`, for example
after [fine-tuning one](training.md#text-to-text-models):

```bash
dllm export flan-tuned.dllm --format safetensors -o flan-tuned/
```

The encoder is written as for a T5 encoder, the decoder under `decoder.block.N` (`layer.0.SelfAttention`,
`layer.1.EncDecAttention`, `layer.2.DenseReluDense`) with its bucket table in the first block, and `lm_head` unless
the head is tied to `shared` (T5 v1.0). The config (`num_decoder_layers`, `tie_word_embeddings`,
`decoder_start_token_id` 0) is read back through the importer first and must give the same model, and importing the
directory gives back the same fingerprint. `T5ForConditionalGeneration.from_pretrained(dir)` loads it and its greedy
`generate` gives the engine's answer. GGUF is refused.

### SentencePiece models through GGUF

GGUF files of TinyLlama, Llama 2, Mistral or Phi-3 usually carry a SentencePiece vocabulary
(`tokenizer.ggml.model = llama`: pieces, scores and token types) rather than merges. `dllm import` reads it and
tokenizes exactly as SentencePiece (and llama.cpp) do:

- every text gets a `▁` in front (also after a special token) when `tokenizer.ggml.add_space_prefix` is true (the
  default), and every space becomes `▁`;
- adjacent pieces merge into the normal piece with the highest score first; two candidates with the same score merge
  left to right, as SentencePiece breaks ties by position;
- characters no piece covers fall back to the `<0xAB>` byte pieces (or `<unk>` without them), and decoding drops the
  prefix's leading space.

Tests check this against the `sentencepiece` library itself, token for token, on models trained with it. A
safetensors export of such an import writes the merges in that order as a `tokenizer.json`; the Hugging Face
tokenizers library resolves two merges with exactly equal scores by list order instead of position, the one case
where the two can differ. A Hugging Face tokenizer whose `Metaspace` adds the space prefix only to the first text
(`prepend_scheme: first`, newer Mistral files) cannot be written as a GGUF `llama` tokenizer, which adds it after
special tokens too, so that export is refused.

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
