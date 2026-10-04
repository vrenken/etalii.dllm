# Retrieval and grounding

EtAlii.Dllm can answer from your own documents: an embedding model turns them into an exact vector index, and every
chat can be grounded in the passages the index finds for the question. Retrieval keeps the project's guarantee. The
same documents and embedding model give a byte-identical index file, the same query finds the same passages on every
machine, and a grounded server reports its own `system_fingerprint`.

## Embedding models

Any imported model can embed text (`/v1/embeddings`, `/api/embed`, `DllmEngine.embed`): a chat model averages its
final hidden states. Trained embedding models do much better. Models published for
[sentence-transformers](https://www.sbert.net/) import with their own recipe, which `dllm import` reads from
`modules.json`, the pooling module's `config.json` and `config_sentence_transformers.json`:

- pooling: `last_token` (the final hidden state of the last token, as in Qwen3-Embedding), `cls` (the first
  token's, as in bge) or `mean` (all-MiniLM);
- normalisation: an L2 `Normalize` module;
- prompts: named instructions such as `query`, put in front of the text (the `input_type` of a request);
- checkpoints saved without the causal LM wrapper (no `model.` prefix, no LM head) import as well.

The import stores this as the `embedding` section of `model.dllm` ([format](model-format.md)). Embedding then
encodes the text with the tokenizer's special tokens (Qwen3-Embedding appends `<|endoftext|>`) and pools as the
model says. Other pooling modes (max) and pooling that leaves out the prompt are refused at import.

### Encoder models

Small BERT encoders are what most retrieval systems use: all-MiniLM-L6-v2 (22M parameters, Apache 2.0) and
bge-small-en-v1.5 (33M, MIT) embed far faster than a decoder and are trained for exactly this. They import with their
WordPiece tokenizer (reproduced exactly, `tokenizers`' own character classes included) and their sentence-transformers
settings; inputs longer than the model's `max_seq_length` are truncated as sentence-transformers truncates them,
keeping `[CLS]` and `[SEP]`:

```bash
dllm import hf:sentence-transformers/all-MiniLM-L6-v2 -o minilm.dllm
dllm --model minilm.dllm embed "What is the capital of France?" --json
dllm --model minilm.dllm index build docs/ -o docs.index
dllm --model chat.dllm --index docs.index --embedding-model minilm.dllm chat "What does the guide say about X?"
```

An encoder runs on the CPU, embeds only (chat and completions refuse it with an error; it fine-tunes, takes LoRA
adapters and exports since Phase 58: [fine-tuning encoders](training.md#encoders)) and
gives the same bits on every machine: the encoder's forward pass and pooling are in the
[specification](specification.md#6-the-encoder) and `dllm verify --reference` checks them against the reference
implementation. A plain BERT checkpoint without sentence-transformers files pools the mean over all positions.

#### RoBERTa and multilingual encoders

RoBERTa and XLM-RoBERTa encoders import too (`model_type` `roberta` and `xlm-roberta`): they are BERT whose
position ids count from past the padding token, with a single token type. Multilingual models usually carry
XLM-RoBERTa's Unigram tokenizer, which is reproduced exactly as `tokenizers` runs it: SentencePiece's precompiled
normaliser (looked up per grapheme cluster, with the cluster tables of `tokenizers`' own segmentation crate), the
`▁` pre-tokenizer and the Viterbi segmentation, ties and unknown pieces included:

```bash
dllm import hf:sentence-transformers/all-distilroberta-v1 -o distilroberta.dllm
dllm import hf:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 -o multilingual.dllm
dllm --model multilingual.dllm embed "Wie viele Menschen leben in Berlin?" --json
```

#### ModernBERT encoders

ModernBERT encoders (`model_type` `modernbert`, Phase 59) import too, for example
[gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) and
[modernbert-embed-base](https://huggingface.co/nomic-ai/modernbert-embed-base) (both Apache 2.0). ModernBERT is a
newer encoder: pre-norm layers with LayerNorms that have no bias, fused query/key/value projections, rotary positions
with one base for global layers and another for local ones, local layers that only see keys within half of
`local_attention` on either side (every `global_attn_every_n_layers`-th layer sees all), a gated GELU MLP and a final
norm. Its tokenizer is a byte-level BPE, reproduced as for the decoders:

```bash
dllm import hf:Alibaba-NLP/gte-modernbert-base -o gte-modernbert.dllm
dllm --model gte-modernbert.dllm embed "What is the capital of France?" --json
```

The forward pass is in the [specification](specification.md#6-the-encoder) and `dllm verify --reference` checks it.
Checkpoints with biases, RoPE scaling or a head activation different from the MLP's are refused at import. ModernBERT
is checked against `transformers` on tiny synthetic models with sequences longer than the local window
(`tests/test_modernbert.py`), not yet on the real checkpoints in CI.

#### DeBERTa encoders

DeBERTa-v3 encoders (`model_type` `deberta-v2`, Phase 60) import too, as embedders or, more often, as
cross-encoders (below). DeBERTa has no absolute positions: its attention adds two terms to every score, the query
against the embedding of the key's relative distance and the key against the query's, over a shared table of
relative distances that grow logarithmically beyond `position_buckets / 2`. The buckets are computed exactly as
transformers computes them (in float32, with the portable logarithm), and the scores go through the
`biased_attention` kernel, so the result has the same bits on every machine. Its Unigram tokenizer is read from
`tokenizer.json`; a checkpoint that only has `spm.model` is refused at import with the one-line fix (save it again
with `AutoTokenizer.from_pretrained(dir).save_pretrained(dir)`).

Only DeBERTa-v3's layout imports (relative attention with shared position projections, both position terms, the
relative LayerNorm, no absolute positions); the convolution layer of some large v2 checkpoints is refused. DeBERTa is
checked against `transformers` on tiny synthetic models with sequences longer than the buckets' exact range
(`tests/test_deberta.py`), not yet on the real checkpoints in CI. Since Phase 61 it fine-tunes (with or without LoRA,
[fine-tuning encoders](training.md#encoders)) and exports to safetensors
([exporting encoders](model-building.md#encoders)).

#### T5 encoders

The encoders of T5 (`model_type` `t5`, Phase 62) import as embedders, as
[sentence-t5-base](https://huggingface.co/sentence-transformers/sentence-t5-base) and
[gtr-t5-base](https://huggingface.co/sentence-transformers/gtr-t5-base) (Apache 2.0) store them:

```bash
dllm import hf:sentence-transformers/gtr-t5-base -o gtr.dllm
dllm --model gtr.dllm embed "How many people live in Berlin?"
```

A full T5 checkpoint (`T5ForConditionalGeneration`) in a sentence-transformers directory imports too, with its
decoder dropped; without `modules.json` it imports as a text-to-text model that generates (see
[getting started](getting-started.md#60-t5-text-to-text-generation)). T5 has no position
embeddings: every layer adds a per-head bias to its attention scores that depends only on the distance from the query
to the key, read from 32 buckets (exact up to 8 tokens, then logarithmic up to 128, the same for anything farther).
The buckets are computed exactly as transformers computes them (in float32, with the portable logarithm), and the
scores go through the `biased_attention` kernel without scaling, as T5 has none; the RMS norms are the `rms_norm`
kernel, the MLP is T5 v1.0's ReLU or v1.1's gated tanh GELU. sentence-t5 and GTR-T5 project the mean-pooled vector
with a `Dense` module (768 to 768, no bias, no activation) before normalising it: the importer reads `2_Dense` and
the engine applies it with the `linear` kernel, for any encoder family (LaBSE's `Dense` has a bias and `tanh`, which
works the same way).

T5 is checked against `transformers`' `T5EncoderModel` on tiny synthetic models with sequences far longer than the
buckets' maximum distance (`tests/test_t5.py`), not yet on the real checkpoints in CI. T5 embedders, and embedders
with a `Dense` projection of any family, fine-tune (with or without LoRA) and export to safetensors (Phase 63;
[fine-tuning encoders](training.md#encoders), [exporting encoders](model-building.md#encoders)).

all-MiniLM-L6-v2, bge-small-en-v1.5, all-distilroberta-v1, paraphrase-multilingual-MiniLM-L12-v2 and
[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (Apache 2.0) are checked against
`transformers` in CI (`tests/test_reference_models.py`):

```bash
dllm import hf:Qwen/Qwen3-Embedding-0.6B -o qwen3-embedding.dllm
curl localhost:5080/v1/embeddings -d '{"input": "What is the capital of France?", "input_type": "query"}'
```

`input_type` is an extension of the OpenAI request; leave it out for documents. Qwen3-Embedding's `query` prompt
is `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:`.

## Building an index

```bash
dllm --model qwen3-embedding.dllm index build docs/ notes.txt -o docs.index
```

`index build` reads the given files and every `.md`, `.markdown`, `.txt`, `.rst`, `.html`, `.htm`, `.json`, `.py`,
`.csv`, `.yaml` and `.yml` file under the given directories. Every step is fixed:

1. Documents are sorted by their path (relative to the directory given, `/` separated, compared as UTF-8 bytes),
   decoded as UTF-8 and their line endings normalised to `\n`.
2. Each document is split into paragraphs at blank lines. Paragraphs are packed in order into chunks of at most
   `--chunk-tokens` tokens (default 256). A paragraph that is longer on its own is split at word boundaries. Only
   ASCII white space separates paragraphs and words, so chunks do not depend on the Python version's Unicode tables.
3. Every chunk is embedded on its own, with the model's `document` prompt when it has one.

The index file is safetensors: the vectors as `vectors [chunks, dimensions]` float32, and in the `index` metadata
entry (canonical JSON) the format version, the chunks (source, character offsets, text), the chunk size and the
embedding model (weights fingerprint, id and the path it was loaded from). The fingerprint is the SHA-256 of all of
it. Loading checks it, so an index that was changed is refused.

## Searching

```bash
dllm index search docs.index "How do I import a gated model?" --top 3
dllm index search docs.index "How do I import a gated model?" --json
```

Without `--model`, search loads the embedding model the index records. The query is embedded with the `query`
prompt and compared with every chunk by the `cosine_similarity` kernel (double sums in a fixed order). There is no
approximate nearest-neighbour structure that could return different passages on another machine. Hits are ranked
by a total order: the higher score first, and the earlier chunk on a tie. An embedding model other than the index's
is refused, because its vectors would not be comparable.

Exact search reads every vector per query. That is fast for collections of up to some hundred thousand chunks, which
covers documentation, notes and code bases.

## Lexical and hybrid search

```bash
dllm index search docs.index "BM25 gated model import" --mode lexical
dllm index search docs.index "How do I import a gated model?" --mode hybrid
```

`--mode lexical` ranks chunks by Okapi BM25 instead of embeddings, and needs no embedding model. `--mode hybrid` fuses
both rankings. Like dense search, both are exact and give the same passages on every machine:

- **Terms.** Text is NFKC-normalised and lower-cased with the project's pinned Unicode tables, and the terms are the
  maximal runs of letters, marks and digits (categories `L*`, `M*`, `N*`). So `Café`, `cafe\u0301` and `CAFÉ` are one
  term, and nothing depends on the Python version's Unicode data or the locale.
- **BM25** with `k1 = 1.2` and `b = 0.75`: a chunk's score adds, for each distinct query term in order of first
  appearance, `idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / average_length))`, with
  `idf = log(1 + (N - df + 0.5) / (df + 0.5))` from the portable `log` kernel. Every step is one IEEE double operation
  in a fixed order. Lexical search returns only chunks that share a term with the query.
- **Statistics.** Term counts, document frequencies and lengths are computed from the chunk texts the index already
  stores, so the file format is unchanged and every existing index supports lexical search.
- **Hybrid** is reciprocal rank fusion: a chunk scores `1 / (60 + rank)` for its rank in the dense ranking, plus the
  same for its rank in the lexical ranking when it is in it (dense first, then lexical, in double). Hybrid search
  helps with exact names, numbers and rare words that embeddings blur.

All three modes rank by a total order: the higher score first, and the earlier chunk on a tie.

## Reranking

```bash
dllm --model qwen2.5-1.5b.dllm rerank "Where is Paris?" "Berlin is in Germany." "Paris is in France."
dllm --model qwen2.5-1.5b.dllm rerank "Where is Paris?" --file passages.txt --top 3 --json
dllm --rerank-model qwen2.5-1.5b.dllm index search docs.index "How do I import a gated model?" --mode hybrid
curl http://localhost:5080/v1/rerank -d '{"query": "Where is Paris?", "documents": ["Berlin ...", "Paris ..."]}'
```

A chat model can judge relevance more closely than an embedding comparison. The reranker follows the Qwen3-Reranker
recipe, and any chat model can run it:

- The prompt is the model's chat template with a fixed system message (judge whether the document meets the query;
  answer only "yes" or "no") and a user message `<Instruct>: ...\n<Query>: ...\n<Document>: ...`, then the generation
  prompt. Thinking is switched off for thinking models. `--instruction` (`instruction` in the API) replaces the default
  instruction, "Given a web search query, retrieve relevant passages that answer the query".
- The score is `sigmoid(logit(yes) - logit(no))` for the next token, in double with the portable kernel, where `yes`
  and `no` are the first tokens of those words. A model whose two words start with the same token is refused.
- Documents are ranked by the higher score first, the earlier document on a tie.

### Cross-encoder rerankers

A cross-encoder is a small BERT model trained to score a query and a passage read together, which is what
rerankers built for search do. [ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
(22M parameters, Apache 2.0) reranks far faster than a chat model judge:

```bash
dllm import hf:cross-encoder/ms-marco-MiniLM-L6-v2 -o ms-marco.dllm
dllm --model ms-marco.dllm rerank "How many people live in Berlin?" "Berlin has 3.5 million people." "Berlin has museums."
dllm --rerank-model ms-marco.dllm index search docs.index "How do I import a gated model?" --mode hybrid
```

- The import keeps the sequence-classification head (the pooler and the classifier on the `[CLS]` state) and
  records its labels, the activation sentence-transformers' `CrossEncoder` applies (`Sigmoid` or none) and the
  longest pair the model takes (`max_length`, else the tokenizer's `model_max_length`).
- The query and the passage are encoded as one pair, `[CLS] query [SEP] passage [SEP]` with token types 0 and 1,
  truncated longest first exactly as `tokenizers` truncates it, so the tokens are those transformers feeds the model.
- The score is the classifier's logit (or its sigmoid when the model says so); the instruction does not apply.
  Documents are ranked as above, the higher score first and the earlier document on a tie.
- `dllm rerank`, `/v1/rerank`, `--rerank-model` with `index search` and grounded chats all use it when the model is
  a cross-encoder. A classifier with several labels is not a reranker and is refused. In Python,
  `engine.classify(query, passage)` returns the logits, the scores and the labels.

The pair encoding and the head are in the [specification](specification.md#6-the-encoder), and
`dllm verify --reference` checks a cross-encoder's scores against the reference implementation bit for bit.

XLM-RoBERTa cross-encoders rerank in many languages, for example
[mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) (118M parameters,
Apache 2.0, checked against `transformers` in CI). Their pair is `<s> query </s></s> passage </s>`, every token of
type 0, and their classification head is the same computation as BERT's:

```bash
dllm import hf:cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 -o mmarco.dllm
dllm --model mmarco.dllm rerank "Wie viele Menschen leben in Berlin?" "Berlin hat 3,5 Millionen Einwohner." "Berlin hat Museen."
```

ModernBERT cross-encoders such as
[gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) (149M parameters,
Apache 2.0) import as rerankers too. Their pair is `[CLS] query [SEP] passage [SEP]` without token types, and their
head pools the `[CLS]` state or the mean over the tokens (`classifier_pooling`), then applies a dense layer, the
GELU, a LayerNorm and the classifier:

```bash
dllm import hf:Alibaba-NLP/gte-reranker-modernbert-base -o gte-reranker.dllm
dllm --model gte-reranker.dllm rerank "How many people live in Berlin?" "Berlin has 3.5 million people." "Berlin has museums."
```

DeBERTa-v3 cross-encoders such as
[mxbai-rerank-xsmall-v1](https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1) (71M parameters, Apache 2.0)
rerank as well, and natural language inference models such as
[nli-deberta-v3-small](https://huggingface.co/cross-encoder/nli-deberta-v3-small) (Apache 2.0) classify a pair into
their three labels with `engine.classify`. Their head is the context pooler, `classifier(gelu(pooler(h[0])))`, and
their pair is `[CLS] query [SEP] passage [SEP]`:

```bash
dllm import hf:mixedbread-ai/mxbai-rerank-xsmall-v1 -o mxbai-rerank.dllm
dllm --model mxbai-rerank.dllm rerank "How many people live in Berlin?" "Berlin has 3.5 million people." "Berlin has museums."
```

`POST /v1/rerank` (also `/rerank`) takes the request shape of Cohere and Jina: `query`, `documents` (strings or
`{"text": ...}`), `top_n` and `return_documents` (default true), and the extension `instruction`. It returns
`{"id", "model", "results": [{"index", "relevance_score", "document"}], "usage": {"total_tokens"}}`. The served model
judges, and the id is derived from the request. `--rerank-model` with `index search` reranks the first `4 * top` hits
and keeps the best `top`.

## Grounded chat

```bash
dllm --model qwen2.5-1.5b.dllm --index docs.index chat "How do I import a gated model?"
dllm-server --model qwen2.5-1.5b.dllm --index docs.index --index-top 4
DLLM_INDEX=docs.index dllm-mcp --model qwen2.5-1.5b.dllm
```

With `--index` (or `DLLM_INDEX`), every chat request is grounded: the last user message is searched for, and the top
`--index-top` passages (`DLLM_INDEX_TOP`, default 3) are added to the system message. If the request has no system
message, a new one is put first. The passages come after a fixed instruction to answer from them and cite them by
number:

```text
Answer using the passages below when they are relevant, and say which passage you used by its number. If they do not contain the answer, say so.

[1] (guide/import.md)
...
```

The index records the path of its embedding model; `--embedding-model` (`DLLM_EMBEDDING_MODEL`) names it when the
file has moved. Raw prompts (Ollama's `raw` mode, `/v1/completions`) are not grounded.

`--index-mode lexical|hybrid` (`DLLM_INDEX_MODE`, default `dense`) chooses how grounding searches, and
`--rerank-model` (`DLLM_RERANK_MODEL`) reranks the first `4 * --index-top` hits with a chat model and grounds in the
best `--index-top`. Lexical grounding needs no embedding model.

Grounding changes the output, so it changes the `system_fingerprint`: the chat model's fingerprint is combined with
the index fingerprint and `--index-top` (and the search mode and reranker when they are not the defaults, so dense
grounding keeps the fingerprints it always had). Equal fingerprints and equal requests still give equal answers on every front
end. Grounding works for the OpenAI, Anthropic and Ollama APIs, the chat page, `dllm chat` and the MCP `chat` tool.

## The MCP `search_documents` tool

A `dllm-mcp` server started with `--index` also offers `search_documents(query, top=5)`, so an MCP client such as
Claude Code can search your documents directly. The tool returns a JSON list of `{rank, score, source, start, end,
text}`, and it is deterministic: the same query always returns the same passages. Without an index the tool reports
an error.

## Python

```python
from etalii_dllm.engine import DllmEngine
from etalii_dllm.retrieval import Index, Retriever, build_index, read_documents

embedder = DllmEngine.from_model_file("qwen3-embedding.dllm")
index = build_index(embedder, read_documents(["docs/"]), model_path="qwen3-embedding.dllm")
index.save("docs.index")
for hit in Index.load("docs.index").search(embedder, "gated models", top=3):
    print(hit.rank, round(hit.score, 3), hit.chunk.source, hit.chunk.text[:60])

chat = DllmEngine.from_model_file("qwen2.5-1.5b.dllm", index="docs.index")  # grounded chat engine
```
