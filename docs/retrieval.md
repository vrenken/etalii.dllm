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

- pooling: `last_token` (the final hidden state of the last token, as in Qwen3-Embedding) or `mean`;
- normalisation: an L2 `Normalize` module;
- prompts: named instructions such as `query`, put in front of the text (the `input_type` of a request);
- checkpoints saved without the causal LM wrapper (no `model.` prefix, no LM head) import as well.

The import stores this as the `embedding` section of `model.dllm` ([format](model-format.md)). Embedding then
encodes the text with the tokenizer's special tokens (Qwen3-Embedding appends `<|endoftext|>`) and pools as the
model says. Other pooling modes (CLS, max) and pooling that leaves out the prompt are refused at import.

[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (Apache 2.0) is checked against
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

Grounding changes the output, so it changes the `system_fingerprint`: the chat model's fingerprint is combined with
the index fingerprint and `--index-top`. Equal fingerprints and equal requests still give equal answers on every front
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
