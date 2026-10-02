# EtAlii.Dllm

**A deterministic large language model, built from scratch in Python with C++ kernels.**

Mainstream LLM inference is not reproducible: the same prompt, the same weights and even the same `seed` can
give different answers from one request to the next. EtAlii.Dllm treats bit-exact reproducibility as a hard
requirement instead of a best-effort hint:

> Same weights + same prompt and context window ⇒ the same tokens, bit for bit, on every run and on every machine.

That holds regardless of server load, batch composition or thread scheduling, and across machines: x86-64 and arm64
CPUs on Linux, Windows and macOS, every SIMD path, and NVIDIA GPUs give the same bits, float32 or Q8_0 alike
(checked in CI with the real models on all five release platforms). `dllm verify` prints one fingerprint to
compare two machines. What it takes: [portable determinism](docs/kernels.md#portable-determinism). Exactly which
bits: the [determinism specification](docs/specification.md), checked by an independent second implementation.

The model speaks the protocols the rest of the ecosystem already uses (the OpenAI, Anthropic and Ollama HTTP APIs, with
streaming, tool calling and JSON-schema structured output, and the Model Context Protocol), so existing clients and
agents can use it without changes.

## Status

Phases 1 to 6 (kernels, importing models, fine-tuning, API parity, MCP, performance) done, including a CUDA backend
that reproduces the CPU bits on NVIDIA GPUs. The full pipeline (tokenizer →
model → sampler → CLI / HTTP API / MCP server) runs end to end and is proven bit-exact by CI, run to run and across Linux, Windows and macOS on
x86-64 and arm64 (Phase 10). The transformer building blocks (aligned `Tensor`, batch-invariant matmul,
RMSNorm, SiLU/GELU, RoPE and grouped-query attention, see [docs/kernels.md](docs/kernels.md)) are in place, and so are
`dllm import` (safetensors/GGUF to our [model.dllm](docs/model-format.md) format, with source and licence recorded)
a Llama/Qwen2/Qwen3 decoder whose KV cache cannot change its output, the models' own BPE tokenizers and chat templates, and
`--model` on every front end. The real SmolLM2-135M-Instruct, Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct and Qwen3-0.6B imports match Hugging Face
`transformers` (logits within about 2e-5, identical greedy answers), checked in CI. `dllm finetune` trains an imported model further with AdamW, reproducibly: equal
runs, and runs resumed from a checkpoint, write byte-identical models (see [docs/training.md](docs/training.md)).
The HTTP API speaks OpenAI, Anthropic and Ollama, with streaming, tool calling, JSON-schema structured output
(constrained decoding), logprobs and embeddings; streamed and non-streamed answers are identical (see
[docs/api.md](docs/api.md)).
The kernels run on all cores with SIMD (AVX2, SSE2 or NEON) and still give the same bits as the plain scalar loops,
whatever the thread count or batch, and `--quantize q8_0` runs 8-bit weights with exact integer accumulation (see
[docs/kernels.md](docs/kernels.md#threads-and-simd)).
Without a model file the engine falls back to a seeded placeholder (a bigram table).

## Which models can it run?

Not every model, but any model in a supported family. Determinism is a property of the engine, not of the weights:
`dllm import` only converts a model's weights, tokenizer and chat template to our [model.dllm](docs/model-format.md)
format, and every model the engine runs is deterministic. So a model becomes deterministic as soon as it imports.
What limits the choice is which architectures the engine implements.

| | Supported | Refused at import (for now) |
| --- | --- | --- |
| Architecture | Llama-style decoders (`model_type` `llama`: SmolLM2, TinyLlama, Llama 2, Llama 3.x, ...), Mistral (sliding-window attention), OLMo 2, IBM Granite 3.x (dense), Phi-3/Phi-3.5-mini/Phi-4-mini/Phi-4 (fused projections, partial rotary, LongRoPE up to the original context), Qwen2/Qwen2.5, dense Qwen3, Gemma 2 (2B, 9B; attention and final logit soft-capping), Gemma 3 text models (`gemma3_text`: 270M, 1B) | multimodal Gemma 3 checkpoints, Phi-1/Phi-2, Phi-3-small, mixture-of-experts models (Mixtral, Qwen3-MoE), GELU MLPs, MLP biases, YaRN RoPE scaling |
| Tokenizer | BPE from `tokenizer.json` (the model's own, with its Jinja chat template): byte-level (GPT-2, Llama 3, Qwen) and SentencePiece-style with `▁` and byte fallback (Llama 2, TinyLlama, Mistral, Phi-3) | Unigram and WordPiece tokenizers (T5, BERT), SentencePiece `tokenizer.model` files without a `tokenizer.json`, GGUF files with a SentencePiece (`llama`) vocabulary |
| Files | Hugging Face safetensors (F32/F16/BF16), GGUF (F32/F16/BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K), PEFT LoRA adapters | Other GGUF quantisations |
| Size | Weights are held in memory as float32 (or Q8_0 with `--quantize q8_0`), so memory and CPU speed set the limit; about 1.5B parameters is practical today | |

Verified against Hugging Face `transformers` in CI: SmolLM2-135M-Instruct, Qwen2.5-0.5B-Instruct,
Qwen2.5-1.5B-Instruct, Qwen3-0.6B, TinyLlama-1.1B-Chat and OLMo-2-1B-Instruct. Other models in these families should work but are not checked. An unsupported
model fails at `dllm import` with an error that names the missing feature, never with silently wrong output. A server
serves one model, chosen at start-up with `--model`/`DLLM_MODEL`.

## Quick start

New here? [Getting started](docs/getting-started.md) walks through installing, importing a real model and using it
from the command line, the HTTP API and MCP.
How it compares with transformers and llama.cpp on speed, memory, quality and determinism:
[benchmarks](docs/benchmarks.md).

Pre-built wheels for Linux, Windows and macOS are attached to every [release](https://github.com/vrenken/etalii.dllm/releases)
(see [getting started](docs/getting-started.md#1-install)). Building from source requires Python 3.11+, CMake and
a C++17 compiler (the numeric kernels are a C++ extension built on install).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest

# Command line
dllm info
dllm generate --prompt "Hello" --temperature 0.8 --seed 7

# OpenAI-, Anthropic- and Ollama-compatible HTTP server on http://localhost:5080 (see docs/api.md)
dllm-server
curl http://localhost:5080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"dllm","messages":[{"role":"user","content":"Hi"}],"temperature":0.7,"seed":5}'

# MCP server over stdio, e.g. registered with Claude Code
claude mcp add dllm -- dllm-mcp

# Import an open-weight model to our own format (source, revision and licence are recorded in the file)
dllm import hf:HuggingFaceTB/SmolLM2-135M-Instruct -o smollm2-135m.dllm
dllm import ./qwen2.5-0.5b-instruct-q8_0.gguf -o qwen2.5-0.5b.dllm
dllm inspect smollm2-135m.dllm

# Run it: every front end takes --model (or the DLLM_MODEL environment variable) and uses the model's own
# tokenizer and chat template
dllm --model smollm2-135m.dllm chat "What is the capital of France?"
dllm-server --model smollm2-135m.dllm
claude mcp add dllm -- dllm-mcp --model /path/to/smollm2-135m.dllm

# Let the model call the tools of other MCP servers during a chat (mcp.json in the usual mcpServers format)
dllm --model qwen2.5-0.5b.dllm chat "What time is it in Amsterdam?" --mcp-config mcp.json

# Fine-tune it on your own text; the same data and options give a byte-identical model
dllm finetune smollm2-135m.dllm --data my-data.jsonl -o smollm2-135m-tuned.dllm --steps 50
```

Run the same request twice and compare: the responses, including `id` and `system_fingerprint`, are identical.

## Why Python and C++

Python is where the model ecosystem lives: loading safetensors/GGUF, tokenizers, reference implementations to
compare against, the official MCP SDK and FastAPI for the HTTP API. Deterministic arithmetic cannot be left to
NumPy or BLAS, whose reduction order depends on array size, thread count and library version, so every kernel that
reduces (sums, dot products, softmax, later matmul and attention) is written in C++ with a fixed order and exposed
to Python through [nanobind](https://github.com/wjakob/nanobind). Python orchestrates; C++ computes.

## Repository layout

| Path | What it is |
| --- | --- |
| `cpp/` | C++ kernels: `include/dllm/random.hpp` (RNG), `include/dllm/math.hpp` (exp, log, sin, cos, tanh, erf, sum, dot, softmax, log-softmax), `include/dllm/nn.hpp` (linear/matmul, RMSNorm, SiLU/GELU, RoPE, attention), `include/dllm/grad.hpp` (their gradients, cross-entropy, AdamW), `kernels.cpp` (Python bindings). Evaluation orders: [docs/kernels.md](docs/kernels.md) |
| `src/etalii_dllm/` | Python package: `tensor` (aligned float32 `Tensor`), `numerics`, `sampling`, `tokenization`, `models`, `generation` (streaming, stop sequences, logprobs), `grammar` (constrained decoding), `tools` (tool calling), `chat`, `engine` (shared facade) |
| `src/etalii_dllm/training/` | Fine-tuning: decoder gradients, AdamW, fixed data order, checkpoints. See [docs/training.md](docs/training.md) |
| `src/etalii_dllm/server/` | HTTP API, FastAPI: OpenAI (`/v1/models`, `/v1/chat/completions`, `/v1/responses`, `/v1/embeddings`) Anthropic (`/v1/messages`) and Ollama (`/api/chat`, `/api/generate`, ...); a chat page at `/` (`static/chat.html`). See [docs/api.md](docs/api.md) |
| `src/etalii_dllm/mcp_server.py` | Model Context Protocol server (stdio, official `mcp` SDK): `chat`, `generate` and `model_info` tools, model card/chat template/determinism resources, prompts. See [docs/mcp.md](docs/mcp.md) |
| `src/etalii_dllm/mcp_host.py` | MCP client host: the model calls external MCP servers' tools during a chat (`dllm chat --mcp-config`) |
| `src/etalii_dllm/cli.py` | `dllm` command line tool |
| `src/etalii_dllm/importing/` | Model import: safetensors and GGUF readers, GGUF dequantisation, Hugging Face download, `dllm import` |
| `src/etalii_dllm/bpe.py`, `chat_template.py` | BPE tokenizer (byte-level or SentencePiece-style) from `tokenizer.json` (or GGUF metadata) and the model's Jinja chat template |
| `src/etalii_dllm/transformer.py` | Llama/Qwen2/Qwen3 decoder with a KV cache that cannot change the logits |
| `src/etalii_dllm/modelfile.py` | The [`model.dllm`](docs/model-format.md) container that imported models are stored in |
| `tests/` | pytest suite, including golden-hash reproducibility tests |
| `benchmarks/` | Performance comparison with transformers and llama.cpp (`benchmark.py`, the `Benchmark` workflow). Results: [docs/benchmarks.md](docs/benchmarks.md) |
| `docs/research/` | Research notes: [deterministic inference](docs/research/deterministic-inference.md), [compatibility targets](docs/research/compatibility.md), [model import](docs/research/model-import.md) |

## How determinism is achieved

Summarised from the [research notes](docs/research/deterministic-inference.md):

- **Own random numbers.** A specified generator (xoshiro256\*\* seeded by SplitMix64) instead of `random` or `numpy.random`.
- **Own transcendental functions.** `exp`, `log`, `sin`, `cos`, `tanh` and `erf` built from IEEE basic operations only.
  Not strictly needed on one machine, but cheap, and it keeps runtime or library updates from shifting results.
- **Fixed reduction order.** Sums, dot products, softmax, matmul, RMSNorm and attention run in C++ and accumulate in one documented order, never in an order
  chosen by thread scheduling or batch size (batch invariance), so concurrent requests cannot change each other's output.
- **Pinned Unicode.** Normalisation, lower-casing and the `\p{L}`-style classes in pre-tokenizer patterns use Unicode
  15.1 tables shipped in the package, not the ones of the installed Python or `regex`, so the same text gives the same
  tokens on every installation.
- **Total-order sampling.** Ties break on token id, so sorting never depends on algorithm stability.
- **Golden hashes in CI.** Tests assert SHA-256 hashes of weights and generated tokens; any drift fails the build.
  SIMD and threads only ever split work between output elements, so they have not changed a single hash.

## Architecture

[docs/architecture](docs/architecture/README.md) describes how the pieces fit together (system context, layers,
module map), with Mermaid diagrams.

## Roadmap

| Phase | Goal |
| --- | --- |
| 0. Bootstrap ✅ | Solution skeleton, deterministic RNG and math, sampler, byte tokenizer, placeholder model, OpenAI-style API, MCP server, CI with golden hashes |
| 1. Kernels ✅ | Tensor type, deterministic matmul with fixed tiling, RMSNorm, RoPE with deterministic `sin`/`cos`, SiLU/GELU, attention with fixed-order softmax |
| 2. Import existing models ✅ | Llama-style decoder with KV cache; `dllm import` converting small open-weight models (SmolLM2, Qwen2.5, ...) from safetensors/GGUF to our own format with licence metadata; BPE tokenizer and chat templates. SmolLM2-135M and Qwen2.5-0.5B verified against `transformers` in CI. See [model import](docs/research/model-import.md) |
| 3. Fine-tuning ✅ | Deterministic backprop and AdamW on top of imported weights, fixed data order, reproducible checkpoints |
| 4. API parity ✅ | Streaming (SSE), tool/function calling, JSON-schema structured output, logprobs, Anthropic Messages endpoint, embeddings. See [HTTP API](docs/api.md) |
| 5. MCP, both directions ✅ | Richer MCP server (prompts, resources); MCP client host so the model can call external tools during a chat. See [MCP](docs/mcp.md) |
| 6. Performance ✅ | ✅ SIMD and multi-threading with fixed, batch-invariant reduction order; ✅ integer quantisation (Q8_0, associative int32 accumulation); ✅ batch-invariance stress tests; ✅ CUDA kernels that give the CPU's bits (`--device cuda`). See [kernels](docs/kernels.md#threads-and-simd) and [GPU](docs/kernels.md#gpu) |
| 7. Usability and releases | ✅ Pre-built wheels for Linux, Windows and macOS and tagged GitHub releases ([releasing](docs/releasing.md)); PyPI; ✅ the CUDA install route checked in CI; ✅ a Docker image (`ghcr.io/vrenken/etalii-dllm`); ✅ a web chat UI at `/` of `dllm-server`; ✅ a larger verified model (Qwen2.5-1.5B) |
| 8. Serving and ecosystem ✅ | ✅ Prompt caching across requests with the same bits as a cold run ([prompt caching](docs/api.md#prompt-caching)); ✅ concurrent requests decoded as one batch, each keeping its solo bits ([batching](docs/api.md#concurrent-requests)); ✅ Ollama-compatible API ([Ollama API](docs/api.md#ollama-api)); ✅ OpenAI Responses API ([Responses API](docs/api.md#responses-api)); ✅ Qwen3 (verified Qwen3-0.6B); ✅ LoRA fine-tuning and PEFT adapter import ([LoRA](docs/training.md#lora-adapters)) |
| 9. Mainstream model families | ✅ SentencePiece-style tokenizers (TinyLlama, Llama 2, Mistral, Phi-3; verified TinyLlama-1.1B-Chat); ✅ Mistral and sliding-window attention; ✅ Gemma 3 (text); ✅ Gemma 2; ✅ Phi-3/Phi-4-mini; ✅ OLMo 2; ✅ Granite 3.x; ✅ verified Llama-3.2-1B-Instruct and gemma-3-270m-it. Each family is checked against `transformers`, see [which models can it run](#which-models-can-it-run) |
| 10. Portable determinism ✅ | Identical output across machines as a guarantee, not a bonus: ✅ kernels run in the IEEE default floating point environment whatever the process set ([kernels](docs/kernels.md#floating-point-environment)); ✅ tokenizer and chat template independent of the Python, `regex` and `jinja2` versions; ✅ real-model golden bits checked on every release platform and SIMD path; ✅ `dllm verify` to compare two machines; ✅ the guarantee documented ([portable determinism](docs/kernels.md#portable-determinism)) |
| 11. Interpretability and editing ✅ | ✅ Activation tracing that cannot change a bit of the output; ✅ logit lens (`dllm lens`); ✅ embedding explorer, analogies and word clouds (`dllm neighbours`); ✅ attention maps (`dllm attention`); ✅ steering vectors (`dllm steer`, `--steer`); ✅ ROME model edits (`dllm edit`); ✅ sparse autoencoders (`dllm sae`). See [interpretability](docs/interpretability.md) |
| 12. Retrieval and grounding ✅ | ✅ Embedding models with their own pooling and query prompts, verified Qwen3-Embedding-0.6B; ✅ exact, reproducible document index (`dllm index build`/`search`); ✅ chats grounded in your documents in every front end (`--index`) and an MCP `search_documents` tool. See [retrieval](docs/retrieval.md) |
| 13. Speed without changing a bit ✅ | ✅ Activations on the thread pool and a fused SwiGLU; ✅ register-tiled float32 and Q8_0 matmuls; ✅ tiled prefill attention; ✅ a thread pool that spins briefly before sleeping, and kernels really built at `-O3`. Prefill is about 3× faster, decoding 1.7×, and every golden hash is unchanged. See [Phase 13](docs/kernels.md#phase-13-tiles) and [benchmarks](docs/benchmarks.md) |
| 14. Lean and fast decoding ✅ | ✅ Speculative decoding that never changes a token (`--speculate`, `--draft-model`); ✅ 4-bit Q4_0 weights; ✅ one copy of the weights in memory; ✅ benchmarks after Phase 13. See [speculative decoding](docs/api.md#speculative-decoding), [Q4_0](docs/kernels.md#q4_0-quantisation) and [memory](docs/kernels.md#memory) |
| 15. Verifiable outputs ✅ | ✅ Receipts for every response that anyone can check later (#133); ✅ `dllm replay` to verify a receipt bit for bit (#134); ✅ a prompt cache that survives restarts with the same bits (`--persistent-cache`, #135); ✅ reproducible evaluation with `dllm eval` (#136). See [receipts](docs/receipts.md) and [evaluation](docs/evaluation.md) |
| 16. Reproducible agents ✅ | ✅ Agent transcripts of MCP tool loops (#139); ✅ offline replay of an agent run with the recorded tool results (#140); ✅ built-in deterministic tools (#141); ✅ receipt chains for whole conversations (#142). See [reproducible agents](docs/agents.md) |
| 17. Verifiable models ✅ | ✅ Model lineage in every model file (#145); ✅ training receipts that replay a fine-tune bit for bit (#146); ✅ byte-identical model files on every platform (#147); ✅ signed receipts, transcripts and models (#148). See [verifiable models](docs/provenance.md) |
| 18. Deterministic serving at scale ✅ | ✅ An exact response cache that answers repeated requests with the same bits (#151); ✅ identical requests in flight share one generation (#152); ✅ `dllm audit` to check that servers agree, and a self-audit at `GET /v1/audit` (#153). See [serving at scale](docs/serving.md) |
| 19. Reproducible model building ✅ | ✅ Exact model merges recorded in the lineage (`dllm merge`: linear, SLERP, TIES, #156); ✅ exports back to safetensors and GGUF that import to the same weights (`dllm export`, #157); ✅ distillation on a teacher's answers that replays bit for bit (`dllm distill`, #158). See [building models](docs/model-building.md) |
| 20. Independent verification ✅ | ✅ A second, independent implementation that gives the same bits (#161); ✅ `dllm verify --reference` to check a machine against it (#162); ✅ conformance vectors for other implementations (`dllm conformance`, #163); ✅ a written determinism specification (#164). See [specification](docs/specification.md) |
| 21. Deterministic decoding controls ✅ | ✅ Repetition, frequency and presence penalties, min-p and logit bias with a fixed order (#166); ✅ several choices per request, each with its solo bits (`n`, #167); ✅ regex-constrained output (#168); ✅ all of them in the specification, the reference implementation and the conformance vectors (#169). See [decoding controls](docs/api.md#decoding-controls) |
| 22. Reproducible batch jobs ✅ | ✅ `dllm batch` with byte-identical output at any concurrency (#171); ✅ exact resume of an interrupted batch (#172); ✅ the OpenAI Files and Batches API with content-derived ids (#173); ✅ a verifiable, signable digest per batch and `--verify` (#174). See [batch jobs](docs/batches.md) |
| 23. Long conversations in a fixed window ✅ | ✅ A context window that is enforced the same way everywhere (#176); ✅ deterministic truncation of the oldest turns (`truncation`, #177); ✅ rolling generation past a full window that equals a fresh run over the kept tokens (#178); ✅ the window rules in the specification, the reference implementation and `dllm verify --reference` (#179). See [long conversations](docs/api.md#long-conversations) |
| 24. Reproducible reasoning ✅ | ✅ The reasoning of thinking models separated from the answer in every API (#181); ✅ thinking switched on or off per request through the model's own template (#182); ✅ an exact thinking budget counted in tokens (#183); ✅ the rules in receipts, the specification, the reference implementation and `dllm verify --reference` (#184). See [reasoning](docs/api.md#reasoning) |
| 25. Reproducible preference tuning ✅ | ✅ DPO fine-tuning with bit-identical results (`dllm finetune --dpo`, #186); ✅ preference data in a fixed order with exact reference log-probabilities (#187); ✅ preference runs in checkpoints, training receipts and `dllm replay` (#188); ✅ preference accuracy in `dllm eval` (#189). See [preference tuning](docs/training.md#preference-tuning) |
| 26. Exact hybrid search and reranking ✅ | ✅ Exact BM25 lexical search over every index (#191); ✅ hybrid ranking with reciprocal rank fusion and a total order, in grounding and MCP search (#192); ✅ reranking with a language model, `dllm rerank` and `/v1/rerank` (#193); ✅ golden rankings and docs (#194). See [retrieval](docs/retrieval.md#lexical-and-hybrid-search) |
| 27. Verifiable text watermarks ✅ | ✅ A keyed green-list watermark in the sampler with exactly defined bits, in every API and the CLI (#196); ✅ exact detection from the text alone, `dllm watermark detect` and `/v1/watermark/detect` (#197); ✅ the watermark in receipts, the specification, the reference implementation and the conformance vectors (#198); ✅ golden values and docs (#199). See [watermarks](docs/watermarks.md) |
| 28. Reproducible scoring and voting ✅ | ✅ The OpenAI completions API, equal to `dllm generate` bit for bit (#201); ✅ exact prompt scoring with `echo` and `logprobs`, and `dllm score` (#202); ✅ self-consistency voting with a fixed rule, `vote` and `--vote` (#203); ✅ score and vote receipts, the specification, the reference check, golden values and docs (#204). See [scoring](docs/api.md#scoring) and [voting](docs/api.md#voting) |
| 29. Exact guided decoding ✅ | ✅ Classifier-free guidance with a negative prompt (#206); ✅ contrastive decoding against a smaller amateur model (#207); ✅ logit ensembles of models that share a tokenizer (#208); ✅ guided answers in receipts, the specification, the reference check, conformance vectors, golden values and docs (#209). See [guided decoding](docs/api.md#guided-decoding) |
| 30. Exact beam search ✅ | ✅ Beam search with a total order on hypotheses (#211); ✅ length penalties and fixed stopping rules (#212); ✅ n-best lists as choices (#213); ✅ beam receipts, the specification, the reference check, golden values and docs (#214). See [beam search](docs/api.md#beam-search) |
| 31. Exact JSON Schema constraints ✅ | ✅ String patterns and formats (#216); ✅ string lengths in code points (#217); ✅ integer bounds (#218); ✅ docs, tests and golden values (#219). See [structured output](docs/api.md#structured-output) |
| 32. Exact context-free grammars ✅ | ✅ GBNF grammars compiled to the byte pushdown automaton, recursion included and left recursion refused (#221); ✅ grammar output in every API and the CLI (#222); ✅ grammars in receipts, replay and the specification (#223); ✅ docs, examples and golden values (#224). See [grammars](docs/api.md#grammars) |
| 33. Exact numeric and object constraints ✅ | ✅ Bounds on decimal numbers, compared exactly (#226); ✅ `multipleOf` on integers and decimals (#227); ✅ property counts and typed map objects (#228); ✅ docs, tests and golden values (#229). See [structured output](docs/api.md#structured-output) |
| 34. Exact tuples, key names and unique items ✅ | ✅ Tuples with `prefixItems` (#231); ✅ constrained property names (#232); ✅ `uniqueItems` over finite item sets (#233); ✅ docs, tests and golden values (#234). See [structured output](docs/api.md#structured-output) |
| 35. Exact schema combinators ✅ | ✅ `allOf` over several schemas (#236); ✅ `not` and `if`/`then`/`else` over decidable conditions (#237); ✅ `patternProperties` (#238); ✅ `contains` with `minContains`/`maxContains` (#239); ✅ docs, tests and golden values (#240). See [structured output](docs/api.md#structured-output) |
| 36. Exact token healing ✅ | ✅ Token healing for raw prompts (#242); ✅ for chat prefills (#243); ✅ in receipts, the specification and the reference implementation (#244); ✅ docs, tests and golden values (#245). See [token healing](docs/api.md#token-healing) |
| 37. Exact fill-in-the-middle ✅ | ✅ Fill-in-the-middle prompts from the model's FIM tokens (#247); ✅ `suffix` on the completions API, Ollama and `dllm generate` (#248); ✅ in receipts, the specification and the reference implementation (#249); ✅ docs, tests and golden values (#250). See [fill-in-the-middle](docs/api.md#fill-in-the-middle) |
| 38. Exact length and stop controls ✅ | ✅ `min_tokens` and `ignore_eos` (#252); ✅ `stop_token_ids` and `include_stop_str_in_output` (#253); ✅ in receipts, the specification and the reference implementation (#254); ✅ docs, tests and golden values (#255). See [length and stop controls](docs/api.md#length-and-stop-controls) |

## Working with Claude Code

This repository is developed largely by Claude Code cloud sessions. [CLAUDE.md](CLAUDE.md) holds the build commands,
conventions and determinism rules those sessions follow.

## License

[Apache 2.0](LICENSE)
