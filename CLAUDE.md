# CLAUDE.md

Guidance for Claude Code sessions working in this repository.

## Project

EtAlii.Dllm is a deterministic LLM written from scratch in Python, with the numeric kernels in C++. The one
non-negotiable requirement: the same weights, prompt and context window produce bit-identical output on every run,
regardless of load, batching or thread scheduling, and (since Phase 10) on every supported machine: x86-64/arm64 CPUs,
every SIMD path and NVIDIA GPUs give the same bits (`docs/kernels.md#portable-determinism`). Vision and roadmap: `README.md`. Background: `docs/research/`.
Weights come from importing small open-weight models (Apache 2.0/MIT by default), not from training from scratch;
see `docs/research/model-import.md`. `huggingface.co` is blocked by the default cloud network policy.

## Commands

```bash
pip install -e ".[dev]"      # builds the C++ extension (scikit-build-core + nanobind); rerun after C++ changes
pytest                       # all tests, including golden-hash reproducibility tests
ruff check . && ruff format --check .    # CI runs both
dllm generate --prompt "Hi" --temperature 0.8 --seed 7
dllm-server                  # http://localhost:5080/v1/chat/completions
dllm-mcp                     # MCP over stdio
dllm-tools --tool calculator    # built-in deterministic tools as an MCP server (also `dllm chat --tool`)
docker build -t etalii-dllm .   # server image (Dockerfile, docker/entrypoint.sh); published by .github/workflows/docker.yml
```

Cloud sessions: `.claude/hooks/session-start.sh` creates `.venv`, installs the package in editable mode and puts
`.venv/bin` on the PATH. If commands are missing, run it (or `source .venv/bin/activate`).

## Layout

- `cpp/include/dllm/`: header-only C++ kernels (`random.hpp`, `math.hpp` transcendentals, `nn.hpp` matmul/norm/RoPE/
  attention, `grad.hpp` their gradients plus cross-entropy and AdamW, `quant.hpp` Q8_0, `interp.hpp` attention probabilities/cosine/Cholesky for interpretability, `parallel.hpp` the thread pool, `fpenv.hpp` the
  floating point environment every binding runs in,
  `simd.hpp` the per-machine AVX2/SSE2/NEON dispatch, `cuda.hpp` the CUDA backend; evaluation orders in `docs/kernels.md`);
  `cpp/kernels.cpp` binds them as `etalii_dllm._kernels`. `cpp/cuda/kernels.cu` holds the GPU kernels: CMake embeds it
  and `math.hpp` in the extension and `cuda.hpp` compiles them at run time with NVRTC (`--fmad=false`), loading the
  driver and NVRTC dynamically, so nothing CUDA is needed to build. GPU kernels must run the CPU kernel's exact order
  (one thread per output element, no atomics, no warp reductions) so they give the CPU bits (`tests/test_cuda.py`). `CMakeLists.txt` sets the floating point flags; never add `-ffast-math`, `-O3 -march=native`
  style reassociation flags or `-ffp-contract=fast`.
- `src/etalii_dllm/`: `tensor` (aligned float32 `Tensor`), `numerics` (thin wrappers over `_kernels`, fingerprints),
  `sampling`, `tokenization`, `models`, `generation`, `prompt_cache` (KV caches reused across requests),
  `batching` (concurrent generations share `forward_batch` steps), `chat`, `engine` (`DllmEngine`, the facade shared by every front
  end; `DLLM_MODEL`/`--model` selects a `model.dllm`), `transformer` (Llama/Qwen2/Qwen3 decoder + KV cache), `bpe`, `unicode` (Unicode tables pinned to one version) and
  `chat_template` (the model's own tokenizer and Jinja template), `cuda` (the GPU backend: NVRTC discovery, `CudaTensor`, device
  ops; `--device cuda`/`DLLM_DEVICE`), `architecture` (`TransformerConfig`), `modelfile`
  (the `model.dllm` container, `docs/model-format.md`), `importing` (safetensors/GGUF readers and `dllm import`),
  `training` (gradients, AdamW, data order, checkpoints and `dllm finetune`, `docs/training.md`), `lora` (LoRA
  adapters, always merged into the weights; the PEFT format), `grammar`
  (JSON-schema constrained decoding over a token trie), `tools` (tool calling in the Hermes `<tool_call>` format), `interpret/` (activation tracing via `LayerHook`, logit lens,
  embedding explorer, attention maps, steering vectors, ROME edits, sparse autoencoders; `dllm lens|attention|
  neighbours|steer|edit|sae`, `docs/interpretability.md`), `receipts` (generation receipts, receipt chains and `dllm replay`, `docs/receipts.md`), `transcripts` (agent transcripts
  and their offline replay), `builtin_tools` (deterministic calculator/files/documents tools, `dllm chat --tool`,
  `dllm-tools`; `docs/agents.md`), `signing` (Ed25519 signatures, `dllm sign`, `--sign-key`, `--trust`;
  `docs/provenance.md`; model lineage lives in `modelfile`, training receipts in `training/receipt.py`), `serving` (exact
  response cache, coalescing of identical in-flight requests, `--audit-every`/`GET /v1/audit`, `dllm audit`, `dllm cache`;
  `docs/serving.md`; `chat_stream(fresh=True)` bypasses both and is what replays use), `merging`/`exporting`
  (`dllm merge`, `dllm export` to safetensors/GGUF; distillation in `training/distill.py`; `docs/model-building.md`), `evaluation` (`dllm eval`,
  `docs/evaluation.md`), `retrieval` (exact document index, `dllm index`, chats
  grounded with `--index`, embedding-model pooling in `engine.embed`; `docs/retrieval.md`).
  Tests compare against the reference packages `gguf`, `safetensors` and `tokenizers` (dev dependencies).
- `src/etalii_dllm/server/` (OpenAI `app.py` and `responses_api.py`, Anthropic `anthropic_api.py`, Ollama `ollama_api.py`, the browser chat page `static/chat.html`
  served at `/`; `docs/api.md`), `mcp_server.py`,
  `cli.py`: thin front ends over `DllmEngine.chat_stream`. Keep logic out of them so all stay output-identical;
  non-streamed responses are assembled from the same event stream as streamed ones. `mcp_host.py` is the MCP client
  host (the model calls external MCP tools in a loop over `chat_stream`); both MCP directions: `docs/mcp.md`.
- `tests/`: pytest. `tests/golden_values.py` holds the reference hashes. `tests/test_reference_models.py` compares the
  real pinned SmolLM2-135M/Qwen2.5-0.5B/Qwen2.5-1.5B/Qwen3-0.6B imports with `transformers` (`DLLM_REFERENCE_MODELS=<dir>`, extra `reference`; the
  `Reference models` workflow downloads them); cached copies live in `/mnt/project-files/models` in cloud sessions.

## Determinism rules (inference and training code)

1. Never use `random`, `numpy.random`, `uuid4`, `time`/`datetime.now` or any ambient entropy in model code. Use
   `numerics.DeterministicRandom`.
2. Any reduction (sum, dot, matmul, softmax, norm, attention) is a C++ kernel with a fixed, documented order and a
   `double` accumulator for `float` data. No NumPy/BLAS reductions (`np.sum`, `@`, `np.dot`, `np.einsum`) in
   inference paths; elementwise NumPy operations are fine. Parallelism only with fixed partitioning (chunks from
   data size, combined in chunk order). SIMD is fine if the code path is fixed per machine. Threads split outputs, never a
   reduction (`parallel.hpp`); SIMD lanes hold different outputs, and every variant must equal the scalar reference
   (`linear_reference`) bit for bit on every thread count (`tests/test_batch_invariance.py`). Integer sums (Q8_0)
   are exact, so their order is free.
3. Use the portable kernels in `math.hpp` instead of `std::exp`, `math.exp` and `numpy.exp`; add new ones built from
   `+ - * /` and `sqrt`, with an accuracy test.
4. Kernels must not change strategy based on batch size or sequence length.
5. Sorting must use a total order (break ties on index/token id).
6. Text processing must not depend on set iteration order, `PYTHONHASHSEED`, locale or the installed Python/`regex`
   Unicode version: use `etalii_dllm.unicode` (pinned Unicode 15.1 tables) instead of `unicodedata`, `str.lower`
   and `regex` `\p{..}` classes.
7. API responses must not contain clock- or entropy-derived values; derive ids from content.

## Golden values

Reproducibility tests assert exact SHA-256 hashes, and every platform must give the same ones: CI runs them on Linux,
Windows and macOS, and the real-model `*_golden` tests on all five release platforms with every SIMD path
(`DLLM_ISA`). Never key golden values per platform; a platform that differs is a determinism bug. New kernels must
equal the scalar reference bit for bit on every path, so only exact operations may differ between paths (an FMA of a
product that is exact in double, integer sums in any order).
If a hash changes:
- unintentionally: it is a determinism bug, find it; do not update the constant.
- intentionally (new weights, new sampler semantics): update `tests/golden_values.py` and say why in the commit
  message. Get new values from the failing test output or from `dllm generate` (it prints the fingerprint to stderr).

## Conventions

- Python 3.11+, type hints everywhere, `from __future__ import annotations`; ruff config in `pyproject.toml`.
- C++17, header-only kernels in the `dllm` namespace; bindings stay thin.
- Work on a feature branch and open a draft PR against `develop` (the default branch).
- The version lives only in `src/etalii_dllm/__init__.py`. Releases (wheels for Linux/Windows/macOS tested against
  the golden hashes, GitHub Release, optional PyPI) come from `.github/workflows/release.yml`; see `docs/releasing.md`.
- Progress is tracked in the GitHub Project EtAlii.Dllm (see `docs/project-board.md`). Roadmap items are issues
  labelled `roadmap`/`phase-N` under "Phase N" milestones; a PR that finishes one says `Closes #n`, and a README
  roadmap change updates the matching issues/milestones.
