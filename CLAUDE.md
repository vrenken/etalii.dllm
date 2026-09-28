# CLAUDE.md

Guidance for Claude Code sessions working in this repository.

## Project

EtAlii.Dllm is a deterministic LLM written from scratch in Python, with the numeric kernels in C++. The one
non-negotiable requirement: on the same hardware, the same weights, prompt and context window produce bit-identical
output on every run, regardless of load, batching or thread scheduling. Identical output across different hardware
is not required. Vision and roadmap: `README.md`. Background: `docs/research/`.
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
```

Cloud sessions: `.claude/hooks/session-start.sh` creates `.venv`, installs the package in editable mode and puts
`.venv/bin` on the PATH. If commands are missing, run it (or `source .venv/bin/activate`).

## Layout

- `cpp/include/dllm/`: header-only C++ kernels (`random.hpp`, `math.hpp` transcendentals, `nn.hpp` matmul/norm/RoPE/
  attention, `grad.hpp` their gradients plus cross-entropy and AdamW; evaluation orders in `docs/kernels.md`); `cpp/kernels.cpp` binds them as
  `etalii_dllm._kernels`. `CMakeLists.txt` sets the floating point flags; never add `-ffast-math`, `-O3 -march=native`
  style reassociation flags or `-ffp-contract=fast`.
- `src/etalii_dllm/`: `tensor` (aligned float32 `Tensor`), `numerics` (thin wrappers over `_kernels`, fingerprints),
  `sampling`, `tokenization`, `models`, `generation`, `chat`, `engine` (`DllmEngine`, the facade shared by every front
  end; `DLLM_MODEL`/`--model` selects a `model.dllm`), `transformer` (Llama/Qwen2 decoder + KV cache), `bpe` and
  `chat_template` (the model's own tokenizer and Jinja template), `architecture` (`TransformerConfig`), `modelfile`
  (the `model.dllm` container, `docs/model-format.md`), `importing` (safetensors/GGUF readers and `dllm import`),
  `training` (gradients, AdamW, data order, checkpoints and `dllm finetune`, `docs/training.md`), `grammar`
  (JSON-schema constrained decoding over a token trie), `tools` (tool calling in the Hermes `<tool_call>` format).
  Tests compare against the reference packages `gguf`, `safetensors` and `tokenizers` (dev dependencies).
- `src/etalii_dllm/server/` (OpenAI `app.py`, Anthropic `anthropic_api.py`; `docs/api.md`), `mcp_server.py`,
  `cli.py`: thin front ends over `DllmEngine.chat_stream`. Keep logic out of them so all stay output-identical;
  non-streamed responses are assembled from the same event stream as streamed ones.
- `tests/`: pytest. `tests/golden_values.py` holds the reference hashes. `tests/test_reference_models.py` compares the
  real pinned SmolLM2/Qwen2.5 imports with `transformers` (`DLLM_REFERENCE_MODELS=<dir>`, extra `reference`; the
  `Reference models` workflow downloads them); cached copies live in `/mnt/project-files/models` in cloud sessions.

## Determinism rules (inference and training code)

1. Never use `random`, `numpy.random`, `uuid4`, `time`/`datetime.now` or any ambient entropy in model code. Use
   `numerics.DeterministicRandom`.
2. Any reduction (sum, dot, matmul, softmax, norm, attention) is a C++ kernel with a fixed, documented order and a
   `double` accumulator for `float` data. No NumPy/BLAS reductions (`np.sum`, `@`, `np.dot`, `np.einsum`) in
   inference paths; elementwise NumPy operations are fine. Parallelism only with fixed partitioning (chunks from
   data size, combined in chunk order). SIMD is fine if the code path is fixed per machine.
3. Prefer the portable kernels in `math.hpp` over `std::exp`, `math.exp` and `numpy.exp`; add new ones built from
   `+ - * /` and `sqrt`, with an accuracy test.
4. Kernels must not change strategy based on batch size or sequence length.
5. Sorting must use a total order (break ties on index/token id).
6. Text processing must not depend on set iteration order, `PYTHONHASHSEED` or locale.
7. API responses must not contain clock- or entropy-derived values; derive ids from content.

## Golden values

Reproducibility tests assert exact SHA-256 hashes. CI runs them on Linux, Windows and macOS; today all agree, but if a
hardware-specific kernel makes them diverge, key the golden values per platform rather than forcing portability.
If a hash changes:
- unintentionally: it is a determinism bug, find it; do not update the constant.
- intentionally (new weights, new sampler semantics): update `tests/golden_values.py` and say why in the commit
  message. Get new values from the failing test output or from `dllm generate` (it prints the fingerprint to stderr).

## Conventions

- Python 3.11+, type hints everywhere, `from __future__ import annotations`; ruff config in `pyproject.toml`.
- C++17, header-only kernels in the `dllm` namespace; bindings stay thin.
- Work on a feature branch and open a draft PR against `develop` (the default branch).
- Progress is tracked in the GitHub Project EtAlii.Dllm (see `docs/project-board.md`). Roadmap items are issues
  labelled `roadmap`/`phase-N` under "Phase N" milestones; a PR that finishes one says `Closes #n`, and a README
  roadmap change updates the matching issues/milestones.
