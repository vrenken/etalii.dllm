# Testing

The whole suite is `pytest`. It needs no network, no model downloads and no GPU: tests build small synthetic models,
tokenizers and files in memory, and the parts that need more skip themselves (see [What is not covered](#what-is-not-covered)).

```bash
pip install -e ".[dev]"
pytest                         # everything
pytest --cov                   # with the coverage report and the floor CI enforces
pytest tests/test_bpe.py -q    # one area
```

## What the suite checks

| Kind | Where | What it proves |
|---|---|---|
| Unit | `test_kernels.py`, `test_numerics.py` | every C++ kernel against a float64 reference, portable `exp`/`log`/`sqrt` accuracy, fingerprints |
| Unit | `test_bpe*.py`, `test_chat_template.py` | the tokenizer against the `tokenizers` reference, byte fallback, special tokens, malformed files |
| Unit | `test_grammar*.py`, `test_tools.py` | JSON-schema constrained decoding and Hermes tool-call parsing |
| Unit | `test_edge_cases.py` | `Tensor`, sampling options, tool-call parsing, prompt cache and MCP host edge cases |
| Unit | `test_import*.py`, `test_modelfile.py` | safetensors/GGUF readers against the `safetensors`/`gguf` reference writers, `model.dllm` round trips, every malformed-input error |
| Determinism | `test_reproducibility.py`, `golden_values.py` | exact SHA-256 hashes of logits, samples and generations (CI runs them on Linux, Windows and macOS) |
| Determinism | `test_batch_invariance.py`, `test_batching.py`, `test_prompt_cache.py` | batching, thread count, SIMD path and cache reuse never change a bit |
| Model | `test_transformer.py`, `test_generation.py`, `test_lora*.py`, `test_training*.py`, `test_engine_edge_cases.py` | the decoder, KV cache, sampling loop, LoRA merge, gradients (finite differences), AdamW and checkpoints |
| Integration | `test_server.py`, `test_responses_api.py`, `test_anthropic_api.py`, `test_ollama_api.py`, `test_api_errors.py` | the HTTP server driven by the official `openai`, `anthropic` and `ollama` clients, streamed and non-streamed output identical, each API's error shape |
| Integration | `test_cli.py` | the `dllm` commands end to end, including their error exits |
| Integration | `test_mcp_server.py`, `test_mcp_host.py` | both MCP directions through the official `mcp` SDK |
| Integration | `test_engine_import.py` | import a model file, load it in `DllmEngine` and generate |
| End to end | `.github/workflows/docker.yml` | the Docker image serves `/v1/chat/completions` twice with identical answers and the chat page |
| Reference | `test_reference_models.py` (`reference.yml`) | real SmolLM2/Qwen2.5/Qwen3 imports match Hugging Face `transformers` |
| GPU | `test_cuda.py` | NVRTC compiles the kernels (CI, `cuda-install` job); CPU and GPU give the same bits (needs a GPU) |

## Coverage

CI measures line and branch coverage of `src/etalii_dllm` on the Ubuntu / Python 3.12 job (`pytest --cov`) and fails
below the floor in `[tool.coverage.report] fail_under` in `pyproject.toml`. Raise the floor when coverage goes up;
never lower it to get a PR through. New code comes with tests: a validation branch gets a `pytest.raises` with the
message, a new endpoint gets a test through the official client, a new kernel gets a reference comparison and a
batch-invariance case.

The C++ kernels have no coverage number of their own: every bound function is called by `test_kernels.py`,
`test_numerics.py` and `test_batch_invariance.py`, which compare against references for all SIMD paths and thread
counts on the CI machines.

## What is not covered

- **The GPU path.** `cuda.py` and the `*_gpu` branches of `transformer.py` run only with an NVIDIA GPU. CI checks that
  NVRTC compiles the kernels, but bit-exactness against the CPU runs only where a GPU is present
  (`pytest tests/test_cuda.py`). A self-hosted GPU runner would close this gap.
- **Real models.** `test_reference_models.py` needs downloaded weights (`DLLM_REFERENCE_MODELS`) and runs in its own
  workflow, not on every PR.
