# Benchmarks

How EtAlii.Dllm compares with the two engines most people run small models on, [Hugging Face
transformers](https://github.com/huggingface/transformers) (PyTorch) and [llama.cpp](https://github.com/ggml-org/llama.cpp),
on the same machine, for the same pinned open-weight models. Speed, memory, output quality and determinism are measured
side by side, the way the established LLM comparisons measure them.

First run: 2026-09-30. Harness: [`benchmarks/benchmark.py`](../benchmarks/benchmark.py); workflow:
[`Benchmark`](../.github/workflows/benchmark.yml). Raw results: [`docs/benchmarks/`](benchmarks/).

## Results (2026-09-30)

Main run: one GitHub Actions `ubuntu-latest` runner (AMD EPYC 7763, 4 vCPUs, 16 GB), all six models, all three
engines, 4 threads. Full tables: [2026-09-30-github-runner.md](benchmarks/2026-09-30-github-runner.md). A second run
of SmolLM2-135M in the Claude Code cloud container (Intel Xeon 2.8 GHz, 4 vCPUs):
[2026-09-30-cloud-xeon.md](benchmarks/2026-09-30-cloud-xeon.md). The GPU backend was not measured this time: the
only CUDA machine was not reachable during the run.

### Headlines

1. **Determinism: EtAlii.Dllm was bit-identical in all 48 checks** (6 models × float32/Q8_0 × 4 probes). The other
   engines were not:
   - transformers changes the logits when the same prompt is padded into a batch with other prompts, on every model
     (by up to 9.8e-5). On the cloud Xeon it also changed them with the thread count.
   - llama.cpp is reproducible run to run, across thread counts and micro-batch sizes, but `llama-server` gives a
     request different log-probabilities when it runs alongside others than when it runs alone, for every float32
     model. For Qwen3-0.6B and TinyLlama-1.1B the generated *text* changed. Its Q8_0 path stayed identical.
2. **Quality: the same numbers as the reference engines.** Float32 perplexity agrees with transformers and llama.cpp
   to within 0.001 for five of the six models, which is what an exact implementation of the same weights should give.
   Q8_0 costs at most about 0.3 % perplexity (KL divergence 0.001 to 0.0035, top token unchanged 96.6 to 98.0 % of the time),
   the same as llama.cpp's Q8_0. TinyLlama-1.1B is the exception, see below.
3. **Generation speed: faster than transformers, slower than llama.cpp.** Single-stream float32 decoding is 1.4 to
   1.5× transformers on every model and 0.73 to 0.89× llama.cpp from 0.5B parameters up (0.44× on the tiny
   SmolLM2). With Q8_0 weights llama.cpp is 2 to 3× faster (5.6× on SmolLM2).
4. **Prompt processing is the biggest gap**: 3.4 to 3.9× slower than llama.cpp and 2.6 to 3× slower than
   transformers.
5. **Concurrency trades speed for bits.** Eight simultaneous requests raise EtAlii.Dllm's total throughput 1.7 to 2× (float32)
   while llama.cpp's server gains 3 to 4×, and the time to first token grows with every request that joins. llama.cpp
   gets there by batching in a way that changes the output (headline 1); EtAlii.Dllm keeps every request's solo bits.

### Single-stream speed and memory

Tokens per second, 4 threads, mean of 5 runs (standard deviations are below 2 %; see the full tables).

| Model | Weights | EtAlii.Dllm pp128 | transformers pp128 | llama.cpp pp128 | EtAlii.Dllm tg64 | transformers tg64 | llama.cpp tg64 | EtAlii.Dllm peak RSS | transformers peak RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SmolLM2-135M | f32 | 139 | 412 | 539 | 31.9 | 22.1 | 71.7 | 1.3 GB | 1.5 GB |
| SmolLM2-135M | Q8_0 | 124 | | 539 | 40.7 | | 229.4 | 1.1 GB | |
| Qwen2.5-0.5B | f32 | 50.3 | 130 | 178 | 15.4 | 10.2 | 20.5 | 4.5 GB | 3.8 GB |
| Qwen2.5-0.5B | Q8_0 | 48.5 | | 178 | 25.0 | | 72.8 | 3.6 GB | |
| Qwen3-0.6B | f32 | 39.2 | 104 | 135 | 12.5 | 8.4 | 17.2 | 5.4 GB | 4.8 GB |
| Qwen3-0.6B | Q8_0 | 39.2 | | 139 | 20.8 | | 55.3 | 4.2 GB | |
| OLMo-2-1B | f32 | 19.0 | 50.1 | 64.6 | 7.2 | 5.1 | 8.1 | 11.0 GB | 8.9 GB |
| OLMo-2-1B | Q8_0 | 19.7 | | 62.5 | 14.4 | | 28.8 | 8.0 GB | |
| TinyLlama-1.1B | f32 | 20.4 | 60.4 | 69.9 | 8.3 | 6.0 | 9.8 | 8.4 GB | 6.7 GB |
| TinyLlama-1.1B | Q8_0 | 20.9 | | 68.0 | 15.9 | | 36.1 | 6.1 GB | |
| Qwen2.5-1.5B | f32 | 15.2 | 42.1 | 51.2 | 5.7 | 4.0 | 6.7 | 11.8 GB | 9.3 GB |
| Qwen2.5-1.5B | Q8_0 | 15.3 | | 50.5 | 10.9 | | 24.0 | 8.9 GB | |

transformers has no Q8_0 CPU path, so those cells are empty. llama-bench does not report memory.

### Perplexity (WikiText-2, 512-token chunks)

| Model | EtAlii.Dllm f32 | transformers f32 | llama.cpp f32 | EtAlii.Dllm Q8_0 | llama.cpp Q8_0 | EtAlii.Dllm Q8_0 KLD | EtAlii.Dllm Q8_0 top-1 same |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SmolLM2-135M | 16.742 | 16.742 | 16.743 | 16.796 | 16.789 | 0.0035 | 96.6 % |
| Qwen2.5-0.5B | 13.895 | 13.895 | 13.894 | 13.889 | 13.912 | 0.0022 | 97.7 % |
| Qwen3-0.6B | 19.087 | 19.087 | 19.086 | 19.063 | 19.087 | 0.0023 | 97.5 % |
| OLMo-2-1B | 14.095 | 14.095 | 14.096 | 14.090 | 14.078 | 0.0012 | 98.0 % |
| TinyLlama-1.1B | 16.259 | 18.098 | 17.785 | 16.278 | 17.804 | 0.0021 | 97.6 % |
| Qwen2.5-1.5B | 10.014 | 10.014 | 10.013 | 10.045 | 10.039 | 0.0021 | 97.5 % |

The TinyLlama row does not compare like with like: all three engines disagree with each other, which points at the
tokenisation of this long text (TinyLlama uses a SentencePiece-style Llama tokenizer; the harness lets each engine
tokenise with its own tokenizer, and llama.cpp also adds a BOS token), not at the model. Its logits match transformers
in `tests/test_reference_models.py`. Follow-up: compare the three token streams for this text.

### Determinism

| Probe | EtAlii.Dllm | transformers | llama.cpp |
| --- | --- | --- | --- |
| Repeat runs | identical (12/12) | identical (6/6) | identical (12/12) |
| Thread counts 1, 2, 4 | identical (12/12) | identical on the runner, **differs** on the cloud Xeon | identical (12/12) |
| Batch composition | identical (12/12) | **differs** on all 6 (max 7.6e-6 to 9.8e-5) | identical (12/12, micro-batch 512 vs 16) |
| Concurrent = solo | identical (12/12) | n/a | **differs** for all 6 f32 models (text changed for 2); Q8_0 identical |

### Where EtAlii.Dllm loses time

- **Prompt processing.** The float32 matmul computes each output with its own fixed-order `double` accumulation;
  llama.cpp and PyTorch use cache-blocked GEMM kernels that reuse loaded weights across many tokens. A blocked kernel
  that keeps the per-output summation order would close much of the gap without touching the bits.
- **Q8_0 decoding.** llama.cpp's Q8_0 is 2 to 3× faster at the same quality; it quantises the activations to 8 bits
  and multiplies in integers throughout.
- **Memory.** Peak float32 RSS is 13 to 27 % above transformers from 0.5B parameters up, because the memory-mapped source tensors and the
  packed copy used by the kernels are both resident.
- **Threads.** On this 4-vCPU runner every engine stops scaling after 2 threads, but EtAlii.Dllm's Q8_0 decoding of
  the smallest model even slows down with more threads: the per-token work is too small to split four ways.

These are measurements, not goals: the one hard requirement stays bit-identical output, and every row above that
says "identical" is the reason the others are slower.


## Methodology

There is no single standard for comparing inference engines, so the harness copies the parts of four well-known ones
that fit a CPU engine for small models.

| Aspect | Copied from | What we measure |
| --- | --- | --- |
| Throughput | llama.cpp [`llama-bench`](https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench) | `pp128`: prompt processing, tokens/s over a 128-token prompt. `tg64`: token generation, tokens/s over 64 greedy tokens. One warm-up, then the mean ± standard deviation of 5 repetitions, like llama-bench's `avg_ts ± stddev_ts`. Thread scaling at 1, 2 and 4 threads. |
| Latency | [MLPerf Inference](https://mlcommons.org/2024/03/mlperf-llama2-70b/) (TTFT, TPOT) and vLLM's [`benchmark_serving`](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) | Time per output token (p50/p99 over all decode steps). Time to first token (p50/p99) and aggregate output tokens/s with 1, 4 and 8 simultaneous greedy requests of 32 tokens. |
| Quality | llama.cpp [`llama-perplexity`](https://github.com/ggml-org/llama.cpp/tree/master/tools/perplexity) | Perplexity on the WikiText-2 test set in 512-token chunks, scoring the second half of each chunk, 8 chunks. For Q8_0: mean KL divergence from the float32 distribution and how often the top token agrees, llama.cpp's own quantisation metrics. |
| Determinism | Thinking Machines, [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) | Bit-for-bit comparison of logits and greedy tokens across repeated runs, thread counts, batch composition, and a request run alone versus alongside 3 or 7 others. |

Leaderboards such as the [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
rank models by task accuracy (MMLU-Pro, IFEval, ...). That compares models, not engines: every engine here runs the same
weights, so their accuracy differs only as much as their logits do, which perplexity and KL divergence capture directly.

### Setup

- **Models**: the pinned revisions from `tests/test_reference_models.py`: SmolLM2-135M-Instruct, Qwen2.5-0.5B-Instruct,
  Qwen3-0.6B, OLMo-2-0425-1B-Instruct, TinyLlama-1.1B-Chat-v1.0 and Qwen2.5-1.5B-Instruct.
- **EtAlii.Dllm**: `dllm import`, float32 and Q8_0 (`--quantize q8_0`), all cores, prompt cache off.
- **transformers**: float32 on CPU PyTorch, `torch.set_num_threads`, a KV cache for decoding.
- **llama.cpp**: tag `b11260` built from source with `GGML_NATIVE=ON`; models converted with its
  `convert_hf_to_gguf.py` to f32 and Q8_0. Speed from `llama-bench`, perplexity from `llama-perplexity`, raw logits for
  the determinism probes from `llama-cpp-python` 0.3.35, concurrency from `llama-server` with 8 parallel slots.
- **Text**: WikiText-2 test split as shipped with `pytorch/examples` (sha256 pinned in the harness). Every engine uses
  the model's own tokenizer; the token counts agree.
- Every configuration runs in a fresh process, so the peak resident memory (RSS) is its own.

### What each determinism probe does

| Probe | EtAlii.Dllm | transformers | llama.cpp |
| --- | --- | --- | --- |
| Repeat runs | 5 prefills and 5 decodes, hashes compared | same | two identical `Llama` instances |
| Thread counts | 1, 2 and 4 threads | `torch.set_num_threads` 1, 2, 4 | 1 vs 4 threads |
| Batch composition | the prompt alone vs stacked with prompts of 17, 91 and 3 tokens in one `forward_batch` | alone vs left-padded in a batch of 4 with an attention mask | micro-batch (`n_ubatch`) 512 vs 16 |
| Concurrent = solo | 1, 4, 8 threads calling the engine at once (continuous batching) vs each prompt alone; tokens compared | n/a (no server) | `llama-server -np 8`, streamed greedy requests; text and the chosen tokens' log-probabilities compared |

"identical" means every bit of the compared logits (or every token and log-probability) matched.

## Reproducing

```bash
pip install -e ".[dev,reference]" sentencepiece protobuf
python tests/test_reference_models.py ~/models          # downloads the pinned snapshots (huggingface.co)
git clone --depth 1 --branch b11260 https://github.com/ggml-org/llama.cpp ../llama.cpp
cmake -S ../llama.cpp -B ../llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_NATIVE=ON
cmake --build ../llama.cpp/build -j --target llama-bench llama-perplexity llama-server
CMAKE_ARGS="-DGGML_NATIVE=ON" pip install llama-cpp-python==0.3.35
python benchmarks/benchmark.py suite --snapshots ~/models --output results.json \
    --llamacpp ../llama.cpp/build/bin --llamacpp-source ../llama.cpp
python benchmarks/benchmark.py report results.json      # the Markdown tables
```

`--models`, `--reps`, `--pp`, `--tg`, `--ppl-chunks`, `--concurrency` and `--thread-sweep` change the run;
`--skip-transformers` or leaving out `--llamacpp` drop an engine. `benchmark.py dllm --model m.dllm --device cuda`
measures the GPU backend on its own. The `Benchmark` workflow runs the whole suite on a GitHub runner
(`workflow_dispatch`, or any PR that changes `benchmarks/`).
