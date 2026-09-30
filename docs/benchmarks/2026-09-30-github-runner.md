# 2026-09-30, GitHub Actions runner

Full report of the [`Benchmark` workflow run](https://github.com/vrenken/etalii.dllm/actions/runs/36648377825) on
PR #106, as printed by `benchmarks/benchmark.py report`. Summary and discussion: [docs/benchmarks.md](../benchmarks.md).

Machine: AMD EPYC 7763 64-Core Processor, 4 cores, 15.6 GB, Linux-6.17.0-1022-azure-x86_64-with-glibc2.39; dllm 0.2.0
(avx2), llama.cpp b11260, llama-cpp-python 0.3.35, transformers/torch CPU from PyPI.

### Throughput (pp128, tg64, 4 threads, mean ± stdev of 5)

| Model | Engine | Weights | pp t/s | tg t/s | TPOT p50 / p99 ms | Load s | Peak RSS MB |
|---|---|---|---:|---:|---:|---:|---:|
| SmolLM2-135M | dllm | f32 | 139.2 ± 0.2 | 31.9 ± 0.2 | 31.2 / 33.8 | 0.81 | 1,337 |
| SmolLM2-135M | dllm | Q8_0 | 124.0 ± 0.6 | 40.7 ± 0.2 | 24.6 / 25.4 | 2.52 | 1,099 |
| SmolLM2-135M | transformers | f32 | 411.5 ± 2.4 | 22.1 ± 0.1 | 45.2 / 47.1 | 2.49 | 1,451 |
| SmolLM2-135M | llama.cpp | f32 | 539.2 ± 0.4 | 71.7 ± 0.9 | n/a | n/a | n/a |
| SmolLM2-135M | llama.cpp | Q8_0 | 539.0 ± 2.8 | 229.4 ± 1.8 | n/a | n/a | n/a |
| Qwen2.5-0.5B | dllm | f32 | 50.3 ± 0.1 | 15.4 ± 0.0 | 64.7 / 66.8 | 2.41 | 4,534 |
| Qwen2.5-0.5B | dllm | Q8_0 | 48.5 ± 0.2 | 25.0 ± 0.1 | 40.1 / 40.9 | 8.86 | 3,639 |
| Qwen2.5-0.5B | transformers | f32 | 130.1 ± 1.2 | 10.2 ± 0.1 | 97.6 / 101.0 | 2.29 | 3,835 |
| Qwen2.5-0.5B | llama.cpp | f32 | 178.0 ± 0.3 | 20.5 ± 0.2 | n/a | n/a | n/a |
| Qwen2.5-0.5B | llama.cpp | Q8_0 | 178.4 ± 1.0 | 72.8 ± 0.5 | n/a | n/a | n/a |
| Qwen3-0.6B | dllm | f32 | 39.2 ± 0.0 | 12.5 ± 0.1 | 79.9 / 87.3 | 2.85 | 5,401 |
| Qwen3-0.6B | dllm | Q8_0 | 39.2 ± 0.1 | 20.8 ± 0.0 | 48.0 / 51.2 | 10.63 | 4,191 |
| Qwen3-0.6B | transformers | f32 | 103.8 ± 0.1 | 8.4 ± 0.0 | 119.2 / 123.0 | 2.40 | 4,755 |
| Qwen3-0.6B | llama.cpp | f32 | 135.4 ± 0.5 | 17.2 ± 0.1 | n/a | n/a | n/a |
| Qwen3-0.6B | llama.cpp | Q8_0 | 138.7 ± 0.3 | 55.3 ± 1.7 | n/a | n/a | n/a |
| OLMo-2-1B | dllm | f32 | 19.0 ± 0.0 | 7.2 ± 0.0 | 138.6 / 144.9 | 10.60 | 11,036 |
| OLMo-2-1B | dllm | Q8_0 | 19.7 ± 0.1 | 14.4 ± 0.0 | 69.6 / 72.9 | 23.63 | 7,965 |
| OLMo-2-1B | transformers | f32 | 50.1 ± 0.4 | 5.1 ± 0.0 | 193.9 / 203.3 | 4.61 | 8,940 |
| OLMo-2-1B | llama.cpp | f32 | 64.6 ± 0.3 | 8.1 ± 0.1 | n/a | n/a | n/a |
| OLMo-2-1B | llama.cpp | Q8_0 | 62.5 ± 0.2 | 28.8 ± 0.3 | n/a | n/a | n/a |
| TinyLlama-1.1B | dllm | f32 | 20.4 ± 0.0 | 8.3 ± 0.0 | 120.0 / 126.0 | 4.23 | 8,437 |
| TinyLlama-1.1B | dllm | Q8_0 | 20.9 ± 0.0 | 15.9 ± 0.1 | 62.9 / 64.8 | 17.61 | 6,061 |
| TinyLlama-1.1B | transformers | f32 | 60.4 ± 0.5 | 6.0 ± 0.0 | 165.2 / 173.2 | 2.14 | 6,671 |
| TinyLlama-1.1B | llama.cpp | f32 | 69.9 ± 0.1 | 9.8 ± 0.1 | n/a | n/a | n/a |
| TinyLlama-1.1B | llama.cpp | Q8_0 | 68.0 ± 0.9 | 36.1 ± 0.1 | n/a | n/a | n/a |
| Qwen2.5-1.5B | dllm | f32 | 15.2 ± 0.0 | 5.7 ± 0.1 | 175.1 / 184.7 | 16.50 | 11,784 |
| Qwen2.5-1.5B | dllm | Q8_0 | 15.3 ± 0.1 | 10.9 ± 0.0 | 91.9 / 94.2 | 28.98 | 8,854 |
| Qwen2.5-1.5B | transformers | f32 | 42.1 ± 0.3 | 4.0 ± 0.0 | 249.6 / 255.1 | 3.21 | 9,279 |
| Qwen2.5-1.5B | llama.cpp | f32 | 51.2 ± 0.0 | 6.7 ± 0.0 | n/a | n/a | n/a |
| Qwen2.5-1.5B | llama.cpp | Q8_0 | 50.5 ± 0.1 | 24.0 ± 0.1 | n/a | n/a | n/a |

### Quality (WikiText-2 test, ctx 512, 8 chunks)

| Model | Engine | Weights | PPL | KLD vs f32 | Top-1 agreement |
|---|---|---|---:|---:|---:|
| SmolLM2-135M | dllm | f32 | 16.742 ± 1.078 |  |  |
| SmolLM2-135M | dllm | Q8_0 | 16.796 ± 1.079 | 0.00350 | 96.57 % |
| SmolLM2-135M | transformers | f32 | 16.742 ± 1.078 |  |  |
| SmolLM2-135M | llama.cpp | f32 | 16.743 ± 1.078 |  |  |
| SmolLM2-135M | llama.cpp | Q8_0 | 16.789 ± 1.078 |  |  |
| Qwen2.5-0.5B | dllm | f32 | 13.895 ± 0.881 |  |  |
| Qwen2.5-0.5B | dllm | Q8_0 | 13.889 ± 0.880 | 0.00221 | 97.65 % |
| Qwen2.5-0.5B | transformers | f32 | 13.895 ± 0.881 |  |  |
| Qwen2.5-0.5B | llama.cpp | f32 | 13.894 ± 0.881 |  |  |
| Qwen2.5-0.5B | llama.cpp | Q8_0 | 13.912 ± 0.882 |  |  |
| Qwen3-0.6B | dllm | f32 | 19.087 ± 1.421 |  |  |
| Qwen3-0.6B | dllm | Q8_0 | 19.063 ± 1.418 | 0.00229 | 97.45 % |
| Qwen3-0.6B | transformers | f32 | 19.087 ± 1.421 |  |  |
| Qwen3-0.6B | llama.cpp | f32 | 19.086 ± 1.421 |  |  |
| Qwen3-0.6B | llama.cpp | Q8_0 | 19.087 ± 1.420 |  |  |
| OLMo-2-1B | dllm | f32 | 14.095 ± 0.934 |  |  |
| OLMo-2-1B | dllm | Q8_0 | 14.090 ± 0.933 | 0.00116 | 98.04 % |
| OLMo-2-1B | transformers | f32 | 14.095 ± 0.934 |  |  |
| OLMo-2-1B | llama.cpp | f32 | 14.096 ± 0.934 |  |  |
| OLMo-2-1B | llama.cpp | Q8_0 | 14.078 ± 0.932 |  |  |
| TinyLlama-1.1B | dllm | f32 | 16.259 ± 1.124 |  |  |
| TinyLlama-1.1B | dllm | Q8_0 | 16.278 ± 1.126 | 0.00210 | 97.55 % |
| TinyLlama-1.1B | transformers | f32 | 18.098 ± 1.252 |  |  |
| TinyLlama-1.1B | llama.cpp | f32 | 17.785 ± 1.300 |  |  |
| TinyLlama-1.1B | llama.cpp | Q8_0 | 17.804 ± 1.302 |  |  |
| Qwen2.5-1.5B | dllm | f32 | 10.014 ± 0.596 |  |  |
| Qwen2.5-1.5B | dllm | Q8_0 | 10.045 ± 0.598 | 0.00208 | 97.45 % |
| Qwen2.5-1.5B | transformers | f32 | 10.014 ± 0.596 |  |  |
| Qwen2.5-1.5B | llama.cpp | f32 | 10.013 ± 0.596 |  |  |
| Qwen2.5-1.5B | llama.cpp | Q8_0 | 10.039 ± 0.597 |  |  |

### Determinism

| Model | Engine | Weights | Repeat runs | Thread counts | Batch composition | Concurrent = solo |
|---|---|---|---|---|---|---|
| SmolLM2-135M | dllm | f32 | identical | identical | identical | identical |
| SmolLM2-135M | dllm | Q8_0 | identical | identical | identical | identical |
| SmolLM2-135M | transformers | f32 | identical | identical | **differs** (max 2.53e-05) | n/a |
| SmolLM2-135M | llama.cpp | f32 | identical | identical | identical | **differs** (text same) |
| SmolLM2-135M | llama.cpp | Q8_0 | identical | identical | identical | identical |
| Qwen2.5-0.5B | dllm | f32 | identical | identical | identical | identical |
| Qwen2.5-0.5B | dllm | Q8_0 | identical | identical | identical | identical |
| Qwen2.5-0.5B | transformers | f32 | identical | identical | **differs** (max 3.72e-05) | n/a |
| Qwen2.5-0.5B | llama.cpp | f32 | identical | identical | identical | **differs** (text same) |
| Qwen2.5-0.5B | llama.cpp | Q8_0 | identical | identical | identical | identical |
| Qwen3-0.6B | dllm | f32 | identical | identical | identical | identical |
| Qwen3-0.6B | dllm | Q8_0 | identical | identical | identical | identical |
| Qwen3-0.6B | transformers | f32 | identical | identical | **differs** (max 2.15e-05) | n/a |
| Qwen3-0.6B | llama.cpp | f32 | identical | identical | identical | **differs** (text **differs**) |
| Qwen3-0.6B | llama.cpp | Q8_0 | identical | identical | identical | identical |
| OLMo-2-1B | dllm | f32 | identical | identical | identical | identical |
| OLMo-2-1B | dllm | Q8_0 | identical | identical | identical | identical |
| OLMo-2-1B | transformers | f32 | identical | identical | **differs** (max 1.72e-05) | n/a |
| OLMo-2-1B | llama.cpp | f32 | identical | identical | identical | **differs** (text same) |
| OLMo-2-1B | llama.cpp | Q8_0 | identical | identical | identical | identical |
| TinyLlama-1.1B | dllm | f32 | identical | identical | identical | identical |
| TinyLlama-1.1B | dllm | Q8_0 | identical | identical | identical | identical |
| TinyLlama-1.1B | transformers | f32 | identical | identical | **differs** (max 7.63e-06) | n/a |
| TinyLlama-1.1B | llama.cpp | f32 | identical | identical | identical | **differs** (text **differs**) |
| TinyLlama-1.1B | llama.cpp | Q8_0 | identical | identical | identical | identical |
| Qwen2.5-1.5B | dllm | f32 | identical | identical | identical | identical |
| Qwen2.5-1.5B | dllm | Q8_0 | identical | identical | identical | identical |
| Qwen2.5-1.5B | transformers | f32 | identical | identical | **differs** (max 9.78e-05) | n/a |
| Qwen2.5-1.5B | llama.cpp | f32 | identical | identical | identical | **differs** (text same) |
| Qwen2.5-1.5B | llama.cpp | Q8_0 | identical | identical | identical | identical |

### Thread scaling (tokens/s)

| Model | Engine | Weights | pp @1 / tg @1 | pp @2 / tg @2 | pp @4 / tg @4 |
|---|---|---|---:|---:|---:|
| SmolLM2-135M | dllm | f32 | 79.1 / 26.9 | 136.1 / 34.6 | 139.3 / 32.5 |
| SmolLM2-135M | dllm | Q8_0 | 77.7 / 52.2 | 122.8 / 46.8 | 124.6 / 40.9 |
| SmolLM2-135M | transformers | f32 | 282.6 / 25.3 | 472.2 / 25.1 | 414.3 / 22.2 |
| SmolLM2-135M | llama.cpp | f32 | 282.6 / 45.6 | 541.0 / 68.4 | 541.7 / 73.8 |
| SmolLM2-135M | llama.cpp | Q8_0 | 287.3 / 107.7 | 546.9 / 181.9 | 541.0 / 237.1 |
| Qwen2.5-0.5B | dllm | f32 | 26.4 / 8.9 | 48.2 / 14.7 | 50.4 / 15.5 |
| Qwen2.5-0.5B | dllm | Q8_0 | 28.3 / 19.7 | 48.6 / 26.1 | 47.9 / 25.1 |
| Qwen2.5-0.5B | transformers | f32 | 82.7 / 11.1 | 145.0 / 11.0 | 132.5 / 10.1 |
| Qwen2.5-0.5B | llama.cpp | f32 | 91.1 / 13.3 | 176.8 / 20.2 | 177.9 / 20.6 |
| Qwen2.5-0.5B | llama.cpp | Q8_0 | 94.5 / 32.8 | 186.7 / 57.5 | 179.6 / 73.1 |
| Qwen3-0.6B | dllm | f32 | 20.7 / 7.3 | 37.6 / 11.6 | 39.1 / 12.7 |
| Qwen3-0.6B | dllm | Q8_0 | 22.3 / 16.2 | 39.3 / 21.5 | 39.3 / 21.1 |
| Qwen3-0.6B | transformers | f32 | 67.0 / 9.2 | 114.7 / 9.3 | 104.0 / 8.4 |
| Qwen3-0.6B | llama.cpp | f32 | 70.9 / 11.1 | 138.9 / 16.9 | 135.5 / 16.9 |
| Qwen3-0.6B | llama.cpp | Q8_0 | 72.0 / 26.5 | 142.8 / 46.1 | 138.8 / 57.2 |
| OLMo-2-1B | dllm | f32 | 9.4 / 3.5 | 18.0 / 6.4 | 19.0 / 7.3 |
| OLMo-2-1B | dllm | Q8_0 | 10.8 / 8.7 | 20.1 / 14.0 | 19.6 / 14.5 |
| OLMo-2-1B | transformers | f32 | 31.9 / 5.5 | 54.0 / 5.5 | 49.5 / 5.2 |
| OLMo-2-1B | llama.cpp | f32 | 31.8 / 5.3 | 63.2 / 8.1 | 64.7 / 8.2 |
| OLMo-2-1B | llama.cpp | Q8_0 | 32.5 / 13.3 | 65.4 / 24.5 | 62.5 / 28.6 |
| TinyLlama-1.1B | dllm | f32 | 10.3 / 4.3 | 19.4 / 7.6 | 20.5 / 8.6 |
| TinyLlama-1.1B | dllm | Q8_0 | 11.5 / 10.3 | 21.1 / 15.8 | 20.9 / 16.0 |
| TinyLlama-1.1B | transformers | f32 | 40.5 / 6.8 | 70.9 / 6.6 | 61.7 / 6.0 |
| TinyLlama-1.1B | llama.cpp | f32 | 34.6 / 6.3 | 69.0 / 9.7 | 69.9 / 9.8 |
| TinyLlama-1.1B | llama.cpp | Q8_0 | 36.5 / 16.6 | 71.7 / 30.7 | 68.2 / 36.0 |
| Qwen2.5-1.5B | dllm | f32 | 7.6 / 2.9 | 14.4 / 5.0 | 15.2 / 5.7 |
| Qwen2.5-1.5B | dllm | Q8_0 | 8.5 / 7.0 | 15.6 / 10.7 | 15.4 / 10.9 |
| Qwen2.5-1.5B | transformers | f32 | 27.3 / 4.3 | 47.3 / 4.3 | 42.6 / 4.1 |
| Qwen2.5-1.5B | llama.cpp | f32 | 25.4 / 4.3 | 50.0 / 6.7 | 51.0 / 6.8 |
| Qwen2.5-1.5B | llama.cpp | Q8_0 | 27.0 / 11.3 | 53.5 / 19.8 | 50.7 / 24.5 |

### Concurrent greedy requests (32 tokens each)

| Model | Engine | Weights | Users | Output t/s | TTFT p50 / p99 ms |
|---|---|---|---:|---:|---:|
| SmolLM2-135M | dllm | f32 | 1 | 31.4 | 69 / 69 |
| SmolLM2-135M | dllm | f32 | 4 | 48.6 | 250 / 250 |
| SmolLM2-135M | dllm | f32 | 8 | 62.3 | 452 / 452 |
| SmolLM2-135M | dllm | Q8_0 | 1 | 38.3 | 76 / 76 |
| SmolLM2-135M | dllm | Q8_0 | 4 | 51.9 | 278 / 278 |
| SmolLM2-135M | dllm | Q8_0 | 8 | 62.4 | 530 / 530 |
| SmolLM2-135M | llama.cpp | f32 | 1 | 65.2 | 29 / 29 |
| SmolLM2-135M | llama.cpp | f32 | 4 | 100.7 | 117 / 118 |
| SmolLM2-135M | llama.cpp | f32 | 8 | 206.5 | 193 / 196 |
| SmolLM2-135M | llama.cpp | Q8_0 | 1 | 178.7 | 21 / 21 |
| SmolLM2-135M | llama.cpp | Q8_0 | 4 | 177.7 | 85 / 86 |
| SmolLM2-135M | llama.cpp | Q8_0 | 8 | 213.6 | 98 / 183 |
| Qwen2.5-0.5B | dllm | f32 | 1 | 14.8 | 177 / 177 |
| Qwen2.5-0.5B | dllm | f32 | 4 | 22.5 | 662 / 663 |
| Qwen2.5-0.5B | dllm | f32 | 8 | 26.3 | 1,306 / 1,307 |
| Qwen2.5-0.5B | dllm | Q8_0 | 1 | 22.7 | 177 / 177 |
| Qwen2.5-0.5B | dllm | Q8_0 | 4 | 27.0 | 691 / 691 |
| Qwen2.5-0.5B | dllm | Q8_0 | 8 | 29.2 | 1,345 / 1,345 |
| Qwen2.5-0.5B | llama.cpp | f32 | 1 | 19.4 | 75 / 75 |
| Qwen2.5-0.5B | llama.cpp | f32 | 4 | 35.2 | 323 / 326 |
| Qwen2.5-0.5B | llama.cpp | f32 | 8 | 72.9 | 412 / 569 |
| Qwen2.5-0.5B | llama.cpp | Q8_0 | 1 | 54.8 | 66 / 66 |
| Qwen2.5-0.5B | llama.cpp | Q8_0 | 4 | 56.8 | 247 / 249 |
| Qwen2.5-0.5B | llama.cpp | Q8_0 | 8 | 79.2 | 440 / 446 |
| Qwen3-0.6B | dllm | f32 | 1 | 12.2 | 211 / 211 |
| Qwen3-0.6B | dllm | f32 | 4 | 18.3 | 837 / 837 |
| Qwen3-0.6B | dllm | f32 | 8 | 21.6 | 1,569 / 1,569 |
| Qwen3-0.6B | dllm | Q8_0 | 1 | 18.9 | 210 / 210 |
| Qwen3-0.6B | dllm | Q8_0 | 4 | 22.5 | 809 / 810 |
| Qwen3-0.6B | dllm | Q8_0 | 8 | 24.2 | 1,607 / 1,607 |
| Qwen3-0.6B | llama.cpp | f32 | 1 | 16.1 | 101 / 101 |
| Qwen3-0.6B | llama.cpp | f32 | 4 | 28.9 | 410 / 410 |
| Qwen3-0.6B | llama.cpp | f32 | 8 | 59.9 | 636 / 642 |
| Qwen3-0.6B | llama.cpp | Q8_0 | 1 | 46.5 | 76 / 76 |
| Qwen3-0.6B | llama.cpp | Q8_0 | 4 | 50.8 | 221 / 339 |
| Qwen3-0.6B | llama.cpp | Q8_0 | 8 | 67.4 | 612 / 620 |
| OLMo-2-1B | dllm | f32 | 1 | 6.8 | 444 / 444 |
| OLMo-2-1B | dllm | f32 | 4 | 10.1 | 1,812 / 1,812 |
| OLMo-2-1B | dllm | f32 | 8 | 11.6 | 3,367 / 3,368 |
| OLMo-2-1B | dllm | Q8_0 | 1 | 12.5 | 407 / 407 |
| OLMo-2-1B | dllm | Q8_0 | 4 | 13.6 | 1,622 / 1,622 |
| OLMo-2-1B | dllm | Q8_0 | 8 | 14.0 | 3,151 / 3,151 |
| OLMo-2-1B | llama.cpp | f32 | 1 | 8.0 | 191 / 191 |
| OLMo-2-1B | llama.cpp | f32 | 4 | 14.4 | 683 / 685 |
| OLMo-2-1B | llama.cpp | f32 | 8 | 32.3 | 1,576 / 1,582 |
| OLMo-2-1B | llama.cpp | Q8_0 | 1 | 25.2 | 147 / 147 |
| OLMo-2-1B | llama.cpp | Q8_0 | 4 | 29.2 | 561 / 562 |
| OLMo-2-1B | llama.cpp | Q8_0 | 8 | 37.6 | 1,195 / 1,201 |
| TinyLlama-1.1B | dllm | f32 | 1 | 8.0 | 394 / 394 |
| TinyLlama-1.1B | dllm | f32 | 4 | 11.4 | 1,710 / 1,710 |
| TinyLlama-1.1B | dllm | f32 | 8 | 13.3 | 3,277 / 3,280 |
| TinyLlama-1.1B | dllm | Q8_0 | 1 | 13.8 | 377 / 377 |
| TinyLlama-1.1B | dllm | Q8_0 | 4 | 14.6 | 1,633 / 1,633 |
| TinyLlama-1.1B | dllm | Q8_0 | 8 | 15.4 | 3,267 / 3,268 |
| TinyLlama-1.1B | llama.cpp | f32 | 1 | 9.6 | 193 / 193 |
| TinyLlama-1.1B | llama.cpp | f32 | 4 | 15.9 | 949 / 952 |
| TinyLlama-1.1B | llama.cpp | f32 | 8 | 30.6 | 1,221 / 1,804 |
| TinyLlama-1.1B | llama.cpp | Q8_0 | 1 | 30.8 | 153 / 153 |
| TinyLlama-1.1B | llama.cpp | Q8_0 | 4 | 34.9 | 610 / 611 |
| TinyLlama-1.1B | llama.cpp | Q8_0 | 8 | 42.2 | 1,282 / 1,284 |
| Qwen2.5-1.5B | dllm | f32 | 1 | 5.4 | 560 / 560 |
| Qwen2.5-1.5B | dllm | f32 | 4 | 8.0 | 2,258 / 2,258 |
| Qwen2.5-1.5B | dllm | f32 | 8 | 9.2 | 4,315 / 4,315 |
| Qwen2.5-1.5B | dllm | Q8_0 | 1 | 9.5 | 524 / 524 |
| Qwen2.5-1.5B | dllm | Q8_0 | 4 | 10.5 | 2,086 / 2,086 |
| Qwen2.5-1.5B | dllm | Q8_0 | 8 | 10.9 | 4,155 / 4,155 |
| Qwen2.5-1.5B | llama.cpp | f32 | 1 | 6.3 | 229 / 229 |
| Qwen2.5-1.5B | llama.cpp | f32 | 4 | 11.9 | 875 / 875 |
| Qwen2.5-1.5B | llama.cpp | f32 | 8 | 26.4 | 1,752 / 1,758 |
| Qwen2.5-1.5B | llama.cpp | Q8_0 | 1 | 20.6 | 180 / 180 |
| Qwen2.5-1.5B | llama.cpp | Q8_0 | 4 | 23.3 | 753 / 755 |
| Qwen2.5-1.5B | llama.cpp | Q8_0 | 8 | 30.9 | 1,358 / 1,364 |
