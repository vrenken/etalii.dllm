# 2026-10-01, GitHub Actions runner (after Phase 13)

Full report of the [`Benchmark` workflow run](https://github.com/vrenken/etalii.dllm/actions/runs/36842595615) on
PR #127 (register-tiled kernels, commit d38d2e0f2d), as printed by `benchmarks/benchmark.py report`. Summary and
discussion: [docs/benchmarks.md](../benchmarks.md). The previous run: [2026-09-30](2026-09-30-github-runner.md).

Machine: AMD EPYC 7763 64-Core Processor, 4 cores, 15.6 GB, Linux-6.17.0-1022-azure-x86_64-with-glibc2.39; dllm 0.2.0
(avx2), llama.cpp b11260, llama-cpp-python 0.3.35, transformers/torch CPU from PyPI.

### Throughput (pp128, tg64, 4 threads, mean ± stdev of 5)

| Model | Engine | Weights | pp t/s | tg t/s | TPOT p50 / p99 ms | Load s | Peak RSS MB |
|---|---|---|---:|---:|---:|---:|---:|
| SmolLM2-135M | dllm | f32 | 308.9 ± 0.8 | 44.5 ± 0.2 | 22.3 / 27.6 | 0.79 | 1,342 |
| SmolLM2-135M | dllm | Q8_0 | 262.2 ± 0.5 | 56.9 ± 0.3 | 17.5 / 18.6 | 3.14 | 1,106 |
| SmolLM2-135M | transformers | f32 | 410.1 ± 5.7 | 21.4 ± 0.1 | 46.6 / 50.8 | 2.54 | 1,453 |
| SmolLM2-135M | llama.cpp | f32 | 537.8 ± 0.3 | 71.7 ± 0.4 | n/a | n/a | n/a |
| SmolLM2-135M | llama.cpp | Q8_0 | 537.5 ± 3.9 | 224.0 ± 2.5 | n/a | n/a | n/a |
| Qwen2.5-0.5B | dllm | f32 | 109.0 ± 0.3 | 17.1 ± 0.1 | 58.0 / 63.1 | 2.64 | 4,558 |
| Qwen2.5-0.5B | dllm | Q8_0 | 112.9 ± 0.1 | 28.0 ± 0.2 | 35.5 / 40.6 | 11.15 | 3,628 |
| Qwen2.5-0.5B | transformers | f32 | 128.9 ± 1.1 | 9.6 ± 0.1 | 103.4 / 111.1 | 2.38 | 3,830 |
| Qwen2.5-0.5B | llama.cpp | f32 | 177.0 ± 0.6 | 20.0 ± 0.3 | n/a | n/a | n/a |
| Qwen2.5-0.5B | llama.cpp | Q8_0 | 178.8 ± 0.4 | 69.8 ± 0.6 | n/a | n/a | n/a |
| Qwen3-0.6B | dllm | f32 | 79.8 ± 0.4 | 14.1 ± 0.1 | 70.4 / 77.7 | 2.93 | 5,342 |
| Qwen3-0.6B | dllm | Q8_0 | 88.7 ± 0.2 | 23.2 ± 0.4 | 42.7 / 55.1 | 13.43 | 4,167 |
| Qwen3-0.6B | transformers | f32 | 100.1 ± 1.3 | 8.0 ± 0.0 | 125.5 / 134.1 | 2.65 | 4,755 |
| Qwen3-0.6B | llama.cpp | f32 | 132.3 ± 2.6 | 16.5 ± 0.1 | n/a | n/a | n/a |
| Qwen3-0.6B | llama.cpp | Q8_0 | 138.5 ± 1.0 | 55.2 ± 0.2 | n/a | n/a | n/a |
| OLMo-2-1B | dllm | f32 | 36.9 ± 0.3 | 6.9 ± 0.0 | 143.7 / 153.7 | 16.08 | 10,731 |
| OLMo-2-1B | dllm | Q8_0 | 48.6 ± 0.1 | 16.6 ± 0.0 | 60.0 / 64.1 | 29.16 | 8,042 |
| OLMo-2-1B | transformers | f32 | 49.8 ± 0.3 | 4.9 ± 0.1 | 202.4 / 218.9 | 3.18 | 8,939 |
| OLMo-2-1B | llama.cpp | f32 | 64.1 ± 0.3 | 7.9 ± 0.1 | n/a | n/a | n/a |
| OLMo-2-1B | llama.cpp | Q8_0 | 62.5 ± 0.1 | 28.6 ± 0.2 | n/a | n/a | n/a |
| TinyLlama-1.1B | dllm | f32 | 41.5 ± 0.1 | 9.0 ± 0.0 | 110.6 / 120.0 | 4.50 | 8,462 |
| TinyLlama-1.1B | dllm | Q8_0 | 51.2 ± 0.1 | 18.8 ± 0.1 | 53.1 / 56.6 | 22.37 | 6,055 |
| TinyLlama-1.1B | transformers | f32 | 60.5 ± 0.8 | 5.7 ± 0.0 | 173.9 / 181.7 | 2.56 | 6,671 |
| TinyLlama-1.1B | llama.cpp | f32 | 69.5 ± 0.1 | 9.7 ± 0.1 | n/a | n/a | n/a |
| TinyLlama-1.1B | llama.cpp | Q8_0 | 67.9 ± 0.1 | 34.1 ± 0.3 | n/a | n/a | n/a |
| Qwen2.5-1.5B | dllm | f32 | 31.1 ± 0.1 | 6.0 ± 0.1 | 167.2 / 185.5 | 17.92 | 12,488 |
| Qwen2.5-1.5B | dllm | Q8_0 | 38.0 ± 0.0 | 12.5 ± 0.1 | 79.9 / 87.5 | 33.73 | 8,817 |
| Qwen2.5-1.5B | transformers | f32 | 41.8 ± 0.2 | 3.9 ± 0.0 | 257.2 / 273.3 | 5.65 | 9,263 |
| Qwen2.5-1.5B | llama.cpp | f32 | 50.3 ± 0.1 | 6.6 ± 0.0 | n/a | n/a | n/a |
| Qwen2.5-1.5B | llama.cpp | Q8_0 | 50.3 ± 0.1 | 23.6 ± 0.2 | n/a | n/a | n/a |

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
| SmolLM2-135M | dllm | f32 | 161.4 / 32.0 | 282.5 / 45.0 | 306.7 / 45.0 |
| SmolLM2-135M | dllm | Q8_0 | 132.0 / 42.6 | 248.2 / 58.0 | 261.3 / 57.4 |
| SmolLM2-135M | transformers | f32 | 279.8 / 24.2 | 446.2 / 24.2 | 408.8 / 21.5 |
| SmolLM2-135M | llama.cpp | f32 | 278.5 / 43.5 | 537.1 / 67.3 | 539.6 / 73.0 |
| SmolLM2-135M | llama.cpp | Q8_0 | 286.8 / 103.5 | 553.5 / 170.8 | 533.6 / 221.6 |
| Qwen2.5-0.5B | dllm | f32 | 53.9 / 10.3 | 102.3 / 15.5 | 108.1 / 17.1 |
| Qwen2.5-0.5B | dllm | Q8_0 | 52.8 / 16.1 | 105.0 / 25.7 | 113.2 / 28.3 |
| Qwen2.5-0.5B | transformers | f32 | 81.9 / 10.4 | 138.1 / 10.5 | 130.5 / 9.9 |
| Qwen2.5-0.5B | llama.cpp | f32 | 90.4 / 12.2 | 174.9 / 19.0 | 177.2 / 19.7 |
| Qwen2.5-0.5B | llama.cpp | Q8_0 | 94.4 / 30.6 | 186.1 / 55.4 | 176.3 / 70.4 |
| Qwen3-0.6B | dllm | f32 | 42.4 / 8.6 | 80.7 / 13.5 | 79.5 / 14.0 |
| Qwen3-0.6B | dllm | Q8_0 | 44.1 / 13.9 | 82.0 / 21.8 | 88.6 / 23.7 |
| Qwen3-0.6B | transformers | f32 | 64.3 / 8.7 | 112.0 / 8.7 | 101.3 / 7.9 |
| Qwen3-0.6B | llama.cpp | f32 | 70.0 / 10.1 | 137.3 / 15.5 | 130.6 / 16.5 |
| Qwen3-0.6B | llama.cpp | Q8_0 | 72.1 / 25.3 | 141.6 / 44.6 | 138.8 / 56.1 |
| OLMo-2-1B | dllm | f32 | 18.4 / 4.0 | 36.2 / 6.3 | 36.6 / 6.9 |
| OLMo-2-1B | dllm | Q8_0 | 23.1 / 9.2 | 45.1 / 14.9 | 48.3 / 16.4 |
| OLMo-2-1B | transformers | f32 | 31.3 / 5.1 | 53.9 / 5.2 | 49.4 / 4.9 |
| OLMo-2-1B | llama.cpp | f32 | 31.8 / 5.0 | 62.5 / 7.6 | 64.1 / 8.1 |
| OLMo-2-1B | llama.cpp | Q8_0 | 32.9 / 12.2 | 65.9 / 23.0 | 62.4 / 28.6 |
| TinyLlama-1.1B | dllm | f32 | 20.8 / 5.1 | 40.4 / 8.3 | 41.5 / 9.1 |
| TinyLlama-1.1B | dllm | Q8_0 | 23.9 / 10.7 | 47.0 / 16.9 | 51.1 / 18.8 |
| TinyLlama-1.1B | transformers | f32 | 40.0 / 6.2 | 70.1 / 6.2 | 61.0 / 5.7 |
| TinyLlama-1.1B | llama.cpp | f32 | 34.6 / 6.0 | 67.5 / 9.5 | 69.6 / 9.9 |
| TinyLlama-1.1B | llama.cpp | Q8_0 | 36.0 / 15.3 | 72.1 / 28.6 | 68.0 / 34.9 |
| Qwen2.5-1.5B | dllm | f32 | 15.3 / 3.4 | 29.6 / 5.5 | 31.3 / 6.0 |
| Qwen2.5-1.5B | dllm | Q8_0 | 17.5 / 7.0 | 34.7 / 11.2 | 37.9 / 12.5 |
| Qwen2.5-1.5B | transformers | f32 | 27.2 / 4.1 | 46.5 / 4.1 | 42.3 / 3.8 |
| Qwen2.5-1.5B | llama.cpp | f32 | 25.2 / 4.1 | 49.6 / 6.3 | 50.5 / 6.6 |
| Qwen2.5-1.5B | llama.cpp | Q8_0 | 26.9 / 10.7 | 53.1 / 19.2 | 49.3 / 23.7 |

### Concurrent greedy requests (32 tokens each)

| Model | Engine | Weights | Users | Output t/s | TTFT p50 / p99 ms |
|---|---|---|---:|---:|---:|
| SmolLM2-135M | dllm | f32 | 1 | 43.7 | 41 / 41 |
| SmolLM2-135M | dllm | f32 | 4 | 69.5 | 134 / 134 |
| SmolLM2-135M | dllm | f32 | 8 | 102.3 | 241 / 242 |
| SmolLM2-135M | dllm | Q8_0 | 1 | 53.7 | 54 / 54 |
| SmolLM2-135M | dllm | Q8_0 | 4 | 70.6 | 189 / 190 |
| SmolLM2-135M | dllm | Q8_0 | 8 | 90.6 | 345 / 345 |
| SmolLM2-135M | llama.cpp | f32 | 1 | 64.2 | 27 / 27 |
| SmolLM2-135M | llama.cpp | f32 | 4 | 96.5 | 132 / 133 |
| SmolLM2-135M | llama.cpp | f32 | 8 | 210.5 | 163 / 164 |
| SmolLM2-135M | llama.cpp | Q8_0 | 1 | 168.5 | 21 / 21 |
| SmolLM2-135M | llama.cpp | Q8_0 | 4 | 161.6 | 93 / 93 |
| SmolLM2-135M | llama.cpp | Q8_0 | 8 | 237.7 | 140 / 143 |
| Qwen2.5-0.5B | dllm | f32 | 1 | 16.8 | 103 / 103 |
| Qwen2.5-0.5B | dllm | f32 | 4 | 29.5 | 377 / 377 |
| Qwen2.5-0.5B | dllm | f32 | 8 | 42.1 | 660 / 661 |
| Qwen2.5-0.5B | dllm | Q8_0 | 1 | 24.7 | 110 / 110 |
| Qwen2.5-0.5B | dllm | Q8_0 | 4 | 38.1 | 397 / 397 |
| Qwen2.5-0.5B | dllm | Q8_0 | 8 | 47.2 | 737 / 738 |
| Qwen2.5-0.5B | llama.cpp | f32 | 1 | 19.1 | 79 / 79 |
| Qwen2.5-0.5B | llama.cpp | f32 | 4 | 34.2 | 278 / 279 |
| Qwen2.5-0.5B | llama.cpp | f32 | 8 | 70.3 | 564 / 573 |
| Qwen2.5-0.5B | llama.cpp | Q8_0 | 1 | 54.9 | 59 / 59 |
| Qwen2.5-0.5B | llama.cpp | Q8_0 | 4 | 57.3 | 219 / 221 |
| Qwen2.5-0.5B | llama.cpp | Q8_0 | 8 | 78.2 | 480 / 489 |
| Qwen3-0.6B | dllm | f32 | 1 | 13.9 | 130 / 130 |
| Qwen3-0.6B | dllm | f32 | 4 | 24.1 | 468 / 471 |
| Qwen3-0.6B | dllm | f32 | 8 | 34.3 | 840 / 840 |
| Qwen3-0.6B | dllm | Q8_0 | 1 | 22.1 | 135 / 135 |
| Qwen3-0.6B | dllm | Q8_0 | 4 | 31.0 | 491 / 491 |
| Qwen3-0.6B | dllm | Q8_0 | 8 | 37.4 | 915 / 916 |
| Qwen3-0.6B | llama.cpp | f32 | 1 | 15.7 | 99 / 99 |
| Qwen3-0.6B | llama.cpp | f32 | 4 | 28.4 | 351 / 353 |
| Qwen3-0.6B | llama.cpp | f32 | 8 | 59.5 | 641 / 647 |
| Qwen3-0.6B | llama.cpp | Q8_0 | 1 | 46.3 | 77 / 77 |
| Qwen3-0.6B | llama.cpp | Q8_0 | 4 | 50.0 | 271 / 272 |
| Qwen3-0.6B | llama.cpp | Q8_0 | 8 | 67.4 | 596 / 601 |
| OLMo-2-1B | dllm | f32 | 1 | 6.8 | 262 / 262 |
| OLMo-2-1B | dllm | f32 | 4 | 12.2 | 1,018 / 1,018 |
| OLMo-2-1B | dllm | f32 | 8 | 17.5 | 1,761 / 1,761 |
| OLMo-2-1B | dllm | Q8_0 | 1 | 15.3 | 215 / 215 |
| OLMo-2-1B | dllm | Q8_0 | 4 | 21.5 | 808 / 810 |
| OLMo-2-1B | dllm | Q8_0 | 8 | 25.1 | 1,506 / 1,507 |
| OLMo-2-1B | llama.cpp | f32 | 1 | 7.7 | 190 / 190 |
| OLMo-2-1B | llama.cpp | f32 | 4 | 14.3 | 701 / 701 |
| OLMo-2-1B | llama.cpp | f32 | 8 | 33.2 | 1,311 / 1,314 |
| OLMo-2-1B | llama.cpp | Q8_0 | 1 | 25.3 | 149 / 149 |
| OLMo-2-1B | llama.cpp | Q8_0 | 4 | 28.9 | 650 / 654 |
| OLMo-2-1B | llama.cpp | Q8_0 | 8 | 37.9 | 918 / 1,254 |
| TinyLlama-1.1B | dllm | f32 | 1 | 8.7 | 222 / 222 |
| TinyLlama-1.1B | dllm | f32 | 4 | 14.0 | 878 / 879 |
| TinyLlama-1.1B | dllm | f32 | 8 | 20.9 | 1,688 / 1,689 |
| TinyLlama-1.1B | dllm | Q8_0 | 1 | 17.4 | 206 / 206 |
| TinyLlama-1.1B | dllm | Q8_0 | 4 | 23.0 | 827 / 827 |
| TinyLlama-1.1B | dllm | Q8_0 | 8 | 27.0 | 1,578 / 1,579 |
| TinyLlama-1.1B | llama.cpp | f32 | 1 | 9.5 | 190 / 190 |
| TinyLlama-1.1B | llama.cpp | f32 | 4 | 20.0 | 854 / 854 |
| TinyLlama-1.1B | llama.cpp | f32 | 8 | 29.9 | 1,665 / 1,666 |
| TinyLlama-1.1B | llama.cpp | Q8_0 | 1 | 29.9 | 154 / 154 |
| TinyLlama-1.1B | llama.cpp | Q8_0 | 4 | 33.8 | 661 / 662 |
| TinyLlama-1.1B | llama.cpp | Q8_0 | 8 | 42.9 | 1,287 / 1,287 |
| Qwen2.5-1.5B | dllm | f32 | 1 | 5.8 | 312 / 312 |
| Qwen2.5-1.5B | dllm | f32 | 4 | 10.4 | 1,128 / 1,131 |
| Qwen2.5-1.5B | dllm | f32 | 8 | 14.9 | 2,203 / 2,204 |
| Qwen2.5-1.5B | dllm | Q8_0 | 1 | 11.6 | 287 / 287 |
| Qwen2.5-1.5B | dllm | Q8_0 | 4 | 16.5 | 1,054 / 1,054 |
| Qwen2.5-1.5B | dllm | Q8_0 | 8 | 19.3 | 2,013 / 2,014 |
| Qwen2.5-1.5B | llama.cpp | f32 | 1 | 6.4 | 230 / 230 |
| Qwen2.5-1.5B | llama.cpp | f32 | 4 | 11.8 | 848 / 849 |
| Qwen2.5-1.5B | llama.cpp | f32 | 8 | 26.8 | 1,615 / 1,621 |
| Qwen2.5-1.5B | llama.cpp | Q8_0 | 1 | 20.3 | 182 / 182 |
| Qwen2.5-1.5B | llama.cpp | Q8_0 | 4 | 22.7 | 715 / 717 |
| Qwen2.5-1.5B | llama.cpp | Q8_0 | 8 | 30.4 | 1,373 / 1,380 |
