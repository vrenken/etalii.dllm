# 2026-09-30, Claude Code cloud container

SmolLM2-135M only (the Qwen and other weights cannot be downloaded there). Raw results: [2026-09-30-cloud-xeon.json](2026-09-30-cloud-xeon.json). Summary: [docs/benchmarks.md](../benchmarks.md).

Machine: Intel(R) Xeon(R) Processor @ 2.80GHz, 4 cores, 15.7 GB, Linux-6.18.44-fc-v50-x86_64-with-glibc2.39; dllm 0.2.0 (avx2), commit 459db237d2.

### Throughput (pp128, tg64, 4 threads, mean ± stdev of 5)

| Model | Engine | Weights | pp t/s | tg t/s | TPOT p50 / p99 ms | Load s | Peak RSS MB |
|---|---|---|---:|---:|---:|---:|---:|
| SmolLM2-135M | dllm | f32 | 97.6 ± 3.5 | 15.8 ± 0.3 | 61.6 / 95.7 | 2.41 | 1,335 |
| SmolLM2-135M | dllm | Q8_0 | 86.5 ± 2.3 | 22.9 ± 0.9 | 42.7 / 60.3 | 3.92 | 1,060 |
| SmolLM2-135M | transformers | f32 | 579.7 ± 77.2 | 21.8 ± 0.9 | 43.9 / 66.5 | 3.17 | 1,816 |
| SmolLM2-135M | llama.cpp | f32 | 726.0 ± 35.5 | 41.5 ± 1.3 | n/a | n/a | n/a |
| SmolLM2-135M | llama.cpp | Q8_0 | 536.8 ± 62.6 | 83.7 ± 2.0 | n/a | n/a | n/a |

### Quality (WikiText-2 test, ctx 512, 8 chunks)

| Model | Engine | Weights | PPL | KLD vs f32 | Top-1 agreement |
|---|---|---|---:|---:|---:|
| SmolLM2-135M | dllm | f32 | 16.742 ± 1.078 |  |  |
| SmolLM2-135M | dllm | Q8_0 | 16.796 ± 1.079 | 0.00350 | 96.57 % |
| SmolLM2-135M | transformers | f32 | 16.742 ± 1.078 |  |  |
| SmolLM2-135M | llama.cpp | f32 | 16.742 ± 1.078 |  |  |
| SmolLM2-135M | llama.cpp | Q8_0 | 16.815 ± 1.080 |  |  |

### Determinism

| Model | Engine | Weights | Repeat runs | Thread counts | Batch composition | Concurrent = solo |
|---|---|---|---|---|---|---|
| SmolLM2-135M | dllm | f32 | identical | identical | identical | identical |
| SmolLM2-135M | dllm | Q8_0 | identical | identical | identical | identical |
| SmolLM2-135M | transformers | f32 | identical | **differs** | **differs** (max 2.77e-05) | n/a |
| SmolLM2-135M | llama.cpp | f32 | identical | identical | identical | **differs** (text same) |
| SmolLM2-135M | llama.cpp | Q8_0 | identical | identical | identical | identical |

### Thread scaling (tokens/s)

| Model | Engine | Weights | pp @1 / tg @1 | pp @2 / tg @2 | pp @4 / tg @4 |
|---|---|---|---:|---:|---:|
| SmolLM2-135M | dllm | f32 | 38.2 / 9.4 | 63.1 / 13.2 | 102.0 / 16.9 |
| SmolLM2-135M | dllm | Q8_0 | 37.0 / 20.7 | 57.7 / 23.4 | 83.0 / 22.5 |
| SmolLM2-135M | transformers | f32 | 231.2 / 10.5 | 390.6 / 15.9 | 637.3 / 22.6 |
| SmolLM2-135M | llama.cpp | f32 | 243.5 / 14.5 | 390.2 / 25.8 | 729.1 / 37.2 |
| SmolLM2-135M | llama.cpp | Q8_0 | 201.3 / 36.3 | 300.2 / 61.9 | 604.3 / 86.9 |

### Concurrent greedy requests (32 tokens each)

| Model | Engine | Weights | Users | Output t/s | TTFT p50 / p99 ms |
|---|---|---|---:|---:|---:|
| SmolLM2-135M | dllm | f32 | 1 | 15.3 | 122 / 122 |
| SmolLM2-135M | dllm | f32 | 4 | 22.6 | 368 / 368 |
| SmolLM2-135M | dllm | f32 | 8 | 32.8 | 666 / 667 |
| SmolLM2-135M | dllm | Q8_0 | 1 | 23.6 | 114 / 114 |
| SmolLM2-135M | dllm | Q8_0 | 4 | 31.1 | 435 / 435 |
| SmolLM2-135M | dllm | Q8_0 | 8 | 39.2 | 770 / 771 |
| SmolLM2-135M | llama.cpp | f32 | 1 | 38.2 | 43 / 43 |
| SmolLM2-135M | llama.cpp | f32 | 4 | 69.4 | 125 / 221 |
| SmolLM2-135M | llama.cpp | f32 | 8 | 120.3 | 271 / 276 |
| SmolLM2-135M | llama.cpp | Q8_0 | 1 | 74.1 | 38 / 38 |
| SmolLM2-135M | llama.cpp | Q8_0 | 4 | 99.4 | 139 / 139 |
| SmolLM2-135M | llama.cpp | Q8_0 | 8 | 175.3 | 191 / 195 |
