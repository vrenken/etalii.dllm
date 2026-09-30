# Getting started

This guide takes you from a fresh clone to a real open-weight model answering over the command line, OpenAI- and
Anthropic-compatible HTTP APIs and MCP. It is kept up to date as features land; the last section says what does not
work yet.

## 1. Install

### Pre-built wheel (no compiler needed)

Every [release](https://github.com/vrenken/etalii.dllm/releases) carries wheels for Linux (x86_64, aarch64),
Windows (x86_64) and macOS (Apple silicon and Intel) for Python 3.11 to 3.13, each tested against the full test
suite, golden hashes included. pip picks the right one for your machine from the release page:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install etalii-dllm --find-links https://github.com/vrenken/etalii.dllm/releases/expanded_assets/v0.2.0
```

Add `"etalii-dllm[cuda]"` instead of `etalii-dllm` for the GPU backend (Linux and Windows). Then continue with
the check below; you need a clone only for the tests and to build from source.

### From source

You need Python 3.11 or newer, CMake 3.18+ and a C++17 compiler (the numeric kernels are a C++ extension that is
compiled during installation):

| OS | Compiler |
| --- | --- |
| Windows | Visual Studio 2022 or its Build Tools, with the "Desktop development with C++" workload |
| macOS | `xcode-select --install` |
| Linux | `sudo apt install build-essential cmake` (or your distribution's equivalent) |

```bash
git clone https://github.com/vrenken/etalii.dllm.git
cd etalii.dllm
python -m venv .venv
source .venv/bin/activate          # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest                              # optional: about 1000 tests, under a minute (docs/testing.md)
```

Check the installation with the built-in placeholder model (a tiny random bigram table, so its text is gibberish,
but it exercises the whole deterministic pipeline):

```bash
dllm info
dllm generate --prompt "Hello" --temperature 0.8 --seed 7
```

Run the last command twice: the output, and the fingerprint printed on stderr, are identical.

## 2. Import a model

Models are converted once into a single `.dllm` file that holds the weights, tokenizer, chat template, the exact
source commit and the licence. The recommended first model is SmolLM2-135M-Instruct (Apache 2.0, about 270 MB
download, about 540 MB converted):

```bash
dllm import hf:HuggingFaceTB/SmolLM2-135M-Instruct -o smollm2-135m.dllm
dllm inspect smollm2-135m.dllm
```

Qwen2.5-0.5B-Instruct (Apache 2.0, about 1 GB download, 2 GB converted) is the second verified model and is
better at following instructions and calling tools:

```bash
dllm import hf:Qwen/Qwen2.5-0.5B-Instruct -o qwen2.5-0.5b.dllm
```

Qwen2.5-1.5B-Instruct (Apache 2.0, about 3 GB download, 6 GB converted) is the third verified model and the best
choice for tool calling and longer answers, if you have the memory: it peaks at about 12 GB in float32 and about
7.5 GB with `--quantize q8_0`, and with three times the parameters it is correspondingly slower than Qwen2.5-0.5B:

```bash
dllm import hf:Qwen/Qwen2.5-1.5B-Instruct -o qwen2.5-1.5b.dllm
dllm --model qwen2.5-1.5b.dllm --quantize q8_0 chat "What is the capital of France?"
```

Qwen3-0.6B (Apache 2.0, about 1.5 GB download, 3 GB converted) is verified too. Qwen3 thinks before it answers:
the reply starts with a `<think>...</think>` block. Add `/no_think` to a message to get a direct answer:

```bash
dllm import hf:Qwen/Qwen3-0.6B -o qwen3-0.6b.dllm
dllm --model qwen3-0.6b.dllm chat "What is the capital of France? /no_think"
```

TinyLlama-1.1B-Chat (Apache 2.0, about 2.2 GB download, 4.4 GB converted) is verified too; it is a Llama 2 model
with a SentencePiece-style tokenizer:

```bash
dllm import hf:TinyLlama/TinyLlama-1.1B-Chat-v1.0 -o tinyllama-1.1b.dllm
dllm --model tinyllama-1.1b.dllm chat "What is the capital of France?"
```

OLMo-2-1B-Instruct (Apache 2.0, fully open data and weights, about 3 GB download, 6 GB converted) is verified too:

```bash
dllm import hf:allenai/OLMo-2-0425-1B-Instruct -o olmo2-1b.dllm
dllm --model olmo2-1b.dllm chat "What is the capital of France?"
```

Pin a revision with `hf:HuggingFaceTB/SmolLM2-135M-Instruct@<commit or tag>`; without one the importer resolves
`main` to its current commit and records that. Downloads are cached in `~/.cache/etalii-dllm/hub` (`--cache` to
change it); set `HF_TOKEN` for gated repositories.

Other ways in:

```bash
# A checkpoint you already downloaded (config.json, tokenizer.json, *.safetensors)
dllm import ./SmolLM2-360M-Instruct -o smollm2-360m.dllm --repo HuggingFaceTB/SmolLM2-360M-Instruct

# A GGUF file from llama.cpp or Ollama (quantised files are dequantised exactly as llama.cpp does)
dllm import ./qwen2.5-0.5b-instruct-q8_0.gguf -o qwen2.5-0.5b.dllm
```

Supported today: Llama-style models (SmolLM2, TinyLlama, Llama), Mistral, OLMo 2, Granite 3.x, Phi-3/Phi-4-mini,
Qwen2/Qwen2.5, Qwen3 (dense), Gemma 2 and Gemma 3 (text: 270M, 1B) with
byte-level or SentencePiece-style BPE tokenizers ([full list](../README.md#which-models-can-it-run)). Anything else is refused with a message saying what is missing. Only Apache-2.0 and MIT models import
without `--accept-licence`; see [model import](research/model-import.md) for the licence policy and candidate
models. Gemma models are gated and use the Gemma terms: accept them on Hugging Face, set `HF_TOKEN` and pass
`--accept-licence`.

**Claude Code cloud sessions:** the default network policy blocks `huggingface.co`. Add `huggingface.co`,
`*.huggingface.co` and `*.hf.co` to the environment's allowed domains, or copy the files in another way. On your own
machine nothing needs configuring.

## 3. Generate

```bash
dllm --model smollm2-135m.dllm chat "What is the capital of France?"
dllm --model smollm2-135m.dllm chat "Write a haiku about rain" --temperature 0.7 --seed 42
dllm --model smollm2-135m.dllm generate --prompt "Once upon a time" --max-tokens 100
```

`chat` wraps your message in the model's own chat template; `generate` continues raw text. Both print the text as
it is generated. Instead of `--model` you can set `DLLM_MODEL=/path/to/smollm2-135m.dllm` once; every command, the
server and the MCP server use it.

Ask for JSON and the answer is guaranteed to parse (constrained decoding: the model can only pick tokens that keep
the output valid):

```bash
dllm --model smollm2-135m.dllm chat "Invent a cat" --json
dllm --model smollm2-135m.dllm chat "Invent a cat" --json-schema '{"type": "object",
  "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}'
```

Determinism: the same model file, prompt, options and seed give the same tokens every time, on any supported
machine, also under concurrent load, where `dllm-server` decodes simultaneous requests as one batch to serve them faster
([concurrent requests](api.md#concurrent-requests)). Temperature 0 (the default) is greedy decoding.

Speed options (they work the same for `dllm`, `dllm-server` and `dllm-mcp`):

```bash
dllm --model smollm2-135m.dllm --threads 2 chat "Hi"            # default: all cores ($DLLM_THREADS)
dllm --model smollm2-135m.dllm --quantize q8_0 chat "Hi"         # 8-bit weights ($DLLM_QUANTIZE)
dllm --model smollm2-135m.dllm --quantize q8_0 info              # shows the quantised system_fingerprint
dllm --model smollm2-135m.dllm --device cuda chat "Hi"           # NVIDIA GPU ($DLLM_DEVICE)
```

`--threads` never changes the output, only the speed. `--quantize q8_0` runs the linear layers on 8-bit weights:
faster and a quarter of the memory traffic, still deterministic, but the numbers differ slightly from the float
model, so it reports its own `system_fingerprint`. Details: [kernels](kernels.md#threads-and-simd).

`--device cuda` runs the model on an NVIDIA GPU and gives exactly the same output as the CPU (same tokens, same
`system_fingerprint`), about twice as fast in float32 and about three times as fast with `--quantize q8_0`. It needs
the NVIDIA driver and NVRTC, the CUDA runtime compiler; nothing CUDA is needed to install the package itself:

```bash
pip install -e ".[dev,cuda]"      # adds NVRTC (nvidia-cuda-nvrtc-cu12); a CUDA toolkit or PyTorch's copy works too
dllm --model smollm2-135m.dllm --device cuda info    # prints the GPU, e.g. "cuda (NVIDIA GeForce RTX 4080, sm_89, ...)"
```

If `--device cuda` fails, check NVRTC on its own (this needs no GPU; 89 is the compute capability, 8.9 for the
RTX 40 series): `python -c "from etalii_dllm import cuda; print(cuda.compile_kernels(89).architecture)"` prints
`sm_89` when NVRTC is found and compiles the kernels. The `[cuda]` extra installs nothing on macOS, which has no CUDA.

NVRTC is found automatically in the `nvidia-cuda-nvrtc` wheel, PyTorch, `$CUDA_PATH`/`$CUDA_HOME` or
`/usr/local/cuda`; set `DLLM_NVRTC` to the library's full path (for example `nvrtc64_120_0.dll`) to pick one, and
`DLLM_CUDA_DEVICE` to choose a GPU other than the first. macOS has no CUDA. Details: [kernels](kernels.md#gpu).

## 4. OpenAI-, Anthropic- and Ollama-compatible server

```bash
dllm-server --model smollm2-135m.dllm        # http://127.0.0.1:5080, --host/--port to change
```

Open <http://127.0.0.1:5080> in a browser for a chat page. Set the temperature and seed under *Settings*, ask
something, then press *Regenerate*: the answer comes back identical, with the same id and `system_fingerprint`, and
the page says so.

```bash
curl http://127.0.0.1:5080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "dllm",
  "messages": [{"role": "user", "content": "What is the capital of France?"}],
  "temperature": 0.7, "seed": 5, "max_tokens": 64
}'
```

Any OpenAI client works by pointing its base URL at `http://127.0.0.1:5080/v1` (the API key is ignored):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:5080/v1", api_key="unused")
reply = client.chat.completions.create(
    model="dllm", messages=[{"role": "user", "content": "Hi!"}], seed=1, temperature=0.5
)
print(reply.choices[0].message.content, reply.system_fingerprint)
```

Streaming (`stream=True`), tool calling (`tools`, `tool_choice`), structured output (`response_format` with
`json_object` or `json_schema`), `logprobs`/`top_logprobs`, `stop` sequences and `/v1/embeddings` work as in the
OpenAI API:

```python
for chunk in client.chat.completions.create(
    model="dllm", messages=[{"role": "user", "content": "Count to five."}], stream=True
):
    print(chunk.choices[0].delta.content or "", end="")

vectors = client.embeddings.create(model="dllm", input=["first text", "second text"])
```

The same server speaks the Anthropic Messages API at `/v1/messages`, so the Anthropic SDK works too:

```python
import anthropic

client = anthropic.Anthropic(base_url="http://127.0.0.1:5080", api_key="unused")
message = client.messages.create(
    model="dllm",
    max_tokens=100,
    messages=[{"role": "user", "content": "Hi!"}],
    extra_body={"temperature": 0.7, "seed": 42},  # sampling options go in extra_body
)
print(message.content[0].text)
```

The OpenAI SDK's newer `client.responses.create(...)` works as well, including `previous_response_id` follow-ups
([Responses API](api.md#responses-api)). Ollama clients work too: point them at `http://127.0.0.1:5080` (for example `ollama.Client(host=...)` in Python or
Open WebUI's Ollama URL); see [Ollama API](api.md#ollama-api).

The response ids and `system_fingerprint` are derived from the request and the weights, so identical requests get
byte-identical responses, and a streamed answer is identical to the non-streamed one. The server reuses the work of
earlier requests that start the same way (a chat's earlier turns, a shared system prompt), so follow-up turns are
several times faster; `usage` reports the reused tokens (`cached_tokens`), which is the only thing that differs.
`--prompt-cache 0` turns this off, see [prompt caching](api.md#prompt-caching). Temperature defaults to 0
(greedy) on every endpoint. All options, how tools and structured output work, and the differences from the real
APIs: [HTTP API](api.md).

## 5. MCP

Register the model as an MCP server in Claude Code (use an absolute path to the model):

```bash
claude mcp add dllm -- dllm-mcp --model /absolute/path/to/smollm2-135m.dllm
```

It exposes three tools: `chat` (answer a conversation with the model's chat template, optionally as JSON matching a
schema), `generate` (continue a prompt) and `model_info` (model id and fingerprint); three resources (the model card
at `dllm://model`, the chat template, and the determinism guarantee); and three prompts (`summarize`, `translate`,
`extract_json`). Other MCP clients (Claude Desktop, IDEs) take the same command: `dllm-mcp --model
/absolute/path/to/model.dllm` over stdio. If the client starts it outside the virtual environment, use the full path
to `.venv/bin/dllm-mcp` (Windows: `.venv\Scripts\dllm-mcp.exe`).

The other direction works too: let the model call the tools of MCP servers during a chat. Put the servers in a
`mcp.json` in the usual `mcpServers` format, or name them with `--mcp-server`:

```bash
dllm --model qwen2.5-0.5b.dllm chat "What time is it in Amsterdam?" --mcp-server "time=uvx mcp-server-time"
dllm --model qwen2.5-0.5b.dllm chat "What time is it in Amsterdam?" --mcp-config mcp.json
```

The tool calls and their results are printed to stderr and the answer to stdout. Tool calling needs a model trained
for it: use Qwen2.5-Instruct here, not SmolLM2-135M. Details, the Python API and what the determinism guarantee
covers when external tools are involved: [MCP](mcp.md).

## 6. Fine-tune

Train an imported model further on your own text. The run is reproducible: the same model, data and options give a
byte-identical result, also when it is interrupted and resumed from a checkpoint.

```bash
# my-data.jsonl: one {"text": "..."} or {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]} per line
dllm finetune smollm2-135m.dllm --data my-data.jsonl -o smollm2-135m-tuned.dllm \
    --steps 50 --batch-size 4 --sequence-length 128 --learning-rate 1e-4 --checkpoint run.dllmckpt
dllm inspect smollm2-135m-tuned.dllm
dllm --model smollm2-135m-tuned.dllm chat "..."
```

Each step prints its loss. `--resume run.dllmckpt` continues a stopped run. All options, the data format and how
the reproducibility is achieved: [training](training.md).

LoRA trains small adapters instead of all weights, and they work with the PEFT ecosystem:

```bash
dllm finetune smollm2-135m.dllm --data my-data.jsonl --lora-rank 8 --adapter-output my-adapter \
    --steps 50 --learning-rate 1e-3
dllm --model smollm2-135m.dllm --adapter my-adapter chat "..."          # or DLLM_ADAPTER=my-adapter
dllm import my-adapter --base smollm2-135m.dllm -o smollm2-135m-lora.dllm   # merge into a new model file
dllm import ./some-peft-adapter --base qwen2.5-0.5b.dllm -o tuned.dllm     # a PEFT adapter trained elsewhere
```

Applying an adapter at load time and serving the merged file give the same answers bit for bit, with the same
`system_fingerprint`. Details: [LoRA adapters](training.md#lora-adapters).

## 7. Docker

The server also comes as an image for linux/amd64 and linux/arm64, published with every release as
`ghcr.io/vrenken/etalii-dllm:<version>` and `:latest` (`:edge` follows `develop`). It runs `dllm-server` on port
5080, chat page included. Put a model in a volume at `/models/model.dllm`, or let the container import one on its
first start:

```bash
# Import SmolLM2 into the volume on first start, then serve it (later starts reuse the file)
docker run -p 5080:5080 -v dllm-models:/models \
  -e DLLM_IMPORT=hf:HuggingFaceTB/SmolLM2-135M-Instruct ghcr.io/vrenken/etalii-dllm:latest

# Serve a model you already converted
docker run -p 5080:5080 -v "$PWD/smollm2-135m.dllm:/models/model.dllm:ro" ghcr.io/vrenken/etalii-dllm:latest

# The other front ends and options work too
docker run --rm -v dllm-models:/models ghcr.io/vrenken/etalii-dllm:latest dllm chat "Hi"
docker run -p 5080:5080 -v dllm-models:/models -e DLLM_QUANTIZE=q8_0 -e DLLM_THREADS=4 ghcr.io/vrenken/etalii-dllm:latest
docker run --gpus all -p 5080:5080 -v dllm-models:/models -e DLLM_DEVICE=cuda ghcr.io/vrenken/etalii-dllm:latest
```

`DLLM_IMPORT_ARGS` passes extra options to the import (for example `--accept-licence`). The image contains the
`[cuda]` extra, so `--gpus all` (NVIDIA container toolkit) is all a GPU needs. On the same machine, the container
gives byte-identical responses to a native install. Build it yourself with `docker build -t etalii-dllm .`.

To check that two machines really give the same bits, run `dllm verify` on both (with the same `--model`,
`--quantize` and `--device` options) and compare the last line:

```bash
dllm --model smollm2-135m.dllm verify
```

It runs a fixed workload (every kernel, the Unicode handling, the model's tokenizer, logits, a greedy and a sampled
answer) and prints one fingerprint per part plus a combined `verify:` line, together with the environment (Python,
instruction set, device). The kernel and Unicode parts are also compared with the values of the release, so a single
machine already shows `(as released)` or `(DIFFERS from release)`; `--json` prints the report as JSON.

## What does not work yet

- Six real models are verified against Hugging Face `transformers` in CI: SmolLM2-135M-Instruct,
  Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, Qwen3-0.6B, TinyLlama-1.1B-Chat and OLMo-2-1B-Instruct (tokenizer, chat template, logits within 1e-3 and the
  same greedy answer; see `tests/test_reference_models.py`). Other Llama/Qwen2/Qwen3 models should work but are not
  checked. Mistral, Granite, Phi-3, Gemma 2 and Gemma 3 are checked against `transformers` only on tiny synthetic models (their real
  checkpoints are gated or too large for a CI runner in float32). Fine-tuning works for Llama, Mistral, Qwen2 and
  Qwen3, not yet for OLMo 2, Granite, Gemma or Phi models that use LongRoPE; Phi-3/Phi-4-mini with LongRoPE run up to their
  original context (4096 tokens) rather than the advertised 128k. If one misbehaves,
  please open an issue with the `dllm inspect` output.
- Speed: on a 4-core cloud VM, SmolLM2-135M reads a prompt at about 150 tokens per second and generates about 20
  tokens per second (about 35 with `--quantize q8_0`); Qwen2.5-0.5B is roughly four times slower. On an RTX 4080,
  Qwen2.5-0.5B generates about 40 tokens per second (about 110 with `--quantize q8_0`). The GPU backend computes in
  double precision to match the CPU bits, so it is far slower than float32 GPU engines. Fine-tuning runs on the CPU
  only and costs roughly three times as much per token as reading a prompt.
- Tool calling works best with models trained for it (Qwen2.5-Instruct uses the same `<tool_call>` format the
  engine asks for). SmolLM2-135M does not know tools, so expect clumsy calls from it; constrained decoding still
  guarantees that every call names a real tool with arguments that fit its schema.
- Structured output supports the common JSON-schema keywords; `pattern`, `minLength`, `minimum` and similar are
  refused with an error (see [HTTP API](api.md)). No images, audio or `n` > 1.
- MCP servers' own resources and prompts are not offered to the model (only their tools), and servers that ask
  the client for sampling or elicitation are not supported.
- Models with Unigram or WordPiece tokenizers, GGUF files with a SentencePiece vocabulary (convert from the
  Hugging Face checkpoint instead), YaRN RoPE scaling or other architectures (including
  Qwen3's mixture-of-experts models) are refused at import; Phase 9 of the roadmap adds the mainstream ones. Qwen3's thinking
  is returned as part of the answer text, not split into a separate reasoning field.
