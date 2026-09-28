# Getting started

This guide takes you from a fresh clone to a real open-weight model answering over the command line, an
OpenAI-compatible HTTP API and MCP. It is kept up to date as features land; the last section says what does not
work yet.

## 1. Install

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
pytest                              # optional: about 300 tests, a few seconds
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

Supported today: Llama-style models (SmolLM2, TinyLlama's architecture, Llama) and Qwen2/Qwen2.5 with byte-level BPE
tokenizers. Anything else is refused with a message saying what is missing. Only Apache-2.0 and MIT models import
without `--accept-licence`; see [model import](research/model-import.md) for the licence policy and candidate
models.

**Claude Code cloud sessions:** the default network policy blocks `huggingface.co`. Add `huggingface.co`,
`*.huggingface.co` and `*.hf.co` to the environment's allowed domains, or copy the files in another way. On your own
machine nothing needs configuring.

## 3. Generate

```bash
dllm --model smollm2-135m.dllm chat "What is the capital of France?"
dllm --model smollm2-135m.dllm chat "Write a haiku about rain" --temperature 0.7 --seed 42
dllm --model smollm2-135m.dllm generate --prompt "Once upon a time" --max-tokens 100
```

`chat` wraps your message in the model's own chat template; `generate` continues raw text. Instead of `--model` you
can set `DLLM_MODEL=/path/to/smollm2-135m.dllm` once; every command, the server and the MCP server use it.

Determinism: the same model file, prompt, options and seed give the same tokens every time on the same machine,
also under concurrent load. Temperature 0 (the default) is greedy decoding.

## 4. OpenAI-compatible server

```bash
dllm-server --model smollm2-135m.dllm        # http://127.0.0.1:5080, --host/--port to change
```

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

The response `id` and `system_fingerprint` are derived from the output and the weights, so identical requests get
byte-identical responses. Streaming and tool calls are not available yet (see below).

## 5. MCP server

Register the model as an MCP server in Claude Code (use an absolute path to the model):

```bash
claude mcp add dllm -- dllm-mcp --model /absolute/path/to/smollm2-135m.dllm
```

It exposes two tools: `generate` (continue a prompt) and `model_info` (model id and fingerprint). Other MCP clients
(Claude Desktop, IDEs) take the same command: `dllm-mcp --model /absolute/path/to/model.dllm` over stdio. If the
client starts it outside the virtual environment, use the full path to `.venv/bin/dllm-mcp` (Windows:
`.venv\Scripts\dllm-mcp.exe`).

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

## What does not work yet

- The importer, decoder, tokenizer and chat templates are tested against reference implementations on small
  synthetic models; the first end-to-end run on the real SmolLM2 weights and the logit comparison against
  Hugging Face `transformers` are still to come (roadmap issues #13 and #16). If a real model misbehaves, please
  open an issue with the `dllm inspect` output.
- Speed: the kernels are single-threaded and unoptimised (Phase 6). For SmolLM2-135M expect roughly 2 to 3 tokens
  per second, both for reading the prompt and for generating (measured on one cloud CPU core), and proportionally
  slower for bigger models. The KV cache is in place, so long answers do not slow down per token. Fine-tuning
  costs roughly three times as much per token as reading a prompt, so on SmolLM2-135M keep runs to a few thousand
  tokens for now.
- Streaming, tool/function calling, structured output and an Anthropic Messages endpoint are Phase 4; the MCP
  server has no chat tool yet.
- Models with SentencePiece tokenizers (TinyLlama, Llama 2), sliding-window attention, YaRN RoPE scaling or
  non-Llama/Qwen2 architectures are refused at import.
