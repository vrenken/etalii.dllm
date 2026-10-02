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

Qwen3-0.6B (Apache 2.0, about 1.5 GB download, 3 GB converted) is verified too. Qwen3 thinks before it answers;
its thinking is printed on stderr after `thinking:` and the answer on stdout. `--no-think` gets a direct answer, and
`--max-reasoning-tokens N` caps the thinking (see [§20](#20-reasoning-models)):

```bash
dllm import hf:Qwen/Qwen3-0.6B -o qwen3-0.6b.dllm
dllm --model qwen3-0.6b.dllm chat "What is the capital of France?" --no-think
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

Llama-3.2-1B-Instruct (Llama 3.2 Community Licence) and gemma-3-270m-it (Gemma terms) are verified too. Both are
gated: accept the licence on the model's Hugging Face page, set `HF_TOKEN` (a read token, or a fine-grained one with
read access to public gated repositories) and pass `--accept-licence`. The Gemma repository has no licence file, so
give the terms with `--licence-file`:

```bash
export HF_TOKEN=hf_...
dllm import hf:meta-llama/Llama-3.2-1B-Instruct -o llama3.2-1b.dllm --accept-licence
echo "Gemma Terms of Use: https://ai.google.dev/gemma/terms" > gemma-terms.txt
dllm import hf:google/gemma-3-270m-it -o gemma3-270m.dllm --accept-licence --licence-file gemma-terms.txt
dllm --model llama3.2-1b.dllm chat "What is the capital of France?"
```

Llama 3.x chat templates put a date in the system prompt; the engine never reads the clock, so they use the date the
template itself falls back to. With `--quantize q8_0`, Gemma loses more accuracy than the other models (Q8_0
quantises the activations too, and Gemma's are large): the 270M model still answers correctly but its top token
sometimes differs from the float32 one.

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
dllm --model smollm2-135m.dllm --quantize q4_0 chat "Hi"         # 4-bit weights, half of Q8_0's memory
dllm --model smollm2-135m.dllm --quantize q8_0 info              # shows the quantised system_fingerprint
dllm --model smollm2-135m.dllm --speculate chat "Repeat: a b c"  # speculative decoding ($DLLM_SPECULATE)
dllm --model qwen2.5-1.5b.dllm --draft-model qwen2.5-0.5b.dllm chat "Hi"   # drafts from a smaller model
dllm --model smollm2-135m.dllm --device cuda chat "Hi"           # NVIDIA GPU ($DLLM_DEVICE)
```

`--threads` never changes the output, only the speed. `--quantize q8_0` runs the linear layers on 8-bit weights:
faster and a quarter of the memory traffic, still deterministic, but the numbers differ slightly from the float
model, so it reports its own `system_fingerprint`. Details: [kernels](kernels.md#threads-and-simd). `--quantize q4_0`
halves the weight memory again at a larger accuracy cost, with its own fingerprint too
([Q4_0](kernels.md#q4_0-quantisation)). Only one copy of the weights stays in memory: SmolLM2-135M needs about
600 MB in float32, 230 MB with Q8_0 and 160 MB with Q4_0.

`--speculate` (8 guesses per step, or `--speculate N`) makes output that repeats earlier text faster, such as code,
quotes or edits of your prompt; `--draft-model` lets a smaller model with the same tokenizer make the guesses.
Unlike any other engine's speculative decoding it never changes a single token or logprob, only the speed
([speculative decoding](api.md#speculative-decoding)).

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
`--prompt-cache 0` turns this off, and `--persistent-cache DIR` keeps that work across restarts (a long shared
system prompt is then answered in 0.4 s instead of 6.3 s right after a restart); see
[prompt caching](api.md#prompt-caching). Temperature defaults to 0
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

## 7. Look inside the model

Because every number inside the model is reproducible, you can inspect it and anyone with the same model file sees
exactly the same thing:

```bash
dllm --model smollm2-135m.dllm lens --prompt "The capital of France is" --top-k 3        # what each layer predicts
dllm --model smollm2-135m.dllm attention --prompt "The cat sat on the mat. The cat" --layer 5 --html attention.html
dllm --model smollm2-135m.dllm neighbours "king - man + woman" --top-k 8 --svg cloud.svg  # ' queen' comes first
```

`--html` writes a self-contained page (a grid of predictions per layer and position, or attention heatmaps) and
`--svg` a word cloud. The Python API (`etalii_dllm.interpret.trace`) returns every activation of a forward pass.

You can also change the model on purpose. A steering vector pushes every answer in a direction, a ROME edit changes
one fact, and a sparse autoencoder finds the features a layer uses:

```bash
dllm --model smollm2-135m.dllm steer --positive "I love this, it is wonderful." --negative "I hate this, it is awful." -o love.json
dllm --model smollm2-135m.dllm --steer love.json chat "Describe your morning."   # also dllm-server, dllm-mcp, DLLM_STEER
dllm edit smollm2-135m.dllm --prompt "The Eiffel Tower is located in the city of" --subject "Eiffel Tower" \
    --target " Rome" -o smollm2-rome.dllm
dllm --model smollm2-135m.dllm sae train --corpus my-text.txt --steps 600 -o smollm2.sae
dllm --model smollm2-135m.dllm sae features smollm2.sae --corpus my-text.txt
```

Steering and edits are reproducible too: a steered model has its own `system_fingerprint`, and equal edits or SAE runs
write byte-identical files. Details: [interpretability](interpretability.md).

## 8. Answer from your documents

An embedding model turns your documents into an index, and a chat model can then answer from them. Both steps are
exact: the same documents give a byte-identical index, and the same question finds the same passages everywhere.

```bash
dllm import hf:Qwen/Qwen3-Embedding-0.6B -o qwen3-embedding.dllm
dllm --model qwen3-embedding.dllm index build my-notes/ -o notes.index      # .md, .txt, ... files
dllm index search notes.index "When is the next release?" --top 3           # uses the index's embedding model
dllm --model smollm2-135m.dllm --index notes.index chat "When is the next release?"
dllm-server --model smollm2-135m.dllm --index notes.index                   # every front end; also DLLM_INDEX
```

With `--index`, each chat gets the passages found for its last user message in its system message, and the
`system_fingerprint` changes to show it. `dllm-mcp --index notes.index` also gives MCP clients a `search_documents`
tool. Details: [retrieval](retrieval.md).

## 9. Docker

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
`dllm verify --reference` also checks the model's answers against an independent second implementation on this one
machine (see section 16).

## 10. Prove an answer later

Any answer can come with a receipt: the request and hashes of the output, enough for anyone with the same model to
re-run it and check every bit.

```bash
dllm --model smollm2-135m.dllm chat "Name three colours." --receipt colours.json
dllm --model smollm2-135m.dllm replay colours.json      # "verified: the replay gave the same output, bit for bit"
```

The server front ends return one when the request says `"receipt": true`, and `POST /v1/receipts/verify` (or the MCP
`verify_receipt` tool) checks one. Details: [receipts](receipts.md).

## 11. Score a model reproducibly

`dllm eval` measures perplexity on a text or multiple-choice accuracy from a JSON lines file, with a fingerprint
over every log-probability, so two machines can check they measured the very same thing:

```bash
dllm --model smollm2-135m.dllm eval my-text.txt
dllm --model smollm2-135m.dllm eval my-questions.jsonl   # {"context", "choices", "answer"} per line
```

Details: [evaluation](evaluation.md).

## 12. Replay an agent run

Give the model tools whose answers never change, record the run, and replay it offline later, round by round:

```bash
dllm --model qwen2.5-0.5b.dllm chat "What is 12.5 times 8.25?" --tool calculator --transcript run.json
dllm --model qwen2.5-0.5b.dllm replay run.json   # "verified: every round gave the same output, bit for bit"
```

`--tool files=DIR` lets the model read a directory and `--tool documents` search your `--index`; `dllm-tools` serves
the same tools to any MCP client. `--transcript` also works with `--mcp-server`: the replay answers every call from
the recording, so no server is started. A conversation over the HTTP APIs chains its receipts (`previous_receipt`,
or automatically with `previous_response_id`), and `dllm replay` of the list checks every turn. Details:
[reproducible agents](agents.md).

## 13. Check where a model came from

```bash
dllm inspect my-model.dllm                    # its lineage: import, fine-tunes, edits
dllm finetune base.dllm --data notes.txt -o tuned.dllm --receipt tuned.train.json
dllm replay tuned.train.json --base base.dllm  # trains again: "the same weights, bit for bit"
dllm sign --keygen me.key && dllm sign tuned.dllm --key me.key
dllm inspect tuned.dllm --trust me.key.pub     # checks tuned.dllm.sig
```

`dllm import` prints the file's SHA-256, which is the same on every platform. Signing needs
`pip install "etalii-dllm[sign]"`. Details: [verifiable models](provenance.md).

## 14. Serve many users with the same bits

```bash
dllm-server --model qwen2.5-0.5b.dllm --response-cache ~/.cache/dllm --audit-every 50
curl localhost:5080/v1/audit                  # cache hits, coalesced requests, audit results
dllm audit --url http://localhost:5080 --url http://other-machine:5080 --prompts prompts.txt
```

Repeated requests are answered from the cache with the same bits, and identical requests in flight share one
generation. Every 50th answer is re-run in the background to check it reproduces, and `dllm audit` checks that two
servers agree. Details: [serving at scale](serving.md).

## 15. Merge, export and distil models

```bash
dllm merge a.dllm b.dllm -o merged.dllm --method slerp --t 0.3
dllm export merged.dllm --format gguf -o merged.gguf          # or --format safetensors -o merged-hf/
dllm distill small.dllm --teacher big.dllm --prompts prompts.txt -o distilled.dllm --steps 200
```

Every result is the same file on every platform, and its lineage says how it was made. An export imports back to
the same weights, and a distillation's training receipt replays with `--teacher`. Details:
[building models](model-building.md).

## 16. Check that a machine computes what the specification says

```bash
dllm --model smollm2-135m.dllm verify --reference    # this machine's kernels against the reference implementation
dllm conformance write vectors/                      # test vectors for an implementation in another language
dllm conformance check vectors/                      # check this build (or --implementation reference) against them
```

`--reference` runs the prompt through the compiled kernels and through a second implementation written in plain
Python, and prints `equal` for the logits and for a greedy, a sampled and a controlled (penalties, min-p, logit
bias) answer (it takes about a minute for
SmolLM2-135M). A difference means this machine's kernels (a SIMD path, the GPU, the compiler) do not compute what the
[determinism specification](specification.md) says.

## 17. Shape the answer without losing reproducibility

```bash
dllm --model smollm2-135m.dllm generate --prompt "List: apple, apple," --repetition-penalty 1.3
dllm --model smollm2-135m.dllm generate --prompt "Once upon a time" --temperature 0.9 --seed 5 --n 3
dllm --model smollm2-135m.dllm chat "Give me a date" --regex '\d{4}-\d{2}-\d{2}'
```

Repetition, frequency and presence penalties, `--min-p` and `--logit-bias TOKEN=BIAS` change the logits in one fixed
order, so the same flags give the same answer everywhere. `--n 3` prints three choices; choice `i` is exactly the
answer you get with `--seed SEED+i`. `--regex` only lets the model write text that matches the pattern in full. The
HTTP API has all of them too: [decoding controls](api.md#decoding-controls).

## 18. Run a batch of requests

```bash
dllm --model smollm2-135m.dllm batch requests.jsonl -o results.jsonl --workers 8
dllm --model smollm2-135m.dllm batch requests.jsonl -o results.jsonl --verify
```

`requests.jsonl` is an OpenAI batch file (one `{"custom_id", "method", "url", "body"}` per line). The results file
is the same bytes whatever `--workers` says. If the run is interrupted, the same command resumes it to that exact
file. `--verify` re-runs the requests and checks every line. The server has the OpenAI Files and Batches API too.
Details: [batch jobs](batches.md).

## 19. Long conversations

```bash
dllm --model smollm2-135m.dllm generate --prompt "Once upon a time" --max-tokens 20000 --context-overflow roll
dllm --model smollm2-135m.dllm chat "Summarise our talk" --truncate
```

A prompt that does not fit the model's context window is refused instead of silently cut, and a full window ends the
answer with finish `length`. `--truncate` drops the oldest turns until the prompt fits. `--context-overflow roll`
keeps generating on a rolled window (the first 4 tokens and the latest half), and every token is still exactly what a
fresh run over those tokens gives. The HTTP API has both: [long conversations](api.md#long-conversations).

## 20. Reasoning models

```bash
dllm --model qwen3-0.6b.dllm chat "Is 391 prime?" --max-reasoning-tokens 200
dllm --model qwen3-0.6b.dllm chat "Is 391 prime?" --no-think
```

A thinking model's `<think>` block is kept apart from its answer: on the command line it goes to stderr, and the HTTP
APIs return it in their own reasoning fields. `--max-reasoning-tokens` closes the thinking after exactly that many
tokens and lets the model answer, and the result is the same on every machine. Details: [reasoning](api.md#reasoning).

## 21. Teach the model your preferences

```bash
dllm finetune smollm2-135m.dllm --dpo --data pairs.jsonl -o smollm2-135m-dpo.dllm --steps 100 --receipt dpo.json
dllm --model smollm2-135m-dpo.dllm eval pairs.jsonl
```

Each line of `pairs.jsonl` holds a prompt (or chat `messages`), a `chosen` answer and a `rejected` one. `--dpo`
trains with direct preference optimization, and the run is as reproducible as any fine-tune: the same file every
time, a resumed run equal to an uninterrupted one, and `dllm replay dpo.json --base smollm2-135m.dllm` to check it.
`dllm eval` on the same file reports how often the model prefers the chosen answer. Details:
[preference tuning](training.md#preference-tuning).

## 22. Search by words, and rerank

```bash
dllm index search docs.index "gated model import" --mode hybrid
dllm --model qwen2.5-1.5b.dllm rerank "Where is Paris?" "Berlin is in Germany." "Paris is in France."
dllm --model qwen2.5-1.5b.dllm --index docs.index --index-mode hybrid --rerank-model qwen2.5-1.5b.dllm chat "..."
```

`--mode lexical` searches an index by words (BM25) and `--mode hybrid` combines words and embeddings, which helps
with names and numbers. `dllm rerank` (and `POST /v1/rerank`) lets a chat model judge which passages answer a
question. Every ranking is exact, so it is the same on every machine. Details: [retrieval](retrieval.md#lexical-and-hybrid-search).

## 23. Watermark your answers

```bash
dllm generate --prompt "Write a short story" --temperature 0.8 --seed 1 --watermark-key my-secret > story.txt
dllm watermark detect story.txt --key my-secret
```

With a key, the sampler slightly prefers a key-dependent quarter of the vocabulary
after every token, and `dllm watermark detect` (or `POST /v1/watermark/detect`) counts how many tokens fall in it. The
same key gives the same text and the same score on every machine. Details: [watermarks](watermarks.md).

## 24. Score a text, and let answers vote

```bash
dllm score story.txt                                             # how likely the model finds every token
dllm chat "What is 17 * 3? End with 'Answer: N'." --temperature 0.8 --vote 7 --vote-extract "Answer: (\d+)"
```

`dllm score` prints the log-probability of every token, the log-likelihood and the perplexity. `--vote 7` samples seven
answers and prints the most common one. Both are exact, so they repeat on every machine. The server has the same as
`POST /v1/completions` with `echo` and `logprobs`, and a `vote` field on chat completions. Details:
[scoring](api.md#scoring) and [voting](api.md#voting).

## 25. Guide answers with a negative prompt, a smaller model or an ensemble

```bash
dllm generate --prompt "Once upon a time" --temperature 0.7 --seed 3 --negative-prompt "It was a dark night" --guidance-scale 3
dllm --model qwen2.5-1.5b.dllm --contrast-model qwen2.5-0.5b.dllm chat "Explain entropy" --contrast 0.5
dllm --model qwen2.5-1.5b.dllm --ensemble-model qwen2.5-0.5b.dllm=0.5 chat "Explain entropy"
```

`--negative-prompt` steers the answer away from another prompt (classifier-free guidance). `--contrast` prefers what
the big model knows better than a small one with the same tokenizer (contrastive decoding), and `--ensemble-model`
averages several models. Each is exact, so the answer repeats on every machine; the server has the same as
`guidance` and `contrast` request fields and the same model options. Details: [guided decoding](api.md#guided-decoding).

## 26. Find the most likely answers with beam search

```bash
dllm generate --prompt "The capital of France is" --max-tokens 12 --beams 4 --n-best 2
```

Beam search follows several likely continuations at once and prints the best ones, best first, with their
log-likelihoods on stderr. It is exact too: the same answers on every machine. The server has the same as a `beam`
field on chat completions and completions. Details: [beam search](api.md#beam-search).

## 27. Answers that fit every rule of a JSON schema

```bash
dllm chat "Book a flight" --json-schema '{"type": "object", "properties": {"from": {"type": "string", "pattern": "^[A-Z]{3}$"},
  "date": {"type": "string", "format": "date"}, "seats": {"type": "integer", "minimum": 1, "maximum": 9}},
  "required": ["from", "date", "seats"]}'
```

Structured output now also enforces string patterns, formats such as dates, string lengths and integer ranges, so
the answer is valid under the whole schema, not only its shape. Details: [structured output](api.md#structured-output).

## 28. Answers that follow your own grammar

```bash
cat > colours.gbnf <<'GBNF'
root   ::= "Colours: " colour (", " colour){1,3} "."
colour ::= "red" | "green" | "blue" | "yellow"
GBNF
dllm chat "Name some colours" --grammar colours.gbnf --temperature 0.8 --seed 3
```

A GBNF grammar (the format llama.cpp uses) can describe answers JSON schemas and regexes cannot, such as nested
expressions. Every answer follows the grammar, and the same request gives the same answer on every machine. The
server takes it as a `grammar` field on chat completions and completions. Details: [grammars](api.md#grammars).

## 29. Numbers within bounds, multiples and maps

```bash
dllm chat "Price a laptop" --json-schema '{"type": "object", "properties": {
  "price": {"type": "number", "minimum": 0, "maximum": 5000, "multipleOf": 0.01},
  "ratings": {"type": "object", "additionalProperties": {"type": "number", "minimum": 0, "maximum": 5}, "maxProperties": 3}},
  "required": ["price", "ratings"]}'
```

Bounds on decimal numbers, `multipleOf` (such as cents), the number of properties and the values of map-like objects
are now enforced exactly too, checked in decimal arithmetic rather than floating point. Details:
[structured output](api.md#structured-output).

## What does not work yet

- Eight real models are verified against Hugging Face `transformers` in CI: SmolLM2-135M-Instruct,
  Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, Qwen3-0.6B, TinyLlama-1.1B-Chat, OLMo-2-1B-Instruct,
  Llama-3.2-1B-Instruct and gemma-3-270m-it (tokenizer, chat template, logits within 1e-3 and the
  same greedy answer; see `tests/test_reference_models.py`), and so is the embedding model Qwen3-Embedding-0.6B. Other Llama/Qwen2/Qwen3 models should work but are not
  checked. Mistral, Granite, Phi-3 and Gemma 2 are checked against `transformers` only on tiny synthetic models (their real
  checkpoints are gated or too large for a CI runner in float32). Fine-tuning works for Llama, Mistral, Qwen2 and
  Qwen3, not yet for OLMo 2, Granite, Gemma or Phi models that use LongRoPE; Phi-3/Phi-4-mini with LongRoPE run up to their
  original context (4096 tokens) rather than the advertised 128k. If one misbehaves,
  please open an issue with the `dllm inspect` output.
- Speed: on a 4-core cloud VM, SmolLM2-135M reads a prompt at about 300 tokens per second and generates about 45
  tokens per second (about 57 with `--quantize q8_0`); Qwen2.5-0.5B is roughly three times slower. On an RTX 4080,
  Qwen2.5-0.5B generates about 40 tokens per second (about 110 with `--quantize q8_0`). The GPU backend computes in
  double precision to match the CPU bits, so it is far slower than float32 GPU engines. Fine-tuning runs on the CPU
  only and costs roughly three times as much per token as reading a prompt. [Benchmarks](benchmarks.md) compares
  six models side by side with transformers and llama.cpp: generation is faster than transformers and somewhat
  slower than llama.cpp, prompt reading is about 0.8× transformers and 0.6× llama.cpp, and only EtAlii.Dllm gives the same
  bits whatever the batch, thread count or concurrent load. `python benchmarks/benchmark.py` reproduces it.
- Tool calling works best with models trained for it (Qwen2.5-Instruct uses the same `<tool_call>` format the
  engine asks for). SmolLM2-135M does not know tools, so expect clumsy calls from it; constrained decoding still
  guarantees that every call names a real tool with arguments that fit its schema.
- Structured output supports the common JSON-schema keywords, including `pattern`, `format`, string lengths, number
  bounds, `multipleOf` and property counts; `uniqueItems`, `patternProperties`, `not` and similar are refused with an error
  (see [HTTP API](api.md#structured-output)). GBNF grammars may not be left-recursive and cannot use llama.cpp's
  token references (`<...>`) ([grammars](api.md#grammars)). No images or audio.
- MCP servers' own resources and prompts are not offered to the model (only their tools), and servers that ask
  the client for sampling or elicitation are not supported.
- Models with Unigram or WordPiece tokenizers, GGUF files with a SentencePiece vocabulary (convert from the
  Hugging Face checkpoint instead), YaRN RoPE scaling or other architectures (including
  Qwen3's mixture-of-experts models) are refused at import; Phase 9 of the roadmap adds the mainstream ones. Qwen3's thinking
  is returned as part of the answer text, not split into a separate reasoning field.
