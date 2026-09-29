# Model import and the model.dllm format

EtAlii.Dllm does not train models from scratch. It imports small open-weight models once, from Hugging Face
safetensors checkpoints or GGUF files, into a single `model.dllm` file, and runs that file from then on. This page
shows the path from a source checkpoint to a running decoder. The byte layout and the conversion rules are
specified in [model format](../model-format.md); the model choice and licence policy in
[model import](../research/model-import.md).

## From source to running model

```mermaid
flowchart LR
    subgraph sources["Sources"]
        direction TB
        hf["hf:org/name@revision<br/>(Hugging Face Hub)"]
        dir["checkpoint directory<br/>config.json, *.safetensors,<br/>tokenizer.json, tokenizer_config.json"]
        gguf["file.gguf<br/>(llama.cpp, Ollama)"]
    end

    subgraph imp["dllm import (importing/)"]
        direction TB
        hub["hub.py<br/>download pinned to a commit"]
        st["safetensors.py<br/>reader"]
        gg["gguf.py + quants.py<br/>reader, dequantisation"]
        conv["importer.py<br/>config, tensor names,<br/>widen to float32"]
        lic["licences.py<br/>licence policy"]
        write["modelfile.write_model_file<br/>canonical header, aligned data,<br/>SHA-256 fingerprint"]
    end

    file[("model.dllm")]

    subgraph run["Loading (engine.py)"]
        direction TB
        mf["ModelFile<br/>memory map, re-hash"]
        cfg["TransformerConfig"]
        tok["BPE tokenizer<br/>from the header"]
        tmpl["ChatTemplate<br/>from the header"]
        tr["Transformer<br/>float32 or Q8_0, cpu or cuda"]
        eng["DllmEngine"]
    end

    hf --> hub --> dir
    dir --> st --> conv
    gguf --> gg --> conv
    conv --> lic --> write --> file
    file --> mf
    mf --> cfg & tok & tmpl
    cfg --> tr
    tr & tok & tmpl --> eng
```

## Import

`import_model` accepts three kinds of source and turns each into the same intermediate form: a
`TransformerConfig`, a list of named float32 tensors, the tokenizer files, the chat template and the licence it
found.

- **`hf:org/name[@revision]`** is downloaded into a cache with the revision resolved to a commit hash, so the file
  records exactly what was read. Then it is imported as a directory.
- **A checkpoint directory** is read through `safetensors.py`. `config.json` (and `generation_config.json` for the
  stop tokens) becomes the `TransformerConfig`; Hugging Face tensor names are mapped to ours
  (`model.layers.3.self_attn.q_proj.weight` → `layers.3.attention.q.weight`). BF16 and F16 widen to float32 exactly.
- **A GGUF file** is read through `gguf.py`. Its metadata becomes the configuration and the tokenizer; quantised
  blocks (`Q4_0` to `Q8_0`, K-quants) are dequantised in llama.cpp's order, bit-identical to `gguf-py`, and the
  rotary permutation llama.cpp applies to q and k is undone.

The importer fails loudly on anything the decoder cannot run exactly (other architectures, non-SiLU activations,
MLP biases, sliding windows, partial rotary, unsupported RoPE scaling) rather than producing a model that behaves
differently from the original.

### Licences

```mermaid
flowchart TB
    start["licence stated by the source<br/>or --licence"] --> known{"stated at all?"}
    known -->|no| stop1["refuse: pass --licence"]
    known -->|yes| perm{"Apache-2.0 or MIT?"}
    perm -->|yes| text
    perm -->|no| acc{"--accept-licence?"}
    acc -->|no| stop2["refuse"]
    acc -->|yes| text{"licence text available?<br/>(source, bundled, --licence-file)"}
    text -->|no| stop3["refuse: pass --licence-file"]
    text -->|yes| rec["record spdx, full text, attribution,<br/>redistributable = permissive"]
```

The licence text and an attribution line travel inside the model file, so a `model.dllm` can be passed on with its
terms attached.

## The container

```mermaid
flowchart LR
    magic["DLLM<br/>magic"] --> ver["format<br/>version"] --> len["header<br/>length"] --> header["canonical JSON header<br/>architecture, tensors, fingerprint,<br/>source, licence, tokenizer,<br/>chat_template, fine_tuning"] --> data["tensor data<br/>float32, 64-byte aligned,<br/>natural name order"]
```

Two properties matter for the rest of the system:

- **The fingerprint is the model's identity.** The SHA-256 of the data section is written into the header and
  becomes the engine's `system_fingerprint` (`fp_` plus its first 12 hex digits). `ModelFile` re-hashes the data on
  open, so a corrupted or edited file cannot pass for the original.
- **Imports are reproducible.** The header is canonical JSON with sorted keys, tensors are in natural name order,
  and nothing comes from a clock or the environment, so importing the same source twice gives byte-identical files.

## Loading

`DllmEngine.from_model_file` memory-maps the file (`ModelFile`), so the tensors are kernel-ready aligned arrays
without a copy. From the header it builds the `TransformerConfig`, the BPE tokenizer (`bpe.py`, from the stored
`tokenizer.json`) and the `ChatTemplate` (with the special tokens from `tokenizer_config.json`), and collects the
stop tokens (the configuration's end-of-sequence ids plus the tokenizer's). The `Transformer` then prepares its
weights once: packed into the 16-output panels the SIMD kernels read, quantised to Q8_0 with `--quantize q8_0`
(which changes the fingerprint), or uploaded to the GPU with `--device cuda` (which does not).

```mermaid
classDiagram
    class ModelFile {
        config: TransformerConfig
        tensors
        fingerprint
        source
        licence
        tokenizer
        chat_template
        fine_tuning
        verify()
    }
    class TransformerConfig {
        family: llama or qwen2
        vocabulary_size, hidden_size
        layers, heads, kv_heads, head_dim
        rope_theta, rope_scaling
        attention_bias, tie_word_embeddings
        eos_token_ids
    }
    class Transformer {
        forward(tokens)
        forward_cached(tokens, cache)
        forward_batch(sequences, caches)
        hidden_states(tokens)
        new_cache()
    }
    class BpeTokenizer {
        encode(text)
        decode_bytes(ids)
    }
    class ChatTemplate {
        render(messages, tools)
    }
    class DllmEngine {
        chat_stream(request)
        chat_completion(request)
        embed(text)
        system_fingerprint
    }
    ModelFile --> TransformerConfig
    Transformer --> TransformerConfig
    DllmEngine --> Transformer
    DllmEngine --> BpeTokenizer
    DllmEngine --> ChatTemplate
```

## Verification against the originals

An imported model is only useful if it is the same model. `tests/test_reference_models.py` checks this against
Hugging Face `transformers` and `tokenizers`: tiny synthetic Llama and Qwen2 checkpoints in every run, and the
pinned real SmolLM2-135M, Qwen2.5-0.5B and Qwen2.5-1.5B instruct models in the `Reference models` workflow. Token
ids and rendered chat templates must match exactly; logits agree to about 1e-5 (bit equality is not expected,
since `transformers` sums in a different order), greedy answers are identical, and our own outputs are pinned
exactly in `tests/golden_values.py`.
