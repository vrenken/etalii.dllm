# Verifiable models

Every output of EtAlii.Dllm is a pure function of its inputs, and since Phase 17 so is every model file. That makes
it possible to check where a model came from and who vouches for it:

- every `model.dllm` records its **lineage**, how its weights were made, step by step;
- a fine-tune writes a **training receipt** that anyone can replay to get the same weights, bit for bit;
- importing the same checkpoint gives the **same file**, byte for byte, on every platform, so a model is identified
  by its SHA-256;
- receipts, transcripts and model files can carry an Ed25519 **signature**.

## Lineage

```bash
dllm inspect my-model.dllm
```

```text
lineage 0:          import (format safetensors, source f72e4e04ac68c221) -> 2e4d93db18c1ce20
lineage 1:          fine_tune (data 9b1c..., steps 100, run 41aa...) -> 7c03e1a9d2b84f10
lineage 2:          edit (method rome, edit f71cf99af44ea676) -> 3dcec89975094b00
system_fingerprint: 3dcec89975094b00bd871f73b4364eb5fe202dd27cc842699463d74d7475c0e0
```

The `lineage` section of the header lists every step, oldest first:

| Step | Recorded |
| --- | --- |
| `import` | the source `format` and a digest of the `source` section (every source file's path, SHA-256 and size) |
| `adapter` | a digest of the `adapter` section (`dllm import ADAPTER --base BASE`) |
| `fine_tune` | the data fingerprint, the number of steps and a digest of the run settings (`dllm finetune`) |
| `edit` | the method and a digest of the edit record (`dllm edit`) |
| `merge` | the method and a digest of the `merge` section, which holds every input's fingerprint and lineage (`dllm merge`, [building models](model-building.md)) |

Every step after the import names the weights it started from (`input`). Every step except the last names the
weights it produced (`output`). The last step produced the file's own fingerprint. `dllm inspect` reports any step
whose input is not the output of the step before. Files written before Phase 17 get the lineage their other sections
imply. Writing an edit onto weights it was not computed from is refused.

## Training receipts

```bash
dllm finetune base.dllm --data notes.txt --steps 100 -o tuned.dllm --receipt tuned.train.json
dllm replay tuned.train.json --base base.dllm      # exit code 0: the same weights, bit for bit
dllm replay tuned.train.json --base base.dllm --data copy-of-notes.txt --json
```

A training receipt (format `dllm-train/1`) records:

- the base model's fingerprint;
- the training data, as the file's SHA-256, the fingerprint of its token windows (or, for a DPO run, its preference pairs) and the path as given;
- every run setting;
- every step's loss, exactly, as hex floats;
- the fingerprint of the weights that came out. For a LoRA run these are the merged weights.

`dllm replay` recognises a training receipt and trains again. The base model comes from `--base`, or else from
`--model`. The data comes from `--data`, or else from the file the receipt names. The replay names the first step
whose loss differs, and checks the final weights. The id (`trn_rcpt_...`) is a hash of the content, so an edited
receipt is detected. As with generation receipts, another engine version is only a note.

## Byte-identical model files

`dllm import` prints the whole file's SHA-256 next to the weights' fingerprint:

```text
system_fingerprint: 2e4d93db18c1ce202f44a8fd4da1333532827e2b84b7979ddcba6c918bfc132f
file_sha256:        ...
```

The header is canonical JSON without clock values or local paths, and the tensors are written in a fixed order, so
the same source files and options give the same bytes on Linux, Windows and macOS. CI checks the file hash of a
tiny import on all three, and every verified real model's file hash on all five release platforms.

The source files are part of the input. A checkout whose files differ, such as a model card with Windows line
endings, gives a different (and correctly different) file.

## Signatures

Signing needs the `sign` extra (`pip install "etalii-dllm[sign]"`, which installs `cryptography`).

```bash
dllm sign --keygen me.key                          # me.key (private, PEM) and me.key.pub (public, hex)
dllm sign answer.receipt.json --key me.key         # adds a signature, in place (or -o FILE)
dllm sign conversation.json --key me.key           # a receipt chain: every receipt is signed
dllm sign tuned.dllm --key me.key                  # writes tuned.dllm.sig
dllm-server --model tuned.dllm --sign-key me.key   # signs every receipt it returns ($DLLM_SIGN_KEY)

dllm replay answer.receipt.json --trust me.key.pub     # verifies the bits AND a signature by that key
dllm inspect tuned.dllm --trust 3f1c...                # checks tuned.dllm.sig
```

- JSON documents (receipts, training receipts, transcripts) carry the signature inline:
  `"signature": {"algorithm": "ed25519", "key": "<public key>", "value": "<signature>"}`. It is computed over the
  canonical JSON of everything else, and the document's id leaves it out, so signing never changes an id.
- A model file gets a detached `FILE.sig` (`{"file_sha256", "signature"}`) over its SHA-256.
- Ed25519 signatures are deterministic: the same key signing the same content gives the same bytes. A signed
  receipt is therefore as reproducible as an unsigned one.
- `--trust` can be repeated. A replay with `--trust` fails (exit 1) when the document is unsigned, when the signing
  key is not trusted, or when the signature does not match.

`dllm sign --keygen` is the one place that uses the operating system's entropy, because a key must be secret.
