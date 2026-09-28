# CLAUDE.md

Guidance for Claude Code sessions working in this repository.

## Project

EtAlii.Dllm is a deterministic LLM written from scratch in C# on .NET 10. The one non-negotiable requirement:
on the same hardware, the same weights, prompt and context window produce bit-identical output on every run,
regardless of load, batching or thread scheduling. Identical output across different hardware is not required.
Vision and roadmap: `README.md`. Background: `docs/research/`.
Weights come from importing small open-weight models (Apache 2.0/MIT by default), not from training from scratch;
see `docs/research/model-import.md`. `huggingface.co` is blocked by the default cloud network policy.

## Commands

```bash
dotnet build                          # warnings are errors
dotnet test                           # all tests, including golden-hash reproducibility tests
dotnet format                         # CI runs `dotnet format --verify-no-changes`
dotnet run --project src/EtAlii.Dllm.Cli -- generate --prompt "Hi" --temperature 0.8 --seed 7
dotnet run --project src/EtAlii.Dllm.Server   # http://localhost:5080/v1/chat/completions
dotnet run --project src/EtAlii.Dllm.Mcp      # MCP over stdio
```

Cloud sessions: `.claude/hooks/session-start.sh` installs `dotnet-sdk-10.0` from the Ubuntu archive (the
Microsoft download host is blocked by the network policy) and restores packages. If `dotnet` is missing, run it.

## Layout

- `src/EtAlii.Dllm.Core`: all model logic. `Numerics/` (RNG, portable math, fingerprints), `Sampling/`,
  `Tokenization/`, `Models/`, `Generation/`, `Chat/`, `Hosting/DllmEngine` (facade shared by every front end).
- `src/EtAlii.Dllm.Server`, `src/EtAlii.Dllm.Mcp`, `src/EtAlii.Dllm.Cli`: thin front ends over `DllmEngine`.
  Keep logic out of them so all three stay output-identical.
- `tests/`: xUnit. `tests/EtAlii.Dllm.Core.Tests/GoldenValues.cs` holds the reference hashes.
- Shared build settings live in `Directory.Build.props` (target framework, nullable, warnings as errors).

## Determinism rules (inference and training code)

1. Never use `System.Random`, `Guid.NewGuid`, `DateTime.Now` or any ambient entropy. Use `DeterministicRandom`.
2. Prefer `DeterministicMath` over `Math`/`MathF` transcendental functions (`Exp`, `Log`, `Sin`, `Tanh`, ...) so
   runtime upgrades cannot shift results; add new ones built from `+ - * /` and `sqrt`, with an accuracy test.
3. Reductions run in a fixed, documented order with a `double` accumulator. Parallelism only with fixed partitioning
   (chunks from data size, combined in chunk order). SIMD and FMA are fine if the code path is fixed per machine.
4. Kernels must not change strategy based on batch size or sequence length.
5. Sorting must use a total order (break ties on index/token id).
6. Text processing is ordinal and culture-invariant. Do not depend on `Dictionary`/`HashSet` enumeration order.
7. API responses must not contain clock- or entropy-derived values; derive ids from content.

## Golden values

Reproducibility tests assert exact SHA-256 hashes. CI runs them on Linux, Windows and macOS; today all agree, but if a
hardware-specific kernel makes them diverge, key the golden values per platform rather than forcing portability.
If a hash changes:
- unintentionally: it is a determinism bug, find it; do not update the constant.
- intentionally (new weights, new sampler semantics): update `GoldenValues.cs` and say why in the commit message.
  Get the new values from the failing test output or from `dllm generate` (it prints the fingerprint to stderr).

## Conventions

- File-scoped namespaces, nullable enabled, analyzers at `latest-recommended`; fix warnings rather than suppress
  them, and justify any suppression in a comment.
- Test names use `Method_Behaviour` (CA1707 is disabled for tests only).
- Work on a feature branch and open a draft PR against `develop` (the default branch).
- Progress is tracked in the GitHub Project EtAlii.Dllm (see `docs/project-board.md`). Roadmap items are issues
  labelled `roadmap`/`phase-N` under "Phase N" milestones; a PR that finishes one says `Closes #n`, and a README
  roadmap change updates the matching issues/milestones.
