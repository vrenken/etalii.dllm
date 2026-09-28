# CLAUDE.md

Guidance for Claude Code sessions working in this repository.

## Project

EtAlii.Dllm is a deterministic LLM written from scratch in C# on .NET 10. The one non-negotiable requirement:
the same weights and the same request produce bit-identical output on every run and every platform.
Vision and roadmap: `README.md`. Background: `docs/research/`.

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
2. Never call `Math`/`MathF` transcendental functions (`Exp`, `Log`, `Pow`, `Sin`, `Cos`, `Tanh`, ...). Add a
   portable version to `DeterministicMath` built only from `+ - * /` and `sqrt`, with an accuracy test against `Math`.
3. Reductions run in a fixed, documented order with a `double` accumulator. No `Parallel.For` reductions, no
   `Vector<T>` (width varies by CPU), no `Math.FusedMultiplyAdd` unless every platform takes the same path.
4. Kernels must not change strategy based on batch size or sequence length.
5. Sorting must use a total order (break ties on index/token id).
6. Text processing is ordinal and culture-invariant. Do not depend on `Dictionary`/`HashSet` enumeration order.
7. API responses must not contain clock- or entropy-derived values; derive ids from content.

## Golden values

Reproducibility tests assert exact SHA-256 hashes, and CI runs them on Linux x64, Windows x64 and macOS Arm64.
If a hash changes:
- unintentionally: it is a determinism bug, find it; do not update the constant.
- intentionally (new weights, new sampler semantics): update `GoldenValues.cs` and say why in the commit message.
  Get the new values from the failing test output or from `dllm generate` (it prints the fingerprint to stderr).

## Conventions

- File-scoped namespaces, nullable enabled, analyzers at `latest-recommended`; fix warnings rather than suppress
  them, and justify any suppression in a comment.
- Test names use `Method_Behaviour` (CA1707 is disabled for tests only).
- Work on a feature branch and open a draft PR against `develop` (the default branch).
