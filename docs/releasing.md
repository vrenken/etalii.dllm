# Releasing

Releases are built and published by `.github/workflows/release.yml`. The version has one home,
`__version__` in `src/etalii_dllm/__init__.py`; `pyproject.toml` reads it from there.

## What a release contains

- An sdist, checked by building and running it on Linux.
- Wheels built with [cibuildwheel](https://cibuildwheel.pypa.io) (settings in `pyproject.toml`,
  `[tool.cibuildwheel]`) for CPython 3.11, 3.12 and 3.13 on:

  | OS | Architectures | Runner |
  | --- | --- | --- |
  | Linux (manylinux_2_28) | x86_64, aarch64 | `ubuntu-latest`, `ubuntu-24.04-arm` |
  | Windows | x86_64 | `windows-latest` |
  | macOS 11+ | arm64, x86_64 | `macos-latest`, `macos-15-intel` |

  Each wheel is installed in a fresh environment and the whole test suite runs against it, so the golden hashes
  are checked on every platform and Python version that is shipped. The wheels are built for the architecture's
  baseline (SSE2 on x86-64, NEON on arm64) and pick AVX2 + FMA at run time where the CPU has it, exactly as a
  source build does, so a wheel gives the same bits as a source build on the same machine.
- The CUDA backend is inside every wheel (its kernels are compiled at run time by NVRTC), so there are no
  separate GPU wheels; `pip install "etalii-dllm[cuda]"` only adds NVRTC.

Pull requests that change `pyproject.toml`, `CMakeLists.txt`, `cpp/` or the workflow run the same builds and tests
without releasing anything.

## Making a release

1. Set `__version__` in `src/etalii_dllm/__init__.py` (for example `0.2.0`) in a pull request and merge it.
2. Either push a tag `v0.2.0` on that commit, or run the *Release* workflow on `develop` with **release** ticked
   (Actions > Release > Run workflow); the workflow then creates the tag itself. A tag that does not match
   `__version__` fails the build.
3. When all wheels pass, the workflow creates the GitHub Release `v0.2.0` with generated notes and every wheel
   and the sdist attached.

## Docker image

A release also runs `.github/workflows/docker.yml`, which builds the image, checks that it answers identical
requests identically, and pushes it for linux/amd64 and linux/arm64 as `ghcr.io/vrenken/etalii-dllm:<version>` and
`:latest`. Every push to `develop` publishes `:edge`, and running the workflow by hand with tags publishes the
current branch under them. The package is linked to this public repository, so anyone can pull it.

## PyPI

Publishing to PyPI is switched off until the repository owner sets it up once (issue #44):

1. On pypi.org, add a pending trusted publisher: project `etalii-dllm`, owner `vrenken`, repository `etalii.dllm`,
   workflow `release.yml`, environment `pypi`.
2. In the repository, create the environment `pypi` (Settings > Environments) and the variable `PYPI_PUBLISH` with
   the value `true` (Settings > Secrets and variables > Actions > Variables).

From then on every release is uploaded to PyPI as well, without any stored token. A version cannot be uploaded
to PyPI twice, so bump `__version__` for every release.
