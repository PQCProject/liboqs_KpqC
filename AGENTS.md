# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains the core C library (`common/`, `kem/`, `sig/`, `sig_stfl/`, and the public headers).
- `tests/` hosts C fixtures (KAT, fuzz, speed) and Python harnesses; vector archives stay under `ACVP_Vectors/` and `Wycheproof_Vectors/`.
- `scripts/` covers helper tooling (`format_code.sh`, `run_doxygen.sh`); integration samples are in `cpp/`, documentation assets in `docs/`.
- Keep build artifacts confined to `build/` or IDE-specific folders such as `cmake-build-debug/`; never commit generated binaries.

## Build, Test, and Development Commands
- `cmake -S . -B build -GNinja -DOQS_DIST_BUILD=ON` sets up a clean Ninja workspace.
- `ninja -C build` compiles the library, CLI utilities, and test binaries.
- `ninja -C build run_tests` or `ctest --test-dir build --output-on-failure` executes the C test suite.
- `pytest -n auto tests/test_kat_all.py` drives the Python harness; set `PYTHONPATH=tests` when calling individual modules.

## Coding Style & Naming Conventions
- Format C sources with `./scripts/format_code.sh`, wrapping `astyle` and `.astylerc` (Google style, width 4, pointer alignment, LF endings).
- Public APIs use `OQS_<module>_<verb>()` and live in `src/oqs.h`; module helpers remain in their subfolders.
- Preserve SPDX headers and keep new files MIT-compatible unless cleared with maintainers.
- Python utilities follow PEP 8; run `astyle` or `pytest tests/test_code_conventions.py` before pushing style changes.

## Testing Guidelines
- Mirror existing layout (`tests/kat_*.c`, `test_kat.py`, etc.) so CMake discovers new coverage automatically.
- Validate Known Answer Tests locally with `pytest tests/test_kat.py`; deposit new vector files beside the relevant algorithm folder.
- For performance-sensitive work, run `pytest tests/test_speed.py` after a full build and note any large deltas in PRs.
- Memory-hygiene and constant-time checks live in `tests/test_leaks.py` and `tests/test_constant_time.py`; run them when changing allocation or timing logic.

## Commit & Pull Request Guidelines
- Write short, imperative commit subjects (e.g., `Add KpqC KEM KATs`); append CI trigger tags like `[full tests]` only when you need extra workflows.
- Keep commits focused and document algorithm-specific rationale in commit bodies or PR notes.
- PRs must summarize scope, link issues, and list completed build/test commands; add benchmarks or platform notes when relevant.
- Request review only after CI passes; use draft PRs to gather early feedback from maintainers.

## Security & Configuration Tips
- Review `CONFIGURE.md` before toggling CMake options such as OpenSSL, AVX2, or dist builds, and sync updates with `PLATFORMS.md`.
- Do not commit secrets or generated key material; rely on encrypted storage or `.gitignore`-excluded artifacts.
