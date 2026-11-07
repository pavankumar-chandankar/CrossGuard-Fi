# Verification evidence — local engineering build

This document records repeatable verification steps and their limitations. It is not a substitute for CI run status, real-data validation, third-party security certification, or cloud deployment evidence.

## Executed locally (7 November 2026)

- Python import/compile checks: passed (`python -m compileall -q src tests scripts`).
- Full pytest suite: passed using available CPU-only PyTorch and FastAPI environment (see GitHub CI for exact Python 3.11 runner status after merge).
- One-command synthetic training, validation, sealed test and artifact inference smoke: passed (`PYTHONPATH=src python scripts/smoke.py`).
- Native C++17 source: compiled with strict `-Wall -Wextra -Werror`; 270-channel numerical parity and invalid-CSV tests passed.
- CMake build: passed on Linux; Clang Static Analyzer: executed with no diagnostic errors.
- Editable package installation: passed with `--no-build-isolation --no-deps` using already-installed local dependencies (container runtime cannot fetch internet packages).

## Not executed locally

- Docker build/run (Docker runtime unavailable in the working environment).
- Terraform `init/validate/plan/apply` (Terraform runtime and AWS credentials unavailable; no billable AWS actions initiated).
- Real CSI collection, training, benchmark, model accuracy, device/room generalization or AWS latency tests (no real dataset supplied).

All UI and infrastructure behavior must be checked in their target environment before claiming deployment. The GitHub Actions pipeline executes the end-to-end synthetic smoke and test suite on Python 3.11 after push.
