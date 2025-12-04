# Repository Guidelines

## Project Structure & Module Organization
- Core server code lives in `src/coral_tpu_mcp/` (`server.py`, `tpu_engine.py`, `__main__.py`). Entry point is `coral_tpu_mcp.server:run`.
- Python packaging is defined in `pyproject.toml`; editable installs pick up packages under `src/`.
- Docs and quick usage notes are in `README.md`. Add new assets or sample data under `src/coral_tpu_mcp/assets/` if needed; avoid repo root clutter.

## Build, Test, and Development Commands
- Set up env: `python -m venv .venv && source .venv/bin/activate && pip install -e .[dev]` (adds pytest and asyncio helpers).
- Run server locally: `python -m coral_tpu_mcp.server` or CLI `coral-tpu-mcp` (uses Coral TPU if present, falls back to CPU for text embeddings).
- Install Coral deps: `pip install pycoral tflite-runtime` inside the same venv (TPU runtime is not vendored).
- Tests: `pytest` from repo root (auto-discovers `tests/` or `test_*.py` you add alongside modules).

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indentation, type hints where practical; keep functions small and synchronous unless clearly async.
- Prefer snake_case for modules and functions; CamelCase for classes; ALL_CAPS for constants/env keys.
- Add lightweight docstrings for public functions and brief comments where flow is non-obvious.
- Format with `ruff format` if added; otherwise follow standard PEP 8. Keep imports sorted (stdlib, third-party, local).

## Testing Guidelines
- Place unit tests next to the code or under `tests/`; name files `test_*.py`.
- For TPU-dependent paths, provide CPU fallbacks or skip markers so CI without hardware passes.
- Include minimal fixtures/sample inputs; avoid embedding large binaries—reference external test data paths instead.

## Commit & Pull Request Guidelines
- Use Conventional Commits (e.g., `feat:`, `fix:`, `chore:`) with a focused scope (`feat(server): add face detection`).
- PRs should state intent, list runnable commands (setup, server run, pytest), and mention any model/runtime changes. Add logs or screenshots if behavior changes.
- Note configuration or dependency changes explicitly (e.g., new optional deps for TPU).

## Security & Configuration Tips
- Do not commit secrets; keep TPU/API keys in `.env` and load via environment variables. Never hardcode device paths.
- Keep test data under a sandbox path; scrub logs before sharing. Confirm TPU drivers are installed locally rather than fetched at runtime.
