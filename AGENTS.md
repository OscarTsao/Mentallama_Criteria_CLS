# Repository Guidelines

## Project Structure & Module Organization
- `src/Project/SubProject/` – main package.
  - `utils/` (logging, MLflow, seeding), `models/`, `engine/`, `data/`.
- `tests/` – pytest tests (add `test_*.py`).
- `configs/` – experiment/config stubs.
- `data/`, `outputs/`, `artifacts/`, `mlruns/` – local data, results, and MLflow runs (do not commit large files).
- `scripts/`, `docs/` – helper scripts and documentation.

## Build, Test, and Development Commands
- Create env + install (editable):
  ```bash
  python -m venv .venv && source .venv/bin/activate
  python -m pip install --upgrade pip
  pip install -e '.[dev]'
  ```
- Lint/format/type-check:
  ```bash
  ruff check src tests
  black src tests
  mypy src
  ```
- Run tests:
  ```bash
  pytest -q
  ```

## Coding Style & Naming Conventions
- Python 3.10+. Use type hints throughout.
- Formatting: Black (line length 100); linting: Ruff (target py310).
- Indentation: 4 spaces. Imports grouped (stdlib, third-party, local).
- Naming: `snake_case` for functions/modules, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Logging via `utils.get_logger` instead of `print`.

## Testing Guidelines
- Place tests under `tests/` with filenames like `test_utils_seed.py`.
- Use pytest; prefer small, deterministic tests (seed with `utils.set_seed`).
- Add tests alongside new modules and for bug fixes.

## Commit & Pull Request Guidelines
- Commits: imperative, scoped messages. Prefer Conventional Commits:
  - e.g., `feat: add mlflow_run context manager`, `fix: correct classifier head shape`.
- PRs must include:
  - Summary, rationale, and how to test (commands).
  - Linked issues (e.g., `Closes #123`).
  - Updated docs/tests when behavior changes.

## Security & Configuration Tips
- Never commit secrets, credentials, or large datasets. Use env vars.
- MLflow: set tracking URI for local runs, e.g. `export MLFLOW_TRACKING_URI=file:./mlruns`.
- Keep experiments and artifacts in `outputs/` or `artifacts/`.

## Agent-Specific Instructions
- Keep diffs minimal and follow `pyproject.toml` tooling.
- Preserve package layout (`src/Project/SubProject/...`) and import style.
- Prefer small, focused changes with accompanying tests and docs.
