# Repository Guidelines

## Project Structure & Modules
- `src/safe_feedback_interpretation/`: library code (models, viz, structs). Entry points live here.
- `experiments/`: runnable scripts and Hydra configs (e.g., `conf/modality_disagreement.yaml`).
- `tests/`: unit tests (`test_*.py`).
- `playground/`: ad-hoc scripts and assets for quick prototyping.
- `outputs/`, `multirun/`: experiment results and Hydra outputs (git-ignored).

## Build, Test, and Dev Commands
- Install (library only): `pip install -e .`
- Install (dev tools): `pip install -e ".[develop]"`
- Format code: `./run_autoformat.sh` (black, isort, docformatter)
- CI checks locally: `./run_ci_checks.sh` (format, mypy, pylint via pytest, tests)
- Run an experiment: `python experiments/modality_disagreement.py -m prompt=a,b,c expression_input=img,txt`

OpenAI setup (required for model-backed runs):
```
export OPENAI_API_KEY=...   # optional: OPENAI_ORG_ID=...
```

## Coding Style & Naming
- Python 3.10, 4-space indentation, type hints expected (mypy enabled).
- Formatting: black (line length 88) + isort (profile=black).
- Linting: pylint (configured via `.pylintrc`).
- Names: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.

## Testing Guidelines
- Framework: pytest. Place tests under `tests/` and name `test_*.py`.
- Keep tests fast and deterministic; prefer unit tests near public APIs in `src/`.
- Run: `pytest tests/` (CI also runs `pytest --pylint` and `mypy .`).

## Commit & Pull Requests
- Messages: short, imperative, and specific (e.g., "add results visualization").
- Reference issues/PRs with `#123` when relevant; group related changes.
- PRs: include a clear description, runnable steps, and screenshots for figures/plots.
- Requirements: passing CI, no secrets committed, updated docs/examples when behavior changes.

## Security & Config Tips
- Do not hardcode keys; use environment variables and avoid committing `.env` files.
- Be mindful of data: keep large or sensitive outputs out of the repo (use `outputs/`/`multirun/`).

## Architecture Overview
- `models/` defines model interfaces (`base_model.py`) and OpenAI-backed implementations.
- Experiments configure and orchestrate runs via Hydra; results are saved under `outputs/`.

