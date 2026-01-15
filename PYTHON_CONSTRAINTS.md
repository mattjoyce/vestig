# Python Constraints (Ruff-first, agent-friendly)

## Goal
- Provide a clear, enforceable policy for Python code quality, tooling, and interfaces.

## Policy language
- MUST: mandatory requirement.
- SHOULD: strong default; deviations require explicit justification.
- MAY: optional; use when it adds clear value.

## Assumptions
- Applies to Python modules, tools, and scripts in this repo.
- Tooling runs in local dev and CI environments.

## Principles (rapid mode)
- Optimize for iteration speed and clarity.
- Fail fast, fail loudly: exceptions are fine; do not hide errors.
- Readable > clever: code is the collaboration interface (humans + agents).

## MUST requirements
### Ruff (format + lint)
- MUST run `ruff format .`
- MUST run `ruff check .`
- CI MUST enforce both.

### Default pre-commit checks (unless overridden)
- SHOULD run `ruff check` (use `--fix` only when agreed) and `ruff format` if formatting is enforced.
- SHOULD run `mypy` on touched modules or a fast subset when types are relevant.
- SHOULD run `deadcode` only for cleanup/refactor work; MAY skip for small behavior changes.

### Repo hygiene
- MUST ensure `README.md` includes: setup, run, validate commands
- MUST ensure `pyproject.toml` owns tool config
- SHOULD use `src/` layout
- SHOULD keep modules shallow; avoid deep nesting

### Docstrings (short & functional)
Required for:
- MUST document every public module, class, function
- MUST document any function reused across modules

Docstrings cover:
- SHOULD cover what it does
- SHOULD cover inputs/outputs (high-level)
- SHOULD cover side effects (files/network/db)
- SHOULD cover expected failure modes (exceptions)

### Types (lightweight, real)
Type hints required for:
- MUST type public functions/classes
- MUST type module constants
- MUST type dataclasses/models
- SHOULD keep type checking non-strict.

## House rules (always)
- MUST keep imports at the top.
- MUST use `pathlib` for file operations.
- MUST add `if __name__ == "__main__":` guard to runnable modules.
- MUST add `from __future__ import annotations` in new/edited modules.
- MUST avoid `..` relative imports outside a package.
- MUST define `__all__` explicitly in public modules.
- SHOULD prefer immutable data via `dataclass(frozen=True)` or `typing.NamedTuple`.

## Execution model
- MUST provide one obvious entrypoint: `python -m package.cli ...` (or a `main()`), not scattered scripts.
- MUST avoid global side effects on import; imports define and execution happens in `main()` / CLI.

## IO & filesystem
- MUST make text IO explicit: `encoding="utf-8"` everywhere.
- SHOULD be consistent with newline handling (pick a convention; keep it stable).

## Config & CLI ergonomics
- MUST follow precedence: `defaults < config < options`  
  Recommended: `defaults < config file < env vars < CLI options`
- MUST keep config explicit: a single config object; no "read env anywhere".
- SHOULD support standard flags: `--config`, `--verbose`, `--debug`, `--dry-run`

## Defaults & required values (fail-fast)
- Agents and templates MUST NOT silently invent important defaults for values that affect correctness, security, or observability.
- For any required configuration or parameter, prefer failing fast and loudly over using a guessed default.
  - Detect missing required values early (configuration loading / validation) and raise a clear, specific exception (e.g. `ValueError`, `ConfigurationError`).
  - Avoid placeholder defaults like `"TODO"`, `"REPLACE_ME"`, `"dummy"`, or empty strings that let code continue with unsafe assumptions.
- Safe, well-reasoned defaults are acceptable for non-critical settings (timeouts, retries, verbosity), but document why a default is safe.
- Use explicit sentinels or validation to make "requiredness" visible in code and tests.

Example pattern:
```python
from dataclasses import dataclass

REQUIRED = object()

@dataclass
class Config:
    api_key: str = REQUIRED  # required — no safe default
    timeout: int = 30        # safe default

def validate_config(cfg: Config) -> None:
    if cfg.api_key is REQUIRED:
        raise ValueError("config.api_key is required and must be provided via config/env/CLI")
```

- CI / PR checks: prefer a quick check that flags common placeholder defaults or sentinel values left in committed config templates.
- For agents: when generating configuration or scaffolding, emit explicit REQUIRED markers and accompanying validation code rather than guessing missing values.

## Error handling (fail hard by design)
Default:
- SHOULD let exceptions propagate.
- SHOULD use `assert` for invariants.
- SHOULD validate at system boundaries only (CLI args, file parsing, external calls).

Avoid:
- MUST NOT catch-and-continue in ways that silently mask failures.
- MUST NOT "return None on error" unless the contract truly means optional.
- SHOULD avoid complex fallback logic early on.

## Logging (minimal but useful)
- SHOULD log at boundaries and major steps.
- SHOULD avoid log spam inside loops.
- SHOULD prefer structured key/value logging when convenient.

## Testing policy (slice-level only)
We do:
- SHOULD add smoke/integration tests per feature slice ("does it run?", "typical input?", "correctly-shaped outputs?")
- SHOULD prefer golden files / snapshots when cheap

We avoid (for now):
- SHOULD NOT add exhaustive unit tests for tiny helpers
- SHOULD NOT add coverage-driven tests
- SHOULD avoid heavy mocking unless unavoidable

## PR expectations
Every PR includes:
- MUST include what changed + why
- MUST include how to validate
- MUST include known limitations / TODOs
- MUST include Ruff passing
- MUST include at least one slice-level validation

Prefer small, frequent PRs.

## Determinism & dependency hygiene
- SHOULD favor determinism where possible: seed randomness; stable ordering when serialising.
- SHOULD default to standard library first unless a dependency clearly buys leverage (agents love adding libraries; keep the garden tidy).
