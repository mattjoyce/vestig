# Python Constraints (Ruff-first, agent-friendly)

## Philosophy (rapid mode)
- Optimize for iteration speed and clarity.
- Fail fast, fail loudly: exceptions are fine; don’t hide errors.
- Readable > clever: code is the collaboration interface (humans + agents).

## Non-negotiables
### Ruff (format + lint)
- `ruff format .`
- `ruff check .`
- CI must enforce both.

### Repo hygiene
- `README.md` includes: setup, run, validate commands
- `pyproject.toml` owns tool config
- Prefer `src/` layout
- Keep modules shallow; avoid deep nesting

### Docstrings (short & functional)
Required for:
- every public module, class, function
- any function reused across modules

Docstrings cover:
- what it does
- inputs/outputs (high-level)
- side effects (files/network/db)
- expected failure modes (exceptions)

### Types (lightweight, real)
Type hints required for:
- public functions/classes
- module constants
- dataclasses/models
Type checking stays non-strict.

## House rules (always)
- **Imports at the top.**
- **Use `pathlib` for file operations.**

## Execution model
- **One obvious entrypoint:** `python -m package.cli ...` (or a `main()`), not scattered scripts.
- **No global side effects on import:** imports define; execution happens in `main()` / CLI.

## IO & filesystem
- **Text IO is explicit:** `encoding="utf-8"` everywhere.
- Be consistent with newline handling (pick a convention; keep it stable).

## Config & CLI ergonomics
- **Precedence:** `defaults < config < options`  
  Recommended: `defaults < config file < env vars < CLI options`
- **Config is explicit:** a single config object; no “read env anywhere”.
- Standard flags: `--config`, `--verbose`, `--debug`, `--dry-run`

## Error handling (fail hard by design)
Default:
- Let exceptions propagate.
- Use `assert` for invariants.
- Validate at system boundaries only (CLI args, file parsing, external calls).

Avoid:
- Catch-and-continue that silently masks failures.
- “return None on error” unless the contract truly means optional.
- Complex fallback logic early on.

## Logging (minimal but useful)
- Log at boundaries and major steps.
- Don’t spam logs inside loops.
- Prefer structured key/value logging when convenient.

## Testing policy (slice-level only)
We do:
- Smoke/integration tests per feature slice (“does it run?”, “typical input?”, “correctly-shaped outputs?”)
- Prefer golden files / snapshots when cheap

We avoid (for now):
- Exhaustive unit tests for tiny helpers
- Coverage-driven tests
- Heavy mocking unless unavoidable

## PR expectations
Every PR includes:
- what changed + why
- how to validate
- known limitations / TODOs
- Ruff passing
- at least one slice-level validation

Prefer small, frequent PRs.

## Determinism & dependency hygiene
- Determinism where possible: seed randomness; stable ordering when serialising.
- Standard library first unless a dependency clearly buys leverage (agents love adding libraries; keep the garden tidy).
