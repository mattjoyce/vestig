# Rapid Python Style Guide (Ruff-first)
**Audience:** AI-assisted dev team building quickly in Python  
**Mode:** Rapid iteration, *fail-fast*, minimal scaffolding (until stable)

---

## 0) Philosophy
- **Optimize for iteration speed and clarity.**
- **Fail fast, fail loudly:** exceptions are fine; don’t hide errors.
- **Minimal correctness scaffolding:** we add robustness when the shape stabilizes.
- **Readable > clever:** code is the collaboration interface (humans + agents).

---

## 1) Non‑negotiables (team standards)

### Formatting & linting (Ruff)
- Formatting: `ruff format .`
- Linting: `ruff check .`
- CI must enforce both.
- **Definition of “quality bar”:** *Ruff clean* under our rule set.

### Docstrings
Required for:
- every **public** module, class, and function
- any function that’s reused across modules

Docstrings should be **short and functional**:
- what it does
- inputs/outputs (high-level)
- side effects (files/network/db)
- expected failure modes (exceptions)

Recommended style: **Google docstrings** (lightweight).

### Types (lightweight, real)
- Type hints required for:
  - public functions/classes
  - module constants
  - dataclasses/models
- Type checking stays **non-strict** (mypy/pyright basic).  
  Don’t fight the type system during exploration.

### Structure (baseline repo hygiene)
- `README.md` includes: setup, run, validate commands
- `pyproject.toml` owns tool config
- Prefer `src/` layout
- Keep modules shallow; avoid deep nesting

---

## 2) Testing policy (explicitly *not granular*)

### We do (Rapid Mode)
- **Smoke/integration tests** per feature slice:
  - “does it run?”
  - “does it handle a typical input?”
  - “does it produce correctly-shaped outputs?”
- Prefer *golden files / snapshots* for outputs when cheap.
- Keep tests minimal and high leverage.

### We avoid (for now)
- No exhaustive unit tests for tiny helpers.
- No tests written purely for “coverage numbers”.
- No heavy mocking unless unavoidable.

### When to increase testing
Increase testing & robustness when:
- the interface stabilizes (churn slows)
- the feature is high-impact / high-risk
- the same bug repeats
- onboarding cost rises due to fragility

---

## 3) Error handling policy (fail hard by design)

### Default behavior
- Let exceptions propagate.
- Use `assert` for invariants and assumptions.
- Validate at **system boundaries** only:
  - API inputs
  - file parsing
  - external service calls

### Avoid
- Catch-and-continue that silently masks failures.
- “return None on error” unless the contract truly means “optional result”.
- Complex fallback logic early on.

### Logging (minimal but useful)
- Log at boundaries and major steps.
- Don’t spam logs inside loops.
- Prefer structured key/value logging when convenient.

---

## 4) Ruff rule set guidance (velocity-friendly)

Enable rules that catch real problems with low noise:
- **E, F**: basic errors / pyflakes
- **I**: import sorting
- **B**: bugbear (common pitfalls)
- **UP**: modern Python upgrades
- **SIM**: simplifications
- **RUF**: ruff-specific rules

Docstrings:
- Enforce “docstrings exist” for public objects, but avoid nitpicking style early.

Optional guardrails:
- Cyclomatic complexity cap (keep functions sane)
- Line length 88 (or 100 if you prefer)

---

## 5) PR expectations (keep velocity *and* coherence)

Every PR must include:
- A short summary: **what changed + why**
- How to validate (commands or steps)
- Any known limitations / TODOs
- `ruff format` + `ruff check` passing
- At least one slice-level validation (test or runnable command)

Prefer **small, frequent PRs**.

---

## 6) Definition of Done (Rapid Mode)
A feature slice is “done” when:
- Works for the **happy path**
- Fails loudly (acceptable)
- Is documented enough that someone else can run it
- Ruff passes
- Has at least smoke-level validation

---

## 7) Commands (what devs run)
```bash
# Format (Black-compatible)
ruff format .

# Lint
ruff check .

# Auto-fix safe issues
ruff check . --fix
```

---

## Appendix A: Example `pyproject.toml` (starter)
> Tune to your preferences; this is a good “rapid mode” baseline.

```toml
[tool.ruff]
target-version = "py311"
line-length = 88

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
line-ending = "auto"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM", "RUF"]
ignore = []
fixable = ["ALL"]

# Optional: exclude generated/build folders
exclude = [
  ".git",
  ".venv",
  "venv",
  "build",
  "dist",
  "__pycache__",
]

# Optional: complexity guardrail (uncomment if desired)
# [tool.ruff.lint.mccabe]
# max-complexity = 12

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

---

## Appendix B: Mantra (print this)
- **Ruff clean, formatted, documented.**
- **Test the slice, not the atom.**
- **Fail loudly; validate at boundaries.**
- **Ship small PRs, fast.**
