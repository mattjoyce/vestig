# EPHEMERAL Temporal Stability Plan

## Objective
Add a new `temporal_stability=ephemeral` bucket to distinguish facts expected to change soon from general dynamic facts, and apply a faster TraceRank decay for EPHEMERAL while keeping existing behavior unchanged for static/dynamic/unknown.

## Scope (M4-safe)
- Additive only: no new CLI flags, no auto-expiry, no backfill of existing rows.
- Preserve existing interfaces and default `unknown` behavior.
- Make the new label observable in recall and memory inspection output.

## Definition
- **static**: permanent facts (never change).
- **dynamic**: could change over time, not necessarily soon.
- **ephemeral**: expected to change soon (hours/days), explicitly indicated in text.
- **unknown**: mixed or unclear.

## Proposed Changes
1) **Schema/validation acceptance**
   - Allow `ephemeral` wherever `temporal_stability` is validated or stored.
   - Do not migrate existing rows.

2) **Prompt guidance**
   - Update `extract_memories_from_session` prompt to include EPHEMERAL guidance and conservative examples.
   - Require explicit near-term markers (e.g., "today", "this week", "temporary", "incident ongoing", "on-call this week").

3) **TraceRank decay (EPHEMERAL only)**
   - Use a shorter tau for EPHEMERAL and leave other stabilities unchanged.
   - Example: `tau_ephemeral = max(3.0, min(7.0, tau_days / 4))`.

4) **Observability**
   - Show `temporal_stability` in `memory show` output.
   - Show `temporal_stability` in recall output (tag or metadata field).

## Acceptance Checks
- `memory show` includes `temporal_stability` and displays `ephemeral` when present.
- `memory recall` includes `temporal_stability` and displays `ephemeral` when present.
- TraceRank: with identical event histories, EPHEMERAL decays faster than dynamic.
- Existing TraceRank tests pass unchanged; add 1-2 targeted EPHEMERAL checks.

## Tests
- Add a new TraceRank test that fixes `now`, creates two event histories with identical timestamps, and asserts:
  - `multiplier(ephemeral) < multiplier(dynamic)`
  - `multiplier(ephemeral) < multiplier(unknown)`

## Non-Goals
- No automatic reclassification of existing memories.
- No auto-expiration logic (`t_expired`) changes.
- No new CLI flags.

## Risks
- Over-classification of EPHEMERAL (mitigate via conservative prompt rules).
- If a DB CHECK constraint exists for stability values, update or relax it.
- Avoid altering static/dynamic decay to prevent ranking regressions.

## Files Likely Touched
- `src/vestig/core/models.py`
- `src/vestig/core/prompts.yaml`
- `src/vestig/core/tracerank.py`
- `src/vestig/core/cli.py` (recall/memory show formatting)
- `tests/test_tracerank.py`
