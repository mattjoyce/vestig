# Guiding Principles

## The mantra
**Earn complexity.**

## Progressive maturation
This project stays coherent (and enjoyable) by treating capability as something we earn: each stage must produce a stable, usable slice that becomes the foundation for the next.

**Rule of thumb:**
> Build the smallest end-to-end loop that proves the next claim, then lock the interfaces, then deepen.

## What “Earn complexity” means in practice
1. **Quality at the boundary beats cleverness in the middle.**
2. **Stable interfaces = sovereignty.** Guard the CLI contract and schema shape.
3. **One thin vertical slice, then deepen.** Add → store → recall (or equivalent backbone) comes first.
4. **Observability is a feature.** Prefer “I can inspect why” over “it seems smart.”
5. **Explicit done-ness per milestone.** Each ends with acceptance checks you can run in minutes.
6. **Agents ship tasks; you ship direction.** One agent task = one capability + one acceptance test + one boundary.

## Agent tasking template
Use this when assigning work to coding agents:

- **Objective:** one capability only
- **Boundary:** which files/modules may change
- **Acceptance checks:** 3–5 bullet checks
- **Non-goals:** explicit out-of-scope items
- **Demo:** minimal script or command sequence that proves it works

## Closing reminder
If you feel the temptation to “just add one more feature” mid-stage, pause and ask:

> Have we earned this complexity yet?
