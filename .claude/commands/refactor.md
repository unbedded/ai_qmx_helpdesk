Refactor (no behavior change): $ARGUMENTS

- Improve clarity, naming, constants vs magic numbers, small functions.
- Preserve public API; add/adjust tests as needed to keep behavior identical.
- Follow CLAUDE.md logging/cfg patterns and STEP_ACTION_TABLE comments.
- Return/apply unified diff; run ruff + pytest; iterate to green.
