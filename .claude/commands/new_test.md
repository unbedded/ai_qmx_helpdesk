Create/extend tests for: $ARGUMENTS

- Add or update tests under tests/test_$ARGUMENTS.py
- Achieve coverage for edge cases and error paths; keep stdlib only.
- Single unified diff patch, apply, run: ruff + pytest; iterate to green.
- Keep CLAUDE.md docstyle (docstrings/examples where helpful).
