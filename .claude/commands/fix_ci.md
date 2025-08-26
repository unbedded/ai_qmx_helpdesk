Fix CI/lint/test failures for: $ARGUMENTS

- Use the latest ruff/pytest output (paste/collect from logs).
- Minimal changes only (NO new features).
- Keep CLAUDE.md standards (headers, STEP_ACTION_TABLE, cfg/logging pattern).
- Return a SINGLE unified diff patch; save/apply as tmp_patch.diff; run:
  ruff format . && ruff check --fix . && pytest -q
- Iterate until green; print a bullet list of what changed and why.
