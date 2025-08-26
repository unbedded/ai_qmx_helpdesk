Implement a new feature: $ARGUMENTS

- Follow repo House Rules in CLAUDE.md strictly (headers with <DATE>, Example usage, STEP_ACTION_TABLE, __init__ logging first, cfg_dict pattern, verbose docstrings, stdlib only).
- Output a SINGLE unified diff patch covering:
  - src/ai_qmx_helpdesk/<module>.py (implementation)
  - tests/test_<module>.py (tests)
  - src/ai_qmx_helpdesk/__init__.py (exports)
  - CHANGELOG.md ([Unreleased] → Added entry)
- Save patch to tmp_patch.diff, apply it, then:
  git apply -p0 tmp_patch.diff || git apply tmp_patch.diff
  ruff format . && ruff check --fix .
  pytest -q
- If anything fails, revise the patch and retry until green.
- Print a short summary of changes and follow-up commands (gh pr create …).
