Prepare release: $ARGUMENTS
# $ARGUMENTS = X.Y.Z (e.g., 1.2.0 or 1.2.0rc1)

- Update src/ai_qmx_helpdesk/__init__.py __version__ to "$ARGUMENTS"
- Update pyproject.toml version to "$ARGUMENTS"
- Move CHANGELOG.md "Unreleased" entries under [$ARGUMENTS] dated <TODAY>; leave a fresh [Unreleased] stub.
- If CI workflow missing, add .github/workflows/ci.yml (ruff + pytest on Python 3.13).
- Unified diff, apply, run ruff + pytest.
- Print follow-ups:
  git switch develop && git pull --ff-only
  git switch -c release/v$ARGUMENTS && git push -u origin HEAD
  gh pr create --base main --head release/v$ARGUMENTS --title "Release $ARGUMENTS" --fill --assignee @me
