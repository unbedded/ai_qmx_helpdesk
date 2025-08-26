Hotfix for: $ARGUMENTS

- Branch from main; fix the specific bug; add targeted tests.
- Bump versions to next PATCH in __init__.py and pyproject.toml.
- CHANGELOG.md: add [X.Y.(Z+1)] with “Fixed”.
- Unified diff; apply; ruff + pytest to green.
- Print follow-ups:
  git switch main && git pull --ff-only
  git switch -c hotfix/vX.Y.(Z+1) && git push -u origin HEAD
  gh pr create --base main --head hotfix/vX.Y.(Z+1) --title "Hotfix vX.Y.(Z+1)" --fill --assignee @me
