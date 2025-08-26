Apply a unified diff patch provided inline

- Accept a ```diff fenced block in the reply.
- Save as tmp_patch.diff; try:
  git apply -p0 tmp_patch.diff || git apply tmp_patch.diff
- Run:
  ruff format . && ruff check --fix . && pytest -q
- If failures, revise the patch and retry until green; print summary.
