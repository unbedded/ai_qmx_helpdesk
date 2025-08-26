Add CI workflow

- Create .github/workflows/ci.yml:
  - on: push, pull_request
  - python-version: 3.13
  - steps: checkout, setup-python, pip install -e .[test] ruff pytest, ruff check ., ruff format --check ., pytest -q
- Unified diff; apply; ensure the workflow passes local checks.
