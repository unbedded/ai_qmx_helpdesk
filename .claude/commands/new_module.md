Create a new Python module named: $ARGUMENTS

- Follow the CLAUDE.md standards strictly.
- Output YAML with a single field `py:` containing the complete module source.
- Save YAML as `tmp_$ARGUMENTS.yaml`.
- Convert to code:
  python tools/yaml_to_py.py tmp_$ARGUMENTS.yaml src/ai_qmx_helpdesk/$ARGUMENTS.py
- Create or update a test: tests/test_$ARGUMENTS.py
- Run: ruff format; ruff check --fix; mypy .; pytest -q
- If any step fails, fix and repeat until tests pass.
