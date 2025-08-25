Docstring & examples audit for: $ARGUMENTS

- Ensure module/class/method docstrings follow CLAUDE.md verbosity.
- Add "Example usage" sections; replace magic numbers with constants.
- No behavior change; tests remain green.
- Unified diff; apply; ruff + pytest.
