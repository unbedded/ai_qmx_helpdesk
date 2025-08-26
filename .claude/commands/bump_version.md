Bump version to: $ARGUMENTS

- Use scripts/bump_version.py if present; otherwise edit __init__.py and pyproject.toml.
- Keep PEP 440 (e.g., 1.3.0.dev1, 1.3.0rc1, 1.3.0).
- Unified diff; apply; ruff + pytest; print what changed.
