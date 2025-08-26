Add CLI subcommand: $ARGUMENTS

- Implement under src/ai_qmx_helpdesk/cli.py (or create it) with argparse.
- Subcommand name and behavior = $ARGUMENTS; include help text and examples.
- Export CLI entry in __init__.py; add tests tests/test_cli_$ARGUMENTS.py
- Update README.md (Usage) and CHANGELOG.md (Unreleased → Added).
- Return/apply unified diff; run ruff + pytest; iterate to green.
