# Claude Commands

Placeholders:
- `$ARGUMENTS` is replaced by what you type after selecting a command in Claude Code.

Usage:
1. Open Claude Code in your repo root.
2. Run a command from `.claude/commands/` (e.g., `feature.md`) and supply `$ARGUMENTS`.
3. Most commands expect/produce a unified diff, then auto-run: ruff + pytest.
4. For multi-file changes, prefer `feature.md` / `refactor.md`. For releases, use `release_prep.md` or `hotfix.md`.
