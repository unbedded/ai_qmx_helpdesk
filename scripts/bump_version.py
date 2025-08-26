#!/usr/bin/env python3
"""bump_version.py — update project version across files.

Usage:
    ./scripts/bump_version.py X.Y.Z[rcN|devN]

This updates:
  - src/ai_qmx_helpdesk/__init__.py  (__version__ = "X.Y.Z")
  - pyproject.toml                    (version = "X.Y.Z")

Exit codes:
  0 on success
  2 on usage or missing matches
"""

import re
import sys
import pathlib

root = pathlib.Path(__file__).resolve().parents[1]
init = root / "src" / "ai_qmx_helpdesk" / "__init__.py"
pyproject = root / "pyproject.toml"


def sub_file(p: pathlib.Path, pattern: str, repl: str) -> None:
    s = p.read_text(encoding="utf-8")
    s2, n = re.subn(pattern, repl, s, flags=re.M)
    if n == 0:
        print(f"No version field matched in {p}", file=sys.stderr)
        sys.exit(2)
    p.write_text(s2, encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: bump_version.py X.Y.Z[rcN|devN]", file=sys.stderr)
        return 2
    ver = sys.argv[1]
    sub_file(init, r'__version__\s*=\s*"[^\"]+"', f'__version__ = "{ver}"')
    sub_file(pyproject, r'^version\s*=\s*"[^\"]+"', f'version = "{ver}"')
    print(f"Bumped to {ver}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
