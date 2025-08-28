#!/usr/bin/env python3
"""
__main__.py - Entry point for python -m ai_qmx_helpdesk
2025-08-28

This module allows running the CLI with: python -m ai_qmx_helpdesk
Automatically routes to the ragdb CLI functionality.
"""

if __name__ == "__main__":
    from .cli import main
    import sys

    # If no arguments provided, show help
    if len(sys.argv) == 1:
        sys.argv.append("--help")

    # If first argument is not a known command, assume ragdb
    known_commands = ["--help", "-h", "--debug-exec", "ragdb"]
    if len(sys.argv) > 1 and sys.argv[1] not in known_commands:
        # Insert 'ragdb' as first argument to route to RAG CLI
        sys.argv.insert(1, "ragdb")

    sys.exit(main())
