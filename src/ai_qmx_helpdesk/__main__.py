#!/usr/bin/env python3
"""
__main__.py - Entry point for python -m ai_qmx_helpdesk.cli
2025-08-26

This module allows running the CLI with: python -m ai_qmx_helpdesk.cli
without the RuntimeWarning about module imports.
"""

if __name__ == "__main__":
    from .cli import main
    import sys

    sys.exit(main())
