Add config key (and docs/tests): $ARGUMENTS
# $ARGUMENTS = <key_name>

- Update relevant module(s) to accept cfg_dict["$ARGUMENTS"] with sane defaults.
- Log config defaults/mismatches in __init__ (per CLAUDE.md).
- Tests covering default and custom values; update README and CHANGELOG.
- Unified diff; apply; ruff + pytest; iterate.
