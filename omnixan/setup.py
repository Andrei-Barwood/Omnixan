"""
Historical packaging shim for OMNIXAN.

Authoritative packaging metadata lives in ../pyproject.toml at the repo root.
Use:

    python -m pip install -e .
    python -m pip install -e '.[cloud,distributed,quantum,dev,docs]'

from the repository root instead of invoking this file directly.
"""

from __future__ import annotations


MESSAGE = (
    "Historical packaging shim: use the repository root pyproject.toml. "
    "Run `python -m pip install -e /path/to/Omnixan` from the repo root."
)


if __name__ == "__main__":
    raise SystemExit(MESSAGE)
