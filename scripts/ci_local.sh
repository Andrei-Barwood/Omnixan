#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-baseline}"

case "${MODE}" in
  baseline)
    exec "${PYTHON_BIN}" -m omnixan validate --json
    ;;
  optional-smokes)
    exec "${PYTHON_BIN}" -m omnixan validate --json --include-optional-smokes
    ;;
  full)
    exec "${PYTHON_BIN}" -m omnixan validate --json --include-optional-smokes --strict-environment
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Usage: $0 [baseline|optional-smokes|full]" >&2
    exit 2
    ;;
esac
