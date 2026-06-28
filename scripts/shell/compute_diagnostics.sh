#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXECUTABLE="${PYTHON:-python3}"
"${PYTHON_EXECUTABLE}" scripts/compute_diagnostics_from_trajectory.py "$@"
