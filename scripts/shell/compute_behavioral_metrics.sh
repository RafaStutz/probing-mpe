#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXECUTABLE="${PYTHON:-python3}"
"${PYTHON_EXECUTABLE}" scripts/compute_behavioral_metrics.py "$@"
