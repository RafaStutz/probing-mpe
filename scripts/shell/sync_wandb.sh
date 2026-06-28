#!/usr/bin/env bash
set -euo pipefail

echo "Syncing W&B artifacts..."

if command -v wandb >/dev/null 2>&1; then
    wandb sync "$@"
else
    echo "[ERROR] wandb command not found. Ensure the virtual environment is activated."
    exit 1
fi
