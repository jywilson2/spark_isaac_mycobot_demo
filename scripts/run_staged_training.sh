#!/usr/bin/env bash
# Reproducible multi-stage Phase 2 training from scratch (spec.md).
#
# Stages: isaac_lab/training_recipe.py
#
# Usage:
#   ./scripts/run_staged_training.sh --headless
#   ./scripts/run_staged_training.sh --headless --skip-verify
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
exec python3 "${REPO_ROOT}/isaac_lab/run_staged_training.py" "$@"
