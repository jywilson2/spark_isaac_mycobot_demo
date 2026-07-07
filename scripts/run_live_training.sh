#!/usr/bin/env bash
# Phase 2 PPO training entry point — Isaac Lab is required (host-side GPU training).
#
# This script checks prerequisites and prints the host command. Training runs
# inside Isaac Lab on the host, not in the Isaac ROS container.
#
# Usage:
#   ./scripts/run_live_training.sh
#   ./scripts/run_live_training.sh --max-iterations 10
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-episodes|--steps-per-episode)
      echo "Note: Isaac Lab training uses --max-iterations instead of episode flags." >&2
      shift 2
      ;;
    --max-iterations)
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --headless)
      EXTRA_ARGS+=("$1")
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -x "${REPO_ROOT}/scripts/host/run_isaac_lab_training.sh" ]]; then
  if [[ -f /.dockerenv ]]; then
    echo "Isaac Lab training must run on the Isaac Sim host." >&2
    echo "From a host terminal:" >&2
    echo "  ${REPO_ROOT}/scripts/host/run_isaac_lab_training.sh train ${EXTRA_ARGS[*]}" >&2
    echo >&2
    echo "If Isaac Lab is not installed yet:" >&2
    echo "  ${REPO_ROOT}/scripts/host/install_isaac_lab.sh" >&2
    exit 1
  fi
  exec "${REPO_ROOT}/scripts/host/run_isaac_lab_training.sh" train "${EXTRA_ARGS[@]}"
fi

echo "Host training script missing: scripts/host/run_isaac_lab_training.sh" >&2
exit 1
