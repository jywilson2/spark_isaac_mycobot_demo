#!/usr/bin/env bash
# Phase 2 PPO training entry point — Isaac Lab on the Isaac Sim host.
#
# From the Isaac ROS container (Cursor), this script **automatically delegates**
# to the host via nsenter (see scripts/host/spark_host_exec.sh and spec.md).
#
# Usage:
#   ./scripts/run_live_training.sh
#   ./scripts/run_live_training.sh --headless --max-duration-minutes 30
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-episodes|--steps-per-episode)
      echo "Note: Isaac Lab training uses --max-duration-minutes (default 30), not episode flags." >&2
      shift 2
      ;;
    --max-iterations|--max-duration-minutes|--headless|--num-arms|--num-envs|--no-motion-glossary)
      EXTRA_ARGS+=("$1")
      if [[ "$1" == "--max-iterations" || "$1" == "--max-duration-minutes" || "$1" == "--num-arms" || "$1" == "--num-envs" ]]; then
        EXTRA_ARGS+=("$2")
        shift 2
      else
        shift
      fi
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -x "${REPO_ROOT}/scripts/host/run_isaac_lab_training.sh" ]]; then
  exec "${REPO_ROOT}/scripts/host/run_isaac_lab_training.sh" train "${EXTRA_ARGS[@]}"
fi

echo "Host training script missing: scripts/host/run_isaac_lab_training.sh" >&2
exit 1
