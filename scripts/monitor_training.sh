#!/usr/bin/env bash
# Monitor an in-progress Isaac Lab training run (reach success, iteration, stage).
#
# Usage:
#   ./scripts/monitor_training.sh
#   ./scripts/monitor_training.sh /tmp/spark_isaac_lab_train_20260708.log
#   ./scripts/monitor_training.sh --watch   # refresh every 10s
set -euo pipefail

WATCH=0
LOG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --watch|-w)
      WATCH=1
      shift
      ;;
    *)
      LOG="$1"
      shift
      ;;
  esac
done

if [[ -z "${LOG}" ]]; then
  LOG="$(ls -t /tmp/spark_isaac_lab_train_*.log 2>/dev/null | head -1 || true)"
fi

if [[ -z "${LOG}" || ! -f "${LOG}" ]]; then
  echo "No training log found. Pass a path or start training first." >&2
  exit 1
fi

show_status() {
  clear 2>/dev/null || true
  echo "=== MyCobot PPO training monitor ==="
  echo "Log: ${LOG}"
  echo "Updated: $(date)"
  echo
  if grep -q "Training sequence complete" "${LOG}"; then
    grep -E "Stop reason:|Reach success rate:|Task requirement met:|Duration limit:" "${LOG}" | tail -4 || true
    echo
  fi
  echo "--- Latest iterations ---"
  grep -E "reach_success_rate=|Learning iteration|curriculum=" "${LOG}" | tail -12 || true
  echo
  if [[ -f "${REPO_ROOT:-}/assets/checkpoints/isaac_lab_ppo/training_summary.json" ]]; then
    echo "--- training_summary.json ---"
    python3 -m json.tool "${REPO_ROOT}/assets/checkpoints/isaac_lab_ppo/training_summary.json" 2>/dev/null \
      | grep -E "reach_success_rate|target_reach_tolerance_m|stop_reason|target_sampling" || true
  fi
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${WATCH}" -eq 1 ]]; then
  while true; do
    show_status
    sleep 10
  done
else
  show_status
fi
