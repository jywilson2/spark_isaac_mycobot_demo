#!/usr/bin/env bash
# Reproducible two-phase Phase 2 PPO training (spec.md).
#
# Phase A — curriculum sampling from scratch until reach target or time budget.
# Phase B — demo-target fine-tune on the same distribution as GUI/headless demo.
#
# Usage (host or container — container auto-delegates via run_isaac_lab_training.sh):
#   ./scripts/run_two_phase_training.sh
#   ./scripts/run_two_phase_training.sh --headless
#   ./scripts/run_two_phase_training.sh --headless --skip-verify
#   ./scripts/run_two_phase_training.sh --resume   # skip phase A if checkpoint exists
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN="${REPO_ROOT}/scripts/host/run_isaac_lab_training.sh"
VERIFY="${REPO_ROOT}/scripts/verify_demo_policy.sh"

HEADLESS=0
SKIP_VERIFY=0
RESUME_PHASE_A=0
EXTRA_TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --headless)
      HEADLESS=1
      EXTRA_TRAIN_ARGS+=(--headless)
      shift
      ;;
    --skip-verify)
      SKIP_VERIFY=1
      shift
      ;;
    --resume)
      RESUME_PHASE_A=1
      shift
      ;;
    *)
      EXTRA_TRAIN_ARGS+=("$1")
      shift
      ;;
  esac
done

# Defaults from training_defaults.py (keep in sync for reproducibility).
CURRICULUM_MINUTES=90
DEMO_MINUTES=30
TARGET_RATE=0.95
EPISODE_LENGTH_S=30

echo "=== Two-phase Phase 2 training ==="
echo "Phase A: curriculum | ${CURRICULUM_MINUTES} min | target ${TARGET_RATE}"
echo "Phase B: demo sampling | ${DEMO_MINUTES} min | no early stop"
echo "Episode length: ${EPISODE_LENGTH_S}s (training = demo default)"
echo

PHASE_A_ARGS=(
  --from-scratch
  --max-duration-minutes "${CURRICULUM_MINUTES}"
  --target-reach-success-rate "${TARGET_RATE}"
  --episode-length-s "${EPISODE_LENGTH_S}"
  --target-sampling curriculum
)

if [[ "${RESUME_PHASE_A}" -eq 1 ]]; then
  echo "Skipping phase A (--resume): expecting an existing checkpoint."
  PHASE_A_ARGS=(--resume --max-duration-minutes "${CURRICULUM_MINUTES}"
    --target-reach-success-rate "${TARGET_RATE}"
    --episode-length-s "${EPISODE_LENGTH_S}"
    --target-sampling curriculum)
fi

echo "--- Phase A: curriculum training ---"
"${TRAIN}" train "${EXTRA_TRAIN_ARGS[@]}" "${PHASE_A_ARGS[@]}"

echo
echo "--- Phase B: demo-target fine-tune ---"
"${TRAIN}" train "${EXTRA_TRAIN_ARGS[@]}" \
  --resume \
  --target-sampling demo \
  --no-early-success-stop \
  --max-duration-minutes "${DEMO_MINUTES}" \
  --episode-length-s "${EPISODE_LENGTH_S}" \
  --target-reach-success-rate "${TARGET_RATE}"

if [[ "${SKIP_VERIFY}" -eq 1 ]]; then
  echo
  echo "Skipping demo verification (--skip-verify)."
  exit 0
fi

echo
echo "--- Demo verification (variable episode lengths) ---"
VERIFY_ARGS=()
if [[ "${HEADLESS}" -eq 1 ]]; then
  VERIFY_ARGS+=(--headless)
fi
exec "${VERIFY}" "${VERIFY_ARGS[@]}"
