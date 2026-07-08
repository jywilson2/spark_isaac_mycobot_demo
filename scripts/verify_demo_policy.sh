#!/usr/bin/env bash
# Headless demo regression: verify trained policy at multiple episode lengths.
#
# Success gate: >= 95% reach at each episode length (spec.md).
#
# Usage:
#   ./scripts/verify_demo_policy.sh
#   ./scripts/verify_demo_policy.sh --headless --checkpoint assets/checkpoints/isaac_lab_ppo/latest_policy
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PLAY="${REPO_ROOT}/scripts/host/run_isaac_lab_training.sh"

HEADLESS=1
CHECKPOINT="${REPO_ROOT}/assets/checkpoints/isaac_lab_ppo/latest_policy"
EPISODES=100
MIN_RATE=0.95
LENGTHS=(20 30 40)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --headless)
      HEADLESS=1
      shift
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --episodes)
      EPISODES="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

PLAY_ARGS=(--demo --demo-max-episodes "${EPISODES}" --num-arms 1 --checkpoint "${CHECKPOINT}")
if [[ "${HEADLESS}" -eq 1 ]]; then
  PLAY_ARGS=(--headless "${PLAY_ARGS[@]}")
fi

OVERALL_OK=1
for length_s in "${LENGTHS[@]}"; do
  echo
  echo "=== Demo verify: episode_length_s=${length_s}, episodes=${EPISODES} ==="
  LOG="${TMPDIR:-/tmp}/spark_demo_verify_${length_s}s_$(date +%Y%m%d_%H%M%S).log"
  set +e
  "${PLAY}" play "${PLAY_ARGS[@]}" --episode-length-s "${length_s}" 2>&1 | tee "${LOG}"
  PLAY_EXIT=$?
  set -e
  if [[ "${PLAY_EXIT}" -ne 0 ]]; then
    echo "FAIL: play exited ${PLAY_EXIT} at ${length_s}s" >&2
    OVERALL_OK=0
    continue
  fi
  # Parse "Demo ended: N targets attempted, M reached (X.X%)."
  RATE_LINE="$(grep -E 'Demo ended:.*reached' "${LOG}" | tail -1 || true)"
  if [[ -z "${RATE_LINE}" ]]; then
    echo "FAIL: could not parse demo success rate from ${LOG}" >&2
    OVERALL_OK=0
    continue
  fi
  RATE_PCT="$(echo "${RATE_LINE}" | sed -n 's/.*(\([0-9.]*\)%).*/\1/p')"
  if [[ -z "${RATE_PCT}" ]]; then
    echo "FAIL: could not extract success percent from: ${RATE_LINE}" >&2
    OVERALL_OK=0
    continue
  fi
  RATE="$(python3 -c "print(float('${RATE_PCT}') / 100.0)")"
  OK="$(python3 -c "print(1 if float('${RATE}') >= float('${MIN_RATE}') else 0)")"
  if [[ "${OK}" -ne 1 ]]; then
    echo "FAIL: ${RATE_PCT}% < 95% at ${length_s}s episode length" >&2
    OVERALL_OK=0
  else
    echo "PASS: ${RATE_PCT}% >= 95% at ${length_s}s"
  fi
done

if [[ "${OVERALL_OK}" -ne 1 ]]; then
  echo
  echo "Demo verification FAILED (see logs above)." >&2
  exit 1
fi

echo
echo "Demo verification PASSED at all episode lengths: ${LENGTHS[*]}s"
