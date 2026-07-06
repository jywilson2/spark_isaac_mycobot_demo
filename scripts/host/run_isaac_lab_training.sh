#!/usr/bin/env bash
# Host Isaac Lab PPO training (required Phase 2 path).
#
# Usage:
#   ./scripts/host/run_isaac_lab_training.sh check
#   ./scripts/host/run_isaac_lab_training.sh install
#   ./scripts/host/run_isaac_lab_training.sh verify
#   ./scripts/host/run_isaac_lab_training.sh train [--headless] [--max-iterations N]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=env.isaac_host.sh
source "${SCRIPT_DIR}/env.isaac_host.sh"
# shellcheck source=../../isaac_lab/versions.env
source "${REPO_ROOT}/isaac_lab/versions.env"

MODE="${1:-check}"
shift || true

spark_host_apply_env || true
export ISAACLAB_PATH="${ISAACLAB_PATH:-${SPARK_ISAACLAB_PATH}}"
export SPARK_REPO_ROOT="${REPO_ROOT}"

case "${MODE}" in
  check)
    echo "=== Isaac Sim ==="
    spark_host_check_prereqs || true
    echo
    echo "=== Isaac Lab (required) ==="
    if [[ -x "${ISAACLAB_PATH}/isaaclab.sh" ]]; then
      (
        cd "${ISAACLAB_PATH}"
        ./isaaclab.sh -p "${REPO_ROOT}/isaac_lab/detect_isaac_lab.py"
      ) || true
    else
      python3 "${REPO_ROOT}/isaac_lab/detect_isaac_lab.py" || true
      echo
      echo "Install Isaac Lab: ${REPO_ROOT}/scripts/host/install_isaac_lab.sh"
    fi
    ;;
  install)
    exec "${SCRIPT_DIR}/install_isaac_lab.sh" "$@"
    ;;
  verify)
    exec "${SCRIPT_DIR}/verify_isaac_lab.sh" "$@"
    ;;
  train)
    if [[ ! -x "${ISAACLAB_PATH}/isaaclab.sh" ]]; then
      echo "Isaac Lab required. Run: ${REPO_ROOT}/scripts/host/install_isaac_lab.sh" >&2
      exit 1
    fi
    if [[ ! -f "${REPO_ROOT}/assets/robots/mycobot_280_m5_limo_cobot/mycobot_280_m5_limo_cobot/mycobot_280_m5_limo_cobot.usda" ]]; then
      echo "Robot USD missing. Run: ${REPO_ROOT}/scripts/host/iter_build_isaac_scene.sh" >&2
      exit 1
    fi
    LOG_PATH="${TMPDIR:-/tmp}/spark_isaac_lab_train_$(date +%Y%m%d_%H%M%S).log"
    echo "Training log: ${LOG_PATH}"
    (
      cd "${ISAACLAB_PATH}"
      export TERM="${TERM:-xterm-256color}"
      ./isaaclab.sh -p "${REPO_ROOT}/isaac_lab/train_ppo.py" "$@"
    ) 2>&1 | tee -a "${LOG_PATH}"
    ;;
  *)
    echo "Usage: $0 {check|install|verify|train} [args...]" >&2
    exit 1
    ;;
esac
