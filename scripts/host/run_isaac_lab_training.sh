#!/usr/bin/env bash
# Host Isaac Lab PPO training (required Phase 2 path).
#
# Usage:
#   ./scripts/host/run_isaac_lab_training.sh check
#   ./scripts/host/run_isaac_lab_training.sh install
#   ./scripts/host/run_isaac_lab_training.sh verify
#   ./scripts/host/run_isaac_lab_training.sh train [--headless] [--num-arms N] [--max-duration-minutes M]
#   ./scripts/host/run_isaac_lab_training.sh play [--checkpoint PATH] [--episodes N]
#   ./scripts/host/run_isaac_lab_training.sh demo [--checkpoint PATH]
#   ./scripts/host/run_isaac_lab_training.sh staged [--headless]
#   ./scripts/host/run_isaac_lab_training.sh monitor [--watch]
#   ./scripts/host/run_isaac_lab_training.sh two-phase [--headless]
#   ./scripts/host/run_isaac_lab_training.sh verify-demo [--headless]
#
# Visualization (Isaac Sim GUI) is the default for train/play. Pass --headless to disable the GUI.
# Default arms: 2 with GUI, 8 headless (DGX Spark). EE cameras use --enable_cameras (auto).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# When Cursor runs inside the Isaac ROS container, delegate to the host Isaac Sim
# install automatically (same pattern as URDF probe / scene build agents).
if [[ -f /.dockerenv ]] && [[ "${SPARK_SKIP_HOST_DELEGATE:-}" != "1" ]]; then
  # shellcheck source=spark_host_exec.sh
  source "${SCRIPT_DIR}/spark_host_exec.sh"
  spark_delegate_to_host "./scripts/host/run_isaac_lab_training.sh" "$@"
  exit $?
fi

# shellcheck source=env.isaac_host.sh
source "${SCRIPT_DIR}/env.isaac_host.sh"
# shellcheck source=../../isaac_lab/versions.env
source "${REPO_ROOT}/isaac_lab/versions.env"

MODE="${1:-check}"
shift || true

spark_host_apply_env || true
export ISAACLAB_PATH="${ISAACLAB_PATH:-${SPARK_ISAACLAB_PATH}}"
export SPARK_REPO_ROOT="${REPO_ROOT}"

spark_ensure_isaac_sim_conda_stub() {
  local stub="${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh"
  if [[ -L "${ISAACLAB_PATH}/_isaac_sim" || -d "${ISAACLAB_PATH}/_isaac_sim" ]] && [[ ! -f "${stub}" ]]; then
    cat > "${stub}" <<'EOF'
#!/usr/bin/env bash
# Stub for pre-built Isaac Sim installs without bundled conda env.
return 0 2>/dev/null || exit 0
EOF
    chmod +x "${stub}"
  fi
}

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
    spark_ensure_isaac_sim_conda_stub
    if [[ ! -x "${ISAACLAB_PATH}/isaaclab.sh" ]]; then
      echo "Isaac Lab required. Run: ${REPO_ROOT}/scripts/host/install_isaac_lab.sh" >&2
      exit 1
    fi
    if [[ ! -f "${REPO_ROOT}/assets/robots/mycobot_280_m5_limo_cobot/mycobot_280_m5_limo_cobot/mycobot_280_m5_limo_cobot.usda" ]]; then
      echo "Robot USD missing. Run: ${REPO_ROOT}/scripts/host/iter_build_isaac_scene.sh" >&2
      exit 1
    fi

    TRAIN_ARGS=()
    HEADLESS=0
    VIZ_EXPLICIT=0
    ENABLE_CAMERAS=0
    NUM_ARMS_EXPLICIT=0
    DEFAULT_GUI_ARMS=2
    DEFAULT_HEADLESS_ARMS=8
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --headless)
          HEADLESS=1
          shift
          ;;
        --num-arms|--num-envs)
          NUM_ARMS_EXPLICIT=1
          TRAIN_ARGS+=("--num-arms" "$2")
          shift 2
          ;;
        --use-red-block-vision)
          ENABLE_CAMERAS=1
          TRAIN_ARGS+=("$1")
          shift
          ;;
        --viz|--visualizer)
          VIZ_EXPLICIT=1
          if [[ "${2:-}" == "none" ]]; then
            HEADLESS=1
          else
            HEADLESS=0
          fi
          TRAIN_ARGS+=("$1" "$2")
          shift 2
          ;;
        *)
          TRAIN_ARGS+=("$1")
          shift
          ;;
      esac
    done

    if [[ "${VIZ_EXPLICIT}" -eq 0 ]]; then
      if [[ "${HEADLESS}" -eq 1 ]]; then
        TRAIN_ARGS=(--viz none "${TRAIN_ARGS[@]}")
      else
        TRAIN_ARGS=(--viz kit "${TRAIN_ARGS[@]}")
      fi
    fi

    if [[ "${ENABLE_CAMERAS}" -eq 1 ]]; then
      TRAIN_ARGS=(--enable_cameras "${TRAIN_ARGS[@]}")
    fi

    if [[ "${NUM_ARMS_EXPLICIT}" -eq 0 ]]; then
      if [[ "${HEADLESS}" -eq 1 ]]; then
        TRAIN_ARGS+=("--num-arms" "${DEFAULT_HEADLESS_ARMS}")
      else
        TRAIN_ARGS+=("--num-arms" "${DEFAULT_GUI_ARMS}")
      fi
    fi

    export HEADLESS="${HEADLESS}"
    LOG_PATH="${TMPDIR:-/tmp}/spark_isaac_lab_train_$(date +%Y%m%d_%H%M%S).log"
    echo "Training log: ${LOG_PATH}"
    if [[ "${HEADLESS}" -eq 1 ]]; then
      echo "Visualization: headless (default ${DEFAULT_HEADLESS_ARMS} arms)"
    else
      echo "Visualization: Isaac Sim GUI (default ${DEFAULT_GUI_ARMS} arms)"
    fi
    (
      cd "${ISAACLAB_PATH}"
      export TERM="${TERM:-xterm-256color}"
      ./isaaclab.sh -p "${REPO_ROOT}/isaac_lab/train_ppo.py" "${TRAIN_ARGS[@]}"
    ) 2>&1 | tee -a "${LOG_PATH}"
    ;;
  play)
    spark_ensure_isaac_sim_conda_stub
    if [[ ! -x "${ISAACLAB_PATH}/isaaclab.sh" ]]; then
      echo "Isaac Lab required. Run: ${REPO_ROOT}/scripts/host/install_isaac_lab.sh" >&2
      exit 1
    fi
    PLAY_ARGS=()
    HEADLESS=0
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --headless)
          HEADLESS=1
          shift
          ;;
        --checkpoint|--seed|--robot-usd|--episode-length-s|--demo-max-episodes|--reach-tolerance-m)
          PLAY_ARGS+=("$1" "$2")
          shift 2
          ;;
        *)
          PLAY_ARGS+=("$1")
          shift
          ;;
      esac
    done
    if [[ "${HEADLESS}" -eq 0 ]]; then
      PLAY_ARGS=(--viz kit --enable_cameras "${PLAY_ARGS[@]}")
    else
      PLAY_ARGS=(--headless --viz none --enable_cameras "${PLAY_ARGS[@]}")
    fi
    echo "Playing trained policy (GUI default). Each episode randomizes EE target."
    (
      cd "${ISAACLAB_PATH}"
      export TERM="${TERM:-xterm-256color}"
      ./isaaclab.sh -p "${REPO_ROOT}/isaac_lab/play_ppo.py" "${PLAY_ARGS[@]}"
    )
    ;;
  demo)
    spark_ensure_isaac_sim_conda_stub
    if [[ ! -x "${ISAACLAB_PATH}/isaaclab.sh" ]]; then
      echo "Isaac Lab required. Run: ${REPO_ROOT}/scripts/host/install_isaac_lab.sh" >&2
      exit 1
    fi
    # shellcheck source=spark_host_exec.sh
    source "${SCRIPT_DIR}/spark_host_exec.sh"
    spark_require_gui_display "${HOME}" || exit 1
    DEMO_ARGS=(--demo --viz kit --num-arms 1)
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --checkpoint|--seed|--robot-usd|--episode-length-s)
          DEMO_ARGS+=("$1" "$2")
          shift 2
          ;;
        --headless)
          echo "demo mode requires Isaac Sim GUI; ignoring --headless" >&2
          shift
          ;;
        *)
          DEMO_ARGS+=("$1")
          shift
          ;;
      esac
    done
    echo "=== MyCobot EE reach showcase (continuous demo) ==="
    echo "Single arm | red target sphere | runs until you close Isaac Sim or Ctrl+C"
    (
      cd "${ISAACLAB_PATH}"
      export TERM="${TERM:-xterm-256color}"
      ./isaaclab.sh -p "${REPO_ROOT}/isaac_lab/play_ppo.py" "${DEMO_ARGS[@]}"
    )
    ;;
  staged)
    exec "${REPO_ROOT}/scripts/run_staged_training.sh" "$@"
    ;;
  monitor)
    exec "${REPO_ROOT}/scripts/monitor_training.sh" "$@"
    ;;
  two-phase)
    exec "${REPO_ROOT}/scripts/run_two_phase_training.sh" "$@"
    ;;
  verify-demo)
    exec "${REPO_ROOT}/scripts/verify_demo_policy.sh" "$@"
    ;;
  *)
    echo "Usage: $0 {check|install|verify|train|play|demo|staged|monitor|two-phase|verify-demo} [args...]" >&2
    exit 1
    ;;
esac
