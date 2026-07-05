#!/usr/bin/env bash
# Initialize mycobot_ros2 submodule and build the Isaac Sim scene on the host.
#
# For logged iteration/debug on DGX Spark, prefer:
#   ./scripts/host/check_prereqs.sh
#   ./scripts/host/iter_urdf_import.sh
#   ./scripts/host/iter_build_isaac_scene.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if ! git submodule update --init --recursive third_party/mycobot_ros2 2>&1; then
  echo >&2
  echo "If git reported 'dubious ownership', run once on the host:" >&2
  echo "  git config --global --add safe.directory ${REPO_ROOT}" >&2
  exit 1
fi

URDF="${REPO_ROOT}/third_party/mycobot_ros2/mycobot_description/urdf/mycobot_280_m5/mycobot_280_m5.urdf"
if [[ ! -f "${URDF}" ]]; then
  echo "Missing URDF after submodule init: ${URDF}" >&2
  exit 1
fi

if [[ -z "${ISAACSIM_PATH:-}" ]] && [[ -z "${ISAACSIM_PYTHON_EXE:-}" ]]; then
  echo "Hint: run ./scripts/isaac_sim_env.sh to auto-detect, or set ISAACSIM_PATH." >&2
fi

# shellcheck source=isaac_sim_env.sh
source "${SCRIPT_DIR}/isaac_sim_env.sh"
require_isaac_python || exit 1
PYTHON_SH="${ISAACSIM_PYTHON_EXE}"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}/lib/aarch64-linux-gnu/libgomp.so.1"

echo "Using Isaac Sim: ${ISAACSIM_PATH}" >&2

exec "${PYTHON_SH}" \
  "${REPO_ROOT}/isaac_sim/build_mycobot_limo_cobot_scene.py" \
  --repo-root "${REPO_ROOT}" \
  --headless \
  --no-play \
  "$@"
