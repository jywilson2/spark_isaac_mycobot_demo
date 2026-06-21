#!/usr/bin/env bash
# Initialize mycobot_ros2 submodule and build the Isaac Sim scene on the host.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
git submodule update --init --recursive third_party/mycobot_ros2

URDF="${REPO_ROOT}/third_party/mycobot_ros2/mycobot_description/urdf/mycobot_280_m5/mycobot_280_m5.urdf"
if [[ ! -f "${URDF}" ]]; then
  echo "Missing URDF after submodule init: ${URDF}" >&2
  exit 1
fi

if [[ -z "${ISAACSIM_PATH:-}" ]]; then
  echo "Set ISAACSIM_PATH to your Isaac Sim install directory." >&2
  exit 1
fi

PYTHON_SH="${ISAACSIM_PATH}/python.sh"
if [[ ! -x "${PYTHON_SH}" ]]; then
  echo "Isaac Sim python launcher not found: ${PYTHON_SH}" >&2
  exit 1
fi

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"

exec "${PYTHON_SH}" \
  "${REPO_ROOT}/isaac_sim/build_mycobot_limo_cobot_scene.py" \
  --repo-root "${REPO_ROOT}" \
  "$@"
