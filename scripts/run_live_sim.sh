#!/usr/bin/env bash
# Launch the MyCobot live Isaac Sim scene with embedded ROS 2 bridge on the host.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"

if [[ -z "${ISAACSIM_PATH:-}" ]]; then
  echo "Set ISAACSIM_PATH to your Isaac Sim install directory." >&2
  exit 1
fi

PYTHON_SH="${ISAACSIM_PATH}/python.sh"
if [[ ! -x "${PYTHON_SH}" ]]; then
  echo "Isaac Sim python launcher not found: ${PYTHON_SH}" >&2
  exit 1
fi

exec "${PYTHON_SH}" \
  "${REPO_ROOT}/isaac_sim/run_mycobot_live_sim.py" \
  --repo-root "${REPO_ROOT}" \
  --build-scene \
  "$@"
