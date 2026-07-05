#!/usr/bin/env bash
# Launch the MyCobot live Isaac Sim scene with embedded ROS 2 bridge on the host.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -z "${ISAACSIM_PATH:-}" ]] && [[ -z "${ISAACSIM_PYTHON_EXE:-}" ]]; then
  echo "Hint: run ./scripts/isaac_sim_env.sh to auto-detect, or set ISAACSIM_PATH." >&2
fi

# shellcheck source=isaac_sim_env.sh
source "${SCRIPT_DIR}/isaac_sim_env.sh"
require_isaac_python || exit 1
PYTHON_SH="${ISAACSIM_PYTHON_EXE}"

if ! git -C "${REPO_ROOT}" submodule update --init --recursive third_party/mycobot_ros2 2>&1; then
  echo >&2
  echo "If git reported 'dubious ownership', run once on the host:" >&2
  echo "  git config --global --add safe.directory ${REPO_ROOT}" >&2
  exit 1
fi

exec "${PYTHON_SH}" \
  "${REPO_ROOT}/isaac_sim/run_mycobot_live_sim.py" \
  --repo-root "${REPO_ROOT}" \
  --build-scene \
  "$@"
