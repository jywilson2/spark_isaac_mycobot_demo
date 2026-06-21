#!/usr/bin/env bash
# Run live Phase 1/2 integration tests against a playing Isaac Sim scene.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WS_ROOT="${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"

source /opt/ros/jazzy/setup.bash
cd "${WS_ROOT}"
colcon build --packages-select spark_verify_pkg
source install/setup.bash

echo "Probing Isaac Sim /clock on ROS_DOMAIN_ID=${ROS_DOMAIN_ID}..."
if ! timeout 8 ros2 topic echo /clock --once >/dev/null 2>&1; then
  echo "Isaac Sim live bridge not detected. Start host live sim first:" >&2
  echo "  ${REPO_ROOT}/scripts/run_live_sim.sh" >&2
  exit 1
fi

colcon test --packages-select spark_verify_pkg \
  --ctest-args -R "test_phase1_live_integration|test_phase2_live_integration|test_live_sim_gate|test_safety_boundary_evaluator_py" \
  --event-handlers console_direct+
colcon test-result --all
