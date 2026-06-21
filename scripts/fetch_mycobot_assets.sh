#!/usr/bin/env bash
# Clone/update the mycobot_ros2 submodule (Limo Cobot arm = myCobot 280 M5).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
git submodule update --init --recursive third_party/mycobot_ros2

URDF="${REPO_ROOT}/third_party/mycobot_ros2/mycobot_description/urdf/mycobot_280_m5/mycobot_280_m5.urdf"
if [[ -f "${URDF}" ]]; then
  echo "mycobot_ros2 ready: ${URDF}"
else
  echo "Submodule initialized but URDF missing: ${URDF}" >&2
  exit 1
fi
