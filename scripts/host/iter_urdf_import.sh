#!/usr/bin/env bash
# Host-only: minimal URDF import probe with timestamped log (fast iteration).
#
#   ./scripts/host/iter_urdf_import.sh
#   ./scripts/host/iter_urdf_import.sh --keep-prepared   # skip mesh copy if unchanged
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.isaac_host.sh
source "${SCRIPT_DIR}/env.isaac_host.sh"

spark_host_require_native_shell
spark_host_check_prereqs

LOG_PATH="$(spark_host_new_log urdf_import_probe)"
echo "Writing log: ${LOG_PATH}"
echo "Tail with:   tail -f ${LOG_PATH}"

set +e
spark_host_run_python \
  "${SPARK_REPO_ROOT}/isaac_sim/probe_urdf_import.py" \
  --repo-root "${SPARK_REPO_ROOT}" \
  "$@" 2>&1 | tee "${LOG_PATH}"
exit_code="${PIPESTATUS[0]}"
set -e

if [[ "${exit_code}" -ne 0 ]]; then
  echo "URDF import probe FAILED (exit ${exit_code})" >&2
  spark_host_print_log_tail "${LOG_PATH}" 60
  exit "${exit_code}"
fi

robot_usd="${SPARK_REPO_ROOT}/assets/robots/mycobot_280_m5_limo_cobot/mycobot_280_m5_limo_cobot.usd"
robot_dir="${SPARK_REPO_ROOT}/assets/robots/mycobot_280_m5_limo_cobot"
if ! find "${robot_dir}" -maxdepth 3 -name '*.usd*' -print -quit | grep -q .; then
  echo "URDF import probe FAILED: no robot USD produced" >&2
  spark_host_print_log_tail "${LOG_PATH}" 60
  exit 1
fi

echo "URDF import probe succeeded."
echo "Log: ${LOG_PATH}"
find "${robot_dir}" -maxdepth 3 -name '*.usd*' -ls 2>/dev/null || true
