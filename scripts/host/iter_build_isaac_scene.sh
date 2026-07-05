#!/usr/bin/env bash
# Host-only: run build_isaac_scene.sh with timestamped log for iteration/debug.
#
#   ./scripts/host/iter_build_isaac_scene.sh
#   ./scripts/host/iter_build_isaac_scene.sh --with-ros2-bridge
#   ./scripts/host/iter_build_isaac_scene.sh --probe-first   # URDF probe, then full build
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.isaac_host.sh
source "${SCRIPT_DIR}/env.isaac_host.sh"

PROBE_FIRST=0
FORWARD_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --probe-first)
      PROBE_FIRST=1
      ;;
    *)
      FORWARD_ARGS+=("${arg}")
      ;;
  esac
done

spark_host_require_native_shell
spark_host_check_prereqs

if [[ "${PROBE_FIRST}" -eq 1 ]]; then
  echo "Running URDF import probe first..."
  "${SCRIPT_DIR}/iter_urdf_import.sh" || exit 1
fi

LOG_PATH="$(spark_host_new_log build_isaac_scene)"
echo "Writing log: ${LOG_PATH}"
echo "Tail with:   tail -f ${LOG_PATH}"

set +e
"${SPARK_REPO_ROOT}/scripts/build_isaac_scene.sh" "${FORWARD_ARGS[@]}" 2>&1 | tee "${LOG_PATH}"
exit_code="${PIPESTATUS[0]}"
set -e

if [[ "${exit_code}" -ne 0 ]]; then
  echo "Scene build FAILED (exit ${exit_code})" >&2
  spark_host_print_log_tail "${LOG_PATH}" 80
  exit "${exit_code}"
fi

echo "Scene build succeeded."
echo "Log: ${LOG_PATH}"
ls -la "${SPARK_REPO_ROOT}/assets/scenes/"*.usd 2>/dev/null || true
