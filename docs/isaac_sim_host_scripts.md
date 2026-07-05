# Isaac Sim Host Scripts

This guide documents shell helpers under `scripts/` and `scripts/host/` used to build the MyCobot Isaac Sim scene on the **DGX Spark host**. Isaac Sim does **not** run inside the Isaac ROS Docker container; these scripts must be executed from a **native host terminal** (or an equivalent host mount namespace), not from Cursor's container shell.

For the full host ↔ container workflow, see [README.md](../README.md#development-workflow-isaac-ros--cursor-on-dgx-spark).

## Why host-only?

| Component | Runs where | Reason |
|-----------|------------|--------|
| Isaac Sim (`python.sh`, GUI/headless kit) | **Host** | GPU + Omniverse install lives on the host filesystem |
| ROS 2 nodes, `colcon test`, live launch files | **Isaac ROS container** (Cursor) | Jazzy toolchain and workspace bind-mount |
| Scene USD build, URDF → USD import | **Host** | Requires Isaac Sim Python API |

If you run host scripts inside the container, they detect `/.dockerenv` and warn. The container has no usable Isaac Sim install at standard paths.

## Environment detection

### `scripts/isaac_sim_env.sh`

Shared helper sourced by other scripts. Resolves `ISAACSIM_PYTHON_EXE` and `ISAACSIM_PATH` by probing, in order:

1. `$ISAACSIM_PATH/python.sh` (if `ISAACSIM_PATH` is set)
2. `$ISAACSIM_PYTHON_EXE` (if set and executable)
3. `$HOME/IsaacSim/_build/linux-aarch64/release/python.sh` (DGX Spark source build)
4. `$HOME/isaacsim/python.sh` (pre-built binary install)
5. `$HOME/isaac-sim/python.sh`
6. `/opt/nvidia/isaac-sim/python.sh`

**Print detected paths:**

```bash
./scripts/isaac_sim_env.sh
```

**Override** when auto-detection fails:

```bash
export ISAACSIM_PATH="$HOME/isaacsim"          # directory containing python.sh
# or
export ISAACSIM_PYTHON_EXE="$HOME/isaacsim/python.sh"
```

Persist host exports in `~/.bashrc` (see README).

### `scripts/host/env.isaac_host.sh`

Host iteration library (sourced, not run directly). Provides:

| Function | Purpose |
|----------|---------|
| `spark_host_require_native_shell` | Warns if `/.dockerenv` is present |
| `spark_host_apply_env` | Sets `ROS_DOMAIN_ID`, `FASTDDS_BUILTIN_TRANSPORTS`, `PYTHONPATH`, `LD_PRELOAD`, log dir |
| `spark_host_check_prereqs` | Verifies Isaac Sim python + submodule URDF exist |
| `spark_host_new_log` | Creates timestamped log under `assets/logs/isaac_host/` |
| `spark_host_print_log_tail` | Prints log tail + error grep on failure |
| `spark_host_run_python` | Runs a script via `${ISAACSIM_PYTHON_EXE}` |

Logs are written to `assets/logs/isaac_host/` (gitignored). Symlinks `latest.log` and `latest_<label>.log` point at the most recent run.

## Script reference

### Container scripts (run in Cursor)

| Script | When to use |
|--------|-------------|
| `scripts/fetch_mycobot_assets.sh` | One-time or after clone — initializes `third_party/mycobot_ros2` submodule |
| `scripts/source_container_env.sh` | Every container session — sets `ROS_DOMAIN_ID=42`, `FASTDDS_BUILTIN_TRANSPORTS=UDPv4` |
| `scripts/fix_repo_permissions.sh` | When host `git` fails with permission errors on `.git/` |

### Host production scripts (run on native host)

| Script | When to use |
|--------|-------------|
| `scripts/build_isaac_scene.sh` | **Production** scene build after URDF/asset changes. Initializes submodule if needed, runs `isaac_sim/build_mycobot_limo_cobot_scene.py` headless. |
| `scripts/run_live_sim.sh` | **Daily live work** — loads scene, enables ROS 2 bridge, presses Play. Keep running while testing in container. |
| `scripts/run_live_tests.sh` | Run live integration tests from container (calls colcon test; tests auto-skip if sim absent). |

### Host iteration scripts (run on native host)

Use these when debugging URDF import or scene build failures. They wrap the production scripts with **timestamped logs** and clearer failure output.

| Script | Purpose | Typical duration |
|--------|---------|------------------|
| `scripts/host/check_prereqs.sh` | Verify Isaac Sim path + submodule URDF before any build | ~1 s |
| `scripts/host/iter_urdf_import.sh` | **Fast loop** — URDF import only (`isaac_sim/probe_urdf_import.py`) | ~10–90 s |
| `scripts/host/iter_build_isaac_scene.sh` | **Full loop** — complete workspace scene USD | ~10–90 s |

#### `scripts/host/check_prereqs.sh`

```bash
cd ~/workspaces/isaac_ros-dev/src/spark_isaac_mycobot_demo
./scripts/host/check_prereqs.sh
```

Prints resolved `ISAACSIM_PATH`, `ISAACSIM_PYTHON_EXE`, and suggests next steps. Run this first when returning to the project or after changing Isaac Sim install location.

#### `scripts/host/iter_urdf_import.sh`

Minimal URDF → USD probe. Faster than a full scene build when iterating on mesh paths, material fixes, or importer API changes.

```bash
./scripts/host/iter_urdf_import.sh              # re-prepare meshes + URDF, then import
./scripts/host/iter_urdf_import.sh --keep-prepared   # skip mesh copy if already prepared
```

**Success criteria:** exits 0 and produces robot USD under:

`assets/robots/mycobot_280_m5_limo_cobot/mycobot_280_m5_limo_cobot/mycobot_280_m5_limo_cobot.usda`

(Isaac Sim 6.x writes a `.usda` tree, not a single flat `.usd` file.)

**On failure:** inspect `assets/logs/isaac_host/latest.log`.

#### `scripts/host/iter_build_isaac_scene.sh`

Full scene build with logging.

```bash
./scripts/host/iter_build_isaac_scene.sh
./scripts/host/iter_build_isaac_scene.sh --probe-first          # URDF probe, then full build
./scripts/host/iter_build_isaac_scene.sh --with-ros2-bridge     # embed ROS 2 OmniGraph in saved USD
```

**Success criteria:** exits 0 and writes:

`assets/scenes/mycobot_280_m5_limo_cobot.usd`

## Recommended iteration workflow

When URDF import or scene build fails after a code change:

```bash
# Host terminal
cd ~/workspaces/isaac_ros-dev/src/spark_isaac_mycobot_demo
./scripts/host/check_prereqs.sh
./scripts/host/iter_urdf_import.sh          # fix URDF/mesh issues first
./scripts/host/iter_build_isaac_scene.sh    # then full scene
tail -f assets/logs/isaac_host/latest.log  # optional live log
```

When URDF import is already known good:

```bash
./scripts/build_isaac_scene.sh              # production path, no extra log wrapper
# or
./scripts/host/iter_build_isaac_scene.sh --keep-prepared   # if iter script supports it — use iter without re-probe
```

After scene build succeeds, continue live verification in Cursor (see README Step 3+).

## URDF preparation notes (Isaac Sim 6.x)

The scene builder (`isaac_sim/build_mycobot_limo_cobot_scene.py`) prepares assets automatically:

1. Copies upstream `mycobot_280_m5` meshes into `assets/robots/mycobot_280_m5_limo_cobot/`
2. Rewrites `package://` URIs to relative mesh paths (`isaac_sim/urdf_utils.py`)
3. Replaces `G_base.dae` with a **box primitive** — upstream `G_base.dae` triggers an Isaac Sim 6.x COLLADA material bug (`getPrimNames` / `None` material names). Arm link meshes import normally.
4. Sanitizes GUID-style COLLADA material IDs in remaining `.dae` files
5. Imports via Isaac Sim 6 `URDFImporter` API (`isaac_sim/urdf_import.py`)

Unit tests for URDF prep (no Isaac Sim required):

```bash
python3 -m pytest isaac_sim/test/test_urdf_utils.py isaac_sim/test/test_urdf_import.py -v
```

## Related Python entry points

| File | Role |
|------|------|
| `isaac_sim/probe_urdf_import.py` | Minimal URDF import probe (used by `iter_urdf_import.sh`) |
| `isaac_sim/build_mycobot_limo_cobot_scene.py` | Full workspace scene (table, block, camera, robot reference) |
| `isaac_sim/run_mycobot_live_sim.py` | Live sim + ROS 2 bridge (used by `run_live_sim.sh`) |
| `isaac_sim/urdf_import.py` | Isaac Sim 6 URDF importer wrapper |
| `isaac_sim/urdf_utils.py` | URDF/mesh preparation helpers |
