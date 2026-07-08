# spark_isaac_mycobot_demo

Automated pipeline for simulating the Elephant Robotics MyCobot 280 inside NVIDIA Isaac Sim via ROS 2 Jazzy, training a pick-and-place policy in Isaac Lab, and deploying to a Raspberry Pi with an AI Hat accelerator.

**Returning after a break?** Read [docs/project_status.md](docs/project_status.md) for a concise “where we left off” briefing (scene build status, live verification gate, Isaac Lab training next steps).

## Phase in development (Phase 2 — EE reach PPO)

**Latest change (2026-07-08):** 1 mm precision, direct-path shaping, 4-stage scripted training.

| Change | Why |
|--------|-----|
| **1 mm EE tolerance** (`EE_REACH_TOLERANCE_M`) | Spec requires sub-millimeter final positioning on the real arm. |
| **Direct-path reward** in approach zone | Penalizes lateral corrective motion when EE is near the target. |
| **4-stage recipe** (`training_recipe.py`, `run_staged_training.sh`) | Reproducible coarse → demo → 5 mm → 1 mm pipeline from scratch. |
| **Phase 5 rename** (`phase5_red_block/`) | Consecutive phase numbering after Phase 4. |
| **Training monitor** (`monitor_training.sh --watch`) | Live reach-rate visibility during long host runs. |

```bash
./scripts/host/run_isaac_lab_training.sh staged --headless
./scripts/host/run_isaac_lab_training.sh monitor --watch
```

Status: [docs/project_status.md](docs/project_status.md).

## Repository Layout

| Path | Purpose |
|------|---------|
| `spark_verify_pkg/` | ROS 2 verification package with mock and live ecosystem nodes and automated tests |
| `isaac_sim/` | Isaac Sim scene builder, ROS 2 bridge config, and host-side live sim runner |
| `scripts/` | Asset fetch, scene build, live sim launch, live tests, container env helper, permission repair |
| `scripts/host/` | **Host-only** Isaac Sim iteration scripts (URDF probe, logged scene build) — see [docs/isaac_sim_host_scripts.md](docs/isaac_sim_host_scripts.md) |
| `docs/` | Extended guides: [host scripts](docs/isaac_sim_host_scripts.md), [project status / resume briefing](docs/project_status.md) |
| `.devcontainer/` | Optional Dev Containers config (not used for the DGX Spark `isaac-ros activate` workflow below) |
| `.vscode/` | Editor settings (integrated terminal drops to `admin` via `gosu` if needed) |
| `spec.md` | Full project specification and incremental backlog |
| `REFERENCES.md` | Curated links: MyCobot 280 background, ROS 2 / Isaac / RL / edge-deployment reference material |
| `commands/` | Agent command playbooks (`initial_project_generation*.md`, `test_live.md`) |
| `initial_project_generation.md` | Phase 1 generation instructions |
| `initial_project_generation_phase2.md` | Phase 2 generation instructions |
| `initial_project_generation_phase_remaining.md` | Phase 3–4 generation instructions |

## Development Workflow (Isaac ROS + Cursor on DGX Spark)

This repository is developed inside an **Isaac ROS CLI** Docker container while **Isaac Sim runs on the host**. The colcon workspace is bind-mounted from the host (`~/workspaces/isaac_ros-dev` → `/workspaces/isaac_ros-dev`), so processes must run as your host user — not root — or git and file saves break on the host.

### Verified daily startup

Use this sequence every session (tested on DGX Spark with Cursor workspace restore):

| Order | Where | Action |
|-------|-------|--------|
| 1 | **Host shell** (native terminal, outside Docker) | `isaac-ros activate` |
| 2 | **Cursor** | Open Cursor — it auto-attaches to the running container when your last workspace is restored |
| 3 | **Cursor terminal** | Confirm `whoami` → `admin` and `id` → uid **1000** (matches your host user) |
| 4 | **Cursor terminal** | `source …/scripts/source_container_env.sh` then build/test ROS packages |

**Step 1 — start the container on the host**

```bash
export ISAAC_ROS_WS="$HOME/workspaces/isaac_ros-dev"   # if not already in ~/.bashrc
isaac-ros activate
```

NVIDIA's `run_dev.py` starts (or re-attaches to) the dev container, passes `HOST_USER_UID` / `HOST_USER_GID` from your host user, and drops interactive shells to **`admin`**. You may exit that host shell after the container is running — it stays up for Cursor.

**Step 2 — Cursor attaches automatically**

After `isaac-ros activate`, open Cursor on the Spark machine. When Cursor restores your last workspace (`spark_isaac_mycobot_demo`), it attaches to the already-running Isaac ROS container. You do **not** need **Dev Containers: Reopen in Container** or **Attach to Running Container** for this workflow.

**Step 3 — verify identity in Cursor**

```bash
whoami          # admin
id              # uid=1000 gid=1000
git status      # should work without Permission denied
```

If `whoami` prints `root`, see [Troubleshooting](#isaac-ros--cursor-troubleshooting) below.

**Step 4 — container environment variables**

Isaac ROS bind-mounts your host `~/.bashrc` into `/home/admin/.bashrc` as **read-only**. Do not append exports there from inside the container. Instead, each Cursor session:

```bash
source /workspaces/isaac_ros-dev/src/spark_isaac_mycobot_demo/scripts/source_container_env.sh
```

That sets `ROS_DOMAIN_ID=42` and `FASTDDS_BUILTIN_TRANSPORTS=UDPv4` (required for Isaac Sim ↔ container ROS traffic).

For **host** terminals (Isaac Sim), persist once in your host `~/.bashrc`:

```bash
export ROS_DOMAIN_ID=42
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export ISAACSIM_PATH="${ISAACSIM_PATH:-$HOME/IsaacSim/_build/linux-aarch64/release}"  # or $HOME/isaacsim for pre-built install
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PYTHON_EXE:-${ISAACSIM_PATH}/python.sh}"
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"
export ISAAC_ROS_WS="$HOME/workspaces/isaac_ros-dev"
```

### One-time setup (host)

- Ubuntu 24.04 with NVIDIA GPU drivers (DGX Spark or compatible)
- [Isaac ROS CLI](https://nvidia-isaac-ros.github.io/getting_started/index.html) installed
- Docker available to your user (`sudo usermod -aG docker $USER`, then log out/in)
- Git LFS (`sudo apt install git-lfs && git lfs install`)
- Isaac Sim 5.x / 6.x on the host (not inside Docker)

Clone into the Isaac ROS workspace:

```bash
mkdir -p ~/workspaces/isaac_ros-dev/src
cd ~/workspaces/isaac_ros-dev/src
git clone git@github.com:jywilson2/spark_isaac_mycobot_demo.git
cd spark_isaac_mycobot_demo
git checkout wip_live_testing   # or your working branch
git submodule update --init --recursive
./scripts/fetch_mycobot_assets.sh
```

Build ROS packages (first time, or after code changes — in Cursor):

```bash
source /workspaces/isaac_ros-dev/src/spark_isaac_mycobot_demo/scripts/source_container_env.sh
source /opt/ros/jazzy/setup.bash
cd /workspaces/isaac_ros-dev
colcon build --packages-select spark_verify_pkg
source install/setup.bash
```

### Terminal layout (live Isaac Sim + training prep)

| Terminal | Machine | Role |
|----------|---------|------|
| **A** | Host | `./scripts/run_live_sim.sh` — MyCobot scene + ROS 2 bridge |
| **B** | Cursor (container) | `ros2 topic hz /clock`, live launch files, integration tests |
| **C** | Host | `./scripts/host/run_isaac_lab_training.sh train` — **Isaac Lab PPO (required, GUI default)** |

Isaac Sim always runs on the **host**. ROS nodes and tests run in the **container** (Cursor).

### Repair root-owned files

If git on the host reports *Permission denied* on `.git/` (usually after editing as root inside the container):

```bash
cd ~/workspaces/isaac_ros-dev/src/spark_isaac_mycobot_demo
./scripts/fix_repo_permissions.sh
```

### Isaac ROS + Cursor troubleshooting

**`whoami` is `root` in Cursor**

The bind-mounted workspace was likely modified as root. Run `./scripts/fix_repo_permissions.sh` on the host, then restart from host: `isaac-ros activate` → reopen Cursor.

**Cursor shows “container is not linked to any local workspace”**

This appears if you manually choose **Attach to Running Container** instead of letting Cursor restore the workspace after `isaac-ros activate`. For the Spark workflow, always start the container with `isaac-ros activate` on the host first, then open Cursor normally — do not attach manually.

**`~/.bashrc: Read-only file system` in the container**

Expected. Isaac ROS mounts the host `.bashrc` read-only. Use `scripts/source_container_env.sh` in container sessions and edit **host** `~/.bashrc` for host-side exports.

**`.devcontainer/` in this repo**

The `.devcontainer/` files are an optional alternative for machines that launch containers through **Reopen in Container**. They are **not** part of the DGX Spark `isaac-ros activate` → Cursor auto-attach workflow above.

## Completed Phases

### Phase 1: URDF, ROS 2 Control Bridge & Perception (Mock Verification)

Phase 1 establishes a mock Isaac Sim ecosystem inside `spark_verify_pkg` that validates the ROS 2 bridge contract before live simulation is wired up.

**Design**

- `mock_articulation_bridge` (C++): subscribes to `/mycobot/joint_commands`, publishes `/mycobot/joint_states` and link TF transforms via a simplified serial-chain kinematic model.
- `mock_camera_publisher` (Python): publishes RGB frames and `NitrosFrameHandle` zero-copy descriptors on NITROS-style topics.
- `joint_command_dispatcher` (Python): dispatches a deterministic 6-DOF joint command sequence for integration tests.
- `phase1_mock_ecosystem.launch.py`: spins up the full Phase 1 mock stack.

**Verification**

- `test_articulation_model.cpp` — C++ gtests for joint kinematics.
- `test_phase1_integration.py` — `launch_testing` integration tests asserting joint commands update articulation TF and the camera publishes RGB + NITROS frame handles.

**Run Phase 1 manually**

```bash
source /workspaces/isaac_ros-dev/install/setup.bash
ros2 launch spark_verify_pkg phase1_mock_ecosystem.launch.py
```

### Phase 2: Reinforcement Learning (Isaac Lab Ecosystem — Mock Verification)

Phase 2 adds the MDP, vision tracking, reward shaping, and safety-boundary logic required before Isaac Lab training begins.

**Design**

- `MyCobotPickPlaceMDP` (Python): defines the observation vector (block centroid, bounding box, end-effector height, grasp flag, joint positions) and reward interface for Isaac Lab integration.
- `reward_function.py` (Python): shaped rewards for visual block tracking, end-effector alignment, grasp state, and vertical lifting, minus safety penalties.
- `safety_boundary_evaluator` (C++): deterministic penalization for joint limits, velocity limits, and workspace bounding boxes.
- `block_vision_tracker` (Python): mock block detector that extracts normalized bounding boxes and centroids from the synthetic camera stream.
- `rl_observation_bridge` (Python): fuses block detections and joint states into `/mycobot/rl/observation` for downstream RL consumers.
- `phase2_mock_ecosystem.launch.py`: includes the Phase 1 stack plus vision tracking and observation bridging.

**Verification**

- `test_safety_boundaries.cpp` — C++ gtests confirming joint-limit, velocity, and workspace violations produce non-zero penalties before training.
- `test_phase2_integration.py` — `launch_testing` tests verifying the vision pipeline publishes accurate block centroids and the RL observation vector contains matching centroid and joint-state data; also validates reward penalization under safety violations.

**Run Phase 2 manually**

```bash
source /workspaces/isaac_ros-dev/install/setup.bash
ros2 launch spark_verify_pkg phase2_mock_ecosystem.launch.py
```

### Phase 3: Sim-to-Real Hardware Prep & Model Export (Mock Verification)

Phase 3 validates the ONNX inference → pymycobot serial command pipeline before physical deployment.

**Design**

- `mock_onnx_policy.py` (Python): deterministic mock ONNX inference mapping RL observations to joint targets.
- `onnx_inference_node` (Python): runs mock inference on `/mycobot/rl/observation`, publishes `/mycobot/policy/inference`.
- `pymycobot_serial_encoder` (C++): encodes joint angles into pymycobot-compatible serial packets.
- `pymycobot_driver` (C++): applies safety gating and publishes `/mycobot/hardware/serial_command`.
- `phase3_mock_ecosystem.launch.py`: includes the Phase 2 stack plus inference and serial driver nodes.

**Verification**

- `test_pymycobot_serial_encoder.cpp` — C++ gtests for deterministic serial encode/decode round-trips, negative/extreme angle preservation, checksum verification, and rejection of corrupted or truncated packets.
- `test_mock_onnx_policy.py` — pytest unit suite covering inference determinism, joint-bias math, centroid steering deltas, zero-padding of truncated observations, and short-observation rejection.
- `test_phase3_integration.py` — `launch_testing` tests passing mock observations through inference to serial output, and confirming out-of-bound joint angles are rejected before transmission.

**Run Phase 3 manually**

```bash
source /workspaces/isaac_ros-dev/install/setup.bash
ros2 launch spark_verify_pkg phase3_mock_ecosystem.launch.py
```

### Phase 4: Standalone Edge Execution & HIL Verification (Mock Verification)

Phase 4 validates the Raspberry Pi + AI Hat edge deployment path with camera optics recommendations and hardware-in-the-loop checks.

**Design**

- `camera_lens_advisor.py` (Python): computes the recommended USB camera focal length from workspace geometry. The lens is selected so the horizontal field of view covers the full manipulator workspace span (2 × 280 mm MyCobot reach, plus margin) at the working distance, while validating that the block still subtends enough image pixels for stable vision tracking; unsatisfiable constraint sets are rejected with a `ValueError`.
- `mock_usb_camera_hil` (Python): simulates USB camera frame publishing and frame-drop statistics.
- `edge_deployment_node` (Python): mock RPi + AI Hat node with inference latency monitoring, frame pipeline health, and bare-metal safety overrides rejecting out-of-bound joint angles.
- `phase4_hil_ecosystem.launch.py`: includes the Phase 3 stack plus USB camera HIL and edge deployment nodes.

**Verification**

- `test_camera_lens_advisor.py` — pytest unit suite verifying the focal-length math against the workspace-coverage model, lens category bucketing across workspace spans, block pixel-extent computation, and `ValueError` rejection of non-physical or unresolvable constraint sets.
- `test_phase4_hil_integration.py` — `launch_testing` HIL tests verifying AI Hat inference latency stays below threshold, USB camera frames process without drops, camera lens recommendation is computed, and out-of-bound joint inferences are rejected before serial transmission.

**Run Phase 4 manually**

```bash
source /workspaces/isaac_ros-dev/install/setup.bash
ros2 launch spark_verify_pkg phase4_hil_ecosystem.launch.py
```

## Build & Test

From the Isaac ROS workspace root:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select spark_verify_pkg
source install/setup.bash
colcon test --packages-select spark_verify_pkg
colcon test-result --all
```

All mock tests must report zero failures before live Isaac Sim verification. Live integration tests auto-skip when Isaac Sim is not playing (or set `SPARK_SKIP_LIVE_SIM_TESTS=1`).

### Live Phase 1 & 2 (Isaac Sim)

Live stacks use Isaac Sim on the host for articulation and camera telemetry. Docker-side nodes consume live topics — **not** mock articulation or mock camera publishers.

**Design (Live Phase 1)**

- `isaac_sim/build_mycobot_limo_cobot_scene.py` + `isaac_sim/ros2_bridge_config.py`: host-side scene with embedded ROS 2 OmniGraph bridge (`/clock`, `/mycobot/joint_commands`, `/mycobot/joint_states`, `/mycobot/camera/rgb`).
- `isaac_sim/run_mycobot_live_sim.py`: load scene and run live sim with bridge on the host.
- `live_nitros_camera_bridge` (Python): publishes `/mycobot/camera/nitros/rgb` NITROS frame handles from live RGB frames.
- `joint_command_dispatcher` with `joint_name_mode:=urdf`: sends URDF joint commands to Isaac Sim.
- `phase1_live_ecosystem.launch.py`: live Phase 1 Docker-side stack (no mock bridge/camera).

**Design (Live Phase 2)**

- `block_vision_tracker`: detects the red workspace block from **live** camera RGB (not synthetic mock painting).
- `rl_observation_bridge`: fuses live detections and joint states into `/mycobot/rl/observation`.
- `IsaacLabMyCobotPickPlaceEnv` (`isaac_lab_mdp_env.py`): Isaac Lab MDP facade wired to live ROS topics.
- `live_mdp_reward_monitor`: publishes live reward and safety penalty metrics for smoke tests.
- `phase2_live_ecosystem.launch.py`: Phase 1 live stack plus vision, observation bridge, and MDP monitor.

**Verification**

- `test_phase1_live_integration.py` — live joint command/state, TF, RGB + NITROS (skips without Isaac Sim).
- `test_phase2_live_integration.py` — live block centroid, RL observation vector, safety/reward penalization (skips without Isaac Sim).
- `test_live_sim_gate.py`, `test_safety_boundary_evaluator_py.py` — always-run unit tests for live gate and safety logic.

**Run Live Phase 1 manually (Docker, Isaac Sim playing on host)**

```bash
export ROS_DOMAIN_ID=42
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
source /workspaces/isaac_ros-dev/install/setup.bash
ros2 launch spark_verify_pkg phase1_live_ecosystem.launch.py
```

**Run Live Phase 2 manually**

```bash
export ROS_DOMAIN_ID=42
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
source /workspaces/isaac_ros-dev/install/setup.bash
ros2 launch spark_verify_pkg phase2_live_ecosystem.launch.py
```

If the default `c++` compiler on the build host is Clang without a matching GCC
runtime (some container images), force the GCC toolchain:

```bash
CC=gcc CXX=g++ colcon build --packages-select spark_verify_pkg
```

## Running Isaac Sim (Live Phase 1 & 2 / Training Prep)

Live Phase 1 and Phase 2 require **Isaac Sim running on the host** with the ROS 2 bridge active. Mock `colcon test` and mock launch files do **not** need Isaac Sim.

This workflow follows the same host ↔ Docker networking pattern as [`spark_isaac_sim_robot_demo`](../spark_isaac_sim_robot_demo/README.md) in this workspace. Complete [Development Workflow](#development-workflow-isaac-ros--cursor-on-dgx-spark) first (`isaac-ros activate` → Cursor as `admin`).

### Prerequisites

- Isaac ROS container running (`isaac-ros activate` on host) and Cursor attached as **`admin`**
- **Isaac Sim 5.x / 6.x** on the **host** (not inside Docker)
- Matching **`ROS_DOMAIN_ID`** on host and in container (this guide uses `42`)
- **`FASTDDS_BUILTIN_TRANSPORTS=UDPv4`** on host and in container (required for Isaac Sim ↔ container communication)

**Host** — add once to `~/.bashrc`:

```bash
export ROS_DOMAIN_ID=42
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export ISAACSIM_PATH="${ISAACSIM_PATH:-$HOME/IsaacSim/_build/linux-aarch64/release}"  # or $HOME/isaacsim for pre-built install
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PYTHON_EXE:-${ISAACSIM_PATH}/python.sh}"
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"
```

**Container (Cursor)** — each session:

```bash
source /workspaces/isaac_ros-dev/src/spark_isaac_mycobot_demo/scripts/source_container_env.sh
```

Set `FASTDDS_BUILTIN_TRANSPORTS=UDPv4` on the host **before** launching Isaac Sim.

### Step 1 — Fetch assets (container, one-time or after clone)

The **Limo Cobot** mobile manipulator ships with a **myCobot 280 M5** arm. This repository vendors upstream meshes via a git submodule.

In a **Cursor terminal**:

```bash
cd /workspaces/isaac_ros-dev/src/spark_isaac_mycobot_demo
./scripts/fetch_mycobot_assets.sh
```

This initializes `third_party/mycobot_ros2` (branch `humble`) and verifies:

`third_party/mycobot_ros2/mycobot_description/urdf/mycobot_280_m5/mycobot_280_m5.urdf`

### Step 2 — Build the MyCobot scene (host, first run or after URDF changes)

Isaac Sim runs on the **host only**. Use a **native host terminal** (not the Cursor container).

**Recommended for debugging** (timestamped logs under `assets/logs/isaac_host/`):

```bash
cd ~/workspaces/isaac_ros-dev/src/spark_isaac_mycobot_demo
./scripts/host/check_prereqs.sh
./scripts/host/iter_urdf_import.sh       # fast URDF-only probe (~10–90 s)
./scripts/host/iter_build_isaac_scene.sh # full scene build
```

See [docs/isaac_sim_host_scripts.md](docs/isaac_sim_host_scripts.md) for what each script does and why.

**Production one-liner** (after URDF import is known good):

```bash
export ROS_DOMAIN_ID=42
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export ISAACSIM_PATH="${ISAACSIM_PATH:-$HOME/isaacsim}"   # or source-build path; see ./scripts/isaac_sim_env.sh

cd ~/workspaces/isaac_ros-dev/src/spark_isaac_mycobot_demo
./scripts/build_isaac_scene.sh
```

Output:

| Artifact | Path |
|----------|------|
| Scene USD | `assets/scenes/mycobot_280_m5_limo_cobot.usd` |
| Isaac-ready URDF + meshes | `assets/robots/mycobot_280_m5_limo_cobot/` |
| Robot USD (Isaac Sim 6) | `assets/robots/mycobot_280_m5_limo_cobot/mycobot_280_m5_limo_cobot/*.usda` |

**Note:** The robot base link uses a **box placeholder** instead of upstream `G_base.dae` because Isaac Sim 6.x fails to import that COLLADA mesh (material ID bug). Arm links use the original meshes.

The scene builder can embed the ROS 2 OmniGraph bridge (`--with-ros2-bridge`) for live topics; `run_live_sim.sh` also configures the bridge at runtime.

### Step 3 — Start live Isaac Sim with ROS 2 bridge (host)

In a **native host terminal** — keep this running for all live work:

```bash
export ROS_DOMAIN_ID=42
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export ISAACSIM_PATH="${ISAACSIM_PATH:-$HOME/IsaacSim/_build/linux-aarch64/release}"  # or $HOME/isaacsim for pre-built install
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PYTHON_EXE:-${ISAACSIM_PATH}/python.sh}"
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"

cd ~/workspaces/isaac_ros-dev/src/spark_isaac_mycobot_demo
./scripts/run_live_sim.sh
```

This rebuilds/opens the scene if needed, enables the bridge, and presses **Play**. For long unattended runs:

```bash
./scripts/run_live_sim.sh --headless
```

Expected host topics:

| Topic | Message type | Role |
|-------|--------------|------|
| `/clock` | `rosgraph_msgs/msg/Clock` | Simulation time |
| `/mycobot/joint_commands` | `sensor_msgs/msg/JointState` | Commands into Isaac Sim articulation |
| `/mycobot/joint_states` | `sensor_msgs/msg/JointState` | Live articulation feedback |
| `/mycobot/camera/rgb` | `sensor_msgs/msg/Image` | Workspace camera |

**Alternative (manual GUI):** open `${ISAACSIM_PATH}/isaac-sim.sh`, load the scene USD, and press **Play** yourself. The scripted path above is preferred because it configures the ROS 2 bridge automatically.

### Step 4 — Verify bridge (Cursor terminal)

```bash
source /workspaces/isaac_ros-dev/src/spark_isaac_mycobot_demo/scripts/source_container_env.sh
ros2 topic hz /clock
ros2 topic list | grep mycobot
```

Expect `/clock` at a non-zero rate (often ~25 Hz). Press `Ctrl+C` to stop.

If topics appear on the host but not in the container, set `FASTDDS_BUILTIN_TRANSPORTS=UDPv4` on **both** sides and restart `./scripts/run_live_sim.sh`. See [Isaac Sim troubleshooting](#isaac-sim-troubleshooting).

### Step 5 — Launch Docker-side live stacks (Cursor)

```bash
source /workspaces/isaac_ros-dev/src/spark_isaac_mycobot_demo/scripts/source_container_env.sh
source /workspaces/isaac_ros-dev/install/setup.bash
```

**Live Phase 1** (joint + NITROS adapter):

```bash
ros2 launch spark_verify_pkg phase1_live_ecosystem.launch.py
```

**Live Phase 2** (vision + RL observation + MDP monitor — required before training):

```bash
ros2 launch spark_verify_pkg phase2_live_ecosystem.launch.py
```

Do **not** use `phase1_mock_ecosystem.launch.py` or `phase2_mock_ecosystem.launch.py` for live acceptance.

**Phase 2 gate** — confirm the training pipeline topics:

```bash
ros2 topic hz /mycobot/rl/observation
ros2 topic echo /mycobot/rl/live_reward --once
ros2 topic echo /mycobot/vision/block_detection --once
```

### Step 6 — Expected live topics (Phase 1 & 2)

These are the **live** endpoints verified by live integration tests (mock topics are **not** substitutes):

| Topic | Message type | Phase | Role |
|-------|--------------|-------|------|
| `/clock` | `rosgraph_msgs/msg/Clock` | 1, 2 | Simulation time (`use_sim_time:=true`) |
| `/mycobot/joint_commands` | `sensor_msgs/msg/JointState` | 1 | Commands into Isaac Sim articulation |
| `/mycobot/joint_states` | `sensor_msgs/msg/JointState` | 1, 2 | Live articulation feedback |
| `/mycobot/camera/rgb` | `sensor_msgs/msg/Image` | 1, 2 | Workspace camera from Isaac Sim |
| `/mycobot/camera/nitros/rgb` | project `NitrosFrameHandle` | 1 | NITROS zero-copy descriptors (Docker adapter) |
| `/mycobot/vision/block_detection` | project `BlockDetection` | 2 | Live block centroid / bbox |
| `/mycobot/rl/observation` | project `RlObservation` | 2 | RL observation vector from live sim |
| `/mycobot/rl/live_reward` | `std_msgs/msg/Float32` | 2 | Live MDP reward smoke metric |
| `/mycobot/rl/live_safety_penalty` | `std_msgs/msg/Float32` | 2 | Live safety penalty smoke metric |
| `/tf` | `tf2_msgs/msg/TFMessage` | 1 | Articulation transforms |

Verify publishers before live tests:

```bash
source /workspaces/isaac_ros-dev/src/spark_isaac_mycobot_demo/scripts/source_container_env.sh
ros2 topic info /clock
ros2 topic list | grep mycobot
```

You need **Publisher count: 1** (or more) on `/clock` and live sim topics while **Play** is active.

### Step 7 — Run live integration tests (pre-training gate)

With Isaac Sim playing (`./scripts/run_live_sim.sh` on host):

```bash
source /workspaces/isaac_ros-dev/src/spark_isaac_mycobot_demo/scripts/source_container_env.sh
cd /workspaces/isaac_ros-dev/src/spark_isaac_mycobot_demo
./scripts/run_live_tests.sh
```

Or from the workspace manually:

```bash
source /workspaces/isaac_ros-dev/src/spark_isaac_mycobot_demo/scripts/source_container_env.sh
source /opt/ros/jazzy/setup.bash
cd /workspaces/isaac_ros-dev
colcon build --packages-select spark_verify_pkg
source install/setup.bash
colcon test --packages-select spark_verify_pkg --event-handlers console_direct+
colcon test-result --all
```

Live tests (`test_phase1_live_integration.py`, `test_phase2_live_integration.py`) **auto-skip** when `/clock` is unavailable. Force skip in CI/mock runs with `SPARK_SKIP_LIVE_SIM_TESTS=1`.

**Do not start Isaac Lab training until Step 7 passes.**

### Step 8 — Agent-driven live verification playbook

Use the live testing playbook:

```text
Please execute the instructions defined in @commands/test_live.md based on @spec.md and @.cursorrules.
```

See [`commands/test_live.md`](commands/test_live.md) for the full Live Phase 1/2 implementation and acceptance sequence. **Do not** use:

```bash
# Mock only — not for live Isaac Sim verification
ros2 launch spark_verify_pkg phase1_mock_ecosystem.launch.py
ros2 launch spark_verify_pkg phase2_mock_ecosystem.launch.py
```

### Step 9 — Isaac Lab PPO training (required)

Phase 2 policy training **requires Isaac Lab** on the Isaac Sim host. Install once, then train:

```bash
# Host terminal — GUI default (2 arms, Isaac Sim window open)
./scripts/host/install_isaac_lab.sh
./scripts/host/verify_isaac_lab.sh
./scripts/host/run_isaac_lab_training.sh train

# Headless integration / throughput (8 arms on DGX Spark)
./scripts/host/run_isaac_lab_training.sh train --headless

# Headless PPO integration smoke (CI / pre-merge gate)
./scripts/host/verify_isaac_lab.sh --smoke-train
```

Add `--headless` to disable the Isaac Sim GUI. With visualization enabled, the default is **2 parallel arms** to avoid out-of-memory kills; headless training defaults to **8 arms**.

Prerequisites:

1. Isaac Sim at `~/isaacsim` (see Step 2 / host scripts)
2. Robot USD built: `./scripts/host/iter_build_isaac_scene.sh`
3. Isaac Lab cloned to `~/IsaacLab` (`develop` branch — see `isaac_lab/versions.env`)

Checkpoints: `assets/checkpoints/isaac_lab_ppo/latest_policy` + `training_summary.json`.

The container `./scripts/run_live_training.sh` redirects to the host Isaac Lab trainer (training does not run inside Docker).

Optional ROS live verification (Steps 5–7) remains recommended before deploying trained policies to the live sim bridge.

### Step 10 — Demonstrate the trained policy (locate + push random cube)

After training completes, run the saved policy in the Isaac Sim GUI. Each episode spawns the red cube at a **random position within arm reach**; the arm uses the **EE-mounted camera** to locate the block, makes contact, and pushes it **5 mm in any direction**.

```bash
# Host terminal — GUI demo (default checkpoint)
./scripts/host/run_isaac_lab_training.sh play

# More episodes or a specific checkpoint
./scripts/host/run_isaac_lab_training.sh play --episodes 20 \
  --checkpoint assets/checkpoints/isaac_lab_ppo/latest_policy
```

Training prints the same play instructions when it finishes. If `task_requirement_met` is `NO`, the policy may still partially locate or push the cube — retrain or continue training until contact/push targets are met.

### Ongoing use of trained policies

Once you have a checkpoint you are satisfied with:

1. **Demo / regression check** — Re-run `./scripts/host/run_isaac_lab_training.sh play` after code or asset changes to confirm the arm still locates and pushes randomly placed cubes.
2. **Resume training** — Run `./scripts/host/run_isaac_lab_training.sh train` again; checkpoints overwrite `assets/checkpoints/isaac_lab_ppo/latest_policy` unless you pass `--checkpoint-dir` to a new folder.
3. **Track quality** — Inspect `assets/checkpoints/isaac_lab_ppo/training_summary.json` for `contact_rate`, `push_success_rate`, and `task_requirement_met`.
4. **ROS / live sim bridge** — For Phase 2 live topic verification, continue using the ROS stack (Steps 5–7) before deploying weights to hardware or the live sim bridge.
5. **Edge export** — Phase 3 ONNX export reads from the same checkpoint directory; re-export after retraining.

To archive a good run, copy the whole `assets/checkpoints/isaac_lab_ppo/` directory and pass `--checkpoint` to `play` pointing at the archived `latest_policy` folder.

### Quick reference checklist

| Step | Where | Action |
|------|-------|--------|
| 0 | Host | `isaac-ros activate`, then open Cursor (auto-attach) |
| 0 | Cursor | `whoami` → `admin`; `source …/scripts/source_container_env.sh` |
| 1 | Cursor | `./scripts/fetch_mycobot_assets.sh` (one-time) |
| 2 | Host | `./scripts/host/iter_build_isaac_scene.sh` or `./scripts/build_isaac_scene.sh` |
| 3 | Host | `./scripts/run_live_sim.sh` — keep running |
| 4 | Cursor | `ros2 topic hz /clock` → non-zero rate |
| 5 | Cursor | `ros2 launch spark_verify_pkg phase2_live_ecosystem.launch.py` |
| 6 | Cursor | `./scripts/run_live_tests.sh` — must pass before training |
| 7 | Host | `./scripts/host/run_isaac_lab_training.sh train` (Step 9; GUI default 2 arms) |
| 8 | Host | `./scripts/host/run_isaac_lab_training.sh play` (Step 10 — demo trained policy) |

### Isaac Sim troubleshooting

**Host sees `/clock` but Docker does not**

Fast DDS may default to shared memory, which does not work across the Isaac Sim host process and the Docker container. Force UDP on both sides:

```bash
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
```

Restart Isaac Sim after setting this on the host.

**Topics listed but no data (`ros2 topic hz` hangs)**

Press **Play** in Isaac Sim. Confirm `ros2 topic info /clock` shows `Publisher count: 1`.

**Simulation time stalled**

If `/clock` is not advancing, nodes using `use_sim_time:=true` will not progress. Resume playback in Isaac Sim.

**Stale nodes after tests**

Stop leftover launch processes before re-testing:

```bash
pkill -f "phase1_mock_ecosystem|phase2_mock_ecosystem|mock_articulation_bridge" || true
export ROS_DOMAIN_ID=42
ros2 node list
```

## Asset ingest & live integration status

| Item | Status |
|------|--------|
| `mycobot_ros2` submodule (`third_party/mycobot_ros2`, `humble`) | Ready — Limo Cobot arm = `mycobot_280_m5.urdf` |
| URDF → USD import (Isaac Sim 6.x) | Ready — `isaac_sim/urdf_import.py`; host probe: `./scripts/host/iter_urdf_import.sh` |
| Isaac Sim scene builder | Ready — output: `assets/scenes/mycobot_280_m5_limo_cobot.usd` |
| Host iteration scripts + logs | Ready — [docs/isaac_sim_host_scripts.md](docs/isaac_sim_host_scripts.md) |
| Host live sim runner | Ready — `./scripts/run_live_sim.sh` |
| Live Phase 1 Docker stack | Ready |
| Live Phase 2 Docker stack | Ready |
| Live integration tests | Ready — auto-skip without Isaac Sim |
| End-to-end live test gate | **Run `./scripts/run_live_tests.sh` before training** |
| Isaac Lab training loop | **Required** — `./scripts/host/install_isaac_lab.sh` then `run_isaac_lab_training.sh train` |

## Live Integration (Phase 1–2 ready)

Mock verification phases validate ROS 2 contracts without Isaac Sim. **Live Phase 1 and 2** are implemented as described above. **Scene build on Isaac Sim 6.x is verified** (2026-07-05).

Remaining work from `spec.md` and [docs/project_status.md](docs/project_status.md):

- **Live verification gate:** Run `./scripts/run_live_tests.sh` with sim playing (if not done recently).
- **Isaac Lab training:** Required on host — `./scripts/host/install_isaac_lab.sh` then `run_isaac_lab_training.sh train`.
- **Live Phase 3:** Export real trained ONNX weights and connect to physical MyCobot via `pymycobot` serial.
- **Live Phase 4:** Deploy to Raspberry Pi + AI Hat with live USB camera and physical MyCobot arm.

## Tech Stack

- **OS:** Ubuntu 24.04 (DGX OS 7.2.3) via NVIDIA Spark Platform
- **ROS:** ROS 2 Jazzy (Isaac ROS CLI container)
- **Simulation:** NVIDIA Isaac Sim / Isaac Lab (required for live Phase 1–2; see [Running Isaac Sim](#running-isaac-sim-live-phase-1--2--training-prep))
- **Hardware target:** Elephant Robotics MyCobot 280
- **Edge target:** Raspberry Pi + AI Hat Board

## License

Apache-2.0
