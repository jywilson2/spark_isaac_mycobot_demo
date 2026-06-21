# spark_isaac_mycobot_demo

Automated pipeline for simulating the Elephant Robotics MyCobot 280 inside NVIDIA Isaac Sim via ROS 2 Jazzy, training a pick-and-place policy in Isaac Lab, and deploying to a Raspberry Pi with an AI Hat accelerator.

## Repository Layout

| Path | Purpose |
|------|---------|
| `spark_verify_pkg/` | ROS 2 verification package with mock and live ecosystem nodes and automated tests |
| `isaac_sim/` | Isaac Sim scene builder, ROS 2 bridge config, and host-side live sim runner |
| `scripts/` | Asset fetch, scene build, live sim launch, and live test helpers |
| `spec.md` | Full project specification and incremental backlog |
| `commands/` | Agent command playbooks (`initial_project_generation*.md`, `test_live.md`) |
| `initial_project_generation.md` | Phase 1 generation instructions |
| `initial_project_generation_phase2.md` | Phase 2 generation instructions |
| `initial_project_generation_phase_remaining.md` | Phase 3–4 generation instructions |

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

- `test_pymycobot_serial_encoder.cpp` — C++ gtests for deterministic serial encode/decode round-trips.
- `test_phase3_integration.py` — `launch_testing` tests passing mock observations through inference to serial output, and confirming out-of-bound joint angles are rejected before transmission.

**Run Phase 3 manually**

```bash
source /workspaces/isaac_ros-dev/install/setup.bash
ros2 launch spark_verify_pkg phase3_mock_ecosystem.launch.py
```

### Phase 4: Standalone Edge Execution & HIL Verification (Mock Verification)

Phase 4 validates the Raspberry Pi + AI Hat edge deployment path with camera optics recommendations and hardware-in-the-loop checks.

**Design**

- `camera_lens_advisor.py` (Python): computes recommended USB camera focal length from workspace geometry (working distance, block size, sensor width).
- `mock_usb_camera_hil` (Python): simulates USB camera frame publishing and frame-drop statistics.
- `edge_deployment_node` (Python): mock RPi + AI Hat node with inference latency monitoring, frame pipeline health, and bare-metal safety overrides rejecting out-of-bound joint angles.
- `phase4_hil_ecosystem.launch.py`: includes the Phase 3 stack plus USB camera HIL and edge deployment nodes.

**Verification**

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

## Running Isaac Sim

Live Phase 1 and Phase 2 require **Isaac Sim running on the host** with the ROS 2 bridge active. Mock `colcon test` and mock launch files do **not** need Isaac Sim.

This workflow follows the same host ↔ Docker networking pattern as [`spark_isaac_sim_robot_demo`](../spark_isaac_sim_robot_demo/README.md) in this workspace.

### Prerequisites

- **Isaac ROS CLI** dev container (Cursor terminal) on DGX Spark or compatible platform
- **Isaac Sim 5.x / 6.x** on the **host** (not inside Docker)
- Matching **`ROS_DOMAIN_ID`** on host and in Docker (this guide uses `42`)
- **`FASTDDS_BUILTIN_TRANSPORTS=UDPv4`** on host and in Docker (required for Isaac Sim ↔ container communication)

Set these in **every** host and Cursor terminal before ROS commands:

```bash
export ROS_DOMAIN_ID=42
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
```

Persist inside the dev container (optional):

```bash
echo 'export ROS_DOMAIN_ID=42' >> ~/.bashrc
echo 'export FASTDDS_BUILTIN_TRANSPORTS=UDPv4' >> ~/.bashrc
```

Also set `FASTDDS_BUILTIN_TRANSPORTS=UDPv4` on the host **before** launching Isaac Sim.

### Step 1 — Launch Isaac Sim on the host

In a **native host terminal** (outside Docker):

```bash
export ROS_DOMAIN_ID=42
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"
${ISAACSIM_PATH}/isaac-sim.sh
```

If Isaac Sim is installed elsewhere, run `./isaac-sim.sh` from your Isaac Sim install directory.

### Step 2 — Fetch assets and build the MyCobot scene

The **Limo Cobot** mobile manipulator ships with a **myCobot 280 M5** arm. This repository vendors upstream meshes via a git submodule and provides an Isaac Sim standalone scene builder.

**Inside the Isaac ROS container (or any git checkout):**

```bash
cd /path/to/spark_isaac_mycobot_demo
./scripts/fetch_mycobot_assets.sh
```

This initializes `third_party/mycobot_ros2` (branch `humble`) and verifies:

`third_party/mycobot_ros2/mycobot_description/urdf/mycobot_280_m5/mycobot_280_m5.urdf`

**On the Isaac Sim host** (outside Docker), build the workspace scene USD:

```bash
export ROS_DOMAIN_ID=42
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export ISAACSIM_PATH=/path/to/isaac-sim   # your install

cd /path/to/spark_isaac_mycobot_demo
./scripts/build_isaac_scene.sh
```

Output:

| Artifact | Path |
|----------|------|
| Scene USD | `assets/scenes/mycobot_280_m5_limo_cobot.usd` |
| Isaac-ready URDF + meshes | `assets/robots/mycobot_280_m5_limo_cobot/` |

The scene builder embeds the ROS 2 OmniGraph bridge (`--with-ros2-bridge`) for live topics.

### Step 3 — Start live Isaac Sim with ROS 2 bridge (host)

In a **native host terminal**:

```bash
export ROS_DOMAIN_ID=42
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export ISAACSIM_PATH=/path/to/isaac-sim

cd /path/to/spark_isaac_mycobot_demo
./scripts/run_live_sim.sh
```

This rebuilds/opens the scene, enables the bridge, and presses **Play**. Expected host topics:

| Topic | Message type | Role |
|-------|--------------|------|
| `/clock` | `rosgraph_msgs/msg/Clock` | Simulation time |
| `/mycobot/joint_commands` | `sensor_msgs/msg/JointState` | Commands into Isaac Sim articulation |
| `/mycobot/joint_states` | `sensor_msgs/msg/JointState` | Live articulation feedback |
| `/mycobot/camera/rgb` | `sensor_msgs/msg/Image` | Workspace camera |

Keep the live sim running for Docker-side verification.

### Step 4 — Launch Docker-side live stacks

**Live Phase 1 (joint + NITROS adapter):**

```bash
export ROS_DOMAIN_ID=42
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
source /workspaces/isaac_ros-dev/install/setup.bash
ros2 launch spark_verify_pkg phase1_live_ecosystem.launch.py
```

**Live Phase 2 (vision + RL observation + MDP monitor):**

```bash
ros2 launch spark_verify_pkg phase2_live_ecosystem.launch.py
```

Do **not** use `phase1_mock_ecosystem.launch.py` or `phase2_mock_ecosystem.launch.py` for live acceptance.

**Bridge sanity check (Cursor terminal):**

```bash
export ROS_DOMAIN_ID=42
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
ros2 topic hz /clock
```

Expect a non-zero rate (often ~25 Hz). Press `Ctrl+C` to stop.

If `ros2 topic list` shows topics but `hz` hangs, set `FASTDDS_BUILTIN_TRANSPORTS=UDPv4` on **both** host and Docker and restart Isaac Sim. See [Troubleshooting](#isaac-sim-troubleshooting).

### Step 5 — Expected live topics (Phase 1 & 2)

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
export ROS_DOMAIN_ID=42
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
ros2 topic info /clock
ros2 topic list | grep mycobot
```

You need **Publisher count: 1** (or more) on `/clock` and live sim topics while **Play** is active.

### Step 6 — Run live integration tests

With Isaac Sim playing (`./scripts/run_live_sim.sh` on host):

```bash
export ROS_DOMAIN_ID=42
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
cd /path/to/spark_isaac_mycobot_demo
./scripts/run_live_tests.sh
```

Or from the workspace manually:

```bash
source /opt/ros/jazzy/setup.bash
cd ${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}
colcon build --packages-select spark_verify_pkg
source install/setup.bash
export ROS_DOMAIN_ID=42
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
colcon test --packages-select spark_verify_pkg --event-handlers console_direct+
colcon test-result --all
```

Live tests (`test_phase1_live_integration.py`, `test_phase2_live_integration.py`) **auto-skip** when `/clock` is unavailable. Force skip in CI/mock runs with `SPARK_SKIP_LIVE_SIM_TESTS=1`.

### Step 7 — Agent-driven live verification playbook

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

### Step 8 — Isaac Lab (Live Phase 2 training)

Live Phase 2 RL work additionally requires **Isaac Lab** with the MDP environment using **live** sim camera observations (not mock `mock_camera_publisher` or synthetic block painting). Start Isaac Sim and load the MyCobot scene first; then launch Isaac Lab training or live MDP smoke tests against the running sim.

### Quick reference checklist

| Step | Where | Action |
|------|-------|--------|
| 1 | Host | Export `ROS_DOMAIN_ID=42` and `FASTDDS_BUILTIN_TRANSPORTS=UDPv4` |
| 2 | Container | `./scripts/fetch_mycobot_assets.sh` |
| 3 | Host | `./scripts/build_isaac_scene.sh` or `./scripts/run_live_sim.sh` |
| 4 | Host | Keep live sim playing (bridge publishing `/clock`) |
| 5 | Cursor | `ros2 topic hz /clock` → expect non-zero rate |
| 6 | Cursor | `ros2 launch spark_verify_pkg phase1_live_ecosystem.launch.py` |
| 7 | Cursor | `ros2 launch spark_verify_pkg phase2_live_ecosystem.launch.py` |
| 8 | Cursor | `./scripts/run_live_tests.sh` or `@commands/test_live.md` |

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
| Isaac Sim scene builder (`isaac_sim/build_mycobot_limo_cobot_scene.py`) | Ready — embeds ROS 2 bridge via `--with-ros2-bridge` |
| Host live sim runner (`isaac_sim/run_mycobot_live_sim.py`) | Ready — `./scripts/run_live_sim.sh` |
| Live Phase 1 Docker stack (`phase1_live_ecosystem.launch.py`) | Ready |
| Live Phase 2 Docker stack (`phase2_live_ecosystem.launch.py`) | Ready |
| Live integration tests (`test_phase*_live_integration.py`) | Ready — auto-skip without Isaac Sim |
| Isaac Lab training loop against live sim | Use `IsaacLabMyCobotPickPlaceEnv` — training orchestration is operator-driven |

## Live Integration (Phase 1–2 ready)

Mock verification phases validate ROS 2 contracts without Isaac Sim. **Live Phase 1 and 2** are implemented as described above. Remaining work from `spec.md`:

- **Live Phase 2 training:** Run Isaac Lab policy training against the live MDP facade (scene + ROS topics must be playing).
- **Live Phase 3:** Export real trained ONNX weights and connect to physical MyCobot via `pymycobot` serial.
- **Live Phase 4:** Deploy to Raspberry Pi + AI Hat with live USB camera and physical MyCobot arm.

## Tech Stack

- **OS:** Ubuntu 24.04 (DGX OS 7.2.3) via NVIDIA Spark Platform
- **ROS:** ROS 2 Jazzy (Isaac ROS CLI container)
- **Simulation:** NVIDIA Isaac Sim / Isaac Lab (required for live Phase 1–2; see [Running Isaac Sim](#running-isaac-sim))
- **Hardware target:** Elephant Robotics MyCobot 280
- **Edge target:** Raspberry Pi + AI Hat Board

## License

Apache-2.0
