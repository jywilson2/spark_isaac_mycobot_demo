# spark_isaac_mycobot_demo

Automated pipeline for simulating the Elephant Robotics MyCobot 280 inside NVIDIA Isaac Sim via ROS 2 Jazzy, training a pick-and-place policy in Isaac Lab, and deploying to a Raspberry Pi with an AI Hat accelerator.

## Repository Layout

| Path | Purpose |
|------|---------|
| `spark_verify_pkg/` | ROS 2 verification package with mock ecosystem nodes and automated tests |
| `spec.md` | Full project specification and incremental backlog |
| `initial_project_generation.md` | Phase 1 generation instructions |
| `initial_project_generation_phase2.md` | Phase 2 generation instructions |

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

## Build & Test

From the Isaac ROS workspace root:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select spark_verify_pkg
source install/setup.bash
colcon test --packages-select spark_verify_pkg
colcon test-result --all
```

All tests must report zero failures before proceeding to the next phase.

## Phases In Development

The following phases from `spec.md` are not yet implemented in this repository:

### Phase 3: Sim-to-Real Hardware Prep & Model Export

- Export trained policies to edge-optimized ONNX weights for AI Hat acceleration.
- Develop native ROS 2 deployment drivers mapping inference outputs to `pymycobot` serial commands.
- Verification: local test harness passing mock visual tensors through ONNX and asserting expected serial byte outputs.

### Phase 4: Standalone Edge Execution & Real-World Hardware Verification

- Configure Raspberry Pi + AI Hat as a standalone edge deployment unit.
- Evaluate USB camera optics/lens requirements based on workspace geometry.
- Deploy ONNX model on AI Hat with live camera inference and bare-metal safety overrides.
- Verification: Hardware-in-the-Loop (HIL) integration test suite for inference latency, frame throughput, and out-of-bound joint rejection.

## Tech Stack

- **OS:** Ubuntu 24.04 (DGX OS 7.2.3) via NVIDIA Spark Platform
- **ROS:** ROS 2 Jazzy (Isaac ROS CLI container)
- **Simulation:** NVIDIA Isaac Sim / Isaac Lab (planned)
- **Hardware target:** Elephant Robotics MyCobot 280
- **Edge target:** Raspberry Pi + AI Hat Board

## License

Apache-2.0
