# MyCobot Isaac Sim Reinforcement Learning Pipeline Spec

## Project Overview

An automated pipeline simulating the Elephant Robotics MyCobot arm inside NVIDIA Isaac Sim via ROS 2 Jazzy, training a block pick-and-place policy in **NVIDIA Isaac Lab** (required), and porting the policy weights to a physical MyCobot device. The system utilizes visual camera data for block detection and tracking, culminating in a standalone edge deployment using a Raspberry Pi paired with an AI Hat accelerator.

## Core Tech Stack

* **Environment:** Ubuntu 24.04 (DGX OS 7.2.3) via NVIDIA Spark Platform
* **ROS Version:** ROS 2 Jazzy (containerized via Isaac ROS CLI) — live bridge and verification
* **Simulation Suite:** NVIDIA Isaac Sim 6.x + **NVIDIA Isaac Lab** (required for Phase 2 PPO)
* **RL Framework:** RSL-RL (PPO) via Isaac Lab on the Isaac Sim host
* **Hardware Target:** Elephant Robotics MyCobot 280 (Limo Cobot / `mycobot_280_m5` URDF)
* **Edge Deployment:** Raspberry Pi (RPi) + AI Hat Board
* **Perception Hardware:** RPi USB Camera (Optics/Lens specification to be evaluated and selected by the AI based on workspace focal length and FOV requirements).

### Isaac Lab requirement

Phase 2 policy training **must** run through Isaac Lab on the Isaac Sim host:

| Step | Command |
|------|---------|
| Install | `./scripts/host/install_isaac_lab.sh` |
| Verify | `./scripts/host/verify_isaac_lab.sh` |
| Train (PPO) | `./scripts/host/run_isaac_lab_training.sh train --headless` |

Isaac Lab is pinned to the Isaac Sim 6.x pre-built binary workflow (`isaac_lab/versions.env`). The ROS container stack remains for live topic verification and sim-to-ROS bridge testing, but **PPO training is not supported without Isaac Lab**.

---

## Incremental Execution Backlog

### Phase 1: URDF, ROS 2 Control Bridge & Perception inside Isaac Sim

* [x] Parse and ingest `https://github.com/elephantrobotics/mycobot_ros2` asset meshes (`third_party/mycobot_ros2` submodule).
* [x] Build Isaac Sim standalone environment script mapping ROS 2 `JointState` positions to the MyCobot articulation tree (`isaac_sim/build_mycobot_limo_cobot_scene.py`, `run_mycobot_live_sim.py`).
* [x] Mount a simulated camera sensor within the Isaac Sim environment oriented toward the manipulator's workspace to provide simulated RGB data (`/World/WorkspaceCamera` → `/mycobot/camera/rgb`).
* [x] **Verification Requirement:** Automated integration tests for mock and live ROS stacks (`colcon test --packages-select spark_verify_pkg`); live tests auto-skip without Isaac Sim.

### Phase 2: Reinforcement Learning (Isaac Lab Ecosystem)

* [x] Define the MDP (Markov Decision Process) environment class (`MyCobotPickPlaceMDP`, `isaac_lab/mdp_core.py`, `IsaacLabMyCobotPickPlaceEnv` for ROS bridge).
* [x] Integrate visual tracking capabilities so the RL network utilizes camera sensor data for locating the block and estimating grasp poses (`block_vision_tracker`, `rl_observation_bridge`; Isaac Lab env uses block state + joint observations matching the 14-dim contract).
* [x] Code the reward function targeting visual block tracking, end-effector alignment, grasp state, and vertical lifting (`reward_function.py`, shared with `isaac_lab/mdp_core.py`).
* [x] **Isaac Lab install & verify scripts** — `scripts/host/install_isaac_lab.sh`, `scripts/host/verify_isaac_lab.sh`, `isaac_lab/detect_isaac_lab.py`.
* [x] **Isaac Lab DirectRLEnv + PPO trainer** — `isaac_lab/mycobot_pick_place_env.py`, `isaac_lab/train_ppo.py` (RSL-RL PPO on host).
* [x] **Verification Requirement:** Safety boundary unit tests (`test_safety_boundaries.cpp`, `test_safety_boundary_evaluator_py.py`); vision/observation integration tests (`test_phase2_integration.py`, `test_phase2_live_integration.py`); Isaac Lab unit tests (`isaac_lab/test/test_mdp_contract.py`, `test_detect_isaac_lab.py`, `test_isaac_lab_integration.py`).

### Phase 3: Sim-to-Real Hardware Prep & Model Export

* [x] Export trained neural net policies into edge-optimized ONNX runtime weights suitable for AI Hat hardware acceleration (mock path + `latest_policy_onnx_ready.json` from training checkpoints).
* [x] Develop native ROS 2 physical deployment drivers mapping inference outputs to `pymycobot` serial commands (`pymycobot_driver`, `edge_deployment_node`).
* [x] **Verification Requirement:** Mock ONNX harness and driver tests (`test_phase3_integration.py`, `test_mock_onnx_policy.py`).

### Phase 4: Standalone Edge Execution & Real-World Hardware Verification

* [x] Configure the Raspberry Pi and AI Hat board as a standalone, isolated edge deployment unit connected directly to the physical MyCobot arm (mock HIL stack).
* [x] Evaluate task constraints (working distance, block size, ambient lighting) and output a mathematical recommendation for the most appropriate RPi USB Camera optics/lens (`camera_lens_advisor.py`).
* [x] Deploy the ONNX model to the AI Hat to execute the RL policy natively on the RPi, deriving motion commands solely from live RPi USB Camera video data (mock HIL).
* [x] Integrate bare-metal safety overrides (joint velocity limits, collision workspace bounding boxes) into the RPi deployment node to intercept malicious or unstable model inferences.
* [x] **Verification Requirement:** HIL integration test suite (`test_phase4_hil_integration.py`).

---

## Key paths

| Artifact | Location |
|----------|----------|
| Isaac Lab env + trainer | `isaac_lab/mycobot_pick_place_env.py`, `isaac_lab/train_ppo.py` |
| Isaac Lab install | `scripts/host/install_isaac_lab.sh` |
| Scene USD | `assets/scenes/mycobot_280_m5_limo_cobot.usd` |
| Training checkpoints | `assets/checkpoints/isaac_lab_ppo/` |
| Live sim runner | `scripts/run_live_sim.sh` |
| Project status | `docs/project_status.md` |
