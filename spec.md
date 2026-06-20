# MyCobot Isaac Sim Reinforcement Learning Pipeline Spec

## Project Overview

An automated pipeline simulating the Elephant Robotics MyCobot arm inside NVIDIA Isaac Sim via ROS 2 Jazzy, migrating to an Isaac Lab Reinforcement Learning environment for block pick-and-place tasks, and porting the policy weights to a physical MyCobot device. The system utilizes visual camera data for block detection and tracking, culminating in a standalone edge deployment using a Raspberry Pi paired with an AI Hat accelerator.

## Core Tech Stack

* **Environment:** Ubuntu 24.04 (DGX OS 7.2.3) via NVIDIA Spark Platform
* **ROS Version:** ROS 2 Jazzy (containerized via Isaac ROS CLI)
* **Simulation Suite:** NVIDIA Isaac Sim / NVIDIA Isaac Lab
* **Hardware Target:** Elephant Robotics MyCobot 280
* **Edge Deployment:** Raspberry Pi (RPi) + AI Hat Board
* **Perception Hardware:** RPi USB Camera (Optics/Lens specification to be evaluated and selected by the AI based on workspace focal length and FOV requirements).

---

## Incremental Execution Backlog

### Phase 1: URDF, ROS 2 Control Bridge & Perception inside Isaac Sim

* [ ] Parse and ingest `https://github.com/elephantrobotics/mycobot_ros2` asset meshes.
* [ ] Build Isaac Sim standalone environment script mapping ROS 2 `JointState` positions to the MyCobot articulation tree.
* [ ] Mount a simulated camera sensor within the Isaac Sim environment oriented toward the manipulator's workspace to provide simulated RGB/Depth data.
* [ ] **Verification Requirement:** Establish automated integration tests verifying that joint commands successfully update articulation transforms, and assert that the simulated camera node actively publishes frame data over ROS 2 topics with zero-copy NITROS delivery.

### Phase 2: Reinforcement Learning (Isaac Lab Ecosystem)

* [ ] Define the MDP (Markov Decision Process) environment class.
* [ ] Integrate visual tracking capabilities so the RL network utilizes camera sensor data for locating the block and estimating grasp poses.
* [ ] Code the reward function targeting visual block tracking, end-effector alignment, grasp state, and vertical lifting.
* [ ] **Verification Requirement:** Add deterministic unit tests for safety boundaries ensuring that penalization metrics trigger correctly before training execution, and verify the simulated vision pipeline feeds accurate bounding/centroid data to the RL observation space.

### Phase 3: Sim-to-Real Hardware Prep & Model Export

* [ ] Export trained neural net policies into edge-optimized ONNX runtime weights suitable for AI Hat hardware acceleration.
* [ ] Develop native ROS 2 physical deployment drivers mapping inference outputs to `pymycobot` serial commands.
* [ ] **Verification Requirement:** Create a local test harness that passes mock visual tensors through the exported ONNX model, evaluates the driver, and asserts expected serial byte outputs for joint states.

### Phase 4: Standalone Edge Execution & Real-World Hardware Verification

* [ ] Configure the Raspberry Pi and AI Hat board as a standalone, isolated edge deployment unit connected directly to the physical MyCobot arm.
* [ ] Evaluate task constraints (working distance, block size, ambient lighting) and output a mathematical recommendation for the most appropriate RPi USB Camera optics/lens (e.g., standard vs. wide-angle, focal length).
* [ ] Deploy the ONNX model to the AI Hat to execute the RL policy natively on the RPi, deriving motion commands solely from live RPi USB Camera video data.
* [ ] Integrate bare-metal safety overrides (joint velocity limits, collision workspace bounding boxes) into the RPi deployment node to intercept malicious or unstable model inferences.
* [ ] **Verification Requirement:** Build a Hardware-in-the-Loop (HIL) integration test suite for the RPi that runs automatically upon deployment. It must verify that the AI Hat inference latency remains below target thresholds, the USB camera frames process without dropping, and the hardware serial nodes reject out-of-bound joint angles before transmitting physical motor commands.