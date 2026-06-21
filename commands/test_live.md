Review @spec.md and @.cursorrules. We are moving into **live** verification for Phase 1 and Phase 2 inside Isaac Sim / Isaac Lab. Do **not** use the mock launch files (`phase1_mock_ecosystem.launch.py`, `phase2_mock_ecosystem.launch.py`) for live acceptance.

## Prerequisites (agent must confirm before testing)

1. Mock regression is green (optional sanity check):
   ```bash
   source /opt/ros/jazzy/setup.bash
   cd ${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}
   colcon build --packages-select spark_verify_pkg
   source install/setup.bash
   colcon test --packages-select spark_verify_pkg
   colcon test-result --all
   ```
2. **Isaac Sim is running on the host** with the MyCobot scene loaded and **Play** pressed. See [README.md](../README.md#running-isaac-sim).
3. **ROS 2 bridge is active** — Docker/Cursor terminal receives sim data:
   ```bash
   export ROS_DOMAIN_ID=42
   export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
   ros2 topic hz /clock
   ```
   Expect a non-zero rate before proceeding.
4. For **Live Phase 2**, Isaac Lab is installed and the sim scene matches the MDP workspace geometry.

If Isaac Sim is not running, start it and confirm the bridge **before** live tests. Do not substitute mock nodes.

---

## Live Phase 1 sequence

Implement and verify (per @spec.md):

1. Parse and ingest `https://github.com/elephantrobotics/mycobot_ros2` asset meshes / URDF.
2. Build an Isaac Sim standalone environment script mapping ROS 2 `JointState` commands to the MyCobot articulation tree (Isaac Sim Articulation API — not CPU mock kinematics).
3. Mount a simulated camera oriented toward the manipulator workspace; publish RGB/Depth over ROS 2 with **zero-copy NITROS** delivery (not mock `NitrosFrameHandle`).
4. Add live integration tests (e.g. `test_phase1_live_integration.py`) separate from mock `test_phase1_integration.py`.

**Acceptance criteria**

- Joint commands update articulation transforms in sim (verify TF and `/mycobot/joint_states`).
- Simulated camera actively publishes frame data on ROS 2 topics with NITROS delivery.
- Live tests pass without mock articulation or mock camera nodes running.

---

## Live Phase 2 sequence

Implement and verify (per @spec.md):

1. Wire the Isaac Lab MDP environment class to **live** camera observations from Isaac Sim (not synthetic mock-camera block painting).
2. Integrate visual tracking so the RL network uses real sim camera data for block location and grasp pose estimation.
3. Implement reward function for visual block tracking, end-effector alignment, grasp state, and vertical lifting in the live MDP.
4. Reuse deterministic safety-boundary gtests (`test_safety_boundaries.cpp`); add live smoke tests for observation/reward pipeline in sim.
5. Add live integration tests (e.g. `test_phase2_live_integration.py`) separate from mock tests.

**Acceptance criteria**

- Safety penalization metrics trigger correctly on boundary violations **before** training execution.
- Live vision pipeline feeds accurate bounding/centroid data into the RL observation space.
- Live tests pass without mock vision or mock observation bridge nodes.

---

## Verification loop

1. Build live packages:
   ```bash
   source /opt/ros/jazzy/setup.bash
   cd ${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}
   colcon build --packages-select spark_verify_pkg
   source install/setup.bash
   ```
2. With Isaac Sim playing, run live integration tests (once implemented):
   ```bash
   export ROS_DOMAIN_ID=42
   export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
   colcon test --packages-select spark_verify_pkg --event-handlers console_direct+
   colcon test-result --all
   ```
3. If tests fail, debug against live sim telemetry (`ros2 topic list`, `ros2 topic hz`, TF echo). Iterate until Live Phase 1 and Phase 2 acceptance criteria pass.
4. Keep mock `colcon test` passing — do not remove or break mock suites.

---

## Git (only when explicitly requested)

Stage live Phase 1/2 implementation and tests. Commit with a message detailing live integration work. Push to `https://github.com/jywilson2/spark_isaac_mycobot_demo.git` on `main` only when the user asks.

---

## Agent invocation example

```
Please execute the instructions defined in @commands/test_live.md based on @spec.md and @.cursorrules.

Prerequisites: Isaac Sim is running with the MyCobot scene loaded, Play is pressed, and ros2 topic hz /clock shows data in Docker.
Do not use mock phase1/phase2 launch files for live verification.
```

If Isaac Sim is not running:

```
Please execute @commands/test_live.md based on @spec.md.
Isaac Sim is not running — start it, load the MyCobot scene, enable the ROS 2 bridge, then proceed with live Phase 1 and 2 implementation and testing.
```
