Review @spec.md and @.cursorrules. We are moving into **live** verification for Phase 1 and Phase 2 inside Isaac Sim / Isaac Lab. Do **not** use the mock launch files (`phase1_mock_ecosystem.launch.py`, `phase2_mock_ecosystem.launch.py`) for live acceptance.

## Prerequisites (agent must confirm before testing)

1. Mock regression is green (optional sanity check):
   ```bash
   source /opt/ros/jazzy/setup.bash
   cd ${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}
   colcon build --packages-select spark_verify_pkg
   source install/setup.bash
   export SPARK_SKIP_LIVE_SIM_TESTS=1
   colcon test --packages-select spark_verify_pkg
   colcon test-result --all
   ```
2. **Isaac Sim is running on the host** with the MyCobot live scene and **Play** pressed:
   ```bash
   export ROS_DOMAIN_ID=42
   export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
   export ISAACSIM_PATH=/path/to/isaac-sim
   cd /path/to/spark_isaac_mycobot_demo
   ./scripts/run_live_sim.sh
   ```
   See [README.md](../README.md#running-isaac-sim).
3. **ROS 2 bridge is active** — Docker/Cursor terminal receives sim data:
   ```bash
   export ROS_DOMAIN_ID=42
   export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
   ros2 topic hz /clock
   ros2 topic list | grep mycobot
   ```
   Expect a non-zero rate before proceeding.
4. For **Live Phase 2**, launch the Docker-side live stack while sim is playing:
   ```bash
   source install/setup.bash
   ros2 launch spark_verify_pkg phase2_live_ecosystem.launch.py
   ```

If Isaac Sim is not running, start it and confirm the bridge **before** live tests. Do not substitute mock nodes.

---

## Live Phase 1 sequence

Implemented in this repository:

1. `third_party/mycobot_ros2` submodule (`mycobot_280_m5.urdf`, Limo Cobot arm).
2. `isaac_sim/build_mycobot_limo_cobot_scene.py` + `isaac_sim/ros2_bridge_config.py` — Isaac Sim articulation + camera ROS 2 bridge on the host.
3. `phase1_live_ecosystem.launch.py` — Docker-side `joint_command_dispatcher` (URDF joint names) + `live_nitros_camera_bridge`.
4. `test_phase1_live_integration.py` — separate from mock `test_phase1_integration.py`.

**Acceptance criteria**

- Joint commands update live articulation state on `/mycobot/joint_states` (verify TF: `g_base` → `joint6_flange`).
- Live camera publishes RGB on `/mycobot/camera/rgb` and NITROS handles on `/mycobot/camera/nitros/rgb`.
- Live tests pass without mock articulation or mock camera nodes running.

**Manual launch**

```bash
ros2 launch spark_verify_pkg phase1_live_ecosystem.launch.py
```

---

## Live Phase 2 sequence

Implemented in this repository:

1. `IsaacLabMyCobotPickPlaceEnv` (`isaac_lab_mdp_env.py`) — MDP facade for live Isaac Lab integration.
2. `block_vision_tracker` — live RGB block detection (red workspace block in scene USD).
3. `rl_observation_bridge` + `live_mdp_reward_monitor` — observation/reward pipeline from live sim topics.
4. `test_safety_boundaries.cpp` (mock gtests) + `test_safety_boundary_evaluator_py.py` + live smoke in `test_phase2_live_integration.py`.

**Acceptance criteria**

- Safety penalization metrics trigger correctly on boundary violations **before** training execution.
- Live vision pipeline feeds block centroid/bbox data into `/mycobot/rl/observation`.
- Live tests pass without mock vision or mock observation bridge substitutes.

**Manual launch**

```bash
ros2 launch spark_verify_pkg phase2_live_ecosystem.launch.py
```

---

## Verification loop

1. Build live packages:
   ```bash
   source /opt/ros/jazzy/setup.bash
   cd ${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}
   colcon build --packages-select spark_verify_pkg
   source install/setup.bash
   ```
2. With Isaac Sim playing, run live integration tests:
   ```bash
   export ROS_DOMAIN_ID=42
   export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
   cd /path/to/spark_isaac_mycobot_demo
   ./scripts/run_live_tests.sh
   ```
   Or full suite (live tests auto-skip when sim is absent):
   ```bash
   colcon test --packages-select spark_verify_pkg --event-handlers console_direct+
   colcon test-result --all
   ```
3. If tests fail, debug against live sim telemetry (`ros2 topic list`, `ros2 topic hz`, TF echo). Iterate until Live Phase 1 and Phase 2 acceptance criteria pass.
4. Keep mock `colcon test` passing — do not remove or break mock suites. Use `SPARK_SKIP_LIVE_SIM_TESTS=1` for mock-only CI runs.

---

## Git (only when explicitly requested)

Stage live Phase 1/2 implementation and tests. Commit with a message detailing live integration work. Push to `https://github.com/jywilson2/spark_isaac_mycobot_demo.git` on `main` only when the user asks.

---

## Agent invocation example

```
Please execute the instructions defined in @commands/test_live.md based on @spec.md and @.cursorrules.

Prerequisites: Isaac Sim is running via ./scripts/run_live_sim.sh, Play is pressed, and ros2 topic hz /clock shows data in Docker.
Do not use mock phase1/phase2 launch files for live verification.
```

If Isaac Sim is not running:

```
Please execute @commands/test_live.md based on @spec.md.
Isaac Sim is not running — start ./scripts/run_live_sim.sh on the host, then proceed with live Phase 1 and 2 testing.
```
