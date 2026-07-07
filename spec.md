# MyCobot Isaac Sim Reinforcement Learning Pipeline Spec

## Project Overview

An automated pipeline simulating the Elephant Robotics MyCobot arm inside NVIDIA Isaac Sim via ROS 2 Jazzy, training an **EE reach-to-target** policy in **NVIDIA Isaac Lab** (Phase 2), and porting weights to a physical MyCobot device. Red-block vision localization and contact-and-push are deferred to **Phase 6** (opt-in only).

### Project objective — tutorial-quality source code

All source code and scripts in this repository **must** contain **verbose inline documentation** so the codebase reads as a **textbook on robotics software development with reinforcement learning**. Comments and docstrings should:

* Explain *why* each design choice exists, not only *what* the code does.
* Link to **internal documentation** (e.g. [spec.md](spec.md), [README.md](README.md), [docs/project_status.md](docs/project_status.md)).
* Link to **authoritative external references** where helpful (Isaac Lab, RSL-RL, ROS 2 Jazzy, operational-space control, PPO).
* Describe how MDP variables (observations, actions, rewards) **translate to physical motion** on the MyCobot arm.

The default training CLI prints a **verbose motion glossary** (`--verbose`, on by default; disable with `--no-verbose`) that mirrors this tutorial intent at runtime.

### Strategic goal — RL for IK, efficient motion, and planning

Phase 2 demonstrates that **inverse kinematics-style reach**, **time-efficient motion**, and rudimentary **motion planning** can be learned by RL **without an analytic IK solver in the policy loop**. The policy acts in **Cartesian task space** (Δx, Δy, Δz); a damped least-squares Jacobian maps task-space commands to joint targets. Phase 2b (obstacles) extends this to planning after Phase 2 reaches **99%** success.

## Core Tech Stack

* **Environment:** Ubuntu 24.04 (DGX OS 7.2.3) via NVIDIA Spark Platform
* **ROS Version:** ROS 2 Jazzy (containerized via Isaac ROS CLI) — live bridge and verification
* **Simulation Suite:** NVIDIA Isaac Sim 6.x + **NVIDIA Isaac Lab** (required for Phase 2 PPO)
* **RL Framework:** RSL-RL (PPO) via Isaac Lab on the Isaac Sim host
* **Hardware Target:** Elephant Robotics MyCobot 280 (Limo Cobot / `mycobot_280_m5` URDF)
* **Edge Deployment:** Raspberry Pi (RPi) + AI Hat Board
* **Perception Hardware:** RPi USB Camera (Phase 6+)

---

## Phase 2 — EE reach-to-target (default PPO task)

Phase 2 trains **efficient inverse-kinematics-style arm motion**: move `joint6_flange` to a **known 3D target** in the robot base frame. **No vision. No block contact.**

### Task definition

1. **Randomize target EE position** each episode within the arm reach envelope (annulus 0.12–0.28 m horizontally; Z 0.08–0.22 m).
2. **Mark the target in simulation** with a visible **red sphere** (visual only — contact with the marker is **not** required).
3. **Observations (11-dim motion policy):** normalized EE-to-target delta (3), `target_valid` (1), `reached` (1), joint positions (6).
4. **Actions (3-dim Cartesian):** policy outputs Δx, Δy, Δz in the robot base frame (scaled, clipped). A **damped least-squares Jacobian** maps Cartesian deltas to joint position targets; **soft joint limits** are enforced. This is *not* analytic IK — the policy learns task-space motion; the Jacobian is only a low-level actuator interface (see [isaac_lab/cartesian_actuation.py](isaac_lab/cartesian_actuation.py)).
5. **Reward:** **potential-based** distance reduction (policy-invariant shaping) + per-step **time penalty** (efficiency) + large **terminal bonus** inside tolerance. No perpetual proximity reward that pays without reaching.
6. **Curriculum:** staged target sampling — near current EE → medium annulus → full workspace — advancing when rolling success exceeds stage thresholds.
7. **Success:** EE within **25 mm** of target; episode terminates early on success.
8. **Training stop:** rolling **reach success rate ≥ 99%** (default) or **`--max-duration-minutes`** (default **30**). Training is **duration-bounded by default**, not iteration-bounded. Use `--fixed-iterations N` only for short smoke tests.
9. **Verbose CLI:** `--verbose` (default **on**) prints a tutorial glossary of MDP variables and how they map to arm motion; `--no-verbose` disables it.

| Mode | Default arms | Rationale |
|------|--------------|-----------|
| **GUI** | **2** | Viewport + marker visualization |
| **Headless** | **8** | DGX Spark throughput |

### Commands (Phase 2 default)

| Step | Command |
|------|---------|
| Train (GUI, 2 arms) | `./scripts/host/run_isaac_lab_training.sh train` |
| Train (headless, 8 arms) | `./scripts/host/run_isaac_lab_training.sh train --headless` |
| Play trained policy | `./scripts/host/run_isaac_lab_training.sh play` |
| Full verify + integration train | `./scripts/host/verify_isaac_lab.sh --smoke-train` |

Cameras are **off** by default for Phase 2 (not required). EE cameras activate only with `--use-red-block-vision` (Phase 6).

---

## Phase 6 — Red-block vision + contact-and-push (opt-in)

Deferred from Phase 2. Enabled **only** with `--use-red-block-vision` (requires `--enable_cameras`).

| Component | Location |
|-----------|----------|
| Red threshold detector | `isaac_lab/phase6_red_block/block_vision.py` |
| Depth → base-frame localization | `isaac_lab/phase6_red_block/block_localization.py` |
| Init scan sequence | `isaac_lab/phase6_red_block/init_scan.py` |
| Contact-and-push env | `isaac_lab/phase6_red_block/red_block_env.py` |
| ROS tracker (mock/live) | `spark_verify_pkg/spark_verify_nodes/block_vision_tracker.py` |

Phase 6 task: locate red block via vision, move EE to target, contact, push 5 mm. Train with:

```bash
./scripts/host/run_isaac_lab_training.sh train --use-red-block-vision --enable_cameras
```

---

## PPO integration testing (headless)

| Step | Command |
|------|---------|
| Unit tests | `pytest isaac_lab/test/test_mdp_contract.py` |
| Pytest gate (1 min time-limit slice) | `pytest isaac_lab/test/test_isaac_lab_integration.py` |
| Full smoke train (30 min, 8 arms) | `./scripts/host/verify_isaac_lab.sh --smoke-train` |

---

## Isaac Lab requirement

| Step | Command |
|------|---------|
| Install | `./scripts/host/install_isaac_lab.sh` |
| Verify | `./scripts/host/verify_isaac_lab.sh` |
| Train | `./scripts/host/run_isaac_lab_training.sh train` |

---

## Documentation maintenance (required)

Every development session that changes behavior, scripts, or verification results **must** update:

| Document | Purpose |
|----------|---------|
| [spec.md](spec.md) | Authoritative requirements |
| [docs/project_status.md](docs/project_status.md) | Execution log |
| [last_prompt.md](last_prompt.md) | Append-only user prompt log (see below) |
| [docs/isaac_lab_warnings_audit.md](docs/isaac_lab_warnings_audit.md) | Warning triage |

### `last_prompt.md` retention policy (required)

When saving the latest user prompt to [last_prompt.md](last_prompt.md):

1. **Never delete** prior prompt entries.
2. **Prepend** the new prompt at the top (immediately after the file header), so the **most recent prompt is first**.
3. Move the previous “last prompt” block under the `# Old prompts:` section, preserving all historical `## BEGIN` / `## END` blocks in order.

README and host-script docs should stay aligned with `spec.md` command paths.

---

## Incremental Execution Backlog

### Phase 1: URDF, ROS 2 Control Bridge & Perception inside Isaac Sim

* [x] Parse and ingest mycobot_ros2 assets
* [x] Build Isaac Sim standalone environment + live sim runner
* [x] Mount simulated workspace camera
* [x] Integration tests (mock + live)

### Phase 2: EE Reach PPO (Isaac Lab)

* [x] MDP contract (`isaac_lab/mdp_core.py`) — reachable EE sampling, reach rewards
* [x] DirectRLEnv (`isaac_lab/mycobot_reach_env.py`) — Cartesian actions, staged curriculum, red marker
* [x] Cartesian actuation (`isaac_lab/cartesian_actuation.py`) — DLS Jacobian map
* [x] PPO trainer (`isaac_lab/train_ppo.py`) — duration-bounded train, verbose CLI (default on)
* [x] Train-until-success loop (`isaac_lab/training_success.py`)
* [x] Unit + integration tests
* [ ] **Verify 99% reach success** on DGX Spark headless integration train

### Phase 3–4: Sim-to-real & edge deployment

* [x] ONNX export path, pymycobot driver, HIL tests

### Phase 6: Red-block vision + contact-and-push (opt-in)

* [x] Isolated modules under `isaac_lab/phase6_red_block/`
* [x] `--use-red-block-vision` execution flag
* [ ] Replace threshold detector with Isaac ROS DNN
* [ ] Train contact-and-push to spec targets

---

## Key paths

| Artifact | Location |
|----------|----------|
| Phase 2 reach env | `isaac_lab/mycobot_reach_env.py` |
| Phase 6 red-block env | `isaac_lab/phase6_red_block/red_block_env.py` |
| Trainer | `isaac_lab/train_ppo.py` |
| Checkpoints | `assets/checkpoints/isaac_lab_ppo/` |
