# MyCobot Isaac Sim Reinforcement Learning Pipeline Spec

## Project Overview

An automated pipeline simulating the Elephant Robotics MyCobot arm inside NVIDIA Isaac Sim via ROS 2 Jazzy, training an **EE reach-to-target** policy in **NVIDIA Isaac Lab** (Phase 2), and porting weights to a physical MyCobot device. Red-block vision localization and contact-and-push are deferred to **Phase 6** (opt-in only).

### Project objective — tutorial-quality source code

All source code and scripts in this repository **must** contain **verbose inline documentation** so the codebase reads as a **textbook on robotics software development with reinforcement learning**. Comments and docstrings should:

* Explain *why* each design choice exists, not only *what* the code does.
* Link to **internal documentation** (e.g. [spec.md](spec.md), [README.md](README.md), [docs/project_status.md](docs/project_status.md)).
* Link to **authoritative external references** where helpful (Isaac Lab, RSL-RL, ROS 2 Jazzy, operational-space control, PPO).
* Describe how MDP variables (observations, actions, rewards) **translate to physical motion** on the MyCobot arm.

The default training CLI prints a **verbose motion glossary** (`--motion-glossary`, on by default; disable with `--no-motion-glossary`) that mirrors this tutorial intent at runtime. Isaac Lab reserves `--verbose` for kit logging.

### Strategic goal — RL for IK, efficient motion, and planning

Phase 2 demonstrates that **inverse kinematics-style reach**, **time-efficient motion**, and rudimentary **motion planning** are **learned by the RL policy** — not delegated to analytic, numeric, or differential IK solvers. The PPO network maps observations (EE-to-target vector + joint state) to **joint position deltas**; coordinating those deltas so the flange reaches a 3D target *is* the learned IK. Phase 2b (obstacles) extends this to planning after Phase 2 reaches **99%** success.

The implementation must remain **sufficiently generic** to train a policy that transfers to a **physical MyCobot 280** in arbitrary reach scenarios: full-workspace target sampling, curriculum → demo fine-tune, and validation at multiple episode horizons — not a single scripted pose.

### Smooth motion (required)

Arm actuation must **not stutter**. Training and inference apply **EMA-smoothed joint deltas** plus **jerk penalties** in the reward so motions ramp up and down smoothly. **Smooth, wear-conscious motion is more important than minimizing episode duration** — longer horizons are acceptable when they improve servo-friendly trajectories.

### Host vs container execution (required)

| Runtime | Runs Isaac Sim / Isaac Lab? | How agents and scripts execute |
|---------|----------------------------|--------------------------------|
| **DGX Spark host** (native shell after `isaac-ros activate`) | **Yes** — `~/isaacsim`, `~/IsaacLab` | `./scripts/host/run_isaac_lab_training.sh train` |
| **Isaac ROS container** (Cursor attached) | **No** — no GPU sim binaries | Scripts **auto-delegate to the host** via `nsenter` ([scripts/host/spark_host_exec.sh](scripts/host/spark_host_exec.sh)) |

**Do not** assume training failed because the container lacks `python.sh`; container invocations of `run_isaac_lab_training.sh` or `./scripts/run_live_training.sh` must transparently re-exec on the host. Override host repo path with `SPARK_HOST_REPO_ROOT` or host user with `SPARK_HOST_USER` (default `admin`) when needed.

Daily workflow: host runs `isaac-ros activate` → Cursor restores workspace → agents call training scripts from the container → host Isaac Sim runs PPO.

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

**Prohibited:** analytic IK, numeric IK, differential IK (`DifferentialIKController`), Jacobian pseudoinverse, or any solver that maps target pose → joint angles. The **PPO policy** must learn that mapping through joint-space actions.

### Task definition

1. **Randomize target EE position** each episode within the arm reach envelope (annulus 0.12–0.28 m horizontally; Z 0.08–0.22 m).
2. **Mark the target in simulation** with a visible **red sphere** (visual only — contact with the marker is **not** required).
3. **Observations (11-dim motion policy):** normalized EE-to-target delta (3), `target_valid` (1), `reached` (1), joint positions (6).
4. **Actions (6-dim joint space):** policy outputs **joint position deltas** (Δq, one per revolute joint), scaled and clipped; **soft joint limits** enforced. Raw actions are **EMA-smoothed** before application; a **jerk penalty** discourages abrupt step-to-step changes. **No IK solver** (analytic, numeric, differential, or Jacobian pseudoinverse) may appear in the control loop — the policy *is* the learned IK mapping.
5. **Reward:** **potential-based** distance reduction + per-step **time penalty** + **action-magnitude** and **jerk** penalties + large **terminal bonus** inside tolerance.
6. **Curriculum:** staged target sampling — near current EE → medium annulus → full workspace.
7. **Success:** EE within **25 mm**; early terminate on success.
8. **Episode length:** default **30 s** for training **and** demo/play (shared constant in `isaac_lab/training_defaults.py`).
9. **Training stop:** rolling **reach success rate ≥ target** (default **99%**; two-phase recipe uses **95%**) or **`--max-duration-minutes`** (default **30**). **Plateau abort is opt-in** (`--plateau-abort`); default is **off** so the duration budget is primary.
10. **Two-phase training (recommended):** `./scripts/run_two_phase_training.sh` — Phase A curriculum from scratch, Phase B demo-target fine-tune, then multi-horizon demo verify.
11. **Demo validation:** after training, **demo mode** is the acceptance gate. Verify with `./scripts/verify_demo_policy.sh` at **20 s, 30 s, and 40 s** episode lengths; each must retain **≥ 95%** reach success.
12. **Motion glossary CLI:** `--motion-glossary` (default **on**) prints MDP/motion tutorial output; `--no-motion-glossary` disables. (Isaac Lab reserves `--verbose` for kit logging.)

| Mode | Default arms | Rationale |
|------|--------------|-----------|
| **GUI** | **2** | Viewport + marker visualization |
| **Headless** | **8** | DGX Spark throughput |

### Commands (Phase 2 default)

| Step | Command |
|------|---------|
| Train (GUI, 2 arms) | `./scripts/host/run_isaac_lab_training.sh train` |
| Train (headless, 8 arms) | `./scripts/host/run_isaac_lab_training.sh train --headless` |
| **Two-phase train + verify** | `./scripts/host/run_isaac_lab_training.sh two-phase --headless` |
| Play trained policy | `./scripts/host/run_isaac_lab_training.sh play` |
| **Continuous GUI demo** (1 arm, until exit) | `./scripts/host/run_isaac_lab_training.sh demo` |
| **Demo regression (20/30/40 s)** | `./scripts/host/run_isaac_lab_training.sh verify-demo --headless` |
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

Every **commit** that changes behavior, scripts, training recipes, or verification results **must** update **all** of:

| Document | Purpose |
|----------|---------|
| [spec.md](spec.md) | Authoritative requirements (this file) |
| [README.md](README.md) | **Phase in development** — latest committed change, what changed, and why |
| [docs/project_status.md](docs/project_status.md) | Operational status after each commit (training/demo gates, blockers) |
| [last_prompt.md](last_prompt.md) | Append-only user prompt log (see below) |
| [docs/isaac_lab_warnings_audit.md](docs/isaac_lab_warnings_audit.md) | Warning triage (when warnings change) |

README and `project_status.md` are **not optional** — they must reflect the same commit as the code change.

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
* [x] DirectRLEnv (`isaac_lab/mycobot_reach_env.py`) — joint-space RL (learned IK), curriculum, red marker
* [x] Host auto-delegate (`scripts/host/spark_host_exec.sh`) — container → host via `nsenter`
* [x] PPO trainer (`isaac_lab/train_ppo.py`) — duration-bounded train, verbose CLI (default on)
* [x] Two-phase training script (`scripts/run_two_phase_training.sh`) + demo verify (`scripts/verify_demo_policy.sh`)
* [x] Smooth motion — EMA action smoothing + jerk penalty in reach reward
* [x] Train-until-success loop (`isaac_lab/training_success.py`)
* [x] Unit + integration tests
* [ ] **Verify ≥ 95% reach** at 20/30/40 s demo horizons on DGX Spark headless two-phase train

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
