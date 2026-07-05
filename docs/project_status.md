# Project Status — Return Briefing

Last updated: **2026-07-05** (branch `wip_live_testing`)

This document summarizes **where the project stands** so you can resume after a long break without re-discovering context. For operational commands, see [README.md](../README.md) and [isaac_sim_host_scripts.md](isaac_sim_host_scripts.md).

## New to this project? Start here

Read these in order if you are joining for the first time:

| Order | Document | Why read it |
|-------|----------|-------------|
| 1 | [README.md](../README.md) | **Main entry point** — repo layout, DGX Spark + Cursor daily workflow, build/test commands, Isaac Sim live steps (Phases 1–2), troubleshooting |
| 2 | [spec.md](../spec.md) | **Full project specification** — goals, phase backlog, acceptance criteria, hardware targets (MyCobot 280, RPi + AI Hat) |
| 3 | [docs/isaac_sim_host_scripts.md](isaac_sim_host_scripts.md) | **Host-only Isaac Sim scripts** — URDF probe, scene build iteration, environment detection, log locations |
| 4 | [commands/test_live.md](../commands/test_live.md) | **Live verification playbook** — prerequisite gate, Phase 1/2 live acceptance sequence, agent prompts (do not use mock launch files for live sign-off) |

### Reference and background

| Document | Purpose |
|----------|---------|
| [REFERENCES.md](../REFERENCES.md) | Curated links: MyCobot 280, ROS 2 Jazzy, Isaac Sim/Lab, RL, edge deployment |
| [LICENSE.md](../LICENSE.md) | Apache 2.0 license |
| [.cursorrules](../.cursorrules) | AI/agent coding constraints (ROS 2 Jazzy, TDD, Isaac Sim API patterns) |

### Phase generation playbooks (`commands/`)

These were used to scaffold the repo incrementally. Useful for understanding *why* a package or node exists, not required for day-to-day operation:

| Document | Scope |
|----------|-------|
| [commands/initial_project_generation.md](../commands/initial_project_generation.md) | Phase 1 — URDF, ROS 2 bridge, mock perception |
| [commands/initial_project_generation_phase2.md](../commands/initial_project_generation_phase2.md) | Phase 2 — MDP, rewards, vision, safety |
| [commands/initial_project_generation_phase_remaining.md](../commands/initial_project_generation_phase_remaining.md) | Phases 3–4 — ONNX export, pymycobot, RPi HIL |

### External assets

| Location | Purpose |
|----------|---------|
| [third_party/mycobot_ros2/](../third_party/mycobot_ros2/) | Upstream Elephant Robotics URDF/meshes (git submodule, branch `humble`) |
| [third_party/mycobot_ros2/README.md](../third_party/mycobot_ros2/README.md) | Upstream package documentation |

### Where this doc fits

- **First visit** → README → spec → host scripts doc → test_live playbook (table above).
- **Returning after a break** → stay on this page (sections below), then run the [resume checklist](#suggested-resume-checklist).
- **Debugging scene/URDF build** → [isaac_sim_host_scripts.md](isaac_sim_host_scripts.md) iteration workflow.

## One-paragraph summary

Mock Phases 1–4 are complete and green via `colcon test`. **Live Phase 1 and 2** ROS stacks and integration tests exist. **Isaac Sim scene build now works on the host** (Isaac Sim 6.x, pre-built install at `~/isaacsim`). The robot base uses a box placeholder instead of `G_base.dae` due to an importer bug. **Isaac Lab PPO training against the live sim is not yet scripted** — the MDP facade exists and live topic plumbing is ready, but the training launcher and end-to-end live training verification remain the next milestone.

## Architecture reminder

```
┌─────────────────────────────────────────────────────────────┐
│  HOST (DGX Spark)                                           │
│  Isaac Sim: scene USD, articulation, camera, /clock bridge  │
│  Future: Isaac Lab trainer (Terminal C)                     │
└──────────────────────────┬──────────────────────────────────┘
                           │ ROS 2 (ROS_DOMAIN_ID=42, UDPv4)
┌──────────────────────────▼──────────────────────────────────┐
│  Isaac ROS container (Cursor, user admin uid 1000)            │
│  phase1/2_live_ecosystem.launch.py, live tests, colcon      │
└─────────────────────────────────────────────────────────────┘
```

**Daily startup:** host `isaac-ros activate` → open Cursor (auto-attach) → `source scripts/source_container_env.sh`.

## Completed ✅

| Area | Status | Notes |
|------|--------|-------|
| Mock Phase 1–4 | ✅ | `colcon test --packages-select spark_verify_pkg` |
| `mycobot_ros2` submodule | ✅ | `third_party/mycobot_ros2`, branch `humble` |
| URDF → USD import (Isaac Sim 6) | ✅ | `URDFImporter` API; see host iteration scripts |
| Scene USD | ✅ | `assets/scenes/mycobot_280_m5_limo_cobot.usd` |
| Host iteration toolkit | ✅ | `scripts/host/*` with logs in `assets/logs/isaac_host/` |
| Live Phase 1 stack | ✅ | `phase1_live_ecosystem.launch.py`, live tests (auto-skip without sim) |
| Live Phase 2 stack | ✅ | Vision, RL observation, `live_mdp_reward_monitor` |
| MDP facade | ✅ | `IsaacLabMyCobotPickPlaceEnv` in `isaac_lab_mdp_env.py` |
| Live sim runner | ✅ | `./scripts/run_live_sim.sh` |

## In progress / next up 🔜

### 1. End-to-end live verification (before training)

Run once after clone or major changes:

| Step | Where | Command |
|------|-------|---------|
| Scene build | Host | `./scripts/host/iter_build_isaac_scene.sh` |
| Live sim | Host | `./scripts/run_live_sim.sh` |
| Bridge check | Container | `ros2 topic hz /clock` |
| Phase 2 stack | Container | `ros2 launch spark_verify_pkg phase2_live_ecosystem.launch.py` |
| Live tests | Container | `./scripts/run_live_tests.sh` |

**Gate:** live tests must pass before starting Isaac Lab training (`README.md` Step 7).

### 2. Isaac Lab training (primary next feature)

**Not implemented yet.** Planned layout:

| Terminal | Role |
|----------|------|
| A (host) | `./scripts/run_live_sim.sh` |
| B (container) | `phase2_live_ecosystem.launch.py` |
| C (host) | Isaac Lab PPO trainer — **to be added** |

**Existing hooks:**

- `spark_verify_pkg/spark_verify_nodes/isaac_lab_mdp_env.py` — `IsaacLabMyCobotPickPlaceEnv` reads live ROS topics (`/mycobot/rl/observation`, joint states, etc.)
- `live_mdp_reward_monitor.py` — publishes `/mycobot/rl/live_reward` and safety penalty for smoke tests
- Reward / safety logic in `reward_function.py`, `safety_boundary_evaluator`

**Still needed:**

- [ ] Isaac Lab install path documented and version-pinned to host Isaac Sim 6.0
- [ ] Training script or launch config (PPO) wired to `IsaacLabMyCobotPickPlaceEnv` or native Isaac Lab env wrapping the same observation space
- [ ] Checkpoint export path toward Phase 3 ONNX pipeline
- [ ] Live test asserting training loop can step at least N episodes with sim playing

### 3. Known limitations (acceptable for now)

| Item | Workaround |
|------|------------|
| `G_base.dae` broken in Isaac Sim 6 importer | Box placeholder in prepared URDF (`0.12 × 0.12 × 0.06` m) |
| Isaac Sim 6 robot USD is `.usda` + payloads dir | Scene builder references imported robot correctly |
| Host scripts fail inside container | Run on native host terminal; see `scripts/host/` |
| Isaac Lab training not scripted | Operator-driven; see section 2 above |

## File locations (quick lookup)

| Artifact | Path |
|----------|------|
| Scene USD | `assets/scenes/mycobot_280_m5_limo_cobot.usd` |
| Robot URDF + meshes | `assets/robots/mycobot_280_m5_limo_cobot/` |
| Host build logs | `assets/logs/isaac_host/latest.log` |
| Live test playbook | [commands/test_live.md](../commands/test_live.md) |
| Full spec / backlog | [spec.md](../spec.md) |
| Host script guide | [docs/isaac_sim_host_scripts.md](isaac_sim_host_scripts.md) |
| Main README | [README.md](../README.md) |

## Branch and remote

- **Working branch:** `wip_live_testing`
- **Remote:** `https://github.com/jywilson2/spark_isaac_mycobot_demo.git`

## Suggested resume checklist

1. Host: `isaac-ros activate` → Cursor open → `whoami` = `admin`
2. Container: `source scripts/source_container_env.sh`
3. Host: `./scripts/host/check_prereqs.sh`
4. If scene missing or URDF changed: `./scripts/host/iter_build_isaac_scene.sh`
5. Host: `./scripts/run_live_sim.sh` (keep running)
6. Container: `ros2 topic hz /clock`
7. Container: `./scripts/run_live_tests.sh`
8. If live tests green → start Isaac Lab training work (section 2)
