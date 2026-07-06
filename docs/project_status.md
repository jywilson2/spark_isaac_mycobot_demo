# Project Status — Return Briefing

Last updated: **2026-07-06** (branch `wip_live_testing`)

This document summarizes **where the project stands** so you can resume after a long break without re-discovering context. For operational commands, see [README.md](../README.md) and [isaac_sim_host_scripts.md](isaac_sim_host_scripts.md).

## New to this project? Start here

Read these in order if you are joining for the first time:

| Order | Document | Why read it |
|-------|----------|-------------|
| 1 | [README.md](../README.md) | **Main entry point** — repo layout, DGX Spark + Cursor daily workflow, build/test commands, Isaac Sim live steps (Phases 1–2), troubleshooting |
| 2 | [spec.md](../spec.md) | **Full project specification** — goals, phase backlog, acceptance criteria, **Isaac Lab requirement** |
| 3 | [docs/isaac_sim_host_scripts.md](isaac_sim_host_scripts.md) | **Host-only Isaac Sim scripts** — URDF probe, scene build iteration, environment detection, log locations |
| 4 | [commands/test_live.md](../commands/test_live.md) | **Live verification playbook** — prerequisite gate, Phase 1/2 live acceptance sequence |

## One-paragraph summary

Mock Phases 1–4 and container tests are green (`colcon test`, 27/27). **Isaac Lab is required and installed** on the host (`~/IsaacLab`, `develop` branch, pinned via `isaac_lab/versions.env`). **Native Isaac Lab PPO training** runs on the Isaac Sim host via `scripts/host/run_isaac_lab_training.sh train`. The MyCobot DirectRLEnv (`Spark-MyCobot-PickPlace-Direct-v0`) uses the 14-dim MDP contract shared with the ROS verification stack. **PPO training verified** on host (8 iterations, 2 envs, mean reward ≈ 74–123, checkpoint written).

## Architecture reminder

```
┌─────────────────────────────────────────────────────────────┐
│  HOST (DGX Spark) — REQUIRED for Phase 2 training           │
│  Isaac Sim 6.x (~/isaacsim) + Isaac Lab (~/IsaacLab)        │
│  PPO: ./scripts/host/run_isaac_lab_training.sh train        │
└──────────────────────────┬──────────────────────────────────┘
                           │ ROS 2 (optional live verification)
┌──────────────────────────▼──────────────────────────────────┐
│  Isaac ROS container (Cursor)                                 │
│  phase1/2 live stacks, colcon test, ROS-bridge smoke tests    │
└─────────────────────────────────────────────────────────────┘
```

## Completed ✅

| Area | Status | Notes |
|------|--------|-------|
| Mock Phase 1–4 | ✅ | `colcon test --packages-select spark_verify_pkg` (27 tests) |
| Isaac Sim scene build | ✅ | `assets/scenes/mycobot_280_m5_limo_cobot.usd` |
| **Isaac Lab install script** | ✅ | `scripts/host/install_isaac_lab.sh` |
| **Isaac Lab verify script** | ✅ | `scripts/host/verify_isaac_lab.sh` |
| **Isaac Lab DirectRLEnv** | ✅ | `isaac_lab/mycobot_pick_place_env.py` |
| **Isaac Lab PPO trainer** | ✅ | `isaac_lab/train_ppo.py` + `rsl_rl_ppo_cfg.py` |
| **Isaac Lab unit tests** | ✅ | `isaac_lab/test/test_mdp_contract.py`, `test_detect_isaac_lab.py` |
| **Host integration tests** | ✅ | `test_isaac_lab_integration.py` (requires host install) |
| ROS live Phase 1/2 | ✅ | Live tests auto-skip without sim |
| ROS-bridge PPO (legacy smoke) | ✅ | `live_ppo_trainer` — not primary training path |

## Isaac Lab workflow (required)

| Step | Command |
|------|---------|
| Install (once) | `./scripts/host/install_isaac_lab.sh` |
| Verify | `./scripts/host/verify_isaac_lab.sh` |
| Train PPO | `./scripts/host/run_isaac_lab_training.sh train --headless --max-iterations 8 --num-envs 2` |
| Check install | `./scripts/host/run_isaac_lab_training.sh check` |

**Pinning:** `isaac_lab/versions.env` — Isaac Sim 6.x + Isaac Lab `develop` branch + RSL-RL.

**Checkpoints:** `assets/checkpoints/isaac_lab_ppo/latest_policy` + `training_summary.json` (gitignored).

## Execution log (2026-07-06)

| Run | Environment | Command | Result |
|-----|-------------|---------|--------|
| Container unit tests | Container | `colcon test --packages-select spark_verify_pkg` | **27/27 passed** |
| Isaac Lab install | Host (jywilson) | `./scripts/host/install_isaac_lab.sh` | **Passed** — cloned `~/IsaacLab` (`develop`), linked `_isaac_sim` → `~/isaacsim`, installed `rsl_rl` |
| Isaac Lab verify | Host | `./scripts/host/install_isaac_lab.sh --verify-only` | **Passed** — imports + headless env smoke (4 steps) |
| **Isaac Lab PPO training** | Host | `run_isaac_lab_training.sh train --headless --max-iterations 8 --num-envs 2` | **Passed** — 8 iterations, 384 steps, mean reward ≈ 74.43, checkpoint at `assets/checkpoints/isaac_lab_ppo/latest_policy` |
| Isaac Lab branch note | Host | `main` branch | **Failed** — incompatible with Isaac Sim 6.0 (`omni.physics.tensors.impl` missing); use `develop` |

**Training summary sample:**

```json
{"task": "Spark-MyCobot-PickPlace-Direct-v0", "num_envs": 2, "max_iterations": 8,
 "checkpoint": ".../assets/checkpoints/isaac_lab_ppo/latest_policy",
 "log_dir": ".../assets/checkpoints/isaac_lab_ppo/logs"}
```

## Suggested resume checklist

1. Host: `isaac-ros activate` → Cursor → `source scripts/source_container_env.sh`
2. Host: `./scripts/host/run_isaac_lab_training.sh check` (Isaac Lab must be installed)
3. If Isaac Lab missing: `./scripts/host/install_isaac_lab.sh`
4. If robot USD missing: `./scripts/host/iter_build_isaac_scene.sh`
5. Train: `./scripts/host/run_isaac_lab_training.sh train --headless --max-iterations 20`
6. Optional ROS live gate: `./scripts/run_live_sim.sh` + `./scripts/run_live_tests.sh`

## Branch and remote

- **Working branch:** `wip_live_testing`
- **Remote:** `https://github.com/jywilson2/spark_isaac_mycobot_demo.git`
