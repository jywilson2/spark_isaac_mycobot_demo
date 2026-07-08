# Project Status — Return Briefing

Last updated: **2026-07-08** (branch `wip_live_testing`, commit after verification)

This document summarizes **where the project stands** so you can resume after a long break without re-discovering context. For operational commands, see [README.md](../README.md) and [isaac_sim_host_scripts.md](isaac_sim_host_scripts.md).

## New to this project? Start here

| Order | Document | Why read it |
|-------|----------|-------------|
| 1 | [README.md](../README.md) | **Main entry point** — repo layout, DGX Spark + Cursor daily workflow, **Phase in development** changelog |
| 2 | [spec.md](../spec.md) | **Full project specification** — smooth motion, two-phase train, demo validation horizons |
| 3 | [docs/isaac_sim_host_scripts.md](isaac_sim_host_scripts.md) | **Host-only Isaac Sim scripts** — URDF probe, scene build iteration |
| 4 | [commands/test_live.md](../commands/test_live.md) | **Live verification playbook** — prerequisite gate |

## One-paragraph summary

Mock Phases 1–4 and container tests are green. **Two-phase from-scratch training** (smooth motion + 30 s horizons) completed on the DGX Spark host: Phase A curriculum reached **88.3%** rolling reach in 90 min; Phase B demo fine-tune reached **97.7%**. **Multi-horizon demo verification passed:** **99/100 (99%)** at **20 s, 30 s, and 40 s** episode lengths. GUI showcase: `./scripts/host/run_isaac_lab_training.sh demo`.

## Architecture reminder

```
┌─────────────────────────────────────────────────────────────┐
│  HOST (DGX Spark) — REQUIRED for Phase 2 training           │
│  Isaac Sim 6.x + Isaac Lab                                  │
│  Two-phase: ./scripts/host/run_isaac_lab_training.sh        │
│             two-phase --headless                            │
│  Demo verify: ... verify-demo --headless                    │
└──────────────────────────┬──────────────────────────────────┘
                           │ ROS 2 (optional live verification)
┌──────────────────────────▼──────────────────────────────────┐
│  Isaac ROS container (Cursor)                                 │
│  colcon test, ROS-bridge smoke tests                          │
└───────────────────────────────────────────────────────────────┘
```

## Completed ✅

| Area | Status | Notes |
|------|--------|-------|
| Mock Phase 1–4 | ✅ | `colcon test --packages-select spark_verify_pkg` |
| Phase 2 reach env | ✅ | Joint-space PPO, EMA smoothing, jerk penalty |
| Two-phase training script | ✅ | `scripts/run_two_phase_training.sh` |
| **From-scratch two-phase train (2026-07-08)** | ✅ | Phase A 88.3% @ 90 min → Phase B **97.7%** @ 30 min |
| **Multi-horizon demo verify (2026-07-08)** | ✅ | **99/100** @ 20 s, 30 s, 40 s (≥ 95% gate) |
| Training defaults | ✅ | 30 s episodes; plateau abort default off |

## Execution log (2026-07-08)

| Run | Result |
|-----|--------|
| Phase A — curriculum, 90 min, from scratch | **88.3%** rolling reach (`max_duration`, target not met) |
| Phase B — demo fine-tune, 30 min, 8 arms | **97.7%** rolling reach (`task_requirement_met: YES`) |
| Demo verify @ 20 s | **99/100 (99%)** |
| Demo verify @ 30 s | **99/100 (99%)** |
| Demo verify @ 40 s | **99/100 (99%)** |

**Checkpoint:** `assets/checkpoints/isaac_lab_ppo/latest_policy` (gitignored)

## Isaac Lab workflow

| Step | Command |
|------|---------|
| **Two-phase train + verify** | `./scripts/host/run_isaac_lab_training.sh two-phase --headless` |
| Demo verify only | `./scripts/host/run_isaac_lab_training.sh verify-demo --headless` |
| **GUI demo** | `./scripts/host/run_isaac_lab_training.sh demo` |

## Suggested resume checklist

1. Host: `isaac-ros activate` → Cursor → `source scripts/source_container_env.sh`
2. `./scripts/host/run_isaac_lab_training.sh check`
3. **Visual demo:** `./scripts/host/run_isaac_lab_training.sh demo`
4. Optional re-verify: `./scripts/host/run_isaac_lab_training.sh verify-demo --headless`

## Branch and remote

- **Working branch:** `wip_live_testing`
- **Remote:** `https://github.com/jywilson2/spark_isaac_mycobot_demo.git`
