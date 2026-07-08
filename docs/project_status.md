# Project Status — Return Briefing

Last updated: **2026-07-08** (branch `wip_live_testing`)

This document summarizes **where the project stands** so you can resume after a long break without re-discovering context. For operational commands, see [README.md](../README.md) and [isaac_sim_host_scripts.md](isaac_sim_host_scripts.md).

## New to this project? Start here

| Order | Document | Why read it |
|-------|----------|-------------|
| 1 | [README.md](../README.md) | **Main entry point** — repo layout, DGX Spark + Cursor daily workflow, **Phase in development** changelog |
| 2 | [spec.md](../spec.md) | **Full project specification** — smooth motion, two-phase train, demo validation horizons |
| 3 | [docs/isaac_sim_host_scripts.md](isaac_sim_host_scripts.md) | **Host-only Isaac Sim scripts** — URDF probe, scene build iteration |
| 4 | [commands/test_live.md](../commands/test_live.md) | **Live verification playbook** — prerequisite gate |

## One-paragraph summary

Mock Phases 1–4 and container tests are green. Phase 2 EE reach PPO now uses **30 s training episodes** (matching demo), **EMA-smoothed actions + jerk penalties** for servo-friendly motion, **plateau abort off by default**, and a **reproducible two-phase script** (curriculum → demo fine-tune → 20/30/40 s demo verify). Prior policy reached **97/100** demo successes before these changes; **from-scratch retrain with the new recipe is in progress** to confirm ≥ 95% at all demo horizons.

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
| Isaac Sim scene build | ✅ | Robot USD under `assets/robots/` |
| Phase 2 reach env | ✅ | Joint-space PPO, curriculum + demo sampling |
| Smooth motion shaping | ✅ | EMA smoothing (`action_smoothing_alpha=0.35`), jerk penalty |
| Two-phase training script | ✅ | `scripts/run_two_phase_training.sh` |
| Multi-horizon demo verify | ✅ | `scripts/verify_demo_policy.sh` — 20/30/40 s, ≥ 95% gate |
| Training defaults aligned | ✅ | 30 s episodes; plateau abort default off |
| Prior demo verify (pre-smooth-motion) | ✅ | 97/100 @ 30 s (2026-07-07 checkpoint) |

## In progress 🔄

| Item | Status | Notes |
|------|--------|-------|
| **From-scratch two-phase retrain + demo verify** | 🔄 **In progress** | Phase A curriculum running on host; multi-horizon verify (20/30/40 s) pending |
| Push to `origin/wip_live_testing` | ⏸ | Blocked by approval gate in prior session |

## Isaac Lab workflow

| Step | Command |
|------|---------|
| **Two-phase train + verify** | `./scripts/host/run_isaac_lab_training.sh two-phase --headless` |
| Train only (curriculum) | `... train --headless --from-scratch --max-duration-minutes 90 --target-reach-success-rate 0.95` |
| Demo fine-tune only | `... train --headless --resume --target-sampling demo --no-early-success-stop --max-duration-minutes 30` |
| Demo verify (20/30/40 s) | `... verify-demo --headless` |
| **GUI demo** | `./scripts/host/run_isaac_lab_training.sh demo` |

**Checkpoints:** `assets/checkpoints/isaac_lab_ppo/latest_policy` + `training_summary.json` (gitignored).

## Suggested resume checklist

1. Host: `isaac-ros activate` → Cursor → `source scripts/source_container_env.sh`
2. `./scripts/host/run_isaac_lab_training.sh check`
3. **Retrain + verify:** `./scripts/host/run_isaac_lab_training.sh two-phase --headless`
4. **Visual demo:** `./scripts/host/run_isaac_lab_training.sh demo`
5. Unit tests: `pytest isaac_lab/test/ -q`

## Branch and remote

- **Working branch:** `wip_live_testing`
- **Remote:** `https://github.com/jywilson2/spark_isaac_mycobot_demo.git`
