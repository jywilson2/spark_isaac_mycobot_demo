# Project Status — Return Briefing

Last updated: **2026-07-07** (branch `wip_live_testing`)

This document summarizes **where the project stands** so you can resume after a long break without re-discovering context. For operational commands, see [README.md](../README.md) and [isaac_sim_host_scripts.md](isaac_sim_host_scripts.md).

## New to this project? Start here

| Order | Document | Why read it |
|-------|----------|-------------|
| 1 | [README.md](../README.md) | **Main entry point** — repo layout, DGX Spark + Cursor daily workflow, build/test commands, Isaac Sim live steps (Phases 1–2), troubleshooting |
| 2 | [spec.md](../spec.md) | **Full project specification** — goals, phase backlog, acceptance criteria, **Isaac Lab requirement** |
| 3 | [docs/isaac_sim_host_scripts.md](isaac_sim_host_scripts.md) | **Host-only Isaac Sim scripts** — URDF probe, scene build iteration, environment detection, log locations |
| 4 | [commands/test_live.md](../commands/test_live.md) | **Live verification playbook** — prerequisite gate, Phase 1/2 live acceptance sequence |

## One-paragraph summary

Mock Phases 1–4 and container tests are green. **Isaac Lab PPO training** for Phase 2 EE reach runs on the Isaac Sim host via `scripts/host/run_isaac_lab_training.sh train`. After iterative fixes to reward shaping, checkpoint resume, plateau gating, and demo-target fine-tuning, the policy **meets a 95% rolling reach-success target in training** and **97/100 successes in headless demo verification** (single arm, stratified workspace targets, 30 s episodes). Continuous GUI showcase: `./scripts/host/run_isaac_lab_training.sh demo`.

## Architecture reminder

```
┌─────────────────────────────────────────────────────────────┐
│  HOST (DGX Spark) — REQUIRED for Phase 2 training           │
│  Isaac Sim 6.x (~/isaacsim) + Isaac Lab (~/IsaacLab)        │
│  PPO: ./scripts/host/run_isaac_lab_training.sh train        │
│  Demo: ./scripts/host/run_isaac_lab_training.sh demo        │
└──────────────────────────┬──────────────────────────────────┘
                           │ ROS 2 (optional live verification)
┌──────────────────────────▼──────────────────────────────────┐
│  Isaac ROS container (Cursor)                                 │
│  phase1/2 live stacks, colcon test, ROS-bridge smoke tests    │
└───────────────────────────────────────────────────────────────┘
```

## Completed ✅

| Area | Status | Notes |
|------|--------|-------|
| Mock Phase 1–4 | ✅ | `colcon test --packages-select spark_verify_pkg` |
| Isaac Sim scene build | ✅ | `assets/scenes/mycobot_280_m5_limo_cobot.usd` |
| Isaac Lab install / verify | ✅ | `scripts/host/install_isaac_lab.sh`, `verify_isaac_lab.sh` |
| Phase 2 EE reach env | ✅ | Joint-space PPO, curriculum + demo target sampling |
| Demo / play inference | ✅ | `play_ppo.py` — stochastic actions, terminal outcome cache, 30 s episodes |
| Host GUI delegation | ✅ | X11 via `spark_host_exec.sh` for `demo` mode |
| Training success reporting | ✅ | Per-iteration criteria, plateau abort, resume, duration bounds |
| **Phase 2 reach training (2026-07-07)** | ✅ | **97.7% rolling reach** (demo sampling); **97/100 demo verify** |

## Development iterations (2026-07-06 — 2026-07-07)

Chronological summary of changes that led to the current working policy:

### 1. Demo mode and host GUI (Jul 6)
- Fixed X11 forwarding for `nsenter` host delegation (`DISPLAY`, `XAUTHORITY`).
- Added continuous `demo` mode: single arm, red sphere, stratified workspace targets with minimum separation.
- Fixed `rsl_rl` checkpoint load API (`load_cfg`); fixed episode outcome reads before DirectRLEnv auto-reset.

### 2. Reward hacking and numerical blow-up (Jul 6–7)
- **Root cause:** rectified progress reward `max(0, Δd)` let the policy farm reward by oscillating without entering the 25 mm success volume; unbounded actions destabilized PhysX (NaN observations).
- **Fixes:** signed potential-based shaping `(d_prev − d_now) × scale`; action clamp `[-1, 1]`; NaN observation guard; lower `entropy_coef` (0.005); action-magnitude penalty.

### 3. Plateau abort and curriculum interaction (Jul 7)
- Gated plateau abort below 50% best reach so slow-starting runs keep their time budget.
- Reset plateau tracker on curriculum stage change (prevents abort when harder stage lowers rolling success).
- Default plateau window remains 120 iterations; use `--no-plateau-abort` for full duration runs.

### 4. Checkpoint resume (Jul 7)
- **`--resume` / `--no-resume` / auto-resume** in `train_ppo.py`.
- `find_newest_checkpoint()` handles `latest_policy` as a **file**, `logs/model_*.pt` snapshots, and numeric (not lexicographic) sorting.
- Fixed relative checkpoint paths in `play_ppo.py` (resolve against repo root).

### 5. Demo-target fine-tuning (Jul 7)
- **`--target-sampling demo`** trains on the same stratified distribution as the GUI showcase (training on curriculum alone reached 95% rolling but only ~85–90% in demo verify).
- **`--no-early-success-stop`** uses the full `--max-duration-minutes` budget instead of stopping at the first 95% rolling window.
- **`--episode-length-s 30`** for demo/play (reduces near-miss timeouts at 26–45 mm).
- **`reach_bonus` raised to 50** (terminal bonus only — proximity bonuses were tried and reverted after reward-hacking regression).
- Final recipe: resume from curriculum checkpoint → 8-arm demo fine-tune (30 min, no early stop) → 97/100 headless demo successes.

### 6. Play / verify helpers (Jul 7)
- **`--demo-max-episodes N`** for headless demo regression (100-episode verify gate).
- **`--policy-mean`** for mean-action inference (optional; stochastic remained better for this policy).
- Multi-env episode counting in `run_play_loop` when `num_arms > 1`.

## Isaac Lab workflow

| Step | Command |
|------|---------|
| Install (once) | `./scripts/host/install_isaac_lab.sh` |
| Verify | `./scripts/host/run_isaac_lab_training.sh check` |
| Train (default 30 min, 8 arms) | `./scripts/host/run_isaac_lab_training.sh train --headless` |
| Train from scratch | `... train --headless --from-scratch --max-duration-minutes 120 --target-reach-success-rate 0.95` |
| Resume | `... train --headless --resume` (auto-resumes when checkpoint exists) |
| Demo fine-tune | `... train --headless --resume --target-sampling demo --no-early-success-stop --max-duration-minutes 30 --episode-length-s 30` |
| **GUI demo** | **`./scripts/host/run_isaac_lab_training.sh demo`** |
| Headless demo verify | `... play --headless --demo --demo-max-episodes 100 --num-arms 1` |

**Checkpoints:** `assets/checkpoints/isaac_lab_ppo/latest_policy` + `logs/model_*.pt` + `training_summary.json` (gitignored).

## Execution log (2026-07-07)

| Run | Result |
|-----|--------|
| Resume curriculum training (no plateau, 120 min budget) | **95.3% rolling reach**, `stop_reason: task_requirement_met` |
| Demo verify (pre fine-tune) | 85/100 (curriculum policy on demo targets) |
| Demo-target fine-tune (8 arm, 30 min × several passes) | Rolling 95–98% on demo sampling |
| **Final demo verify** | **97/100 reaches (97%)**, 30 s episodes, stochastic inference |
| Proximity bonus experiment | **Reverted** — reward hacking (48% reach, reward >1200) |

**Latest `training_summary.json` snapshot:**

```json
{
  "reach_success_rate": 0.9765625,
  "target_reach_success_rate": 0.98,
  "target_sampling": "demo",
  "stop_reason": "max_duration",
  "task_requirement_met": false
}
```

(`task_requirement_met` is false because the 98% *session* target was not hit before the 30 min cap; rolling reach was 97.7%. Demo verify exceeded the user-facing 95% gate.)

## Suggested resume checklist

1. Host: `isaac-ros activate` → Cursor → `source scripts/source_container_env.sh`
2. `./scripts/host/run_isaac_lab_training.sh check`
3. **Visual demo:** `./scripts/host/run_isaac_lab_training.sh demo`
4. Optional re-verify: `./scripts/host/run_isaac_lab_training.sh play --headless --demo --demo-max-episodes 100 --num-arms 1`
5. Optional ROS live gate: `./scripts/run_live_sim.sh` + `./scripts/run_live_tests.sh`

## Branch and remote

- **Working branch:** `wip_live_testing`
- **Remote:** `https://github.com/jywilson2/spark_isaac_mycobot_demo.git`
