# Project Status — Return Briefing

Last updated: **2026-07-09** (branch `wip_live_testing`, rebased on `main`)

This document is the **first file to read** after a long break. For daily commands see [README.md](../README.md); for host Isaac Sim scripts see [isaac_sim_host_scripts.md](isaac_sim_host_scripts.md).

## One-paragraph summary

Mock Phases 1–4 and container tests are green. **Phase 2 EE reach PPO** trains on the Isaac Sim host via Isaac Lab. The **best verified policy** achieves **99/100** headless demo success at **25 mm** tolerance across 20/30/40 s horizons (archived under `assets/checkpoints/verified_demo_25mm/`). A **7-stage precision recipe** (25 mm → 1 mm) was added but **sub-8 mm stages failed** (0% reach); the Jul 9 staged v5 run **hung on Stage 2** (Isaac Sim reload) and was **killed without restart**. Development is paused at a known-good 25 mm checkpoint.

---

## New to this project? Read in order

| Order | Document | Why |
|-------|----------|-----|
| 1 | [README.md](../README.md) | Daily workflow, build/test, training commands |
| 2 | [spec.md](../spec.md) | Requirements, Phase 2 reach, precision tiers |
| 3 | [isaac_sim_host_scripts.md](isaac_sim_host_scripts.md) | Host-only URDF probe, scene build, logs |
| 4 | [commands/test_live.md](../commands/test_live.md) | Live ROS Phase 1/2 gate (optional) |
| 5 | This file | Current state, verified results, known blockers |

---

## Architecture (two processes)

```
┌──────────────────────────────────────────────────────────────┐
│  HOST (DGX Spark) — REQUIRED for Phase 2 training            │
│  Isaac Sim 6.x (~/isaacsim) + Isaac Lab (~/IsaacLab)         │
│  PPO: ./scripts/host/run_isaac_lab_training.sh …             │
└──────────────────────────┬───────────────────────────────────┘
                           │ ROS 2 (optional live verification)
┌──────────────────────────▼───────────────────────────────────┐
│  Isaac ROS container (Cursor)                                 │
│  colcon test, ROS-bridge smoke tests                          │
└───────────────────────────────────────────────────────────────┘
```

Training **never runs inside Docker** — the container delegates to the host via `nsenter`.

---

## Verified result (USE THIS CHECKPOINT)

| Metric | Value |
|--------|-------|
| **Demo verify** | **99/100** @ 20 s, 30 s, and 40 s episode lengths |
| **Reach tolerance** | **25 mm** (`EE_REACH_TOLERANCE_COARSE_M`) |
| **Two-phase Phase B training reach** | **97.7%** rolling |
| **Archived policy** | `assets/checkpoints/verified_demo_25mm/policy.pt` |
| **Metadata** | `demo_verify_results.json`, `training_summary_phase_b.json` |
| **Git commit** | `d92a3cc` — "Verify two-phase training: 99/100 demo success" |

```bash
# Replay verified policy (headless, 100 episodes)
./scripts/host/run_isaac_lab_training.sh play --headless --demo \
  --demo-max-episodes 100 --num-arms 1 \
  --checkpoint assets/checkpoints/verified_demo_25mm/policy.pt

# GUI showcase (single arm, respawning targets)
./scripts/host/run_isaac_lab_training.sh demo \
  --checkpoint assets/checkpoints/verified_demo_25mm/policy.pt
```

---

## Training recipes (current code)

| Recipe | Script | Purpose |
|--------|--------|---------|
| **Two-phase** (verified) | `run_two_phase_training.sh` | 90 min curriculum → 30 min demo @ 25 mm |
| **Staged precision** (experimental) | `run_staged_training.py` | 7 stages: 25→20→15→12→6→3→1 mm (~6.5 h) |
| **Monitor** | `scripts/monitor_training.sh` | Tail logs + `training_summary.json` |

Staged recipe definition: `isaac_lab/training_recipe.py`.

---

## What happened on 2026-07-09 (paused state)

| Run | Outcome |
|-----|---------|
| **staged v4** (7 stages, ~6 h) | Stages 1–4 OK @ 25–15 mm; **8/3/1 mm → 0%** reach |
| **staged v5** (restart from scratch) | Stage 1: **90.6%** @ 25 mm (90 min); **Stage 2 hung** ~6 h at Isaac Sim reload (0% CPU, empty log) |
| **Action taken** | Hung processes **killed**; training **not restarted** |

**Known blocker:** Isaac Sim **stage-transition hang** between staged phases (reload after Stage 1). Workaround for resume: run stages individually or restart from `--start-stage N` after confirming Isaac Sim is healthy.

---

## Completed ✅

| Area | Status |
|------|--------|
| Mock Phase 1–4 | ✅ `colcon test --packages-select spark_verify_pkg` |
| Isaac Sim scene | ✅ `assets/scenes/mycobot_280_m5_limo_cobot.usd` |
| Isaac Lab install/verify | ✅ `scripts/host/install_isaac_lab.sh`, `verify_isaac_lab.sh` |
| Phase 2 reach env | ✅ Joint-space PPO, curriculum + demo + precision sampling |
| Smooth motion | ✅ EMA smoothing, jerk penalty, 30 s aligned horizons |
| Two-phase training | ✅ Verified 99/100 demo @ 25 mm |
| Resume checkpoints | ✅ `--resume`, numeric checkpoint sort |
| Plateau / curriculum fixes | ✅ Stage-aware plateau reset |
| **Verified checkpoint in git** | ✅ `assets/checkpoints/verified_demo_25mm/` |

## Not working yet ⚠️

| Item | Notes |
|------|-------|
| Sub-8 mm precision training | Policy collapses to 0% when tolerance tightens abruptly |
| Staged pipeline reliability | Stage 2+ Isaac Sim reload can hang indefinitely |
| 1 mm spec target | Requires recipe/training fixes before re-attempt |

---

## Resume checklist (after a long hiatus)

### 1. Environment (every session)

```bash
# Host
isaac-ros activate

# Cursor terminal
whoami          # → admin (uid 1000)
source /workspaces/isaac_ros-dev/src/spark_isaac_mycobot_demo/scripts/source_container_env.sh
```

Host `~/.bashrc` should export `ROS_DOMAIN_ID=42`, `FASTDDS_BUILTIN_TRANSPORTS=UDPv4`, `ISAACSIM_PATH`, `ISAAC_ROS_WS`.

### 2. Sanity checks

```bash
./scripts/host/run_isaac_lab_training.sh check
./scripts/host/verify_isaac_lab.sh
cd /workspaces/isaac_ros-dev && source /opt/ros/jazzy/setup.bash && \
  colcon test --packages-select spark_verify_pkg && colcon test-result --all
pytest isaac_lab/test/ -q
```

### 3. Confirm verified policy still works

```bash
./scripts/host/run_isaac_lab_training.sh play --headless --demo \
  --demo-max-episodes 20 --num-arms 1 \
  --checkpoint assets/checkpoints/verified_demo_25mm/policy.pt
```

Expect ≥ 95% reach (historically 99/100 @ 100 episodes).

### 4. Before retraining

- Do **not** use `--from-scratch` unless intentional — it deletes `assets/checkpoints/isaac_lab_ppo/`.
- Prefer **two-phase** for 25 mm targets; treat **staged 1 mm** as experimental.
- If resuming staged training: `./scripts/host/run_isaac_lab_training.sh staged --headless --skip-verify --start-stage 1` (index 1 = demo coarse) **only after** killing any stuck Isaac Sim processes.
- Monitor: `./scripts/host/run_isaac_lab_training.sh monitor --watch` or `tail -f /tmp/spark_staged_train_v*.log`.

### 5. Optional ROS live gate

```bash
# Host: ./scripts/run_live_sim.sh
# Container: ./scripts/run_live_tests.sh
```

---

## Key paths

| Path | Purpose |
|------|---------|
| `assets/checkpoints/verified_demo_25mm/` | **Best verified policy + metadata (in git)** |
| `assets/checkpoints/isaac_lab_ppo/` | Active training dir (gitignored, overwritten each run) |
| `isaac_lab/training_recipe.py` | Staged precision recipe |
| `scripts/run_two_phase_training.sh` | Verified two-phase recipe |
| `scripts/verify_demo_policy.sh` | Multi-horizon demo regression |
| `/tmp/spark_staged_train_v*.log` | Host training tee logs |

---

## Branch and remote

- **Working branch:** `wip_live_testing` (rebased on `main`)
- **Remote:** `https://github.com/jywilson2/spark_isaac_mycobot_demo.git`

---

## Suggested next engineering steps

1. Fix Isaac Sim **stage-transition hang** (timeout + kill/restart between stages).
2. Add **gradual tolerance ramp** or longer budgets before 8 mm precision.
3. Re-run **two-phase** only if 25 mm policy regresses; otherwise iterate precision from `verified_demo_25mm/policy.pt` with `--resume`.
4. Run `verify-demo` after any training change before archiving new checkpoints.
