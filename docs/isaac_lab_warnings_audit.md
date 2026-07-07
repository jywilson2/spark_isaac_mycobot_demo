# Isaac Lab / Isaac Sim warnings audit

Last reviewed: **2026-07-06**

This document records each recurring warning seen during PPO training, whether it
indicates a **project-fixable issue**, and what action was taken. Warnings are **not**
globally suppressed in code; only root-cause fixes or documented upstream limitations
apply.

## Summary

| Warning | Meaningful? | Action |
|---------|-------------|--------|
| `setup_conda_env.sh is missing` | No (pre-built binary workflow) | **Fixed** — stub created by install/train scripts |
| `Seed not set for the environment` | Yes (non-deterministic rollouts) | **Fixed** — `seed=42` passed in `make_env_cfg()` |
| `TGS solver … enable_external_forces_every_iteration=False` | Yes (noisy velocities) | **Fixed** — `PhysxCfg.enable_external_forces_every_iteration=True` |
| RSL-RL `obs_groups` empty / missing actor-critic keys | Yes (future RSL-RL breakage) | **Fixed** — explicit `obs_groups` in `rsl_rl_ppo_cfg.py` |
| PyTorch `cuda capability 12.1` vs max `12.0` (GB10) | **Partial** — upstream stack lag | **Documented** — training runs; upgrade PyTorch/Isaac Sim when NVIDIA ships sm_121 build |
| `Enable omni.materialx.libs extension` | No | Upstream Kit hint; no project impact |
| `USD stage is not available` (simulation_manager) | No | Startup ordering in Isaac Sim; resolves after stage open |
| `Last global time was not created` (usdrt) | No | USD notice-handling timing; no project impact |
| gRPC `File already exists in database: grpc/health/v1/health.proto` | No | Duplicate proto registration across Kit extensions |
| `WARNING: All log messages before absl::InitializeLog()` | No | gRPC/absl bootstrap ordering |
| `carb.audio` device misconfigured / null streamer | No | No audio device on DGX Spark host session; simulation unaffected |
| `rtx.hydra` readTransformsFromFabric + geometry streaming | No | Kit rendering default; no project misconfiguration |
| `--headless` CLI deprecated (Isaac Lab develop) | Yes (API drift) | **Fixed** — use `--viz kit` (GUI default) or `--viz none` |

## Details

### Pre-built Isaac Sim: missing `setup_conda_env.sh`

**Message:** `[WARNING] _isaac_sim is present but _isaac_sim/setup_conda_env.sh is missing`

**Assessment:** Not meaningful for the pre-built Isaac Sim 6.x binary at `~/isaacsim`.
Isaac Lab's `isaaclab.sh` expects a full source install layout. The conda script is
only needed to export env vars from a conda-based build.

**Fix:** `scripts/host/install_isaac_lab.sh` and `run_isaac_lab_training.sh` create a
no-op stub at `_isaac_sim/setup_conda_env.sh` when linking the pre-built binary.

### Environment seed

**Message:** `Seed not set for the environment. The environment creation may not be deterministic.`

**Assessment:** Meaningful — without a seed, PPO rollouts and checkpoint comparison are
non-reproducible.

**Fix:** `make_env_cfg(..., seed=args.seed)` with default `42` in `train_ppo.py`.

### PhysX TGS external forces

**Message:** `TGS solver with enable_external_forces_every_iteration=False may cause noisy velocities.`

**Assessment:** Meaningful for articulated robots with implicit actuators — velocity
noise can destabilize joint-target control.

**Fix:** `env_cfg.sim.physics = PhysxCfg()` with
`enable_external_forces_every_iteration=True` in `mycobot_pick_place_env.py`.

### RSL-RL observation groups

**Message:** `obs_groups is empty` / missing `actor`/`critic` keys (deprecated fallback to `policy`)

**Assessment:** Meaningful — RSL-RL 5.x will remove the fallback; actor/critic would
silently mis-wire if observation groups change.

**Fix:** `obs_groups = {"actor": ["policy"], "critic": ["policy"]}` in `rsl_rl_ppo_cfg.py`.

### PyTorch CUDA capability on GB10 (DGX Spark)

**Message:** `Found GPU0 NVIDIA GB10 which is of cuda capability 12.1. Minimum and Maximum cuda capability supported by this version of PyTorch is (8.0) - (12.0)`

**Assessment:** **Partially meaningful.** The bundled Isaac Sim PyTorch build predates
full Blackwell (sm_121) support. Training has been verified to complete, but NVIDIA
may not guarantee optimal kernels until a matching PyTorch/Isaac Sim release.

**Action:** Leave the warning visible. Track Isaac Sim / Isaac Lab release notes for
GB10 support. No in-repo code fix until upstream ships compatible wheels.

### Omniverse Kit startup messages

MaterialX, USD stage deferral, and gRPC proto duplication occur during Kit extension
load before the simulation stage exists. These are upstream timing/registration issues
and do not indicate misconfiguration of this project's env or MDP.

## Policy

1. **Fix in code** when the warning points to incorrect project configuration.
2. **Document** when the warning is upstream or hardware/stack lag (GB10 PyTorch).
3. **Do not** blanket-suppress Python, Kit, or gRPC logs in training scripts.
