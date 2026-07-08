# Project Status — Return Briefing

Last updated: **2026-07-08** (branch `wip_live_testing`)

## Summary

Phase 2 now targets **1 mm EE tolerance** with **direct-path reward shaping** and a **4-stage scripted training recipe** (`isaac_lab/training_recipe.py`). Phase 6 red-block code renamed to **Phase 5** (`phase5_red_block/`). Prior 25 mm policy achieved **99/100** demo success; **from-scratch 1 mm staged retrain is in progress**.

## In progress

| Item | Status |
|------|--------|
| Staged from-scratch train (4 stages, ~195 min) | Running on host |
| Demo verify @ 1 mm, 20/30/40 s | Pending training completion |

## Commands

```bash
./scripts/host/run_isaac_lab_training.sh staged --headless
./scripts/host/run_isaac_lab_training.sh monitor --watch
./scripts/host/run_isaac_lab_training.sh verify-demo --headless
```

## Prior verified result (25 mm tolerance, 2026-07-08)

| Stage | Result |
|-------|--------|
| Two-phase train | Phase B **97.7%** rolling reach |
| Demo verify 20/30/40 s | **99/100** each @ 25 mm |
