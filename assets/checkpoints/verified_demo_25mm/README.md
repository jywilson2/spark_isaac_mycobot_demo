# Verified demo policy @ 25 mm tolerance

Archived weights and metadata for the best **verified** Phase 2 EE reach run (2026-07-08).

| File | Purpose |
|------|---------|
| `policy.pt` | RSL-RL checkpoint (`model_1730.pt` from Phase B demo fine-tune) |
| `training_summary_phase_b.json` | Training metrics at end of two-phase Phase B |
| `demo_verify_results.json` | Headless 20/30/40 s demo regression (**99/100** each) |

```bash
./scripts/host/run_isaac_lab_training.sh play --headless --demo \
  --demo-max-episodes 100 --num-arms 1 \
  --checkpoint assets/checkpoints/verified_demo_25mm/policy.pt
```

Subsequent staged 1 mm precision retrains did **not** beat this checkpoint; precision stages below 8 mm collapsed to 0% reach.
