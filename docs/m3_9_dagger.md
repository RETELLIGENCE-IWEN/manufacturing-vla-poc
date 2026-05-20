# M3.9 DAgger Pipeline

This milestone adds policy-visited state correction data to the phase-aware BC
pipeline.

## 1. Smoke Test

Check both state restore parity and whether the PickCube motion-planning solver
can emit a first action from a restored policy-visited state.

```bash
python scripts/m3_dagger_smoke.py \
  --model runs/m3_bc_phase_aware/best_model.pt \
  --normalization runs/m3_bc_phase_aware/normalization_stats.npz \
  --expert-action-bounds outputs/m3_phase_aware_dataset_100/action_bounds.json \
  --seed 4000 \
  --rollout-steps 40 \
  --phase-horizon 80 \
  --out-dir outputs/m3_dagger_smoke_v0
```

The smoke test defaults to `--expert-mode continuation`, which mirrors the
PickCube planner sequence without the packaged solution's initial reset. To
diagnose the packaged solution directly, use `--expert-mode stock_solve`.

Important fields:

- `restore_smoke.state_obs_restore_max_abs_diff`
- `planner_restore_smoke.first_pre_step_state_obs_max_abs_diff`
- `planner_restore_smoke.can_use_first_action_as_correction`

If the planner starts from a different state, try `--solver-call-mode no_seed`,
`--solver-call-mode seed_none`, or `--solver-call-mode seed_value` explicitly.

## 2. Collect Policy Rollouts

```bash
python scripts/m3_collect_dagger_rollouts.py \
  --model runs/m3_bc_phase_aware/best_model.pt \
  --normalization runs/m3_bc_phase_aware/normalization_stats.npz \
  --expert-action-bounds outputs/m3_phase_aware_dataset_100/action_bounds.json \
  --num-episodes 50 \
  --seed 4000 \
  --max-steps 120 \
  --phase-horizon 80 \
  --selection-mode focused \
  --near-goal-dist 0.12 \
  --max-selected-per-episode 40 \
  --out-dir outputs/m3_dagger_rollouts_v0
```

Outputs:

- `outputs/m3_dagger_rollouts_v0/rollouts/*.npz`
- `outputs/m3_dagger_rollouts_v0/state_index.jsonl`
- `outputs/m3_dagger_rollouts_v0/summary.json`

## 3. Planner Relabel

```bash
python scripts/m3_relabel_dagger_with_planner.py \
  --rollout-dir outputs/m3_dagger_rollouts_v0 \
  --out-dir outputs/m3_dagger_corrections_v0 \
  --expert-mode continuation \
  --dagger-weight 2.0 \
  --pre-step-tolerance 1e-4 \
  --restore-tolerance 1e-4
```

For a small first run:

```bash
python scripts/m3_relabel_dagger_with_planner.py \
  --rollout-dir outputs/m3_dagger_rollouts_v0 \
  --out-dir outputs/m3_dagger_corrections_smoke_v0 \
  --max-states 20
```

Outputs:

- `outputs/m3_dagger_corrections_v0/episodes/*.npz`
- `outputs/m3_dagger_corrections_v0/relabel_failures.jsonl`
- `outputs/m3_dagger_corrections_v0/summary.json`

## 4. Aggregate Dataset

The default rebuilds expert progress with the same fixed `phase_horizon` used
at closed-loop inference.

```bash
python scripts/m3_build_dagger_aggregate_dataset.py \
  --expert-dataset-dir outputs/m3_phase_aware_dataset_100 \
  --dagger-dataset-dir outputs/m3_dagger_corrections_v0 \
  --out-dir outputs/m3_agg_phase_dagger_dataset_v0 \
  --expert-progress-mode phase_horizon \
  --phase-horizon 80 \
  --expert-weight 1.0
```

## 5. Train

```bash
python scripts/m3_train_bc_phase_weighted.py \
  --config configs/m3_bc_dagger_v0.yaml
```

The trainer now reads optional per-sample `sample_weight` from each episode and
multiplies it with the existing progress-based phase weight.

## 6. Evaluate

```bash
python scripts/m3_eval_closedloop_bc_phase_safe.py \
  --model runs/m3_bc_dagger_v0/best_model.pt \
  --normalization runs/m3_bc_dagger_v0/normalization_stats.npz \
  --expert-action-bounds outputs/m3_agg_phase_dagger_dataset_v0/action_bounds.json \
  --num-episodes 30 \
  --seed 3000 \
  --max-steps 120 \
  --phase-horizon 80 \
  --out-dir runs/m3_bc_dagger_v0/closedloop_eval_safe
```
