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

---

## 7. v0 → v1 → v2 Experiment Summary

DAgger was iterated twice on top of the phase-aware BC policy.

### Closed-loop comparison (PickCube-v1, 30 episodes, seed=3000, phase_horizon=80, max_steps=120)

| Policy                          | grasped_once | final_grasped | placed_once | mean_return | mean_final_dist | mean_min_dist |
| --                              | --           | --            | --          | --          | --              | --            |
| phase_aware BC (baseline)       | 0.567        | 0.000         | 0.033       | 14.56       | 0.289           | 0.165         |
| DAgger v1                       | 0.467        | 0.167         | 0.000       | 20.96       | 0.197           | 0.182         |
| DAgger v1 + force_grip          | **0.467**    | **0.367**     | 0.000       | **24.10**   | 0.206           | 0.190         |
| DAgger v2 (release-bias)        | 0.333        | 0.000         | 0.000       | 16.08       | 0.208           | 0.197         |
| DAgger v2 + force_grip          | 0.300        | 0.167         | 0.000       | 18.57       | 0.209           | 0.193         |

**Chosen BC base: DAgger v1 + force_grip.**

### v1 vs v2 collector configs

```text
v1 selection: near_goal + is_grasped + placed_once_failed_late + deteriorated_after_min_dist + high_action_norm
v1 dataset:   1553 selected → 1121 successful corrections (72.2%), 3 correction episodes, sample_weight=2.0

v2 selection: near_goal + is_grasped + late_grasp_far_from_goal + high_action_norm
              (deteriorated disabled, late_grasp_min_progress=0.6, late_grasp_min_dist=0.05)
v2 dataset:   899 selected → 830 successful corrections (92.3%), 2 correction episodes, sample_weight=2.0
```

### Why v2 regressed

Debug trajectories (seed=3021 and seed=3028) showed v2 fails to approach the cube at all
(grasp_count=0 across the entire episode), whereas v1 grasps and partially transfers
(seed=3021: min_cube_goal_dist=0.061 with v1 vs 0.230 with v2).

The dominant hypothesis: **`deteriorated_after_min_dist` was unintentionally
providing an "actively re-approach the cube after losing it" signal**, not just retreat data.
Removing it caused the policy to drift away from cube-approach behavior. The 240
`late_grasp_far_from_goal` correction states were not enough to compensate, possibly because
they were concentrated in 2 correction episodes (vs 3 for v1) which amplified pattern overfitting.

### Why force_grip helps

Step logs on v1 showed the dominant failure mode: at progress≈0.95 the policy opens
the gripper regardless of cube position, dropping the cube even when far from the goal.
The `force_grip_while_far` inference heuristic (in
[scripts/m3_eval_closedloop_bc_phase_safe.py](../scripts/m3_eval_closedloop_bc_phase_safe.py))
clamps `action[-1] = -1.0` once the policy has ever grasped the cube and the cube is still
beyond the threshold from the goal. This roughly doubles `final_grasped_rate` (0.167 → 0.367).

Attempts to fix release timing without the heuristic (training-time DAgger v2; inference-time
phase_horizon adjustments h100/h120) both regressed: phase_horizon changes caused input
distribution shift that degraded grasp itself.

### Conclusion

Single-task BC + DAgger on PickCube-v1 plateaus at this level. The remaining gap to
closed-loop success is unlikely to close with more BC iterations; downstream milestones
should pivot toward language conditioning (M4) and multi-task generalization (M5+),
which is the project's actual goal.
