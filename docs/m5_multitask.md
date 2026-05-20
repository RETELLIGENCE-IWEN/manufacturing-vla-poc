# M5 — Multi-Task Language-Conditioned BC

M5 extends the M4 infrastructure with a multi-task dataset so the instruction
signal *can* carry a meaningful differentiator (Pick vs Push vs Pull). The
M4 `LangBCPolicy` architecture is reused as-is.

## 1. Task selection

Three ManiSkill tasks were chosen because they share the same panda + cube +
goal state schema but require qualitatively different action sequences:

| env_id        | verb | required action      | actor names          |
| --            | --   | --                   | --                   |
| `PickCube-v1` | pick | grasp + lift + place | cube, goal_site      |
| `PushCube-v1` | push | push horizontally    | cube, goal_region    |
| `PullCube-v1` | pull | pull back            | cube, goal_region    |

Per-task expert demos are generated with the corresponding ManiSkill
motion-planning solution (`mani_skill.examples.motionplanning.panda.solutions`).

```bash
for env in PickCube-v1 PushCube-v1 PullCube-v1; do
  python scripts/m5_generate_multitask_expert.py \
    --env-id $env --num-traj 100 --seed 1000 --only-count-success \
    --record-dir outputs/m5_expert_demos_multitask
done
```

Result: 100/100 successful demos per task, mean episode length ≈ 73 steps.

## 2. Unified dataset

The converter merges the three HDF5 files into a single dataset with task
metadata. It normalizes the goal actor by trying `goal_site` then `goal_region`.

```bash
python scripts/m5_convert_multitask_h5_to_dataset.py \
  --record-dir outputs/m5_expert_demos_multitask \
  --config configs/manufacturing_multitask_v0.yaml \
  --out-dir outputs/m5_multitask_dataset
```

Then standard M3 + M4 augmentation:

```bash
python scripts/m3_make_phase_aware_dataset.py \
  --src-dir outputs/m5_multitask_dataset \
  --out-dir outputs/m5_multitask_phase_aware_dataset

python scripts/m4_add_instruction_embeddings.py \
  --in-dir outputs/m5_multitask_phase_aware_dataset \
  --out-dir outputs/m5_multitask_lang_phase_aware_dataset
```

Dataset summary:

```text
num_episodes       : 300 (100 per task)
unique instructions: 128
obs_dim            : 66   (state 57 + progress 1 + prev_action 8)
lang_emb_dim       : 512  (CLIP pooled output, broadcast across timesteps)
splits             : 240 train / 60 val (per-task stratified by m5 converter,
                     re-shuffled by phase-aware converter — see warning below)
```

Per-task instruction templates (excerpt):

```text
pick: "Pick the {object_name} and place it at the {target_name}."
push: "Push the {object_name} across the table to the {target_name}."
pull: "Pull the {object_name} back toward the {target_name}."
```

The full template list is in [configs/manufacturing_multitask_v0.yaml](../configs/manufacturing_multitask_v0.yaml).

## 3. Training

The M4 trainer is reused unchanged.

```bash
python scripts/m4_train_bc_lang.py --config configs/m5_bc_lang_multitask_v0.yaml
```

Result:

```text
best_epoch          : 174 / 500
best_val_mse_norm   : 0.00426
lang_emb_dim        : 512
lang_proj_dim       : 64
```

Compared to M4 (`best_val_mse_norm = 0.00584` on 100-demo PickCube only), the
multi-task dataset gives a clearly lower validation MSE despite the harder
heterogeneous targets — the extra 200 episodes outweigh the added difficulty.

## 4. Closed-loop swap matrix

The key evaluation: does the policy obey the instruction?

For each (env_id, instruction) cell, 30 episodes at seed=3000.

| env       | instruction        | success | grasped | placed | min_dist | mean_return |
| --        | --                 | --      | --      | --     | --       | --          |
| PickCube  | Pick (matched)     | 0.00    | 0.10    | 0.00   | 0.195    | 11.91       |
| PickCube  | Push (swap)        | 0.00    | 0.10    | 0.00   | 0.199    | 10.15       |
| PickCube  | Pull (swap)        | 0.00    | 0.07    | 0.00   | 0.201    | 11.27       |
| PushCube  | **Push (matched)** | **0.10** | 0.00   | 0.00   | 0.191    | 15.13       |
| PushCube  | Pick (swap)        | 0.00    | 0.00    | 0.00   | 0.196    | 15.69       |
| PullCube  | Pull (matched)     | 0.00    | 0.00    | 0.00   | 0.193    | 15.70       |

### Reading the matrix

- **Vertical signal (env changes, instruction fixed)** is strong: PickCube
  rows show ~10% grasp; PushCube/PullCube rows show 0% grasp (no need to lift).
- **Horizontal signal (env fixed, instruction changes)** is essentially zero:
  in PickCube the grasp rate is 10% / 10% / 7% — within noise.

Conclusion: the policy decides its behavior almost entirely from `state_obs`.
The CLIP embedding + 64-dim projection is added but **not used**.

This is the well-known "instruction shortcut" problem in BC: when state alone
sufficiently determines a good action under the training distribution, gradient
descent has no incentive to attend to language.

### What does succeed

PushCube hits 10% closed-loop success — the project's first non-zero
`success_rate_once` after M3.0+ — purely because the motion-planning expert
trajectories for PushCube are simpler (no grasp/lift) and the multi-task data
plus phase-weighted loss happen to produce a working push policy when the
environment really is PushCube.

## 5. Limitations + handoff to M5.1

What M5 confirms:

- ✅ Pipeline (expert generation → unified dataset → CLIP embedding → BC training → multi-env eval) works end to end across three real ManiSkill tasks.
- ✅ Multi-task data improves open-loop accuracy.
- ✅ Some closed-loop success is achievable (PushCube 10%).

What M5 does **not** demonstrate:

- ❌ Genuine language conditioning. Instruction-swap leaves behavior nearly identical.
- ❌ Multi-task PickCube grasp — PickCube grasp rate dropped from M4's 50% to M5's 10%, suggesting capacity contention between the three tasks with no language signal to disambiguate them.

M5.1 will add an auxiliary instruction→task_id classification head to force
the language path to contribute gradient. Once the policy is provably using
the instruction (matched-instruction success rises and swap-instruction
success drops correspondingly), M6 (vision) becomes meaningful — until then,
visual conditioning would inherit the same shortcut problem.

## 6. Notes on splits

The per-task stratified split produced by `m5_convert_multitask_h5_to_dataset.py`
is overwritten by `m3_make_phase_aware_dataset.py`, which re-shuffles globally.
The downstream val split (PickCube 23 / PushCube 13 / PullCube 24) is therefore
slightly imbalanced. M5.1 may want to plumb stratified splits through
`m3_make_phase_aware_dataset.py` or write a small post-fix step.
