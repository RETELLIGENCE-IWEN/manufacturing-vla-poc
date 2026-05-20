# M5.1 — Auxiliary `task_id` classification loss

The M5 multi-task policy ignored the instruction (matched vs swapped
instructions in the PickCube environment produced essentially identical
behavior). M5.1 forces the language path to carry useful signal by
training it to discriminate the task.

## 1. Why this fixes the M5 shortcut

In M5 the policy was free to predict every action from `state_obs` alone
because state already disambiguates the task (PushCube and PullCube don't
even have a graspable cube goal; PickCube has a goal_site in the air, etc.).
Under that information geometry, gradient descent has no reason to attend
to `lang_proj(lang_emb)` — its scale is small (64 dims vs 66 of state) and
the BC loss is satisfied without it.

The fix is to add a second objective that forces `lang_proj` to be
informative on its own:

```text
task_logits  = task_head(lang_proj(lang_emb))           ; shape (B, num_tasks)
aux_loss     = CrossEntropy(task_logits, task_id)
total_loss   = BC_weighted_MSE(action) + aux_weight * aux_loss
```

After training, `lang_proj` activations are linearly separable by task,
so the concatenated MLP input changes meaningfully when the instruction
changes — the BC head cannot ignore it anymore.

## 2. Implementation

[scripts/m5_1_train_bc_lang_aux.py](../scripts/m5_1_train_bc_lang_aux.py)
forks `m4_train_bc_lang.py` and adds:

- `PhaseWeightedLangAuxDataset` returns `(obs, lang_emb, action, sample_weight, task_int)`.
  `task_int` is loaded from `episodes.jsonl` (`task_id` field) and mapped to
  a 0-based integer per the order tasks appear.
- `LangBCPolicyAux` adds `self.task_head = Linear(lang_proj_dim, num_tasks)`
  on top of `lang_proj`. The BC head is unchanged.
- `forward_with_aux(obs, lang_emb) → (action, task_logits)` returns both;
  `forward(obs, lang_emb) → action` keeps the inference contract identical
  to M4/M5 so the eval scripts only need to swap the model class.
- Joint loss `total = bc + aux_weight * ce`.

[scripts/m5_1_eval_closedloop_bc_lang_aux.py](../scripts/m5_1_eval_closedloop_bc_lang_aux.py)
forks `m5_eval_closedloop_bc_lang.py` and only replaces `LangBCPolicy` with
the M5.1 variant (which has `task_head`). Inference path is unchanged.

[configs/m5_1_bc_lang_multitask_aux_v0.yaml](../configs/m5_1_bc_lang_multitask_aux_v0.yaml):

```yaml
aux:
  aux_weight: 1.0
```

Dataset is reused as-is: `outputs/m5_multitask_lang_phase_aware_dataset/`
already carries `task_id` strings in `episodes.jsonl`.

## 3. Training

```bash
python scripts/m5_1_train_bc_lang_aux.py --config configs/m5_1_bc_lang_multitask_aux_v0.yaml
```

```text
best_epoch          : 305 / 500
best_val_mse_norm   : 0.00384
best_val_task_acc   : 1.000
aux_weight          : 1.0
num_tasks           : 3
task_id_strings     : [pick_place, push_horizontal, pull_horizontal]
```

`val_task_acc = 1.0` proves `lang_proj` activations are now perfectly
task-discriminative. `best_val_mse_norm = 0.00384` is also lower than M5
(0.00426) and M4 (0.00584), so the aux loss is not hurting BC accuracy.

## 4. Closed-loop swap matrix

30 episodes per cell, seed=3000, phase_horizon=80, max_steps=120, safe-action filter
on (multi-task `action_bounds.json`).

| env       | instruction         | success | grasped | placed | min_dist | mean_return |
| --        | --                  | --      | --      | --     | --       | --          |
| PickCube  | Pick (matched)      | 0.00    | 0.07    | **0.03** | 0.192    | 11.69       |
| PickCube  | Push (swap)         | 0.00    | **0.00** | 0.00 | 0.207    | 9.19        |
| PickCube  | Pull (swap)         | 0.00    | **0.00** | 0.00 | 0.203    | 10.11       |
| PushCube  | Push (matched)      | **0.20** | 0.00    | 0.00   | 0.173    | 16.52       |
| PushCube  | Pick (swap)         | 0.00    | 0.00    | 0.00   | 0.201    | 19.76       |
| PullCube  | Pull (matched)      | **0.33** | 0.00    | 0.00   | 0.163    | 17.80       |

### Side-by-side with M5 (no aux loss)

| Metric                | M5     | M5.1   | Δ                                      |
| --                    | --     | --     | --                                     |
| PickCube grasp matched | 0.10   | 0.07   | similar                                |
| PickCube grasp swap_push| 0.10   | **0.00** | **dropped to 0** — instruction obeyed |
| PickCube grasp swap_pull| 0.07   | **0.00** | **dropped to 0** — instruction obeyed |
| PickCube placed matched | 0.00   | **0.03** | first non-zero PickCube placement     |
| PushCube success match  | 0.10   | **0.20** | doubled                                |
| PullCube success match  | 0.00   | **0.33** | first non-zero PullCube success        |

### Reading the results

The crucial column is "PickCube + swap instruction":

- M5 ignored the instruction (10% grasp regardless of Pick/Push/Pull text).
- M5.1 obeys it: when the user says "Push the cube" in a PickCube env,
  the policy stops grasping (0%).

This is the textbook "instruction-following" success signal — and it
generalizes:

- In the matching env, the success rate **rises** for two of three tasks
  (PushCube ×2, PullCube from zero to 33%, PickCube gains its first
  non-zero placement).
- In the mismatched env (PushCube + Pick instruction etc.), success
  stays at 0% but the policy still produces non-degenerate state
  trajectories (mean_return 15-19 vs random ~0.4).

## 5. Limitations + handoff to M6

- The dataset is still pure state (no vision). M6 will swap or augment
  `state_obs` with rendered RGB and reuse the CLIP vision tower; that is
  the genuine VLA setting.
- The current task vocabulary is 3 verbs (pick / push / pull). Adding
  multiple targets within each verb (e.g., "Push to the **left** goal"
  vs "Push to the **right** goal") would test finer-grained instruction
  following — a useful intermediate step before M6.
- Aux weight is fixed at 1.0 with no sweep. A sweep could find a sweet
  spot that lowers BC loss further without sacrificing instruction
  discrimination.

What M6 inherits from M5.1:

- Working `task_id` slot in `episodes.jsonl` and the load helper in
  `m5_1_train_bc_lang_aux.py`.
- `LangBCPolicyAux` architecture (BC head + auxiliary classification head).
- The swap-matrix evaluation pattern, which is the right diagnostic for
  any language-conditioned policy.
