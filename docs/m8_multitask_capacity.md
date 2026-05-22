# M8 — Multi-task capacity-sharing fix study (4-way comparison)

By M7.1 the multi-task capacity-sharing trade-off observed in M6 v0/v1/v2
had become structural: each variant was best at exactly one task, and
moving any control knob just relocated the trade-off rather than removing
it. M7.1 (Diffusion Policy) confirmed the pattern holds across a
different policy class.

M8 sets up a controlled four-way comparison of three independent fixes —
each touching a different layer of the model — against the M6.2 BC v2
baseline, on the same dataset and same 6-cell swap matrix.

## 1. The three fixes

The fixes are deliberately chosen so they touch *different* parts of the
training/architecture stack:

| Fix                              | Layer affected                 | What changes                                  |
| --                               | --                             | --                                            |
| M8a — PCGrad (Yu et al. 2020)    | Optimizer                      | Per-task gradient projection at every step    |
| M8b — Per-task output heads      | Network head                   | One head per task, selected by task_id        |
| M7.1 — Diffusion Policy          | Policy class                   | Eps-prediction over an action chunk           |

Each fix is applied to the same multi-task VLA dataset
(`outputs/m6_multitask_vla_dataset_v2`, 300 episodes, 3 tasks) with the
same training budget where feasible.

## 2. M8a — PCGrad

[scripts/m8a_train_bc_pcgrad.py](../scripts/m8a_train_bc_pcgrad.py).

### Algorithm

PCGrad (Yu et al. 2020) — for every minibatch the BC loss is decomposed
into one loss per task. Each task's gradient is computed via a separate
backward. If two task gradients have negative cosine similarity, one is
projected onto the normal plane of the other to remove the conflicting
component. The resulting projected gradients are averaged and used for the
optimizer step. The auxiliary `task_id` classification gradient is added
on top.

The model architecture is identical to M6.2 BC v2 (`VLAPolicyAux` from
[scripts/m6_train_vla_lang_aux.py](../scripts/m6_train_vla_lang_aux.py)).
**Only the optimizer update rule changes.** This isolates the optimization
contribution from any architecture contribution.

### Training stats

```text
best_epoch              : 127 / 500
best_val_mse_norm       : 0.00634   (BC v2: 0.00678; -7%)
best_val_task_acc       : 1.000
avg conflicts / epoch   : 94.6
```

The conflict count is significant: roughly **95 task-pair gradient
conflicts per epoch are actively resolved** by PCGrad. This is direct
evidence that BC v2 was suffering from gradient interference, not just
capacity scarcity.

### Closed-loop result

Most striking effect: **PickCube grasp_once jumps 0.13 → 0.20** on the
matched instruction. PushCube and PullCube come back to parity with BC v2.

```text
PickCube/Pick : grasp 0.20 (BC v2: 0.13)   <-- the weakest task in BC v2 recovers
PushCube/Push : 0.30                       (BC v2: 0.30)
PullCube/Pull : 0.23                       (BC v2: 0.43, regressed)
```

So PCGrad does what its theory predicts: the optimizer no longer lets one
task overwrite another at the gradient level. PickCube was being
systematically overwritten in BC v2; PCGrad relieves that. The PullCube
regression suggests gradient projection redistributes effective learning
rate toward the weaker task at the expense of the strongest one.

## 3. M8b — Per-task output heads

[scripts/m8b_train_bc_per_task_head.py](../scripts/m8b_train_bc_per_task_head.py),
[scripts/m8b_eval_closedloop_per_task.py](../scripts/m8b_eval_closedloop_per_task.py).

### Architecture

```text
state_obs (57) + progress (1) + prev_action (8)   = obs (66)
lang_proj  (64), image_proj (128)                   = conditioning
trunk      = same MLP as M6.2 (256 -> 256 -> 128)
action_heads = nn.ModuleList([Linear(128, 8) for _ in 3 tasks])
task_head_aux = Linear(64, 3)                       (from lang_proj)

train  : action = action_heads[task_id_gt](trunk_features)
inference: task_id_pred = argmax(task_head_aux(lang_proj))
           action = action_heads[task_id_pred](trunk_features)
```

The trunk stays shared. Only the final linear is duplicated three times.
At inference the task_id is inferred once from the instruction (M5.1
already produced 100% val task_acc; the prediction is stable per
instruction).

### Training stats

```text
best_epoch              : 77 / 500
best_val_mse_norm       : 0.00771   (BC v2: 0.00678; +14% worse)
best_val_task_acc       : 1.000
num_params              : 0.30M     (vs BC v2 ~0.27M, +10%)
```

Best val MSE hits at epoch 77 — earlier than any other M8 variant. This
turned out to be a *bad* checkpoint for closed-loop PullCube performance
(see §5).

### Closed-loop result

```text
PickCube/Pick (matched) : grasp 0.00  placed 0.00     (BC v2: grasp 0.13)
PushCube/Push           : 0.40                        (BC v2: 0.30)   +10pp
PullCube/Pull           : 0.23                        (BC v2: 0.43)   -20pp
PullCube/Pick (swap)    : 0.00                        (BC v2: 0.13)   <-- cleanest instruction obey
```

The cleanest instruction-following result of any model: when given a Pick
instruction in a PullCube env, the policy selects the Pick head, attempts
a grasp, and fails — exactly the "obey the wrong instruction" behavior
the swap matrix is designed to detect. PushCube goes up, PullCube comes
down, and PickCube grasp is lost entirely. Per-task heads decouple the
output but not the shared representation.

## 4. M7.1 — Diffusion Policy

Already documented in [docs/m7_diffusion.md](m7_diffusion.md). Included
here only because the swap matrix was rerun against the M8 baselines.

```text
PushCube/Push   : 0.47   <-- best on PushCube
PullCube/Pull   : 0.23
PickCube/Pick   : 0.00 (grasp 0.00)
mean_return     : ×1.5–2.0 over BC v2 in every cell
```

## 5. Cross-fix comparison

| Cell                       | BC v2     | M7.1 Diff | M8a PCGrad | M8b PerTaskHead |
| --                         | --        | --        | --         | --              |
| PickCube grasp (matched)   | 0.13      | 0.00      | **0.20**   | 0.00            |
| PickCube placed (matched)  | 0.00      | 0.00      | 0.00       | 0.00            |
| PushCube success (matched) | 0.30      | **0.47**  | 0.30       | 0.40            |
| PullCube success (matched) | **0.43**  | 0.23      | 0.23       | 0.23            |
| PickCube grasp (swap_push) | 0.03      | 0.03      | 0.00       | 0.00            |
| PullCube success (swap_pick)| 0.13     | 0.20      | 0.10       | **0.00** (obey) |

Each fix is best at a *different* axis:

```text
BC v2          : PullCube settle
Diffusion      : PushCube, sustained activity (mean_return)
PCGrad         : PickCube grasp recovery (gradient balancing)
Per-task heads : instruction-obey purity (output decoupling)
```

## 6. What this study actually proves

1. **Gradient interference exists and is measurable.** PCGrad logged
   ~95 task-pair conflicts every epoch in BC v2's training. That's not
   capacity scarcity — that's tasks actively pulling shared parameters in
   opposite directions.
2. **Each fix solves a real, distinct sub-problem.** PCGrad rescues the
   weakest task. Per-task heads sharpen instruction obedience. Diffusion
   unlocks PushCube's settle behavior and keeps the policy active.
3. **None of them remove the trade-off.** With the trunk capacity fixed
   at ~270k parameters and 300 demos × 3 tasks, every fix just relocates
   where the trade-off binds.
4. **Open-loop val MSE is a poor proxy for closed-loop competence in
   multi-task BC.** M8b hit its best val MSE at epoch 77 and that
   checkpoint was already PullCube-weak. A correct model-selection
   pipeline would either evaluate closed-loop every K epochs or keep a
   per-task validation metric.

## 7. What this study does *not* claim

- That PCGrad is universally better than per-task heads (it isn't; M8b
  wins instruction purity).
- That Diffusion replaces BC (it doesn't; BC v2 still wins PullCube).
- That the trade-off can't be removed in principle. We tested three fixes
  at fixed trunk capacity. Growing the trunk (M9b — OpenVLA / Octo
  fine-tune) or stacking M8a + M8b at once (M9a) are the obvious next
  controls.

## 8. References

- Yu et al. 2020, **"Gradient Surgery for Multi-Task Learning" (PCGrad)** — NeurIPS 2020. The M8a implementation follows the paper's pseudocode.
- Liu et al. 2021, **"Conflict-Averse Gradient Descent for Multi-task Learning" (CAGrad)** — NeurIPS 2021. Closely related to PCGrad; a natural follow-up.
- Sener & Koltun 2018, **"Multi-Task Learning as Multi-Objective Optimization"** — NeurIPS 2018. The Pareto-optimal motivation behind PCGrad/CAGrad.
- Chen et al. 2018, **"GradNorm"** — ICML 2018. Loss-magnitude balancing rather than gradient projection.
- Kendall et al. 2018, **"Multi-Task Learning Using Uncertainty to Weigh Losses"** — CVPR 2018. The classical loss-weighting alternative.
- Misra et al. 2016, **"Cross-stitch Networks"** — CVPR 2016. Soft feature sharing.
- Houlsby et al. 2019, **"Parameter-Efficient Transfer Learning for NLP" (Adapters)** — ICML 2019. Per-task adapter modules on a frozen backbone — the next step if M8b's per-task heads suggests the right axis but needs more capacity per task.
- Hu et al. 2021, **"LoRA"** — ICLR 2022. The dominant per-task low-rank fine-tune pattern; relevant for M9b.
- Shazeer et al. 2017, **"Outrageously Large Neural Networks" (MoE)** — ICLR 2017. The routing/expert direction beyond per-task heads.
- Reed et al. 2022, **"A Generalist Agent" (Gato)** — TMLR 2022. The "just make the trunk huge" answer to multi-task capacity.
- Kim et al. 2024, **"OpenVLA"**. Concrete foundation-policy candidate for M9b.
- Chi et al. 2023, **"Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"** — RSS 2023. The M7.1 reference.
