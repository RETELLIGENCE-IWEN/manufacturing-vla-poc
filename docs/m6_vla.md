# M6 — VLA: state + CLIP-text + CLIP-vision

M6 closes the V in VLA. The M5.1 multi-task policy already used the
instruction (CLIP text); M6 adds the matching frozen CLIP visual tower
so the policy also sees the world.

## 1. Architecture

```text
state_obs   (panda 31 + cube 13 + goal 13)  =  57
progress    1
prev_action 8
                                              -----
obs                                            66

lang_emb    512  (frozen CLIPText.pooler_output(instruction))
image_emb   768  (frozen CLIPVision.pooler_output(env.render()))

lang_proj   64   = ReLU(Linear(512 -> 64))   (learned)
image_proj  128  = ReLU(Linear(768 -> 128))  (learned)

policy_input = concat(obs, lang_proj, image_proj)   shape 258
MLP backbone : same hidden dims as M5.1 [256, 256, 128] with dropout 0.05
output       : action (8) = pd_joint_pos

task_head    : Linear(64 -> 3)   on top of lang_proj   (auxiliary)
```

The auxiliary task_id head sits on `lang_proj` only (not on
`image_proj`), preserving the M5.1 pressure on the language path. This
matters: putting an auxiliary head on the image path as well would let
the image path satisfy the auxiliary objective alone, and we would lose
the M5.1 instruction signal.

## 2. Image embedding extraction

The M5 expert HDF5 files store `episode_seed` and the full `actions`
sequence. M6 replays each expert trajectory in an `rgb_array` env at
224×224, renders the camera at every step, and batches the resulting
images through `CLIPVisionModel.pooler_output`.

```bash
python scripts/m6_add_image_embeddings.py \
  --in-dir outputs/m5_multitask_lang_phase_aware_dataset \
  --out-dir outputs/m6_multitask_vla_dataset \
  --render-resolution 224
```

Each `episodes/ep_NNNNNN.npz` gains an `image_emb` array of shape
`(T, 768)`. Consecutive-step embedding differences (mean ≈ 1.87 in L2
norm, max ≈ 5.1) confirm the image actually changes step-to-step — the
embedding is not a constant context.

Storage cost: 768 × 4B × ~70 steps × 300 episodes ≈ 65 MB. Cheap.

## 3. Training

[scripts/m6_train_vla_lang_aux.py](../scripts/m6_train_vla_lang_aux.py)
forks M5.1's trainer and adds:

- `VLADataset.__getitem__` returns `(obs, lang_emb, image_emb, action, sample_weight, task_int)`.
- `VLAPolicyAux` holds `lang_proj`, `image_proj`, `task_head` and the
  MLP backbone. `forward(obs, lang_emb, image_emb) -> action`,
  `forward_with_aux(...) -> (action, task_logits)`.
- Loss is unchanged in shape: `BC_weighted_MSE + aux_weight * CE(task_logits, task_id)`.

```bash
python scripts/m6_train_vla_lang_aux.py --config configs/m6_vla_aux_v0.yaml
```

Result (seed=42, 500 epochs):

```text
best_epoch          : 269
best_val_mse_norm   : 0.00437   (M5.1: 0.00384; M5: 0.00426)
best_val_task_acc   : 1.000
image_emb_dim       : 768
image_proj_dim      : 128
aux_weight          : 1.0
```

`val_task_acc = 1.0` is preserved from M5.1: even though `image_proj`
now exists alongside `lang_proj`, the classification head still
perfectly discriminates tasks from `lang_proj` alone — so the language
signal stays informative on its own terms.

`best_val_mse_norm = 0.00437` is slightly higher than M5.1's 0.00384.
With CE driving `lang_proj` to be perfectly separable and BC pulling
both projections toward action-prediction usefulness, the extra capacity
(image_proj_dim=128) does not translate to lower validation MSE in this
data regime.

## 4. Closed-loop swap matrix

For each (env_id, instruction), 30 episodes at seed=3000. Inference
renders at every step and re-encodes the image through CLIP visual on GPU.

| env       | instruction      | success | grasped | placed | min_dist | mean_return |
| --        | --               | --      | --      | --     | --       | --          |
| PickCube  | Pick (matched)   | 0.00    | **0.30** | 0.03   | 0.187    | 13.42       |
| PickCube  | Push (swap)      | 0.00    | 0.07    | 0.03   | 0.202    | 12.59       |
| PickCube  | Pull (swap)      | 0.00    | 0.10    | 0.00   | 0.199    | 12.20       |
| PushCube  | Push (matched)   | 0.13    | 0.00    | 0.00   | 0.166    | 17.99       |
| PushCube  | Pick (swap)      | 0.00    | 0.00    | 0.00   | 0.195    | 21.15       |
| PullCube  | Pull (matched)   | 0.30    | 0.00    | 0.00   | 0.163    | 19.02       |

Side-by-side with M5.1 (no vision):

| Metric                              | M5.1 | M6   | Δ                                                |
| --                                  | --   | --   | --                                               |
| PickCube grasp matched              | 0.07 | 0.30 | **+0.23**  spatial competence jump               |
| PickCube min_dist matched           | 0.192 | 0.187 | -0.005   cube reached slightly closer           |
| PickCube grasp swap (push/pull)     | 0.00 | 0.07/0.10 | **+0.07/+0.10**   instruction-following weakened |
| PushCube success matched            | 0.20 | 0.13 | **-0.07**   regression                           |
| PullCube success matched            | 0.33 | 0.30 | -0.03    similar                                 |
| mean_return (all cells)             | -    | up   | M6 ≥ M5.1 in every single cell                   |

## 5. The trade-off

**Where vision helps**

- PickCube needs cube-localization to grasp. M5.1 only had state for that
  (which obviously contains the cube position), but the state-based
  policy did not always exploit it. Adding image_emb provides a redundant
  cube-position signal *plus* visual context the policy can lean on — and
  grasp rate jumps from 7% to 30%.
- Every cell in the matrix shows lower `min_cube_goal_dist` and higher
  `mean_return`. The robot is more competent at *getting to* the goal.

**Where vision hurts**

- PickCube + swap-instruction grasp rate went 0% → 7%/10%. The policy
  partially overrides the swapped instruction because the image tells
  it "there is a cube right there, you can grasp it." The instruction
  signal is no longer the sole determinant of grasp.
- PushCube matched success dropped 20% → 13%. The push trajectory is
  short and tightly state-determined; the extra image input adds noise
  that the M5.1 policy did not have to deal with.

This is the classic VLA shortcut tension: with two strong modalities
both correlated with the action, gradient descent will exploit whichever
is locally easier per state. In M5.1 there was only one source of
task discrimination (language) so the policy obeyed it cleanly; in M6
the image is also task-discriminative and competes for influence.

## 6. Engineering notes

- Render resolution 224×224 matches CLIP input — no extra resize.
- Inference cost: ~1.2s per cell × 30 episodes ≈ 36s per cell.
- The aux classification head deliberately sits on `lang_proj` only.
  An earlier consideration was to put it on the concatenation
  `concat(lang_proj, image_proj)`, but that would let the image satisfy
  the auxiliary objective and undo the M5.1 instruction-following pressure.

## 7. Limitations + handoff to M7

What M6 confirms:

- ✅ Full VLA stack — image + instruction + state → action — works end to
  end with frozen CLIP towers.
- ✅ Vision adds real spatial competence (PickCube grasp 7% → 30%).
- ✅ Validation task_acc stays at 1.0 — language signal is still meaningful.

What M6 surfaces:

- ❌ Adding vision regresses instruction-following purity.
  The M5.1 "obey-instruction" behavior is partially diluted.
- ❌ Matched-success rate goes down on PushCube and PullCube.

Possible M7 directions:

- **M7a — Image-only VLA**: drop `state_obs` entirely; force the policy
  to localize cube/goal purely from vision. Hardest setting.
- **M7b — Vision-language alignment loss**: add contrastive or
  alignment objective so image and language are pulled into a shared
  representation, reducing competition.
- **M7c — Multi-target instructions**: same env, instructions that vary
  the target (left vs right fixture), so language must encode finer
  spatial concepts.
