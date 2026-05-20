# M7 / M7.1 — Diffusion Policy on the multi-task VLA dataset

By M6.2 the BC ceiling on the multi-task PickCube/PushCube/PullCube setup
was clearly capacity-bound: each BC variant (v0, v1, v2) was best at a
different task, and `late_weight` boosts only shifted the trade-off rather
than removing it. M7 swaps the deterministic BC head for a Diffusion
Policy (Chi et al. 2023): a noise-prediction model over an action chunk.

## 1. Architecture

```text
state_obs (57) + progress (1) + prev_action (8)   = obs (66)
lang_emb (512) + image_emb (768)                  = conditioning inputs

lang_proj  : 512 → 64                              (learned)
image_proj : 768 → 128                             (learned)
cond_enc   : MLP(66 + 64 + 128) → 256              (learned)
time_emb   : sinusoidal(t) → 128 → MLP → 128       (timestep encoding)

global_cond = concat(time_emb (128), cond (256))   → 384-dim

action sequence (noisy) [B, T_chunk, A] → permute → [B, A, T_chunk]
  → Conv1d input_proj                            (A → hidden_dim)
  → ResBlock1D × num_blocks                      (Conv1d + GroupNorm + Mish + FiLM(global_cond))
  → Conv1d output_proj                           (hidden_dim → A)
  → permute → [B, T_chunk, A] = predicted noise

aux: task_head(lang_proj) → CE(task_id)            (kept from M5.1/M6)
```

Defaults: `hidden_dim=256`, `num_blocks=3`, `cond_dim=256`,
`action_chunk=8`, `num_train_timesteps=100`, `beta_schedule=squaredcos_cap_v2`,
`prediction_type=epsilon`.

Code:
- [scripts/m7_train_diffusion_policy.py](../scripts/m7_train_diffusion_policy.py)
- [scripts/m7_eval_closedloop_diffusion.py](../scripts/m7_eval_closedloop_diffusion.py)
- [scripts/m7_record_diffusion_debug.py](../scripts/m7_record_diffusion_debug.py)

## 2. Training

```bash
python scripts/m7_train_diffusion_policy.py --config configs/m7_diffusion_v0.yaml
```

DDPM training, 200 (v0) or 500 (v1) epochs, batch_size=256, AdamW
lr=3e-4, weight_decay=1e-6, grad clip 1.0.

Results:

| Variant | best_epoch | best_val_diffusion_mse | val_task_acc |
| --      | --         | --                     | --           |
| v0      | 186 / 200  | 0.0126                 | 1.000        |
| **v1**  | 381 / 500  | **0.0093**             | 1.000        |

Auxiliary task discrimination stays at 1.0 — the instruction signal is
load-bearing even under the diffusion objective.

## 3. Inference — DDIM + receding horizon

Each "step" of the closed-loop rollout pulls one action from a sampled
chunk. After `action_exec` real-env steps, the chunk is exhausted and we
sample a fresh chunk conditioned on the current observation.

```bash
python scripts/m7_eval_closedloop_diffusion.py \
  --env-id PushCube-v1 \
  --model runs/m7_diffusion_v1/best_model.pt \
  --normalization runs/m7_diffusion_v1/normalization_stats.npz \
  --expert-action-bounds outputs/m6_multitask_vla_dataset_v2/action_bounds.json \
  --instruction "Push the bolt-like part across the table to the forward goal area." \
  --action-exec 2 --num-inference-steps 24 \
  --skip-random --out-dir runs/m7_diffusion_v1/eval_pushcube_push
```

`action_exec=2` means the policy replans every 2 env steps (more
reactive). `num_inference_steps=24` is the DDIM sampling budget per
replan.

## 4. v0 → v1 hyperparameter changes

| Hyperparameter        | v0 | v1 | rationale                                                 |
| --                    | -- | -- | --                                                        |
| `num_epochs`          | 200 | **500** | v0 stopped improving around ep 186; v1 trained to 381   |
| `action_exec`         | 4  | **2** | finer reaction to env feedback (closer to MPC)            |
| `num_inference_steps` | 16 | **24** | higher-quality DDIM samples                              |

Other settings (model size, action_chunk, train timesteps) stayed the same.

## 5. Closed-loop results

30 episodes per cell, seed=3000, phase_horizon=80, max_steps=120.

### Matched-instruction cells

| env       | v2 BC (M6.2) | v0 M7 | **v1 M7.1** |
| --        | --           | --    | --          |
| PickCube  | 0.00 (grasp 0.13) | 0.00 (grasp 0.00) | 0.00 (grasp 0.00) |
| **PushCube** | 0.30         | 0.30  | **0.47** ⭐  |
| PullCube  | **0.43**     | 0.30  | 0.23        |

### Swap-instruction cells (PickCube env with Push/Pull instruction etc.)

| env       | instr           | v2 BC | v0 M7 | v1 M7.1 |
| --        | --              | --    | --    | --      |
| PickCube  | Push (swap)     | 0.00  | 0.00  | 0.00    |
| PullCube  | Pick (swap)     | 0.13  | 0.17  | 0.20    |

### Activity / behavior metrics (mean_return)

`mean_return` rises across every single cell from BC v2 → M7.1:

| env       | v2 BC | v1 M7.1 | ratio  |
| --        | --    | --      | --     |
| PickCube  | 11.23 | 13.47   | ×1.20  |
| PushCube  | 18.03 | 18.74   | ×1.04  |
| PullCube  | 13.47 | 27.69   | ×2.06  |
| PullCube (swap) | 16.00 | 25.04 | ×1.57 |

The diffusion policy keeps producing non-trivial actions throughout the
episode, where the deterministic BC tends to settle into static behavior
after a few steps. This is the most visible signature of a multi-modal
action distribution in our results.

## 6. The trade-off didn't disappear

The capacity-sharing pattern from M6 v0/v1/v2 carries over to M7:

| Variant       | Best at      |
| --            | --           |
| BC v0         | PickCube     |
| BC v1         | PushCube     |
| BC v2         | PullCube     |
| **M7 v0**     | parity       |
| **M7.1 v1**   | **PushCube** |

A single shared multi-task model still can't dominate all three tasks
simultaneously — neither under deterministic BC nor under diffusion. The
diffusion head changes *where* the trade-off sits (PushCube goes up,
PullCube goes down) but doesn't remove it. To break this ceiling we
would need either:

- **per-task output heads** (M8d) — capacity decoupled at the head
- **a larger pretrained backbone** (M8e: OpenVLA/Octo) — capacity sharing
  becomes affordable because the trunk has more to share
- **more demos per task** — the obvious lever

## 7. What this milestone is worth in the portfolio narrative

- ✅ A complete diffusion-policy implementation from scratch on top of
  the multi-task VLA stack: 1D U-Net + FiLM conditioning + DDPM training
  + DDIM receding-horizon inference + auxiliary classification loss.
- ✅ A clean, head-to-head comparison with deterministic BC on the same
  dataset, with stricter metrics (xy distance, sustained-in-goal,
  `--ignore-termination`).
- ✅ A genuine **+17 pp success rate jump on PushCube** (30% → 47%) once
  tuned, demonstrating the value of the multi-modal action distribution.
- ✅ A clean demonstration that the multi-task capacity-sharing ceiling
  is **architecture-agnostic** at our scale — supporting the motivation
  for per-task heads or foundation backbones in subsequent milestones.

## 8. Limitations

- Stochastic inference: the same env seed does not produce a deterministic
  policy rollout. We document this in the M7.1 PushCube `seed=3029`
  video (settles at `min_xy=0.144`) vs the `seed=3017` video
  (`min_xy=0.048`) — both with the same trained model.
- PullCube settle behavior degraded (v2 BC 43% → M7.1 23%), likely because
  the longer training and tighter replan favored short-horizon push-style
  trajectories over the longer pull-and-hold trajectories.
- Diffusion sampling adds inference latency (~3-4 s per env episode
  at action_exec=2, num_inference_steps=24, single RTX 3060). Manageable
  for evaluation, but noticeable.
