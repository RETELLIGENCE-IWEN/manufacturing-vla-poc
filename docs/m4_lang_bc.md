# M4 — Language-Conditioned BC

M4 adds a language encoder to the phase-aware BC pipeline so that downstream
multi-task / multi-instruction milestones (M5+) can sit on a working
language-action code path.

This milestone is **infrastructure**, not capability. PickCube still has only
one underlying task, so the instruction is a dummy signal at this point. The
sanity-check criterion is that closed-loop performance must not regress vs the
matching non-language BC baseline trained on the same 100 demos.

## 1. Pipeline

```text
instruction (text)
       │
       │  CLIPTokenizer + CLIPTextModel (openai/clip-vit-base-patch32, frozen)
       ▼
lang_emb       (512-dim pooled output)
       │
       │  nn.Linear(512 → 64) + ReLU   ← learned during BC training
       ▼
lang_proj      (64-dim)
       │
state_obs (57) + progress (1) + prev_action (8) ─┐
                                                 ├─► concat (130) ─► MLP backbone ─► action (8)
lang_proj  (64) ─────────────────────────────────┘
```

The CLIP weights themselves are not fine-tuned in M4 — the BC dataset is
far too small (100 episodes) and the goal of M4 is the wiring, not text
representation quality.

## 2. Dataset augmentation

`m4_add_instruction_embeddings.py` reads an existing phase-aware dataset
(`outputs/m3_phase_aware_dataset_100/`), encodes each episode's instruction
with CLIP, broadcasts the resulting 512-dim embedding across all time steps,
and writes a new dataset directory with `lang_emb` as an additional NPZ key.

```bash
python scripts/m4_add_instruction_embeddings.py \
  --in-dir outputs/m3_phase_aware_dataset_100 \
  --out-dir outputs/m4_lang_phase_aware_dataset_100 \
  --text-encoder openai/clip-vit-base-patch32
```

Output extends each `episodes/ep_NNNNNN.npz`:

```text
obs           (T, 66)
actions       (T, 8)
rewards       (T,)
terminated    (T,)
truncated     (T,)
success       (T,)
state_obs     (T, 57)
progress      (T, 1)
prev_action   (T, 8)
lang_emb      (T, 512)   ← new (broadcast from per-episode embedding)
```

Embedding statistics on the 100-demo set:

```text
num_unique_instructions : 33
mean_norm               : 23.5
min_norm                : 23.0
max_norm                : 24.0
```

## 3. Training

```bash
python scripts/m4_train_bc_lang.py --config configs/m4_bc_lang_v0.yaml
```

The trainer is `m3_train_bc_phase_weighted.py` extended:

- dataset returns `(obs, lang_emb, actions, sample_weight)` instead of `(obs, actions, sample_weight)`
- the policy is `LangBCPolicy(obs_dim, lang_emb_dim, lang_proj_dim, action_dim, hidden_dims, dropout)` and projects `lang_emb` to `lang_proj_dim=64` before concatenation with `obs`
- checkpoint stores `lang_emb_dim` and `lang_proj_dim` so the evaluator can reconstruct the architecture

Result:

```text
best_epoch         : 251 / 500
best_val_mse_norm  : 0.00584
lang_emb_dim       : 512
lang_proj_dim      : 64
```

The matching non-language BC (`runs/m3_bc_phase_aware/`) had
`best_val_mse_norm = 0.00595` — essentially identical, which means the additional
language input is neither helping nor hurting open-loop action prediction.

## 4. Evaluation

```bash
python scripts/m4_eval_closedloop_bc_lang.py \
  --model runs/m4_bc_lang_v0/best_model.pt \
  --normalization runs/m4_bc_lang_v0/normalization_stats.npz \
  --expert-action-bounds outputs/m3_phase_aware_dataset_100/action_bounds.json \
  --num-episodes 30 \
  --seed 3000 \
  --max-steps 120 \
  --phase-horizon 80 \
  --instruction "Pick the bolt-like part and place it into the left fixture." \
  --out-dir runs/m4_bc_lang_v0/closedloop_eval_safe
```

The instruction is encoded **once** at startup and reused at every step. All
existing eval features carry over (safe action filter, final-hold wrapper,
force_grip_while_far heuristic).

## 5. Sanity-check result

| Metric                    | M3 phase_aware (no lang) | M4 lang_v0 (CLIP) |
| --                        | --                       | --                |
| success_rate_once         | 0.000                    | 0.000             |
| grasped_once_rate         | 0.567                    | 0.500             |
| placed_once_rate          | 0.033                    | **0.033** ✓       |
| mean_return               | 14.56                    | 13.92             |
| mean_final_cube_goal_dist | 0.289                    | 0.277             |
| mean_min_cube_goal_dist   | 0.165                    | 0.164             |

Conclusion: the policy still reaches the same closed-loop performance as
the non-language baseline. The language code path does not corrupt behavior
when the language signal carries no useful information, which is the
prerequisite for M5 — where the instruction will actually need to drive
different action sequences.

## 6. Limitations / what M4 does not do

- M4 does **not** improve PickCube success. It cannot; the instruction
  carries no new task signal.
- M4 does **not** fine-tune CLIP. The CLIP text tower is used as a frozen
  feature extractor only.
- M4 does **not** include vision. State input is unchanged. M6 will swap
  state for rendered RGB and reuse CLIP's visual tower.

## 7. What M5 will need from M4

- The `LangBCPolicy` architecture and the dataset NPZ schema with `lang_emb`.
- The eval script's `--instruction` argument and the `encode_instruction` helper.
- The Dockerfile entry for the `transformers` package.

When M5 produces a multi-task dataset, the M4 code can be reused directly with
new dataset paths and longer training.
