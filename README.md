# Manufacturing VLA PoC

A compact applied robotics AI project for manufacturing-style, language-conditioned robotic manipulation.

This project builds a small but complete VLA-style robotics pipeline that connects:

- language-conditioned task metadata
- robot simulation
- expert demonstration generation
- step-level dataset logging
- behavior cloning
- closed-loop policy evaluation
- failure diagnosis and policy iteration

The current base task is built on ManiSkill `PickCube-v1`, using a Panda arm to perform pick-and-place manipulation.

---

## Project Goal

The goal is not to train a large foundation VLA model from scratch.

Instead, this project demonstrates an applied robotics AI workflow:

```text
Language instruction
+ robot/object/goal state
+ expert demonstration
→ behavior cloning policy
→ closed-loop robotic manipulation evaluation
```

The project is intended as a small portfolio-grade PoC for manufacturing robotics, embodied AI, and VLA-style robot learning.

---

## Current Status

Current milestone status:

```text
M0     Completed — Dockerized ManiSkill bring-up
M1     Completed — Manufacturing-style language task wrapper
M2A    Completed — Step-level dataset logger
M2B    Completed — Expert demonstration generation (100 + 5000 demo sets)
M2C    Completed — Expert H5 to M3-ready dataset conversion
M3.0   Completed — State-only behavior cloning training
M3.1   Completed — Open-loop action prediction evaluation
M3.2   Completed — Closed-loop rollout evaluation
M3.3A  Completed — Closed-loop failure diagnosis with video/logs
M3.3L  Completed — Longer state-only BC training experiment
M3.4   Completed — Phase-aware BC (obs = state + progress + prev_action)
M3.5   Completed — Safe action filter (expert-bounded action clipping)
M3.6   Completed — Final-hold stabilization wrapper for phase-aware BC
M3.7   Completed — 5000-demo expert phase-aware BC
M3.8   Completed — Phase-weighted loss for late-trajectory emphasis
M3.9   Completed — DAgger pipeline (rollout, planner-relabel, aggregate, train)
M3.9H  Completed — force_grip_while_far inference heuristic
M4     Completed — Language-conditioned BC infrastructure (CLIP-text + state)
M5     Completed — Multi-task (PickCube + PushCube + PullCube) language-conditioned BC
M5.1   Completed — Auxiliary task_id classification loss (policy now actually uses the instruction)
M6     Completed — VLA: state + CLIP-text + CLIP-vision policy
M6.1   Completed — PushCube settle solver + late_weight 8 (PushCube success 13%→50%)
M6.2   Completed — PullCube settle solver + xy/sustained metrics + ignore-termination eval flag
                   (PullCube success 30%→43%; multi-task capacity-sharing trade-off documented)
M7     Completed — Diffusion Policy (Chi et al. 2023) on the multi-task VLA dataset
M7.1   Completed — Diffusion Policy with 500 epochs + tuned DDIM eval (PushCube 30%→47%)
M8a    Completed — BC with PCGrad (Yu et al. 2020) gradient surgery (PickCube grasp 13%→20%)
M8b    Completed — Per-task output heads (M8d originally; instruction-obey purity ↑)
M8     Completed — 4-way comparison study: BC v2 / Diffusion / PCGrad / PerTaskHead
```

BC base summary:

> The phase-aware BC + DAgger v1 + force_grip heuristic (M3.4 + M3.9 + M3.9H) is the chosen BC base policy:
> grasped_once 47%, final_grasped 37%, mean_return 24.1, closed-loop success 0%.
> DAgger v2 (release-phase bias selection) underperformed v1 because removing the `deteriorated_after_min_dist`
> selection unintentionally weakened cube-approach behavior; v1 remains the best BC variant.

VLA project context:

> The project's ultimate goal is a **language-conditioned manipulation policy (VLM/VLA-style)**.
> The current state-only BC/DAgger pipeline establishes the action-policy foundation; instruction metadata is
> already captured in the dataset (`instruction`, `object_id`, `target_id`) but is not yet fed into the policy.
> M4 will integrate a text encoder and extend the observation with instruction embeddings, building the
> infrastructure on which multi-task language conditioning (M5+) will sit.

---

## Environment

The project was validated with:

```text
OS: Ubuntu 22.04
GPU: NVIDIA RTX 3060
Driver: NVIDIA 535.288.01
Docker GPU: enabled
Simulator: ManiSkill
Base environment: PickCube-v1
Control mode: pd_joint_pos
```

Validation checks:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
python -c "import mani_skill; print('ManiSkill import OK')"
```

---

## Repository Structure

```text
manufacturing-vla-poc/
  configs/
    manufacturing_pick_place_v0.yaml
    m3_bc_state.yaml
    m3_bc_state_long.yaml
    m3_bc_phase_aware.yaml
    m3_bc_phase_aware_5000.yaml
    m3_bc_phase_weighted_5000.yaml
    m3_bc_dagger_v0.yaml
    m3_bc_dagger_v1.yaml
    m3_bc_dagger_v2.yaml

  docker/
    Dockerfile

  scripts/
    m0_random_rollout.py
    m1_language_task_rollout.py
    m2_*.py
    m3_make_phase_aware_dataset.py
    m3_train_bc_state.py
    m3_train_bc_phase_weighted.py
    m3_eval_openloop_bc_state.py
    m3_eval_closedloop_bc_state.py
    m3_eval_closedloop_bc_phase.py
    m3_eval_closedloop_bc_phase_safe.py        # current evaluator (with force_grip)
    m3_record_closedloop_debug.py
    m3_record_phase_safe_debug.py
    m3_dagger_common.py
    m3_dagger_smoke.py
    m3_collect_dagger_rollouts.py
    m3_relabel_dagger_with_planner.py
    m3_build_dagger_aggregate_dataset.py

  docs/
    m2_dataset_report.md
    m3_9_dagger.md

  outputs/
    m0_random_rollout/ m1_language_task_rollout/ m2_*/
    m3_phase_aware_dataset_100/ m3_phase_aware_dataset_5000/
    m3_dagger_rollouts_{v0,v1,v2}/
    m3_dagger_corrections_{smoke_v0,v1,v2}/
    m3_agg_phase_dagger_dataset_{v1,v2}/

  runs/
    m3_bc_state/ m3_bc_state_long/
    m3_bc_phase_aware/ m3_bc_phase_aware_5000/ m3_bc_phase_weighted_5000/
    m3_bc_dagger_v1/ m3_bc_dagger_v2/           # m3_bc_dagger_v1 is the chosen BC base
```

`outputs/` and `runs/` are generated artifacts and are not expected to be committed.

---

## M0 — Dockerized ManiSkill Bring-up

M0 verifies that ManiSkill runs inside a Docker container with GPU support.

Example command:

```bash
python scripts/m0_random_rollout.py \
  --env-id PickCube-v1 \
  --obs-mode state \
  --num-episodes 3 \
  --max-steps 200 \
  --seed 42
```

Output:

```text
outputs/m0_random_rollout/summary.json
```

Purpose:

> Confirm that the simulation environment, Docker setup, and basic rollout pipeline work.

---

## M1 — Manufacturing-Style Language Task Wrapper

M1 adds manufacturing-style task metadata and language instructions on top of the base ManiSkill task.

Example instructions:

```text
Pick the red component and place it into the left fixture.
Move the bolt-like part to the inspection tray.
Grasp the blue component, then put it in the right fixture.
```

Example command:

```bash
python scripts/m1_language_task_rollout.py \
  --config configs/manufacturing_pick_place_v0.yaml \
  --obs-mode state \
  --num-episodes 10 \
  --max-steps 200 \
  --seed 42
```

Outputs:

```text
outputs/m1_language_task_rollout/summary.json
outputs/m1_language_task_rollout/episodes.jsonl
```

Purpose:

> Convert a simple manipulation task into a language-conditioned manufacturing-style task interface.

---

## M2A — Step-Level Dataset Logger

M2A defines a step-level dataset contract.

Each episode stores:

```text
instruction
observation
action
reward
terminated
truncated
success
task metadata
```

Example command:

```bash
python scripts/m2_collect_step_dataset.py \
  --config configs/manufacturing_pick_place_v0.yaml \
  --obs-mode state \
  --num-episodes 50 \
  --max-steps 200 \
  --seed 42 \
  --action-source random \
  --out-dir outputs/m2_step_dataset
```

Outputs:

```text
outputs/m2_step_dataset/
  dataset_schema.json
  summary.json
  episodes.jsonl
  splits.json
  episodes/
    ep_000000.npz
    ep_000001.npz
```

Purpose:

> Establish the training-oriented data format before training any policy.

---

## M2B — Expert Demonstration Generation

M2B generates expert trajectories using a local PickCube-only wrapper around ManiSkill's Panda motion-planning solution.

The official ManiSkill motion-planning runner had an import mismatch in the installed package version, so this project uses a minimal reproducible PickCube-specific expert generation script.

Example command:

```bash
python scripts/m2_generate_pickcube_expert.py \
  --env-id PickCube-v1 \
  --num-traj 100 \
  --seed 1000 \
  --only-count-success \
  --record-dir outputs/m2_expert_demos_100 \
  --traj-name pickcube_expert_100
```

Expected outputs:

```text
outputs/m2_expert_demos_100/PickCube-v1/motionplanning/
  pickcube_expert_100.h5
  pickcube_expert_100.json
  pickcube_expert_100_summary.json
```

The generated HDF5 trajectories contain:

```text
traj_i/actions
traj_i/terminated
traj_i/truncated
traj_i/success
traj_i/env_states/actors/cube
traj_i/env_states/actors/goal_site
traj_i/env_states/articulations/panda
```

Purpose:

> Obtain successful pick-and-place demonstrations for imitation learning.

---

## M2C — Expert H5 to M3-Ready Dataset Conversion

M2C converts the expert HDF5 trajectories into the project dataset format.

Observation layout:

```text
obs_t = panda state + cube state + goal_site state
```

Current state dimension:

```text
panda:     31 dims
cube:      13 dims
goal_site: 13 dims
total:     57 dims
```

Action dimension:

```text
action: 8 dims
```

Example command:

```bash
python scripts/m2_convert_expert_h5_to_dataset.py \
  --h5 outputs/m2_expert_demos_100/PickCube-v1/motionplanning/pickcube_expert_100.h5 \
  --config configs/manufacturing_pick_place_v0.yaml \
  --out-dir outputs/m2_expert_dataset_100 \
  --seed 42
```

Outputs:

```text
outputs/m2_expert_dataset_100/
  dataset_schema.json
  summary.json
  episodes.jsonl
  splits.json
  episodes/
    ep_000000.npz
    ep_000001.npz
```

Purpose:

> Prepare a clean state-action dataset for behavior cloning.

---

## M3.0 — State-Only Behavior Cloning

M3.0 trains a state-only MLP behavior cloning policy.

Input:

```text
obs_t = panda state + cube state + goal_site state
```

Output:

```text
action_t = expert action
```

Loss:

```text
MSE(predicted_action, expert_action)
```

Example command:

```bash
python scripts/m3_train_bc_state.py \
  --config configs/m3_bc_state.yaml
```

Generated artifacts:

```text
runs/m3_bc_state/
  metadata.json
  normalization_stats.npz
  training_curve.csv
  best_model.pt
  metrics.json
```

Initial result:

```text
best_epoch: 175
best_val_mse_norm: 0.00595
final_val_mse_norm: about 0.00644
final_val_mae_raw: about 0.0148
```

Purpose:

> Verify that the expert state-action mapping can be learned in open-loop supervised training.

---

## M3.1 — Open-Loop Action Prediction Evaluation

M3.1 evaluates the trained BC policy on validation episodes without rolling it out in the simulator.

Example command:

```bash
python scripts/m3_eval_openloop_bc_state.py \
  --dataset-dir outputs/m2_expert_dataset_100 \
  --run-dir runs/m3_bc_state \
  --model runs/m3_bc_state/best_model.pt \
  --normalization runs/m3_bc_state/normalization_stats.npz \
  --split val \
  --num-episodes 5 \
  --out-dir runs/m3_bc_state/openloop_eval
```

Outputs:

```text
runs/m3_bc_state/openloop_eval/
  openloop_summary.json
  ep_XXXXXX_action_compare.csv
```

Purpose:

> Check whether predicted actions match expert actions before running the policy closed-loop.

---

## M3.2 — Closed-Loop Rollout Evaluation

M3.2 places the trained BC policy back into the ManiSkill environment and compares it against a random-action baseline.

Example command:

```bash
python scripts/m3_eval_closedloop_bc_state.py \
  --env-id PickCube-v1 \
  --model runs/m3_bc_state/best_model.pt \
  --normalization runs/m3_bc_state/normalization_stats.npz \
  --num-episodes 30 \
  --max-steps 120 \
  --seed 3000 \
  --out-dir runs/m3_bc_state/closedloop_eval
```

Observed result:

```text
success_rate_once_delta: 0.0
final_success_rate_delta: 0.0
mean_return_delta: +6.65
```

Interpretation:

> The BC policy produced valid closed-loop actions and achieved slightly higher mean return than random, but did not improve task success rate.  
> This indicates that open-loop action prediction alone is insufficient for robust closed-loop grasping.

---

## M3.3A — Closed-Loop Failure Diagnosis

M3.3A records debug videos and per-step logs for BC rollouts.

Example command:

```bash
python scripts/m3_record_closedloop_debug.py \
  --policy bc \
  --env-id PickCube-v1 \
  --model runs/m3_bc_state/best_model.pt \
  --normalization runs/m3_bc_state/normalization_stats.npz \
  --seed 3000 \
  --max-steps 120 \
  --save-video \
  --out-dir runs/m3_bc_state/closedloop_debug
```

Observed failure mode:

```text
The policy failed to properly grip the cube and appeared to push it away.
```

Debug summary:

```text
initial_cube_goal_dist: 0.1747
final_cube_goal_dist:   1.4038
min_cube_goal_dist:     0.1747
success_once:           false
final_success:          false
```

Interpretation:

> The policy does not simply produce small or inactive actions.  
> It produces non-trivial actions, but fails around the grasp phase and destabilizes the cube.

---

## M3.3L — Longer State-Only BC Training

M3.3L trains the same state-only BC architecture for a longer schedule.

Purpose:

> Check whether the closed-loop failure is caused by insufficient optimization.

Result:

```text
Closed-loop success rate remained 0%.
```

Conclusion:

> The failure is unlikely to be solved by longer training alone.  
> The state-only policy lacks phase/progress information required to distinguish approach, grasp, lift, and place stages.

---

## M3.4 — Phase-Aware Behavior Cloning

Phase-aware BC extends the observation with phase signals.

```text
obs = state_57 + progress_1 + prev_action_8   (obs_dim = 66)

progress     = step_idx / (phase_horizon - 1)
prev_action  = previous executed action
```

This reduces phase ambiguity that pure state-only BC suffered from
(approach vs align vs grasp vs lift vs transfer vs place).

Outputs:

```text
runs/m3_bc_phase_aware/           — 100-demo phase-aware BC
runs/m3_bc_phase_aware_5000/      — 5000-demo phase-aware BC
runs/m3_bc_phase_weighted_5000/   — 5000-demo with phase-weighted loss
```

Result on PickCube-v1, 30 episodes, seed=3000, phase_horizon=80, max_steps=120:

```text
grasped_once_rate            : 0.567
placed_once_rate             : 0.033
mean_return                  : 14.56
success_rate_once            : 0.000
mean_final_cube_goal_dist    : 0.289
mean_min_cube_goal_dist      : 0.165
```

The cube reaches the goal occasionally (3.3%) but the policy does not robustly close the loop.

---

## M3.5 — Safe Action Filter

Clips policy actions to the expert action distribution (`min..max` per dim, with optional margin).
This prevents the policy from emitting actions outside what the motion-planner expert ever produced.

Used by all M3.6+ evaluations (`--expert-action-bounds`).

---

## M3.6 — Final-Hold Stabilization

Wrapper that latches the previous action whenever cube reaches near-goal or `is_obj_placed` becomes True,
preventing the policy from continuing to push the cube past placement.

Result on 100-demo phase-aware BC + final-hold wrapper (best variant):

```text
success_rate_once : 0.033
placed_once_rate  : 0.033
grasped_once_rate : 0.567
```

First non-zero closed-loop success.

---

## M3.7 / M3.8 — 5000-Demo and Phase-Weighted Loss

Scaled to 5000 expert demos and added a phase-weighted loss
(`early : mid : late` = 1 : 2 : 4) to emphasize the place phase.
Despite more data and a stronger late-phase signal, closed-loop success on
PickCube-v1 remained 0.0%, reinforcing the conclusion that single-task BC alone
plateaus here — additional signals (DAgger, language conditioning, vision) are needed.

---

## M3.9 — DAgger Pipeline

Adds policy-visited state correction:

```text
1. Collect rollouts from the BC policy (m3_collect_dagger_rollouts.py)
2. Select policy-visited states (focused: near_goal / is_grasped / late_grasp / high_action / deteriorated)
3. Restore each state and run the PickCube motion-planning expert to produce correction labels
   (m3_relabel_dagger_with_planner.py)
4. Aggregate expert + corrections with per-sample weights (m3_build_dagger_aggregate_dataset.py)
5. Retrain phase-aware BC on the aggregated dataset (m3_train_bc_phase_weighted.py)
```

Iterations:

```text
v0  — smoke test of restore + planner first-action correction
v1  — full run: 50 episodes from phase-aware BC, 1553 selected states,
       1121 successful corrections (72.2%), sample_weight=2.0
v2  — release-phase biased: collected from v1 policy with new
       `late_grasp_far_from_goal` selection and `--disable-deteriorated`.
       899 selected, 830 corrections (92.3%), but underperformed v1.
```

DAgger v1 closed-loop result (best BC variant before heuristic):

```text
success_rate_once       : 0.0
grasped_once_rate       : 0.467
placed_once_rate        : 0.0
final_grasped_rate      : 0.167
mean_return             : 20.96
mean_final_cube_goal_dist : 0.197
final_robot_static_rate : 0.933
```

DAgger v1 noticeably improves trajectory shape (cube ends closer to goal) and end-stability,
but does not break through to placed success.

See [docs/m3_9_dagger.md](docs/m3_9_dagger.md) for the full DAgger workflow and v0→v1→v2 analysis.

---

## M3.9H — force_grip_while_far Inference Heuristic

Debug rollouts on DAgger v1 revealed the dominant remaining failure mode:
after the policy lifts the cube and reaches `min_cube_goal_dist` around 0.06m,
it opens the gripper at progress≈0.95 regardless of whether the cube is at the goal,
causing the cube to drop.

The inference-time heuristic forces `action[-1] = -1.0` (gripper closed) once the
policy has ever grasped the cube **and** the cube is still further than a threshold
from the goal:

```bash
python scripts/m3_eval_closedloop_bc_phase_safe.py \
  --model runs/m3_bc_dagger_v1/best_model.pt \
  --normalization runs/m3_bc_dagger_v1/normalization_stats.npz \
  --expert-action-bounds outputs/m3_agg_phase_dagger_dataset_v1/action_bounds.json \
  --force-grip-while-far \
  --force-grip-dist-threshold 0.05 \
  --out-dir runs/m3_bc_dagger_v1/closedloop_eval_safe_forcegrip
```

Result (final BC base used for downstream language-conditioned milestones):

```text
grasped_once_rate     : 0.467
final_grasped_rate    : 0.367   (vs 0.167 without heuristic)
mean_return           : 24.10   (vs 20.96 without heuristic)
placed_once_rate      : 0.0
```

The heuristic roughly doubles end-state grasp retention.

---

## M4 — Language-Conditioned BC Infrastructure (VLA prerequisite)

M4 wires a CLIP text encoder into the BC pipeline so the policy can consume the
instruction attached to every demo.

```text
state_obs = panda_31 + cube_13 + goal_site_13              (= 57)
obs       = state_obs + progress_1 + prev_action_8         (= 66)
lang_emb  = CLIPTextModel("clip-vit-base-patch32").pooler_output(instruction)   (= 512)

policy_input = concat(obs, lang_proj(lang_emb))
             = 66 + 64
             = 130
```

Components delivered:

- [scripts/m4_add_instruction_embeddings.py](scripts/m4_add_instruction_embeddings.py)
  precomputes per-episode CLIP embeddings and copies them into a new dataset
  directory (`outputs/m4_lang_phase_aware_dataset_100/`).
- [scripts/m4_train_bc_lang.py](scripts/m4_train_bc_lang.py)
  trains a `LangBCPolicy` that holds a learned `lang_proj` layer
  (`lang_emb_dim=512 → lang_proj_dim=64`).
- [scripts/m4_eval_closedloop_bc_lang.py](scripts/m4_eval_closedloop_bc_lang.py)
  encodes the user-provided `--instruction` once with CLIP at startup and
  reuses the embedding every step. All existing eval features
  (safe-action filter, final-hold wrapper, force-grip heuristic) are preserved.
- [configs/m4_bc_lang_v0.yaml](configs/m4_bc_lang_v0.yaml) — training config.

Build + train + evaluate:

```bash
python scripts/m4_add_instruction_embeddings.py \
  --in-dir outputs/m3_phase_aware_dataset_100 \
  --out-dir outputs/m4_lang_phase_aware_dataset_100 \
  --text-encoder openai/clip-vit-base-patch32

python scripts/m4_train_bc_lang.py --config configs/m4_bc_lang_v0.yaml

python scripts/m4_eval_closedloop_bc_lang.py \
  --model runs/m4_bc_lang_v0/best_model.pt \
  --normalization runs/m4_bc_lang_v0/normalization_stats.npz \
  --expert-action-bounds outputs/m3_phase_aware_dataset_100/action_bounds.json \
  --instruction "Pick the bolt-like part and place it into the left fixture." \
  --out-dir runs/m4_bc_lang_v0/closedloop_eval_safe
```

Sanity check (vs the non-language M3 phase-aware BC trained on the same 100 demos):

| Metric                  | M3 phase_aware | M4 lang_v0 (CLIP) |
| --                      | --             | --                |
| success_rate_once       | 0.000          | 0.000             |
| grasped_once_rate       | 0.567          | 0.500             |
| placed_once_rate        | 0.033          | **0.033** ✓       |
| mean_return             | 14.56          | 13.92             |
| mean_final_cube_goal_dist | 0.289        | 0.277             |
| mean_min_cube_goal_dist | 0.165          | 0.164             |

Closed-loop performance is statistically indistinguishable from the non-language
baseline, which is exactly what the sanity check requires:
the language pathway is wired correctly without breaking the action policy.
Because every demo solves the same PickCube task, the instruction is currently a
**dummy signal** that the network is free to ignore — that the policy *can*
ignore it (rather than being corrupted by it) is the precondition for M5.

See [docs/m4_lang_bc.md](docs/m4_lang_bc.md) for the full M4 design notes.

---

## M5 — Multi-Task Language-Conditioned BC

M5 combines three ManiSkill cube-manipulation tasks (`PickCube-v1`,
`PushCube-v1`, `PullCube-v1`) into a single dataset and trains the M4
`LangBCPolicy` on the merged set. All three tasks share the same
state observation layout (panda 31 + cube 13 + goal 13 = 57), so no policy
architecture change is needed.

```bash
# 100 episodes per task with the ManiSkill motion-planning expert
python scripts/m5_generate_multitask_expert.py --env-id PickCube-v1 --num-traj 100 --only-count-success
python scripts/m5_generate_multitask_expert.py --env-id PushCube-v1 --num-traj 100 --only-count-success
python scripts/m5_generate_multitask_expert.py --env-id PullCube-v1 --num-traj 100 --only-count-success

# Merge HDF5 files into a unified M3-style dataset with task metadata + per-task instruction templates
python scripts/m5_convert_multitask_h5_to_dataset.py \
  --record-dir outputs/m5_expert_demos_multitask \
  --config configs/manufacturing_multitask_v0.yaml \
  --out-dir outputs/m5_multitask_dataset

# Add phase-aware features (progress, prev_action)
python scripts/m3_make_phase_aware_dataset.py \
  --src-dir outputs/m5_multitask_dataset \
  --out-dir outputs/m5_multitask_phase_aware_dataset

# Add CLIP instruction embeddings
python scripts/m4_add_instruction_embeddings.py \
  --in-dir outputs/m5_multitask_phase_aware_dataset \
  --out-dir outputs/m5_multitask_lang_phase_aware_dataset

# Reuse the M4 trainer
python scripts/m4_train_bc_lang.py --config configs/m5_bc_lang_multitask_v0.yaml

# Multi-task eval (the m5 eval auto-resolves goal_site vs goal_region)
python scripts/m5_eval_closedloop_bc_lang.py \
  --env-id PickCube-v1 \
  --model runs/m5_bc_lang_multitask_v0/best_model.pt \
  --normalization runs/m5_bc_lang_multitask_v0/normalization_stats.npz \
  --expert-action-bounds outputs/m5_multitask_lang_phase_aware_dataset/action_bounds.json \
  --instruction "Pick the bolt-like part and place it at the left fixture." \
  --out-dir runs/m5_bc_lang_multitask_v0/closedloop_eval_pickcube
```

Result on 300-episode multi-task dataset, 30 eval episodes per cell,
seed=3000, phase_horizon=80, max_steps=120:

| env / instruction         | success | grasped | placed | min_dist | mean_return |
| --                        | --      | --      | --     | --       | --          |
| PickCube / Pick (matched) | 0.00    | 0.10    | 0.00   | 0.195    | 11.91       |
| PickCube / Push (swap)    | 0.00    | 0.10    | 0.00   | 0.199    | 10.15       |
| PickCube / Pull (swap)    | 0.00    | 0.07    | 0.00   | 0.201    | 11.27       |
| **PushCube / Push (matched)** | **0.10** | 0.00 | 0.00 | 0.191    | 15.13       |
| PushCube / Pick (swap)    | 0.00    | 0.00    | 0.00   | 0.196    | 15.69       |
| PullCube / Pull (matched) | 0.00    | 0.00    | 0.00   | 0.193    | 15.70       |

**Two takeaways:**

- ✅ **First closed-loop success on PushCube (10%)** — the project's first
  non-zero `success_rate_once` after M3.0 onward. `best_val_mse_norm` also
  drops to **0.00426** (vs M4's 0.00584), so multi-task data improves
  open-loop accuracy.
- ❌ **Instruction is not used as a differentiator.** Swapping Pick→Push or
  Pick→Pull in the PickCube env barely changes behavior (grasp rate
  10%→10%→7%, min_dist 0.195→0.199→0.201). The policy decides almost
  entirely from `state_obs`; CLIP+lang_proj is effectively dummy input.

Why: the lang signal goes through a small 64-dim projection next to a 66-dim
state input, the CLIP encoder is frozen, and the dataset has only 100 episodes
per task. Behavior cloning has no direct incentive to attend to language when
state alone resolves the action.

See [docs/m5_multitask.md](docs/m5_multitask.md) for the full pipeline + swap-matrix analysis.

---

## M5.1 — Instruction-Conditional Auxiliary Loss (genuine language conditioning)

M5.1 adds a tiny classification head on top of `lang_proj` predicting
`task_id ∈ {pick_place, push_horizontal, pull_horizontal}` and trains jointly:

```text
total_loss = BC_phase_weighted_MSE(action) + aux_weight × CrossEntropy(task_logits, task_id)
```

This forces `lang_proj` activations to be linearly separable by task, which
gives the BC head no excuse to ignore them — gradient through `lang_proj` is
no longer dominated by uninformative noise.

Implementation: [scripts/m5_1_train_bc_lang_aux.py](scripts/m5_1_train_bc_lang_aux.py),
[scripts/m5_1_eval_closedloop_bc_lang_aux.py](scripts/m5_1_eval_closedloop_bc_lang_aux.py),
[configs/m5_1_bc_lang_multitask_aux_v0.yaml](configs/m5_1_bc_lang_multitask_aux_v0.yaml).

Training result:

```text
best_epoch          : 305 / 500
best_val_mse_norm   : 0.00384   (M5: 0.00426; M4: 0.00584)
best_val_task_acc   : 1.000     (lang_proj is now perfectly task-discriminative)
aux_weight          : 1.0
```

### Closed-loop swap matrix — M5 vs M5.1

30 episodes per cell, seed=3000, phase_horizon=80, max_steps=120.

| env       | instruction      | M5 success | **M5.1 success** | M5 grasped | M5.1 grasped | M5 placed | M5.1 placed |
| --        | --               | --         | --               | --         | --           | --        | --          |
| PickCube  | Pick (matched)   | 0.00       | 0.00             | 0.10       | 0.07         | 0.00      | **0.03** ⭐  |
| PickCube  | Push (swap)      | 0.00       | 0.00             | **0.10**   | **0.00** ⭐   | 0.00      | 0.00        |
| PickCube  | Pull (swap)      | 0.00       | 0.00             | 0.07       | **0.00** ⭐   | 0.00      | 0.00        |
| PushCube  | Push (matched)   | 0.10       | **0.20** ⭐       | 0.00       | 0.00         | 0.00      | 0.00        |
| PushCube  | Pick (swap)      | 0.00       | 0.00             | 0.00       | 0.00         | 0.00      | 0.00        |
| PullCube  | Pull (matched)  | 0.00       | **0.33** ⭐⭐       | 0.00       | 0.00         | 0.00      | 0.00        |

**Key observations:**

- ✅ **Genuine instruction following** — in PickCube the grasp rate drops
  10%/7% → **0%/0%** when the instruction is swapped to Push/Pull. The
  policy now consults the instruction before deciding to grasp.
- ✅ **Matched-instruction success jumps**:
  - PullCube: **0% → 33%** (project's first non-zero PullCube success)
  - PushCube: 10% → **20%**
  - PickCube placed: 0% → **3%** (project's first non-zero placement)
- ✅ Min cube-goal distance is also instruction-sensitive (PullCube:
  0.193 → 0.163 with the matching instruction).
- ✅ `val_task_acc = 1.0` — lang_proj activations encode the task perfectly,
  and the BC head clearly uses them.

This is the first milestone where the project ships a real language-conditioned policy: instruction is no longer a dummy passenger.

See [docs/m5_1_aux_loss.md](docs/m5_1_aux_loss.md) for the full analysis.

---

## M6 — VLA: state + CLIP-text + CLIP-vision

M6 wires the CLIP vision tower into the M5.1 multi-task policy. Each
transition now feeds *three* inputs to the BC head:

```text
state_obs   (57)  + progress (1) + prev_action (8)   = obs (66)
lang_emb    (512)  ← frozen CLIPTextModel.pooler_output(instruction)
image_emb   (768)  ← frozen CLIPVisionModel.pooler_output(env.render())

policy_input = concat(obs, lang_proj(lang_emb), image_proj(image_emb))
             = 66 + 64 + 128
             = 258
```

`task_head` (auxiliary classification) stays on top of `lang_proj` only,
to keep the M5.1 instruction-following pressure intact.

Pipeline:

```bash
# Precompute per-step CLIP-vision embeddings by replaying every expert traj
# (uses episode_seed from H5 metadata; renders at 224x224 to match CLIP input)
python scripts/m6_add_image_embeddings.py \
  --in-dir outputs/m5_multitask_lang_phase_aware_dataset \
  --out-dir outputs/m6_multitask_vla_dataset

# Train VLA: BC weighted-MSE + aux task_id cross-entropy
python scripts/m6_train_vla_lang_aux.py --config configs/m6_vla_aux_v0.yaml

# Closed-loop eval: instruction is encoded once; image is encoded every step
python scripts/m6_eval_closedloop_vla.py \
  --env-id PickCube-v1 \
  --model runs/m6_vla_aux_v0/best_model.pt \
  --normalization runs/m6_vla_aux_v0/normalization_stats.npz \
  --expert-action-bounds outputs/m6_multitask_vla_dataset/action_bounds.json \
  --instruction "Pick the bolt-like part and place it at the left fixture." \
  --skip-random \
  --out-dir runs/m6_vla_aux_v0/eval_pickcube_pick
```

Training result:

```text
best_epoch          : 269 / 500
best_val_mse_norm   : 0.00437   (M5.1: 0.00384; M5: 0.00426)
best_val_task_acc   : 1.000
image_emb_dim       : 768
image_proj_dim      : 128
```

### Closed-loop swap matrix — M5.1 vs M6

30 episodes per cell, seed=3000, phase_horizon=80, max_steps=120.

| env       | instruction      | M5.1 success | M6 success | M5.1 grasp | **M6 grasp** | M5.1 min_dist | M6 min_dist |
| --        | --               | --           | --         | --         | --           | --            | --          |
| PickCube  | Pick (matched)   | 0.00         | 0.00       | 0.07       | **0.30** ⭐   | 0.192         | 0.187       |
| PickCube  | Push (swap)      | 0.00         | 0.00       | **0.00**   | 0.07         | 0.207         | 0.202       |
| PickCube  | Pull (swap)      | 0.00         | 0.00       | **0.00**   | 0.10         | 0.203         | 0.199       |
| PushCube  | Push (matched)   | **0.20**     | 0.13       | 0.00       | 0.00         | 0.173         | 0.166       |
| PushCube  | Pick (swap)      | 0.00         | 0.00       | 0.00       | 0.00         | 0.201         | 0.195       |
| PullCube  | Pull (matched)   | **0.33**     | 0.30       | 0.00       | 0.00         | 0.163         | 0.163       |

### Trade-off observed

- ✅ **Vision adds spatial competence:** PickCube grasp rate jumps **7% → 30%**.
  `min_cube_goal_dist` improves in every cell. `mean_return` rises in every
  cell. The policy gets to the cube faster and more often.
- ❌ **Vision dilutes instruction following:** in PickCube + swap-instruction,
  grasp goes from 0% → 7-10%. The policy now sees "cube is there" from the
  image and partially overrides the instruction. PushCube matched-success
  also drops 20% → 13%.

This is the classic vision/language shortcut tension: with two strong
modalities, gradient descent will exploit whichever is locally easier. M6
demonstrates the full VLA stack works end-to-end (real image inputs at
inference time, frozen CLIP towers, joint BC + auxiliary classification)
while making the trade-off visible.

See [docs/m6_vla.md](docs/m6_vla.md) for the full pipeline + analysis.

---

## M6.1 — PushCube settle solver + stricter metrics

Watching the M6 success videos revealed that ManiSkill's native success
criterion is lax: PushCube/PullCube fire `success=True` the instant the
cube center crosses the goal-region boundary (`xy_distance < 0.1`), and
the env terminates. The cube never settles. PickCube's `is_obj_placed`
is similarly distance-only; one fleeting moment within 0.025 m of the goal
fires it.

Diagnosis after expert-video review:

- **PickCube expert** settles the cube cleanly inside the goal_site.
- **PullCube expert** also settles the cube near the goal_region center.
- **PushCube expert** stops at the boundary — it inherits the lax success
  criterion.

Two fixes in M6.1:

1. [scripts/m6_generate_multitask_expert_v1.py](scripts/m6_generate_multitask_expert_v1.py)
   adds a third motion-planning stage to PushCube (`tcp = goal_region.x - 0.06`)
   so the cube ends up near the goal center, not on the boundary.
2. [configs/m6_vla_aux_v1.yaml](configs/m6_vla_aux_v1.yaml) raises
   `late_weight: 4 → 8` so the BC head pays more attention to the
   late-trajectory settle behavior.

Result on PushCube + Push (matched), 30 episodes, seed=3000:

| Metric                 | M6 (v0) | **M6.1 (v1)** |
| --                     | --      | --            |
| success_rate_once      | 0.13    | **0.50**      |
| mean_min_xy_dist       | —       | 0.144         |

Debug video (`runs/m6_vla_aux_v1/debug_video/PushCube-v1_seed_3029/...`,
recorded with `--ignore-termination`) shows the policy pushing the cube
to `min_cube_goal_dist = 0.027` (27 mm), past the boundary toward the
goal center.

Stricter eval metrics added to
[scripts/m6_eval_closedloop_vla.py](scripts/m6_eval_closedloop_vla.py):

```text
final_cube_xy_goal_dist      (xy distance at final step)
min_cube_xy_goal_dist        (best xy distance reached)
xy_in_{100,50,25}mm_steps    (step counts inside successive radii)
xy_sustained_{100,50,25}mm_10 (cube spent ≥10 consecutive steps in goal)
```

## M6.2 — PullCube settle solver + xy/sustained metrics

Same idea applied to PullCube ([scripts/m6_generate_multitask_expert_v2.py](scripts/m6_generate_multitask_expert_v2.py)):
add a third pulling stage to bring the cube toward the goal_region center.

Stack of changes in v2:

- PushCube settle (from v1)
- PullCube settle (new in v2)
- `late_weight: 8` (from v1)
- `--ignore-termination` flag on eval so the policy can be observed past
  the moment native success fires

Result on PullCube + Pull (matched), 30 episodes, seed=3000:

| Metric                 | M6.1 (v1) | **M6.2 (v2)** |
| --                     | --        | --            |
| success_rate_once      | 0.30      | **0.43**      |
| mean_min_xy_dist       | 0.167     | **0.154**     |

### v0 → v1 → v2 scorecard

| Model | PickCube grasp | PushCube success | PullCube success |
| --    | --             | --               | --               |
| v0    | **0.30**       | 0.13             | 0.30             |
| v1    | 0.17           | **0.50**         | 0.30             |
| v2    | 0.00 / 0.13*   | 0.30             | **0.43**         |

(\*: with the `--force-grip-while-far` heuristic.)

Each variant is best at a different task. This is the textbook **multi-task
capacity-sharing trade-off** — a single shared MLP can't simultaneously
strengthen three task-specific late-phase behaviors when capacity is fixed.
The findings are documented in [docs/m6_1_settle.md](docs/m6_1_settle.md).

---

## M7 — Diffusion Policy on multi-task VLA dataset

The deterministic BC head was hitting a multi-task capacity ceiling (see M6.2).
M7 replaces it with a **Diffusion Policy** (Chi et al. 2023): the network
predicts noise on top of a noised action chunk instead of a single action.

```text
network = ConditionalUnet1D-style block stack with FiLM conditioning
input   = noisy_action_chunk[T_chunk, action_dim] + timestep + cond
cond    = MLP(concat(obs, lang_proj(lang_emb), image_proj(image_emb)))
loss    = MSE(eps_pred, eps) + aux_weight * CE(task_logits, task_id)
        (eps-prediction DDPM, squaredcos beta schedule, T=100)

inference = DDIM, receding horizon (sample 8 actions, execute K, replan)
```

Code:
[scripts/m7_train_diffusion_policy.py](scripts/m7_train_diffusion_policy.py),
[scripts/m7_eval_closedloop_diffusion.py](scripts/m7_eval_closedloop_diffusion.py),
[scripts/m7_record_diffusion_debug.py](scripts/m7_record_diffusion_debug.py),
[configs/m7_diffusion_v0.yaml](configs/m7_diffusion_v0.yaml),
[configs/m7_diffusion_v1.yaml](configs/m7_diffusion_v1.yaml).

### M7 v0 (200 epoch, action_exec=4, infer_steps=16)

```text
best_val_diffusion_mse : 0.0126
best_val_task_acc      : 1.000
```

Closed-loop versus M6.2 BC v2 (30 ep per cell, seed=3000, max_steps=120):

| env       | instr             | v2 BC | v0 M7  | notes                                          |
| --        | --                | --    | --     | --                                             |
| PickCube  | Pick (matched)    | 0.00  | 0.00   | grasp drops 0.13 → 0.00                        |
| PushCube  | Push (matched)    | 0.30  | 0.30   | parity                                         |
| PullCube  | Pull (matched)    | 0.43  | 0.30   | regression                                     |
| PickCube  | Push (swap)       | 0.00  | 0.00   | parity                                         |
| PullCube  | Pick (swap)       | 0.13  | 0.17   | slight instruction-following weakening         |

Diffusion v0 did not exceed BC v2 — likely under-trained and/or chunk
horizon mismatched for fine grasp behavior.

### M7.1 v1 (500 epoch, action_exec=2, infer_steps=24)

Tuned hyperparameters: longer training, more frequent receding-horizon
replanning (every 2 steps instead of every 4), more DDIM steps for higher
sampling quality.

```text
best_val_diffusion_mse : 0.0093   (vs v0: 0.0126; -26%)
best_val_task_acc      : 1.000
```

Closed-loop versus BC v2 and M7 v0:

| env       | instr             | v2 BC | v0 M7 | **v1 M7.1** |
| --        | --                | --    | --    | --          |
| PickCube  | Pick (matched)    | 0.00  | 0.00  | 0.00        |
| **PushCube** | **Push (matched)** | 0.30  | 0.30  | **0.47** ⭐  |
| PullCube  | Pull (matched)    | 0.43  | 0.30  | 0.23        |
| PickCube  | Push (swap)       | 0.00  | 0.00  | 0.00        |
| PullCube  | Pick (swap)       | 0.13  | 0.17  | 0.20        |

`mean_return` is consistently higher for the diffusion policy across every
cell (×1.5 to ×2.1 vs BC). The diffusion policy keeps interacting with the
cube through the whole episode, while the BC head tends to freeze after
a few steps.

### What this milestone showed

- ✅ Full **Diffusion Policy pipeline** (1D U-Net + DDPM training +
  DDIM receding-horizon inference + auxiliary task classification) works
  end-to-end on the multi-task VLA dataset.
- ✅ **First clean win over BC on PushCube** (30% → 47%) once trained long
  enough and tuned with frequent receding-horizon replanning.
- ✅ Diffusion produces more active policies (higher `mean_return`
  everywhere) — likely the multi-modal action distribution is keeping
  reasonable actions available even after the M6 BC would have collapsed
  to a static state.
- ❌ **No universal improvement**: M7.1 helps PushCube, hurts PullCube,
  PickCube grasp still flat. The multi-task capacity-sharing trade-off
  from M6.2 (each variant best at a different task) **is preserved**, just
  shifted: now M7.1 owns PushCube while BC v2 still owns PullCube and
  PickCube.

See [docs/m7_diffusion.md](docs/m7_diffusion.md) for the full design + analysis.

---

## M8 — Multi-task capacity-sharing fix comparison (4-way study)

Three independent fixes were applied to the multi-task VLA dataset and
compared head-to-head against the M6.2 BC v2 baseline and the M7.1
Diffusion policy, on the same 6-cell swap matrix.

```text
BC v2 (M6.2)       — deterministic MLP, no fix, multi-task BC
Diffusion (M7.1)   — MLP -> Conv1D diffusion U-Net, eps-prediction
PCGrad (M8a)       — Yu et al. 2020. Same architecture as BC v2; per-task
                     gradient projection at every optimizer step
PerTaskHead (M8b)  — Same shared trunk as BC v2; replace the single output
                     head with one head per task, selected by task_id
```

Code:
- [scripts/m8a_train_bc_pcgrad.py](scripts/m8a_train_bc_pcgrad.py)
- [scripts/m8b_train_bc_per_task_head.py](scripts/m8b_train_bc_per_task_head.py)
- [scripts/m8b_eval_closedloop_per_task.py](scripts/m8b_eval_closedloop_per_task.py)
- [configs/m8a_bc_pcgrad_v0.yaml](configs/m8a_bc_pcgrad_v0.yaml), [configs/m8b_bc_per_task_head_v0.yaml](configs/m8b_bc_per_task_head_v0.yaml)

### Open-loop training summary

| Model              | best_val_mse_norm | best_val_task_acc | best_epoch | notes                                  |
| --                 | --                | --                | --         | --                                     |
| BC v2 (M6.2)       | 0.0068            | 1.000             | 107        | baseline                               |
| Diffusion (M7.1)   | 0.0093 (eps)      | 1.000             | 381 / 500  | DDIM eps-MSE                           |
| PCGrad (M8a)       | **0.0063**        | 1.000             | 127        | + avg **95 task-gradient conflicts/epoch** detected |
| PerTaskHead (M8b)  | 0.0077            | 1.000             | 77         | best epoch hit early                   |

### Closed-loop comparison (matched-instruction cells, 30 ep, seed=3000, max_steps=120)

| Cell                              | BC v2   | Diff M7.1 | **PCGrad M8a** | **PerTaskHead M8b** | best                |
| --                                | --      | --        | --             | --                  | --                  |
| PickCube/Pick — grasp_once        | 0.13    | 0.00      | **0.20**       | 0.00                | **M8a**             |
| PickCube/Pick — placed_once       | 0.00    | 0.00      | 0.00           | 0.00                | tie                 |
| **PushCube/Push — success**       | 0.30    | **0.47**  | 0.30           | 0.40                | **Diffusion**       |
| **PullCube/Pull — success**       | **0.43**| 0.23      | 0.23           | 0.23                | **BC v2**           |
| PickCube/Push (swap) — grasp_once | 0.03    | 0.03      | 0.00           | 0.00                | M8a/M8b tie (obey)  |
| PullCube/Pick (swap) — success    | 0.13    | 0.20      | 0.10           | **0.00**            | **M8b** (obey)      |

### What each fix actually does

- **M8a (PCGrad) — gradient surgery, no architecture change**
  PCGrad detects ~95 task-pair gradient conflicts per epoch on this dataset
  and resolves them by projecting away the conflicting component. The
  immediate visible effect is **PickCube grasp recovery from 13% → 20%** —
  the weakest task in BC v2 gets back the capacity it had lost. The other
  two tasks come back to parity with BC v2.

- **M8b (Per-task heads) — output capacity decoupled, trunk still shared**
  Replacing the final linear head with three task-specific heads (selected
  by the task classifier on `lang_proj`) yields the **cleanest
  instruction-following purity**: PullCube + Pick(swap) drops 13% → 0%
  (the Pick head is selected and tries to grasp in the PullCube env, which
  correctly fails). PushCube matched climbs 30% → 40%. But PickCube grasp
  collapses to 0% — the shared trunk still re-allocates capacity away from
  PickCube. Per-task heads alone don't fix trunk-level interference.

- **M7.1 (Diffusion) — multi-modal action sampling**
  Brings the multi-modal action distribution into play. Wins PushCube
  (47%) cleanly and inflates mean_return × 1.5–2 across every cell, but
  pays for it on PullCube (23%) and PickCube grasp (0%).

- **BC v2 (M6.2) — deterministic baseline**
  Strongest on PullCube (43%). Each subsequent fix moves the trade-off,
  not the ceiling.

### The pattern

Every model is "best at one task and worst at another." The trade-off shape
shifts depending on the fix:

| Variant        | Best at                  | Worst at         |
| --             | --                       | --               |
| BC v2          | PullCube                 | PickCube grasp   |
| Diffusion M7.1 | PushCube                 | PullCube         |
| PCGrad M8a     | PickCube grasp           | swap-following   |
| PerTaskHead M8b| Instruction-obey purity  | PickCube grasp   |

This is the textbook multi-task capacity-sharing trade-off — and the M8
study shows it persists across **three structurally different fixes**:
optimization (PCGrad), architecture (PerTaskHead), and policy class
(Diffusion). At our 300-episode, ~300k-parameter scale **no single fix
removes it**.

### Implications for M9+

- The remaining lever that wasn't touched here is **trunk capacity itself**.
  Stacking M8a + M8b (gradient surgery + per-task heads) would address the
  same trunk; a larger trunk or a foundation-model backbone (OpenVLA / Octo
  LoRA) would actually grow the shared representation space.
- Open-loop validation MSE was not a reliable model-selection signal for
  closed-loop performance — M8b hit `best_val_mse_norm` at epoch 77 and
  this checkpoint was already PullCube-weak. Real closed-loop performance
  would need a search over checkpoints, not blind reliance on val MSE.

See [docs/m8_multitask_capacity.md](docs/m8_multitask_capacity.md) for the
full design rationale, PCGrad implementation notes, per-task head
architecture, and the references list.

---

## M9+ — Possible next steps

- **M9a: Stack PCGrad + PerTaskHead** — combine the two compatible M8
  fixes in a single model. The two interventions are orthogonal
  (optimization-side vs head-side) and may compound.
- **M9b: OpenVLA / Octo LoRA fine-tune** — grow the shared trunk by using
  a foundation policy backbone. Strongest portfolio narrative; GPU memory
  + inference latency to verify on RTX 3060.
- **M9c: Image-only VLA** — drop `state_obs`, force the policy to localize
  cube/goal from vision only.
- **M9d: Multi-target instructions** — within-task instruction variation
  (e.g. "left goal" vs "right goal") to test fine-grained conditioning.
- **M9e: Vision-language alignment loss** — contrastive objective on
  image_emb/lang_emb to reduce shortcut competition (cf. M6 trade-off).

---

## Video Artifacts

Expert and closed-loop videos can be generated with `--save-video`.

If the generated MP4 has compatibility issues, convert it with:

```bash
ffmpeg -y \
  -i input.mp4 \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  -an \
  output_compat.mp4
```

For slower playback:

```bash
ffmpeg -y \
  -i input.mp4 \
  -filter:v "setpts=2.0*PTS" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  -an \
  output_slow.mp4
```

If files are generated by a root Docker container, fix host-side ownership with:

```bash
sudo chown -R "$USER:$USER" outputs runs
```

---

## Key Takeaways

This project currently demonstrates:

```text
1. Dockerized GPU-enabled robotics simulation
2. Language-conditioned manufacturing task metadata
3. Expert demonstration generation
4. Step-level dataset construction
5. Behavior cloning training
6. Open-loop prediction evaluation
7. Closed-loop rollout evaluation
8. Failure diagnosis through video and logs
9. Iterative policy improvement planning
```

Current technical conclusion:

> Single-task BC + DAgger on PickCube-v1 plateaus at closed-loop success ≈ 0% despite
> 36.7% end-grasp retention (with the force_grip inference heuristic).
> This is consistent with the well-known limitation of pure BC under distribution shift
> and the fact that PickCube has a tight place tolerance.
> The chosen BC base for downstream work is **DAgger v1 + force_grip**.
>
> The project's true target is a **language-conditioned VLA-style policy**.
> The next milestone (M4) extends the observation with instruction embeddings,
> establishing the language-conditioning code path on top of the existing BC base.
> M5+ will add multi-task variants where the instruction becomes a real action differentiator.

---

## Portfolio Summary

One-line summary:

> Built a Dockerized ManiSkill-based manufacturing VLA PoC that connects language-conditioned task metadata, expert demonstration generation, step-level dataset logging, behavior cloning, closed-loop evaluation, and policy failure diagnosis.

Korean summary:

> Docker 기반 ManiSkill 환경에서 제조형 언어 지시 로봇 조작 task를 구성하고, expert demonstration 생성, 데이터셋 변환, behavior cloning 학습, closed-loop 평가 및 실패 분석까지 연결하는 VLA-style 로봇러닝 PoC를 구축했습니다.
