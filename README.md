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
M2B    Completed — Expert demonstration generation
M2C    Completed — Expert H5 to M3-ready dataset conversion
M3.0   Completed — State-only behavior cloning training
M3.1   Completed — Open-loop action prediction evaluation
M3.2   Completed — Closed-loop rollout evaluation
M3.3A  Completed — Closed-loop failure diagnosis with video/logs
M3.3L  Completed — Longer state-only BC training experiment
M3.4   Planned   — Phase-aware behavior cloning
```

Summary:

> The project has reached a working simulation-to-dataset-to-BC-training-to-closed-loop-evaluation pipeline.  
> The current state-only BC policy learns expert actions in open-loop evaluation but fails to reliably complete closed-loop grasping.  
> The next step is phase-aware BC using progress and previous-action conditioning.

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

  docker/
    Dockerfile

  scripts/
    m0_random_rollout.py
    m1_language_task_rollout.py
    m2_collect_step_dataset.py
    m2_inspect_dataset.py
    m2_generate_pickcube_expert.py
    m2_inspect_expert_h5.py
    m2_convert_expert_h5_to_dataset.py
    m3_train_bc_state.py
    m3_eval_openloop_bc_state.py
    m3_eval_closedloop_bc_state.py
    m3_record_closedloop_debug.py

  docs/
    m2_dataset_report.md

  outputs/
    m0_random_rollout/
    m1_language_task_rollout/
    m2_step_dataset/
    m2_expert_demos/
    m2_expert_dataset/
    m2_expert_demos_100/
    m2_expert_dataset_100/

  runs/
    m3_bc_state/
    m3_bc_state_long/
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

## M3.4 — Planned: Phase-Aware Behavior Cloning

The next milestone is phase-aware BC.

Current observation:

```text
obs = state_57
```

Planned observation:

```text
obs = state_57 + progress_1 + prev_action_8
```

Expected dimension:

```text
obs_dim = 66
```

Where:

```text
progress = t / T
prev_action = action_{t-1}
```

Motivation:

```text
The expert trajectory is phase-dependent:
approach → align → grasp → lift → move → place

A pure state-only MLP has difficulty identifying the current phase.
Adding progress and previous action should reduce phase ambiguity.
```

Planned steps:

```text
M3.4A — Convert expert dataset into phase-aware dataset
M3.4B — Train phase-aware BC
M3.4C — Open-loop evaluation
M3.4D — Closed-loop evaluation
M3.4E — Debug video comparison
```

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

> State-only BC can learn expert actions in open-loop evaluation, but it is not robust enough for closed-loop grasping.  
> The next improvement is phase-aware BC with progress and previous-action conditioning.

---

## Portfolio Summary

One-line summary:

> Built a Dockerized ManiSkill-based manufacturing VLA PoC that connects language-conditioned task metadata, expert demonstration generation, step-level dataset logging, behavior cloning, closed-loop evaluation, and policy failure diagnosis.

Korean summary:

> Docker 기반 ManiSkill 환경에서 제조형 언어 지시 로봇 조작 task를 구성하고, expert demonstration 생성, 데이터셋 변환, behavior cloning 학습, closed-loop 평가 및 실패 분석까지 연결하는 VLA-style 로봇러닝 PoC를 구축했습니다.
