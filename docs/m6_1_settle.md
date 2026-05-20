# M6.1 / M6.2 — Settle solvers and the multi-task capacity trade-off

The M6 success rate numbers (PushCube 13%, PullCube 30%) hid two things
that only became visible when we watched the videos:

1. ManiSkill's `success_rate_once` for cube-displacement tasks fires the
   moment the cube center crosses the goal-region boundary
   (`xy_distance < 0.1`). The env then terminates. The cube never settles.
2. PickCube's `is_obj_placed` is also distance-only; a single frame within
   `0.025 m` flips it, even if the policy never grasped the cube.

So the M6 success rates were measuring **boundary-tap rate**, not real
manipulation. The expert demos themselves carried that limitation for
PushCube (the stock motion-planning solution stops at the boundary).

M6.1 and M6.2 attack the problem from both ends — the expert side
(richer demos), the eval side (stricter metrics), and the loss side
(stronger late-phase weighting).

## Stricter eval metrics

Added in [scripts/m6_eval_closedloop_vla.py](../scripts/m6_eval_closedloop_vla.py):

```text
final_cube_xy_goal_dist          xy distance at the last logged step
min_cube_xy_goal_dist            best xy distance reached during the episode
xy_in_{100,50,25}mm_steps        step counts inside each threshold
xy_sustained_{100,50,25}mm_10    True if ≥10 consecutive steps with xy_dist ≤ threshold
--ignore-termination             do not break when env terminated/truncated fires;
                                 keep stepping until max_steps so the policy can
                                 be observed past the moment native success fires
```

The "sustained" metrics are designed to ignore the boundary-tap shortcut:
the cube has to stay inside the goal region for at least ~0.33 s, not
just touch it.

## Expert side fixes

### M6.1 — PushCube settle solver

Stock `mani_skill/.../push_cube.solve` ends with the tcp at
`goal_region.x - 0.12`. With cube half-width 0.02 the cube ends up
just inside the boundary. M6.1 adds a third planning stage to push
further:

```python
# stage 3 (new): tcp goes to goal_region.x - 0.06 → cube ends near goal center
settle_pose = sapien.Pose(p=env.goal_region.pose.sp.p + np.array([-0.06, 0, 0]),
                          q=env.agent.tcp.pose.sp.q)
planner.move_to_pose_with_screw(settle_pose)
```

Code: [scripts/m6_generate_multitask_expert_v1.py](../scripts/m6_generate_multitask_expert_v1.py).

### M6.2 — PullCube settle solver

Stock `pull_cube.solve` ends at `goal_region.x + 0.05`. M6.2 adds the
analogous stage that pulls further:

```python
# stage 3 (new): tcp goes to goal_region.x - 0.01 → cube ends near goal center
settle_pose = sapien.Pose(p=env.goal_region.pose.sp.p + np.array([-0.01, 0, 0]),
                          q=env.agent.tcp.pose.sp.q)
planner.move_to_pose_with_screw(settle_pose)
```

Code: [scripts/m6_generate_multitask_expert_v2.py](../scripts/m6_generate_multitask_expert_v2.py).

PickCube uses the stock solver unchanged.

## Training side

`late_weight` in the phase-weighted MSE loss was raised from **4 → 8**:

```yaml
# configs/m6_vla_aux_v1.yaml and configs/m6_vla_aux_v2.yaml
loss:
  late_weight: 8.0
```

The intention is to make the BC head fit the new settle stage rather
than treat it as noise relative to the early reach/approach.

## Results

### v0 → v1 → v2 scorecard (matched-instruction cells, 30 ep, seed=3000)

| Model       | PickCube grasp | PushCube success | PullCube success |
| --          | --             | --               | --               |
| M6  (v0)    | **0.30**       | 0.13             | 0.30             |
| M6.1 (v1)   | 0.17           | **0.50**         | 0.30             |
| M6.2 (v2)   | 0.00 / 0.13*   | 0.30             | **0.43**         |

(\*: with `--force-grip-while-far` heuristic.)

### What each milestone fixed

| Milestone | Verified by                                                                                  |
| --        | --                                                                                           |
| v0 → v1   | PushCube success jumps 13% → 50%; debug video shows cube driven to `min_xy = 0.027`.        |
| v1 → v2   | PullCube success 30% → 43%; `min_xy` 0.167 → 0.154. PushCube regresses 50% → 30%.            |
| v2 force_grip | PickCube grasp 0% → 13%; placed 0% → 3%. Still far below pre-multi-task BC (~ 50%).       |

### Debug videos (recorded with `--ignore-termination`)

- `runs/m6_vla_aux_v1/debug_video/PushCube-v1_seed_3029/...mp4`
  — cube pushed past the boundary to `min_xy ≈ 0.027`.
- `runs/m6_vla_aux_v2/debug_video/PullCube-v1_seed_3024/...mp4`
  — cube pulled into the goal region, `min_xy ≈ 0.101` (still on boundary).
- `runs/m6_vla_aux_v2/debug_video/PushCube-v1_seed_3029/...mp4`
  — v2's push competence is preserved (`min_xy ≈ 0.025`).

## The trade-off: multi-task capacity sharing

The scorecard shows a clean pattern: **each variant is best at a different
task**, and improving one task tends to regress another.

| Variant | Best at      | Worst at         |
| --      | --           | --               |
| v0      | PickCube     | PushCube         |
| v1      | PushCube     | PickCube         |
| v2      | PullCube     | PickCube         |

This is the textbook multi-task BC capacity-sharing problem: a shared MLP
trunk has finite capacity, and pushing it harder on one task's late-phase
behavior (via expert modification or `late_weight` boost) eats into
capacity that previously served other tasks. The instruction signal
(`lang_emb`) plus the auxiliary `task_id` head from M5.1 provide enough
context to select between tasks at runtime, but they don't enlarge the
trunk that has to fit all three terminal behaviors.

### What would actually fix this

- **Per-task output heads** (a small head selected by `task_id` after the
  shared trunk). This is M7d in the README.
- **Larger trunk + more demos**. Likely effective; expensive.
- **Off-line RL fine-tuning** on the trunk for late-phase robustness.

For PoC purposes the current BC ceiling is honest: a single shared
multi-task language-and-vision-conditioned BC policy hits real-task
success rates that depend on which slice of the data you optimize for.

## Things that did *not* work

- `force_grip_while_far` on M6.2 PickCube: grasp 0% → 13%, placed 0% → 3%.
  The heuristic still helps (some grasp competence is recovered), but the
  underlying capacity loss from multi-task training is the dominant
  bottleneck on PickCube. M3.9H force_grip on M5.1 (state-only, no vision)
  yielded `final_grasped = 37%`; the gap is the multi-task cost, not the
  heuristic.

## Suggested M7 directions (in order of value)

1. **M7d — per-task heads** to escape the capacity-sharing ceiling
   without giving up the shared VLA conditioning.
2. **M7c — multi-target instructions** to test finer-grained
   instruction following on top of the v2 base.
3. **M7a — image-only VLA** to verify whether vision can shoulder
   state's role under the same multi-task setting.
