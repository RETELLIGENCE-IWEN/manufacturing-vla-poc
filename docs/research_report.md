# Manufacturing VLA PoC — Research Report

**Project:** Manufacturing-style language-conditioned manipulation
**Scope:** M0 → M8 (Dockerized ManiSkill bring-up → 4-way multi-task capacity study)
**Hardware:** Single RTX 3060, Ubuntu 22.04, ManiSkill / PickCube–PushCube–PullCube
**Status:** All planned milestones complete; M9+ design space mapped
**Companion artifacts:** [README.md](../README.md), per-milestone notes in [docs/](.)

---

## Part I — Executive Summary

### One-paragraph statement

This PoC builds a complete language-conditioned manipulation pipeline on a single consumer GPU and uses it to investigate **how each layer of a VLA-style policy contributes to closed-loop success on a multi-task manipulation benchmark.** Starting from state-only behavior cloning and ending at a 4-way comparison (BC / Diffusion / PCGrad / Per-task heads) on a shared state+text+vision dataset, the project reaches three substantive findings: (1) **CLIP-text becomes a real action differentiator only with an auxiliary task-classification objective on the language projection** (M5.1); (2) **adding CLIP-vision improves spatial competence but dilutes instruction following** (M6); and (3) at the 300-episode / ~300k-parameter scale studied here, **no single structural fix removes the multi-task capacity-sharing trade-off** — each variant ends up best at one task and worst at another (M8).

### Headline numbers

| Stage | Best result it produced |
| -- | -- |
| State-only BC + DAgger v1 + force_grip (M3.9H) | grasped_once 47%, final_grasped 37%, mean_return 24.1, closed-loop success 0% |
| Multi-task BC + CLIP-text (M5) | First non-zero closed-loop success (PushCube 10%); instruction not yet used as differentiator |
| + auxiliary task classification loss (M5.1) | PullCube 0%→33%, PushCube 10%→20%; grasp drops to 0% when instruction is swapped (genuine instruction following) |
| + CLIP-vision tower (M6 / M6.1 / M6.2) | PickCube grasp 7%→30%; PushCube success 13%→50% after expert-settle + late_weight tuning; PullCube 30%→43% |
| Diffusion policy on the same dataset (M7.1) | PushCube success 30%→47%; mean_return 1.5–2× vs BC across all cells |
| PCGrad gradient surgery (M8a) | PickCube grasp 13%→20% (recovers the weakest task without changing architecture) |
| Per-task output heads (M8b) | Cleanest instruction-following purity (swap PullCube+Pick: 13%→0%) |

### Three things the project actually shows

1. **Instruction following is not free.** With multi-task BC alone (M5) the policy ignores CLIP-text and decides from state. Adding a one-line auxiliary classification objective on the language projection (M5.1) turns the instruction into a load-bearing input. The diagnostic was a **swap matrix**, not validation MSE.
2. **Vision is a double-edged modality.** M6 lifted PickCube grasp 7%→30% via image conditioning, but swapping the instruction in the same state began producing partial grasps (0%→10%). With two strong modalities and a deterministic BC head, gradient descent exploits whichever is locally easier.
3. **Capacity-sharing is the load-bearing constraint, not the policy class.** Diffusion (M7), PCGrad gradient surgery (M8a), and per-task heads (M8b) each *moved* the trade-off — diffusion owns PushCube, PCGrad recovers PickCube grasp, per-task heads give the cleanest swap purity — but none of them removed it. At this scale, capacity is the lever.

### What this proves about the engineer

A complete VLA-style stack — Dockerized simulation, expert demonstration generation, step-level dataset, CLIP text + vision integration, multiple training objectives (BC weighted-MSE, auxiliary CE, eps-prediction DDPM), receding-horizon DDIM inference, gradient-conflict resolution (PCGrad), per-task head selection, closed-loop swap-matrix evaluation, video failure diagnosis — assembled, ablated, and analyzed end-to-end on a single consumer GPU.

---

## Part II — Research Report

### 1. Introduction

#### 1.1 Motivation

Vision-Language-Action (VLA) models are the dominant direction in robot learning, but most VLA literature reports results at foundation-model scale (hundreds of millions of parameters, datacenter training, multi-camera demonstrations). This project asks the inverse question: **on a single consumer GPU with ~300 expert demonstrations per task, which layer of the VLA stack actually contributes to closed-loop success, and which is dead weight?**

The PoC is framed in manufacturing terms ("pick the bolt-like part and place it at the left fixture") to suggest a near-term application target, but the technical substrate is generic — ManiSkill cube manipulation with a Panda arm.

#### 1.2 Contributions

1. A reproducible Dockerized pipeline from ManiSkill bring-up to closed-loop swap-matrix evaluation, runnable on a single RTX 3060 (M0–M2C).
2. A complete BC → DAgger → language-conditioned BC → VLA progression, with the design rationale for each step grounded in **observed failure modes** (M3.0–M6.2).
3. An auxiliary-classification trick (M5.1) that converts CLIP-text from a dummy input into a load-bearing action differentiator, demonstrated by a 6-cell swap matrix rather than open-loop MSE.
4. A 4-way multi-task capacity-sharing study (M8) comparing BC, Diffusion, PCGrad, and per-task heads on a common dataset — the same dataset, the same eval matrix, the same seeds — and the observation that **the trade-off pattern persists across three structurally different fixes.**

#### 1.3 Scope and non-goals

In scope: behavior cloning, DAgger, language-conditioned BC, VLA (state+text+vision), diffusion policy, gradient-surgery and architectural fixes for multi-task interference, and closed-loop evaluation with strict metrics (xy distance, sustained-in-region step counts).

Explicit non-goals: foundation-model fine-tuning, real-robot deployment, sim-to-real transfer, multi-camera setups, vision-only policies (M9c is mapped but unrun).

### 2. Problem Setup

#### 2.1 Tasks

Three ManiSkill cube-manipulation tasks selected because they share the same state observation layout (panda 31 + cube 13 + goal 13 = 57) but require **structurally different end-phase behaviors**:

- **PickCube-v1** — grasp the cube and lift it to a `goal_site`. Strict placement tolerance (`is_obj_placed` ≤ 0.025 m).
- **PushCube-v1** — push the cube horizontally into a `goal_region`. Native success criterion is lax (boundary crossing).
- **PullCube-v1** — pull the cube toward the agent into a `goal_region`. Also lax native success criterion.

The lax native success criteria in PushCube/PullCube became a methodological finding in their own right (see §6.2).

#### 2.2 Observation and action spaces

```text
state_obs        = panda(31) + cube(13) + goal(13)                = 57
phase_aware obs  = state_obs + progress(1) + prev_action(8)        = 66
lang_emb         = CLIPTextModel("clip-vit-base-patch32").pooled    = 512
image_emb        = CLIPVisionModel.pooled (224x224 RGB)             = 768
action           = pd_joint_pos (7 joints + gripper)                = 8
```

`progress = step_idx / (phase_horizon − 1)` and `prev_action` were added in M3.4 after the state-only BC was diagnosed as failing because of phase ambiguity (approach vs. align vs. grasp vs. lift vs. transfer vs. place all look similar in instantaneous state).

#### 2.3 Evaluation protocol

All closed-loop numbers in this report use the same protocol:

- 30 episodes per cell
- seed = 3000
- `phase_horizon = 80`, `max_steps = 120`
- `--expert-action-bounds` (M3.5 safe-action filter) always on
- `--ignore-termination` for VLA evaluation (M6.2) so policies are observed past the moment native success fires
- For multi-task models, a **6-cell swap matrix** of (env, instruction) is reported, not just matched-instruction success — this is what surfaces dummy-instruction policies

Strict metrics added in M6.1/M6.2 beyond ManiSkill's native `success`:

```text
final_cube_xy_goal_dist
min_cube_xy_goal_dist
xy_in_{100, 50, 25}mm_steps
xy_sustained_{100, 50, 25}mm_10   (≥10 consecutive in-region steps)
```

### 3. Related Work

- **Behavior Cloning baseline** — Pomerleau (1989); Ross et al., DAgger (2011). Used as the M3 baseline and M3.9 correction loop.
- **Phase-aware BC / temporal conditioning** — internal extension; closest published work is Florence et al.'s implicit BC (2021) and Mandi et al.'s phase-conditioned policies. Here `progress` is appended explicitly rather than learned.
- **Language-conditioned BC** — RT-1 (Brohan et al. 2022), CLIPort (Shridhar et al. 2022). This project uses frozen CLIP text features (no vision-language joint encoder), closer to BC-Z (Jang et al. 2022) than to fully cross-attention conditioned policies.
- **VLA stacks** — RT-2 (Brohan et al. 2023), OpenVLA (Kim et al. 2024), Octo (Octo team 2024). This project's M6 is a faithful but minimal implementation of the architectural pattern (frozen CLIP towers + small projection heads + shared trunk), not a foundation-model fine-tune.
- **Diffusion policies** — Chi et al. (2023). M7 implements the 1D conditional U-Net + DDPM training + DDIM receding-horizon inference recipe with the M6 conditioning vector as `cond`.
- **Multi-task gradient methods** — PCGrad (Yu et al. 2020). Implemented in M8a.
- **Multi-task architectures** — per-task heads on a shared trunk; classical multi-task learning (Caruana 1997) revisited in M8b.

### 4. Methodology

#### 4.1 Data pipeline (M0–M2C)

```text
ManiSkill PickCube/PushCube/PullCube
  → motion-planning expert (project-local PickCube/PushCube/PullCube solvers,
    because the installed ManiSkill MP runner had an import mismatch)
  → HDF5 trajectories
  → step-level NPZ dataset (panda/cube/goal state + action + instruction
    + task metadata + success)
  → phase-aware augmentation (progress, prev_action)
  → CLIP-text embeddings (frozen pooler output, per episode)
  → CLIP-vision embeddings (frozen pooler output, per step,
    via deterministic replay of expert trajectories at the recorded seed)
```

PushCube and PullCube expert solvers were extended in M6.1/M6.2 with a third "settle" motion-planning stage so the cube ends near the goal-region center rather than at the boundary; the resulting v1/v2 datasets carry richer late-phase supervision.

#### 4.2 Policy architectures

All policies share the M5.1 `LangBCPolicy` skeleton with progressively more inputs and progressively different action heads:

```text
state_obs (57) ─┐
progress (1)   ─┼─→ concat → obs (66) ─────────────┐
prev_action(8)─┘                                   │
                                                   ├─→ trunk (MLP) ──→ head ──→ action (8)
lang_emb (512) ──→ lang_proj (64) ─→ task_head ────┘
                                  (auxiliary CE)
image_emb(768) ──→ image_proj (128) (M6+) ─────────┘
```

Variants studied:

| Variant | Action head | Notes |
| -- | -- | -- |
| M3 BC | trunk → 8-dim MLP | state only |
| M3.4–M3.9 BC | trunk → 8-dim MLP | + progress, prev_action, DAgger aggregation, force_grip heuristic |
| M4 / M5 lang BC | + frozen CLIP text | aux loss off; instruction is dummy |
| M5.1 lang BC | + aux task CE on lang_proj | instruction becomes load-bearing |
| M6 VLA | + frozen CLIP vision | aux CE retained on `lang_proj` only |
| M7 Diffusion | 1D conditional U-Net with FiLM cond | eps-prediction DDPM, DDIM receding-horizon inference (T=100/24, K=2) |
| M8a PCGrad | same as BC v2 | per-task gradient projection at every optimizer step |
| M8b Per-task heads | trunk → 3 separate output heads | head selected by task classifier on lang_proj |

#### 4.3 Training objectives

```text
BC weighted MSE     : MSE(action) with phase weighting   early:mid:late = 1:2:4 (M3.8, M4, M5)
                                                             1:2:8 (M6.1, M6.2, M8)
Auxiliary task CE   : CE(task_logits, task_id) with weight = 1.0 (M5.1, M6, M8)
Diffusion eps-MSE   : MSE(eps_pred, eps) on a noised action chunk + aux CE (M7)
PCGrad              : project g_i ← g_i − sum_j (g_i · g_j_hat / ||g_j||) when g_i·g_j < 0 (M8a)
```

#### 4.4 Inference-time heuristics

- **Safe-action filter (M3.5)** — clip every output action to the per-dimension `(min, max)` envelope of the expert dataset (with a small margin). Prevents the policy from drifting into out-of-distribution joint commands.
- **Final-hold wrapper (M3.6)** — once the cube reaches the goal region, latch the previous action. Suppresses post-placement push-throughs.
- **force_grip_while_far (M3.9H)** — once the policy has ever closed the gripper on the cube and the cube is still far from the goal, force `action[-1] = −1.0` (closed). Roughly doubles end-state grasp retention in the single-task BC base.

### 5. Experiments and Results

#### 5.1 Single-task BC saturation (M3 line)

The first part of the project sought to find out how far pure BC could go on PickCube. The answer: not far enough.

| Variant | grasped_once | final_grasped | success_once | mean_return |
| -- | -- | -- | -- | -- |
| State-only BC (M3.0) | low | — | 0.00 | +6.65 vs random |
| Phase-aware BC, 100 demos (M3.4) | 0.567 | — | 0.000 | 14.56 |
| + final-hold (M3.6) | 0.567 | — | **0.033** | — |
| Phase-aware BC, 5000 demos (M3.7) | — | — | 0.000 | — |
| + phase-weighted loss 1:2:4 (M3.8) | — | — | 0.000 | — |
| + DAgger v1 (M3.9) | 0.467 | 0.167 | 0.000 | 20.96 |
| + force_grip heuristic (M3.9H) | **0.467** | **0.367** | 0.000 | **24.10** |

Findings:

- Longer training (M3.3L) and more data (M3.7) did not break the success-rate ceiling. The ceiling is not a data-quantity problem.
- The state-only policy cannot disambiguate phases from instantaneous state alone; **phase-aware features were the largest single improvement** in the M3 line (grasped_once 0 → 56.7%).
- DAgger v1 improves trajectory shape and end-stability but does not push success past zero on PickCube.
- The dominant remaining failure mode (revealed by debug video at M3.9H) is **premature gripper release at progress ≈ 0.95**, which the `force_grip_while_far` heuristic patches. The fact that an inference-time rule helps this much is itself a sign that BC capacity, not BC training, is the bottleneck.
- **DAgger v2 (release-phase biased selection) underperformed v1**, indicating that the v1 selection rule (`deteriorated_after_min_dist`) was load-bearing for cube-approach behavior. v1 + force_grip is the chosen BC base for the rest of the project.

#### 5.2 Language conditioning becomes load-bearing only with the aux loss (M4 → M5 → M5.1)

| Stage | best_val_mse_norm | PickCube grasp matched | PickCube grasp swapped | PushCube success matched | PullCube success matched |
| -- | -- | -- | -- | -- | -- |
| M4 single-task lang BC | 0.00584 | 0.500 | n/a | n/a | n/a |
| M5 multi-task lang BC | 0.00426 | 0.10 | **0.10** (dummy) | 0.10 | 0.00 |
| **M5.1 + aux CE on lang_proj** | **0.00384** | 0.07 | **0.00** ⭐ | **0.20** | **0.33** ⭐ |

Key observations:

- M4 wires the language pathway without breaking the action policy (success and grasp parity vs. M3 single-task BC). This is the necessary sanity check before multi-task.
- **M5 trains successfully but ignores the instruction.** Swapping `Pick → Push → Pull` in the PickCube env barely changes grasp rate (10% → 10% → 7%). The policy resolves the action from state.
- **M5.1's auxiliary task-classification loss** forces `lang_proj` to be linearly separable by task and turns the instruction into a load-bearing signal: PickCube grasp drops 10%/7% → 0% under instruction swap, **and** matched-instruction success jumps on the other two tasks (PullCube 0% → 33%).
- Validation `task_acc = 1.000` from epoch ~50 onward; the model achieves *perfect* instruction classification long before its actions become instruction-conditional in closed-loop. **Open-loop accuracy is not a sufficient signal — swap-matrix grasp/success is.**

#### 5.3 Vision adds spatial competence and dilutes instruction following (M6 → M6.1 → M6.2)

| Cell | M5.1 grasp/success | **M6 v0** | M6.1 (PushCube settle, late_weight 8) | M6.2 (+ PullCube settle, ignore-termination) |
| -- | -- | -- | -- | -- |
| PickCube/Pick grasp | 0.07 | **0.30** ⭐ | 0.17 | 0.00 (0.13 with force_grip) |
| PickCube/Push (swap) grasp | 0.00 | 0.07 | — | — |
| PushCube/Push success | 0.20 | 0.13 | **0.50** ⭐⭐ | 0.30 |
| PullCube/Pull success | **0.33** | 0.30 | 0.30 | **0.43** |

Key observations:

- Adding CLIP-vision (M6 v0) quadruples PickCube grasp (7% → 30%) — vision provides spatial localization the policy was previously inferring from joint angles alone.
- But under an instruction-swap in the PickCube env, **grasp goes from 0% → 7–10%**: the image now says "cube is there" loudly enough that the policy partly overrides the instruction. PushCube matched success drops 20% → 13%.
- M6.1 adds an expert-side fix: PushCube's motion-planning expert is extended with a third stage that pushes past the goal-region boundary toward the center. Pushing `late_weight` to 8 in the BC objective doubles PushCube matched success (13% → 50%).
- M6.2 applies the same expert-settle fix to PullCube (30% → 43%) and adds strict xy/sustained metrics. Watching expert videos revealed that the native ManiSkill success criterion fires the instant the cube center crosses the goal-region boundary — *the policy that triggers `success=True` may still be holding a cube on the line, not settled.* Strict metrics replace this.
- **No single VLA variant is best on all three tasks** (v0 best on PickCube grasp, v1 best on PushCube, v2 best on PullCube). This is the first appearance of the multi-task capacity-sharing trade-off in this project.

#### 5.4 Diffusion policy vs. deterministic BC (M7 → M7.1)

| Cell | BC v2 (M6.2) | M7 v0 (200 ep, K=4, T=16) | **M7.1 v1 (500 ep, K=2, T=24)** |
| -- | -- | -- | -- |
| PickCube/Pick — grasp | 0.13 | 0.00 | 0.00 |
| **PushCube/Push — success** | 0.30 | 0.30 | **0.47** ⭐ |
| PullCube/Pull — success | **0.43** | 0.30 | 0.23 |
| PickCube/Push (swap) — grasp | 0.03 | 0.00 | 0.00 |
| PullCube/Pick (swap) — success | 0.13 | 0.17 | 0.20 |

Key observations:

- **M7 v0 (under-trained) did not exceed BC.** This is informative: a diffusion policy is not free; chunk horizon, replan frequency, and DDIM step count all matter.
- **M7.1 (more training + more frequent replanning + more DDIM steps) wins PushCube cleanly** (30% → 47%), but PullCube drops (43% → 23%) and PickCube grasp stays flat (0%).
- `mean_return` is consistently **1.5–2× higher** for diffusion than for BC across every cell — the multi-modal action distribution keeps reasonable actions available even when the deterministic BC head would have frozen.
- The trade-off pattern from M6.2 is **preserved, not removed.** It has shifted: diffusion now owns PushCube while BC v2 still owns PullCube and PickCube.

#### 5.5 Three independent fixes, same trade-off (M8)

The M8 study runs three structurally different fixes — optimization-side (PCGrad), policy-class (Diffusion), and architecture-side (per-task heads) — on the **same** multi-task VLA dataset, with the **same** evaluation matrix, against the **same** BC v2 baseline.

Open-loop training summary:

| Model | best_val_mse_norm | best_val_task_acc | notes |
| -- | -- | -- | -- |
| BC v2 | 0.0068 | 1.000 | baseline |
| Diffusion M7.1 | 0.0093 (eps-MSE) | 1.000 | not directly comparable to BC MSE |
| **PCGrad M8a** | **0.0063** | 1.000 | + **~95 task-gradient conflicts per epoch** detected and resolved |
| Per-task heads M8b | 0.0077 | 1.000 | best epoch hit early (77/500) |

Closed-loop matched-instruction comparison:

| Cell | BC v2 | Diffusion M7.1 | **PCGrad M8a** | **PerTaskHead M8b** | Best |
| -- | -- | -- | -- | -- | -- |
| PickCube/Pick — grasp | 0.13 | 0.00 | **0.20** | 0.00 | M8a |
| PushCube/Push — success | 0.30 | **0.47** | 0.30 | 0.40 | Diffusion |
| PullCube/Pull — success | **0.43** | 0.23 | 0.23 | 0.23 | BC v2 |
| PickCube/Push (swap) — grasp | 0.03 | 0.03 | **0.00** | **0.00** | M8a/M8b (purity) |
| PullCube/Pick (swap) — success | 0.13 | 0.20 | 0.10 | **0.00** | M8b (purity) |

The four-way pattern:

| Variant | Best at | Worst at |
| -- | -- | -- |
| BC v2 | PullCube | PickCube grasp |
| Diffusion M7.1 | PushCube | PullCube |
| PCGrad M8a | PickCube grasp | swap-following |
| Per-task heads M8b | Instruction-obey purity | PickCube grasp |

### 6. Discussion

#### 6.1 The capacity-sharing constraint

The M8 study is the project's strongest empirical finding: **at 300-episode / ~300k-parameter scale, the multi-task trade-off persists across three structurally different fixes.** PCGrad eliminates gradient conflicts (∼95 per epoch are detected). Per-task output heads eliminate output-side interference. Diffusion changes the action class. None remove the pattern — they only shift which task wins.

The control variable is the **shared trunk**. PCGrad acts on the optimizer, leaving the trunk's parameter count unchanged. Per-task heads leave the trunk shared. Diffusion changes the head class, again leaving the trunk untouched. The remaining lever — **trunk capacity** itself — is exactly M9a (stacking PCGrad + per-task heads still does not address it) and M9b (foundation-model backbone).

#### 6.2 Open-loop MSE is not closed-loop performance

Multiple times in the project the open-loop validation signal was misleading:

- **M5 had a lower `best_val_mse_norm` than M4** (0.00426 vs 0.00584) but the instruction was still dummy.
- **M5.1's `val_task_acc` hit 1.000 long before** the policy's actions became instruction-conditional in closed-loop.
- **M8b's best `val_mse_norm` was at epoch 77**, and that checkpoint was already PullCube-weak.

The diagnostic that consistently worked was the **closed-loop swap matrix**. Open-loop MSE measures point-wise action accuracy; instruction following is a *distributional* property that only shows up when the action distribution shifts as the instruction shifts. For VLA model selection, this is the right level of abstraction.

#### 6.3 Hidden simulator pitfalls

ManiSkill's native `success` flags fire on instantaneous distance thresholds (e.g., PushCube fires when `xy_distance < 0.1` regardless of whether the cube is settled or moving through the region). Two episodes can both report `success=True` while having structurally different end states. M6.1 added strict xy / sustained-in-region metrics; M6.2 added an `--ignore-termination` evaluation flag so the policy could be observed past the moment native success fires. **Without these changes, several of the comparison cells in M6/M7/M8 would have been measuring noise.**

#### 6.4 Inference heuristics as a capacity diagnostic

`force_grip_while_far` (M3.9H) is technically a hack: a 5-line rule appended at policy output time. The reason it almost doubles end-state grasp retention (16.7% → 36.7%) is that the underlying policy *has* learned how to grasp; it just also learned the spurious correlation "open the gripper at progress ≈ 0.95." The size of the heuristic's effect is itself a measurement of how much capacity is being spent on the wrong correlation — and it predicted, correctly, that M5+ would need more than additional training data on the BC head.

### 7. Limitations

- **Scale.** ~300 episodes per task, single Panda arm, single camera angle, ~300k trainable parameters in the BC head + projections. No foundation-model backbone.
- **Frozen CLIP encoders.** All language/vision features come from frozen `clip-vit-base-patch32`. The aux task-classification loss steers `lang_proj`, but the CLIP towers themselves do not adapt.
- **No real-robot validation.** All numbers are simulator closed-loop. The PoC was scoped this way intentionally.
- **State-conditioned vision.** M6 still feeds `state_obs` alongside the image; M9c (image-only) is not yet run, so the contribution of vision in isolation is undermeasured.
- **Open-loop checkpoint selection.** Best checkpoints were selected by `best_val_mse_norm`, which §6.2 documents as misleading. A closed-loop checkpoint search would likely change several of the closed-loop numbers above.

### 8. Future Work (M9+)

Mapped but not yet implemented:

- **M9a — Stack PCGrad + Per-task heads.** Two compatible M8 fixes; the interventions are orthogonal (optimizer-side vs head-side) and may compound. This is the cheapest experiment and directly addressable on the existing dataset.
- **M9b — OpenVLA / Octo LoRA fine-tune.** Grow the shared trunk by using a foundation-policy backbone. The strongest single lever against the capacity-sharing constraint. Needs to verify RTX 3060 fits the model + inference latency.
- **M9c — Image-only VLA.** Drop `state_obs` and force the policy to localize cube/goal from vision only. Tests how much M6's spatial gains came from the image vs from state.
- **M9d — Multi-target instructions.** Within-task instruction variation ("place at the left fixture" vs "place at the right fixture") to test fine-grained conditioning beyond task identity.
- **M9e — Vision-language alignment loss.** Contrastive objective on `image_emb` / `lang_emb` to reduce shortcut competition (cf. the M6 trade-off in §5.3).

### 9. Conclusion

This project demonstrates that a complete VLA-style stack — Dockerized simulation, expert demonstration generation, step-level dataset construction, CLIP text + vision integration, multiple training objectives (BC weighted-MSE, auxiliary CE, eps-prediction DDPM), receding-horizon DDIM inference, gradient-conflict resolution, and per-task head selection — can be built, ablated, and analyzed end-to-end on a single consumer GPU.

The most consequential finding is **negative in form but positive in implication**: at PoC scale, no single optimization-, architecture-, or policy-class fix removes the multi-task capacity-sharing trade-off. The trade-off shifts, it does not vanish. That this result is *reproducible across three structurally different interventions* identifies trunk capacity itself — i.e., foundation-model backbones — as the load-bearing lever for the next milestone.

---

## Part III — Retrospective Appendix

A milestone-by-milestone log in *hypothesis → experiment → finding → next-step* form. Cross-links to the per-milestone docs.

### M0 — Dockerized ManiSkill bring-up

- **Hypothesis:** ManiSkill + GPU runs reproducibly in a Docker container on this hardware.
- **Experiment:** Random rollouts on PickCube-v1 inside the container.
- **Finding:** Confirmed; baseline established.
- **Next:** Add manufacturing-style task wrapper (M1).

### M1 — Manufacturing language task wrapper

- **Hypothesis:** A thin metadata wrapper can convert PickCube into a language-conditioned manufacturing task without modifying ManiSkill internals.
- **Experiment:** Per-episode instruction templates + task metadata logging.
- **Finding:** Works; instruction is metadata only (not yet a policy input).
- **Next:** Define the step-level dataset contract (M2A).

### M2A → M2C — Step-level dataset

- **Hypothesis:** A clean step-level NPZ format is necessary for downstream training; ManiSkill's MP expert can be wrapped locally because the installed runner has an import mismatch.
- **Experiment:** Built `m2_collect_step_dataset.py`, `m2_generate_pickcube_expert.py`, `m2_convert_expert_h5_to_dataset.py`.
- **Finding:** Yes; this format carried unchanged through M3–M8.
- **Notes:** See [docs/m2_dataset_report.md](m2_dataset_report.md).

### M3.0 → M3.3L — State-only BC saturation

- **Hypothesis:** State-only BC will solve PickCube given enough data and training.
- **Experiment:** Train MLP, evaluate open-loop, then closed-loop, then with longer training.
- **Finding:** Open-loop converges; closed-loop success = 0%. Longer training does not break it.
- **Diagnosis:** Debug video at M3.3A shows the policy pushing the cube away rather than producing inactive actions. The policy has learned something, but the learned function lacks phase resolution.
- **Next:** Add phase signals (M3.4).

### M3.4 → M3.6 — Phase-aware BC + safe filter + final-hold

- **Hypothesis:** Appending `progress` and `prev_action` to the observation will resolve the approach/grasp/lift/place phase ambiguity.
- **Experiment:** `obs = state(57) + progress(1) + prev_action(8) = 66`.
- **Finding:** `grasped_once` jumps to 56.7%. First non-zero closed-loop success (3.3%) once the final-hold wrapper is added.
- **Next:** Scale data (M3.7) and adjust loss (M3.8).

### M3.7 → M3.8 — 5000 demos, phase-weighted loss

- **Hypothesis:** More data + late-phase loss weighting will break the 0% ceiling.
- **Experiment:** Trained on 5000-episode dataset with `early:mid:late = 1:2:4`.
- **Finding:** Closed-loop success remained 0%. Confirmed the ceiling is not a data-quantity issue at this architecture scale.
- **Next:** DAgger (M3.9).

### M3.9 — DAgger pipeline

- **Hypothesis:** Correcting policy-visited states with the motion-planning expert will close the distribution-shift gap that pure BC suffers from.
- **Experiment:** Rollout selection + planner relabel + aggregate retrain. Two iterations: v1 with `deteriorated_after_min_dist` selection, v2 with release-phase biased selection.
- **Finding:** v1 noticeably improves trajectory shape and end-stability (`mean_return` 14.56 → 20.96, `final_grasped` 0 → 0.167). v2 underperformed v1 — removing the `deteriorated` selection rule unintentionally weakened cube-approach behavior.
- **Lesson:** Selection-rule design matters more than DAgger iteration count at this data scale.
- **Notes:** See [docs/m3_9_dagger.md](m3_9_dagger.md).

### M3.9H — force_grip_while_far

- **Hypothesis:** The dominant failure mode after DAgger v1 is premature gripper release at progress ≈ 0.95.
- **Experiment:** Inference-time rule: once the policy has ever closed on the cube and the cube is still far from the goal, force `action[-1] = -1.0`.
- **Finding:** `final_grasped` 0.167 → 0.367, `mean_return` 20.96 → 24.10, `success` still 0.
- **Lesson:** That a 5-line rule helps this much is itself a capacity diagnostic.

### M4 — Language-conditioned BC infrastructure

- **Hypothesis:** Adding CLIP-text + a learned `lang_proj` will not break the BC base.
- **Experiment:** Single-task (PickCube) BC with CLIP-text appended.
- **Finding:** Closed-loop parity with M3 phase-aware BC. The language pathway is wired correctly, instruction is currently dummy (only one task). **This is the precondition for M5.**
- **Notes:** See [docs/m4_lang_bc.md](m4_lang_bc.md).

### M5 — Multi-task language-conditioned BC

- **Hypothesis:** Combining PickCube + PushCube + PullCube into a single dataset will (a) produce the first non-zero closed-loop success on the easier tasks and (b) cause the policy to use the instruction as a differentiator.
- **Experiment:** 100 episodes per task; same `LangBCPolicy`.
- **Finding:** (a) Yes — PushCube matched-instruction success 10%, the first non-zero. (b) No — swap matrix shows the policy decides almost entirely from state.
- **Lesson:** BC has no direct incentive to attend to language when state alone resolves the action.
- **Notes:** See [docs/m5_multitask.md](m5_multitask.md).

### M5.1 — Auxiliary task-classification loss

- **Hypothesis:** Adding `CE(task_logits, task_id)` on top of `lang_proj` will force the language projection to be linearly task-separable, removing the BC head's excuse to ignore it.
- **Experiment:** `total_loss = BC_phase_weighted_MSE + 1.0 × CE(task_logits, task_id)`.
- **Finding:** Genuine instruction following. PickCube grasp under swap: 10% → 0%. PullCube matched: 0% → 33%. PushCube matched: 10% → 20%. `val_task_acc = 1.000`.
- **Lesson:** Open-loop MSE was not what differentiated M5 from M5.1 in any meaningful way; the swap matrix did.
- **Notes:** See [docs/m5_1_aux_loss.md](m5_1_aux_loss.md).

### M6 → M6.2 — Vision-language-action

- **Hypothesis:** Adding CLIP-vision to the conditioning input will improve spatial competence.
- **Experiment:** Frozen CLIP-vision tower, per-step embeddings precomputed by deterministic replay of expert trajectories.
- **Finding (M6 v0):** PickCube grasp 7% → 30% — vision provides cube/goal localization. **But** instruction following is partially diluted (swap-grasp 0% → 7–10%).
- **Finding (M6.1, M6.2):** Watching expert videos revealed that PushCube and PullCube experts stop at the boundary because ManiSkill's native success fires there. Adding a third motion-planning settle stage + raising `late_weight` from 4 to 8 pushed PushCube success 13% → 50% and PullCube 30% → 43%. Strict xy/sustained metrics + `--ignore-termination` eval added.
- **Lesson:** Sim-side success criteria can silently distort the training target. Expert-side fixes can be larger interventions than policy-side fixes.
- **Notes:** See [docs/m6_vla.md](m6_vla.md), [docs/m6_1_settle.md](m6_1_settle.md).

### M7 → M7.1 — Diffusion policy

- **Hypothesis:** A multi-modal action head (diffusion) will break the capacity-sharing trade-off observed in M6.2 where each VLA variant is best at a different task.
- **Experiment:** Chi et al.'s 1D conditional U-Net + DDPM training + DDIM receding-horizon inference (K=2, T=24, 500 epochs) on the same multi-task dataset, with the M6 conditioning vector as `cond` and aux task CE retained.
- **Finding (M7 v0):** Under-trained variant did not exceed BC. Hyperparameters (chunk horizon, K, T) matter.
- **Finding (M7.1 v1):** PushCube 30% → 47% (first clean win over BC); `mean_return` 1.5–2× higher than BC everywhere. But PullCube drops 43% → 23% and PickCube grasp stays 0%.
- **Lesson:** The trade-off pattern is preserved, just rotated. Diffusion now owns PushCube; BC still owns PullCube and PickCube.
- **Notes:** See [docs/m7_diffusion.md](m7_diffusion.md).

### M8 — 4-way capacity-sharing study

- **Hypothesis:** A structural fix (PCGrad or per-task heads) on the same shared trunk will remove the trade-off observed in M6.2/M7.1.
- **Experiment:** Same dataset, same eval matrix, same baseline. PCGrad (Yu et al. 2020) at every optimizer step; per-task output heads selected by the task classifier on `lang_proj`.
- **Finding:** Trade-off persists across all three fixes (Diffusion / PCGrad / per-task heads). PCGrad detects ~95 gradient conflicts per epoch and resolves them, recovering PickCube grasp 13% → 20% but giving up nothing in return. Per-task heads give the cleanest instruction-following purity (swap PullCube+Pick: 13% → 0%) but the shared trunk still re-allocates capacity away from PickCube grasp (0%).
- **Lesson:** The control variable across all three fixes was the shared trunk. **Trunk capacity is the next lever** — i.e., M9b (OpenVLA / Octo LoRA fine-tune).
- **Notes:** See [docs/m8_multitask_capacity.md](m8_multitask_capacity.md).

---

## Cross-cutting findings (one-line summary)

1. **Phase signals were the largest single BC improvement** (M3.4): `grasped_once` 0 → 56.7%.
2. **Open-loop MSE and `val_task_acc` are not reliable VLA model-selection signals** — only the closed-loop swap matrix is (M5 vs M5.1, M8b val_mse vs closed-loop).
3. **A 5-line inference rule (`force_grip_while_far`) almost doubles end-state grasp** (M3.9H) — a capacity diagnostic.
4. **Sim-native success criteria can be deceptively lax**; strict xy/sustained metrics changed how M6/M7/M8 were compared (M6.1, M6.2).
5. **Multi-task capacity-sharing is the load-bearing constraint at this scale**, persisting across three structurally different fixes (M8).
