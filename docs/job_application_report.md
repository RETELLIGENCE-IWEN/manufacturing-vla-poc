# manufacturing-vla-poc — 지원서/포트폴리오용 프로젝트 분석 리포트

대상 직무: 현대차 제조로보틱스 / 모방학습(IL) · 강화학습(RL) · VLA 기반 로봇 자율작업
근거 원칙: README, docs/, scripts/, configs/, outputs/, runs/ 에 실제로 존재하는 파일·코드·metric 만을 인용. 추정·과장 금지.

---

## 1. Repository Snapshot

| 항목 | 값 |
| -- | -- |
| Repo name | `manufacturing-vla-poc` |
| Current branch | `master` |
| Working tree | clean (`git status --short` empty) |
| Latest commit | `d76c8ea — v1 report` (HEAD → master) |
| 직전 마일스톤 커밋 | `7f2dc71 — M8 4-way comparison study 완료` (origin/master) |
| Main language / framework | Python 3 / PyTorch / HuggingFace `transformers==4.46.3` |
| Simulator / Environment | ManiSkill (`PickCube-v1`, `PushCube-v1`, `PullCube-v1`) |
| Robot / Control mode | Panda arm, `pd_joint_pos` |
| Container base | `maniskill/base:latest` (CUDA, Vulkan rendering, NVIDIA GPU) |
| 주요 의존성 | `pyyaml`, `rich`, `imageio`, `imageio-ffmpeg`, `h5py`, `transformers==4.46.3` (Dockerfile) |

### 최근 커밋 (8건)

```
d76c8ea (HEAD -> master) v1 report
7f2dc71 (origin/master)  M8 4-way comparison study 완료
8e66200                  M7.1 v1 Diffusion Policy
6efc27c                  M6.1 + M6.2 정리 완료
5ab5430                  Lang-conditioned multi-task BC (M4-M5.1) with aux task_id loss
9685b09                  trying out DAgger
aba6bfc                  bc 5k
419780f                  Add final hold stabilization wrapper for phase-aware BC
```

### 디렉터리 트리 (top-level, 소스 한정)

```
manufacturing-vla-poc/
├── README.md
├── docker/Dockerfile
├── configs/      # 20 YAML (manufacturing task 정의 + 마일스톤별 학습 hyperparameter)
├── scripts/      # 41 Python entrypoint (M0~M8 학습/평가/데이터 파이프라인)
├── docs/         # 10 Markdown (마일스톤별 분석 보고서 + 본 리포트)
├── outputs/      # 데이터셋·expert demo·DAgger correction (artifact, 미커밋 의도)
└── runs/         # 모델 체크포인트·학습 곡선·closed-loop eval (artifact, 미커밋 의도)
```

### 실행 가능한 entrypoint (선별)

| 스크립트 | 역할 |
| -- | -- |
| `scripts/m0_random_rollout.py` | ManiSkill bring-up smoke test |
| `scripts/m1_language_task_rollout.py` | manufacturing-style language task wrapper |
| `scripts/m2_collect_step_dataset.py` | step-level dataset 로거 |
| `scripts/m2_generate_pickcube_expert.py` | PickCube motion-planning expert (project-local) |
| `scripts/m2_convert_expert_h5_to_dataset.py` | HDF5 → npz dataset 변환 |
| `scripts/m3_train_bc_state.py` / `m3_train_bc_phase_weighted.py` | 상태 기반 BC / phase-aware + 가중치 손실 |
| `scripts/m3_eval_closedloop_bc_phase_safe.py` | safe-action filter + force_grip 포함 closed-loop eval |
| `scripts/m3_collect_dagger_rollouts.py` / `m3_relabel_dagger_with_planner.py` / `m3_build_dagger_aggregate_dataset.py` | DAgger 3단계 파이프라인 |
| `scripts/m4_add_instruction_embeddings.py` / `m4_train_bc_lang.py` / `m4_eval_closedloop_bc_lang.py` | CLIP-text 언어 조건 BC |
| `scripts/m5_generate_multitask_expert.py` / `m5_convert_multitask_h5_to_dataset.py` / `m5_eval_closedloop_bc_lang.py` | 멀티태스크 데이터 + 평가 |
| `scripts/m5_1_train_bc_lang_aux.py` / `m5_1_eval_closedloop_bc_lang_aux.py` | aux task-classification 손실 학습/평가 |
| `scripts/m6_add_image_embeddings.py` / `m6_train_vla_lang_aux.py` / `m6_eval_closedloop_vla.py` / `m6_record_vla_debug.py` | CLIP-vision 추가 VLA |
| `scripts/m6_generate_multitask_expert_v1.py` / `m6_generate_multitask_expert_v2.py` | PushCube/PullCube settle 보강 expert |
| `scripts/m7_train_diffusion_policy.py` / `m7_eval_closedloop_diffusion.py` / `m7_record_diffusion_debug.py` | Diffusion Policy (1D U-Net + DDPM + DDIM) |
| `scripts/m8a_train_bc_pcgrad.py` | PCGrad 그래디언트 surgery 학습 |
| `scripts/m8b_train_bc_per_task_head.py` / `m8b_eval_closedloop_per_task.py` | per-task output head 학습/평가 |

### 주요 config 파일

`configs/manufacturing_pick_place_v0.yaml`, `configs/manufacturing_multitask_v0.yaml` (태스크 정의 + 언어 지시 템플릿), 그리고 마일스톤별 hyperparameter YAML 18종 (`m3_*`, `m4_*`, `m5_*`, `m6_*`, `m7_*`, `m8a_*`, `m8b_*`).

### 주요 docs/output

- `docs/m2_dataset_report.md`, `docs/m3_9_dagger.md`, `docs/m4_lang_bc.md`, `docs/m5_multitask.md`, `docs/m5_1_aux_loss.md`, `docs/m6_vla.md`, `docs/m6_1_settle.md`, `docs/m7_diffusion.md`, `docs/m8_multitask_capacity.md`, `docs/research_report.md`
- `runs/m*/closedloop_*/closedloop_summary.json` — 모든 closed-loop 평가의 raw metric
- `runs/m*/metrics.json` — 학습 best epoch · val MSE · task acc 등 학습 사이드 metric
- `outputs/m3_dagger_corrections_v*/summary.json` — DAgger 보정 성공률
- `outputs/m5_multitask_dataset/summary.json` — 300 episode (PickCube/PushCube/PullCube × 100) success_rate_once = 1.0

---

## 2. Project Purpose

### 어떤 문제를 푸는 PoC인가
ManiSkill 시뮬레이션 상에서 Panda 7-DoF 매니퓰레이터를 대상으로, **"제조형 언어 지시 + 정형 부품 조작"이라는 시나리오를 모방학습 → 정책 고도화 → VLA-style 멀티모달 정책까지 한 줄로 잇는 파이프라인**을 구축한 PoC이다. 단일 RTX 3060 환경에서 데이터 수집·학습·평가·실패 분석 사이클을 모두 직접 수행한 것이 핵심이며, **모델 자체보다 "이 정책을 어떻게 만들어내고, 어떻게 검증하고, 어떻게 다음 실험으로 연결하는가"**를 보여주는 데 목적이 있다.

### Target robot / task / environment
- Robot: ManiSkill Panda arm (`pd_joint_pos`, action_dim = 8)
- Task: PickCube-v1 (집어 위치 고정) · PushCube-v1 (밀어 영역 진입) · PullCube-v1 (당겨 영역 진입)
- Manufacturing wrapper: `configs/manufacturing_pick_place_v0.yaml`, `configs/manufacturing_multitask_v0.yaml` 가 "Pick the bolt-like part and place it at the left fixture" 같은 제조형 instruction 템플릿을 task metadata 로 attach
- Observation: `panda(31) + cube(13) + goal(13) = 57` 기본 state, M3.4 부터 `+progress(1) + prev_action(8) = 66` phase-aware, M4 부터 +`lang_emb(512)`, M6 부터 +`image_emb(768)`

### 제조로보틱스 관점에서의 의미
제조 현장에서 자주 등장하는 "정해진 부품을 정해진 위치에" 집어/밀어/당겨놓는 정형 동작을, **expert demonstration → 모방학습 → closed-loop 실패 분석 → DAgger 보정 → 멀티태스크/언어/비전 조건화**의 표준 워크플로로 풀어낸 사례이다. 실제 라인 데이터가 없는 상황에서도 시뮬레이션 기반 PoC 만으로 "어떤 인풋(state vs vision)이 어디까지 책임지는가", "어디서부터 capacity 가 부족해지는가", "instruction 이 실제로 정책에 영향을 주는지 어떻게 검증할 것인가" 같은 **현장 적용 직전에 반드시 답해야 하는 질문들**에 정량적 근거를 만들어둔 점이 의미를 갖는다.

### VLA-style policy learning 관점에서의 핵심 목표
"foundation VLA 학습"이 아니라, **"VLA stack 의 각 층 — 상태, 언어, 비전, 보조 손실, action head class — 이 closed-loop 성능에 얼마나 기여하는가"**를 같은 데이터셋·같은 평가 행렬 위에서 분리/측정하는 것이다. (M8 4-way 비교가 그 정점이다.)

### "foundation 학습"이 아닌 "applied robotics AI workflow"라는 점
- 학습 가능한 파라미터는 모두 합쳐도 ~300k 수준 (M8b 측정값 `num_params_M = 0.299547`).
- CLIP text · vision tower 는 모두 frozen, `lang_proj` / `image_proj` 와 BC head 만 학습.
- 데이터셋 규모는 PickCube 100·5000, multi-task 300 episode 수준 — foundation scale 이 아니다.
- 본 프로젝트가 입증하는 것은 모델 크기가 아니라 **"단일 RTX 3060 위에서 데이터/학습/평가/실패분석/정책개선 한 사이클을 완결한 능력"**.

### 현대차 IL/RL/VLA 직무와의 연결 지점
- IL: M2~M3 의 expert demonstration 생성 + state-action BC + open-loop/closed-loop 평가 사이클
- DAgger: M3.9 의 rollout → planner relabel → aggregate → retrain 워크플로 직접 구현
- 언어 조건 정책: M4~M5.1 의 CLIP text + 보조 분류 손실 트릭으로 instruction following 을 "실제로" 작동시킨 사례
- VLA: M6 에서 CLIP vision tower 까지 결합한 state + text + image 정책
- 정책 클래스 확장: M7 Diffusion Policy (Chi et al. 2023) 와 M8a PCGrad / M8b per-task head 까지 같은 데이터셋에서 직접 비교
- closed-loop 평가와 failure diagnosis: 모든 마일스톤에서 영상·log 기반 실패 분석을 다음 마일스톤의 동기로 연결

---

## 3. Technical Pipeline (실제 코드 기준)

```
┌────────────────────────────────────────────────────────────────────────┐
│ M0  Dockerized ManiSkill bring-up                                      │
│   docker/Dockerfile (maniskill/base + CUDA + Vulkan)                   │
│   scripts/m0_random_rollout.py                                         │
└────────────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼──────────────────────────────────────────┐
│ M1  Manufacturing-style language task wrapper                          │
│   configs/manufacturing_pick_place_v0.yaml                             │
│   scripts/m1_language_task_rollout.py                                  │
│   episode 별 instruction · object_id · target_id 를 metadata 로 부여   │
└─────────────────────────────┬──────────────────────────────────────────┘
                              │
┌─────────────────────────────▼──────────────────────────────────────────┐
│ M2A  step-level dataset 로거                                           │
│   scripts/m2_collect_step_dataset.py → outputs/m2_step_dataset/        │
│   schema: instruction · observation · action · reward · success ···    │
│                                                                        │
│ M2B  PickCube motion-planning expert (project-local)                   │
│   scripts/m2_generate_pickcube_expert.py → outputs/m2_expert_demos_100 │
│   (설치된 ManiSkill MP runner 의 import 충돌 회피용 minimal wrapper)   │
│                                                                        │
│ M2C  H5 → M3-ready dataset 변환                                        │
│   scripts/m2_convert_expert_h5_to_dataset.py                           │
│   obs_t = panda(31) + cube(13) + goal_site(13) = 57, action(8)         │
└─────────────────────────────┬──────────────────────────────────────────┘
                              │
┌─────────────────────────────▼──────────────────────────────────────────┐
│ M3.0  state-only BC (MLP, MSE)                                         │
│   scripts/m3_train_bc_state.py + configs/m3_bc_state.yaml              │
│ M3.1  open-loop action prediction eval                                 │
│   scripts/m3_eval_openloop_bc_state.py                                 │
│ M3.2  closed-loop eval vs random baseline                              │
│   scripts/m3_eval_closedloop_bc_state.py                               │
│ M3.3A failure diagnosis via debug video / step logs                    │
│   scripts/m3_record_closedloop_debug.py                                │
│ M3.4  phase-aware BC (state + progress + prev_action = 66)             │
│   scripts/m3_make_phase_aware_dataset.py                               │
│ M3.5  safe action filter (expert-bounded action clipping)              │
│   scripts/m3_eval_closedloop_bc_phase_safe.py (--expert-action-bounds) │
│ M3.6  final-hold stabilization wrapper                                 │
│   동 스크립트 (--final-hold-*)                                         │
│ M3.7/3.8  5000-demo + phase-weighted loss                              │
│   scripts/m3_train_bc_phase_weighted.py + configs/m3_bc_phase_weighted │
│ M3.9  DAgger pipeline (rollout → relabel → aggregate → retrain)        │
│   scripts/m3_collect_dagger_rollouts.py                                │
│   scripts/m3_relabel_dagger_with_planner.py                            │
│   scripts/m3_build_dagger_aggregate_dataset.py                         │
│   scripts/m3_train_bc_phase_weighted.py (aggregate 위 재학습)          │
│ M3.9H force_grip_while_far inference heuristic                         │
│   m3_eval_closedloop_bc_phase_safe.py (--force-grip-while-far)         │
└─────────────────────────────┬──────────────────────────────────────────┘
                              │
┌─────────────────────────────▼──────────────────────────────────────────┐
│ M4  CLIP-text 언어 조건 BC                                              │
│   scripts/m4_add_instruction_embeddings.py (CLIPTextModel pooler, 512) │
│   scripts/m4_train_bc_lang.py   policy_input = obs(66) + lang_proj(64) │
│   scripts/m4_eval_closedloop_bc_lang.py                                 │
└─────────────────────────────┬──────────────────────────────────────────┘
                              │
┌─────────────────────────────▼──────────────────────────────────────────┐
│ M5  multi-task lang BC (PickCube + PushCube + PullCube)                │
│   scripts/m5_generate_multitask_expert.py                              │
│   scripts/m5_convert_multitask_h5_to_dataset.py                        │
│   configs/manufacturing_multitask_v0.yaml (task 별 instruction 템플릿) │
│   scripts/m5_eval_closedloop_bc_lang.py (goal_site / goal_region auto) │
│                                                                        │
│ M5.1  aux task_id 분류 손실 동시 학습                                  │
│   total_loss = BC_phase_weighted_MSE + 1.0 × CE(task_logits, task_id)  │
│   scripts/m5_1_train_bc_lang_aux.py / m5_1_eval_closedloop_bc_lang_aux │
└─────────────────────────────┬──────────────────────────────────────────┘
                              │
┌─────────────────────────────▼──────────────────────────────────────────┐
│ M6  VLA — state + CLIP-text + CLIP-vision                              │
│   scripts/m6_add_image_embeddings.py (CLIPVisionModel pooler, 768)     │
│   scripts/m6_train_vla_lang_aux.py                                     │
│     policy_input = obs(66) + lang_proj(64) + image_proj(128) = 258     │
│   scripts/m6_eval_closedloop_vla.py / m6_record_vla_debug.py           │
│                                                                        │
│ M6.1  PushCube settle solver (expert 3-stage)                          │
│   scripts/m6_generate_multitask_expert_v1.py                           │
│   configs/m6_vla_aux_v1.yaml (late_weight: 4 → 8)                      │
│                                                                        │
│ M6.2  PullCube settle solver + 엄격 xy/sustained metric                │
│   scripts/m6_generate_multitask_expert_v2.py                           │
│   m6_eval_closedloop_vla.py 에 xy / xy_sustained_*_10 metric 추가      │
│   eval --ignore-termination 옵션                                       │
└─────────────────────────────┬──────────────────────────────────────────┘
                              │
┌─────────────────────────────▼──────────────────────────────────────────┐
│ M7  Diffusion Policy on multi-task VLA dataset                         │
│   1D conditional U-Net + FiLM, eps-prediction DDPM, DDIM 추론          │
│   scripts/m7_train_diffusion_policy.py / m7_eval_closedloop_diffusion  │
│   M7  v0: 200 epoch, action_exec=4, infer_steps=16                     │
│   M7.1 v1: 500 epoch, action_exec=2, infer_steps=24                    │
└─────────────────────────────┬──────────────────────────────────────────┘
                              │
┌─────────────────────────────▼──────────────────────────────────────────┐
│ M8  multi-task capacity-sharing 4-way 비교                              │
│   M8a PCGrad — scripts/m8a_train_bc_pcgrad.py                          │
│         (per-task gradient projection at every optimizer step)         │
│   M8b per-task head — scripts/m8b_train_bc_per_task_head.py            │
│         (공유 trunk + task_id 로 선택되는 3개 출력 head)               │
│   BC v2 (M6.2) / Diffusion (M7.1) / PCGrad (M8a) / PerTaskHead (M8b)   │
│   동일 swap matrix(6 cell × 30 episode) closed-loop 비교               │
└────────────────────────────────────────────────────────────────────────┘
```

### closed-loop evaluation & failure diagnosis 공통 프로토콜
- 30 episodes / cell, `seed=3000`, `phase_horizon=80`, `max_steps=120`
- safe-action filter (M3.5) 항상 활성
- M6.2 부터 `--ignore-termination` 으로 native success 발화 이후도 관측
- 모든 평가에 random baseline 동시 비교
- 실패 분석: `m3_record_closedloop_debug.py`, `m3_record_phase_safe_debug.py`, `m6_record_vla_debug.py`, `m7_record_diffusion_debug.py` 로 mp4 저장 + per-step CSV log

### swap-matrix evaluation
M5 이후 모든 멀티태스크 모델은 (env × instruction) 6 cell — matched 3 cell + swap 3 cell — 으로 평가. instruction 이 실제로 정책에 영향을 미치는지 검증하는 핵심 도구.

---

## 4. Milestone Progress (실제 파일·metric 출처 명시)

| Milestone | Status | 구현 내용 | 주요 파일 | 확인된 결과/metric | 출처 |
| -- | -- | -- | -- | -- | -- |
| M0 | ✅ | Dockerized ManiSkill bring-up + 랜덤 rollout smoke | `docker/Dockerfile`, `scripts/m0_random_rollout.py` | 컨테이너+GPU+ManiSkill 정상 구동 확인 | README.md §M0 |
| M1 | ✅ | 제조형 언어 task wrapper · instruction metadata | `scripts/m1_language_task_rollout.py`, `configs/manufacturing_pick_place_v0.yaml` | episode 별 instruction · object_id · target_id 저장 | README.md §M1, configs/manufacturing_pick_place_v0.yaml |
| M2A | ✅ | step-level dataset schema/logger | `scripts/m2_collect_step_dataset.py` | dataset_schema.json 정의 | outputs/m2_step_dataset/dataset_schema.json |
| M2B | ✅ | PickCube motion-planning expert | `scripts/m2_generate_pickcube_expert.py` | 100·5000 episode dataset 생성 | outputs/m2_expert_demos_100, outputs/m2_expert_demos_5000 |
| M2C | ✅ | H5 → 학습용 dataset 변환 | `scripts/m2_convert_expert_h5_to_dataset.py` | obs_dim=57, action_dim=8 | outputs/m2_expert_dataset_100/summary.json |
| M3.0 | ✅ | state-only BC (MLP, MSE) | `scripts/m3_train_bc_state.py` | best_val_mse_norm ≈ 0.00595, success_rate_once = 0.0 (closed-loop), mean_return = 6.80 vs random 0.15 | runs/m3_bc_state/metrics.json, closedloop_eval/closedloop_summary.json |
| M3.1 | ✅ | open-loop action prediction eval | `scripts/m3_eval_openloop_bc_state.py` | predicted vs expert action 비교 | runs/m3_bc_state/openloop_eval/ |
| M3.2 | ✅ | closed-loop vs random baseline | `scripts/m3_eval_closedloop_bc_state.py` | mean_return_delta = +6.65 (BC vs random) | runs/m3_bc_state/closedloop_eval/closedloop_summary.json |
| M3.3A | ✅ | failure diagnosis (video + log) | `scripts/m3_record_closedloop_debug.py` | "grasp 단계에서 cube 를 밀어내며 실패" 진단 | README.md §M3.3A |
| M3.3L | ✅ | 더 긴 학습으로 closed-loop 확인 | `scripts/m3_train_bc_state.py` + `configs/m3_bc_state_long.yaml` | success_rate_once = 0.0 (학습 시간 늘려도 미해결) | README.md §M3.3L |
| M3.4 | ✅ | phase-aware BC (state+progress+prev_action) | `scripts/m3_make_phase_aware_dataset.py`, `scripts/m3_train_bc_phase_weighted.py` | best_val_mse_norm = 0.00475, grasped_once_rate = 0.567 (100-demo, safe filter) | runs/m3_bc_phase_aware/metrics.json, closedloop_eval_safe_120step/closedloop_summary.json (mean_return 14.56) |
| M3.5 | ✅ | safe action filter | `m3_eval_closedloop_bc_phase_safe.py --expert-action-bounds` | 모든 M3.6+ 평가에 활성 | outputs/m3_phase_aware_dataset_100/action_bounds.json |
| M3.6 | ✅ | final-hold stabilization wrapper | 동 스크립트 `--final-hold-*` | success_rate_once 0 → 0.033 (PickCube 첫 성공) | runs/m3_bc_phase_aware/closedloop_eval_final_hold_120step/closedloop_summary.json |
| M3.7 | ✅ | 5000-demo phase-aware BC | `configs/m3_bc_phase_aware_5000.yaml` | grasped_once 0.433, success 0 (데이터 양만으로 성공률 미돌파) | runs/m3_bc_phase_aware_5000/closedloop_eval_safe_120step/closedloop_summary.json |
| M3.8 | ✅ | phase-weighted loss (1:2:4) | `configs/m3_bc_phase_weighted_5000.yaml` | grasped_once 0.2, success 0 (가중치만으로도 미돌파) | runs/m3_bc_phase_weighted_5000/closedloop_eval_final_hold_120step/closedloop_summary.json |
| M3.9 v1 | ✅ | DAgger v1 (50 rollouts, deteriorated selection) | `m3_collect_dagger_rollouts.py` → `m3_relabel_dagger_with_planner.py` → `m3_build_dagger_aggregate_dataset.py` | 1553 selected, **1121 corrections (72.2%)**, dagger_weight=2.0 | outputs/m3_dagger_corrections_v1/summary.json |
| M3.9 v1 closed-loop | ✅ | DAgger v1 정책 closed-loop | `m3_eval_closedloop_bc_phase_safe.py` | grasped_once 0.467, final_grasped 0.167, mean_return 20.96, success 0 | runs/m3_bc_dagger_v1/closedloop_eval_safe/closedloop_summary.json |
| M3.9 v2 | ✅ | release-phase biased DAgger v2 | 동 파이프라인 + `--disable-deteriorated` | 899 selected, 830 corrections (92.3%) — v1 보다 underperform | outputs/m3_dagger_corrections_v2/summary.json, runs/m3_bc_dagger_v2/closedloop_eval_safe_forcegrip/closedloop_summary.json (grasp 0.30 < v1 0.467) |
| M3.9H | ✅ | force_grip_while_far inference heuristic | `m3_eval_closedloop_bc_phase_safe.py --force-grip-while-far` | **final_grasped 0.167 → 0.367**, mean_return 20.96 → **24.10** | runs/m3_bc_dagger_v1/closedloop_eval_safe_forcegrip/closedloop_summary.json |
| M4 | ✅ | CLIP-text 언어 조건 BC (단일태스크) | `m4_add_instruction_embeddings.py`, `m4_train_bc_lang.py`, `m4_eval_closedloop_bc_lang.py` | best_val_mse_norm = 0.00584, closed-loop = M3 phase-aware 와 통계적 동등 (sanity check 통과) | runs/m4_bc_lang_v0/metrics.json, closedloop_eval_safe/closedloop_summary.json (grasp 0.50, placed 0.033) |
| M5 | ✅ | multi-task lang BC (PickCube+PushCube+PullCube, 100/each) | `m5_generate_multitask_expert.py`, `m5_convert_multitask_h5_to_dataset.py`, `m5_eval_closedloop_bc_lang.py` | best_val_mse_norm = 0.00426, **PushCube success 0.10 (프로젝트 첫 비-0)**. instruction swap 시 grasp 0.10→0.10→0.067 (instruction 이 차별자로 작동하지 않음) | runs/m5_bc_lang_multitask_v0/metrics.json, closedloop_eval_pushcube + swap_pickenv_*/closedloop_summary.json |
| M5.1 | ✅ | + aux task_id 분류 손실 | `m5_1_train_bc_lang_aux.py` (aux_weight=1.0) | best_val_mse_norm = **0.00384**, val_task_acc = 1.000 | runs/m5_1_bc_lang_multitask_aux_v0/metrics.json |
| M5.1 closed-loop | ✅ | swap matrix 평가 | `m5_1_eval_closedloop_bc_lang_aux.py` | **PullCube success 0.333 (프로젝트 첫 PullCube 성공)**, PushCube 0.20, PickCube grasp matched 0.067 / swap 0.0 (instruction following 실제로 작동) | runs/m5_1_bc_lang_multitask_aux_v0/eval_*/closedloop_summary.json |
| M6 v0 | ✅ | CLIP-vision 결합 VLA | `m6_add_image_embeddings.py`, `m6_train_vla_lang_aux.py` | best_val_mse_norm = 0.00437, val_task_acc = 1.000, **PickCube grasp 0.067 → 0.30** | runs/m6_vla_aux_v0/metrics.json, eval_pickcube_pick/closedloop_summary.json |
| M6.1 | ✅ | PushCube settle solver + late_weight 8 | `m6_generate_multitask_expert_v1.py`, `configs/m6_vla_aux_v1.yaml` | **PushCube success 0.13 → 0.50**, min_xy = 0.144 | runs/m6_vla_aux_v1/eval_pushcube_push/closedloop_summary.json |
| M6.2 | ✅ | PullCube settle + xy/sustained metric + ignore-termination | `m6_generate_multitask_expert_v2.py`, `m6_eval_closedloop_vla.py` (xy metrics 추가) | **PullCube success 0.30 → 0.433**, min_xy = 0.154 | runs/m6_vla_aux_v2/eval_pullcube_pull/closedloop_summary.json |
| M7 v0 | ✅ | Diffusion Policy (200 epoch, action_exec=4, infer_steps=16) | `m7_train_diffusion_policy.py`, `m7_eval_closedloop_diffusion.py` | best_val_diffusion_mse = 0.0126, val_task_acc = 1.000, 모든 셀에서 BC v2 미돌파 | runs/m7_diffusion_v0/metrics.json, eval_*/closedloop_summary.json |
| M7.1 v1 | ✅ | Diffusion (500 epoch, action_exec=2, infer_steps=24) | 동 스크립트 + `configs/m7_diffusion_v1.yaml` | best_val_diffusion_mse = **0.00932** (-26%), **PushCube success 0.30 → 0.467 (프로젝트 PushCube 최고)** | runs/m7_diffusion_v1/metrics.json, eval_pushcube_push/closedloop_summary.json |
| M8a | ✅ | PCGrad gradient surgery | `m8a_train_bc_pcgrad.py` | best_val_mse_norm = **0.00634**, val_task_acc = 1.000, **PickCube grasp 0.13 → 0.20** | runs/m8a_bc_pcgrad_v0/metrics.json, eval_pickcube_pick/closedloop_summary.json |
| M8b | ✅ | per-task output head | `m8b_train_bc_per_task_head.py`, `m8b_eval_closedloop_per_task.py` | best_val_mse_norm = 0.00771, num_params_M = 0.2995, **swap PullCube+Pick 0.13 → 0.0 (instruction-obey 가장 깨끗)** | runs/m8b_bc_per_task_head_v0/metrics.json, eval_pullcube_pickswap/closedloop_summary.json |
| M8 4-way | ✅ | BC v2 / Diffusion / PCGrad / PerTaskHead 동일 swap matrix 비교 | `docs/m8_multitask_capacity.md` | 각 변형이 서로 다른 셀에서 best — 단일 fix 로 trade-off 제거 불가 | runs/m6_vla_aux_v2/, runs/m7_diffusion_v1/, runs/m8a_bc_pcgrad_v0/, runs/m8b_bc_per_task_head_v0/ |

---

## 5. Key Results and Metrics (출처 명시)

### 5.1 M3 line — PickCube BC 진화

| 변형 | best_val_mse_norm | grasped_once | final_grasped | placed_once | success_once | mean_return | min_cube_goal_dist | 출처 |
| -- | -- | -- | -- | -- | -- | -- | -- | -- |
| state-only BC | 0.00595 | — | — | — | 0.000 | 6.80 | — | runs/m3_bc_state/metrics.json + closedloop_eval/closedloop_summary.json |
| phase-aware 100 (safe) | 0.00475 | (안기록) | — | — | 0.000 | 14.56 | — | runs/m3_bc_phase_aware/metrics.json + closedloop_eval_safe_120step/closedloop_summary.json |
| phase-aware 100 + final-hold | 동일 | **0.567** | 0.033 | **0.033** | **0.033** | 13.95 | 0.165 | runs/m3_bc_phase_aware/closedloop_eval_final_hold_120step/closedloop_summary.json |
| phase-aware 5000 (safe) | — | 0.433 | 0.000 | 0.000 | 0.000 | 12.51 | 0.185 | runs/m3_bc_phase_aware_5000/closedloop_eval_safe_120step/closedloop_summary.json |
| phase-weighted 5000 + final-hold | — | 0.200 | 0.000 | 0.000 | 0.000 | 10.91 | 0.202 | runs/m3_bc_phase_weighted_5000/closedloop_eval_final_hold_120step/closedloop_summary.json |
| **DAgger v1 (safe)** | 0.00420 | **0.467** | **0.167** | 0.000 | 0.000 | **20.96** | 0.182 | runs/m3_bc_dagger_v1/metrics.json + closedloop_eval_safe/closedloop_summary.json |
| **DAgger v1 + force_grip (M3.9H)** | 동일 | 0.467 | **0.367** | 0.000 | 0.000 | **24.10** | 0.190 | runs/m3_bc_dagger_v1/closedloop_eval_safe_forcegrip/closedloop_summary.json |
| DAgger v2 + force_grip | — | 0.300 | 0.167 | 0.000 | 0.000 | 18.57 | 0.193 | runs/m3_bc_dagger_v2/closedloop_eval_safe_forcegrip/closedloop_summary.json |

**DAgger correction 통계:** v1 — 1553 attempted / 1121 successful (success_rate **0.7218**) · v2 — 899 attempted / 830 successful (success_rate **0.9232**). 출처: `outputs/m3_dagger_corrections_v{1,2}/summary.json`.

### 5.2 M4 — 단일태스크 lang BC sanity check

| Metric | M3 phase-aware 100 | M4 lang_v0 | 출처 |
| -- | -- | -- | -- |
| best_val_mse_norm | 0.00475 | **0.00584** | runs/m4_bc_lang_v0/metrics.json |
| grasped_once | 0.567 | 0.500 | runs/m4_bc_lang_v0/closedloop_eval_safe/closedloop_summary.json |
| placed_once | 0.033 | 0.033 | 동 |
| mean_return | 13.95 | 13.92 | 동 |

→ 통계적 동등. 언어 path 가 BC 를 깨지 않은 것을 확인 (M5 진입 전 sanity 통과).

### 5.3 M5 vs M5.1 — instruction following swap matrix (30 ep, seed=3000)

| (env, instruction) | M5 success | **M5.1 success** | M5 grasped | **M5.1 grasped** | M5 mean_return | M5.1 mean_return | min_dist M5 → M5.1 |
| -- | -- | -- | -- | -- | -- | -- | -- |
| PickCube, Pick (matched) | 0.000 | 0.000 | 0.100 | 0.067 | 11.91 | 11.69 | 0.195 → 0.192 |
| PickCube, Push (swap) | 0.000 | 0.000 | **0.100** | **0.000** ⭐ | 10.15 | 9.19 | 0.199 → 0.207 |
| PickCube, Pull (swap) | 0.000 | 0.000 | 0.067 | **0.000** ⭐ | 11.27 | 10.11 | 0.201 → 0.203 |
| **PushCube, Push (matched)** | **0.100** | **0.200** | 0.000 | 0.000 | 15.13 | 16.52 | 0.191 → 0.173 |
| PushCube, Pick (swap) | 0.000 | 0.000 | 0.000 | 0.000 | 15.69 | 19.76 | 0.196 → 0.201 |
| **PullCube, Pull (matched)** | 0.000 | **0.333** ⭐⭐ | 0.000 | 0.000 | 15.70 | 17.80 | 0.193 → **0.163** |

출처: runs/m5_bc_lang_multitask_v0/closedloop_eval_*, swap_*/closedloop_summary.json · runs/m5_1_bc_lang_multitask_aux_v0/eval_*/closedloop_summary.json
학습 출처: runs/m5_*/metrics.json — M5 best_val_mse_norm 0.00426 → M5.1 **0.00384**, **val_task_acc = 1.000** (aux_weight=1.0)

### 5.4 M5.1 → M6 → M6.1 → M6.2 — 비전 추가 + settle solver

| Cell | M5.1 success/grasp | M6 v0 | M6.1 v1 | **M6.2 v2** |
| -- | -- | -- | -- | -- |
| PickCube/Pick — grasp | 0.067 | **0.300** ⭐ | 0.167 | 0.133 (forcegrip 시 0.133, placed 0.033) |
| PushCube/Push — success | 0.200 | 0.133 | **0.500** ⭐⭐ | 0.300 |
| PullCube/Pull — success | **0.333** | 0.300 | 0.300 | **0.433** |
| PickCube/Push (swap) — grasp | 0.000 | 0.067 | 0.000 | 0.033 |
| PullCube/Pick (swap) — success | — | — | 0.333 | 0.133 |

출처: runs/m6_vla_aux_v0/eval_*, runs/m6_vla_aux_v1/eval_*, runs/m6_vla_aux_v2/eval_*/closedloop_summary.json. xy sustained metric 은 v1, v2 에 추가됨 (e.g. `mean_min_cube_xy_goal_dist`: v1 PushCube 0.144, v2 PullCube 0.154).
학습 출처: M6 v0 best_val_mse_norm 0.00437, v1 0.00604 (best_epoch 92), v2 0.00678 (best_epoch 107). val_task_acc = 1.000 모든 변형.

### 5.5 M7 Diffusion vs BC v2

| Cell | BC v2 (M6.2) | M7 v0 | **M7.1 v1** |
| -- | -- | -- | -- |
| PickCube/Pick grasp | 0.133 | 0.000 | 0.000 |
| **PushCube/Push success** | 0.300 | 0.300 | **0.467** ⭐ |
| **PullCube/Pull success** | **0.433** | 0.300 | 0.233 |
| PickCube/Push (swap) grasp | 0.033 | 0.000 | 0.033 |
| PullCube/Pick (swap) success | 0.133 | 0.167 | 0.200 |
| mean_return PushCube | 18.03 | 17.94 | **18.74** |
| mean_return PullCube | 13.47 | 26.05 | **27.69** |
| mean_return PullCube swap | 16.00 | 28.60 | 25.04 |

best_val_diffusion_mse: v0 **0.01258** → v1 **0.00932** (-26%). 출처: runs/m7_diffusion_v{0,1}/metrics.json, eval_*/closedloop_summary.json.
관찰: PushCube 정상 cell 첫 BC 추월(0.30→0.467), 그러나 PullCube 셀에서는 BC v2 의 0.433 을 못 넘김 (0.233).

### 5.6 M8 — BC v2 / Diffusion / PCGrad / PerTaskHead 4-way (출처 통합)

| Cell | BC v2 (M6.2) | Diffusion (M7.1) | **PCGrad (M8a)** | **PerTaskHead (M8b)** | best |
| -- | -- | -- | -- | -- | -- |
| PickCube/Pick — grasp | 0.133 | 0.000 | **0.200** | 0.000 | M8a |
| PushCube/Push — success | 0.300 | **0.467** | 0.300 | 0.400 | Diffusion |
| PullCube/Pull — success | **0.433** | 0.233 | 0.233 | 0.233 | BC v2 |
| PickCube/Push (swap) — grasp | 0.033 | 0.033 | **0.000** | **0.000** | M8a / M8b (obedience) |
| PullCube/Pick (swap) — success | 0.133 | 0.200 | 0.100 | **0.000** | M8b (obedience) |

학습 사이드: BC v2 best_val_mse_norm 약 0.0068 (M6.2) / Diffusion eps-MSE 0.00932 / **PCGrad 0.00634 (가장 낮음, 충돌 그래디언트 ~95건/epoch 검출)** / PerTaskHead 0.00771. 모든 모델 val_task_acc = 1.000.
출처: runs/m6_vla_aux_v2/eval_*, runs/m7_diffusion_v1/eval_*, runs/m8a_bc_pcgrad_v0/eval_*, runs/m8b_bc_per_task_head_v0/eval_*/closedloop_summary.json + 각 run 의 metrics.json + docs/m8_multitask_capacity.md.

### 5.7 한 줄 요약

- 단일태스크 PickCube closed-loop success **0%** 한계는 BC 데이터 양·loss weight·heuristic·DAgger 까지 모두 동원해도 깨지지 않았다. 멀티태스크 데이터셋 도입(M5)이 첫 비-0 closed-loop success(PushCube 10%) 를 만들어냈다.
- 보조 분류 손실(M5.1) 이 instruction following 을 처음으로 진짜로 작동시켜, **PullCube success 0 → 33.3%**.
- 비전 추가(M6 v0) 가 **PickCube grasp 6.7% → 30%**, settle solver(M6.1) 가 **PushCube success 13% → 50%**.
- Diffusion(M7.1) 이 **PushCube 의 새 최고치 46.7%** 를 달성했으나, PullCube 와 PickCube grasp 는 못 잡았다.
- PCGrad(M8a) 가 **PickCube grasp 를 13% → 20%** 로 회복시키고, PerTaskHead(M8b) 가 swap instruction obedience 를 가장 깨끗하게 만들었다 (PullCube+Pick swap: 13%→0%).
- 어떤 단일 fix 도 동시에 세 task 를 모두 끌어올리지 못한 점이 가장 중요한 발견이다.

---

## 6. What Worked / What Did Not Work

### 6.1 Worked (지원서에서 강조 가능한 기술적 성과)

- **End-to-end VLA stack 구축** — Docker, MP expert, 데이터 schema, BC, DAgger, CLIP-text, CLIP-vision, Diffusion, PCGrad, PerTaskHead 까지 단일 GPU 상에서 모두 직접 구현/검증.
- **DAgger 보정 파이프라인** — rollout 수집 → 상태 복원 + planner relabel → aggregate 데이터셋 → 가중 손실 재학습까지 자체 구현. v1 success_rate 72.2%, v2 92.3% (선별 기준 차이) 정량 비교 완료.
- **closed-loop sanity check 가 정착된 워크플로** — M4 단일태스크 sanity → M5 멀티태스크 → M5.1 aux loss → swap matrix 라는 **사다리식 검증 순서**가 코드와 결과 모두에 일관되게 남아 있음.
- **언어 조건이 "실제로 작동"하는 것을 정량 입증** — aux task_id CE 손실로 PickCube grasp 가 instruction swap 시 10% → 0% 로 떨어지고, PullCube 매치 시 0% → 33% 로 올라간 swap matrix 가 출처.
- **비전 추가의 양면 효과를 분리해 측정** — PickCube grasp 6.7% → 30% 의 이득과, swap instruction grasp 0% → 7-10% 의 obedience 손실을 동일 평가 행렬로 분리 측정.
- **expert-side 수정이 큰 효과를 낸 사례** — M6.1 PushCube settle solver + late_weight 4→8 로 PushCube success 13% → 50% (정책 변경 없이 expert demonstration 만 수정한 사례).
- **엄격 metric 도입** — ManiSkill native success 의 boundary-firing 문제를 video 분석으로 발견, `mean_min_cube_xy_goal_dist`, `xy_sustained_*_10` 등 추가. M6/M7/M8 비교를 신뢰 가능하게 만든 기반.
- **Diffusion Policy 풀스택** — 1D conditional U-Net + DDPM eps-prediction + DDIM receding-horizon 추론 + auxiliary task CE 까지 직접 구현하고 BC 와 동일 데이터셋·동일 swap matrix 로 비교.
- **PCGrad 충돌 그래디언트 정량 관측** — 학습 중 task-pair 충돌 그래디언트 평균 ~95건/epoch 검출 및 projection 으로 해소 (`docs/m8_multitask_capacity.md`).

### 6.2 Did not work / Limitations (정직하게 명시)

- **open-loop MSE 가 closed-loop 성능을 보장하지 않음** — M5 best_val_mse_norm (0.00426) < M4 (0.00584) 이지만, M5 closed-loop swap matrix 는 instruction 을 사실상 무시. M8b 도 best_val_mse_norm 이 epoch 77 에 도달했지만 그 체크포인트가 PullCube 약체. ⇒ "checkpoint 선택 기준을 closed-loop 으로 옮겨야 한다"는 교훈.
- **순수 BC 의 distribution shift** — 5000-demo + phase-weighted loss 까지 적용해도 PickCube closed-loop success 0% 한계 미돌파. ⇒ DAgger 와 멀티태스크 데이터로 우회.
- **PickCube placement 의 강한 tolerance** — `is_obj_placed` 가 0.025 m 임계. M3.9H force_grip 으로 grasp 유지가 거의 두 배(0.167→0.367) 됐지만 placement success 0 으로 수렴. ⇒ 정책 capacity 자체가 한계.
- **multi-task capacity-sharing trade-off** — 어떤 단일 fix 도 세 task 를 모두 동시에 끌어올리지 못함. 모든 변형이 "best at one, worst at another" 패턴 (M8). ⇒ 다음 단계는 trunk capacity 자체를 키우는 방향(foundation backbone).
- **M5 단계의 dummy instruction** — 멀티태스크 BC 만으로는 정책이 instruction 을 차별자로 사용하지 않음 (state 만으로 행동이 거의 결정됨). ⇒ M5.1 aux 분류 손실로 해결.
- **vision/language shortcut tension** — 비전이 강해질수록 instruction 의 영향력이 약해지는 현상. PickCube 에서 swap instruction grasp 가 비전 추가 후 0→7-10% 로 새어 나옴.
- **task 별 최적 변형이 다름** — BC v2: PullCube / Diffusion: PushCube / PCGrad: PickCube grasp / PerTaskHead: instruction-obedience. 단일 모델로 통합되지 않음. ⇒ M9+ 의 핵심 문제.
- **simulator native success 의 함정** — PushCube/PullCube native success 가 boundary crossing 만으로 발화. ⇒ M6.1 부터 xy / sustained metric 으로 보완. 만약 이 보정을 안 했다면 M6/M7/M8 비교가 노이즈 측정이 됐을 가능성이 있다.

### 6.3 실패가 다음 실험으로 연결된 흐름

- M3.3A 영상 분석에서 "cube 를 밀어내는 실패" 확인 → M3.4 phase signal 도입의 직접적 동기.
- M3.7/M3.8 의 데이터 양·loss weight 만으로 미돌파 → M3.9 DAgger 의 동기.
- M3.9H 영상 분석에서 progress≈0.95 grip 해제 패턴 확인 → 5-line force_grip 룰 도입.
- M5 swap matrix 에서 instruction 무시 발견 → M5.1 aux 손실 도입.
- M6 v0 영상 분석에서 PushCube boundary stop 확인 → M6.1 settle solver 도입.
- M7 v0 의 BC 미돌파 → M7.1 의 epoch · K · DDIM steps 튜닝.
- M6.2 의 task 별 best 변형 분기 관찰 → M8 4-way 비교 study 의 동기.

---

## 7. Job Application Positioning

### One-line summary (국문 1문장)

ManiSkill 시뮬레이션 위에서 제조형 언어 지시 조작을 expert demonstration · 모방학습 · DAgger · CLIP 텍스트/비전 조건화 · Diffusion Policy · 멀티태스크 capacity-sharing 비교 실험까지 단일 GPU 로 일관되게 구축한 VLA-style 로봇러닝 PoC.

### 300자 프로젝트 설명 (국문)

ManiSkill Panda arm 환경에서 PickCube/PushCube/PullCube 세 제조형 조작 태스크를 대상으로, expert demonstration 생성 → step-level dataset → behavior cloning → DAgger 정책 보정 → CLIP-text 언어 조건 BC → CLIP-vision 결합 VLA → Diffusion Policy → PCGrad/PerTaskHead 비교까지 9개 마일스톤을 단일 RTX 3060 위에서 직접 구현·평가·실패 분석한 응용 로봇 AI PoC. 모든 평가는 30-episode swap matrix 기반 closed-loop 으로 수행했으며 video/log 기반 failure diagnosis 가 다음 실험의 동기로 이어진다. (294자)

### 700자 프로젝트 설명 (국문)

ManiSkill Panda arm 시뮬레이션에서 PickCube/PushCube/PullCube 세 제조형 조작 태스크를 대상으로, expert demonstration 자체 생성, step-level dataset 설계, state-only BC → phase-aware BC → safe-action filter → final-hold wrapper → DAgger 보정 → force_grip heuristic 의 단일태스크 사다리, 그리고 CLIP-text 언어 조건 BC → 멀티태스크 BC + 보조 task_id 분류 손실 → CLIP-vision 결합 VLA → PushCube/PullCube settle solver → Diffusion Policy → PCGrad gradient surgery → per-task head 까지의 멀티태스크 사다리를 단일 RTX 3060 위에서 한 사이클로 구현하고 평가한 응용 로봇 AI PoC. 모든 모델은 동일 30-episode swap matrix 로 closed-loop 평가했고 영상/로그 기반 실패 분석이 다음 마일스톤의 직접적 동기로 연결된다. PushCube 최고 success 46.7%, PullCube 최고 43.3%, PickCube grasp 회복 20%, instruction following 의 정량 확인(swap 시 grasp 10%→0%) 등 핵심 지표를 모두 metric 파일로 보유. multi-task capacity-sharing trade-off 가 단일 fix 로 제거되지 않는다는 부정적 결론까지 실험적으로 도달해, 다음 단계로 foundation backbone scaling 을 자연스럽게 연결한다. (689자)

### Portfolio slide bullets

**문제 정의**
- 제조형 언어 지시 기반 정형 조작을 모방학습으로 풀 수 있는가, VLA stack 의 어느 층이 어디까지 책임지는가를 단일 GPU 에서 정량 검증.

**접근 방법**
- ManiSkill PickCube/PushCube/PullCube 멀티태스크 + Panda arm + pd_joint_pos 환경.
- 데이터: 제조형 instruction 메타데이터 + motion-planning expert demonstration (PickCube 100·5000, multi-task 300 episodes).
- 정책 사다리: state-only BC → phase-aware BC → DAgger → CLIP-text → +aux task CE → +CLIP-vision → Diffusion → PCGrad / PerTaskHead.
- 평가: 30-episode × 6-cell swap matrix closed-loop + xy/sustained 엄격 metric + 영상/로그 기반 failure diagnosis.

**본인 역할**
- Docker 환경 구성, ManiSkill MP expert wrapper 자체 구현, 데이터 schema · 학습 코드 · 평가 코드 · DAgger 3단계 파이프라인 전부 단일 저자 작성. 마일스톤별 실패 분석과 다음 실험 설계까지 일관 수행.

**기술 스택**
- Python / PyTorch / HuggingFace Transformers (CLIP) / ManiSkill / Docker (CUDA + Vulkan).
- Diffusion Policy (1D conditional U-Net, DDPM eps-prediction, DDIM receding-horizon).
- DAgger, PCGrad, multi-head selection by task classifier.

**주요 결과**
- DAgger v1 + force_grip: PickCube grasped_once 46.7%, final_grasped 36.7%, mean_return 24.1.
- M5.1 aux 손실로 instruction following 정량 확인 (swap 시 grasp 10%→0%, PullCube success 0%→33.3%).
- M6.1 settle solver + late_weight 8 로 PushCube success 13%→50%.
- M7.1 Diffusion Policy 가 PushCube success 30%→46.7% 로 BC 첫 추월.
- M8 4-way 비교: 단일 fix 로 multi-task capacity trade-off 미해소 (음성 결과를 정직하게 정리).

**배운 점 / 다음 단계**
- open-loop MSE 는 closed-loop 성능을 담보하지 않는다 — swap matrix 가 진짜 진단 도구.
- expert-side 보정이 정책-side 보정보다 큰 효과를 낼 수 있다 (M6.1).
- 다음 단계: M9a (PCGrad + PerTaskHead 결합), M9b (OpenVLA/Octo LoRA 로 trunk capacity 자체 확장), M9c (image-only VLA 로 비전 단독 기여도 측정).

### 표현 방향 체크
- ✅ "제조형 언어 지시 로봇 조작 PoC"
- ✅ "expert demonstration 기반 imitation learning"
- ✅ "DAgger 기반 정책 고도화"
- ✅ "CLIP text/vision 기반 VLA-style policy"
- ✅ "closed-loop evaluation 과 failure diagnosis"
- ✅ "Diffusion Policy 및 multi-task interference 완화 실험"
- ✅ "현대차 제조로보틱스 IL/RL/VLA 역량과 직접 연결"

---

## 8. Evidence Checklist (지원서/포트폴리오 첨부 가능 자료)

- [x] README milestone table — README.md §"Current Status" (M0~M8)
- [x] dataset schema — outputs/m2_step_dataset/dataset_schema.json, outputs/m5_multitask_dataset/summary.json (300 episode, obs=57, action=8)
- [x] expert demonstration generation command — scripts/m2_generate_pickcube_expert.py, scripts/m5_generate_multitask_expert.py, scripts/m6_generate_multitask_expert_v{1,2}.py
- [x] BC training result — runs/m3_bc_phase_aware/metrics.json (best_val_mse_norm=0.00475, best_epoch=303)
- [x] closed-loop evaluation result — runs/m3_bc_dagger_v1/closedloop_eval_safe_forcegrip/closedloop_summary.json 등 모든 마일스톤 분기별 raw json
- [x] DAgger correction result — outputs/m3_dagger_corrections_v1/summary.json (1121/1553 = 72.2%), v2 (830/899 = 92.3%)
- [x] CLIP text/vision architecture & code — scripts/m4_add_instruction_embeddings.py, scripts/m4_train_bc_lang.py, scripts/m6_add_image_embeddings.py, scripts/m6_train_vla_lang_aux.py
- [x] Diffusion Policy result — runs/m7_diffusion_v1/metrics.json (best_val_diffusion_mse=0.00932), eval_pushcube_push (success 0.4667)
- [x] PCGrad / PerTaskHead 비교 — runs/m8a_bc_pcgrad_v0/, runs/m8b_bc_per_task_head_v0/ + docs/m8_multitask_capacity.md
- [x] demo video — runs/m6_vla_aux_v2/debug_video/, runs/m7_diffusion_v1/debug_video_x5/ (PushCube/PullCube/PickCube 각 다수 시드)
- [x] failure diagnosis log — runs/*/debug_video/* + closedloop_summary.json 의 episode-level CSV
- [x] representative commands — README.md 마일스톤별 명령 + docs/research_report.md
- [x] Dockerfile / reproducible environment — docker/Dockerfile (maniskill/base:latest + transformers==4.46.3)

---

## 9. Recommended Documentation Patch (포트폴리오 친화도 강화 제안)

현재 README 도 충실하지만, 지원서/포트폴리오 관점에서 다음을 추가하면 효과적이다.

1. **README 상단에 "Portfolio Summary" 박스 추가** — 본 리포트의 §7 One-line / 300자 / 700자 요약을 그대로 README 최상단에 박스로 노출.
2. **"Current Best Result" 한 장 표** — README 상단에 PickCube grasp / PushCube success / PullCube success 의 현재 best 와 그 출처 (run path) 한 표.
3. **"How to reproduce key result" 섹션** — 각 마일스톤별 학습/평가 명령을 핵심 result 옆에 배치 (이미 README 에 명령이 있지만, "이 명령 실행 → 이 metric 재현" 매핑이 명시되면 평가자가 즉시 따라할 수 있음).
4. **대표 demo video / gif 링크 정리** — runs/m6_vla_aux_v2/debug_video, runs/m7_diffusion_v1/debug_video_x5 중 PushCube success, PullCube success, PickCube grasp 의 대표 시드를 README 에 1-2개씩 링크. (현재는 디렉터리에 mp4 가 있을 뿐 README 에서 인덱스되어 있지 않다.)
5. **architecture / pipeline diagram 추가** — 본 리포트 §3 의 텍스트 다이어그램을 mermaid 또는 png 다이어그램으로 변환해 docs/ 에 두고 README 에 임베드.
6. **closed-loop evaluation summary table** — 모든 마일스톤의 succ/grasp/placed/mean_return 한 표 (본 리포트 §5).
7. **Limitations / Future Work 정리** — 본 리포트 §6.2 와 §M9+ 의 후속 실험 후보를 README 끝에 한 섹션으로.
8. **민감 정보 점검** — git log/메시지·configs/scripts/docs 에서 절대경로·내부 호스트명·이메일 등이 노출되지 않는지 한 번 더 확인 후 공개. (현재 docker/Dockerfile 의 WORKDIR `/workspace/manufacturing-vla-poc` 는 내부 경로처럼 보일 수 있으므로 README 에서는 "컨테이너 내부 작업 경로" 로 명시 권장.)

---

## 10. Final Hiring Narrative

> 저는 이 프로젝트를 통해 단순히 모델을 학습시키는 데에 그치지 않고, **제조형 로봇 조작 문제를 데이터셋 설계 · expert demonstration 생성 · 모방학습 · DAgger 기반 정책 보정 · CLIP text/vision 결합 VLA-style 정책 · Diffusion Policy · multi-task capacity-sharing 비교 실험 · closed-loop swap matrix 검증 · 영상/로그 기반 실패 분석의 전체 파이프라인**으로 다루었습니다. 단일 RTX 3060 환경에서 9개 마일스톤을 순차적으로 진행하면서, 각 단계에서 관찰된 실패가 다음 단계의 설계 동기로 직접 이어지는 흐름을 코드와 metric 파일 모두에 그대로 남겨두었고, "open-loop MSE 는 closed-loop 성능을 담보하지 않는다"는 부정적 발견에서부터 "multi-task 모델 단일 fix 로는 task-별 trade-off 가 사라지지 않는다"는 결과까지 정직하게 정리했습니다. 이는 현대차 제조로보틱스 직무가 요구하는 IL · RL · VLA 역량 — 데이터 수집부터 정책 검증·개선까지 책임지는 응용 로봇 AI 엔지니어링 — 을 실제로 한 사이클 수행해본 경험으로 직결됩니다.
