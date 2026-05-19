# Manufacturing VLA PoC

A compact applied robotics AI project for language-conditioned manufacturing-style manipulation.

This project connects:

- language instructions,
- robot observations,
- task/action representations,
- learned manipulation policies,
- simulation-based evaluation,

into a small VLA-style robotics pipeline.

## Goal

The goal is to build a small but reproducible PoC for manufacturing-oriented robotic manipulation.

Initial task concept:

> Pick a component and place it into a target fixture, bin, or inspection tray.

## Current Milestones

### M0 — Dockerized ManiSkill Bring-up

M0 verifies that ManiSkill can run inside Docker and execute a basic manipulation environment.

Command:

```bash
python scripts/m0_random_rollout.py \
  --env-id PickCube-v1 \
  --obs-mode state \
  --num-episodes 3 \
  --max-steps 200 \
  --seed 42
```

## 지금까지 한 일
M0 — Dockerized ManiSkill 환경 구축

Docker container 위에서 ManiSkill을 실행할 수 있게 했다.

완료한 것:

NVIDIA driver 설치 성공
Docker GPU 연결 성공
torch.cuda.is_available() == True
ManiSkill import OK
PickCube-v1 실행 확인

의미:

재현 가능한 GPU 기반 로봇 시뮬레이션 개발 환경을 만들었다.

M1 — 제조형 language task wrapper

기본 PickCube-v1 조작 환경 위에 제조 작업 느낌의 언어 지시와 task metadata를 붙였다.

예:

Pick the red component and place it into the left fixture.
Move the bolt-like part to the inspection tray.
Grasp the blue component, then put it in the right fixture.

의미:

단순 로봇 조작 task를 language-conditioned manufacturing task로 확장할 준비를 했다.

M2A — step-level dataset logger

random rollout이라도 학습용 데이터 구조로 저장할 수 있게 했다.

저장 구조:

instruction
observation
action
reward
terminated
truncated
success
task metadata

의미:

M3에서 imitation learning이나 behavior cloning으로 이어질 수 있는 데이터 포맷을 만들었다.

M2B — expert demonstration 생성

ManiSkill의 Panda motion-planning solver를 우회 실행해서 PickCube-v1 expert trajectory를 생성했다.

결과:

traj_0 ~ traj_4
actions: [T, 8]
success: [T]
env_states: [T+1, ...]

의미:

random action이 아니라, 실제 pick-and-place 성공 궤적을 확보했다.

M2C — expert H5를 학습용 dataset으로 변환

expert HDF5 trajectory를 우리가 만든 M3-ready dataset format으로 바꿨다.

최종 데이터 구조:

obs_t = panda state + cube state + goal_site state
action_t = expert action
success_t = success signal
instruction = 제조형 언어 지시

의미:

이제 behavior cloning 학습을 바로 시작할 수 있는 상태가 되었다.

현재 PoC 상태

지금 프로젝트는 여기까지 왔다.

M0: 환경 구축 완료
M1: 언어 지시 task wrapper 완료
M2A: step-level dataset logger 완료
M2B: expert demo 생성 완료
M2C: expert dataset 변환 완료

한 문장으로 말하면:

Dockerized ManiSkill 기반으로, 언어 지시가 붙은 제조형 로봇 조작 task를 정의하고, expert demonstration을 학습 가능한 state-action dataset으로 변환하는 파이프라인을 구축했다.