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