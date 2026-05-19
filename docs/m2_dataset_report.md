# M2 Dataset Report

## Goal

M2 builds a dataset foundation for language-conditioned robotic manipulation.

The dataset work has two parts:

1. Step-level dataset logging for the project-level VLA-style data contract.
2. Expert demonstration generation using ManiSkill motion-planning trajectories.

## M2A — Step-Level Dataset Logger

The project-level logger stores language-conditioned manipulation episodes as compressed NPZ files.

Generated files:

```text
outputs/m2_step_dataset/
  dataset_schema.json
  summary.json
  episodes.jsonl
  splits.json
  episodes/
    ep_000000.npz
    ep_000001.npz