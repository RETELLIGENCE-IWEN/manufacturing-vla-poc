"""M5-B: convert per-task expert HDF5 files into a unified multi-task M3-ready dataset.

Inputs : a directory containing per-task subdirs of the form
           <record_dir>/{env_id}/motionplanning/<traj_name>.h5
Outputs: a single dataset directory with the same layout as
         outputs/m3_phase_aware_dataset_100 but containing episodes from
         multiple env_ids, each with task-specific instruction templates
         and a `task_id` metadata field for later analysis.

The state layout is unified to (panda 31 + cube 13 + goal 13 = 57).
- PickCube actor names: "cube", "goal_site"
- PushCube actor names: "cube", "goal_region"
- PullCube actor names: "cube", "goal_region"
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml


GOAL_ACTOR_CANDIDATES = ["goal_site", "goal_region"]


def load_multitask_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sample_instruction(task_cfg: dict[str, Any], rng: random.Random) -> dict[str, str]:
    obj = rng.choice(task_cfg["object_names"])
    tgt = rng.choice(task_cfg["target_names"])
    template = rng.choice(task_cfg["instruction_templates"])
    return {
        "instruction": template.format(object_name=obj, target_name=tgt),
        "object_name": obj,
        "target_name": tgt,
        "template": template,
    }


def build_state_obs(traj_group: h5py.Group, env_id: str) -> np.ndarray:
    actions = np.asarray(traj_group["actions"], dtype=np.float32)
    T = actions.shape[0]

    panda = np.asarray(traj_group["env_states"]["articulations"]["panda"], dtype=np.float32)[:T]
    cube = np.asarray(traj_group["env_states"]["actors"]["cube"], dtype=np.float32)[:T]

    actors = traj_group["env_states"]["actors"]
    goal_key = next((k for k in GOAL_ACTOR_CANDIDATES if k in actors), None)
    if goal_key is None:
        raise KeyError(f"No goal actor found in {env_id} traj. tried {GOAL_ACTOR_CANDIDATES}, got {list(actors.keys())}")
    goal = np.asarray(actors[goal_key], dtype=np.float32)[:T]

    return np.concatenate([panda, cube, goal], axis=1).astype(np.float32)


def make_splits(num_episodes: int, val_ratio: float, seed: int, group_ids: list[int]) -> dict[str, list[int]]:
    """Per-group stratified split (so each task gets val coverage)."""
    rng = random.Random(seed)
    groups: dict[int, list[int]] = {}
    for ep_id, gid in enumerate(group_ids):
        groups.setdefault(gid, []).append(ep_id)

    train_ids: list[int] = []
    val_ids: list[int] = []
    for gid, ids in groups.items():
        ids = list(ids)
        rng.shuffle(ids)
        num_val = max(1, int(round(len(ids) * val_ratio)))
        val_ids.extend(ids[:num_val])
        train_ids.extend(ids[num_val:])

    return {"train": sorted(train_ids), "val": sorted(val_ids)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-dir", type=str, default="outputs/m5_expert_demos_multitask")
    parser.add_argument("--config", type=str, default="configs/manufacturing_multitask_v0.yaml")
    parser.add_argument("--out-dir", type=str, default="outputs/m5_multitask_dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    record_root = Path(args.record_dir)
    cfg = load_multitask_config(Path(args.config))
    out_dir = Path(args.out_dir)
    ep_dir = out_dir / "episodes"
    ep_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    episode_records: list[dict[str, Any]] = []
    task_group_ids: list[int] = []
    episode_id = 0

    for task_idx, task_cfg in enumerate(cfg["tasks"]):
        env_id = task_cfg["env_id"]
        task_dir = record_root / env_id / "motionplanning"
        h5_files = sorted(task_dir.glob("*.h5"))
        if not h5_files:
            raise FileNotFoundError(f"No H5 files under {task_dir}")
        if len(h5_files) > 1:
            print(f"[warn] multiple H5 files under {task_dir}, using all of them.")

        for h5_path in h5_files:
            with h5py.File(h5_path, "r") as h5:
                traj_keys = sorted(
                    (k for k in h5.keys() if k.startswith("traj_")),
                    key=lambda x: int(x.split("_")[1]),
                )

                for traj_key in traj_keys:
                    traj = h5[traj_key]
                    actions = np.asarray(traj["actions"], dtype=np.float32)
                    rewards = np.zeros((actions.shape[0],), dtype=np.float32)
                    terminated = np.asarray(traj["terminated"], dtype=np.bool_)
                    truncated = np.asarray(traj["truncated"], dtype=np.bool_)
                    success = np.asarray(traj["success"], dtype=np.bool_)

                    obs = build_state_obs(traj, env_id)
                    if obs.shape[0] != actions.shape[0]:
                        raise ValueError(f"Length mismatch in {traj_key}@{env_id}: obs={obs.shape}, actions={actions.shape}")

                    instr = sample_instruction(task_cfg, rng)

                    ep_name = f"ep_{episode_id:06d}"
                    npz_path = ep_dir / f"{ep_name}.npz"
                    np.savez_compressed(
                        npz_path,
                        obs=obs,
                        actions=actions,
                        rewards=rewards,
                        terminated=terminated,
                        truncated=truncated,
                        success=success,
                    )

                    record = {
                        "episode_id": episode_id,
                        "source_traj_key": traj_key,
                        "source_h5": str(h5_path),
                        "npz_path": str(npz_path),
                        "task_family": cfg["task_family"],
                        "task_id": task_cfg["id"],
                        "task_verb": task_cfg["verb"],
                        "base_env_id": env_id,
                        "instruction": instr["instruction"],
                        "object_name": instr["object_name"],
                        "target_name": instr["target_name"],
                        "template": instr["template"],
                        "num_steps": int(actions.shape[0]),
                        "obs_dim": int(obs.shape[1]),
                        "action_dim": int(actions.shape[1]),
                        "success_once": bool(success.any()),
                        "final_success": bool(success[-1]) if success.size > 0 else False,
                        "dataset_type": "expert_motionplanning",
                    }
                    episode_records.append(record)
                    task_group_ids.append(task_idx)
                    episode_id += 1

    episodes_jsonl_path = out_dir / "episodes.jsonl"
    with episodes_jsonl_path.open("w", encoding="utf-8") as f:
        for record in episode_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    splits = make_splits(len(episode_records), val_ratio=args.val_ratio, seed=args.seed, group_ids=task_group_ids)
    with (out_dir / "splits.json").open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    obs_dim = episode_records[0]["obs_dim"] if episode_records else 0
    action_dim = episode_records[0]["action_dim"] if episode_records else 0

    schema = {
        "milestone": "M5B",
        "format_version": 1,
        "description": "Multi-task (Pick/Push/Pull) expert motion-planning dataset.",
        "tasks": [t["env_id"] for t in cfg["tasks"]],
        "episode_npz_fields": {
            "obs": "[T, 57] float32; concatenated panda/cube/goal env state at time t",
            "actions": "[T, 8] float32; pd_joint_pos expert action",
            "rewards": "[T] float32; placeholder zeros",
            "terminated": "[T] bool",
            "truncated": "[T] bool",
            "success": "[T] bool",
        },
        "obs_layout": {"panda": 31, "cube": 13, "goal": 13, "total": obs_dim},
        "action_layout": {"total": action_dim, "note": "pd_joint_pos"},
    }
    with (out_dir / "dataset_schema.json").open("w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    summary = {
        "milestone": "M5B",
        "out_dir": str(out_dir),
        "config": args.config,
        "num_episodes": len(episode_records),
        "per_task_counts": {
            task_cfg["env_id"]: sum(1 for r in episode_records if r["base_env_id"] == task_cfg["env_id"])
            for task_cfg in cfg["tasks"]
        },
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "splits": {k: len(v) for k, v in splits.items()},
        "success_rate_once": float(np.mean([r["success_once"] for r in episode_records])) if episode_records else 0.0,
        "mean_steps": float(np.mean([r["num_steps"] for r in episode_records])) if episode_records else 0.0,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[done] M5B multi-task dataset built")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
