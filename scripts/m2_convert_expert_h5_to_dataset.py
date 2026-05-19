from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml


def load_task_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sample_instruction(cfg: dict[str, Any], rng: random.Random) -> dict[str, str]:
    obj = rng.choice(cfg["objects"])
    tgt = rng.choice(cfg["targets"])
    template = rng.choice(cfg["instruction_templates"])

    instruction = template.format(
        object_name=obj["display_name"],
        target_name=tgt["display_name"],
    )

    return {
        "instruction": instruction,
        "object_id": obj["id"],
        "object_name": obj["display_name"],
        "target_id": tgt["id"],
        "target_name": tgt["display_name"],
        "template": template,
    }


def build_state_obs(traj_group: h5py.Group) -> np.ndarray:
    """
    Build M3-ready state observation from ManiSkill env_states.

    actions has length T.
    env_states has length T+1.
    We use env_states[t] as observation before action[t].
    """
    actions = np.asarray(traj_group["actions"], dtype=np.float32)
    T = actions.shape[0]

    panda = np.asarray(traj_group["env_states"]["articulations"]["panda"], dtype=np.float32)
    cube = np.asarray(traj_group["env_states"]["actors"]["cube"], dtype=np.float32)
    goal_site = np.asarray(traj_group["env_states"]["actors"]["goal_site"], dtype=np.float32)

    panda_t = panda[:T]
    cube_t = cube[:T]
    goal_t = goal_site[:T]

    obs = np.concatenate([panda_t, cube_t, goal_t], axis=1).astype(np.float32)
    return obs


def make_splits(num_episodes: int, val_ratio: float, seed: int) -> dict[str, list[int]]:
    ids = list(range(num_episodes))
    rng = random.Random(seed)
    rng.shuffle(ids)

    if num_episodes <= 1:
        return {"train": ids, "val": []}

    num_val = max(1, int(round(num_episodes * val_ratio)))
    val_ids = sorted(ids[:num_val])
    train_ids = sorted(ids[num_val:])

    return {
        "train": train_ids,
        "val": val_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/manufacturing_pick_place_v0.yaml")
    parser.add_argument("--out-dir", type=str, default="outputs/m2_expert_dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    h5_path = Path(args.h5)
    cfg_path = Path(args.config)
    out_dir = Path(args.out_dir)
    episode_dir = out_dir / "episodes"

    if not h5_path.exists():
        raise FileNotFoundError(h5_path)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    episode_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_task_config(cfg_path)
    rng = random.Random(args.seed)

    episode_records: list[dict[str, Any]] = []

    with h5py.File(h5_path, "r") as h5:
        traj_keys = sorted(
            [k for k in h5.keys() if k.startswith("traj_")],
            key=lambda x: int(x.split("_")[1]),
        )

        episodes_jsonl = out_dir / "episodes.jsonl"

        with episodes_jsonl.open("w", encoding="utf-8") as f_jsonl:
            for episode_id, traj_key in enumerate(traj_keys):
                traj = h5[traj_key]

                actions = np.asarray(traj["actions"], dtype=np.float32)
                rewards = np.zeros((actions.shape[0],), dtype=np.float32)
                terminated = np.asarray(traj["terminated"], dtype=np.bool_)
                truncated = np.asarray(traj["truncated"], dtype=np.bool_)
                success = np.asarray(traj["success"], dtype=np.bool_)

                obs = build_state_obs(traj)

                if obs.shape[0] != actions.shape[0]:
                    raise ValueError(
                        f"Length mismatch in {traj_key}: obs={obs.shape}, actions={actions.shape}"
                    )

                task = sample_instruction(cfg, rng)

                ep_name = f"ep_{episode_id:06d}"
                npz_path = episode_dir / f"{ep_name}.npz"

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
                    "base_env_id": cfg["base_env_id"],
                    "instruction": task["instruction"],
                    "object_id": task["object_id"],
                    "object_name": task["object_name"],
                    "target_id": task["target_id"],
                    "target_name": task["target_name"],
                    "template": task["template"],
                    "num_steps": int(actions.shape[0]),
                    "obs_dim": int(obs.shape[1]),
                    "action_dim": int(actions.shape[1]),
                    "success_once": bool(success.any()),
                    "final_success": bool(success[-1]) if success.size > 0 else False,
                    "dataset_type": "expert_motionplanning",
                }

                episode_records.append(record)
                f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

    splits = make_splits(len(episode_records), val_ratio=args.val_ratio, seed=args.seed)

    with (out_dir / "splits.json").open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    success_rate = float(np.mean([r["success_once"] for r in episode_records])) if episode_records else 0.0
    mean_steps = float(np.mean([r["num_steps"] for r in episode_records])) if episode_records else 0.0

    obs_dim = episode_records[0]["obs_dim"] if episode_records else 0
    action_dim = episode_records[0]["action_dim"] if episode_records else 0

    schema = {
        "milestone": "M2C",
        "format_version": 1,
        "description": "Expert motion-planning trajectories converted into M3-ready state-action dataset.",
        "source_h5": str(h5_path),
        "episode_npz_fields": {
            "obs": "[T, obs_dim] float32; concatenated panda/cube/goal_site env state at time t",
            "actions": "[T, action_dim] float32; ManiSkill expert action",
            "rewards": "[T] float32; placeholder zeros for expert BC dataset",
            "terminated": "[T] bool",
            "truncated": "[T] bool",
            "success": "[T] bool",
        },
        "obs_layout": {
            "panda": "31 dims",
            "cube": "13 dims",
            "goal_site": "13 dims",
            "total": obs_dim,
        },
        "action_layout": {
            "total": action_dim,
            "note": "ManiSkill pd_joint_pos action from expert motion-planning trajectory.",
        },
    }

    with (out_dir / "dataset_schema.json").open("w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    summary = {
        "milestone": "M2C",
        "source_h5": str(h5_path),
        "config": str(cfg_path),
        "out_dir": str(out_dir),
        "num_episodes": len(episode_records),
        "success_rate_once": success_rate,
        "mean_steps": mean_steps,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "episodes_jsonl": str(out_dir / "episodes.jsonl"),
        "splits_json": str(out_dir / "splits.json"),
        "dataset_schema_json": str(out_dir / "dataset_schema.json"),
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[done] converted expert H5 to M3-ready dataset")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()