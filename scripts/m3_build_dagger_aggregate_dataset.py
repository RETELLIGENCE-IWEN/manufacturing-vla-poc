from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_splits(path: Path) -> dict[str, list[int]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {
        "train": [int(x) for x in raw.get("train", [])],
        "val": [int(x) for x in raw.get("val", [])],
    }


def make_dagger_splits(
    ids: list[int],
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not ids:
        return [], []
    if val_ratio <= 0.0 or len(ids) <= 1:
        return ids, []

    rng = np.random.default_rng(seed)
    shuffled = np.asarray(ids, dtype=np.int32)
    rng.shuffle(shuffled)
    num_val = max(1, int(round(len(ids) * val_ratio)))
    val = sorted(shuffled[:num_val].astype(int).tolist())
    train = sorted(shuffled[num_val:].astype(int).tolist())
    return train, val


def rebuild_expert_obs_with_phase_horizon(
    data: Any,
    phase_horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "state_obs" not in data or "prev_action" not in data:
        obs = data["obs"].astype(np.float32)
        state_obs = obs[:, :57].astype(np.float32)
        prev_action = obs[:, 58:66].astype(np.float32)
        return obs, state_obs, prev_action

    state_obs = data["state_obs"].astype(np.float32)
    prev_action = data["prev_action"].astype(np.float32)
    T = state_obs.shape[0]
    denom = max(1, phase_horizon - 1)
    progress = np.minimum(np.arange(T, dtype=np.float32) / float(denom), 1.0).reshape(T, 1)
    obs = np.concatenate([state_obs, progress, prev_action], axis=1).astype(np.float32)
    return obs, state_obs, prev_action


def copy_episode(
    out_episode_dir: Path,
    out_episode_id: int,
    obs: np.ndarray,
    actions: np.ndarray,
    sample_weight: np.ndarray,
    source_is_dagger: np.ndarray,
    source_episode_id: int,
    source_dataset: str,
    extra_fields: dict[str, np.ndarray] | None = None,
) -> Path:
    out_path = out_episode_dir / f"ep_{out_episode_id:06d}.npz"
    fields: dict[str, np.ndarray] = {
        "obs": obs.astype(np.float32),
        "actions": actions.astype(np.float32),
        "sample_weight": sample_weight.astype(np.float32),
        "source_is_dagger": source_is_dagger.astype(np.bool_),
        "source_episode_id": np.full((obs.shape[0],), source_episode_id, dtype=np.int32),
        "source_dataset_id": np.full((obs.shape[0],), source_dataset, dtype="<U32"),
    }
    if extra_fields is not None:
        fields.update(extra_fields)
    np.savez_compressed(out_path, **fields)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-dataset-dir", type=str, default="outputs/m3_phase_aware_dataset_100")
    parser.add_argument("--dagger-dataset-dir", type=str, default="outputs/m3_dagger_corrections_v0")
    parser.add_argument("--out-dir", type=str, default="outputs/m3_agg_phase_dagger_dataset_v0")
    parser.add_argument(
        "--expert-progress-mode",
        type=str,
        default="phase_horizon",
        choices=["original", "phase_horizon"],
    )
    parser.add_argument("--phase-horizon", type=int, default=80)
    parser.add_argument("--expert-weight", type=float, default=1.0)
    parser.add_argument("--dagger-val-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    expert_dir = Path(args.expert_dataset_dir)
    dagger_dir = Path(args.dagger_dataset_dir)
    out_dir = Path(args.out_dir)
    out_episode_dir = out_dir / "episodes"

    if not expert_dir.exists():
        raise FileNotFoundError(expert_dir)
    if not dagger_dir.exists():
        raise FileNotFoundError(dagger_dir)

    out_episode_dir.mkdir(parents=True, exist_ok=True)

    expert_splits = load_splits(expert_dir / "splits.json")
    expert_split_by_id: dict[int, str] = {}
    for split_name, ids in expert_splits.items():
        for episode_id in ids:
            expert_split_by_id[int(episode_id)] = split_name

    aggregate_train: list[int] = []
    aggregate_val: list[int] = []
    episode_records: list[dict[str, Any]] = []
    action_min: np.ndarray | None = None
    action_max: np.ndarray | None = None

    def update_action_bounds(actions: np.ndarray) -> None:
        nonlocal action_min, action_max
        current_min = actions.min(axis=0)
        current_max = actions.max(axis=0)
        if action_min is None:
            action_min = current_min
            action_max = current_max
        else:
            action_min = np.minimum(action_min, current_min)
            action_max = np.maximum(action_max, current_max)

    next_episode_id = 0
    expert_paths = sorted((expert_dir / "episodes").glob("ep_*.npz"))
    for src_path in expert_paths:
        source_episode_id = int(src_path.stem.split("_")[1])
        data = np.load(src_path)
        actions = data["actions"].astype(np.float32)

        if args.expert_progress_mode == "phase_horizon":
            obs, state_obs, prev_action = rebuild_expert_obs_with_phase_horizon(
                data=data,
                phase_horizon=args.phase_horizon,
            )
        else:
            obs = data["obs"].astype(np.float32)
            state_obs = data["state_obs"].astype(np.float32) if "state_obs" in data else obs[:, :57]
            prev_action = data["prev_action"].astype(np.float32) if "prev_action" in data else obs[:, 58:66]

        sample_weight = np.full((obs.shape[0],), args.expert_weight, dtype=np.float32)
        source_is_dagger = np.zeros((obs.shape[0],), dtype=np.bool_)
        extra_fields = {
            "state_obs": state_obs.astype(np.float32),
            "prev_action": prev_action.astype(np.float32),
        }

        out_path = copy_episode(
            out_episode_dir=out_episode_dir,
            out_episode_id=next_episode_id,
            obs=obs,
            actions=actions,
            sample_weight=sample_weight,
            source_is_dagger=source_is_dagger,
            source_episode_id=source_episode_id,
            source_dataset="expert",
            extra_fields=extra_fields,
        )
        update_action_bounds(actions)

        split_name = expert_split_by_id.get(source_episode_id, "train")
        if split_name == "val":
            aggregate_val.append(next_episode_id)
        else:
            aggregate_train.append(next_episode_id)

        episode_records.append(
            {
                "episode_id": next_episode_id,
                "source_episode_id": source_episode_id,
                "source_npz_path": str(src_path),
                "npz_path": str(out_path),
                "dataset_type": "phase_aware_expert_bc",
                "split": split_name,
                "num_steps": int(obs.shape[0]),
                "obs_dim": int(obs.shape[1]),
                "action_dim": int(actions.shape[1]),
                "mean_sample_weight": float(np.mean(sample_weight)),
            }
        )
        next_episode_id += 1

    dagger_paths = sorted((dagger_dir / "episodes").glob("ep_*.npz"))
    dagger_new_ids: list[int] = []
    for src_path in dagger_paths:
        source_episode_id = int(src_path.stem.split("_")[1])
        data = np.load(src_path)
        obs = data["obs"].astype(np.float32)
        actions = data["actions"].astype(np.float32)
        sample_weight = (
            data["sample_weight"].astype(np.float32)
            if "sample_weight" in data
            else np.ones((obs.shape[0],), dtype=np.float32)
        )
        source_is_dagger = np.ones((obs.shape[0],), dtype=np.bool_)

        extra_fields = {
            "policy_action_env": data["policy_action_env"].astype(np.float32)
            if "policy_action_env" in data
            else np.zeros_like(actions, dtype=np.float32),
        }

        out_path = copy_episode(
            out_episode_dir=out_episode_dir,
            out_episode_id=next_episode_id,
            obs=obs,
            actions=actions,
            sample_weight=sample_weight,
            source_is_dagger=source_is_dagger,
            source_episode_id=source_episode_id,
            source_dataset="dagger",
            extra_fields=extra_fields,
        )
        update_action_bounds(actions)
        dagger_new_ids.append(next_episode_id)

        episode_records.append(
            {
                "episode_id": next_episode_id,
                "source_episode_id": source_episode_id,
                "source_npz_path": str(src_path),
                "npz_path": str(out_path),
                "dataset_type": "dagger_planner_correction",
                "split": "pending",
                "num_steps": int(obs.shape[0]),
                "obs_dim": int(obs.shape[1]),
                "action_dim": int(actions.shape[1]),
                "mean_sample_weight": float(np.mean(sample_weight)),
            }
        )
        next_episode_id += 1

    dagger_train, dagger_val = make_dagger_splits(
        ids=dagger_new_ids,
        val_ratio=args.dagger_val_ratio,
        seed=args.seed,
    )
    aggregate_train.extend(dagger_train)
    aggregate_val.extend(dagger_val)

    dagger_split_by_id = {episode_id: "train" for episode_id in dagger_train}
    dagger_split_by_id.update({episode_id: "val" for episode_id in dagger_val})
    for record in episode_records:
        if record["dataset_type"] == "dagger_planner_correction":
            record["split"] = dagger_split_by_id.get(record["episode_id"], "train")

    splits = {
        "train": sorted(aggregate_train),
        "val": sorted(aggregate_val),
    }
    with (out_dir / "splits.json").open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    with (out_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for record in episode_records:
            f.write(json.dumps(record) + "\n")

    if action_min is None or action_max is None:
        raise RuntimeError("No actions found while building aggregate dataset.")

    action_bounds = {
        "source_dataset": str(out_dir),
        "min": action_min.astype(float).tolist(),
        "max": action_max.astype(float).tolist(),
    }
    with (out_dir / "action_bounds.json").open("w", encoding="utf-8") as f:
        json.dump(action_bounds, f, indent=2)

    num_expert_episodes = len(expert_paths)
    num_dagger_episodes = len(dagger_paths)
    num_expert_transitions = int(
        sum(record["num_steps"] for record in episode_records if record["dataset_type"] == "phase_aware_expert_bc")
    )
    num_dagger_transitions = int(
        sum(record["num_steps"] for record in episode_records if record["dataset_type"] == "dagger_planner_correction")
    )

    schema = {
        "milestone": "M3.9D",
        "format_version": 1,
        "description": "Aggregated phase-aware expert and DAgger correction dataset.",
        "episode_npz_fields": {
            "obs": "[T, 66] float32; state_57 + progress_1 + prev_action_8",
            "actions": "[T, 8] float32; expert/planner action label",
            "sample_weight": "[T] float32; source-specific sample multiplier",
            "source_is_dagger": "[T] bool",
        },
        "expert_progress_mode": args.expert_progress_mode,
        "phase_horizon": args.phase_horizon,
    }

    with (out_dir / "dataset_schema.json").open("w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    summary = {
        "milestone": "M3.9D",
        "expert_dataset_dir": str(expert_dir),
        "dagger_dataset_dir": str(dagger_dir),
        "out_dir": str(out_dir),
        "expert_progress_mode": args.expert_progress_mode,
        "phase_horizon": args.phase_horizon,
        "expert_weight": args.expert_weight,
        "dagger_val_ratio": args.dagger_val_ratio,
        "num_episodes": len(episode_records),
        "num_expert_episodes": num_expert_episodes,
        "num_dagger_episodes": num_dagger_episodes,
        "num_expert_transitions": num_expert_transitions,
        "num_dagger_transitions": num_dagger_transitions,
        "num_train_episodes": len(splits["train"]),
        "num_val_episodes": len(splits["val"]),
        "obs_dim": 66,
        "action_dim": int(action_min.shape[0]),
        "episodes_jsonl": str(out_dir / "episodes.jsonl"),
        "splits_json": str(out_dir / "splits.json"),
        "dataset_schema_json": str(out_dir / "dataset_schema.json"),
        "action_bounds_json": str(out_dir / "action_bounds.json"),
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[done] M3.9D aggregate DAgger dataset created")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

