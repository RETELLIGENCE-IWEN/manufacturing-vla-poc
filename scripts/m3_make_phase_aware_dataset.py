from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def make_splits(num_episodes: int, val_ratio: float, seed: int) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    ids = np.arange(num_episodes)
    rng.shuffle(ids)

    if num_episodes <= 1:
        return {"train": ids.tolist(), "val": []}

    num_val = max(1, int(round(num_episodes * val_ratio)))
    val_ids = sorted(ids[:num_val].tolist())
    train_ids = sorted(ids[num_val:].tolist())

    return {"train": train_ids, "val": val_ids}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", type=str, default="outputs/m2_expert_dataset_100")
    parser.add_argument("--out-dir", type=str, default="outputs/m3_phase_aware_dataset_100")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_episode_dir = out_dir / "episodes"

    if not src_dir.exists():
        raise FileNotFoundError(src_dir)

    out_episode_dir.mkdir(parents=True, exist_ok=True)

    src_episodes = sorted((src_dir / "episodes").glob("ep_*.npz"))
    if not src_episodes:
        raise RuntimeError(f"No episode npz files found under {src_dir / 'episodes'}")

    src_metadata_path = src_dir / "episodes.jsonl"
    src_metadata = load_jsonl(src_metadata_path) if src_metadata_path.exists() else []

    episode_records: list[dict[str, Any]] = []

    for episode_id, src_path in enumerate(src_episodes):
        data = np.load(src_path)

        state_obs = data["obs"].astype(np.float32)
        actions = data["actions"].astype(np.float32)
        rewards = data["rewards"].astype(np.float32)
        terminated = data["terminated"].astype(np.bool_)
        truncated = data["truncated"].astype(np.bool_)
        success = data["success"].astype(np.bool_)

        if state_obs.shape[0] != actions.shape[0]:
            raise ValueError(
                f"Length mismatch in {src_path}: obs={state_obs.shape}, actions={actions.shape}"
            )

        T = actions.shape[0]
        state_dim = state_obs.shape[1]
        action_dim = actions.shape[1]

        if T <= 1:
            progress = np.zeros((T, 1), dtype=np.float32)
        else:
            progress = (np.arange(T, dtype=np.float32) / float(T - 1)).reshape(T, 1)

        prev_action = np.zeros_like(actions, dtype=np.float32)
        if T > 1:
            prev_action[1:] = actions[:-1]

        phase_obs = np.concatenate(
            [state_obs, progress, prev_action],
            axis=1,
        ).astype(np.float32)

        out_path = out_episode_dir / f"ep_{episode_id:06d}.npz"

        np.savez_compressed(
            out_path,
            obs=phase_obs,
            actions=actions,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            success=success,
            state_obs=state_obs,
            progress=progress,
            prev_action=prev_action,
        )

        base_meta = src_metadata[episode_id] if episode_id < len(src_metadata) else {}

        record = {
            **base_meta,
            "episode_id": episode_id,
            "npz_path": str(out_path),
            "dataset_type": "phase_aware_expert_bc",
            "num_steps": int(T),
            "state_dim": int(state_dim),
            "action_dim": int(action_dim),
            "obs_dim": int(phase_obs.shape[1]),
            "phase_features": {
                "state_obs": state_dim,
                "progress": 1,
                "prev_action": action_dim,
            },
            "success_once": bool(success.any()),
            "final_success": bool(success[-1]) if success.size > 0 else False,
        }

        episode_records.append(record)

    splits = make_splits(len(episode_records), val_ratio=args.val_ratio, seed=args.seed)

    with (out_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for record in episode_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with (out_dir / "splits.json").open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    schema = {
        "milestone": "M3.4A",
        "format_version": 1,
        "description": "Phase-aware BC dataset with progress and previous-action conditioning.",
        "source_dataset": str(src_dir),
        "episode_npz_fields": {
            "obs": "[T, 66] float32; state_57 + progress_1 + prev_action_8",
            "actions": "[T, 8] float32; expert action",
            "rewards": "[T] float32",
            "terminated": "[T] bool",
            "truncated": "[T] bool",
            "success": "[T] bool",
            "state_obs": "[T, 57] float32; original state observation",
            "progress": "[T, 1] float32; normalized phase progress t/(T-1)",
            "prev_action": "[T, 8] float32; previous expert action, zero at t=0",
        },
        "obs_layout": {
            "state_obs": 57,
            "progress": 1,
            "prev_action": 8,
            "total": 66,
        },
    }

    with (out_dir / "dataset_schema.json").open("w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    summary = {
        "milestone": "M3.4A",
        "src_dir": str(src_dir),
        "out_dir": str(out_dir),
        "num_episodes": len(episode_records),
        "success_rate_once": float(np.mean([r["success_once"] for r in episode_records])),
        "mean_steps": float(np.mean([r["num_steps"] for r in episode_records])),
        "state_dim": 57,
        "action_dim": 8,
        "obs_dim": 66,
        "episodes_jsonl": str(out_dir / "episodes.jsonl"),
        "splits_json": str(out_dir / "splits.json"),
        "dataset_schema_json": str(out_dir / "dataset_schema.json"),
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[done] phase-aware dataset created")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()