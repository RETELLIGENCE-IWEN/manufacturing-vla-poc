from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import yaml


@dataclass(frozen=True)
class TaskInstance:
    task_family: str
    base_env_id: str
    instruction: str
    object_id: str
    object_name: str
    target_id: str
    target_name: str
    template: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_numpy_safe(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, dict):
        return {k: to_numpy_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_numpy_safe(v) for v in x]
    return x


def flatten_numeric_tree(x: Any, prefix: str = "") -> tuple[np.ndarray, list[str]]:
    """
    Convert a nested observation/action object into a flat float32 vector.

    This is intentionally simple and robust for M2.
    Later, M3 can replace this with structured encoders.
    """
    x = to_numpy_safe(x)
    values: list[np.ndarray] = []
    names: list[str] = []

    def visit(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key in sorted(node.keys()):
                visit(node[key], f"{path}/{key}" if path else str(key))
            return

        if isinstance(node, (list, tuple)):
            for idx, item in enumerate(node):
                visit(item, f"{path}/{idx}" if path else str(idx))
            return

        arr = np.asarray(node)

        if arr.dtype == object:
            return

        if np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_):
            flat = arr.astype(np.float32).reshape(-1)
            if flat.size > 0:
                values.append(flat)
                names.extend([path] * flat.size)

    visit(x, prefix)

    if not values:
        return np.zeros((0,), dtype=np.float32), []

    return np.concatenate(values).astype(np.float32), names


def scalar_bool(x: Any) -> bool:
    arr = np.asarray(to_numpy_safe(x))
    return bool(arr.mean() > 0.5)


def scalar_float(x: Any) -> float:
    arr = np.asarray(to_numpy_safe(x), dtype=np.float64)
    return float(arr.mean())


def load_task_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sample_task_instance(cfg: dict[str, Any], rng: random.Random) -> TaskInstance:
    obj = rng.choice(cfg["objects"])
    tgt = rng.choice(cfg["targets"])
    template = rng.choice(cfg["instruction_templates"])

    instruction = template.format(
        object_name=obj["display_name"],
        target_name=tgt["display_name"],
    )

    return TaskInstance(
        task_family=cfg["task_family"],
        base_env_id=cfg["base_env_id"],
        instruction=instruction,
        object_id=obj["id"],
        object_name=obj["display_name"],
        target_id=tgt["id"],
        target_name=tgt["display_name"],
        template=template,
    )


def sample_action(env: gym.Env, action_source: str) -> Any:
    if action_source == "random":
        return env.action_space.sample()

    if action_source == "zero":
        sample = env.action_space.sample()
        if isinstance(sample, np.ndarray):
            return np.zeros_like(sample, dtype=np.float32)
        return sample

    raise ValueError(f"Unsupported action_source: {action_source}")


def run_episode(
    env: gym.Env,
    task: TaskInstance,
    episode_idx: int,
    max_steps: int,
    seed: int,
    action_source: str,
    episode_dir: Path,
) -> dict[str, Any]:
    obs, info = env.reset(seed=seed)

    obs_vectors: list[np.ndarray] = []
    action_vectors: list[np.ndarray] = []
    rewards: list[float] = []
    terminated_list: list[bool] = []
    truncated_list: list[bool] = []
    success_list: list[bool] = []

    obs_paths_ref: list[str] | None = None
    action_paths_ref: list[str] | None = None

    final_success = False
    steps = 0

    for t in range(max_steps):
        obs_vec, obs_paths = flatten_numeric_tree(obs, prefix="obs")
        if obs_paths_ref is None:
            obs_paths_ref = obs_paths

        action = sample_action(env, action_source=action_source)
        action_vec, action_paths = flatten_numeric_tree(action, prefix="action")
        if action_paths_ref is None:
            action_paths_ref = action_paths

        next_obs, reward, terminated, truncated, info = env.step(action)

        info_np = to_numpy_safe(info)
        success = False
        if isinstance(info_np, dict) and "success" in info_np:
            success = scalar_bool(info_np["success"])

        obs_vectors.append(obs_vec)
        action_vectors.append(action_vec)
        rewards.append(scalar_float(reward))
        terminated_list.append(scalar_bool(terminated))
        truncated_list.append(scalar_bool(truncated))
        success_list.append(success)

        final_success = final_success or success
        steps = t + 1
        obs = next_obs

        if scalar_bool(terminated) or scalar_bool(truncated):
            break

    if not obs_vectors:
        raise RuntimeError("No steps collected. Check environment/reset/action logic.")

    obs_array = np.stack(obs_vectors, axis=0).astype(np.float32)
    action_array = np.stack(action_vectors, axis=0).astype(np.float32)

    ep_stem = f"ep_{episode_idx:06d}"
    npz_path = episode_dir / f"{ep_stem}.npz"

    np.savez_compressed(
        npz_path,
        obs=obs_array,
        actions=action_array,
        rewards=np.asarray(rewards, dtype=np.float32),
        terminated=np.asarray(terminated_list, dtype=np.bool_),
        truncated=np.asarray(truncated_list, dtype=np.bool_),
        success=np.asarray(success_list, dtype=np.bool_),
    )

    return {
        "episode_id": episode_idx,
        "task": asdict(task),
        "seed": seed,
        "action_source": action_source,
        "num_steps": steps,
        "return": float(np.sum(rewards)),
        "success_once": bool(final_success),
        "final_success": bool(success_list[-1]) if success_list else False,
        "obs_dim": int(obs_array.shape[1]),
        "action_dim": int(action_array.shape[1]),
        "npz_path": str(npz_path),
        "obs_paths": obs_paths_ref or [],
        "action_paths": action_paths_ref or [],
    }


def make_splits(num_episodes: int, val_ratio: float, seed: int) -> dict[str, list[int]]:
    ids = list(range(num_episodes))
    rng = random.Random(seed)
    rng.shuffle(ids)

    num_val = max(1, int(round(num_episodes * val_ratio))) if num_episodes > 1 else 0
    val_ids = sorted(ids[:num_val])
    train_ids = sorted(ids[num_val:])

    return {
        "train": train_ids,
        "val": val_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/manufacturing_pick_place_v0.yaml")
    parser.add_argument("--obs-mode", type=str, default="state")
    parser.add_argument("--control-mode", type=str, default=None)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-source", type=str, default="random", choices=["random", "zero"])
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--out-dir", type=str, default="outputs/m2_step_dataset")
    args = parser.parse_args()

    set_seed(args.seed)

    cfg_path = Path(args.config)
    cfg = load_task_config(cfg_path)

    out_dir = Path(args.out_dir)
    episode_dir = out_dir / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    env_kwargs: dict[str, Any] = {"obs_mode": args.obs_mode}
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode

    env = gym.make(cfg["base_env_id"], **env_kwargs)

    episodes_jsonl = out_dir / "episodes.jsonl"
    episode_records: list[dict[str, Any]] = []

    with episodes_jsonl.open("w", encoding="utf-8") as f:
        for ep in range(args.num_episodes):
            task = sample_task_instance(cfg, rng)
            record = run_episode(
                env=env,
                task=task,
                episode_idx=ep,
                max_steps=args.max_steps,
                seed=args.seed + ep,
                action_source=args.action_source,
                episode_dir=episode_dir,
            )
            episode_records.append(record)

            # Keep JSONL metadata compact.
            compact_record = dict(record)
            compact_record.pop("obs_paths", None)
            compact_record.pop("action_paths", None)

            f.write(json.dumps(compact_record, ensure_ascii=False) + "\n")
            print(json.dumps(compact_record, indent=2, ensure_ascii=False))

    env.close()

    success_rate = float(np.mean([r["success_once"] for r in episode_records])) if episode_records else 0.0
    mean_return = float(np.mean([r["return"] for r in episode_records])) if episode_records else 0.0
    mean_steps = float(np.mean([r["num_steps"] for r in episode_records])) if episode_records else 0.0

    splits = make_splits(args.num_episodes, val_ratio=args.val_ratio, seed=args.seed)
    with (out_dir / "splits.json").open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    schema = {
        "milestone": "M2",
        "format_version": 1,
        "description": "Step-level language-conditioned manipulation dataset.",
        "episode_npz_fields": {
            "obs": "[T, obs_dim] float32 flattened observation vector before action",
            "actions": "[T, action_dim] float32 flattened action vector",
            "rewards": "[T] float32 reward after action",
            "terminated": "[T] bool Gymnasium terminated signal",
            "truncated": "[T] bool Gymnasium truncated signal",
            "success": "[T] bool task success signal from info, if available",
        },
        "metadata_files": {
            "episodes_jsonl": "Episode-level metadata including instruction and task aliases.",
            "splits_json": "Train/validation split by episode id.",
            "summary_json": "Dataset summary statistics.",
        },
        "obs_paths": episode_records[0]["obs_paths"] if episode_records else [],
        "action_paths": episode_records[0]["action_paths"] if episode_records else [],
    }

    with (out_dir / "dataset_schema.json").open("w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    summary = {
        "milestone": "M2",
        "config": str(cfg_path),
        "base_env_id": cfg["base_env_id"],
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "action_source": args.action_source,
        "seed": args.seed,
        "num_episodes": args.num_episodes,
        "max_steps": args.max_steps,
        "success_rate_once": success_rate,
        "mean_return": mean_return,
        "mean_steps": mean_steps,
        "out_dir": str(out_dir),
        "episodes_jsonl": str(episodes_jsonl),
        "splits_json": str(out_dir / "splits.json"),
        "dataset_schema_json": str(out_dir / "dataset_schema.json"),
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[done] M2 dataset collection complete")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()