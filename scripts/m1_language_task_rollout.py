from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
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
    if isinstance(x, dict):
        return {k: to_numpy_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_numpy_safe(v) for v in x]
    return x


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


def run_episode(
    env: gym.Env,
    task: TaskInstance,
    episode_idx: int,
    max_steps: int,
    seed: int,
) -> dict[str, Any]:
    obs, info = env.reset(seed=seed)

    rewards: list[float] = []
    success = False
    steps = 0

    for t in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(scalar_float(reward))

        info_np = to_numpy_safe(info)
        if isinstance(info_np, dict) and "success" in info_np:
            success = scalar_bool(info_np["success"])

        steps = t + 1

        if scalar_bool(terminated) or scalar_bool(truncated):
            break

    return {
        "episode": episode_idx,
        "task_family": task.task_family,
        "base_env_id": task.base_env_id,
        "instruction": task.instruction,
        "object_id": task.object_id,
        "object_name": task.object_name,
        "target_id": task.target_id,
        "target_name": task.target_name,
        "template": task.template,
        "num_steps": steps,
        "return": float(np.sum(rewards)),
        "success": success,
        "policy": "random_action_baseline",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/manufacturing_pick_place_v0.yaml",
    )
    parser.add_argument("--obs-mode", type=str, default="state")
    parser.add_argument("--control-mode", type=str, default=None)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="outputs/m1_language_task_rollout")
    args = parser.parse_args()

    set_seed(args.seed)

    cfg_path = Path(args.config)
    cfg = load_task_config(cfg_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    env_kwargs: dict[str, Any] = {
        "obs_mode": args.obs_mode,
    }
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode

    env = gym.make(cfg["base_env_id"], **env_kwargs)

    episodes: list[dict[str, Any]] = []

    jsonl_path = out_dir / "episodes.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as f_jsonl:
        for ep in range(args.num_episodes):
            task = sample_task_instance(cfg, rng)
            result = run_episode(
                env=env,
                task=task,
                episode_idx=ep,
                max_steps=args.max_steps,
                seed=args.seed + ep,
            )

            episodes.append(result)
            f_jsonl.write(json.dumps(result, ensure_ascii=False) + "\n")

            print(json.dumps(result, indent=2, ensure_ascii=False))

    success_rate = float(np.mean([ep["success"] for ep in episodes])) if episodes else 0.0
    mean_return = float(np.mean([ep["return"] for ep in episodes])) if episodes else 0.0
    mean_steps = float(np.mean([ep["num_steps"] for ep in episodes])) if episodes else 0.0

    summary = {
        "milestone": "M1",
        "description": "Language-conditioned manufacturing-style task wrapper over ManiSkill PickCube-v1.",
        "config": str(cfg_path),
        "seed": args.seed,
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "num_episodes": args.num_episodes,
        "policy": "random_action_baseline",
        "success_rate": success_rate,
        "mean_return": mean_return,
        "mean_steps": mean_steps,
        "episodes_jsonl": str(jsonl_path),
    }

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    env.close()
    print(f"[done] wrote {summary_path}")
    print(f"[done] wrote {jsonl_path}")


if __name__ == "__main__":
    main()