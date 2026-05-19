from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="PickCube-v1")
    parser.add_argument("--obs-mode", type=str, default="state")
    parser.add_argument("--control-mode", type=str, default=None)
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="outputs/m0_random_rollout")
    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs: dict[str, Any] = {
        "obs_mode": args.obs_mode,
    }
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode

    env = gym.make(args.env_id, **env_kwargs)

    summaries: list[dict[str, Any]] = []

    for ep in range(args.num_episodes):
        obs, info = env.reset(seed=args.seed + ep)

        rewards: list[float] = []
        success = False
        steps = 0

        for t in range(args.max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            rewards.append(scalar_float(reward))

            info_np = to_numpy_safe(info)
            if isinstance(info_np, dict) and "success" in info_np:
                success = scalar_bool(info_np["success"])

            steps = t + 1

            if scalar_bool(terminated) or scalar_bool(truncated):
                break

        summary = {
            "episode": ep,
            "env_id": args.env_id,
            "obs_mode": args.obs_mode,
            "control_mode": args.control_mode,
            "num_steps": steps,
            "return": float(np.sum(rewards)),
            "success": success,
        }
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

    output = {
        "milestone": "M0",
        "description": "Random-action ManiSkill rollout smoke test.",
        "seed": args.seed,
        "env_id": args.env_id,
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "num_episodes": args.num_episodes,
        "episodes": summaries,
    }

    output_path = out_dir / "summary.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    env.close()
    print(f"[done] wrote {output_path}")


if __name__ == "__main__":
    main()