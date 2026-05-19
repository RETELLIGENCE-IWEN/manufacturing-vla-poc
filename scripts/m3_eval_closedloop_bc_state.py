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
import torch.nn as nn


class BCPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        dropout: float,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = obs_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def scalar_bool(x: Any) -> bool:
    return bool(np.asarray(to_numpy(x)).mean() > 0.5)


def scalar_float(x: Any) -> float:
    return float(np.asarray(to_numpy(x), dtype=np.float64).mean())


def get_state_dict(env: gym.Env) -> dict[str, Any]:
    """
    ManiSkill wrappers/version may expose get_state_dict at different levels.
    Try common access paths.
    """
    candidates = [
        env,
        getattr(env, "unwrapped", None),
        getattr(env, "base_env", None),
    ]

    for candidate in candidates:
        if candidate is None:
            continue
        if hasattr(candidate, "get_state_dict"):
            return candidate.get_state_dict()

    raise RuntimeError(
        "Could not find get_state_dict() on env/env.unwrapped/env.base_env. "
        "Check ManiSkill API version."
    )


def squeeze_first_batch(x: Any) -> np.ndarray:
    arr = to_numpy(x).astype(np.float32)

    # ManiSkill often returns batched tensors with shape [1, D].
    if arr.ndim >= 2 and arr.shape[0] == 1:
        arr = arr[0]

    return arr.reshape(-1).astype(np.float32)


def build_bc_obs_from_env(env: gym.Env) -> np.ndarray:
    """
    Build the exact M3 training observation layout:

    panda articulation state: 31 dims
    cube actor state: 13 dims
    goal_site actor state: 13 dims
    total: 57 dims
    """
    state = get_state_dict(env)

    try:
        panda = squeeze_first_batch(state["articulations"]["panda"])
        cube = squeeze_first_batch(state["actors"]["cube"])
        goal_site = squeeze_first_batch(state["actors"]["goal_site"])
    except KeyError as exc:
        available = {
            "top_level": list(state.keys()),
            "actors": list(state.get("actors", {}).keys()) if isinstance(state.get("actors", {}), dict) else None,
            "articulations": list(state.get("articulations", {}).keys())
            if isinstance(state.get("articulations", {}), dict)
            else None,
        }
        raise KeyError(f"Missing expected state key: {exc}. Available keys: {available}") from exc

    obs = np.concatenate([panda, cube, goal_site], axis=0).astype(np.float32)

    if obs.shape[0] != 57:
        raise ValueError(
            f"Unexpected BC obs dim: {obs.shape[0]}. "
            f"Expected 57 = panda(31) + cube(13) + goal_site(13)."
        )

    return obs


def load_policy(
    model_path: Path,
    norm_path: Path,
    device: torch.device,
) -> tuple[BCPolicy, dict[str, np.ndarray]]:
    checkpoint = torch.load(model_path, map_location=device)

    model = BCPolicy(
        obs_dim=int(checkpoint["obs_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        hidden_dims=[int(x) for x in checkpoint["hidden_dims"]],
        dropout=float(checkpoint["dropout"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    norm = np.load(norm_path)
    stats = {
        "obs_mean": norm["obs_mean"].astype(np.float32),
        "obs_std": norm["obs_std"].astype(np.float32),
        "action_mean": norm["action_mean"].astype(np.float32),
        "action_std": norm["action_std"].astype(np.float32),
    }

    return model, stats


@torch.no_grad()
def predict_bc_action(
    model: BCPolicy,
    stats: dict[str, np.ndarray],
    obs_raw: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    obs_norm = ((obs_raw - stats["obs_mean"]) / stats["obs_std"]).astype(np.float32)

    obs_t = torch.from_numpy(obs_norm[None, :]).to(device)
    pred_norm = model(obs_t).cpu().numpy()[0].astype(np.float32)

    action = pred_norm * stats["action_std"] + stats["action_mean"]
    return action.astype(np.float32)


def clip_action(env: gym.Env, action: np.ndarray) -> np.ndarray:
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    action = action.reshape(-1).astype(np.float32)

    if low.shape == action.shape and high.shape == action.shape:
        return np.clip(action, low, high).astype(np.float32)

    return action.astype(np.float32)


def run_episode(
    env: gym.Env,
    policy_name: str,
    seed: int,
    max_steps: int,
    model: BCPolicy | None,
    stats: dict[str, np.ndarray] | None,
    device: torch.device,
) -> dict[str, Any]:
    env.reset(seed=seed)

    rewards: list[float] = []
    success_flags: list[bool] = []
    steps = 0

    for t in range(max_steps):
        if policy_name == "random":
            action = env.action_space.sample()
        elif policy_name == "bc":
            if model is None or stats is None:
                raise RuntimeError("BC policy requested but model/stats are missing.")
            obs_raw = build_bc_obs_from_env(env)
            action = predict_bc_action(
                model=model,
                stats=stats,
                obs_raw=obs_raw,
                device=device,
            )
            action = clip_action(env, action)
        else:
            raise ValueError(f"Unknown policy_name: {policy_name}")

        _, reward, terminated, truncated, info = env.step(action)

        reward_value = scalar_float(reward)
        rewards.append(reward_value)

        success = False
        if isinstance(info, dict) and "success" in info:
            success = scalar_bool(info["success"])

        success_flags.append(success)
        steps = t + 1

        if scalar_bool(terminated) or scalar_bool(truncated):
            break

    return {
        "policy": policy_name,
        "seed": seed,
        "num_steps": steps,
        "return": float(np.sum(rewards)),
        "success_once": bool(any(success_flags)),
        "final_success": bool(success_flags[-1]) if success_flags else False,
    }


def evaluate_policy(
    policy_name: str,
    num_episodes: int,
    seed_start: int,
    max_steps: int,
    model_path: Path,
    norm_path: Path,
    device: torch.device,
    env_id: str,
    sim_backend: str,
) -> dict[str, Any]:
    env = gym.make(
        env_id,
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode=None,
        sim_backend=sim_backend,
    )

    model = None
    stats = None

    if policy_name == "bc":
        model, stats = load_policy(
            model_path=model_path,
            norm_path=norm_path,
            device=device,
        )

    episodes: list[dict[str, Any]] = []

    for ep in range(num_episodes):
        seed = seed_start + ep
        result = run_episode(
            env=env,
            policy_name=policy_name,
            seed=seed,
            max_steps=max_steps,
            model=model,
            stats=stats,
            device=device,
        )
        result["episode"] = ep
        episodes.append(result)

        print(
            f"[{policy_name} ep={ep:03d}] "
            f"seed={seed} "
            f"success_once={result['success_once']} "
            f"final_success={result['final_success']} "
            f"return={result['return']:.3f} "
            f"steps={result['num_steps']}"
        )

    env.close()

    return {
        "policy": policy_name,
        "num_episodes": num_episodes,
        "seed_start": seed_start,
        "max_steps": max_steps,
        "success_rate_once": float(np.mean([x["success_once"] for x in episodes])),
        "final_success_rate": float(np.mean([x["final_success"] for x in episodes])),
        "mean_return": float(np.mean([x["return"] for x in episodes])),
        "mean_steps": float(np.mean([x["num_steps"] for x in episodes])),
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="PickCube-v1")
    parser.add_argument("--model", type=str, default="runs/m3_bc_state/best_model.pt")
    parser.add_argument("--normalization", type=str, default="runs/m3_bc_state/normalization_stats.npz")
    parser.add_argument("--num-episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--sim-backend", type=str, default="auto")
    parser.add_argument("--out-dir", type=str, default="runs/m3_bc_state/closedloop_eval")
    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model)
    norm_path = Path(args.normalization)

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not norm_path.exists():
        raise FileNotFoundError(norm_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random_summary = evaluate_policy(
        policy_name="random",
        num_episodes=args.num_episodes,
        seed_start=args.seed,
        max_steps=args.max_steps,
        model_path=model_path,
        norm_path=norm_path,
        device=device,
        env_id=args.env_id,
        sim_backend=args.sim_backend,
    )

    bc_summary = evaluate_policy(
        policy_name="bc",
        num_episodes=args.num_episodes,
        seed_start=args.seed,
        max_steps=args.max_steps,
        model_path=model_path,
        norm_path=norm_path,
        device=device,
        env_id=args.env_id,
        sim_backend=args.sim_backend,
    )

    summary = {
        "milestone": "M3.2",
        "description": "Closed-loop rollout evaluation for state-only BC policy.",
        "env_id": args.env_id,
        "control_mode": "pd_joint_pos",
        "obs_mode": "none",
        "model": str(model_path),
        "normalization": str(norm_path),
        "device": str(device),
        "num_episodes": args.num_episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "random": random_summary,
        "bc": bc_summary,
        "comparison": {
            "success_rate_once_delta": bc_summary["success_rate_once"] - random_summary["success_rate_once"],
            "final_success_rate_delta": bc_summary["final_success_rate"] - random_summary["final_success_rate"],
            "mean_return_delta": bc_summary["mean_return"] - random_summary["mean_return"],
        },
    }

    with (out_dir / "closedloop_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[done] M3.2 closed-loop evaluation complete")
    print(json.dumps(summary["comparison"], indent=2))


if __name__ == "__main__":
    main()