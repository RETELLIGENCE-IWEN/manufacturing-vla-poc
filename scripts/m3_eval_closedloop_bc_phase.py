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
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int], dropout: float) -> None:
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
    candidates = [env, getattr(env, "unwrapped", None), getattr(env, "base_env", None)]

    for candidate in candidates:
        if candidate is None:
            continue
        if hasattr(candidate, "get_state_dict"):
            return candidate.get_state_dict()

    raise RuntimeError("Could not find get_state_dict() on env.")


def set_env_time_limit(env: gym.Env, max_steps: int) -> None:
    """
    Override ManiSkill/Gymnasium TimeLimitWrapper horizon.

    ManiSkill PickCube-v1 may wrap the environment with an internal
    TimeLimitWrapper whose _max_episode_steps defaults to 50. This can
    prematurely truncate closed-loop policy evaluation even when the
    script-level --max-steps argument is larger.

    This helper aligns the wrapper time limit with the requested rollout
    horizon.
    """
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")

    candidates = [
        env,
        getattr(env, "unwrapped", None),
        getattr(env, "base_env", None),
    ]

    for candidate in candidates:
        if candidate is None:
            continue

        if hasattr(candidate, "_max_episode_steps"):
            try:
                candidate._max_episode_steps = int(max_steps)
            except Exception:
                pass

        if hasattr(candidate, "max_episode_steps"):
            try:
                candidate.max_episode_steps = int(max_steps)
            except Exception:
                pass


def squeeze_first_batch(x: Any) -> np.ndarray:
    arr = to_numpy(x).astype(np.float32)

    if arr.ndim >= 2 and arr.shape[0] == 1:
        arr = arr[0]

    return arr.reshape(-1).astype(np.float32)


def build_state_obs_from_env(env: gym.Env) -> np.ndarray:
    state = get_state_dict(env)

    panda = squeeze_first_batch(state["articulations"]["panda"])
    cube = squeeze_first_batch(state["actors"]["cube"])
    goal_site = squeeze_first_batch(state["actors"]["goal_site"])

    obs = np.concatenate([panda, cube, goal_site], axis=0).astype(np.float32)

    if obs.shape[0] != 57:
        raise ValueError(f"Unexpected state obs dim: {obs.shape[0]}, expected 57")

    return obs


def build_phase_obs(
    env: gym.Env,
    step_idx: int,
    phase_horizon: int,
    prev_action: np.ndarray,
) -> np.ndarray:
    state_obs = build_state_obs_from_env(env)

    denom = max(1, phase_horizon - 1)
    progress_value = min(float(step_idx) / float(denom), 1.0)
    progress = np.asarray([progress_value], dtype=np.float32)

    obs = np.concatenate(
        [state_obs, progress, prev_action.astype(np.float32).reshape(-1)],
        axis=0,
    ).astype(np.float32)

    if obs.shape[0] != 66:
        raise ValueError(f"Unexpected phase-aware obs dim: {obs.shape[0]}, expected 66")

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
def predict_action(
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
    phase_horizon: int,
    model: BCPolicy | None,
    stats: dict[str, np.ndarray] | None,
    device: torch.device,
) -> dict[str, Any]:
    env.reset(seed=seed)

    rewards: list[float] = []
    success_flags: list[bool] = []
    action_norms: list[float] = []

    prev_action = np.zeros((8,), dtype=np.float32)

    for t in range(max_steps):
        if policy_name == "random":
            action = env.action_space.sample()
        elif policy_name == "phase_bc":
            if model is None or stats is None:
                raise RuntimeError("phase_bc requested but model/stats are missing.")

            obs = build_phase_obs(
                env=env,
                step_idx=t,
                phase_horizon=phase_horizon,
                prev_action=prev_action,
            )

            action = predict_action(
                model=model,
                stats=stats,
                obs_raw=obs,
                device=device,
            )
            action = clip_action(env, action)
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        _, reward, terminated, truncated, info = env.step(action)

        prev_action = np.asarray(action, dtype=np.float32).reshape(-1)

        rewards.append(scalar_float(reward))
        action_norms.append(float(np.linalg.norm(prev_action)))

        success = False
        if isinstance(info, dict) and "success" in info:
            success = scalar_bool(info["success"])
        success_flags.append(success)

        if scalar_bool(terminated) or scalar_bool(truncated):
            break

    return {
        "policy": policy_name,
        "seed": seed,
        "num_steps": len(rewards),
        "return": float(np.sum(rewards)),
        "success_once": bool(any(success_flags)),
        "final_success": bool(success_flags[-1]) if success_flags else False,
        "mean_action_norm": float(np.mean(action_norms)) if action_norms else 0.0,
        "max_action_norm": float(np.max(action_norms)) if action_norms else 0.0,
    }


def evaluate_policy(
    policy_name: str,
    num_episodes: int,
    seed_start: int,
    max_steps: int,
    phase_horizon: int,
    env_id: str,
    sim_backend: str,
    model: BCPolicy | None,
    stats: dict[str, np.ndarray] | None,
    device: torch.device,
) -> dict[str, Any]:
    env = gym.make(
        env_id,
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode=None,
        sim_backend=sim_backend,
    )

    set_env_time_limit(env, max_steps=max_steps)

    episodes: list[dict[str, Any]] = []

    for ep in range(num_episodes):
        seed = seed_start + ep

        result = run_episode(
            env=env,
            policy_name=policy_name,
            seed=seed,
            max_steps=max_steps,
            phase_horizon=phase_horizon,
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

    env_time_limit = int(getattr(env, "_max_episode_steps", max_steps))
    env.close()

    return {
        "policy": policy_name,
        "num_episodes": num_episodes,
        "seed_start": seed_start,
        "max_steps": max_steps,
        "phase_horizon": phase_horizon,
        "env_time_limit": env_time_limit,
        "success_rate_once": float(np.mean([x["success_once"] for x in episodes])),
        "final_success_rate": float(np.mean([x["final_success"] for x in episodes])),
        "mean_return": float(np.mean([x["return"] for x in episodes])),
        "mean_steps": float(np.mean([x["num_steps"] for x in episodes])),
        "mean_action_norm": float(np.mean([x["mean_action_norm"] for x in episodes])),
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="PickCube-v1")
    parser.add_argument("--model", type=str, default="runs/m3_bc_phase_aware/best_model.pt")
    parser.add_argument("--normalization", type=str, default="runs/m3_bc_phase_aware/normalization_stats.npz")
    parser.add_argument("--num-episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--phase-horizon", type=int, default=80)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--sim-backend", type=str, default="auto")
    parser.add_argument("--out-dir", type=str, default="runs/m3_bc_phase_aware/closedloop_eval")
    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, stats = load_policy(
        model_path=Path(args.model),
        norm_path=Path(args.normalization),
        device=device,
    )

    random_summary = evaluate_policy(
        policy_name="random",
        num_episodes=args.num_episodes,
        seed_start=args.seed,
        max_steps=args.max_steps,
        phase_horizon=args.phase_horizon,
        env_id=args.env_id,
        sim_backend=args.sim_backend,
        model=None,
        stats=None,
        device=device,
    )

    phase_bc_summary = evaluate_policy(
        policy_name="phase_bc",
        num_episodes=args.num_episodes,
        seed_start=args.seed,
        max_steps=args.max_steps,
        phase_horizon=args.phase_horizon,
        env_id=args.env_id,
        sim_backend=args.sim_backend,
        model=model,
        stats=stats,
        device=device,
    )

    summary = {
        "milestone": "M3.4D",
        "description": "Closed-loop rollout evaluation for phase-aware BC policy.",
        "env_id": args.env_id,
        "control_mode": "pd_joint_pos",
        "obs_mode": "none",
        "model": str(args.model),
        "normalization": str(args.normalization),
        "device": str(device),
        "num_episodes": args.num_episodes,
        "max_steps": args.max_steps,
        "phase_horizon": args.phase_horizon,
        "seed": args.seed,
        "random": random_summary,
        "phase_bc": phase_bc_summary,
        "comparison": {
            "success_rate_once_delta": phase_bc_summary["success_rate_once"] - random_summary["success_rate_once"],
            "final_success_rate_delta": phase_bc_summary["final_success_rate"] - random_summary["final_success_rate"],
            "mean_return_delta": phase_bc_summary["mean_return"] - random_summary["mean_return"],
        },
    }

    with (out_dir / "closedloop_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[done] M3.4D phase-aware closed-loop evaluation complete")
    print(json.dumps(summary["comparison"], indent=2))


if __name__ == "__main__":
    main()
