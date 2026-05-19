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
from mani_skill.utils.wrappers.record import RecordEpisode


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

    raise RuntimeError("Could not find get_state_dict() on env.")


def squeeze_first_batch(x: Any) -> np.ndarray:
    arr = to_numpy(x).astype(np.float32)

    if arr.ndim >= 2 and arr.shape[0] == 1:
        arr = arr[0]

    return arr.reshape(-1).astype(np.float32)


def extract_state_features(env: gym.Env) -> dict[str, np.ndarray]:
    state = get_state_dict(env)

    panda = squeeze_first_batch(state["articulations"]["panda"])
    cube = squeeze_first_batch(state["actors"]["cube"])
    goal_site = squeeze_first_batch(state["actors"]["goal_site"])

    obs = np.concatenate([panda, cube, goal_site], axis=0).astype(np.float32)

    if obs.shape[0] != 57:
        raise ValueError(f"Unexpected obs dim: {obs.shape[0]}, expected 57")

    cube_pos = cube[:3]
    goal_pos = goal_site[:3]
    cube_goal_dist = np.linalg.norm(cube_pos - goal_pos)

    return {
        "obs": obs,
        "panda": panda,
        "cube": cube,
        "goal_site": goal_site,
        "cube_pos": cube_pos,
        "goal_pos": goal_pos,
        "cube_goal_dist": np.asarray([cube_goal_dist], dtype=np.float32),
    }


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, choices=["bc", "random"], default="bc")
    parser.add_argument("--env-id", type=str, default="PickCube-v1")
    parser.add_argument("--model", type=str, default="runs/m3_bc_state/best_model.pt")
    parser.add_argument("--normalization", type=str, default="runs/m3_bc_state/normalization_stats.npz")
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--out-dir", type=str, default="runs/m3_bc_state/closedloop_debug")
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir) / args.policy / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = None
    stats = None

    if args.policy == "bc":
        model, stats = load_policy(
            model_path=Path(args.model),
            norm_path=Path(args.normalization),
            device=device,
        )

    env = gym.make(
        args.env_id,
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array" if args.save_video else None,
        sim_backend="auto",
    )

    if args.save_video:
        env = RecordEpisode(
            env,
            output_dir=str(out_dir),
            trajectory_name=f"{args.policy}_seed_{args.seed}",
            save_video=True,
            source_type="policy_rollout",
            source_desc=f"M3.3 closed-loop debug rollout for {args.policy}",
            video_fps=30,
            record_reward=True,
            save_on_reset=False,
        )

    env.reset(seed=args.seed)

    step_log_path = out_dir / "steps.jsonl"
    summary_path = out_dir / "summary.json"

    rewards: list[float] = []
    success_flags: list[bool] = []
    cube_goal_dists: list[float] = []
    action_norms: list[float] = []

    with step_log_path.open("w", encoding="utf-8") as f_log:
        for t in range(args.max_steps):
            state_features = extract_state_features(env)

            if args.policy == "random":
                action = env.action_space.sample()
            else:
                assert model is not None
                assert stats is not None
                action = predict_bc_action(
                    model=model,
                    stats=stats,
                    obs_raw=state_features["obs"],
                    device=device,
                )
                action = clip_action(env, action)

            _, reward, terminated, truncated, info = env.step(action)

            success = False
            if isinstance(info, dict) and "success" in info:
                success = scalar_bool(info["success"])

            reward_value = scalar_float(reward)
            action_norm = float(np.linalg.norm(np.asarray(action).reshape(-1)))
            cube_goal_dist = float(state_features["cube_goal_dist"][0])

            rewards.append(reward_value)
            success_flags.append(success)
            cube_goal_dists.append(cube_goal_dist)
            action_norms.append(action_norm)

            row = {
                "step": t,
                "policy": args.policy,
                "seed": args.seed,
                "reward": reward_value,
                "success": success,
                "terminated": scalar_bool(terminated),
                "truncated": scalar_bool(truncated),
                "cube_goal_dist": cube_goal_dist,
                "action_norm": action_norm,
                "cube_pos": state_features["cube_pos"].tolist(),
                "goal_pos": state_features["goal_pos"].tolist(),
                "action": np.asarray(action).reshape(-1).astype(float).tolist(),
            }

            f_log.write(json.dumps(row, ensure_ascii=False) + "\n")

            if scalar_bool(terminated) or scalar_bool(truncated):
                break

    if args.save_video:
        env.flush_trajectory()
        env.flush_video()

    env.close()

    summary = {
        "milestone": "M3.3A",
        "description": "Closed-loop rollout debug video and step log.",
        "policy": args.policy,
        "env_id": args.env_id,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "num_steps": len(rewards),
        "return": float(np.sum(rewards)),
        "success_once": bool(any(success_flags)),
        "final_success": bool(success_flags[-1]) if success_flags else False,
        "initial_cube_goal_dist": float(cube_goal_dists[0]) if cube_goal_dists else None,
        "final_cube_goal_dist": float(cube_goal_dists[-1]) if cube_goal_dists else None,
        "min_cube_goal_dist": float(np.min(cube_goal_dists)) if cube_goal_dists else None,
        "mean_action_norm": float(np.mean(action_norms)) if action_norms else None,
        "max_action_norm": float(np.max(action_norms)) if action_norms else None,
        "step_log": str(step_log_path),
        "out_dir": str(out_dir),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[done] M3.3A closed-loop debug rollout complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()