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


def set_env_time_limit(env: gym.Env, max_steps: int) -> None:
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


def get_env_time_limit(env: gym.Env, fallback: int) -> int:
    if hasattr(env, "_max_episode_steps"):
        value = getattr(env, "_max_episode_steps")
        if value is not None:
            return int(value)
    return int(fallback)


def get_state_dict(env: gym.Env) -> dict[str, Any]:
    candidates = [
        getattr(env, "unwrapped", None),
        getattr(env, "base_env", None),
        env,
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

    state_obs = np.concatenate([panda, cube, goal_site], axis=0).astype(np.float32)

    if state_obs.shape[0] != 57:
        raise ValueError(f"Unexpected state obs dim: {state_obs.shape[0]}, expected 57")

    cube_pos = cube[:3].astype(np.float32)
    goal_pos = goal_site[:3].astype(np.float32)
    cube_goal_dist = np.asarray([np.linalg.norm(cube_pos - goal_pos)], dtype=np.float32)

    return {
        "state_obs": state_obs,
        "panda": panda,
        "cube": cube,
        "goal_site": goal_site,
        "cube_pos": cube_pos,
        "goal_pos": goal_pos,
        "cube_goal_dist": cube_goal_dist,
    }


def build_phase_obs(
    env: gym.Env,
    step_idx: int,
    phase_horizon: int,
    prev_action: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    features = extract_state_features(env)

    denom = max(1, phase_horizon - 1)
    progress_value = min(float(step_idx) / float(denom), 1.0)
    progress = np.asarray([progress_value], dtype=np.float32)

    prev_action = prev_action.astype(np.float32).reshape(-1)

    obs = np.concatenate(
        [features["state_obs"], progress, prev_action],
        axis=0,
    ).astype(np.float32)

    if obs.shape[0] != 66:
        raise ValueError(f"Unexpected phase-aware obs dim: {obs.shape[0]}, expected 66")

    return obs, features


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


def load_expert_action_bounds(path: Path, margin: float) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        bounds = json.load(f)

    low = np.asarray(bounds["min"], dtype=np.float32)
    high = np.asarray(bounds["max"], dtype=np.float32)

    span = high - low
    low = low - margin * span
    high = high + margin * span

    return low.astype(np.float32), high.astype(np.float32)


def apply_safe_action_filter(
    action: np.ndarray,
    expert_low: np.ndarray,
    expert_high: np.ndarray,
    hard_threshold_gripper: bool,
) -> np.ndarray:
    action = action.reshape(-1).astype(np.float32)
    action = np.clip(action, expert_low, expert_high).astype(np.float32)

    if hard_threshold_gripper:
        action[-1] = 1.0 if action[-1] >= 0.0 else -1.0

    return action.astype(np.float32)


def clip_action_to_env(env: gym.Env, action: np.ndarray) -> np.ndarray:
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    action = action.reshape(-1).astype(np.float32)

    if low.shape == action.shape and high.shape == action.shape:
        return np.clip(action, low, high).astype(np.float32)

    return action.astype(np.float32)


def read_info_flag(info: Any, key: str) -> bool:
    if isinstance(info, dict) and key in info:
        return scalar_bool(info[key])
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="PickCube-v1")
    parser.add_argument("--model", type=str, default="runs/m3_bc_phase_aware/best_model.pt")
    parser.add_argument("--normalization", type=str, default="runs/m3_bc_phase_aware/normalization_stats.npz")
    parser.add_argument("--expert-action-bounds", type=str, default="outputs/m3_phase_aware_dataset_100/action_bounds.json")
    parser.add_argument("--expert-bound-margin", type=float, default=0.05)
    parser.add_argument("--hard-threshold-gripper", action="store_true")
    parser.add_argument("--seed", type=int, default=3021)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--phase-horizon", type=int, default=80)
    parser.add_argument("--out-dir", type=str, default="runs/m3_bc_phase_aware/closedloop_debug_safe")
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir) / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, stats = load_policy(
        model_path=Path(args.model),
        norm_path=Path(args.normalization),
        device=device,
    )

    expert_low, expert_high = load_expert_action_bounds(
        Path(args.expert_action_bounds),
        margin=args.expert_bound_margin,
    )

    env = gym.make(
        args.env_id,
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array" if args.save_video else None,
        sim_backend="auto",
    )

    set_env_time_limit(env, max_steps=args.max_steps)
    env_time_limit = get_env_time_limit(env, fallback=args.max_steps)

    if args.save_video:
        env = RecordEpisode(
            env,
            output_dir=str(out_dir),
            trajectory_name=f"phase_safe_seed_{args.seed}",
            save_video=True,
            source_type="policy_rollout",
            source_desc="M3.5C phase-aware safe BC debug rollout",
            video_fps=30,
            record_reward=True,
            save_on_reset=False,
        )

        set_env_time_limit(env, max_steps=args.max_steps)

    env.reset(seed=args.seed)

    step_log_path = out_dir / "steps.jsonl"
    summary_path = out_dir / "summary.json"

    rewards: list[float] = []
    success_flags: list[bool] = []
    grasped_flags: list[bool] = []
    placed_flags: list[bool] = []
    static_flags: list[bool] = []
    cube_goal_dists: list[float] = []
    action_norms: list[float] = []

    prev_action = np.zeros((8,), dtype=np.float32)

    with step_log_path.open("w", encoding="utf-8") as f_log:
        for t in range(args.max_steps):
            obs, features = build_phase_obs(
                env=env,
                step_idx=t,
                phase_horizon=args.phase_horizon,
                prev_action=prev_action,
            )

            action = predict_action(
                model=model,
                stats=stats,
                obs_raw=obs,
                device=device,
            )

            action_raw = action.copy()

            action = apply_safe_action_filter(
                action=action,
                expert_low=expert_low,
                expert_high=expert_high,
                hard_threshold_gripper=args.hard_threshold_gripper,
            )

            action_safe = action.copy()
            action = clip_action_to_env(env, action)

            _, reward, terminated, truncated, info = env.step(action)

            prev_action = np.asarray(action, dtype=np.float32).reshape(-1)

            reward_value = scalar_float(reward)
            success = read_info_flag(info, "success")
            is_grasped = read_info_flag(info, "is_grasped")
            is_obj_placed = read_info_flag(info, "is_obj_placed")
            is_robot_static = read_info_flag(info, "is_robot_static")
            term_value = scalar_bool(terminated)
            trunc_value = scalar_bool(truncated)

            cube_goal_dist = float(features["cube_goal_dist"][0])
            action_norm = float(np.linalg.norm(prev_action))

            rewards.append(reward_value)
            success_flags.append(success)
            grasped_flags.append(is_grasped)
            placed_flags.append(is_obj_placed)
            static_flags.append(is_robot_static)
            cube_goal_dists.append(cube_goal_dist)
            action_norms.append(action_norm)

            row = {
                "step": t,
                "seed": args.seed,
                "reward": reward_value,
                "success": success,
                "is_grasped": is_grasped,
                "is_obj_placed": is_obj_placed,
                "is_robot_static": is_robot_static,
                "terminated": term_value,
                "truncated": trunc_value,
                "cube_goal_dist": cube_goal_dist,
                "cube_pos": features["cube_pos"].tolist(),
                "goal_pos": features["goal_pos"].tolist(),
                "progress": min(float(t) / float(max(1, args.phase_horizon - 1)), 1.0),
                "action_norm": action_norm,
                "action_raw": action_raw.reshape(-1).astype(float).tolist(),
                "action_safe": action_safe.reshape(-1).astype(float).tolist(),
                "action_env": np.asarray(action).reshape(-1).astype(float).tolist(),
            }

            f_log.write(json.dumps(row, ensure_ascii=False) + "\n")

            if term_value or trunc_value:
                break

    if args.save_video:
        env.flush_trajectory()
        env.flush_video()

    env.close()

    summary = {
        "milestone": "M3.5C",
        "description": "Near-success phase-aware safe BC debug video and step log.",
        "env_id": args.env_id,
        "seed": args.seed,
        "requested_max_steps": args.max_steps,
        "env_time_limit": env_time_limit,
        "phase_horizon": args.phase_horizon,
        "num_steps": len(rewards),
        "return": float(np.sum(rewards)),
        "success_once": bool(any(success_flags)),
        "final_success": bool(success_flags[-1]) if success_flags else False,
        "grasped_once": bool(any(grasped_flags)),
        "placed_once": bool(any(placed_flags)),
        "robot_static_once": bool(any(static_flags)),
        "final_is_grasped": bool(grasped_flags[-1]) if grasped_flags else False,
        "final_is_obj_placed": bool(placed_flags[-1]) if placed_flags else False,
        "final_is_robot_static": bool(static_flags[-1]) if static_flags else False,
        "initial_cube_goal_dist": float(cube_goal_dists[0]) if cube_goal_dists else None,
        "final_cube_goal_dist": float(cube_goal_dists[-1]) if cube_goal_dists else None,
        "min_cube_goal_dist": float(np.min(cube_goal_dists)) if cube_goal_dists else None,
        "mean_action_norm": float(np.mean(action_norms)) if action_norms else None,
        "max_action_norm": float(np.max(action_norms)) if action_norms else None,
        "safe_action_filter": {
            "expert_action_bounds": args.expert_action_bounds,
            "expert_bound_margin": args.expert_bound_margin,
            "hard_threshold_gripper": args.hard_threshold_gripper,
        },
        "step_log": str(step_log_path),
        "out_dir": str(out_dir),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[done] M3.5C debug rollout recorded")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()