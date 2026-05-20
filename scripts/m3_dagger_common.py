from __future__ import annotations

import json
import random
import importlib
from pathlib import Path
from typing import Any

import gymnasium as gym
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


def read_info_flag(info: Any, key: str) -> bool:
    if isinstance(info, dict) and key in info:
        return scalar_bool(info[key])
    return False


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
        if "_max_episode_steps" in vars(candidate):
            candidate._max_episode_steps = int(max_steps)


def get_env_time_limit(env: gym.Env, fallback: int) -> int:
    candidates = [
        env,
        getattr(env, "unwrapped", None),
        getattr(env, "base_env", None),
    ]

    for candidate in candidates:
        if candidate is None:
            continue
        if "_max_episode_steps" in vars(candidate):
            value = getattr(candidate, "_max_episode_steps")
            if value is not None:
                return int(value)

    return int(fallback)


def _state_candidates(env: gym.Env) -> list[Any]:
    return [
        getattr(env, "unwrapped", None),
        getattr(env, "base_env", None),
        env,
    ]


def find_state_method(env: gym.Env, method_name: str) -> tuple[Any, Any]:
    for candidate in _state_candidates(env):
        if candidate is None:
            continue
        if hasattr(candidate, method_name):
            return candidate, getattr(candidate, method_name)

    raise RuntimeError(f"Could not find {method_name}() on env.")


def get_state_dict(env: gym.Env) -> dict[str, Any]:
    _, getter = find_state_method(env, "get_state_dict")
    return getter()


def tree_to_numpy(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().copy()
    if isinstance(x, np.ndarray):
        return x.copy()
    if isinstance(x, dict):
        return {key: tree_to_numpy(value) for key, value in x.items()}
    if isinstance(x, list):
        return [tree_to_numpy(value) for value in x]
    if isinstance(x, tuple):
        return tuple(tree_to_numpy(value) for value in x)
    return x


def tree_to_torch(x: Any, device: torch.device | str | None = None) -> Any:
    if isinstance(x, torch.Tensor):
        return x.to(device) if device is not None else x
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x, device=device)
    if isinstance(x, dict):
        return {key: tree_to_torch(value, device=device) for key, value in x.items()}
    if isinstance(x, list):
        return [tree_to_torch(value, device=device) for value in x]
    if isinstance(x, tuple):
        return tuple(tree_to_torch(value, device=device) for value in x)
    return x


def get_env_device(env: gym.Env) -> torch.device | str | None:
    for candidate in _state_candidates(env):
        if candidate is None:
            continue
        device = getattr(candidate, "device", None)
        if device is not None:
            return device
    return None


def set_state_dict(env: gym.Env, state: dict[str, Any]) -> None:
    candidate, setter = find_state_method(env, "set_state_dict")

    try:
        setter(state)
        return
    except Exception as numpy_exc:
        device = getattr(candidate, "device", None) or get_env_device(env)
        try:
            setter(tree_to_torch(state, device=device))
            return
        except Exception as torch_exc:
            raise RuntimeError(
                "set_state_dict failed with both numpy-like and torch-like state payloads. "
                f"numpy_error={numpy_exc!r}; torch_error={torch_exc!r}"
            ) from torch_exc


def squeeze_first_batch(x: Any) -> np.ndarray:
    arr = to_numpy(x).astype(np.float32)

    if arr.ndim >= 2 and arr.shape[0] == 1:
        arr = arr[0]

    return arr.reshape(-1).astype(np.float32)


def extract_state_features_from_state(state: dict[str, Any]) -> dict[str, np.ndarray]:
    try:
        panda = squeeze_first_batch(state["articulations"]["panda"])
        cube = squeeze_first_batch(state["actors"]["cube"])
        goal_site = squeeze_first_batch(state["actors"]["goal_site"])
    except KeyError as exc:
        available = {
            "top_level": list(state.keys()),
            "actors": list(state.get("actors", {}).keys())
            if isinstance(state.get("actors", {}), dict)
            else None,
            "articulations": list(state.get("articulations", {}).keys())
            if isinstance(state.get("articulations", {}), dict)
            else None,
        }
        raise KeyError(f"Missing expected state key: {exc}. Available keys: {available}") from exc

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


def extract_state_features(env: gym.Env) -> dict[str, np.ndarray]:
    return extract_state_features_from_state(get_state_dict(env))


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
    if prev_action.shape[0] != 8:
        raise ValueError(f"Unexpected prev_action dim: {prev_action.shape[0]}, expected 8")

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


def clip_action_to_env(env: gym.Env, action: np.ndarray) -> np.ndarray:
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    action = action.reshape(-1).astype(np.float32)

    if low.shape == action.shape and high.shape == action.shape:
        return np.clip(action, low, high).astype(np.float32)

    return action.astype(np.float32)


def load_expert_action_bounds(path: Path, margin: float) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        bounds = json.load(f)

    low = np.asarray(bounds["min"], dtype=np.float32)
    high = np.asarray(bounds["max"], dtype=np.float32)

    if low.shape != high.shape:
        raise ValueError(f"Action bound shape mismatch: low={low.shape}, high={high.shape}")

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

    if action.shape != expert_low.shape or action.shape != expert_high.shape:
        raise ValueError(
            f"Action/bound shape mismatch: action={action.shape}, "
            f"low={expert_low.shape}, high={expert_high.shape}"
        )

    action = np.clip(action, expert_low, expert_high).astype(np.float32)

    if hard_threshold_gripper:
        action[-1] = 1.0 if action[-1] >= 0.0 else -1.0

    return action.astype(np.float32)


def should_activate_final_hold(
    hold_trigger: str,
    cube_goal_dist: float,
    hold_dist_threshold: float,
    is_obj_placed_latched: bool,
) -> bool:
    distance_triggered = cube_goal_dist <= hold_dist_threshold

    if hold_trigger == "distance":
        return distance_triggered
    if hold_trigger == "placed":
        return is_obj_placed_latched
    if hold_trigger == "distance_or_placed":
        return distance_triggered or is_obj_placed_latched

    raise ValueError(f"Unsupported hold_trigger: {hold_trigger}")


def solve_pick_cube_continuation(env: gym.Env, debug: bool = False, vis: bool = False) -> Any:
    """
    Run the PickCube motion-planning solution from the current env state.

    ManiSkill's packaged PickCube solution resets the env at the top of solve().
    DAgger relabeling needs the opposite: keep the restored policy-visited state
    and ask the planner for a continuation action.
    """
    pick_cube_solution = importlib.import_module(
        "mani_skill.examples.motionplanning.panda.solutions.pick_cube"
    )
    sapien = getattr(pick_cube_solution, "sapien")
    PandaArmMotionPlanningSolver = getattr(
        pick_cube_solution,
        "PandaArmMotionPlanningSolver",
    )
    compute_grasp_info_by_obb = getattr(
        pick_cube_solution,
        "compute_grasp_info_by_obb",
    )
    get_actor_obb = getattr(pick_cube_solution, "get_actor_obb")

    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    task_env = env.unwrapped
    finger_length = 0.025

    try:
        obb = get_actor_obb(task_env.cube)
        approaching = np.array([0, 0, -1])
        target_closing = task_env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=finger_length,
        )

        closing = grasp_info["closing"]
        grasp_pose = task_env.agent.build_grasp_pose(
            approaching,
            closing,
            task_env.cube.pose.sp.p,
        )

        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        result = planner.move_to_pose_with_screw(reach_pose)
        if result == -1:
            return -1

        result = planner.move_to_pose_with_screw(grasp_pose)
        if result == -1:
            return -1

        planner.close_gripper()

        goal_pose = sapien.Pose(task_env.goal_site.pose.sp.p, grasp_pose.q)
        result = planner.move_to_pose_with_screw(goal_pose)
        return result
    finally:
        planner.close()


class ActionLoggingWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.logged_actions: list[np.ndarray] = []
        self.pre_step_state_obs: list[np.ndarray] = []
        self.pre_step_state_obs_max_abs_diff: list[float] = []
        self.expected_pre_state_obs: np.ndarray | None = None

    def reset_log(self, expected_pre_state_obs: np.ndarray | None = None) -> None:
        self.logged_actions.clear()
        self.pre_step_state_obs.clear()
        self.pre_step_state_obs_max_abs_diff.clear()
        if expected_pre_state_obs is None:
            self.expected_pre_state_obs = None
        else:
            self.expected_pre_state_obs = expected_pre_state_obs.reshape(-1).astype(np.float32)

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        action_np = np.asarray(to_numpy(action), dtype=np.float32).reshape(-1)
        self.logged_actions.append(action_np.copy())

        try:
            state_obs = extract_state_features(self)["state_obs"]
            self.pre_step_state_obs.append(state_obs.copy())
            if self.expected_pre_state_obs is not None:
                diff = float(np.max(np.abs(state_obs - self.expected_pre_state_obs)))
                self.pre_step_state_obs_max_abs_diff.append(diff)
        except Exception:
            pass

        return self.env.step(action)
