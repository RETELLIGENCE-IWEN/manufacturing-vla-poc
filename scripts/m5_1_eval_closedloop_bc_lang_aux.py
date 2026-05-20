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
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer


class LangBCPolicy(nn.Module):
    """M5.1 LangBCPolicyAux with the auxiliary task_head also loaded.

    Inference uses only `forward` (BC head). The classification head is loaded
    so the checkpoint's state_dict matches and may also be queried for debugging.
    """

    def __init__(
        self,
        obs_dim: int,
        lang_emb_dim: int,
        lang_proj_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        dropout: float,
        num_tasks: int = 3,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.lang_emb_dim = lang_emb_dim
        self.lang_proj_dim = lang_proj_dim
        self.num_tasks = num_tasks

        self.lang_proj = nn.Linear(lang_emb_dim, lang_proj_dim)
        self.task_head = nn.Linear(lang_proj_dim, num_tasks)

        layers: list[nn.Module] = []
        in_dim = obs_dim + lang_proj_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, lang_emb: torch.Tensor) -> torch.Tensor:
        lang_p = F.relu(self.lang_proj(lang_emb))
        x = torch.cat([obs, lang_p], dim=-1)
        return self.net(x)

    def predict_task(self, lang_emb: torch.Tensor) -> torch.Tensor:
        lang_p = F.relu(self.lang_proj(lang_emb))
        return self.task_head(lang_p)


def encode_instruction(
    text: str,
    text_encoder_name: str,
    device: torch.device,
) -> np.ndarray:
    tokenizer = CLIPTokenizer.from_pretrained(text_encoder_name)
    model = CLIPTextModel.from_pretrained(text_encoder_name).to(device)
    model.eval()
    with torch.no_grad():
        tokens = tokenizer([text], padding=True, truncation=True, return_tensors="pt").to(device)
        out = model(**tokens)
    return out.pooler_output[0].cpu().numpy().astype(np.float32)


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
    """
    Override ManiSkill TimeLimitWrapper horizon.

    PickCube-v1 may be wrapped with _max_episode_steps=50.
    This function aligns that internal horizon with the requested
    evaluation horizon.
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

        # Avoid touching max_episode_steps property because Gymnasium may warn.
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


def get_state_dict(env: gym.Env) -> dict[str, Any]:
    """
    Get ManiSkill state dict.

    Prefer unwrapped/base_env first to avoid Gymnasium wrapper warnings.
    """
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


GOAL_ACTOR_CANDIDATES = ["goal_site", "goal_region"]


def extract_state_features(env: gym.Env) -> dict[str, np.ndarray]:
    """
    Build the M5 multi-task live state layout.

    state_obs = panda (31) + cube (13) + goal (13) = 57.
    PickCube uses actor "goal_site"; PushCube/PullCube use "goal_region";
    fall back across the candidates.
    """
    state = get_state_dict(env)

    try:
        panda = squeeze_first_batch(state["articulations"]["panda"])
        cube = squeeze_first_batch(state["actors"]["cube"])
        actors = state["actors"]
        goal_key = next((k for k in GOAL_ACTOR_CANDIDATES if k in actors), None)
        if goal_key is None:
            raise KeyError(f"no goal actor found, tried {GOAL_ACTOR_CANDIDATES}, got {list(actors.keys())}")
        goal_site = squeeze_first_batch(actors[goal_key])
    except KeyError as exc:
        available = {
            "top_level": list(state.keys()),
            "actors": list(state.get("actors", {}).keys()) if isinstance(state.get("actors", {}), dict) else None,
            "articulations": list(state.get("articulations", {}).keys()) if isinstance(state.get("articulations", {}), dict) else None,
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


def build_phase_obs(
    env: gym.Env,
    step_idx: int,
    phase_horizon: int,
    prev_action: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Build phase-aware observation.

    obs = state_57 + progress_1 + prev_action_8
    total dim = 66
    """
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
) -> tuple[LangBCPolicy, dict[str, np.ndarray]]:
    checkpoint = torch.load(model_path, map_location=device)

    model = LangBCPolicy(
        obs_dim=int(checkpoint["obs_dim"]),
        lang_emb_dim=int(checkpoint["lang_emb_dim"]),
        lang_proj_dim=int(checkpoint["lang_proj_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        hidden_dims=[int(x) for x in checkpoint["hidden_dims"]],
        dropout=float(checkpoint["dropout"]),
        num_tasks=int(checkpoint.get("num_tasks", 3)),
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
    model: LangBCPolicy,
    stats: dict[str, np.ndarray],
    obs_raw: np.ndarray,
    lang_emb: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    obs_norm = ((obs_raw - stats["obs_mean"]) / stats["obs_std"]).astype(np.float32)

    obs_t = torch.from_numpy(obs_norm[None, :]).to(device)
    lang_t = torch.from_numpy(lang_emb[None, :]).to(device)
    pred_norm = model(obs_t, lang_t).cpu().numpy()[0].astype(np.float32)

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
    """
    Keep policy actions inside the expert action distribution.

    This is a diagnostic/safety filter:
    - clip all channels to expert min/max range
    - optionally force the last gripper-like dimension to {-1, +1}
    """
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


def read_info_flag(info: Any, key: str) -> bool:
    if isinstance(info, dict) and key in info:
        return scalar_bool(info[key])
    return False


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


def run_episode(
    env: gym.Env,
    policy_name: str,
    seed: int,
    max_steps: int,
    phase_horizon: int,
    model: LangBCPolicy | None,
    stats: dict[str, np.ndarray] | None,
    device: torch.device,
    lang_emb: np.ndarray | None = None,
    expert_low: np.ndarray | None = None,
    expert_high: np.ndarray | None = None,
    hard_threshold_gripper: bool = False,
    enable_final_hold: bool = False,
    hold_dist_threshold: float = 0.05,
    hold_trigger: str = "distance_or_placed",
    force_grip_while_far: bool = False,
    force_grip_dist_threshold: float = 0.05,
) -> dict[str, Any]:
    env.reset(seed=seed)

    rewards: list[float] = []
    success_flags: list[bool] = []
    is_grasped_flags: list[bool] = []
    is_obj_placed_flags: list[bool] = []
    is_robot_static_flags: list[bool] = []
    truncated_flags: list[bool] = []
    terminated_flags: list[bool] = []
    action_norms: list[float] = []
    cube_goal_dists: list[float] = []
    final_hold_used_flags: list[bool] = []

    prev_action = np.zeros((8,), dtype=np.float32)

    final_hold_active = False
    final_hold_activation_step: int | None = None
    is_obj_placed_latched = False
    has_grasped_before = False
    force_grip_used_flags: list[bool] = []

    for t in range(max_steps):
        used_final_hold_this_step = False

        if policy_name == "random":
            action = env.action_space.sample()
            features = extract_state_features(env)

        elif policy_name == "phase_bc":
            if model is None or stats is None:
                raise RuntimeError("phase_bc requested but model/stats are missing.")

            obs, features = build_phase_obs(
                env=env,
                step_idx=t,
                phase_horizon=phase_horizon,
                prev_action=prev_action,
            )

            if lang_emb is None:
                raise RuntimeError("phase_bc with LangBCPolicy requires a precomputed lang_emb.")

            action = predict_action(
                model=model,
                stats=stats,
                obs_raw=obs,
                lang_emb=lang_emb,
                device=device,
            )

            if expert_low is not None and expert_high is not None:
                action = apply_safe_action_filter(
                    action=action,
                    expert_low=expert_low,
                    expert_high=expert_high,
                    hard_threshold_gripper=hard_threshold_gripper,
                )

            cube_goal_dist_before_action = float(features["cube_goal_dist"][0])

            used_force_grip_this_step = False
            if (
                force_grip_while_far
                and has_grasped_before
                and cube_goal_dist_before_action > force_grip_dist_threshold
            ):
                action[-1] = -1.0
                used_force_grip_this_step = True
            force_grip_used_flags.append(used_force_grip_this_step)

            if enable_final_hold:
                hold_now = final_hold_active or should_activate_final_hold(
                    hold_trigger=hold_trigger,
                    cube_goal_dist=cube_goal_dist_before_action,
                    hold_dist_threshold=hold_dist_threshold,
                    is_obj_placed_latched=is_obj_placed_latched,
                )

                if hold_now:
                    if not final_hold_active:
                        final_hold_activation_step = t
                    final_hold_active = True
                    used_final_hold_this_step = True

                    # For pd_joint_pos, holding the previous joint target is safer
                    # than zeroing the action. This attempts to prevent the policy
                    # from continuing to push the cube after near-placement.
                    action = prev_action.copy()

            action = clip_action_to_env(env, action)

        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        _, reward, terminated, truncated, info = env.step(action)

        prev_action = np.asarray(action, dtype=np.float32).reshape(-1)

        reward_value = scalar_float(reward)
        term_value = scalar_bool(terminated)
        trunc_value = scalar_bool(truncated)

        success = read_info_flag(info, "success")
        is_grasped = read_info_flag(info, "is_grasped")
        is_obj_placed = read_info_flag(info, "is_obj_placed")
        is_robot_static = read_info_flag(info, "is_robot_static")

        if is_grasped:
            has_grasped_before = True

        if enable_final_hold and is_obj_placed:
            if hold_trigger in ["placed", "distance_or_placed"]:
                is_obj_placed_latched = True
                if not final_hold_active:
                    # This will be applied from the next step.
                    final_hold_activation_step = t + 1
                    final_hold_active = True

        rewards.append(reward_value)
        terminated_flags.append(term_value)
        truncated_flags.append(trunc_value)
        success_flags.append(success)
        is_grasped_flags.append(is_grasped)
        is_obj_placed_flags.append(is_obj_placed)
        is_robot_static_flags.append(is_robot_static)
        final_hold_used_flags.append(used_final_hold_this_step)

        action_norms.append(float(np.linalg.norm(prev_action)))
        cube_goal_dists.append(float(features["cube_goal_dist"][0]))

        if term_value or trunc_value:
            break

    return {
        "policy": policy_name,
        "seed": seed,
        "num_steps": len(rewards),
        "return": float(np.sum(rewards)),
        "success_once": bool(any(success_flags)),
        "final_success": bool(success_flags[-1]) if success_flags else False,
        "grasped_once": bool(any(is_grasped_flags)),
        "placed_once": bool(any(is_obj_placed_flags)),
        "robot_static_once": bool(any(is_robot_static_flags)),
        "final_is_grasped": bool(is_grasped_flags[-1]) if is_grasped_flags else False,
        "final_is_obj_placed": bool(is_obj_placed_flags[-1]) if is_obj_placed_flags else False,
        "final_is_robot_static": bool(is_robot_static_flags[-1]) if is_robot_static_flags else False,
        "terminated_once": bool(any(terminated_flags)),
        "truncated_once": bool(any(truncated_flags)),
        "final_terminated": bool(terminated_flags[-1]) if terminated_flags else False,
        "final_truncated": bool(truncated_flags[-1]) if truncated_flags else False,
        "mean_action_norm": float(np.mean(action_norms)) if action_norms else 0.0,
        "max_action_norm": float(np.max(action_norms)) if action_norms else 0.0,
        "initial_cube_goal_dist": float(cube_goal_dists[0]) if cube_goal_dists else None,
        "final_cube_goal_dist": float(cube_goal_dists[-1]) if cube_goal_dists else None,
        "min_cube_goal_dist": float(np.min(cube_goal_dists)) if cube_goal_dists else None,
        "final_hold_active": bool(final_hold_active),
        "final_hold_used_once": bool(any(final_hold_used_flags)),
        "final_hold_step_count": int(np.sum(final_hold_used_flags)) if final_hold_used_flags else 0,
        "final_hold_activation_step": final_hold_activation_step,
        "force_grip_used_once": bool(any(force_grip_used_flags)),
        "force_grip_step_count": int(np.sum(force_grip_used_flags)) if force_grip_used_flags else 0,
    }


def aggregate_bool_rate(episodes: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([bool(x[key]) for x in episodes])) if episodes else 0.0


def aggregate_float_mean(episodes: list[dict[str, Any]], key: str) -> float:
    values = [x[key] for x in episodes if x.get(key) is not None]
    return float(np.mean(values)) if values else 0.0


def aggregate_int_mean(episodes: list[dict[str, Any]], key: str) -> float:
    values = [int(x[key]) for x in episodes if x.get(key) is not None]
    return float(np.mean(values)) if values else 0.0


def evaluate_policy(
    policy_name: str,
    num_episodes: int,
    seed_start: int,
    max_steps: int,
    phase_horizon: int,
    env_id: str,
    sim_backend: str,
    model: LangBCPolicy | None,
    stats: dict[str, np.ndarray] | None,
    device: torch.device,
    lang_emb: np.ndarray | None = None,
    expert_low: np.ndarray | None = None,
    expert_high: np.ndarray | None = None,
    hard_threshold_gripper: bool = False,
    enable_final_hold: bool = False,
    hold_dist_threshold: float = 0.05,
    hold_trigger: str = "distance_or_placed",
    force_grip_while_far: bool = False,
    force_grip_dist_threshold: float = 0.05,
) -> dict[str, Any]:
    env = gym.make(
        env_id,
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode=None,
        sim_backend=sim_backend,
    )

    set_env_time_limit(env, max_steps=max_steps)
    env_time_limit = get_env_time_limit(env, fallback=max_steps)

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
            lang_emb=lang_emb,
            expert_low=expert_low,
            expert_high=expert_high,
            hard_threshold_gripper=hard_threshold_gripper,
            enable_final_hold=enable_final_hold,
            hold_dist_threshold=hold_dist_threshold,
            hold_trigger=hold_trigger,
            force_grip_while_far=force_grip_while_far,
            force_grip_dist_threshold=force_grip_dist_threshold,
        )

        result["episode"] = ep
        episodes.append(result)

        print(
            f"[{policy_name} ep={ep:03d}] "
            f"seed={seed} "
            f"success_once={result['success_once']} "
            f"grasped_once={result['grasped_once']} "
            f"placed_once={result['placed_once']} "
            f"final_static={result['final_is_robot_static']} "
            f"final_hold={result['final_hold_used_once']} "
            f"return={result['return']:.3f} "
            f"steps={result['num_steps']}"
        )

    env.close()

    return {
        "policy": policy_name,
        "num_episodes": num_episodes,
        "seed_start": seed_start,
        "max_steps": max_steps,
        "env_time_limit": env_time_limit,
        "phase_horizon": phase_horizon,
        "success_rate_once": aggregate_bool_rate(episodes, "success_once"),
        "final_success_rate": aggregate_bool_rate(episodes, "final_success"),
        "grasped_once_rate": aggregate_bool_rate(episodes, "grasped_once"),
        "placed_once_rate": aggregate_bool_rate(episodes, "placed_once"),
        "robot_static_once_rate": aggregate_bool_rate(episodes, "robot_static_once"),
        "final_grasped_rate": aggregate_bool_rate(episodes, "final_is_grasped"),
        "final_placed_rate": aggregate_bool_rate(episodes, "final_is_obj_placed"),
        "final_robot_static_rate": aggregate_bool_rate(episodes, "final_is_robot_static"),
        "terminated_once_rate": aggregate_bool_rate(episodes, "terminated_once"),
        "truncated_once_rate": aggregate_bool_rate(episodes, "truncated_once"),
        "final_hold_once_rate": aggregate_bool_rate(episodes, "final_hold_used_once"),
        "mean_final_hold_step_count": aggregate_int_mean(episodes, "final_hold_step_count"),
        "force_grip_once_rate": aggregate_bool_rate(episodes, "force_grip_used_once"),
        "mean_force_grip_step_count": aggregate_int_mean(episodes, "force_grip_step_count"),
        "mean_return": aggregate_float_mean(episodes, "return"),
        "mean_steps": aggregate_float_mean(episodes, "num_steps"),
        "mean_action_norm": aggregate_float_mean(episodes, "mean_action_norm"),
        "max_action_norm": float(np.max([x["max_action_norm"] for x in episodes])) if episodes else 0.0,
        "mean_initial_cube_goal_dist": aggregate_float_mean(episodes, "initial_cube_goal_dist"),
        "mean_final_cube_goal_dist": aggregate_float_mean(episodes, "final_cube_goal_dist"),
        "mean_min_cube_goal_dist": aggregate_float_mean(episodes, "min_cube_goal_dist"),
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
    parser.add_argument("--out-dir", type=str, default="runs/m3_bc_phase_aware/closedloop_eval_safe")
    parser.add_argument(
        "--expert-action-bounds",
        type=str,
        default="outputs/m3_phase_aware_dataset_100/action_bounds.json",
    )
    parser.add_argument("--expert-bound-margin", type=float, default=0.05)
    parser.add_argument("--hard-threshold-gripper", action="store_true")

    parser.add_argument("--enable-final-hold", action="store_true")
    parser.add_argument("--hold-dist-threshold", type=float, default=0.05)
    parser.add_argument(
        "--hold-trigger",
        type=str,
        default="distance_or_placed",
        choices=["distance", "placed", "distance_or_placed"],
    )

    parser.add_argument(
        "--force-grip-while-far",
        action="store_true",
        help="After first grasp, force gripper closed (action[-1]=-1) while cube_goal_dist > threshold.",
    )
    parser.add_argument("--force-grip-dist-threshold", type=float, default=0.05)

    parser.add_argument(
        "--instruction",
        type=str,
        default="Pick the bolt-like part and place it into the left fixture.",
        help="Natural-language instruction. Encoded once with CLIP at start and reused every step.",
    )
    parser.add_argument(
        "--text-encoder",
        type=str,
        default="openai/clip-vit-base-patch32",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(args.model)
    norm_path = Path(args.normalization)
    bounds_path = Path(args.expert_action_bounds)

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not norm_path.exists():
        raise FileNotFoundError(norm_path)
    if not bounds_path.exists():
        raise FileNotFoundError(bounds_path)

    model, stats = load_policy(
        model_path=model_path,
        norm_path=norm_path,
        device=device,
    )

    expert_low, expert_high = load_expert_action_bounds(
        bounds_path,
        margin=args.expert_bound_margin,
    )

    print(f"[m4-eval] encoding instruction with {args.text_encoder}: {args.instruction!r}")
    lang_emb = encode_instruction(
        text=args.instruction,
        text_encoder_name=args.text_encoder,
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
        expert_low=None,
        expert_high=None,
        hard_threshold_gripper=False,
        enable_final_hold=False,
        hold_dist_threshold=args.hold_dist_threshold,
        hold_trigger=args.hold_trigger,
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
        lang_emb=lang_emb,
        expert_low=expert_low,
        expert_high=expert_high,
        hard_threshold_gripper=args.hard_threshold_gripper,
        enable_final_hold=args.enable_final_hold,
        hold_dist_threshold=args.hold_dist_threshold,
        hold_trigger=args.hold_trigger,
        force_grip_while_far=args.force_grip_while_far,
        force_grip_dist_threshold=args.force_grip_dist_threshold,
    )

    summary = {
        "milestone": "M5C",
        "description": (
            "Multi-task language-conditioned phase-aware BC closed-loop evaluation. "
            "Handles PickCube/PushCube/PullCube envs (goal actor name auto-resolved)."
        ),
        "instruction": args.instruction,
        "text_encoder": args.text_encoder,
        "lang_emb_dim": int(lang_emb.shape[0]),
        "env_id": args.env_id,
        "control_mode": "pd_joint_pos",
        "obs_mode": "none",
        "model": str(model_path),
        "normalization": str(norm_path),
        "device": str(device),
        "num_episodes": args.num_episodes,
        "requested_max_steps": args.max_steps,
        "phase_horizon": args.phase_horizon,
        "seed": args.seed,
        "safe_action_filter": {
            "expert_action_bounds": str(bounds_path),
            "expert_bound_margin": args.expert_bound_margin,
            "hard_threshold_gripper": args.hard_threshold_gripper,
        },
        "final_hold_wrapper": {
            "enabled": args.enable_final_hold,
            "hold_dist_threshold": args.hold_dist_threshold,
            "hold_trigger": args.hold_trigger,
            "mode": "previous_action_hold",
        },
        "force_grip_while_far": {
            "enabled": args.force_grip_while_far,
            "dist_threshold": args.force_grip_dist_threshold,
            "trigger": "after_first_grasp",
        },
        "random": random_summary,
        "phase_bc": phase_bc_summary,
        "comparison": {
            "success_rate_once_delta": phase_bc_summary["success_rate_once"] - random_summary["success_rate_once"],
            "final_success_rate_delta": phase_bc_summary["final_success_rate"] - random_summary["final_success_rate"],
            "grasped_once_rate_delta": phase_bc_summary["grasped_once_rate"] - random_summary["grasped_once_rate"],
            "placed_once_rate_delta": phase_bc_summary["placed_once_rate"] - random_summary["placed_once_rate"],
            "final_robot_static_rate_delta": phase_bc_summary["final_robot_static_rate"] - random_summary["final_robot_static_rate"],
            "final_hold_once_rate_delta": phase_bc_summary["final_hold_once_rate"] - random_summary["final_hold_once_rate"],
            "mean_return_delta": phase_bc_summary["mean_return"] - random_summary["mean_return"],
            "mean_final_cube_goal_dist_delta": (
                phase_bc_summary["mean_final_cube_goal_dist"] - random_summary["mean_final_cube_goal_dist"]
            ),
            "mean_min_cube_goal_dist_delta": (
                phase_bc_summary["mean_min_cube_goal_dist"] - random_summary["mean_min_cube_goal_dist"]
            ),
        },
    }

    with (out_dir / "closedloop_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[done] M3.6A final-hold evaluation complete")
    print(json.dumps(summary["comparison"], indent=2))


if __name__ == "__main__":
    main()