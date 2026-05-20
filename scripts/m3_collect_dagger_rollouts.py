from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch

from m3_dagger_common import (
    apply_safe_action_filter,
    build_phase_obs,
    clip_action_to_env,
    get_env_time_limit,
    get_state_dict,
    load_expert_action_bounds,
    load_policy,
    predict_action,
    read_info_flag,
    scalar_bool,
    scalar_float,
    set_env_time_limit,
    set_seed,
    should_activate_final_hold,
    tree_to_numpy,
)


def selection_reasons(
    step_idx: int,
    progress: float,
    cube_goal_dist: float,
    is_grasped: bool,
    placed_once_failed_late: bool,
    deteriorated_after_min_dist: bool,
    high_action_norm: bool,
    args: argparse.Namespace,
) -> list[str]:
    reasons: list[str] = []

    if args.use_progress_as_selection and progress >= args.min_progress:
        reasons.append("min_progress")
    if cube_goal_dist <= args.near_goal_dist:
        reasons.append("near_goal")
    if is_grasped:
        reasons.append("is_grasped")
    if placed_once_failed_late:
        reasons.append("placed_once_failed_late")
    if deteriorated_after_min_dist:
        reasons.append("deteriorated_after_min_dist")
    if high_action_norm:
        reasons.append("high_action_norm")
    if args.selection_mode == "all":
        reasons.append("all")

    return reasons


def build_selected_mask(
    progress: np.ndarray,
    cube_goal_dist: np.ndarray,
    is_grasped: np.ndarray,
    is_obj_placed: np.ndarray,
    success: np.ndarray,
    action_norm: np.ndarray,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[list[str]]]:
    T = int(progress.shape[0])

    if args.selection_mode == "all":
        selected = np.ones((T,), dtype=np.bool_)
    else:
        selected = np.zeros((T,), dtype=np.bool_)
        if args.use_progress_as_selection:
            selected |= progress.reshape(-1) >= args.min_progress
        selected |= cube_goal_dist.reshape(-1) <= args.near_goal_dist
        selected |= is_grasped.reshape(-1).astype(np.bool_)

        if T > 0 and not bool(success[-1]) and bool(np.any(is_obj_placed)):
            first_place = int(np.flatnonzero(is_obj_placed.astype(np.bool_))[0])
            selected[first_place:] = True

        if T > 0 and not bool(success[-1]):
            min_idx = int(np.argmin(cube_goal_dist))
            min_dist = float(cube_goal_dist[min_idx])
            deteriorated = (
                np.arange(T) >= min_idx
            ) & (cube_goal_dist.reshape(-1) >= min_dist + args.deterioration_margin)
            selected |= deteriorated

        if args.high_action_norm_quantile > 0.0 and T > 0:
            threshold = float(np.quantile(action_norm, args.high_action_norm_quantile))
            selected |= action_norm.reshape(-1) >= threshold

    if args.max_selected_per_episode > 0 and int(np.sum(selected)) > args.max_selected_per_episode:
        selected_indices = np.flatnonzero(selected)
        kept = rng.choice(
            selected_indices,
            size=args.max_selected_per_episode,
            replace=False,
        )
        new_selected = np.zeros_like(selected)
        new_selected[np.sort(kept)] = True
        selected = new_selected

    if T > 0 and not bool(success[-1]) and bool(np.any(is_obj_placed)):
        first_place = int(np.flatnonzero(is_obj_placed.astype(np.bool_))[0])
    else:
        first_place = T + 1

    min_idx = int(np.argmin(cube_goal_dist)) if T > 0 else T + 1
    min_dist = float(cube_goal_dist[min_idx]) if T > 0 else 0.0
    action_threshold = (
        float(np.quantile(action_norm, args.high_action_norm_quantile))
        if args.high_action_norm_quantile > 0.0 and T > 0
        else float("inf")
    )

    reasons_by_step: list[list[str]] = []
    for t in range(T):
        placed_once_failed_late = bool(
            not bool(success[-1])
            and first_place <= t
        )
        deteriorated_after_min_dist = bool(
            not bool(success[-1])
            and t >= min_idx
            and float(cube_goal_dist[t]) >= min_dist + args.deterioration_margin
        )
        high_action_norm = bool(float(action_norm[t]) >= action_threshold)
        reasons_by_step.append(
            selection_reasons(
                step_idx=t,
                progress=float(progress[t]),
                cube_goal_dist=float(cube_goal_dist[t]),
                is_grasped=bool(is_grasped[t]),
                placed_once_failed_late=placed_once_failed_late,
                deteriorated_after_min_dist=deteriorated_after_min_dist,
                high_action_norm=high_action_norm,
                args=args,
            )
        )

    return selected, reasons_by_step


def run_episode(
    env: gym.Env,
    episode_id: int,
    seed: int,
    model: torch.nn.Module,
    stats: dict[str, np.ndarray],
    device: torch.device,
    expert_low: np.ndarray | None,
    expert_high: np.ndarray | None,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    env.reset(seed=seed)

    env_states: list[dict[str, Any]] = []
    obs_rows: list[np.ndarray] = []
    state_obs_rows: list[np.ndarray] = []
    progress_rows: list[np.ndarray] = []
    prev_action_rows: list[np.ndarray] = []
    policy_action_raw_rows: list[np.ndarray] = []
    policy_action_safe_rows: list[np.ndarray] = []
    policy_action_env_rows: list[np.ndarray] = []
    reward_rows: list[float] = []
    success_rows: list[bool] = []
    is_grasped_rows: list[bool] = []
    is_obj_placed_rows: list[bool] = []
    is_robot_static_rows: list[bool] = []
    terminated_rows: list[bool] = []
    truncated_rows: list[bool] = []
    cube_goal_dist_rows: list[float] = []
    action_norm_rows: list[float] = []
    final_hold_used_rows: list[bool] = []

    prev_action = np.zeros((8,), dtype=np.float32)
    final_hold_active = False
    final_hold_activation_step: int | None = None
    is_obj_placed_latched = False

    for t in range(args.max_steps):
        env_state = tree_to_numpy(get_state_dict(env))

        obs, features = build_phase_obs(
            env=env,
            step_idx=t,
            phase_horizon=args.phase_horizon,
            prev_action=prev_action,
        )

        action_raw = predict_action(
            model=model,
            stats=stats,
            obs_raw=obs,
            device=device,
        )

        action_safe = action_raw.copy()
        if expert_low is not None and expert_high is not None:
            action_safe = apply_safe_action_filter(
                action=action_safe,
                expert_low=expert_low,
                expert_high=expert_high,
                hard_threshold_gripper=args.hard_threshold_gripper,
            )

        action_env = action_safe.copy()
        used_final_hold_this_step = False

        cube_goal_dist_before_action = float(features["cube_goal_dist"][0])
        if args.enable_final_hold:
            hold_now = final_hold_active or should_activate_final_hold(
                hold_trigger=args.hold_trigger,
                cube_goal_dist=cube_goal_dist_before_action,
                hold_dist_threshold=args.hold_dist_threshold,
                is_obj_placed_latched=is_obj_placed_latched,
            )

            if hold_now:
                if not final_hold_active:
                    final_hold_activation_step = t
                final_hold_active = True
                used_final_hold_this_step = True
                action_env = prev_action.copy()

        action_env = clip_action_to_env(env, action_env)

        _, reward, terminated, truncated, info = env.step(action_env)

        reward_value = scalar_float(reward)
        success = read_info_flag(info, "success")
        is_grasped = read_info_flag(info, "is_grasped")
        is_obj_placed = read_info_flag(info, "is_obj_placed")
        is_robot_static = read_info_flag(info, "is_robot_static")
        terminated_value = scalar_bool(terminated)
        truncated_value = scalar_bool(truncated)

        if args.enable_final_hold and is_obj_placed:
            if args.hold_trigger in ["placed", "distance_or_placed"]:
                is_obj_placed_latched = True
                if not final_hold_active:
                    final_hold_activation_step = t + 1
                    final_hold_active = True

        env_states.append(env_state)
        obs_rows.append(obs)
        state_obs_rows.append(features["state_obs"])
        progress_rows.append(np.asarray([obs[57]], dtype=np.float32))
        prev_action_rows.append(prev_action.copy())
        policy_action_raw_rows.append(action_raw.reshape(-1).astype(np.float32))
        policy_action_safe_rows.append(action_safe.reshape(-1).astype(np.float32))
        policy_action_env_rows.append(action_env.reshape(-1).astype(np.float32))
        reward_rows.append(reward_value)
        success_rows.append(success)
        is_grasped_rows.append(is_grasped)
        is_obj_placed_rows.append(is_obj_placed)
        is_robot_static_rows.append(is_robot_static)
        terminated_rows.append(terminated_value)
        truncated_rows.append(truncated_value)
        cube_goal_dist_rows.append(cube_goal_dist_before_action)
        action_norm_rows.append(float(np.linalg.norm(action_env)))
        final_hold_used_rows.append(used_final_hold_this_step)

        prev_action = action_env.reshape(-1).astype(np.float32)

        if terminated_value or truncated_value:
            break

    obs_arr = np.asarray(obs_rows, dtype=np.float32)
    progress_arr = np.asarray([x[0] for x in progress_rows], dtype=np.float32)
    cube_goal_dist_arr = np.asarray(cube_goal_dist_rows, dtype=np.float32)
    is_grasped_arr = np.asarray(is_grasped_rows, dtype=np.bool_)
    is_obj_placed_arr = np.asarray(is_obj_placed_rows, dtype=np.bool_)
    success_arr = np.asarray(success_rows, dtype=np.bool_)
    action_norm_arr = np.asarray(action_norm_rows, dtype=np.float32)

    selected_mask, reasons_by_step = build_selected_mask(
        progress=progress_arr,
        cube_goal_dist=cube_goal_dist_arr,
        is_grasped=is_grasped_arr,
        is_obj_placed=is_obj_placed_arr,
        success=success_arr,
        action_norm=action_norm_arr,
        rng=rng,
        args=args,
    )

    state_index_rows: list[dict[str, Any]] = []
    for t, selected in enumerate(selected_mask):
        if not selected:
            continue
        state_index_rows.append(
            {
                "episode_id": episode_id,
                "seed": seed,
                "step": t,
                "selected": True,
                "selection_reasons": reasons_by_step[t],
                "progress": float(progress_arr[t]),
                "cube_goal_dist": float(cube_goal_dist_arr[t]),
                "is_grasped": bool(is_grasped_arr[t]),
                "is_obj_placed": bool(is_obj_placed_arr[t]),
                "success": bool(success_arr[t]),
                "action_norm": float(action_norm_arr[t]),
            }
        )

    arrays = {
        "obs": obs_arr,
        "state_obs": np.asarray(state_obs_rows, dtype=np.float32),
        "progress": np.asarray(progress_rows, dtype=np.float32),
        "prev_action": np.asarray(prev_action_rows, dtype=np.float32),
        "policy_action_raw": np.asarray(policy_action_raw_rows, dtype=np.float32),
        "policy_action_safe": np.asarray(policy_action_safe_rows, dtype=np.float32),
        "policy_action_env": np.asarray(policy_action_env_rows, dtype=np.float32),
        "reward": np.asarray(reward_rows, dtype=np.float32),
        "success": np.asarray(success_rows, dtype=np.bool_),
        "is_grasped": np.asarray(is_grasped_rows, dtype=np.bool_),
        "is_obj_placed": np.asarray(is_obj_placed_rows, dtype=np.bool_),
        "is_robot_static": np.asarray(is_robot_static_rows, dtype=np.bool_),
        "terminated": np.asarray(terminated_rows, dtype=np.bool_),
        "truncated": np.asarray(truncated_rows, dtype=np.bool_),
        "cube_goal_dist": np.asarray(cube_goal_dist_rows, dtype=np.float32),
        "action_norm": np.asarray(action_norm_rows, dtype=np.float32),
        "final_hold_used": np.asarray(final_hold_used_rows, dtype=np.bool_),
        "selected_for_relabel": selected_mask.astype(np.bool_),
        "env_states": np.asarray(env_states, dtype=object),
    }

    summary = {
        "episode_id": episode_id,
        "seed": seed,
        "num_steps": int(obs_arr.shape[0]),
        "return": float(np.sum(reward_rows)) if reward_rows else 0.0,
        "success_once": bool(np.any(success_arr)) if success_arr.size else False,
        "final_success": bool(success_arr[-1]) if success_arr.size else False,
        "grasped_once": bool(np.any(is_grasped_arr)) if is_grasped_arr.size else False,
        "placed_once": bool(np.any(is_obj_placed_arr)) if is_obj_placed_arr.size else False,
        "final_is_grasped": bool(is_grasped_arr[-1]) if is_grasped_arr.size else False,
        "final_is_obj_placed": bool(is_obj_placed_arr[-1]) if is_obj_placed_arr.size else False,
        "final_cube_goal_dist": float(cube_goal_dist_arr[-1]) if cube_goal_dist_arr.size else None,
        "min_cube_goal_dist": float(np.min(cube_goal_dist_arr)) if cube_goal_dist_arr.size else None,
        "mean_action_norm": float(np.mean(action_norm_arr)) if action_norm_arr.size else None,
        "max_action_norm": float(np.max(action_norm_arr)) if action_norm_arr.size else None,
        "num_selected_for_relabel": int(np.sum(selected_mask)),
        "final_hold_active": bool(final_hold_active),
        "final_hold_activation_step": final_hold_activation_step,
    }

    return summary, arrays, state_index_rows


def aggregate_bool_rate(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([bool(row[key]) for row in rows])) if rows else 0.0


def aggregate_float_mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [row[key] for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="PickCube-v1")
    parser.add_argument("--model", type=str, default="runs/m3_bc_phase_aware/best_model.pt")
    parser.add_argument(
        "--normalization",
        type=str,
        default="runs/m3_bc_phase_aware/normalization_stats.npz",
    )
    parser.add_argument(
        "--expert-action-bounds",
        type=str,
        default="outputs/m3_phase_aware_dataset_100/action_bounds.json",
    )
    parser.add_argument("--expert-bound-margin", type=float, default=0.05)
    parser.add_argument("--disable-safe-filter", action="store_true")
    parser.add_argument("--hard-threshold-gripper", action="store_true")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=4000)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--phase-horizon", type=int, default=80)
    parser.add_argument("--sim-backend", type=str, default="auto")
    parser.add_argument("--out-dir", type=str, default="outputs/m3_dagger_rollouts_v0")
    parser.add_argument("--selection-mode", type=str, default="focused", choices=["focused", "all"])
    parser.add_argument("--min-progress", type=float, default=0.25)
    parser.add_argument(
        "--use-progress-as-selection",
        action="store_true",
        help="Select every state after min-progress. Off by default to keep focused DAgger recovery data.",
    )
    parser.add_argument("--near-goal-dist", type=float, default=0.12)
    parser.add_argument("--deterioration-margin", type=float, default=0.03)
    parser.add_argument("--high-action-norm-quantile", type=float, default=0.95)
    parser.add_argument("--max-selected-per-episode", type=int, default=40)
    parser.add_argument("--enable-final-hold", action="store_true")
    parser.add_argument("--hold-dist-threshold", type=float, default=0.05)
    parser.add_argument(
        "--hold-trigger",
        type=str,
        default="distance_or_placed",
        choices=["distance", "placed", "distance_or_placed"],
    )
    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    rollout_dir = out_dir / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, stats = load_policy(
        model_path=Path(args.model),
        norm_path=Path(args.normalization),
        device=device,
    )

    expert_low = None
    expert_high = None
    if not args.disable_safe_filter:
        expert_low, expert_high = load_expert_action_bounds(
            Path(args.expert_action_bounds),
            margin=args.expert_bound_margin,
        )

    env = gym.make(
        args.env_id,
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode=None,
        sim_backend=args.sim_backend,
    )
    set_env_time_limit(env, max_steps=args.max_steps)
    env_time_limit = get_env_time_limit(env, fallback=args.max_steps)

    rng = np.random.default_rng(args.seed)
    episode_summaries: list[dict[str, Any]] = []
    all_state_index_rows: list[dict[str, Any]] = []

    for episode_id in range(args.num_episodes):
        seed = args.seed + episode_id
        summary, arrays, state_index_rows = run_episode(
            env=env,
            episode_id=episode_id,
            seed=seed,
            model=model,
            stats=stats,
            device=device,
            expert_low=expert_low,
            expert_high=expert_high,
            rng=rng,
            args=args,
        )

        rollout_path = rollout_dir / f"rollout_{episode_id:06d}.npz"
        np.savez_compressed(rollout_path, **arrays)

        summary["rollout_path"] = str(rollout_path)
        for row in state_index_rows:
            row["rollout_path"] = str(rollout_path)

        episode_summaries.append(summary)
        all_state_index_rows.extend(state_index_rows)

        print(
            f"[collect ep={episode_id:03d}] "
            f"seed={seed} "
            f"success_once={summary['success_once']} "
            f"grasped_once={summary['grasped_once']} "
            f"placed_once={summary['placed_once']} "
            f"selected={summary['num_selected_for_relabel']} "
            f"steps={summary['num_steps']}"
        )

    env.close()

    state_index_path = out_dir / "state_index.jsonl"
    with state_index_path.open("w", encoding="utf-8") as f:
        for row in all_state_index_rows:
            f.write(json.dumps(row) + "\n")

    episodes_path = out_dir / "episodes.jsonl"
    with episodes_path.open("w", encoding="utf-8") as f:
        for row in episode_summaries:
            f.write(json.dumps(row) + "\n")

    summary = {
        "milestone": "M3.9B",
        "description": "Policy-visited phase-aware BC states collected for DAgger relabeling.",
        "env_id": args.env_id,
        "control_mode": "pd_joint_pos",
        "obs_mode": "none",
        "sim_backend": args.sim_backend,
        "model": args.model,
        "normalization": args.normalization,
        "device": str(device),
        "num_episodes": args.num_episodes,
        "seed_start": args.seed,
        "requested_max_steps": args.max_steps,
        "env_time_limit": env_time_limit,
        "phase_horizon": args.phase_horizon,
        "safe_action_filter": {
            "enabled": not args.disable_safe_filter,
            "expert_action_bounds": args.expert_action_bounds if not args.disable_safe_filter else None,
            "expert_bound_margin": args.expert_bound_margin,
            "hard_threshold_gripper": args.hard_threshold_gripper,
        },
        "final_hold": {
            "enabled": args.enable_final_hold,
            "hold_dist_threshold": args.hold_dist_threshold,
            "hold_trigger": args.hold_trigger,
        },
        "selection": {
            "selection_mode": args.selection_mode,
            "min_progress": args.min_progress,
            "use_progress_as_selection": args.use_progress_as_selection,
            "near_goal_dist": args.near_goal_dist,
            "deterioration_margin": args.deterioration_margin,
            "high_action_norm_quantile": args.high_action_norm_quantile,
            "max_selected_per_episode": args.max_selected_per_episode,
        },
        "success_rate_once": aggregate_bool_rate(episode_summaries, "success_once"),
        "final_success_rate": aggregate_bool_rate(episode_summaries, "final_success"),
        "grasped_once_rate": aggregate_bool_rate(episode_summaries, "grasped_once"),
        "placed_once_rate": aggregate_bool_rate(episode_summaries, "placed_once"),
        "mean_final_cube_goal_dist": aggregate_float_mean(episode_summaries, "final_cube_goal_dist"),
        "mean_min_cube_goal_dist": aggregate_float_mean(episode_summaries, "min_cube_goal_dist"),
        "num_selected_states": len(all_state_index_rows),
        "selection_reason_counts": dict(
            Counter(
                reason
                for row in all_state_index_rows
                for reason in row.get("selection_reasons", [])
            )
        ),
        "rollout_dir": str(rollout_dir),
        "state_index_jsonl": str(state_index_path),
        "episodes_jsonl": str(episodes_path),
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[done] M3.9B DAgger rollout collection complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
