from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch

from mani_skill.examples.motionplanning.panda.solutions.pick_cube import solve as solve_pick_cube

from m3_dagger_common import (
    ActionLoggingWrapper,
    apply_safe_action_filter,
    build_phase_obs,
    clip_action_to_env,
    extract_state_features,
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
    set_state_dict,
    solve_pick_cube_continuation,
    tree_to_numpy,
)


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.reshape(-1).astype(np.float32) - b.reshape(-1).astype(np.float32))))


def run_restore_smoke(
    env: gym.Env,
    seed: int,
    num_steps_before_restore: int,
) -> dict[str, Any]:
    env.reset(seed=seed)
    saved_state = tree_to_numpy(get_state_dict(env))
    before = extract_state_features(env)["state_obs"]

    for _ in range(num_steps_before_restore):
        action = clip_action_to_env(env, env.action_space.sample())
        env.step(action)

    drifted = extract_state_features(env)["state_obs"]
    set_state_dict(env, saved_state)
    restored = extract_state_features(env)["state_obs"]

    return {
        "seed": seed,
        "num_steps_before_restore": num_steps_before_restore,
        "state_obs_drift_before_restore": max_abs_diff(before, drifted),
        "state_obs_restore_max_abs_diff": max_abs_diff(before, restored),
    }


def make_policy_action(
    env: gym.Env,
    policy_source: str,
    model: torch.nn.Module | None,
    stats: dict[str, np.ndarray] | None,
    device: torch.device,
    step_idx: int,
    phase_horizon: int,
    prev_action: np.ndarray,
    expert_low: np.ndarray | None,
    expert_high: np.ndarray | None,
    hard_threshold_gripper: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    obs, features = build_phase_obs(
        env=env,
        step_idx=step_idx,
        phase_horizon=phase_horizon,
        prev_action=prev_action,
    )

    if policy_source == "phase_bc":
        if model is None or stats is None:
            raise RuntimeError("phase_bc policy_source requires model and normalization stats.")
        action = predict_action(model=model, stats=stats, obs_raw=obs, device=device)
    elif policy_source == "random":
        action = np.asarray(env.action_space.sample(), dtype=np.float32).reshape(-1)
    elif policy_source == "zero":
        action = np.zeros((8,), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported policy_source: {policy_source}")

    if expert_low is not None and expert_high is not None:
        action = apply_safe_action_filter(
            action=action,
            expert_low=expert_low,
            expert_high=expert_high,
            hard_threshold_gripper=hard_threshold_gripper,
        )

    return clip_action_to_env(env, action), obs, features


def roll_to_policy_state(
    env: gym.Env,
    seed: int,
    rollout_steps: int,
    policy_source: str,
    model: torch.nn.Module | None,
    stats: dict[str, np.ndarray] | None,
    device: torch.device,
    phase_horizon: int,
    expert_low: np.ndarray | None,
    expert_high: np.ndarray | None,
    hard_threshold_gripper: bool,
) -> dict[str, Any]:
    env.reset(seed=seed)
    prev_action = np.zeros((8,), dtype=np.float32)
    last_info: dict[str, Any] = {}
    elapsed_steps = 0

    for step_idx in range(rollout_steps):
        action, _, _ = make_policy_action(
            env=env,
            policy_source=policy_source,
            model=model,
            stats=stats,
            device=device,
            step_idx=step_idx,
            phase_horizon=phase_horizon,
            prev_action=prev_action,
            expert_low=expert_low,
            expert_high=expert_high,
            hard_threshold_gripper=hard_threshold_gripper,
        )

        _, reward, terminated, truncated, info = env.step(action)
        last_info = dict(info) if isinstance(info, dict) else {}
        prev_action = action.reshape(-1).astype(np.float32)
        elapsed_steps = step_idx + 1

        if scalar_bool(terminated) or scalar_bool(truncated):
            break

    obs, features = build_phase_obs(
        env=env,
        step_idx=elapsed_steps,
        phase_horizon=phase_horizon,
        prev_action=prev_action,
    )

    return {
        "env_state": tree_to_numpy(get_state_dict(env)),
        "obs": obs,
        "state_obs": features["state_obs"],
        "prev_action": prev_action,
        "elapsed_policy_steps": elapsed_steps,
        "cube_goal_dist": float(features["cube_goal_dist"][0]),
        "last_reward": scalar_float(last_info["reward"]) if "reward" in last_info else None,
        "last_success": read_info_flag(last_info, "success"),
        "last_is_grasped": read_info_flag(last_info, "is_grasped"),
        "last_is_obj_placed": read_info_flag(last_info, "is_obj_placed"),
        "last_is_robot_static": read_info_flag(last_info, "is_robot_static"),
    }


def call_solve_pick_cube(
    env: gym.Env,
    solver_call_mode: str,
    solver_seed: int | None,
    rollout_seed: int,
    vis: bool,
) -> tuple[Any, str]:
    if solver_call_mode == "no_seed":
        return solve_pick_cube(env, debug=False, vis=vis), "no_seed"
    if solver_call_mode == "seed_none":
        return solve_pick_cube(env, seed=None, debug=False, vis=vis), "seed_none"
    if solver_call_mode == "seed_value":
        seed_value = rollout_seed if solver_seed is None else solver_seed
        return solve_pick_cube(env, seed=seed_value, debug=False, vis=vis), "seed_value"
    if solver_call_mode != "auto":
        raise ValueError(f"Unsupported solver_call_mode: {solver_call_mode}")

    attempts: list[tuple[str, dict[str, Any]]] = [
        ("no_seed", {}),
        ("seed_none", {"seed": None}),
        ("seed_value", {"seed": rollout_seed if solver_seed is None else solver_seed}),
    ]
    type_errors: list[str] = []

    for mode, kwargs in attempts:
        try:
            return solve_pick_cube(env, debug=False, vis=vis, **kwargs), mode
        except TypeError as exc:
            type_errors.append(f"{mode}: {exc}")

    raise RuntimeError("All solve_pick_cube call modes failed with TypeError: " + "; ".join(type_errors))


def call_expert_planner(
    env: gym.Env,
    expert_mode: str,
    solver_call_mode: str,
    solver_seed: int | None,
    rollout_seed: int,
    vis: bool,
) -> tuple[Any, str]:
    if expert_mode == "continuation":
        return solve_pick_cube_continuation(env, debug=False, vis=vis), "continuation"
    if expert_mode == "stock_solve":
        return call_solve_pick_cube(
            env=env,
            solver_call_mode=solver_call_mode,
            solver_seed=solver_seed,
            rollout_seed=rollout_seed,
            vis=vis,
        )
    raise ValueError(f"Unsupported expert_mode: {expert_mode}")


def summarize_solver_result(result: Any) -> dict[str, Any]:
    if result == -1:
        return {"planner_returned_failure": True, "final_success": False, "elapsed_steps": 0}

    if not result:
        return {"planner_returned_failure": False, "final_success": False, "elapsed_steps": 0}

    final_info = result[-1]
    if not isinstance(final_info, dict):
        return {"planner_returned_failure": False, "final_success": False, "elapsed_steps": len(result)}

    return {
        "planner_returned_failure": False,
        "final_success": read_info_flag(final_info, "success"),
        "elapsed_steps": int(round(scalar_float(final_info.get("elapsed_steps", len(result))))),
    }


def run_planner_restore_smoke(
    env: ActionLoggingWrapper,
    args: argparse.Namespace,
    model: torch.nn.Module | None,
    stats: dict[str, np.ndarray] | None,
    device: torch.device,
    expert_low: np.ndarray | None,
    expert_high: np.ndarray | None,
) -> dict[str, Any]:
    candidate = roll_to_policy_state(
        env=env,
        seed=args.seed,
        rollout_steps=args.rollout_steps,
        policy_source=args.policy_source,
        model=model,
        stats=stats,
        device=device,
        phase_horizon=args.phase_horizon,
        expert_low=expert_low,
        expert_high=expert_high,
        hard_threshold_gripper=args.hard_threshold_gripper,
    )

    env.reset(seed=args.seed)
    set_state_dict(env, candidate["env_state"])
    restored_features = extract_state_features(env)
    restore_diff = max_abs_diff(candidate["state_obs"], restored_features["state_obs"])

    env.reset_log(expected_pre_state_obs=candidate["state_obs"])

    try:
        solver_result, solver_call_mode_used = call_expert_planner(
            env=env,
            expert_mode=args.expert_mode,
            solver_call_mode=args.solver_call_mode,
            solver_seed=args.solver_seed,
            rollout_seed=args.seed,
            vis=args.vis,
        )
        solver_error = None
    except Exception as exc:
        solver_result = -1
        solver_call_mode_used = args.solver_call_mode
        solver_error = repr(exc)

    solver_summary = summarize_solver_result(solver_result)

    first_action = env.logged_actions[0] if env.logged_actions else None
    first_pre_step_diff = (
        float(env.pre_step_state_obs_max_abs_diff[0])
        if env.pre_step_state_obs_max_abs_diff
        else None
    )

    can_use_first_action = (
        first_action is not None
        and solver_error is None
        and not solver_summary["planner_returned_failure"]
        and first_pre_step_diff is not None
        and first_pre_step_diff <= args.pre_step_tolerance
    )

    return {
        "seed": args.seed,
        "policy_source": args.policy_source,
        "rollout_steps": args.rollout_steps,
        "elapsed_policy_steps": candidate["elapsed_policy_steps"],
        "candidate_cube_goal_dist": candidate["cube_goal_dist"],
        "candidate_last_success": candidate["last_success"],
        "candidate_last_is_grasped": candidate["last_is_grasped"],
        "candidate_last_is_obj_placed": candidate["last_is_obj_placed"],
        "restore_state_obs_max_abs_diff": restore_diff,
        "solver_call_mode_requested": args.solver_call_mode,
        "solver_call_mode_used": solver_call_mode_used,
        "expert_mode": args.expert_mode,
        "solver_error": solver_error,
        "solver": solver_summary,
        "logged_num_actions": len(env.logged_actions),
        "first_pre_step_state_obs_max_abs_diff": first_pre_step_diff,
        "first_action": first_action.astype(float).tolist() if first_action is not None else None,
        "first_action_norm": float(np.linalg.norm(first_action)) if first_action is not None else None,
        "can_use_first_action_as_correction": bool(can_use_first_action),
        "pre_step_tolerance": args.pre_step_tolerance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="PickCube-v1")
    parser.add_argument("--seed", type=int, default=4000)
    parser.add_argument("--sim-backend", type=str, default="auto")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--phase-horizon", type=int, default=80)
    parser.add_argument("--restore-step-count", type=int, default=5)
    parser.add_argument("--rollout-steps", type=int, default=40)
    parser.add_argument("--policy-source", type=str, default="phase_bc", choices=["phase_bc", "random", "zero"])
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
    parser.add_argument(
        "--expert-mode",
        type=str,
        default="continuation",
        choices=["continuation", "stock_solve"],
    )
    parser.add_argument(
        "--solver-call-mode",
        type=str,
        default="auto",
        choices=["auto", "no_seed", "seed_none", "seed_value"],
    )
    parser.add_argument("--solver-seed", type=int, default=None)
    parser.add_argument("--pre-step-tolerance", type=float, default=1e-4)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--out-dir", type=str, default="outputs/m3_dagger_smoke_v0")
    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = None
    stats = None
    if args.policy_source == "phase_bc":
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

    base_env = gym.make(
        args.env_id,
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode=None,
        sim_backend=args.sim_backend,
    )
    env = ActionLoggingWrapper(base_env)
    set_env_time_limit(env, max_steps=args.max_steps)
    env_time_limit = get_env_time_limit(env, fallback=args.max_steps)

    restore_smoke = run_restore_smoke(
        env=env,
        seed=args.seed,
        num_steps_before_restore=args.restore_step_count,
    )
    planner_smoke = run_planner_restore_smoke(
        env=env,
        args=args,
        model=model,
        stats=stats,
        device=device,
        expert_low=expert_low,
        expert_high=expert_high,
    )

    env.close()

    summary = {
        "milestone": "M3.9A-smoke",
        "description": "State restore parity and restored-state PickCube planner relabel smoke test.",
        "env_id": args.env_id,
        "control_mode": "pd_joint_pos",
        "obs_mode": "none",
        "sim_backend": args.sim_backend,
        "device": str(device),
        "requested_max_steps": args.max_steps,
        "env_time_limit": env_time_limit,
        "phase_horizon": args.phase_horizon,
        "expert_mode": args.expert_mode,
        "safe_action_filter": {
            "enabled": not args.disable_safe_filter,
            "expert_action_bounds": args.expert_action_bounds if not args.disable_safe_filter else None,
            "expert_bound_margin": args.expert_bound_margin,
            "hard_threshold_gripper": args.hard_threshold_gripper,
        },
        "restore_smoke": restore_smoke,
        "planner_restore_smoke": planner_smoke,
        "passed": bool(
            restore_smoke["state_obs_restore_max_abs_diff"] <= args.pre_step_tolerance
            and planner_smoke["can_use_first_action_as_correction"]
        ),
    }

    summary_path = out_dir / "smoke_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[done] M3.9A DAgger smoke complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
