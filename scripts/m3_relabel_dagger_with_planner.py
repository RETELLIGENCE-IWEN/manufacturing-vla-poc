from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np

from mani_skill.examples.motionplanning.panda.solutions.pick_cube import solve as solve_pick_cube

from m3_dagger_common import (
    ActionLoggingWrapper,
    extract_state_features,
    read_info_flag,
    scalar_float,
    set_env_time_limit,
    set_state_dict,
    solve_pick_cube_continuation,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def as_state_dict(x: Any) -> dict[str, Any]:
    if isinstance(x, dict):
        return x
    if isinstance(x, np.ndarray) and x.shape == ():
        value = x.item()
        if isinstance(value, dict):
            return value
    if hasattr(x, "item"):
        value = x.item()
        if isinstance(value, dict):
            return value
    raise TypeError(f"Could not convert stored env state to dict: type={type(x)!r}")


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.reshape(-1).astype(np.float32) - b.reshape(-1).astype(np.float32))))


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
    if isinstance(result, int) and result == -1:
        return {
            "planner_returned_failure": True,
            "final_success": False,
            "elapsed_steps": 0,
        }

    if not result:
        return {
            "planner_returned_failure": False,
            "final_success": False,
            "elapsed_steps": 0,
        }

    final_info = result[-1]
    if not isinstance(final_info, dict):
        return {
            "planner_returned_failure": False,
            "final_success": False,
            "elapsed_steps": len(result),
        }

    elapsed_steps = final_info.get("elapsed_steps", len(result))
    return {
        "planner_returned_failure": False,
        "final_success": read_info_flag(final_info, "success"),
        "elapsed_steps": int(round(scalar_float(elapsed_steps))),
    }


def write_correction_episode(
    episode_dir: Path,
    episode_id: int,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    obs = np.asarray([row["obs"] for row in rows], dtype=np.float32)
    actions = np.asarray([row["expert_action"] for row in rows], dtype=np.float32)
    sample_weight = np.asarray([row["sample_weight"] for row in rows], dtype=np.float32)
    policy_action_env = np.asarray([row["policy_action_env"] for row in rows], dtype=np.float32)
    rollout_episode = np.asarray([row["rollout_episode"] for row in rows], dtype=np.int32)
    rollout_step = np.asarray([row["rollout_step"] for row in rows], dtype=np.int32)
    rollout_seed = np.asarray([row["rollout_seed"] for row in rows], dtype=np.int32)
    first_pre_step_state_obs_max_abs_diff = np.asarray(
        [row["first_pre_step_state_obs_max_abs_diff"] for row in rows],
        dtype=np.float32,
    )
    restored_state_obs_max_abs_diff = np.asarray(
        [row["restored_state_obs_max_abs_diff"] for row in rows],
        dtype=np.float32,
    )

    out_path = episode_dir / f"ep_{episode_id:06d}.npz"
    np.savez_compressed(
        out_path,
        obs=obs,
        actions=actions,
        sample_weight=sample_weight,
        policy_action_env=policy_action_env,
        rollout_episode=rollout_episode,
        rollout_step=rollout_step,
        rollout_seed=rollout_seed,
        first_pre_step_state_obs_max_abs_diff=first_pre_step_state_obs_max_abs_diff,
        restored_state_obs_max_abs_diff=restored_state_obs_max_abs_diff,
        source_is_dagger=np.ones((obs.shape[0],), dtype=np.bool_),
    )

    return {
        "episode_id": episode_id,
        "npz_path": str(out_path),
        "num_steps": int(obs.shape[0]),
        "obs_dim": int(obs.shape[1]),
        "action_dim": int(actions.shape[1]),
        "dataset_type": "dagger_planner_correction",
        "mean_sample_weight": float(np.mean(sample_weight)),
        "mean_first_pre_step_state_obs_max_abs_diff": float(np.mean(first_pre_step_state_obs_max_abs_diff)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=str, default="outputs/m3_dagger_rollouts_v0")
    parser.add_argument("--out-dir", type=str, default="outputs/m3_dagger_corrections_v0")
    parser.add_argument("--env-id", type=str, default="PickCube-v1")
    parser.add_argument("--sim-backend", type=str, default="auto")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--max-states", type=int, default=-1)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--dagger-weight", type=float, default=2.0)
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
    parser.add_argument("--restore-tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--require-planner-final-success",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--keep-failed-planner-first-action", action="store_true")
    parser.add_argument("--vis", action="store_true")
    args = parser.parse_args()

    rollout_dir = Path(args.rollout_dir)
    out_dir = Path(args.out_dir)
    episode_dir = out_dir / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)

    state_index_path = rollout_dir / "state_index.jsonl"
    if not state_index_path.exists():
        raise FileNotFoundError(state_index_path)

    state_rows = load_jsonl(state_index_path)
    if args.max_states > 0:
        state_rows = state_rows[: args.max_states]

    env = ActionLoggingWrapper(
        gym.make(
            args.env_id,
            obs_mode="none",
            control_mode="pd_joint_pos",
            render_mode=None,
            sim_backend=args.sim_backend,
        )
    )
    set_env_time_limit(env, max_steps=args.max_steps)

    rollout_cache: dict[str, Any] = {}
    correction_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []
    fail_reasons: Counter[str] = Counter()
    solver_modes: Counter[str] = Counter()

    def get_rollout_npz(path: str) -> Any:
        if path not in rollout_cache:
            rollout_cache[path] = np.load(path, allow_pickle=True)
        return rollout_cache[path]

    def flush_chunk() -> None:
        nonlocal correction_rows
        if not correction_rows:
            return
        episode_id = len(episode_records)
        record = write_correction_episode(
            episode_dir=episode_dir,
            episode_id=episode_id,
            rows=correction_rows,
        )
        episode_records.append(record)
        correction_rows = []

    for index_id, row in enumerate(state_rows):
        rollout_path = row["rollout_path"]
        step = int(row["step"])
        rollout = get_rollout_npz(rollout_path)
        obs = rollout["obs"][step].astype(np.float32)
        expected_state_obs = rollout["state_obs"][step].astype(np.float32)
        policy_action_env = rollout["policy_action_env"][step].astype(np.float32)
        env_state = as_state_dict(rollout["env_states"][step])
        rollout_seed = int(row["seed"])

        fail_reason: str | None = None
        solver_call_mode_used = args.solver_call_mode
        solver_summary: dict[str, Any] = {}
        solver_error: str | None = None

        try:
            env.reset(seed=rollout_seed)
            set_state_dict(env, env_state)
            restored_state_obs = extract_state_features(env)["state_obs"]
            restore_diff = max_abs_diff(expected_state_obs, restored_state_obs)
        except Exception as exc:
            restore_diff = float("inf")
            fail_reason = "restore_failed"
            solver_error = repr(exc)

        if fail_reason is None and restore_diff > args.restore_tolerance:
            fail_reason = "restore_state_mismatch"

        if fail_reason is None:
            env.reset_log(expected_pre_state_obs=expected_state_obs)
            try:
                solver_result, solver_call_mode_used = call_expert_planner(
                    env=env,
                    expert_mode=args.expert_mode,
                    solver_call_mode=args.solver_call_mode,
                    solver_seed=args.solver_seed,
                    rollout_seed=rollout_seed,
                    vis=args.vis,
                )
                solver_summary = summarize_solver_result(solver_result)
            except Exception as exc:
                solver_summary = {
                    "planner_returned_failure": True,
                    "final_success": False,
                    "elapsed_steps": 0,
                }
                solver_error = repr(exc)
                fail_reason = "planner_exception"

        if fail_reason is None:
            solver_modes[solver_call_mode_used] += 1
            first_pre_step_diff = (
                float(env.pre_step_state_obs_max_abs_diff[0])
                if env.pre_step_state_obs_max_abs_diff
                else None
            )

            if first_pre_step_diff is None:
                fail_reason = "planner_logged_no_pre_step_state"
            elif first_pre_step_diff > args.pre_step_tolerance:
                fail_reason = "planner_started_from_different_state"
            elif not env.logged_actions:
                fail_reason = "planner_logged_no_actions"
            elif solver_summary.get("planner_returned_failure", False):
                fail_reason = "planner_returned_failure"
            elif args.require_planner_final_success and not solver_summary.get("final_success", False):
                fail_reason = "planner_final_not_success"
        else:
            first_pre_step_diff = None

        if (
            fail_reason is not None
            and fail_reason in ["planner_returned_failure", "planner_final_not_success"]
            and args.keep_failed_planner_first_action
            and env.logged_actions
            and first_pre_step_diff is not None
            and first_pre_step_diff <= args.pre_step_tolerance
        ):
            fail_reason = None

        if fail_reason is None:
            expert_action = env.logged_actions[0].reshape(-1).astype(np.float32)
            if expert_action.shape[0] != policy_action_env.reshape(-1).shape[0]:
                fail_reason = "expert_action_shape_mismatch"
            else:
                correction_rows.append(
                    {
                        "obs": obs,
                        "expert_action": expert_action,
                        "sample_weight": float(args.dagger_weight),
                        "policy_action_env": policy_action_env,
                        "rollout_episode": int(row["episode_id"]),
                        "rollout_step": step,
                        "rollout_seed": rollout_seed,
                        "first_pre_step_state_obs_max_abs_diff": float(first_pre_step_diff),
                        "restored_state_obs_max_abs_diff": float(restore_diff),
                    }
                )

                if len(correction_rows) >= args.chunk_size:
                    flush_chunk()

        if fail_reason is not None:
            fail_reasons[fail_reason] += 1
            failure_rows.append(
                {
                    "index_id": index_id,
                    "episode_id": int(row["episode_id"]),
                    "seed": rollout_seed,
                    "step": step,
                    "rollout_path": rollout_path,
                    "fail_reason": fail_reason,
                    "solver_call_mode_used": solver_call_mode_used,
                    "expert_mode": args.expert_mode,
                    "solver_error": solver_error,
                    "solver_summary": solver_summary,
                    "restored_state_obs_max_abs_diff": restore_diff,
                    "first_pre_step_state_obs_max_abs_diff": first_pre_step_diff,
                    "selection_reasons": row.get("selection_reasons", []),
                    "cube_goal_dist": row.get("cube_goal_dist"),
                    "is_grasped": row.get("is_grasped"),
                    "is_obj_placed": row.get("is_obj_placed"),
                }
            )

        print(
            f"[relabel {index_id + 1:05d}/{len(state_rows):05d}] "
            f"ep={row['episode_id']} step={step} "
            f"ok={fail_reason is None} "
            f"reason={fail_reason or 'ok'}"
        )

    flush_chunk()

    for data in rollout_cache.values():
        data.close()
    env.close()

    splits = {
        "train": [record["episode_id"] for record in episode_records],
        "val": [],
    }

    with (out_dir / "splits.json").open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    with (out_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for record in episode_records:
            f.write(json.dumps(record) + "\n")

    with (out_dir / "relabel_failures.jsonl").open("w", encoding="utf-8") as f:
        for failure in failure_rows:
            f.write(json.dumps(failure) + "\n")

    num_success = int(sum(record["num_steps"] for record in episode_records))
    summary = {
        "milestone": "M3.9C",
        "description": "Planner first-action correction labels for selected policy-visited states.",
        "rollout_dir": str(rollout_dir),
        "out_dir": str(out_dir),
        "env_id": args.env_id,
        "control_mode": "pd_joint_pos",
        "obs_mode": "none",
        "sim_backend": args.sim_backend,
        "expert_mode": args.expert_mode,
        "max_requested_states": args.max_states,
        "num_attempted_states": len(state_rows),
        "num_successful_corrections": num_success,
        "success_rate": float(num_success / max(1, len(state_rows))),
        "num_failed": len(failure_rows),
        "fail_reasons": dict(fail_reasons),
        "solver_call_modes": dict(solver_modes),
        "dagger_weight": args.dagger_weight,
        "require_planner_final_success": args.require_planner_final_success,
        "keep_failed_planner_first_action": args.keep_failed_planner_first_action,
        "pre_step_tolerance": args.pre_step_tolerance,
        "restore_tolerance": args.restore_tolerance,
        "num_correction_episodes": len(episode_records),
        "episodes_jsonl": str(out_dir / "episodes.jsonl"),
        "splits_json": str(out_dir / "splits.json"),
        "failures_jsonl": str(out_dir / "relabel_failures.jsonl"),
    }

    schema = {
        "milestone": "M3.9C",
        "format_version": 1,
        "episode_npz_fields": {
            "obs": "[N, 66] float32; policy-visited state_57 + rollout progress + rollout prev_action",
            "actions": "[N, 8] float32; planner first-action correction",
            "sample_weight": "[N] float32; DAgger source weight",
            "policy_action_env": "[N, 8] float32; action taken by policy at this state",
            "source_is_dagger": "[N] bool",
        },
    }

    with (out_dir / "dataset_schema.json").open("w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[done] M3.9C DAgger planner relabeling complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
