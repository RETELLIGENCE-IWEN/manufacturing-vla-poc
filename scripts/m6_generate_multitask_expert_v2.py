"""M6.2: motion-planning expert demos with settle solvers for both PushCube and PullCube.

PushCube and PullCube both inherit the same problem from ManiSkill stock solvers:
the planner stops the moment the cube center crosses the goal-region boundary,
i.e. native success_rate triggers but the cube does not actually settle inside
the goal. M6.1 fixed PushCube; M6.2 adds the same fix for PullCube.

PickCube is left to use the stock motion-planning solver (it already settles
inside the goal site).
"""

from __future__ import annotations

import argparse
import json
import os.path as osp
import time
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import sapien
import torch
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.solutions.pick_cube import solve as solve_pick_cube
from mani_skill.utils.wrappers.record import RecordEpisode
from tqdm import tqdm


def solve_push_cube_settle(env, seed=None, debug=False, vis=False):
    """Modified PushCube solver: extra push to drive cube toward goal center."""
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env, debug=debug, vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis, print_env_info=False,
    )
    env = env.unwrapped
    planner.close_gripper()

    reach_pose = sapien.Pose(
        p=env.obj.pose.sp.p + np.array([-0.05, 0, 0]),
        q=env.agent.tcp.pose.sp.q,
    )
    planner.move_to_pose_with_screw(reach_pose)

    boundary_pose = sapien.Pose(
        p=env.goal_region.pose.sp.p + np.array([-0.12, 0, 0]),
        q=env.agent.tcp.pose.sp.q,
    )
    planner.move_to_pose_with_screw(boundary_pose)

    settle_pose = sapien.Pose(
        p=env.goal_region.pose.sp.p + np.array([-0.06, 0, 0]),
        q=env.agent.tcp.pose.sp.q,
    )
    res = planner.move_to_pose_with_screw(settle_pose)

    planner.close()
    return res


def solve_pull_cube_settle(env, seed=None, debug=False, vis=False):
    """Modified PullCube solver: extra pull to drive cube toward goal center.

    Stock pull_cube.solve ends with tcp at goal_region.x + 0.05, leaving the cube
    just inside the goal-region boundary. We add a second pull that takes the tcp
    closer to the goal center (cube ends up nearer goal_region.x).
    """
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env, debug=debug, vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis, print_env_info=False,
    )
    env = env.unwrapped
    planner.close_gripper()

    # stage 1: reach in front of the cube (positive x side)
    reach_pose = sapien.Pose(
        p=env.obj.pose.sp.p + np.array([0.05, 0, 0]),
        q=env.agent.tcp.pose.sp.q,
    )
    planner.move_to_pose_with_screw(reach_pose)

    # stage 2: pull cube to the near boundary of the goal region (ManiSkill default)
    boundary_pose = sapien.Pose(
        p=env.goal_region.pose.sp.p + np.array([0.05, 0, 0]),
        q=env.agent.tcp.pose.sp.q,
    )
    planner.move_to_pose_with_screw(boundary_pose)

    # stage 3 (NEW): pull further toward goal center
    settle_pose = sapien.Pose(
        p=env.goal_region.pose.sp.p + np.array([-0.01, 0, 0]),
        q=env.agent.tcp.pose.sp.q,
    )
    res = planner.move_to_pose_with_screw(settle_pose)

    planner.close()
    return res


TASK_SOLVERS: dict[str, Callable] = {
    "PickCube-v1": solve_pick_cube,
    "PushCube-v1": solve_push_cube_settle,
    "PullCube-v1": solve_pull_cube_settle,
}


def to_bool(x: Any) -> bool:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return bool(np.asarray(x).mean() > 0.5)


def to_int(x: Any) -> int:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return int(np.asarray(x).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, required=True, choices=sorted(TASK_SOLVERS.keys()))
    parser.add_argument("--num-traj", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--only-count-success", action="store_true")
    parser.add_argument("--obs-mode", type=str, default="none")
    parser.add_argument("--sim-backend", type=str, default="auto")
    parser.add_argument("--render-mode", type=str, default="rgb_array")
    parser.add_argument("--shader", type=str, default="default")
    parser.add_argument("--record-dir", type=str, default="outputs/m6_expert_demos_multitask_v2")
    parser.add_argument("--traj-name", type=str, default=None)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--vis", action="store_true")
    args = parser.parse_args()

    solver = TASK_SOLVERS[args.env_id]
    traj_name = args.traj_name or f"{args.env_id.replace('-v1','').lower()}_expert_v2"

    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        sim_backend=args.sim_backend,
    )

    output_dir = osp.join(args.record_dir, args.env_id, "motionplanning")

    env = RecordEpisode(
        env,
        output_dir=output_dir,
        trajectory_name=traj_name,
        save_video=args.save_video,
        source_type="motionplanning",
        source_desc=f"M6.2 motion planning expert ({args.env_id})",
        video_fps=30, record_reward=False, save_on_reset=False,
    )

    successes: list[bool] = []
    episode_lengths: list[int] = []
    failed_motion_plans = 0
    passed = 0
    seed = args.seed

    pbar = tqdm(total=args.num_traj, desc=f"{args.env_id} expert v2")

    while passed < args.num_traj:
        try:
            result = solver(env, seed=seed, debug=False, vis=args.vis)
        except Exception as exc:
            print(f"[warn] motion planning failed at seed={seed}: {exc}")
            result = -1

        if result == -1:
            success = False
            elapsed_steps = 0
            failed_motion_plans += 1
        else:
            final_info = result[-1]
            success = to_bool(final_info["success"]) if "success" in final_info else False
            elapsed_steps = to_int(final_info["elapsed_steps"]) if "elapsed_steps" in final_info else 0

        if args.only_count_success and not success:
            env.flush_trajectory(save=False)
            if args.save_video:
                env.flush_video(save=False)
            seed += 1
            continue

        env.flush_trajectory()
        if args.save_video:
            env.flush_video()

        successes.append(success)
        episode_lengths.append(elapsed_steps)
        passed += 1
        seed += 1
        pbar.update(1)
        pbar.set_postfix(
            success_rate=float(np.mean(successes)) if successes else 0.0,
            failed_mp_rate=failed_motion_plans / max(1, seed - args.seed),
            avg_len=float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        )

    pbar.close()
    h5_path = env._h5_file.filename
    env.close()

    summary = {
        "milestone": "M6.2A",
        "env_id": args.env_id,
        "solver": "settle" if args.env_id in ("PushCube-v1","PullCube-v1") else "stock",
        "num_traj": args.num_traj,
        "seed_start": args.seed,
        "seed_end_exclusive": seed,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "num_success": int(np.sum(successes)) if successes else 0,
        "num_failed_motion_plans": failed_motion_plans,
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "h5_path": h5_path,
        "json_path": h5_path.replace(".h5", ".json"),
        "record_dir": output_dir,
    }

    summary_path = Path(output_dir) / f"{traj_name}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[done] M6.2A expert demos for {args.env_id} complete")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
