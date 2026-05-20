"""M5-A: motion-planning expert demos for PickCube/PushCube/PullCube.

Same per-task pattern as m2_generate_pickcube_expert.py, but dispatches to the
correct ManiSkill solve() function based on `--env-id`.
Writes a separate H5 trajectory per task into a shared record directory so the
M5-B converter can merge them.
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
import torch
from tqdm import tqdm

from mani_skill.examples.motionplanning.panda.solutions.pick_cube import solve as solve_pick_cube
from mani_skill.examples.motionplanning.panda.solutions.pull_cube import solve as solve_pull_cube
from mani_skill.examples.motionplanning.panda.solutions.push_cube import solve as solve_push_cube
from mani_skill.utils.wrappers.record import RecordEpisode

TASK_SOLVERS: dict[str, Callable] = {
    "PickCube-v1": solve_pick_cube,
    "PushCube-v1": solve_push_cube,
    "PullCube-v1": solve_pull_cube,
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
    parser.add_argument("--record-dir", type=str, default="outputs/m5_expert_demos_multitask")
    parser.add_argument("--traj-name", type=str, default=None)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--vis", action="store_true")
    args = parser.parse_args()

    solver = TASK_SOLVERS[args.env_id]
    traj_name = args.traj_name or f"{args.env_id.replace('-v1','').lower()}_expert"

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
        source_desc=f"M5 motion planning expert ({args.env_id})",
        video_fps=30,
        record_reward=False,
        save_on_reset=False,
    )

    successes: list[bool] = []
    episode_lengths: list[int] = []
    failed_motion_plans = 0
    passed = 0
    seed = args.seed

    pbar = tqdm(total=args.num_traj, desc=f"{args.env_id} expert")

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
        "milestone": "M5A",
        "env_id": args.env_id,
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

    print(f"[done] M5A expert demos for {args.env_id} complete")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
