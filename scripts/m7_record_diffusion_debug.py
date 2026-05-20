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
import math
import torch.nn.functional as F
from diffusers.schedulers import DDIMScheduler
from mani_skill.utils.wrappers.record import RecordEpisode
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModel


GOAL_ACTOR_CANDIDATES = ["goal_site", "goal_region"]


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        half = self.dim // 2
        emb = math.log(10000) / max(1, half - 1)
        emb = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb = x[:, None].float() * emb[None]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, padding=k // 2),
            nn.GroupNorm(min(n_groups, c_out), c_out),
            nn.Mish(),
        )
    def forward(self, x):
        return self.block(x)


class ResBlock1D(nn.Module):
    def __init__(self, c_in, c_out, cond_dim, k=3):
        super().__init__()
        self.b1 = Conv1dBlock(c_in, c_out, k)
        self.b2 = Conv1dBlock(c_out, c_out, k)
        self.cond_proj = nn.Linear(cond_dim, c_out * 2)
        self.residual = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
    def forward(self, x, cond):
        out = self.b1(x)
        scale_shift = self.cond_proj(cond)
        scale, shift = scale_shift.chunk(2, dim=-1)
        out = out * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        out = self.b2(out)
        return out + self.residual(x)


class VLAPolicy(nn.Module):
    """M7 DiffusionPolicy used for video recording. forward returns eps prediction."""

    def __init__(
        self,
        obs_dim: int,
        lang_emb_dim: int,
        lang_proj_dim: int,
        image_emb_dim: int,
        image_proj_dim: int,
        action_dim: int,
        num_tasks: int,
        cond_dim: int,
        hidden_dim: int,
        num_blocks: int,
        action_chunk: int,
        num_train_timesteps: int,
        time_dim: int = 128,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_chunk = action_chunk
        self.num_train_timesteps = num_train_timesteps
        self.lang_proj = nn.Linear(lang_emb_dim, lang_proj_dim)
        self.image_proj = nn.Linear(image_emb_dim, image_proj_dim)
        self.task_head = nn.Linear(lang_proj_dim, num_tasks)
        in_cond = obs_dim + lang_proj_dim + image_proj_dim
        self.cond_encoder = nn.Sequential(
            nn.Linear(in_cond, cond_dim),
            nn.Mish(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.time_emb = SinusoidalPosEmb(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )
        global_cond_dim = time_dim + cond_dim
        self.input_proj = nn.Conv1d(action_dim, hidden_dim, 3, padding=1)
        self.blocks = nn.ModuleList([
            ResBlock1D(hidden_dim, hidden_dim, global_cond_dim) for _ in range(num_blocks)
        ])
        self.output_proj = nn.Conv1d(hidden_dim, action_dim, 3, padding=1)

    def forward(self, noisy_actions, timesteps, obs, lang_emb, image_emb):
        lang_p = F.relu(self.lang_proj(lang_emb))
        image_p = F.relu(self.image_proj(image_emb))
        cond = self.cond_encoder(torch.cat([obs, lang_p, image_p], dim=-1))
        t_emb = self.time_mlp(self.time_emb(timesteps))
        global_cond = torch.cat([t_emb, cond], dim=-1)
        x = noisy_actions.permute(0, 2, 1)
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x, global_cond)
        return self.output_proj(x).permute(0, 2, 1)


class _LegacyVLAPolicy(nn.Module):
    """Unused. Kept for type tag only."""
    def __init__(
        self,
        obs_dim: int,
        lang_emb_dim: int,
        lang_proj_dim: int,
        image_emb_dim: int,
        image_proj_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        dropout: float,
        num_tasks: int = 3,
    ) -> None:
        super().__init__()
        self.lang_proj = nn.Linear(lang_emb_dim, lang_proj_dim)
        self.image_proj = nn.Linear(image_emb_dim, image_proj_dim)
        self.task_head = nn.Linear(lang_proj_dim, num_tasks)

        layers: list[nn.Module] = []
        in_dim = obs_dim + lang_proj_dim + image_proj_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs, lang_emb, image_emb):
        lang_p = F.relu(self.lang_proj(lang_emb))
        image_p = F.relu(self.image_proj(image_emb))
        x = torch.cat([obs, lang_p, image_p], dim=-1)
        return self.net(x)


class VisionEncoder:
    def __init__(self, name: str, device: torch.device):
        self.processor = CLIPImageProcessor.from_pretrained(name)
        self.model = CLIPVisionModel.from_pretrained(name).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def encode(self, image_uint8: np.ndarray) -> np.ndarray:
        if image_uint8.ndim == 4 and image_uint8.shape[0] == 1:
            image_uint8 = image_uint8[0]
        inputs = self.processor(images=image_uint8, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        return out.pooler_output[0].cpu().numpy().astype(np.float32)


def encode_instruction(text: str, encoder_name: str, device: torch.device) -> np.ndarray:
    tok = CLIPTokenizer.from_pretrained(encoder_name)
    mdl = CLIPTextModel.from_pretrained(encoder_name).to(device)
    mdl.eval()
    with torch.no_grad():
        inputs = tok([text], padding=True, truncation=True, return_tensors="pt").to(device)
        out = mdl(**inputs)
    return out.pooler_output[0].cpu().numpy().astype(np.float32)


def render_image_uint8(env: gym.Env) -> np.ndarray:
    img = env.render()
    if hasattr(img, "cpu"):
        img = img.cpu().numpy()
    arr = np.asarray(img)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


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
    actors = state["actors"]
    goal_key = next((k for k in GOAL_ACTOR_CANDIDATES if k in actors), None)
    if goal_key is None:
        raise KeyError(f"no goal actor in {list(actors.keys())}")
    goal_site = squeeze_first_batch(actors[goal_key])

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
) -> tuple[VLAPolicy, dict[str, np.ndarray]]:
    checkpoint = torch.load(model_path, map_location=device)

    model = VLAPolicy(
        obs_dim=int(checkpoint["obs_dim"]),
        lang_emb_dim=int(checkpoint["lang_emb_dim"]),
        lang_proj_dim=int(checkpoint["lang_proj_dim"]),
        image_emb_dim=int(checkpoint["image_emb_dim"]),
        image_proj_dim=int(checkpoint["image_proj_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        num_tasks=int(checkpoint.get("num_tasks", 3)),
        cond_dim=int(checkpoint["cond_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_blocks=int(checkpoint["num_blocks"]),
        action_chunk=int(checkpoint["action_chunk"]),
        num_train_timesteps=int(checkpoint["num_train_timesteps"]),
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
def sample_action_chunk(
    model: VLAPolicy,
    scheduler: DDIMScheduler,
    stats: dict[str, np.ndarray],
    obs_raw: np.ndarray,
    lang_emb: np.ndarray,
    image_emb: np.ndarray,
    device: torch.device,
    num_inference_steps: int,
) -> np.ndarray:
    obs_norm = ((obs_raw - stats["obs_mean"]) / stats["obs_std"]).astype(np.float32)
    obs_t = torch.from_numpy(obs_norm[None, :]).to(device)
    lang_t = torch.from_numpy(lang_emb[None, :]).to(device)
    image_t = torch.from_numpy(image_emb[None, :]).to(device)
    scheduler.set_timesteps(num_inference_steps)
    noisy = torch.randn((1, model.action_chunk, model.action_dim), device=device)
    for t in scheduler.timesteps:
        t_batch = torch.tensor([int(t)], device=device, dtype=torch.long)
        eps_pred = model(noisy, t_batch, obs_t, lang_t, image_t)
        noisy = scheduler.step(eps_pred, int(t), noisy).prev_sample
    chunk_norm = noisy[0].cpu().numpy().astype(np.float32)
    return (chunk_norm * stats["action_std"] + stats["action_mean"]).astype(np.float32)


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
    parser.add_argument("--env-id", type=str, required=True)
    parser.add_argument("--model", type=str, default="runs/m6_vla_aux_v0/best_model.pt")
    parser.add_argument("--normalization", type=str, default="runs/m6_vla_aux_v0/normalization_stats.npz")
    parser.add_argument("--expert-action-bounds", type=str, default="outputs/m6_multitask_vla_dataset/action_bounds.json")
    parser.add_argument("--expert-bound-margin", type=float, default=0.05)
    parser.add_argument("--hard-threshold-gripper", action="store_true")
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--phase-horizon", type=int, default=80)
    parser.add_argument("--out-dir", type=str, default="runs/m6_vla_aux_v0/debug_video")
    parser.add_argument("--save-video", action="store_true", default=True)
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--text-encoder", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--vision-encoder", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--render-resolution-policy", type=int, default=224,
                        help="Resolution of the image fed to the policy's CLIP visual encoder.")
    parser.add_argument("--render-resolution-video", type=int, default=512,
                        help="Resolution used when saving the .mp4 (separate camera).")
    parser.add_argument("--ignore-termination", action="store_true",
                        help="Do not stop the rollout when terminated/truncated fires; keep stepping until max_steps.")
    parser.add_argument("--num-inference-steps", type=int, default=16)
    parser.add_argument("--action-exec", type=int, default=4)
    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir) / f"{args.env_id}_seed_{args.seed}"
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

    print(f"[m6-record] encoding instruction: {args.instruction!r}")
    lang_emb = encode_instruction(args.instruction, args.text_encoder, device)
    print(f"[m6-record] loading vision encoder {args.vision_encoder}")
    vision_encoder = VisionEncoder(args.vision_encoder, device)

    env = gym.make(
        args.env_id,
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        human_render_camera_configs=dict(width=args.render_resolution_video, height=args.render_resolution_video),
        sim_backend="auto",
    )

    set_env_time_limit(env, max_steps=args.max_steps)
    env_time_limit = get_env_time_limit(env, fallback=args.max_steps)

    if args.save_video:
        env = RecordEpisode(
            env,
            output_dir=str(out_dir),
            trajectory_name=f"vla_{args.env_id.replace('-v1','').lower()}_seed_{args.seed}",
            save_video=True,
            source_type="policy_rollout",
            source_desc=f"M6 VLA debug rollout (env={args.env_id} instruction={args.instruction!r})",
            video_fps=30,
            record_reward=True,
            save_on_reset=False,
        )
        set_env_time_limit(env, max_steps=args.max_steps)

    scheduler = DDIMScheduler(
        num_train_timesteps=model.num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=True,
    )

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
    chunk_buffer = None
    chunk_idx = 0
    action_exec = args.action_exec
    num_inference_steps = args.num_inference_steps

    with step_log_path.open("w", encoding="utf-8") as f_log:
        for t in range(args.max_steps):
            obs, features = build_phase_obs(
                env=env,
                step_idx=t,
                phase_horizon=args.phase_horizon,
                prev_action=prev_action,
            )

            if chunk_buffer is None or chunk_idx >= action_exec:
                image_uint8 = render_image_uint8(env)
                image_emb = vision_encoder.encode(image_uint8)
                chunk_buffer = sample_action_chunk(
                    model=model,
                    scheduler=scheduler,
                    stats=stats,
                    obs_raw=obs,
                    lang_emb=lang_emb,
                    image_emb=image_emb,
                    device=device,
                    num_inference_steps=num_inference_steps,
                )
                chunk_idx = 0
            action = chunk_buffer[chunk_idx]
            chunk_idx += 1

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

            if (term_value or trunc_value) and not args.ignore_termination:
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