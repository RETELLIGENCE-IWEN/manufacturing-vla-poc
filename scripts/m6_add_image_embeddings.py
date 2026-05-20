"""M6-A: precompute per-step CLIP-vision embeddings by replaying expert trajectories.

For every episode in the M5 multi-task language-augmented dataset, this script:
  1. opens the source ManiSkill HDF5 + JSON to read `episode_seed` and `actions`
  2. spins up an `rgb_array` env at 224x224 with the matching env_id
  3. resets to the recorded seed
  4. steps through the recorded actions, rendering the camera image at every step
  5. batches the images through CLIPVisionModel to get per-timestep 768-dim embeddings
  6. writes a new dataset directory copying the existing NPZs with an additional
     `image_emb` array of shape (T, 768)

The resulting dataset has the layout

    obs (T, 66) + lang_emb (T, 512) + image_emb (T, 768)

which M6 trainers consume jointly.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import gymnasium as gym
import h5py
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
from transformers import CLIPImageProcessor, CLIPVisionModel


def to_numpy_image(img) -> np.ndarray:
    if hasattr(img, "cpu"):
        img = img.cpu().numpy()
    arr = np.asarray(img)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def load_h5_index(h5_path: Path) -> dict:
    """Return mapping traj_key -> {seed, actions(T, A)}."""
    json_path = h5_path.with_suffix(".json")
    with json_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    seed_by_ep = {f"traj_{ep['episode_id']}": int(ep["episode_seed"]) for ep in meta["episodes"]}

    index = {}
    with h5py.File(h5_path, "r") as h5:
        for traj_key, seed in seed_by_ep.items():
            actions = np.asarray(h5[traj_key]["actions"], dtype=np.float32)
            index[traj_key] = {"seed": seed, "actions": actions}
    return index


@torch.no_grad()
def encode_images_batch(
    images: list,
    processor: CLIPImageProcessor,
    model: CLIPVisionModel,
    device: torch.device,
) -> np.ndarray:
    inputs = processor(images=images, return_tensors="pt").to(device)
    out = model(**inputs)
    return out.pooler_output.cpu().numpy().astype(np.float32)


def render_and_embed_episode(
    env: gym.Env,
    seed: int,
    actions: np.ndarray,
    processor: CLIPImageProcessor,
    model: CLIPVisionModel,
    device: torch.device,
    encode_batch_size: int,
) -> np.ndarray:
    env.reset(seed=seed)
    images = []
    images.append(to_numpy_image(env.render()))

    for t in range(actions.shape[0] - 1):
        _, _, terminated, truncated, _ = env.step(actions[t])
        images.append(to_numpy_image(env.render()))
        term = bool(np.asarray(terminated).mean() > 0.5)
        trunc = bool(np.asarray(truncated).mean() > 0.5)
        if term or trunc:
            break

    T = actions.shape[0]
    while len(images) < T:
        images.append(images[-1])

    embeddings_chunks = []
    for start in range(0, T, encode_batch_size):
        chunk = images[start : start + encode_batch_size]
        embeddings_chunks.append(encode_images_batch(chunk, processor, model, device))
    return np.concatenate(embeddings_chunks, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=str, default="outputs/m5_multitask_lang_phase_aware_dataset")
    parser.add_argument("--out-dir", type=str, default="outputs/m6_multitask_vla_dataset")
    parser.add_argument("--clip-vision", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--render-resolution", type=int, default=224)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=-1, help="If >0, only process the first N episodes (smoke).")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "episodes").mkdir(exist_ok=True)

    with (in_dir / "episodes.jsonl").open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        records = records[: args.limit]
    print(f"[m6-A] {len(records)} episodes to process from {in_dir}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    processor = CLIPImageProcessor.from_pretrained(args.clip_vision)
    model = CLIPVisionModel.from_pretrained(args.clip_vision).to(device)
    model.eval()

    h5_caches: dict = {}
    env_cache: dict = {}

    def get_env(env_id: str) -> gym.Env:
        if env_id not in env_cache:
            env_cache[env_id] = gym.make(
                env_id,
                obs_mode="none",
                control_mode="pd_joint_pos",
                render_mode="rgb_array",
                human_render_camera_configs=dict(width=args.render_resolution, height=args.render_resolution),
                sim_backend="auto",
            )
        return env_cache[env_id]

    out_records = []
    for rec_idx, rec in enumerate(records):
        h5_path = Path(rec["source_h5"])
        if str(h5_path) not in h5_caches:
            h5_caches[str(h5_path)] = load_h5_index(h5_path)
        h5_index = h5_caches[str(h5_path)]

        traj_key = rec["source_traj_key"]
        if traj_key not in h5_index:
            raise KeyError(f"traj_key {traj_key} not in H5 {h5_path}")
        seed = h5_index[traj_key]["seed"]
        actions = h5_index[traj_key]["actions"]

        env_id = rec["base_env_id"]
        env = get_env(env_id)

        embeddings = render_and_embed_episode(
            env=env,
            seed=seed,
            actions=actions,
            processor=processor,
            model=model,
            device=device,
            encode_batch_size=args.encode_batch_size,
        )

        ep_path_in = in_dir / "episodes" / Path(rec["npz_path"]).name
        ep_path_out = out_dir / "episodes" / ep_path_in.name
        with np.load(ep_path_in, allow_pickle=True) as data:
            arrays = {k: data[k] for k in data.files}
        T = arrays["obs"].shape[0]
        if embeddings.shape[0] != T:
            raise ValueError(f"length mismatch for {ep_path_in}: obs T={T} image_emb T={embeddings.shape[0]}")
        arrays["image_emb"] = embeddings.astype(np.float32)
        np.savez_compressed(ep_path_out, **arrays)

        new_rec = dict(rec)
        new_rec["npz_path"] = str(ep_path_out)
        new_rec["image_emb_dim"] = int(embeddings.shape[1])
        out_records.append(new_rec)

        if (rec_idx + 1) % 10 == 0 or rec_idx == len(records) - 1:
            print(f"[m6-A] {rec_idx + 1}/{len(records)} episodes done ({env_id})")

    for env in env_cache.values():
        env.close()

    with (out_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    for fname in ["splits.json", "action_bounds.json", "dataset_schema.json"]:
        src = in_dir / fname
        if src.exists():
            shutil.copy(src, out_dir / fname)

    schema_path = out_dir / "dataset_schema.json"
    schema = json.load(schema_path.open("r", encoding="utf-8")) if schema_path.exists() else {}
    schema["image_emb_dim"] = int(out_records[0]["image_emb_dim"]) if out_records else 0
    schema["vision_encoder"] = args.clip_vision
    schema["render_resolution"] = args.render_resolution
    with schema_path.open("w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    summary_in = in_dir / "summary.json"
    summary = json.load(summary_in.open("r", encoding="utf-8")) if summary_in.exists() else {}
    summary["milestone"] = "M6A"
    summary["description"] = "Multi-task lang dataset augmented with CLIP-vision image embeddings (frozen)."
    summary["source_dataset_dir"] = str(in_dir)
    summary["vision_encoder"] = args.clip_vision
    summary["image_emb_dim"] = schema["image_emb_dim"]
    summary["render_resolution"] = args.render_resolution
    summary["num_episodes"] = len(out_records)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[m6-A] done. out_dir={out_dir}")
    print(json.dumps({k: summary[k] for k in ["milestone","num_episodes","vision_encoder","image_emb_dim","render_resolution"]}, indent=2))


if __name__ == "__main__":
    main()
