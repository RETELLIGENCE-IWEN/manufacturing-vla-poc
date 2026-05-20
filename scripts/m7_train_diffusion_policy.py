"""M7: Diffusion Policy (Chi et al. 2023) on the M6.2 multi-task VLA dataset.

Architecture:
    cond_vector = MLP(concat(obs, lang_proj(lang_emb), image_proj(image_emb)))  [B, cond_dim]
    aux: task_head(lang_proj)  -> CE(task_id)        (kept from M5.1/M6 for instruction grounding)

    noisy_actions [B, T_chunk, A] + timestep -> ConditionalUnet1D -> eps_pred [B, T_chunk, A]
    DDPM training (T=100, eps prediction, squaredcos beta schedule).

Inference (in the eval script): DDIM with 16 steps + receding-horizon control.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from diffusers.schedulers import DDPMScheduler
from torch.utils.data import DataLoader, Dataset


# ---------- config ----------


@dataclass(frozen=True)
class TrainConfig:
    dataset_dir: str
    run_dir: str
    seed: int
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    hidden_dim: int
    num_blocks: int
    cond_dim: int
    action_chunk: int
    lang_proj_dim: int
    image_proj_dim: int
    aux_weight: float
    num_train_timesteps: int
    print_every_epochs: int


# ---------- dataset ----------


def load_episode_task_ids(dataset_dir: Path):
    with (dataset_dir / "episodes.jsonl").open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    task_ids_seen = []
    for rec in records:
        tid = rec.get("task_id")
        if tid is None:
            raise KeyError(f"missing task_id in episode {rec.get('episode_id')}")
        if tid not in task_ids_seen:
            task_ids_seen.append(tid)
    tid_to_int = {tid: i for i, tid in enumerate(task_ids_seen)}
    ep_to_int = {int(rec["episode_id"]): tid_to_int[rec["task_id"]] for rec in records}
    return ep_to_int, task_ids_seen


def load_splits(d: Path):
    with (d / "splits.json").open("r", encoding="utf-8") as f:
        s = json.load(f)
    return list(s["train"]), list(s["val"])


class DiffusionDataset(Dataset):
    """Random (episode, t) -> obs[t], lang_emb[t], image_emb[t] as conditioning;
    actions[t : t+chunk] as target (right-padded with the last action if short)."""

    def __init__(
        self,
        dataset_dir: Path,
        episode_ids,
        episode_to_task_int,
        action_chunk: int,
        obs_mean=None,
        obs_std=None,
        action_mean=None,
        action_std=None,
    ):
        self.action_chunk = action_chunk
        self.eps_data = []  # list of dicts per episode

        all_obs = []
        all_actions = []

        for ep_id in episode_ids:
            ep_path = dataset_dir / "episodes" / f"ep_{ep_id:06d}.npz"
            data = np.load(ep_path)
            obs = data["obs"].astype(np.float32)
            actions = data["actions"].astype(np.float32)
            lang_emb = data["lang_emb"].astype(np.float32)
            image_emb = data["image_emb"].astype(np.float32)
            task_int = episode_to_task_int[int(ep_id)]
            T = obs.shape[0]
            for arr, name in [(actions, "actions"), (lang_emb, "lang_emb"), (image_emb, "image_emb")]:
                if arr.shape[0] != T:
                    raise ValueError(f"{ep_path} {name} length mismatch")
            self.eps_data.append({
                "obs": obs,
                "actions": actions,
                "lang_emb": lang_emb,
                "image_emb": image_emb,
                "task_int": task_int,
                "T": T,
            })
            all_obs.append(obs)
            all_actions.append(actions)

        self.all_obs_raw = np.concatenate(all_obs, axis=0)
        self.all_actions_raw = np.concatenate(all_actions, axis=0)

        # index list of (ep_idx, t) for all valid sampling positions: t in [0, T-1]
        # we right-pad if t + chunk > T
        self._index = []
        for ep_idx, ep in enumerate(self.eps_data):
            for t in range(ep["T"]):
                self._index.append((ep_idx, t))

        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.action_mean = action_mean
        self.action_std = action_std

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        ep_idx, t = self._index[idx]
        ep = self.eps_data[ep_idx]
        T = ep["T"]
        obs_t = ep["obs"][t]
        lang_t = ep["lang_emb"][t]
        image_t = ep["image_emb"][t]
        task_int = ep["task_int"]

        # action chunk with right-pad (repeat last action)
        end = min(T, t + self.action_chunk)
        act_chunk = ep["actions"][t:end]
        if act_chunk.shape[0] < self.action_chunk:
            pad = np.repeat(act_chunk[-1:], self.action_chunk - act_chunk.shape[0], axis=0)
            act_chunk = np.concatenate([act_chunk, pad], axis=0)

        # normalize obs / actions if stats provided
        if self.obs_mean is not None:
            obs_t = (obs_t - self.obs_mean) / self.obs_std
        if self.action_mean is not None:
            act_chunk = (act_chunk - self.action_mean) / self.action_std

        return (
            torch.from_numpy(obs_t.astype(np.float32)),
            torch.from_numpy(lang_t.astype(np.float32)),
            torch.from_numpy(image_t.astype(np.float32)),
            torch.from_numpy(act_chunk.astype(np.float32)),
            torch.tensor(task_int, dtype=torch.long),
        )


# ---------- model ----------


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        emb = math.log(10000) / max(1, half - 1)
        emb = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb = x[:, None].float() * emb[None]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, padding=k // 2),
            nn.GroupNorm(min(n_groups, c_out), c_out),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ResBlock1D(nn.Module):
    """Conv1D residual block with FiLM conditioning (timestep + obs_cond)."""

    def __init__(self, c_in: int, c_out: int, cond_dim: int, k: int = 3):
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


class DiffusionPolicy(nn.Module):
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
        time_dim: int = 128,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.lang_emb_dim = lang_emb_dim
        self.lang_proj_dim = lang_proj_dim
        self.image_emb_dim = image_emb_dim
        self.image_proj_dim = image_proj_dim
        self.action_dim = action_dim
        self.num_tasks = num_tasks
        self.cond_dim = cond_dim

        self.lang_proj = nn.Linear(lang_emb_dim, lang_proj_dim)
        self.image_proj = nn.Linear(image_emb_dim, image_proj_dim)
        self.task_head = nn.Linear(lang_proj_dim, num_tasks)

        in_cond_dim = obs_dim + lang_proj_dim + image_proj_dim
        self.cond_encoder = nn.Sequential(
            nn.Linear(in_cond_dim, cond_dim),
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

    def encode_conditioning(self, obs, lang_emb, image_emb):
        lang_p = F.relu(self.lang_proj(lang_emb))
        image_p = F.relu(self.image_proj(image_emb))
        return self.cond_encoder(torch.cat([obs, lang_p, image_p], dim=-1)), lang_p

    def forward(self, noisy_actions, timesteps, obs, lang_emb, image_emb):
        # noisy_actions: [B, T_chunk, A]
        cond, lang_p = self.encode_conditioning(obs, lang_emb, image_emb)
        t_emb = self.time_mlp(self.time_emb(timesteps))
        global_cond = torch.cat([t_emb, cond], dim=-1)

        x = noisy_actions.permute(0, 2, 1)  # [B, A, T]
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x, global_cond)
        x = self.output_proj(x)
        eps = x.permute(0, 2, 1)  # [B, T, A]

        task_logits = self.task_head(lang_p)
        return eps, task_logits


# ---------- training ----------


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_yaml(p):
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_config(cfg):
    t = cfg["training"]; m = cfg["model"]; d = cfg["diffusion"]; lc = cfg.get("lang", {}); ic = cfg.get("image", {}); ac = cfg.get("aux", {})
    return TrainConfig(
        dataset_dir=str(cfg["dataset_dir"]),
        run_dir=str(cfg["run_dir"]),
        seed=int(cfg["seed"]),
        batch_size=int(t["batch_size"]),
        num_epochs=int(t["num_epochs"]),
        learning_rate=float(t["learning_rate"]),
        weight_decay=float(t["weight_decay"]),
        grad_clip_norm=float(t["grad_clip_norm"]),
        hidden_dim=int(m["hidden_dim"]),
        num_blocks=int(m["num_blocks"]),
        cond_dim=int(m["cond_dim"]),
        action_chunk=int(d["action_chunk"]),
        num_train_timesteps=int(d["num_train_timesteps"]),
        lang_proj_dim=int(lc.get("lang_proj_dim", 64)),
        image_proj_dim=int(ic.get("image_proj_dim", 128)),
        aux_weight=float(ac.get("aux_weight", 1.0)),
        print_every_epochs=int(cfg["logging"]["print_every_epochs"]),
    )


def compute_normalization(dataset: DiffusionDataset):
    eps = 1e-6
    om = dataset.all_obs_raw.mean(axis=0).astype(np.float32)
    os_ = (dataset.all_obs_raw.std(axis=0) + eps).astype(np.float32)
    am = dataset.all_actions_raw.mean(axis=0).astype(np.float32)
    as_ = (dataset.all_actions_raw.std(axis=0) + eps).astype(np.float32)
    return om, os_, am, as_


def train_one_epoch(model, loader, optimizer, scheduler, device, grad_clip, aux_weight, num_train_steps):
    model.train()
    tot = bc = aux = 0.0
    count = 0
    for obs, lang, image, actions, task_int in loader:
        obs = obs.to(device); lang = lang.to(device); image = image.to(device)
        actions = actions.to(device); task_int = task_int.to(device)

        noise = torch.randn_like(actions)
        timesteps = torch.randint(0, num_train_steps, (actions.shape[0],), device=device, dtype=torch.long)
        noisy_actions = scheduler.add_noise(actions, noise, timesteps)

        eps_pred, task_logits = model(noisy_actions, timesteps, obs, lang, image)
        loss_diff = F.mse_loss(eps_pred, noise)
        loss_aux = F.cross_entropy(task_logits, task_int)
        loss = loss_diff + aux_weight * loss_aux

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        n = obs.shape[0]
        tot += float(loss.item()) * n
        bc += float(loss_diff.item()) * n
        aux += float(loss_aux.item()) * n
        count += n
    return {"loss": tot / max(1, count), "diffusion_mse": bc / max(1, count), "aux_ce": aux / max(1, count)}


@torch.no_grad()
def evaluate(model, loader, scheduler, device, num_train_steps):
    model.eval()
    tot_mse = 0.0
    correct = 0
    total = 0
    count = 0
    for obs, lang, image, actions, task_int in loader:
        obs = obs.to(device); lang = lang.to(device); image = image.to(device)
        actions = actions.to(device); task_int = task_int.to(device)

        noise = torch.randn_like(actions)
        timesteps = torch.randint(0, num_train_steps, (actions.shape[0],), device=device, dtype=torch.long)
        noisy_actions = scheduler.add_noise(actions, noise, timesteps)

        eps_pred, task_logits = model(noisy_actions, timesteps, obs, lang, image)
        tot_mse += float(F.mse_loss(eps_pred, noise, reduction="sum").item())
        preds = task_logits.argmax(dim=-1)
        correct += int((preds == task_int).sum().item())
        total += int(task_int.shape[0])
        count += int(eps_pred.numel())
    return {"val_diffusion_mse": tot_mse / max(1, count), "val_task_acc": correct / max(1, total)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = parse_config(load_yaml(Path(args.config)))
    set_seed(cfg.seed)

    dataset_dir = Path(cfg.dataset_dir)
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ids, val_ids = load_splits(dataset_dir)
    ep_to_int, task_id_strings = load_episode_task_ids(dataset_dir)
    print(f"[m7] tasks={task_id_strings} | train={len(train_ids)} val={len(val_ids)}")

    train_raw = DiffusionDataset(dataset_dir, train_ids, ep_to_int, cfg.action_chunk)
    obs_mean, obs_std, action_mean, action_std = compute_normalization(train_raw)

    train_ds = DiffusionDataset(dataset_dir, train_ids, ep_to_int, cfg.action_chunk,
                                obs_mean=obs_mean, obs_std=obs_std, action_mean=action_mean, action_std=action_std)
    val_ds = DiffusionDataset(dataset_dir, val_ids, ep_to_int, cfg.action_chunk,
                              obs_mean=obs_mean, obs_std=obs_std, action_mean=action_mean, action_std=action_std)
    print(f"[m7] train transitions={len(train_ds)} val transitions={len(val_ds)}")

    sample = train_ds[0]
    obs_dim = sample[0].shape[0]
    lang_emb_dim = sample[1].shape[0]
    image_emb_dim = sample[2].shape[0]
    action_dim = sample[3].shape[1]
    num_tasks = len(task_id_strings)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffusionPolicy(
        obs_dim=obs_dim,
        lang_emb_dim=lang_emb_dim,
        lang_proj_dim=cfg.lang_proj_dim,
        image_emb_dim=image_emb_dim,
        image_proj_dim=cfg.image_proj_dim,
        action_dim=action_dim,
        num_tasks=num_tasks,
        cond_dim=cfg.cond_dim,
        hidden_dim=cfg.hidden_dim,
        num_blocks=cfg.num_blocks,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[m7] DiffusionPolicy params={n_params/1e6:.2f}M")

    scheduler = DDPMScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, drop_last=False)

    np.savez(run_dir / "normalization_stats.npz",
             obs_mean=obs_mean, obs_std=obs_std, action_mean=action_mean, action_std=action_std)

    with (run_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump({
            "milestone": "M7",
            "description": "Diffusion Policy (Chi et al. 2023) on multi-task VLA dataset.",
            "config": asdict(cfg),
            "obs_dim": obs_dim,
            "lang_emb_dim": lang_emb_dim,
            "image_emb_dim": image_emb_dim,
            "action_dim": action_dim,
            "num_tasks": num_tasks,
            "task_id_strings": task_id_strings,
            "num_params_M": n_params / 1e6,
        }, f, indent=2)

    curve = run_dir / "training_curve.csv"
    best = float("inf"); best_epoch = -1; best_task_acc = 0.0

    with curve.open("w", encoding="utf-8", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=["epoch", "train_loss", "train_diff_mse", "train_aux_ce", "val_diff_mse", "val_task_acc"])
        writer.writeheader()

        for epoch in range(1, cfg.num_epochs + 1):
            tm = train_one_epoch(model, train_loader, optimizer, scheduler, device, cfg.grad_clip_norm, cfg.aux_weight, cfg.num_train_timesteps)
            vm = evaluate(model, val_loader, scheduler, device, cfg.num_train_timesteps)
            writer.writerow({
                "epoch": epoch,
                "train_loss": tm["loss"],
                "train_diff_mse": tm["diffusion_mse"],
                "train_aux_ce": tm["aux_ce"],
                "val_diff_mse": vm["val_diffusion_mse"],
                "val_task_acc": vm["val_task_acc"],
            })
            f_csv.flush()

            if vm["val_diffusion_mse"] < best:
                best = vm["val_diffusion_mse"]
                best_epoch = epoch
                best_task_acc = vm["val_task_acc"]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "obs_dim": obs_dim,
                    "lang_emb_dim": lang_emb_dim,
                    "lang_proj_dim": cfg.lang_proj_dim,
                    "image_emb_dim": image_emb_dim,
                    "image_proj_dim": cfg.image_proj_dim,
                    "action_dim": action_dim,
                    "num_tasks": num_tasks,
                    "task_id_strings": task_id_strings,
                    "cond_dim": cfg.cond_dim,
                    "hidden_dim": cfg.hidden_dim,
                    "num_blocks": cfg.num_blocks,
                    "action_chunk": cfg.action_chunk,
                    "num_train_timesteps": cfg.num_train_timesteps,
                    "epoch": epoch,
                    "val_metrics": vm,
                    "training_type": "m7_diffusion_policy",
                }, run_dir / "best_model.pt")

            if epoch == 1 or epoch % cfg.print_every_epochs == 0 or epoch == cfg.num_epochs:
                print(f"[ep {epoch:04d}] train_loss={tm['loss']:.5f} diff={tm['diffusion_mse']:.5f} aux={tm['aux_ce']:.4f} val_diff={vm['val_diffusion_mse']:.5f} val_task={vm['val_task_acc']:.3f}")

    final = {
        "best_epoch": best_epoch,
        "best_val_diffusion_mse": best,
        "best_val_task_acc": best_task_acc,
        "curve_csv": str(curve),
        "best_model": str(run_dir / "best_model.pt"),
        "normalization_stats": str(run_dir / "normalization_stats.npz"),
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

    print("[done] M7 diffusion policy training complete")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
