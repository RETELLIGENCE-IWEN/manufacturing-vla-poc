"""M6-B: VLA training — state + lang + image (with aux task_id loss on lang_proj).

Input layout per transition:
    state_obs (57) + progress (1) + prev_action (8) = obs (66)
    lang_emb (512)   — CLIP text pooled output (frozen, broadcast over the episode)
    image_emb (768)  — CLIP vision pooled output (frozen, recomputed per step)

Architecture:
    lang_proj  : 512 -> lang_proj_dim   (learned)
    image_proj : 768 -> image_proj_dim  (learned)
    task_head  : lang_proj_dim -> num_tasks (auxiliary classifier on lang only)
    MLP input  : concat(obs, lang_proj, image_proj)

Loss:
    BC weighted-MSE + aux_weight * CE(task_logits, task_id)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset


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
    hidden_dims: list
    dropout: float
    print_every_epochs: int
    early_end: float
    mid_end: float
    early_weight: float
    mid_weight: float
    late_weight: float
    normalize_phase_weights: bool
    use_dataset_sample_weight: bool
    normalize_total_weights: bool
    gripper_weight: float
    lang_proj_dim: int
    image_proj_dim: int
    aux_weight: float


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


class VLADataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        episode_ids: list,
        episode_to_task_int: dict,
        early_end: float,
        mid_end: float,
        early_weight: float,
        mid_weight: float,
        late_weight: float,
        normalize_phase_weights: bool,
        use_dataset_sample_weight: bool,
        normalize_total_weights: bool,
        obs_mean=None,
        obs_std=None,
        action_mean=None,
        action_std=None,
    ):
        obs_list, lang_list, img_list, act_list, w_list, task_list = [], [], [], [], [], []

        for episode_id in episode_ids:
            ep_path = dataset_dir / "episodes" / f"ep_{episode_id:06d}.npz"
            if not ep_path.exists():
                raise FileNotFoundError(ep_path)

            data = np.load(ep_path)
            obs = data["obs"].astype(np.float32)
            actions = data["actions"].astype(np.float32)
            for k in ("lang_emb", "image_emb"):
                if k not in data.files:
                    raise KeyError(f"{ep_path} missing {k}")
            lang_emb = data["lang_emb"].astype(np.float32)
            image_emb = data["image_emb"].astype(np.float32)

            if use_dataset_sample_weight and "sample_weight" in data.files:
                weights = data["sample_weight"].astype(np.float32).reshape(-1)
            else:
                weights = np.ones((obs.shape[0],), dtype=np.float32)

            T = obs.shape[0]
            for arr, name in [(actions, "actions"), (lang_emb, "lang_emb"), (image_emb, "image_emb")]:
                if arr.shape[0] != T:
                    raise ValueError(f"length mismatch in {ep_path}: {name}={arr.shape}")

            task_int = episode_to_task_int[int(episode_id)]
            task_arr = np.full((T,), task_int, dtype=np.int64)

            obs_list.append(obs)
            lang_list.append(lang_emb)
            img_list.append(image_emb)
            act_list.append(actions)
            w_list.append(weights)
            task_list.append(task_arr)

        self.obs_raw = np.concatenate(obs_list, axis=0).astype(np.float32)
        self.lang_emb = np.concatenate(lang_list, axis=0).astype(np.float32)
        self.image_emb = np.concatenate(img_list, axis=0).astype(np.float32)
        self.actions_raw = np.concatenate(act_list, axis=0).astype(np.float32)
        self.dataset_weights_raw = np.concatenate(w_list, axis=0).astype(np.float32)
        self.task_int = np.concatenate(task_list, axis=0).astype(np.int64)

        self.progress_raw = self.obs_raw[:, 57].astype(np.float32)
        self.phase_weights = self._build_phase_weights(
            self.progress_raw, early_end, mid_end, early_weight, mid_weight, late_weight, normalize_phase_weights
        ).astype(np.float32)
        self.sample_weights = (self.phase_weights * self.dataset_weights_raw).astype(np.float32)
        if normalize_total_weights:
            mean_w = float(np.mean(self.sample_weights))
            if mean_w > 1e-8:
                self.sample_weights = (self.sample_weights / mean_w).astype(np.float32)

        self.obs_mean, self.obs_std = obs_mean, obs_std
        self.action_mean, self.action_std = action_mean, action_std

        self.obs = ((self.obs_raw - obs_mean) / obs_std).astype(np.float32) if obs_mean is not None else self.obs_raw
        self.actions = ((self.actions_raw - action_mean) / action_std).astype(np.float32) if action_mean is not None else self.actions_raw

    @staticmethod
    def _build_phase_weights(progress, early_end, mid_end, early_weight, mid_weight, late_weight, normalize):
        if not (0.0 < early_end < mid_end < 1.0):
            raise ValueError(f"Expected 0 < early_end < mid_end < 1, got {early_end}, {mid_end}")
        w = np.full_like(progress, fill_value=late_weight, dtype=np.float32)
        w[progress < mid_end] = mid_weight
        w[progress < early_end] = early_weight
        if normalize:
            m = float(np.mean(w))
            if m > 1e-8:
                w = w / m
        return w.astype(np.float32)

    def __len__(self):
        return int(self.obs.shape[0])

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.lang_emb[idx]),
            torch.from_numpy(self.image_emb[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.tensor(self.sample_weights[idx], dtype=torch.float32),
            torch.tensor(self.task_int[idx], dtype=torch.long),
        )


class VLAPolicyAux(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        lang_emb_dim: int,
        lang_proj_dim: int,
        image_emb_dim: int,
        image_proj_dim: int,
        action_dim: int,
        num_tasks: int,
        hidden_dims: list,
        dropout: float,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.lang_emb_dim = lang_emb_dim
        self.lang_proj_dim = lang_proj_dim
        self.image_emb_dim = image_emb_dim
        self.image_proj_dim = image_proj_dim
        self.num_tasks = num_tasks

        self.lang_proj = nn.Linear(lang_emb_dim, lang_proj_dim)
        self.image_proj = nn.Linear(image_emb_dim, image_proj_dim)
        self.task_head = nn.Linear(lang_proj_dim, num_tasks)

        layers = []
        in_dim = obs_dim + lang_proj_dim + image_proj_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def _projections(self, lang_emb, image_emb):
        return F.relu(self.lang_proj(lang_emb)), F.relu(self.image_proj(image_emb))

    def forward(self, obs, lang_emb, image_emb):
        lang_p, image_p = self._projections(lang_emb, image_emb)
        x = torch.cat([obs, lang_p, image_p], dim=-1)
        return self.net(x)

    def forward_with_aux(self, obs, lang_emb, image_emb):
        lang_p, image_p = self._projections(lang_emb, image_emb)
        x = torch.cat([obs, lang_p, image_p], dim=-1)
        action = self.net(x)
        task_logits = self.task_head(lang_p)
        return action, task_logits


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_yaml(p):
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_config(cfg):
    training = cfg["training"]; model = cfg["model"]; loss = cfg["loss"]
    lang = cfg.get("lang", {}); image = cfg.get("image", {}); aux = cfg.get("aux", {})
    return TrainConfig(
        dataset_dir=str(cfg["dataset_dir"]),
        run_dir=str(cfg["run_dir"]),
        seed=int(cfg["seed"]),
        batch_size=int(training["batch_size"]),
        num_epochs=int(training["num_epochs"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        grad_clip_norm=float(training["grad_clip_norm"]),
        hidden_dims=[int(x) for x in model["hidden_dims"]],
        dropout=float(model["dropout"]),
        print_every_epochs=int(cfg["logging"]["print_every_epochs"]),
        early_end=float(loss["early_end"]),
        mid_end=float(loss["mid_end"]),
        early_weight=float(loss["early_weight"]),
        mid_weight=float(loss["mid_weight"]),
        late_weight=float(loss["late_weight"]),
        normalize_phase_weights=bool(loss.get("normalize_phase_weights", True)),
        use_dataset_sample_weight=bool(loss.get("use_dataset_sample_weight", True)),
        normalize_total_weights=bool(loss.get("normalize_total_weights", False)),
        gripper_weight=float(loss.get("gripper_weight", 1.0)),
        lang_proj_dim=int(lang.get("lang_proj_dim", 64)),
        image_proj_dim=int(image.get("image_proj_dim", 128)),
        aux_weight=float(aux.get("aux_weight", 1.0)),
    )


def load_splits(d):
    with (d / "splits.json").open("r", encoding="utf-8") as f:
        s = json.load(f)
    return list(s["train"]), list(s["val"])


def compute_normalization(ds):
    eps = 1e-6
    return (
        ds.obs_raw.mean(0).astype(np.float32),
        (ds.obs_raw.std(0) + eps).astype(np.float32),
        ds.actions_raw.mean(0).astype(np.float32),
        (ds.actions_raw.std(0) + eps).astype(np.float32),
    )


@torch.no_grad()
def evaluate(model, loader, device, action_mean, action_std):
    model.eval()
    mse_sum = mse_raw_sum = mae_raw_sum = 0.0
    count = 0
    correct = 0
    total_clf = 0
    am = torch.from_numpy(action_mean).to(device)
    asd = torch.from_numpy(action_std).to(device)

    for obs, lang_emb, image_emb, actions, _w, task_int in loader:
        obs = obs.to(device); lang_emb = lang_emb.to(device); image_emb = image_emb.to(device)
        actions = actions.to(device); task_int = task_int.to(device)
        pred, logits = model.forward_with_aux(obs, lang_emb, image_emb)
        mse_sum += float(F.mse_loss(pred, actions, reduction="sum").item())
        pred_raw = pred * asd + am
        act_raw = actions * asd + am
        mse_raw_sum += float(F.mse_loss(pred_raw, act_raw, reduction="sum").item())
        mae_raw_sum += float(F.l1_loss(pred_raw, act_raw, reduction="sum").item())
        preds = logits.argmax(dim=-1)
        correct += int((preds == task_int).sum().item())
        total_clf += int(task_int.shape[0])
        count += obs.shape[0] * actions.shape[1]

    return {
        "mse_norm": mse_sum / max(1, count),
        "mse_raw": mse_raw_sum / max(1, count),
        "mae_raw": mae_raw_sum / max(1, count),
        "task_acc": correct / max(1, total_clf),
    }


def weighted_action_mse(pred, target, w, channel_w):
    sq = (pred - target) ** 2 * channel_w[None, :]
    return (sq.mean(dim=1) * w).mean()


def train_one_epoch(model, loader, optim, device, clip, channel_w, aux_weight):
    model.train()
    tot = bc = aux = 0.0
    count = 0
    for obs, lang_emb, image_emb, actions, w, task_int in loader:
        obs = obs.to(device); lang_emb = lang_emb.to(device); image_emb = image_emb.to(device)
        actions = actions.to(device); w = w.to(device); task_int = task_int.to(device)
        pred, logits = model.forward_with_aux(obs, lang_emb, image_emb)
        bc_loss = weighted_action_mse(pred, actions, w, channel_w)
        aux_loss = F.cross_entropy(logits, task_int)
        loss = bc_loss + aux_weight * aux_loss
        optim.zero_grad(set_to_none=True)
        loss.backward()
        if clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optim.step()
        n = obs.shape[0]
        tot += float(loss.item()) * n
        bc += float(bc_loss.item()) * n
        aux += float(aux_loss.item()) * n
        count += n
    return {"total_loss": tot / max(1, count), "bc_loss": bc / max(1, count), "aux_loss": aux / max(1, count)}


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
    num_tasks = len(task_id_strings)
    print(f"[m6] task_id mapping: {task_id_strings}")

    train_raw = VLADataset(
        dataset_dir, train_ids, ep_to_int,
        cfg.early_end, cfg.mid_end, cfg.early_weight, cfg.mid_weight, cfg.late_weight,
        cfg.normalize_phase_weights, cfg.use_dataset_sample_weight, cfg.normalize_total_weights,
    )
    obs_mean, obs_std, action_mean, action_std = compute_normalization(train_raw)

    train_ds = VLADataset(
        dataset_dir, train_ids, ep_to_int,
        cfg.early_end, cfg.mid_end, cfg.early_weight, cfg.mid_weight, cfg.late_weight,
        cfg.normalize_phase_weights, cfg.use_dataset_sample_weight, cfg.normalize_total_weights,
        obs_mean=obs_mean, obs_std=obs_std, action_mean=action_mean, action_std=action_std,
    )
    val_ds = VLADataset(
        dataset_dir, val_ids, ep_to_int,
        cfg.early_end, cfg.mid_end, cfg.early_weight, cfg.mid_weight, cfg.late_weight,
        cfg.normalize_phase_weights, cfg.use_dataset_sample_weight, cfg.normalize_total_weights,
        obs_mean=obs_mean, obs_std=obs_std, action_mean=action_mean, action_std=action_std,
    )

    obs_dim = int(train_ds.obs.shape[1])
    lang_emb_dim = int(train_ds.lang_emb.shape[1])
    image_emb_dim = int(train_ds.image_emb.shape[1])
    action_dim = int(train_ds.actions.shape[1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VLAPolicyAux(
        obs_dim=obs_dim,
        lang_emb_dim=lang_emb_dim,
        lang_proj_dim=cfg.lang_proj_dim,
        image_emb_dim=image_emb_dim,
        image_proj_dim=cfg.image_proj_dim,
        action_dim=action_dim,
        num_tasks=num_tasks,
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    action_w = np.ones((action_dim,), dtype=np.float32)
    action_w[-1] = float(cfg.gripper_weight)
    channel_w = torch.from_numpy(action_w).to(device)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, drop_last=False)

    np.savez(run_dir / "normalization_stats.npz",
             obs_mean=obs_mean, obs_std=obs_std, action_mean=action_mean, action_std=action_std)

    with (run_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump({
            "milestone": "M6B",
            "description": "Multi-task VLA: state + lang + image with aux task_id classification.",
            "config": asdict(cfg),
            "obs_dim": obs_dim,
            "lang_emb_dim": lang_emb_dim,
            "lang_proj_dim": cfg.lang_proj_dim,
            "image_emb_dim": image_emb_dim,
            "image_proj_dim": cfg.image_proj_dim,
            "action_dim": action_dim,
            "num_tasks": num_tasks,
            "task_id_strings": task_id_strings,
            "aux_weight": cfg.aux_weight,
            "num_train_episodes": len(train_ids),
            "num_val_episodes": len(val_ids),
            "num_train_transitions": len(train_ds),
            "num_val_transitions": len(val_ds),
        }, f, indent=2)

    curve = run_dir / "training_curve.csv"
    best = float("inf"); best_epoch = -1; best_task_acc = 0.0

    with curve.open("w", encoding="utf-8", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=[
            "epoch","train_total","train_bc","train_aux","val_mse_norm","val_mse_raw","val_mae_raw","val_task_acc",
        ])
        writer.writeheader()

        for epoch in range(1, cfg.num_epochs + 1):
            tm = train_one_epoch(model, train_loader, optim, device, cfg.grad_clip_norm, channel_w, cfg.aux_weight)
            vm = evaluate(model, val_loader, device, action_mean, action_std)
            writer.writerow({
                "epoch": epoch,
                "train_total": tm["total_loss"],
                "train_bc": tm["bc_loss"],
                "train_aux": tm["aux_loss"],
                "val_mse_norm": vm["mse_norm"],
                "val_mse_raw": vm["mse_raw"],
                "val_mae_raw": vm["mae_raw"],
                "val_task_acc": vm["task_acc"],
            })
            f_csv.flush()

            if vm["mse_norm"] < best:
                best = vm["mse_norm"]
                best_epoch = epoch
                best_task_acc = vm["task_acc"]
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
                    "hidden_dims": cfg.hidden_dims,
                    "dropout": cfg.dropout,
                    "epoch": epoch,
                    "val_metrics": vm,
                    "training_type": "m6_vla_aux",
                }, run_dir / "best_model.pt")

            if epoch == 1 or epoch % cfg.print_every_epochs == 0 or epoch == cfg.num_epochs:
                print(f"[epoch {epoch:04d}] train_total={tm['total_loss']:.6f} bc={tm['bc_loss']:.6f} aux={tm['aux_loss']:.6f} val_mse_norm={vm['mse_norm']:.6f} task_acc={vm['task_acc']:.3f}")

    final = {
        "best_epoch": best_epoch,
        "best_val_mse_norm": best,
        "best_val_task_acc": best_task_acc,
        "curve_csv": str(curve),
        "best_model": str(run_dir / "best_model.pt"),
        "normalization_stats": str(run_dir / "normalization_stats.npz"),
        "lang_emb_dim": lang_emb_dim,
        "lang_proj_dim": cfg.lang_proj_dim,
        "image_emb_dim": image_emb_dim,
        "image_proj_dim": cfg.image_proj_dim,
        "num_tasks": num_tasks,
        "task_id_strings": task_id_strings,
        "aux_weight": cfg.aux_weight,
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

    print("[done] M6B VLA training complete")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
