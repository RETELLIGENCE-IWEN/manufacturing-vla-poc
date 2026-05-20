"""M5.1: language-conditioned BC with an auxiliary task_id classification head.

Same as scripts/m4_train_bc_lang.py but adds:
  - per-episode task_id (read from episodes.jsonl)
  - a classification head on top of lang_proj predicting task_id
  - joint loss = BC weighted-MSE + aux_weight * CrossEntropy(task_logits, task_id)

The motivation is the M5 finding that the policy ignores instruction:
forcing the lang_proj activations to discriminate the task should make the
gradient through lang_proj non-trivial, so the BC head must actually use it.
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
    aux_weight: float


def load_episode_task_ids(dataset_dir: Path) -> tuple[dict, list]:
    """Return (episode_id -> task_int, ordered list of task_id strings)."""
    jsonl_path = dataset_dir / "episodes.jsonl"
    with jsonl_path.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    task_ids_seen: list = []
    for rec in records:
        tid = rec.get("task_id")
        if tid is None:
            raise KeyError(f"episodes.jsonl entry for episode {rec.get('episode_id')} lacks 'task_id'")
        if tid not in task_ids_seen:
            task_ids_seen.append(tid)

    tid_to_int = {tid: i for i, tid in enumerate(task_ids_seen)}
    ep_to_int = {int(rec["episode_id"]): tid_to_int[rec["task_id"]] for rec in records}
    return ep_to_int, task_ids_seen


class PhaseWeightedLangAuxDataset(Dataset):
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
    ) -> None:
        obs_list = []
        lang_list = []
        action_list = []
        weight_list = []
        task_int_list = []

        for episode_id in episode_ids:
            ep_path = dataset_dir / "episodes" / f"ep_{episode_id:06d}.npz"
            if not ep_path.exists():
                raise FileNotFoundError(ep_path)

            data = np.load(ep_path)
            obs = data["obs"].astype(np.float32)
            actions = data["actions"].astype(np.float32)
            if "lang_emb" not in data.files:
                raise KeyError(f"{ep_path} missing 'lang_emb'.")
            lang_emb = data["lang_emb"].astype(np.float32)

            if use_dataset_sample_weight and "sample_weight" in data.files:
                weights = data["sample_weight"].astype(np.float32).reshape(-1)
            else:
                weights = np.ones((obs.shape[0],), dtype=np.float32)

            T = obs.shape[0]
            if actions.shape[0] != T or lang_emb.shape[0] != T:
                raise ValueError(f"length mismatch in {ep_path}")

            task_int = episode_to_task_int[int(episode_id)]
            task_int_arr = np.full((T,), task_int, dtype=np.int64)

            obs_list.append(obs)
            lang_list.append(lang_emb)
            action_list.append(actions)
            weight_list.append(weights)
            task_int_list.append(task_int_arr)

        self.obs_raw = np.concatenate(obs_list, axis=0).astype(np.float32)
        self.lang_emb = np.concatenate(lang_list, axis=0).astype(np.float32)
        self.actions_raw = np.concatenate(action_list, axis=0).astype(np.float32)
        self.dataset_weights_raw = np.concatenate(weight_list, axis=0).astype(np.float32)
        self.task_int = np.concatenate(task_int_list, axis=0).astype(np.int64)

        self.progress_raw = self.obs_raw[:, 57].astype(np.float32)

        self.phase_weights = self._build_phase_weights(
            progress=self.progress_raw,
            early_end=early_end,
            mid_end=mid_end,
            early_weight=early_weight,
            mid_weight=mid_weight,
            late_weight=late_weight,
            normalize=normalize_phase_weights,
        ).astype(np.float32)
        self.sample_weights = (self.phase_weights * self.dataset_weights_raw).astype(np.float32)

        if normalize_total_weights:
            mean_weight = float(np.mean(self.sample_weights))
            if mean_weight > 1e-8:
                self.sample_weights = (self.sample_weights / mean_weight).astype(np.float32)

        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.action_mean = action_mean
        self.action_std = action_std

        if self.obs_mean is not None:
            self.obs = ((self.obs_raw - self.obs_mean) / self.obs_std).astype(np.float32)
        else:
            self.obs = self.obs_raw

        if self.action_mean is not None:
            self.actions = ((self.actions_raw - self.action_mean) / self.action_std).astype(np.float32)
        else:
            self.actions = self.actions_raw

    @staticmethod
    def _build_phase_weights(progress, early_end, mid_end, early_weight, mid_weight, late_weight, normalize):
        if not (0.0 < early_end < mid_end < 1.0):
            raise ValueError(f"Expected 0 < early_end < mid_end < 1, got {early_end}, {mid_end}")
        weights = np.full_like(progress, fill_value=late_weight, dtype=np.float32)
        weights[progress < mid_end] = mid_weight
        weights[progress < early_end] = early_weight
        if normalize:
            mean_weight = float(np.mean(weights))
            if mean_weight > 1e-8:
                weights = weights / mean_weight
        return weights.astype(np.float32)

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.lang_emb[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.tensor(self.sample_weights[idx], dtype=torch.float32),
            torch.tensor(self.task_int[idx], dtype=torch.long),
        )


class LangBCPolicyAux(nn.Module):
    """LangBCPolicy + auxiliary task_id classification head on top of lang_proj."""

    def __init__(
        self,
        obs_dim: int,
        lang_emb_dim: int,
        lang_proj_dim: int,
        action_dim: int,
        num_tasks: int,
        hidden_dims: list,
        dropout: float,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.lang_emb_dim = lang_emb_dim
        self.lang_proj_dim = lang_proj_dim
        self.num_tasks = num_tasks

        self.lang_proj = nn.Linear(lang_emb_dim, lang_proj_dim)
        self.task_head = nn.Linear(lang_proj_dim, num_tasks)

        layers: list = []
        in_dim = obs_dim + lang_proj_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def lang_proj_activation(self, lang_emb: torch.Tensor) -> torch.Tensor:
        return F.relu(self.lang_proj(lang_emb))

    def forward(self, obs: torch.Tensor, lang_emb: torch.Tensor) -> torch.Tensor:
        lang_p = self.lang_proj_activation(lang_emb)
        x = torch.cat([obs, lang_p], dim=-1)
        return self.net(x)

    def forward_with_aux(self, obs: torch.Tensor, lang_emb: torch.Tensor):
        lang_p = self.lang_proj_activation(lang_emb)
        x = torch.cat([obs, lang_p], dim=-1)
        action = self.net(x)
        task_logits = self.task_head(lang_p)
        return action, task_logits


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_config(cfg: dict) -> TrainConfig:
    training = cfg["training"]
    model = cfg["model"]
    logging_cfg = cfg["logging"]
    loss_cfg = cfg["loss"]
    lang_cfg = cfg.get("lang", {})
    aux_cfg = cfg.get("aux", {})

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
        print_every_epochs=int(logging_cfg["print_every_epochs"]),
        early_end=float(loss_cfg["early_end"]),
        mid_end=float(loss_cfg["mid_end"]),
        early_weight=float(loss_cfg["early_weight"]),
        mid_weight=float(loss_cfg["mid_weight"]),
        late_weight=float(loss_cfg["late_weight"]),
        normalize_phase_weights=bool(loss_cfg.get("normalize_phase_weights", True)),
        use_dataset_sample_weight=bool(loss_cfg.get("use_dataset_sample_weight", True)),
        normalize_total_weights=bool(loss_cfg.get("normalize_total_weights", False)),
        gripper_weight=float(loss_cfg.get("gripper_weight", 1.0)),
        lang_proj_dim=int(lang_cfg.get("lang_proj_dim", 64)),
        aux_weight=float(aux_cfg.get("aux_weight", 1.0)),
    )


def load_splits(dataset_dir: Path):
    with (dataset_dir / "splits.json").open("r", encoding="utf-8") as f:
        splits = json.load(f)
    return list(splits["train"]), list(splits["val"])


def compute_normalization(dataset: PhaseWeightedLangAuxDataset):
    eps = 1e-6
    obs_mean = dataset.obs_raw.mean(axis=0)
    obs_std = dataset.obs_raw.std(axis=0) + eps
    action_mean = dataset.actions_raw.mean(axis=0)
    action_std = dataset.actions_raw.std(axis=0) + eps
    return (
        obs_mean.astype(np.float32),
        obs_std.astype(np.float32),
        action_mean.astype(np.float32),
        action_std.astype(np.float32),
    )


@torch.no_grad()
def evaluate(model, loader, device, action_mean, action_std):
    model.eval()
    mse_norm_sum = 0.0
    mse_raw_sum = 0.0
    mae_raw_sum = 0.0
    count = 0
    correct = 0
    total_classified = 0
    action_mean_t = torch.from_numpy(action_mean).to(device)
    action_std_t = torch.from_numpy(action_std).to(device)

    for obs, lang_emb, actions_norm, _w, task_int in loader:
        obs = obs.to(device)
        lang_emb = lang_emb.to(device)
        actions_norm = actions_norm.to(device)
        task_int = task_int.to(device)

        pred_norm, task_logits = model.forward_with_aux(obs, lang_emb)
        mse_norm = F.mse_loss(pred_norm, actions_norm, reduction="sum")
        pred_raw = pred_norm * action_std_t + action_mean_t
        actions_raw = actions_norm * action_std_t + action_mean_t
        mse_raw = F.mse_loss(pred_raw, actions_raw, reduction="sum")
        mae_raw = F.l1_loss(pred_raw, actions_raw, reduction="sum")

        preds = task_logits.argmax(dim=-1)
        correct += int((preds == task_int).sum().item())
        total_classified += int(task_int.shape[0])

        n = obs.shape[0] * actions_norm.shape[1]
        mse_norm_sum += float(mse_norm.item())
        mse_raw_sum += float(mse_raw.item())
        mae_raw_sum += float(mae_raw.item())
        count += n

    return {
        "mse_norm": mse_norm_sum / max(1, count),
        "mse_raw": mse_raw_sum / max(1, count),
        "mae_raw": mae_raw_sum / max(1, count),
        "task_acc": correct / max(1, total_classified),
    }


def weighted_action_mse(pred, target, sample_weights, action_channel_weights):
    sq_err = (pred - target) ** 2
    sq_err = sq_err * action_channel_weights[None, :]
    per_sample = sq_err.mean(dim=1)
    return (per_sample * sample_weights).mean()


def train_one_epoch(model, loader, optimizer, device, grad_clip_norm, action_channel_weights, aux_weight):
    model.train()
    total_loss_sum = 0.0
    total_bc_sum = 0.0
    total_aux_sum = 0.0
    total_count = 0

    for obs, lang_emb, actions, w, task_int in loader:
        obs = obs.to(device)
        lang_emb = lang_emb.to(device)
        actions = actions.to(device)
        w = w.to(device)
        task_int = task_int.to(device)

        pred, task_logits = model.forward_with_aux(obs, lang_emb)
        bc_loss = weighted_action_mse(pred, actions, w, action_channel_weights)
        aux_loss = F.cross_entropy(task_logits, task_int)
        loss = bc_loss + aux_weight * aux_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        n = obs.shape[0]
        total_loss_sum += float(loss.item()) * n
        total_bc_sum += float(bc_loss.item()) * n
        total_aux_sum += float(aux_loss.item()) * n
        total_count += n

    return {
        "total_loss": total_loss_sum / max(1, total_count),
        "bc_loss": total_bc_sum / max(1, total_count),
        "aux_loss": total_aux_sum / max(1, total_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = parse_config(load_yaml(cfg_path))
    set_seed(cfg.seed)

    dataset_dir = Path(cfg.dataset_dir)
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ids, val_ids = load_splits(dataset_dir)
    ep_to_int, task_id_strings = load_episode_task_ids(dataset_dir)
    num_tasks = len(task_id_strings)
    print(f"[m5.1] task_id mapping: {task_id_strings}")

    train_raw = PhaseWeightedLangAuxDataset(
        dataset_dir=dataset_dir,
        episode_ids=train_ids,
        episode_to_task_int=ep_to_int,
        early_end=cfg.early_end,
        mid_end=cfg.mid_end,
        early_weight=cfg.early_weight,
        mid_weight=cfg.mid_weight,
        late_weight=cfg.late_weight,
        normalize_phase_weights=cfg.normalize_phase_weights,
        use_dataset_sample_weight=cfg.use_dataset_sample_weight,
        normalize_total_weights=cfg.normalize_total_weights,
    )
    obs_mean, obs_std, action_mean, action_std = compute_normalization(train_raw)

    train_dataset = PhaseWeightedLangAuxDataset(
        dataset_dir=dataset_dir,
        episode_ids=train_ids,
        episode_to_task_int=ep_to_int,
        early_end=cfg.early_end,
        mid_end=cfg.mid_end,
        early_weight=cfg.early_weight,
        mid_weight=cfg.mid_weight,
        late_weight=cfg.late_weight,
        normalize_phase_weights=cfg.normalize_phase_weights,
        use_dataset_sample_weight=cfg.use_dataset_sample_weight,
        normalize_total_weights=cfg.normalize_total_weights,
        obs_mean=obs_mean,
        obs_std=obs_std,
        action_mean=action_mean,
        action_std=action_std,
    )
    val_dataset = PhaseWeightedLangAuxDataset(
        dataset_dir=dataset_dir,
        episode_ids=val_ids,
        episode_to_task_int=ep_to_int,
        early_end=cfg.early_end,
        mid_end=cfg.mid_end,
        early_weight=cfg.early_weight,
        mid_weight=cfg.mid_weight,
        late_weight=cfg.late_weight,
        normalize_phase_weights=cfg.normalize_phase_weights,
        use_dataset_sample_weight=cfg.use_dataset_sample_weight,
        normalize_total_weights=cfg.normalize_total_weights,
        obs_mean=obs_mean,
        obs_std=obs_std,
        action_mean=action_mean,
        action_std=action_std,
    )

    obs_dim = int(train_dataset.obs.shape[1])
    lang_emb_dim = int(train_dataset.lang_emb.shape[1])
    action_dim = int(train_dataset.actions.shape[1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LangBCPolicyAux(
        obs_dim=obs_dim,
        lang_emb_dim=lang_emb_dim,
        lang_proj_dim=cfg.lang_proj_dim,
        action_dim=action_dim,
        num_tasks=num_tasks,
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    action_weights = np.ones((action_dim,), dtype=np.float32)
    action_weights[-1] = float(cfg.gripper_weight)
    action_channel_weights = torch.from_numpy(action_weights).to(device)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0, drop_last=False)

    np.savez(
        run_dir / "normalization_stats.npz",
        obs_mean=obs_mean,
        obs_std=obs_std,
        action_mean=action_mean,
        action_std=action_std,
    )

    metadata = {
        "milestone": "M5.1A",
        "description": "Multi-task lang BC with auxiliary task_id classification head.",
        "config_path": str(cfg_path),
        "config": asdict(cfg),
        "dataset_dir": str(dataset_dir),
        "run_dir": str(run_dir),
        "device": str(device),
        "obs_dim": obs_dim,
        "lang_emb_dim": lang_emb_dim,
        "lang_proj_dim": cfg.lang_proj_dim,
        "action_dim": action_dim,
        "num_tasks": num_tasks,
        "task_id_strings": task_id_strings,
        "aux_weight": cfg.aux_weight,
        "num_train_episodes": len(train_ids),
        "num_val_episodes": len(val_ids),
        "num_train_transitions": len(train_dataset),
        "num_val_transitions": len(val_dataset),
        "action_channel_weights": action_weights.tolist(),
    }
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    curve_path = run_dir / "training_curve.csv"
    best_val = float("inf")
    best_epoch = -1
    best_task_acc = 0.0

    with curve_path.open("w", encoding="utf-8", newline="") as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=[
                "epoch", "train_total", "train_bc", "train_aux",
                "val_mse_norm", "val_mse_raw", "val_mae_raw", "val_task_acc",
            ],
        )
        writer.writeheader()

        for epoch in range(1, cfg.num_epochs + 1):
            train_metrics = train_one_epoch(
                model=model, loader=train_loader, optimizer=optimizer, device=device,
                grad_clip_norm=cfg.grad_clip_norm, action_channel_weights=action_channel_weights,
                aux_weight=cfg.aux_weight,
            )
            val_metrics = evaluate(model=model, loader=val_loader, device=device, action_mean=action_mean, action_std=action_std)

            writer.writerow({
                "epoch": epoch,
                "train_total": train_metrics["total_loss"],
                "train_bc": train_metrics["bc_loss"],
                "train_aux": train_metrics["aux_loss"],
                "val_mse_norm": val_metrics["mse_norm"],
                "val_mse_raw": val_metrics["mse_raw"],
                "val_mae_raw": val_metrics["mae_raw"],
                "val_task_acc": val_metrics["task_acc"],
            })
            f_csv.flush()

            if val_metrics["mse_norm"] < best_val:
                best_val = val_metrics["mse_norm"]
                best_epoch = epoch
                best_task_acc = val_metrics["task_acc"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "obs_dim": obs_dim,
                        "lang_emb_dim": lang_emb_dim,
                        "lang_proj_dim": cfg.lang_proj_dim,
                        "action_dim": action_dim,
                        "num_tasks": num_tasks,
                        "task_id_strings": task_id_strings,
                        "hidden_dims": cfg.hidden_dims,
                        "dropout": cfg.dropout,
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                        "training_type": "lang_phase_weighted_bc_aux",
                    },
                    run_dir / "best_model.pt",
                )

            if epoch == 1 or epoch % cfg.print_every_epochs == 0 or epoch == cfg.num_epochs:
                print(
                    f"[epoch {epoch:04d}] "
                    f"train_total={train_metrics['total_loss']:.6f} "
                    f"train_bc={train_metrics['bc_loss']:.6f} "
                    f"train_aux={train_metrics['aux_loss']:.6f} "
                    f"val_mse_norm={val_metrics['mse_norm']:.6f} "
                    f"val_task_acc={val_metrics['task_acc']:.4f}"
                )

    final = {
        "best_epoch": best_epoch,
        "best_val_mse_norm": best_val,
        "best_val_task_acc": best_task_acc,
        "curve_csv": str(curve_path),
        "best_model": str(run_dir / "best_model.pt"),
        "normalization_stats": str(run_dir / "normalization_stats.npz"),
        "lang_emb_dim": lang_emb_dim,
        "lang_proj_dim": cfg.lang_proj_dim,
        "num_tasks": num_tasks,
        "task_id_strings": task_id_strings,
        "aux_weight": cfg.aux_weight,
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

    print("[done] M5.1A lang-conditioned BC + aux classification complete")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
