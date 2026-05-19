from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class NormalizationStats:
    obs_mean: list[float]
    obs_std: list[float]
    action_mean: list[float]
    action_std: list[float]


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
    hidden_dims: list[int]
    dropout: float
    print_every_epochs: int


class TransitionDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        episode_ids: list[int],
        obs_mean: np.ndarray | None = None,
        obs_std: np.ndarray | None = None,
        action_mean: np.ndarray | None = None,
        action_std: np.ndarray | None = None,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.episode_ids = episode_ids

        obs_list: list[np.ndarray] = []
        action_list: list[np.ndarray] = []

        for episode_id in episode_ids:
            ep_path = dataset_dir / "episodes" / f"ep_{episode_id:06d}.npz"
            if not ep_path.exists():
                raise FileNotFoundError(ep_path)

            data = np.load(ep_path)
            obs = data["obs"].astype(np.float32)
            actions = data["actions"].astype(np.float32)

            if obs.shape[0] != actions.shape[0]:
                raise ValueError(
                    f"Length mismatch for {ep_path}: obs={obs.shape}, actions={actions.shape}"
                )

            obs_list.append(obs)
            action_list.append(actions)

        if not obs_list:
            raise RuntimeError("No episode data loaded.")

        self.obs_raw = np.concatenate(obs_list, axis=0).astype(np.float32)
        self.actions_raw = np.concatenate(action_list, axis=0).astype(np.float32)

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

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.actions[idx]),
        )


class BCPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        dropout: float,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = obs_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, action_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_config(cfg: dict[str, Any]) -> TrainConfig:
    training = cfg["training"]
    model = cfg["model"]
    logging_cfg = cfg["logging"]

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
    )


def load_splits(dataset_dir: Path) -> tuple[list[int], list[int]]:
    split_path = dataset_dir / "splits.json"
    if not split_path.exists():
        raise FileNotFoundError(split_path)

    with split_path.open("r", encoding="utf-8") as f:
        splits = json.load(f)

    return list(splits["train"]), list(splits["val"])


def compute_normalization(dataset: TransitionDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> dict[str, float]:
    model.eval()

    mse_norm_sum = 0.0
    mse_raw_sum = 0.0
    mae_raw_sum = 0.0
    count = 0

    action_mean_t = torch.from_numpy(action_mean).to(device)
    action_std_t = torch.from_numpy(action_std).to(device)

    for obs, actions_norm in loader:
        obs = obs.to(device)
        actions_norm = actions_norm.to(device)

        pred_norm = model(obs)

        mse_norm = F.mse_loss(pred_norm, actions_norm, reduction="sum")

        pred_raw = pred_norm * action_std_t + action_mean_t
        actions_raw = actions_norm * action_std_t + action_mean_t

        mse_raw = F.mse_loss(pred_raw, actions_raw, reduction="sum")
        mae_raw = F.l1_loss(pred_raw, actions_raw, reduction="sum")

        batch_size = obs.shape[0]
        action_dim = actions_norm.shape[1]
        n = batch_size * action_dim

        mse_norm_sum += float(mse_norm.item())
        mse_raw_sum += float(mse_raw.item())
        mae_raw_sum += float(mae_raw.item())
        count += n

    return {
        "mse_norm": mse_norm_sum / max(1, count),
        "mse_raw": mse_raw_sum / max(1, count),
        "mae_raw": mae_raw_sum / max(1, count),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
) -> float:
    model.train()

    total_loss = 0.0
    total_count = 0

    for obs, actions in loader:
        obs = obs.to(device)
        actions = actions.to(device)

        pred = model(obs)
        loss = F.mse_loss(pred, actions)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        total_loss += float(loss.item()) * obs.shape[0]
        total_count += obs.shape[0]

    return total_loss / max(1, total_count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/m3_bc_state.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    raw_cfg = load_yaml(cfg_path)
    cfg = parse_config(raw_cfg)

    set_seed(cfg.seed)

    dataset_dir = Path(cfg.dataset_dir)
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ids, val_ids = load_splits(dataset_dir)

    train_raw = TransitionDataset(dataset_dir=dataset_dir, episode_ids=train_ids)
    obs_mean, obs_std, action_mean, action_std = compute_normalization(train_raw)

    train_dataset = TransitionDataset(
        dataset_dir=dataset_dir,
        episode_ids=train_ids,
        obs_mean=obs_mean,
        obs_std=obs_std,
        action_mean=action_mean,
        action_std=action_std,
    )

    val_dataset = TransitionDataset(
        dataset_dir=dataset_dir,
        episode_ids=val_ids,
        obs_mean=obs_mean,
        obs_std=obs_std,
        action_mean=action_mean,
        action_std=action_std,
    )

    obs_dim = int(train_dataset.obs.shape[1])
    action_dim = int(train_dataset.actions.shape[1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BCPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    np.savez(
        run_dir / "normalization_stats.npz",
        obs_mean=obs_mean,
        obs_std=obs_std,
        action_mean=action_mean,
        action_std=action_std,
    )

    norm_stats = NormalizationStats(
        obs_mean=obs_mean.tolist(),
        obs_std=obs_std.tolist(),
        action_mean=action_mean.tolist(),
        action_std=action_std.tolist(),
    )

    metadata = {
        "milestone": "M3.0",
        "description": "State-only behavior cloning baseline for PickCube expert trajectories.",
        "config_path": str(cfg_path),
        "config": asdict(cfg),
        "dataset_dir": str(dataset_dir),
        "run_dir": str(run_dir),
        "device": str(device),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "num_train_episodes": len(train_ids),
        "num_val_episodes": len(val_ids),
        "num_train_transitions": len(train_dataset),
        "num_val_transitions": len(val_dataset),
        "normalization_stats": asdict(norm_stats),
    }

    with (run_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    curve_path = run_dir / "training_curve.csv"
    with curve_path.open("w", encoding="utf-8", newline="") as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=[
                "epoch",
                "train_loss_norm",
                "val_mse_norm",
                "val_mse_raw",
                "val_mae_raw",
            ],
        )
        writer.writeheader()

        best_val = float("inf")
        best_epoch = -1

        for epoch in range(1, cfg.num_epochs + 1):
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                grad_clip_norm=cfg.grad_clip_norm,
            )

            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                action_mean=action_mean,
                action_std=action_std,
            )

            row = {
                "epoch": epoch,
                "train_loss_norm": train_loss,
                "val_mse_norm": val_metrics["mse_norm"],
                "val_mse_raw": val_metrics["mse_raw"],
                "val_mae_raw": val_metrics["mae_raw"],
            }
            writer.writerow(row)
            f_csv.flush()

            if val_metrics["mse_norm"] < best_val:
                best_val = val_metrics["mse_norm"]
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "obs_dim": obs_dim,
                        "action_dim": action_dim,
                        "hidden_dims": cfg.hidden_dims,
                        "dropout": cfg.dropout,
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                    },
                    run_dir / "best_model.pt",
                )

            if epoch == 1 or epoch % cfg.print_every_epochs == 0 or epoch == cfg.num_epochs:
                print(
                    f"[epoch {epoch:04d}] "
                    f"train_loss_norm={train_loss:.6f} "
                    f"val_mse_norm={val_metrics['mse_norm']:.6f} "
                    f"val_mse_raw={val_metrics['mse_raw']:.6f} "
                    f"val_mae_raw={val_metrics['mae_raw']:.6f}"
                )

    final_metrics = {
        "best_epoch": best_epoch,
        "best_val_mse_norm": best_val,
        "curve_csv": str(curve_path),
        "best_model": str(run_dir / "best_model.pt"),
        "normalization_stats": str(run_dir / "normalization_stats.npz"),
    }

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    print("[done] M3.0 state-only BC training complete")
    print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()