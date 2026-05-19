from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


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


def load_splits(dataset_dir: Path) -> dict[str, list[int]]:
    split_path = dataset_dir / "splits.json"
    if not split_path.exists():
        raise FileNotFoundError(split_path)

    with split_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_episode(
    model: BCPolicy,
    episode_path: Path,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    device: torch.device,
    out_csv: Path,
) -> dict[str, Any]:
    data = np.load(episode_path)

    obs_raw = data["obs"].astype(np.float32)
    expert_actions = data["actions"].astype(np.float32)

    obs_norm = ((obs_raw - obs_mean) / obs_std).astype(np.float32)

    model.eval()
    with torch.no_grad():
        obs_t = torch.from_numpy(obs_norm).to(device)
        pred_norm = model(obs_t).cpu().numpy().astype(np.float32)

    pred_actions = pred_norm * action_std + action_mean

    errors = pred_actions - expert_actions
    abs_errors = np.abs(errors)
    sq_errors = errors**2

    per_step_mae = abs_errors.mean(axis=1)
    per_step_mse = sq_errors.mean(axis=1)
    per_step_l2 = np.linalg.norm(errors, axis=1)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    action_dim = expert_actions.shape[1]
    fieldnames = (
        ["step", "mae", "mse", "l2"]
        + [f"expert_a{i}" for i in range(action_dim)]
        + [f"pred_a{i}" for i in range(action_dim)]
        + [f"err_a{i}" for i in range(action_dim)]
    )

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for t in range(expert_actions.shape[0]):
            row: dict[str, Any] = {
                "step": t,
                "mae": float(per_step_mae[t]),
                "mse": float(per_step_mse[t]),
                "l2": float(per_step_l2[t]),
            }

            for i in range(action_dim):
                row[f"expert_a{i}"] = float(expert_actions[t, i])
                row[f"pred_a{i}"] = float(pred_actions[t, i])
                row[f"err_a{i}"] = float(errors[t, i])

            writer.writerow(row)

    return {
        "episode_path": str(episode_path),
        "csv_path": str(out_csv),
        "num_steps": int(expert_actions.shape[0]),
        "action_dim": int(action_dim),
        "mae_raw": float(abs_errors.mean()),
        "mse_raw": float(sq_errors.mean()),
        "rmse_raw": float(np.sqrt(sq_errors.mean())),
        "mean_step_l2": float(per_step_l2.mean()),
        "max_step_l2": float(per_step_l2.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default="outputs/m2_expert_dataset_100")
    parser.add_argument("--run-dir", type=str, default="runs/m3_bc_state")
    parser.add_argument("--model", type=str, default="runs/m3_bc_state/best_model.pt")
    parser.add_argument("--normalization", type=str, default="runs/m3_bc_state/normalization_stats.npz")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default="runs/m3_bc_state/openloop_eval")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    run_dir = Path(args.run_dir)
    model_path = Path(args.model)
    norm_path = Path(args.normalization)
    out_dir = Path(args.out_dir)

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not norm_path.exists():
        raise FileNotFoundError(norm_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)

    model = BCPolicy(
        obs_dim=int(checkpoint["obs_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        hidden_dims=[int(x) for x in checkpoint["hidden_dims"]],
        dropout=float(checkpoint["dropout"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    norm = np.load(norm_path)
    obs_mean = norm["obs_mean"].astype(np.float32)
    obs_std = norm["obs_std"].astype(np.float32)
    action_mean = norm["action_mean"].astype(np.float32)
    action_std = norm["action_std"].astype(np.float32)

    splits = load_splits(dataset_dir)
    episode_ids = list(splits[args.split])[: args.num_episodes]

    if not episode_ids:
        raise RuntimeError(f"No episodes found in split: {args.split}")

    episode_summaries: list[dict[str, Any]] = []

    for episode_id in episode_ids:
        ep_path = dataset_dir / "episodes" / f"ep_{episode_id:06d}.npz"
        out_csv = out_dir / f"ep_{episode_id:06d}_action_compare.csv"

        summary = evaluate_episode(
            model=model,
            episode_path=ep_path,
            obs_mean=obs_mean,
            obs_std=obs_std,
            action_mean=action_mean,
            action_std=action_std,
            device=device,
            out_csv=out_csv,
        )

        summary["episode_id"] = int(episode_id)
        episode_summaries.append(summary)

        print(
            f"[episode {episode_id:06d}] "
            f"mae_raw={summary['mae_raw']:.6f} "
            f"rmse_raw={summary['rmse_raw']:.6f} "
            f"mean_step_l2={summary['mean_step_l2']:.6f}"
        )

    aggregate = {
        "milestone": "M3.1",
        "description": "Open-loop BC action prediction evaluation.",
        "dataset_dir": str(dataset_dir),
        "run_dir": str(run_dir),
        "model": str(model_path),
        "normalization": str(norm_path),
        "split": args.split,
        "num_episodes": len(episode_summaries),
        "device": str(device),
        "mean_mae_raw": float(np.mean([x["mae_raw"] for x in episode_summaries])),
        "mean_mse_raw": float(np.mean([x["mse_raw"] for x in episode_summaries])),
        "mean_rmse_raw": float(np.mean([x["rmse_raw"] for x in episode_summaries])),
        "mean_step_l2": float(np.mean([x["mean_step_l2"] for x in episode_summaries])),
        "max_step_l2": float(np.max([x["max_step_l2"] for x in episode_summaries])),
        "episodes": episode_summaries,
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "openloop_summary.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    print("[done] M3.1 open-loop evaluation complete")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()