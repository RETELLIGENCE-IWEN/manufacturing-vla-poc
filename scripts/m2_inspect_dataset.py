from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default="outputs/m2_step_dataset")
    parser.add_argument("--episode-id", type=int, default=0)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    ep_path = dataset_dir / "episodes" / f"ep_{args.episode_id:06d}.npz"

    if not ep_path.exists():
        raise FileNotFoundError(ep_path)

    data = np.load(ep_path)

    print(f"[episode] {ep_path}")
    for key in data.files:
        arr = data[key]
        print(f"{key:12s} shape={arr.shape} dtype={arr.dtype}")

    summary_path = dataset_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        print("\n[summary]")
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()