from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def print_h5_tree(name: str, obj, indent: int = 0) -> None:
    prefix = "  " * indent

    if isinstance(obj, h5py.Dataset):
        print(f"{prefix}- {name}: shape={obj.shape}, dtype={obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"{prefix}+ {name}/")
        for key in obj.keys():
            print_h5_tree(key, obj[key], indent + 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=str, required=True)
    args = parser.parse_args()

    h5_path = Path(args.h5)
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    print(f"[h5] {h5_path}")

    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            print_h5_tree(key, f[key], indent=0)


if __name__ == "__main__":
    main()