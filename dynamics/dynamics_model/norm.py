#!/usr/bin/env python3
"""Compute per-dataset action normalization statistics from LeRobot-style parquet episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


SPLIT_ACTION_COLUMNS = [
    "action.left_arm",
    "action.left_gripper",
    "action.right_arm",
    "action.right_gripper",
]


class OnlineStatistics:
    """Online min/max/mean/variance calculator for action vectors."""

    def __init__(self, dim: int):
        self.dim = dim
        self.count = 0
        self.min = np.full(dim, np.inf, dtype=np.float64)
        self.max = np.full(dim, -np.inf, dtype=np.float64)
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros(dim, dtype=np.float64)

    def update(self, batch: np.ndarray):
        if batch.ndim != 2 or batch.shape[1] != self.dim:
            raise ValueError(f"Expected batch shape (N, {self.dim}), got {batch.shape}")

        self.min = np.minimum(self.min, np.min(batch, axis=0))
        self.max = np.maximum(self.max, np.max(batch, axis=0))

        for row in batch:
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            delta2 = row - self.mean
            self.M2 += delta * delta2

    def get_statistics(self) -> dict | None:
        if self.count == 0:
            return None

        var = self.M2 / self.count if self.count > 1 else np.zeros(self.dim, dtype=np.float64)
        std = np.sqrt(var)
        return {
            "count": self.count,
            "min": self.min.astype(np.float32),
            "max": self.max.astype(np.float32),
            "mean": self.mean.astype(np.float32),
            "std": std.astype(np.float32),
            "var": var.astype(np.float32),
        }


def format_vector(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:.6f}" for x in values) + "]"


def print_statistics(dataset_name: str, action_key: str, stats: dict):
    print("\n" + "=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Action key: {action_key}")
    print(f"Dimensions: {len(stats['min'])}")
    print(f"Count: {stats['count']}")
    print("=" * 80)
    print(f"min = {format_vector(stats['min'])}")
    print(f"max = {format_vector(stats['max'])}")
    print(f"mean = {format_vector(stats['mean'])}")
    print(f"std = {format_vector(stats['std'])}")
    print("=" * 80)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_dataset_dirs(dataset_base: Path, dataset_names: list[str] | None) -> list[Path]:
    if dataset_names:
        dataset_dirs = [dataset_base / name for name in dataset_names]
    else:
        dataset_dirs = sorted(path for path in dataset_base.iterdir() if path.is_dir())

    missing = [str(path) for path in dataset_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Dataset(s) not found: {', '.join(missing)}")

    return dataset_dirs


def infer_action_key(meta_info: dict) -> str | None:
    features = meta_info.get("features", {})

    for candidate in ("action", "actions"):
        spec = features.get(candidate)
        if spec and spec.get("dtype", "").startswith("float"):
            return candidate

    if all(column in features for column in SPLIT_ACTION_COLUMNS):
        return "split"

    return None


def infer_action_dim(meta_info: dict, action_key: str) -> int | None:
    features = meta_info.get("features", {})

    if action_key == "split":
        dim = 0
        for column in SPLIT_ACTION_COLUMNS:
            shape = features.get(column, {}).get("shape", [])
            dim += int(shape[0]) if shape else 1
        return dim or None

    shape = features.get(action_key, {}).get("shape", [])
    if shape:
        return int(shape[0])
    return None


def ensure_array(value) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        array = array.reshape(1)
    return array


def extract_actions(df, action_key: str) -> np.ndarray | None:
    if action_key in ("action", "actions"):
        if action_key not in df.columns:
            return None
        values = [ensure_array(value) for value in df[action_key].values]
        if not values:
            return None
        return np.stack(values, axis=0)

    if action_key == "split":
        if not all(column in df.columns for column in SPLIT_ACTION_COLUMNS):
            return None
        rows = [
            np.concatenate([ensure_array(df.at[idx, column]) for column in SPLIT_ACTION_COLUMNS], axis=0)
            for idx in range(len(df))
        ]
        return np.stack(rows, axis=0) if rows else None

    return None


try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def get_parquet_files(dataset_dir: Path) -> list[Path]:
    return sorted(dataset_dir.glob("data/chunk-*/episode_*.parquet"))


def compute_dataset_stats(dataset_dir: Path, batch_size: int) -> tuple[str, dict] | None:
    dataset_name = dataset_dir.name
    meta_path = dataset_dir / "meta" / "info.json"
    if not meta_path.exists():
        print(f"Skipping {dataset_name}: missing {meta_path}", file=sys.stderr)
        return None

    meta_info = load_json(meta_path)
    action_key = infer_action_key(meta_info)
    if action_key is None:
        print(f"Skipping {dataset_name}: could not infer action key from {meta_path}", file=sys.stderr)
        return None

    action_dim = infer_action_dim(meta_info, action_key)
    if action_dim is None:
        print(f"Skipping {dataset_name}: could not infer action dimension from {meta_path}", file=sys.stderr)
        return None

    parquet_files = get_parquet_files(dataset_dir)
    if not parquet_files:
        print(f"Skipping {dataset_name}: no parquet episodes found", file=sys.stderr)
        return None

    stats = OnlineStatistics(dim=action_dim)
    for parquet_file in tqdm(parquet_files, desc=f"  {dataset_name}", leave=False):
        try:
            df = pd.read_parquet(parquet_file)
            actions = extract_actions(df, action_key)
            if actions is None:
                continue
            if actions.shape[1] != action_dim:
                raise ValueError(
                    f"Action dim mismatch in {parquet_file}: expected {action_dim}, got {actions.shape[1]}"
                )

            for start in range(0, len(actions), batch_size):
                stats.update(actions[start:start + batch_size])
        except Exception as exc:
            print(f"Error reading {parquet_file}: {exc}", file=sys.stderr)

    result = stats.get_statistics()
    if result is None:
        print(f"Skipping {dataset_name}: no usable action data found", file=sys.stderr)
        return None

    return action_key, result


def save_to_config(config_path: Path, dataset_name: str, action_key: str, stats: dict):
    config_path.parent.mkdir(parents=True, exist_ok=True)

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {}

    config[dataset_name] = {
        "action_key": action_key,
        "dim": len(stats["min"]),
        "count": int(stats["count"]),
        "min": stats["min"].tolist(),
        "max": stats["max"].tolist(),
        "mean": stats["mean"].tolist(),
        "std": stats["std"].tolist(),
    }

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved normalization statistics to {config_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute action statistics from parquet files")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset",
        help="Base directory containing datasets. Default: ./dataset",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        help="Specific dataset names to process. Default: process all datasets under --dataset-dir.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for statistics updates.",
    )
    parser.add_argument(
        "--save-config",
        type=str,
        default=None,
        help="Path to save per-dataset normalization config as JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_base = Path(args.dataset_dir)

    if pd is None:
        print("Error: pandas is not installed in the current Python environment.", file=sys.stderr)
        sys.exit(1)

    if not dataset_base.exists():
        print(f"Error: dataset directory not found: {dataset_base}", file=sys.stderr)
        sys.exit(1)

    try:
        dataset_dirs = discover_dataset_dirs(dataset_base, args.datasets)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if not dataset_dirs:
        print("No datasets found")
        sys.exit(0)

    print(f"Found {len(dataset_dirs)} dataset(s)")

    processed = 0
    for dataset_dir in tqdm(dataset_dirs, desc="Processing datasets"):
        result = compute_dataset_stats(dataset_dir, args.batch_size)
        if result is None:
            continue

        action_key, stats = result
        print_statistics(dataset_dir.name, action_key, stats)
        if args.save_config:
            save_to_config(Path(args.save_config), dataset_dir.name, action_key, stats)
        processed += 1

    if processed == 0:
        print("No action data found in any dataset")
        sys.exit(1)


if __name__ == "__main__":
    main()
