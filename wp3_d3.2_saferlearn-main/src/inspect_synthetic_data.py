#!/usr/bin/env python3
"""Inspect synthetic_data structure and sample compatibility."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


SAMPLE_EXTENSIONS = {".pt", ".pth", ".npy", ".npz", ".png", ".jpg", ".jpeg", ".bmp"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def infer_sample_count(obj: Any) -> int:
    if isinstance(obj, torch.Tensor):
        return int(obj.shape[0]) if obj.ndim > 0 else 1
    if isinstance(obj, np.ndarray):
        return int(obj.shape[0]) if obj.ndim > 0 else 1
    if isinstance(obj, dict):
        for key in ("samples", "data", "images", "x"):
            value = obj.get(key)
            if isinstance(value, (torch.Tensor, np.ndarray)):
                return int(value.shape[0]) if value.ndim > 0 else 1
    return 1


def extract_data_and_labels(obj: Any, sample_file: Path) -> tuple[Any | None, np.ndarray | None]:
    data = None
    labels = None

    if isinstance(obj, torch.Tensor):
        data = obj
    elif isinstance(obj, np.ndarray):
        data = obj
    elif isinstance(obj, dict):
        for key in ("samples", "data", "images", "x"):
            if key in obj and isinstance(obj[key], (torch.Tensor, np.ndarray)):
                data = obj[key]
                break
        for key in ("labels", "y", "targets"):
            if key in obj and isinstance(obj[key], (torch.Tensor, np.ndarray, list, tuple)):
                labels = np.array(obj[key], dtype=np.int64).reshape(-1)
                break

    # Common repository pattern: labels stored as sibling labels.csv.
    if labels is None:
        candidate = sample_file.with_name("labels.csv")
        if candidate.exists():
            # Handle csv files that may include header and index column.
            csv_data = np.genfromtxt(candidate, delimiter=",", names=True, dtype=None, encoding="utf-8")
            if csv_data.dtype.names:
                preferred_cols = [name for name in ("label", "labels", "target", "y") if name in csv_data.dtype.names]
                if preferred_cols:
                    labels = np.array(csv_data[preferred_cols[0]], dtype=np.int64).reshape(-1)
                else:
                    # Fallback to last column when label column name is unknown.
                    labels = np.array(csv_data[csv_data.dtype.names[-1]], dtype=np.int64).reshape(-1)
            else:
                loaded = np.genfromtxt(candidate, delimiter=",", dtype=np.int64)
                labels = np.array(loaded, dtype=np.int64).reshape(-1)

    return data, labels


def to_numpy_shape(x: Any) -> tuple[int, ...]:
    if isinstance(x, torch.Tensor):
        return tuple(int(v) for v in x.shape)
    if isinstance(x, np.ndarray):
        return tuple(int(v) for v in x.shape)
    return tuple()


def iter_sample_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SAMPLE_EXTENSIONS:
            yield p


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    synthetic_root = repo_root / "synthetic_data"
    if not synthetic_root.exists():
        raise FileNotFoundError(f"synthetic_data folder not found: {synthetic_root}")

    subfolders = sorted([p for p in synthetic_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    all_files = sorted([p for p in synthetic_root.rglob("*") if p.is_file()], key=lambda p: p.name)
    sample_files = sorted(iter_sample_files(synthetic_root), key=lambda p: p.as_posix())

    print(f"SYNTHETIC_ROOT={synthetic_root.as_posix()}")
    print(f"SUBFOLDER_COUNT={len(subfolders)}")
    print("SUBFOLDERS=" + ",".join([p.name for p in subfolders]))
    print(f"TOTAL_FILES={len(all_files)}")
    print("FILE_NAMES=" + ",".join([p.name for p in all_files]))
    print(f"SAMPLE_FILE_COUNT={len(sample_files)}")
    if sample_files:
        print(f"EXAMPLE_SAMPLE_FILE={sample_files[0].as_posix()}")

    # Step 3: load one file safely and inspect.
    if not sample_files:
        print("No sample files found.")
        return

    first = sample_files[0]
    ext = first.suffix.lower()
    if ext in {".pt", ".pth"}:
        first_obj = torch.load(first, map_location="cpu")
    elif ext == ".npy":
        first_obj = np.load(first, allow_pickle=True)
    elif ext == ".npz":
        first_obj = dict(np.load(first, allow_pickle=True))
    elif ext in IMAGE_EXTENSIONS:
        first_obj = np.array([[0]], dtype=np.uint8)  # placeholder for type reporting
    else:
        first_obj = None

    first_data, first_labels = extract_data_and_labels(first_obj, first)
    print("")
    print("FIRST_FILE_INSPECTION")
    print(f"  file={first.as_posix()}")
    print(f"  object_type={type(first_obj).__name__}")
    if isinstance(first_obj, dict):
        print(f"  keys={list(first_obj.keys())}")
    print(f"  tensor_shape={to_numpy_shape(first_data)}")
    print(f"  dtype={getattr(first_data, 'dtype', None)}")
    print(f"  labels_exist={first_labels is not None}")
    print(f"  number_of_samples_in_file={infer_sample_count(first_data if first_data is not None else first_obj)}")

    # Full scan.
    total_samples = 0
    label_counter: Counter[int] = Counter()
    sample_shape_counter: Counter[tuple[int, ...]] = Counter()
    mismatches: list[str] = []
    compatible_count = 0
    inspected_sample_files = 0
    label_tensor_shape = None

    for sample_file in sample_files:
        ext = sample_file.suffix.lower()
        if ext in {".pt", ".pth"}:
            obj = torch.load(sample_file, map_location="cpu")
        elif ext == ".npy":
            obj = np.load(sample_file, allow_pickle=True)
        elif ext == ".npz":
            obj = dict(np.load(sample_file, allow_pickle=True))
        elif ext in IMAGE_EXTENSIONS:
            # Each image file is treated as one sample; shape info omitted.
            obj = None
        else:
            continue

        data, labels = extract_data_and_labels(obj, sample_file)
        if ext in IMAGE_EXTENSIONS:
            total_samples += 1
            continue

        if data is None:
            continue

        inspected_sample_files += 1
        data_shape = to_numpy_shape(data)
        sample_count = int(data_shape[0]) if len(data_shape) > 0 else 1
        total_samples += sample_count
        sample_shape_counter[data_shape[1:] if len(data_shape) > 1 else data_shape] += sample_count

        # CNN compatibility checks for per-sample shape.
        per_sample_shape = data_shape[1:] if len(data_shape) > 1 else data_shape
        if per_sample_shape == (1, 28, 28):
            compatible_count += sample_count
        elif per_sample_shape == (784,):
            compatible_count += sample_count
        else:
            mismatches.append(f"{sample_file.as_posix()} -> per_sample_shape={per_sample_shape}")

        if labels is not None:
            if label_tensor_shape is None:
                label_tensor_shape = labels.shape
            for value in labels.tolist():
                label_counter[int(value)] += 1

    dominant_sample_shape = sample_shape_counter.most_common(1)[0][0] if sample_shape_counter else tuple()
    unique_labels = sorted(label_counter.keys())
    all_compatible = compatible_count == total_samples and total_samples > 0

    print("")
    print("Synthetic dataset summary")
    print("-------------------------")
    print(f"Total files: {len(all_files)}")
    print(f"Total sample files: {len(sample_files)}")
    print(f"Inspected sample files: {inspected_sample_files}")
    print(f"Total samples: {total_samples}")
    print(f"Sample tensor shape: {dominant_sample_shape}")
    print(f"Label tensor shape: {label_tensor_shape}")
    print(f"Unique labels: {unique_labels}")
    print(f"Label counts: {dict(sorted(label_counter.items()))}")
    print(f"Compatible with UCStubModel: {'YES' if all_compatible else 'NO'}")
    if mismatches:
        print("Shape mismatches:")
        for msg in mismatches[:20]:
            print(f"  - {msg}")
        if len(mismatches) > 20:
            print(f"  - ... and {len(mismatches) - 20} more")


if __name__ == "__main__":
    main()
