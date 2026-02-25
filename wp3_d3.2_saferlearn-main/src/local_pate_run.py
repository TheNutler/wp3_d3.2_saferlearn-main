#!/usr/bin/env python3
"""Single-process local PATE baseline on MNIST t10k.

This script intentionally avoids all distributed orchestration (MPC/Kafka/RPyC).
It loads pre-trained teacher checkpoints, aggregates teacher votes with Gaussian
noise, trains a student model on noisy labels, and evaluates on a held-out split.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


class UCStubModel(nn.Module):
    """Teacher/student CNN architecture from src/usecases/data_owner_example.py."""

    def __init__(self):
        super(UCStubModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def estimate_epsilon_gaussian(
    num_queries: int, sigma: float, delta: float = 1e-5
) -> float:
    """Basic epsilon estimate via zCDP->(eps,delta)-DP conversion.

    This gives a simple upper-bound style estimate for repeated Gaussian releases.
    """
    if sigma <= 0:
        return float("inf")
    rho = num_queries / (2.0 * sigma * sigma)
    return float(rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta)))


def estimate_epsilon_laplace_basic(num_queries: int, laplace_scale: float) -> float:
    """Basic Laplace composition estimate: epsilon ~= queries / scale."""
    if laplace_scale <= 0:
        return float("inf")
    return float(num_queries / laplace_scale)


def load_teacher_paths(teachers_root: Path, teacher_count: int) -> List[Path]:
    paths: List[Path] = []
    for teacher_id in range(teacher_count):
        ckpt = teachers_root / str(teacher_id) / "model.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing teacher checkpoint: {ckpt}")
        paths.append(ckpt)
    return paths


def load_synthetic_dataset(synthetic_root: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and pool synthetic query dataset from all teacher folders."""
    if not synthetic_root.exists():
        raise RuntimeError(
            f"Synthetic dataset folder not found: {synthetic_root.as_posix()}"
        )

    teacher_dirs = sorted(
        [p for p in synthetic_root.iterdir() if p.is_dir() and p.name.startswith("teacher_")]
    )
    if not teacher_dirs:
        raise RuntimeError(
            f"No teacher folders found under: {synthetic_root.as_posix()}"
        )
    print(f"[synthetic] teachers_found={len(teacher_dirs)}")

    all_images: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for teacher_dir in teacher_dirs:
        sample_path = teacher_dir / "synthetic_samples.pt"
        labels_path = teacher_dir / "labels.csv"
        if not sample_path.exists():
            raise RuntimeError(f"Missing synthetic samples file: {sample_path.as_posix()}")
        if not labels_path.exists():
            raise RuntimeError(f"Missing synthetic labels file: {labels_path.as_posix()}")

        images = torch.load(sample_path, map_location="cpu")
        if not isinstance(images, torch.Tensor):
            raise RuntimeError(
                f"Expected tensor in {sample_path.as_posix()}, got {type(images).__name__}"
            )
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
            raise RuntimeError(
                f"Synthetic sample shape mismatch in {sample_path.as_posix()}: "
                f"expected (N,1,28,28), got {tuple(images.shape)}"
            )

        csv_data = np.genfromtxt(
            labels_path, delimiter=",", names=True, dtype=None, encoding="utf-8"
        )
        if csv_data.dtype.names:
            preferred = [
                name
                for name in ("label", "labels", "target", "y")
                if name in csv_data.dtype.names
            ]
            if preferred:
                labels_np = np.array(csv_data[preferred[0]], dtype=np.int64).reshape(-1)
            else:
                labels_np = np.array(
                    csv_data[csv_data.dtype.names[-1]], dtype=np.int64
                ).reshape(-1)
        else:
            labels_np = np.genfromtxt(labels_path, delimiter=",", dtype=np.int64).reshape(-1)

        labels = torch.tensor(labels_np, dtype=torch.long)
        if images.shape[0] != labels.shape[0]:
            raise RuntimeError(
                f"Sample/label count mismatch in {teacher_dir.as_posix()}: "
                f"{images.shape[0]} samples vs {labels.shape[0]} labels"
            )

        all_images.append(images)
        all_labels.append(labels)

    pooled_images = torch.cat(all_images, dim=0)
    pooled_labels = torch.cat(all_labels, dim=0)
    label_counter = Counter(int(v) for v in pooled_labels.tolist())
    print(f"[synthetic] total_samples={pooled_images.shape[0]}")
    print(f"[synthetic] label_distribution={dict(sorted(label_counter.items()))}")
    return pooled_images, pooled_labels


def collect_teacher_votes(
    teacher_paths: List[Path],
    query_loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Return vote counts tensor with shape [num_queries, num_classes]."""
    num_queries = len(query_loader.dataset)
    vote_counts = torch.zeros((num_queries, num_classes), dtype=torch.int32)
    total_iterations = len(teacher_paths)
    start_time = time.time()

    for teacher_idx, teacher_path in enumerate(teacher_paths, start=1):
        model = UCStubModel().to(device)
        checkpoint = torch.load(teacher_path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=True)
        model.eval()

        print(
            f"[teacher] Loaded {teacher_idx}/{len(teacher_paths)}: "
            f"{teacher_path.as_posix()}"
        )

        offset = 0
        with torch.no_grad():
            for images, _ in query_loader:
                images = images.to(device)
                logits = model(images)
                preds = torch.argmax(logits, dim=1).cpu()
                batch_size = preds.size(0)
                row_indices = torch.arange(offset, offset + batch_size)
                vote_counts[row_indices, preds] += 1
                offset += batch_size

        if teacher_idx % 100 == 0 or teacher_idx == total_iterations:
            elapsed = time.time() - start_time
            progress = teacher_idx / total_iterations
            if progress > 0:
                eta = elapsed * (1 - progress) / progress
            else:
                eta = 0.0
            print(
                f"[PATE Progress] {teacher_idx}/{total_iterations} "
                f"({progress*100:.2f}%) | "
                f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s"
            )

    return vote_counts


def noisy_max_laplace_from_votes(
    vote_counts: torch.Tensor, lap_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Laplace noisy-max to per-query vote counts."""
    if lap_scale <= 0:
        raise RuntimeError(f"laplace-scale must be > 0 for Laplace mechanism, got {lap_scale}.")

    counts_np = vote_counts.cpu().numpy().astype(np.float32, copy=True)
    laplace_noise = np.random.laplace(
        loc=0.0, scale=float(lap_scale), size=counts_np.shape
    ).astype(np.float32)
    noisy_counts_np = counts_np + laplace_noise
    noisy_labels_np = np.argmax(noisy_counts_np, axis=1).astype(np.int64)
    return torch.from_numpy(noisy_counts_np), torch.from_numpy(noisy_labels_np)


def train_student(
    student_model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> None:
    student_model.train()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        num_samples = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = student_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += float(loss.item()) * batch_size
            num_samples += batch_size

        epoch_loss = running_loss / max(num_samples, 1)
        print(f"[student] epoch {epoch}/{epochs} loss={epoch_loss:.4f}")


def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = torch.argmax(model(images), dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())
    return 100.0 * correct / max(total, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local single-process PATE baseline on MNIST t10k."
    )
    parser.add_argument("--teacher-count", type=int, default=250)
    parser.add_argument(
        "--query-source", choices=["real", "synthetic"], default="real"
    )
    parser.add_argument("--laplace-scale", type=float, default=20.0)
    parser.add_argument("--mechanism", choices=["gaussian", "laplace"], default="gaussian")
    parser.add_argument("--max-queries", type=int, default=1000)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--student-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delta", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    repo_root = Path(__file__).resolve().parents[1]
    teachers_root = repo_root / "trained_nets_gpu"
    synthetic_root = repo_root / "synthetic_data"
    mnist_root = repo_root / "public-dataset"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}")

    if args.teacher_count <= 0:
        raise RuntimeError(
            f"teacher-count must be a positive integer, got {args.teacher_count}."
        )
    teacher_paths = load_teacher_paths(teachers_root, args.teacher_count)
    if len(teacher_paths) != args.teacher_count:
        raise RuntimeError(
            f"Expected {args.teacher_count} teacher checkpoints, "
            f"found {len(teacher_paths)}."
        )
    print(f"[setup] teacher_count={len(teacher_paths)}")
    print(f"[setup] sigma={args.sigma}")
    print(f"[setup] laplace_scale={args.laplace_scale}")
    print(f"[setup] mechanism={args.mechanism}")
    print(f"[setup] query_source={args.query_source}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    mnist_test = datasets.MNIST(
        root=str(mnist_root), train=False, download=True, transform=transform
    )

    if args.query_source == "real":
        # t10k split: 90% query set for noisy labels, 10% held-out eval set.
        all_indices = torch.randperm(len(mnist_test))
        query_size = int(0.9 * len(mnist_test))
        query_indices = all_indices[:query_size]
        eval_indices = all_indices[query_size:]
        total_available_queries = len(query_indices)
        query_indices = query_indices[: min(args.max_queries, total_available_queries)]
        queries_used = len(query_indices)
        print(f"[setup] total_available_queries={total_available_queries}")
        print(f"[setup] queries_used={queries_used}")

        query_images = torch.stack(
            [mnist_test[int(i)][0] for i in query_indices], dim=0
        )
        eval_images = torch.stack([mnist_test[int(i)][0] for i in eval_indices], dim=0)
        eval_labels = torch.tensor(
            [int(mnist_test[int(i)][1]) for i in eval_indices], dtype=torch.long
        )
    else:
        synthetic_images, synthetic_labels = load_synthetic_dataset(synthetic_root)
        _ = synthetic_labels  # labels are not used for PATE querying
        total_available_queries = synthetic_images.shape[0]
        queries_used = min(args.max_queries, total_available_queries)
        print(f"[setup] total_available_queries={total_available_queries}")
        print(f"[setup] queries_used={queries_used}")
        query_images = synthetic_images[:queries_used]

        # In synthetic mode, evaluate on the full MNIST t10k benchmark set.
        eval_images = torch.stack([mnist_test[i][0] for i in range(len(mnist_test))], dim=0)
        eval_labels = torch.tensor(
            [int(mnist_test[i][1]) for i in range(len(mnist_test))], dtype=torch.long
        )
    print("Evaluation dataset: MNIST t10k (held-out benchmark set)")

    query_loader = DataLoader(
        TensorDataset(query_images, torch.zeros(query_images.size(0), dtype=torch.long)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    vote_counts = collect_teacher_votes(
        teacher_paths=teacher_paths,
        query_loader=query_loader,
        num_classes=10,
        device=device,
    )

    counts_float = vote_counts.float()
    if args.mechanism == "gaussian":
        if args.sigma <= 0:
            raise RuntimeError(
                f"sigma must be > 0 for Gaussian mechanism, got {args.sigma}."
            )
        noisy_counts = counts_float + torch.randn_like(counts_float) * args.sigma
        noisy_labels = torch.argmax(noisy_counts, dim=1).long()
    else:
        noisy_counts, noisy_labels = noisy_max_laplace_from_votes(
            vote_counts=vote_counts, lap_scale=args.laplace_scale
        )
    print(f"[aggregate] query_samples={len(noisy_labels)}")
    print(f"Answered queries: {len(noisy_labels)}")

    student_train_loader = DataLoader(
        TensorDataset(query_images, noisy_labels),
        batch_size=args.batch_size,
        shuffle=True,
    )
    student_eval_loader = DataLoader(
        TensorDataset(eval_images, eval_labels),
        batch_size=args.batch_size,
        shuffle=False,
    )

    student_model = UCStubModel().to(device)
    train_student(
        student_model=student_model,
        train_loader=student_train_loader,
        device=device,
        epochs=args.student_epochs,
        lr=args.lr,
    )
    accuracy = evaluate(student_model, student_eval_loader, device)

    if args.mechanism == "gaussian":
        epsilon_per_query = (1.0 / args.sigma) * math.sqrt(
            2.0 * math.log(1.25 / args.delta)
        )
        epsilon_total = len(noisy_labels) * epsilon_per_query
    else:
        epsilon_per_query = 1.0 / args.laplace_scale
        epsilon_total = len(noisy_labels) * epsilon_per_query
    print("")
    print("=============================")
    print("PATE EXPERIMENT SUMMARY")
    print("=============================")
    print(f"Query Source: {args.query_source}")
    print(f"Teachers: {len(teacher_paths)}")
    print(f"Mechanism: {args.mechanism}")
    print(f"Sigma: {args.sigma}")
    print(f"Laplace Scale: {args.laplace_scale}")
    print(f"Queries Used: {len(noisy_labels)}")
    print(f"Student Accuracy: {accuracy:.4f}")
    print(f"{args.mechanism.capitalize()} mechanism used")
    print(f"Epsilon per query: {epsilon_per_query}")
    print(f"Epsilon total: {epsilon_total}")
    print(f"Delta: {args.delta}")
    print("=============================")


if __name__ == "__main__":
    main()
