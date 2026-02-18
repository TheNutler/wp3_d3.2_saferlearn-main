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


def load_teacher_paths(teachers_root: Path, teacher_count: int) -> List[Path]:
    paths: List[Path] = []
    for teacher_id in range(teacher_count):
        ckpt = teachers_root / str(teacher_id) / "model.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing teacher checkpoint: {ckpt}")
        paths.append(ckpt)
    return paths


def collect_teacher_votes(
    teacher_paths: List[Path],
    query_loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Return vote counts tensor with shape [num_queries, num_classes]."""
    num_queries = len(query_loader.dataset)
    vote_counts = torch.zeros((num_queries, num_classes), dtype=torch.int32)

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

    return vote_counts


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
    mnist_root = repo_root / "public-dataset"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}")

    teacher_paths = load_teacher_paths(teachers_root, args.teacher_count)
    if len(teacher_paths) != args.teacher_count:
        raise RuntimeError(
            f"Expected {args.teacher_count} teacher checkpoints, "
            f"found {len(teacher_paths)}."
        )
    print(f"[setup] teacher_count={len(teacher_paths)}")
    print(f"[setup] sigma={args.sigma}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    mnist_test = datasets.MNIST(
        root=str(mnist_root), train=False, download=True, transform=transform
    )

    # t10k split: 90% query set for noisy labels, 10% held-out eval set.
    all_indices = torch.randperm(len(mnist_test))
    query_size = int(0.9 * len(mnist_test))
    query_indices = all_indices[:query_size]
    eval_indices = all_indices[query_size:]

    query_images = torch.stack([mnist_test[int(i)][0] for i in query_indices], dim=0)
    eval_images = torch.stack([mnist_test[int(i)][0] for i in eval_indices], dim=0)
    eval_labels = torch.tensor(
        [int(mnist_test[int(i)][1]) for i in eval_indices], dtype=torch.long
    )

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

    noisy_counts = vote_counts.float() + torch.randn_like(vote_counts.float()) * args.sigma
    noisy_labels = torch.argmax(noisy_counts, dim=1).long()
    print(f"[aggregate] query_samples={len(noisy_labels)}")

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

    epsilon = estimate_epsilon_gaussian(
        num_queries=len(noisy_labels), sigma=args.sigma, delta=args.delta
    )
    print("")
    print("-----------------------------------------")
    print("BASE PATE (MNIST)")
    print(f"Teachers: {len(teacher_paths)}")
    print(f"Queries: {len(noisy_labels)}")
    print(f"Sigma: {args.sigma:.1f}")
    print(f"Student Accuracy: {accuracy:.2f}%")
    print(f"Epsilon (delta={args.delta}): {epsilon:.4f}")
    print("-----------------------------------------")


if __name__ == "__main__":
    main()
