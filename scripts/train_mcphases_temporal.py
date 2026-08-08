#!/usr/bin/env python
"""Train matched temporal heads over frozen mcPHASES daily embeddings."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from femmhc import TEMPORAL_ENCODERS, count_trainable_parameters
from femmhc.data import McPhasesEmbeddingHistoryDataset
from femmhc.tasks import MCPHASES_TASKS, TaskDefinition


@dataclass(frozen=True)
class Evaluation:
    primary_metric: str
    primary_value: float
    metrics: dict[str, float]
    samples: int
    participants: int


class TemporalTaskModel(nn.Module):
    def __init__(
        self,
        architecture: str,
        task: TaskDefinition,
        *,
        input_dim: int,
        hidden_dim: int,
        history_days: int,
        modes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if architecture not in TEMPORAL_ENCODERS:
            raise ValueError(f"unknown architecture: {architecture}")
        self.task = task
        self.encoder = TEMPORAL_ENCODERS[architecture](
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            maximum_days=history_days,
            modes=modes,
            dropout=dropout,
        )
        output_dim = task.classes if task.kind != "regression" else 1
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim or 1),
        )

    def forward(
        self,
        daily_embeddings: torch.Tensor,
        day_present: torch.Tensor,
    ) -> torch.Tensor:
        representation = self.encoder(daily_embeddings, day_present).representation
        output = self.head(representation)
        return output.squeeze(-1) if self.task.kind == "regression" else output


def _task(name: str) -> TaskDefinition:
    try:
        return next(task for task in MCPHASES_TASKS if task.name == name)
    except StopIteration as error:
        raise ValueError(f"unknown mcPHASES task: {name}") from error


def _class_weights(
    dataset: McPhasesEmbeddingHistoryDataset,
    classes: int,
) -> torch.Tensor:
    target = np.asarray(
        [int(dataset.targets[item["sample_index"]]) for item in dataset.examples]
    )
    count = np.bincount(target, minlength=classes).astype(np.float64)
    weight = target.size / (classes * np.maximum(count, 1.0))
    return torch.tensor(weight, dtype=torch.float32)


def _metrics(
    task: TaskDefinition,
    target: np.ndarray,
    output: np.ndarray,
    participant: np.ndarray,
) -> Evaluation:
    if task.kind == "regression":
        metrics = {
            "mae": float(mean_absolute_error(target, output)),
            "rmse": float(mean_squared_error(target, output) ** 0.5),
        }
        primary = metrics["mae"]
    else:
        probabilities = torch.softmax(torch.from_numpy(output), dim=-1).numpy()
        classes = np.arange(task.classes or probabilities.shape[1])
        if task.kind == "ordinal":
            continuous = probabilities @ classes.astype(np.float64)
            prediction = np.clip(
                np.rint(continuous), 0, (task.classes or 2) - 1
            ).astype(int)
            metrics = {
                "mae": float(mean_absolute_error(target, continuous)),
                "macro_f1": float(
                    f1_score(
                        target,
                        prediction,
                        labels=classes,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "quadratic_kappa": float(
                    cohen_kappa_score(target, prediction, weights="quadratic")
                ),
            }
            primary = metrics["mae"]
        elif task.classes == 2:
            probability = probabilities[:, 1]
            prediction = probability >= 0.5
            metrics = {
                "auprc": float(average_precision_score(target, probability)),
                "auroc": float(roc_auc_score(target, probability)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(target, prediction)
                ),
            }
            primary = metrics["auprc"]
        else:
            prediction = probabilities.argmax(axis=1)
            metrics = {
                "macro_f1": float(
                    f1_score(
                        target,
                        prediction,
                        labels=classes,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "balanced_accuracy": float(
                    balanced_accuracy_score(target, prediction)
                ),
            }
            primary = metrics["macro_f1"]
    return Evaluation(
        primary_metric=task.primary_metric,
        primary_value=float(primary),
        metrics=metrics,
        samples=int(target.size),
        participants=int(np.unique(participant).size),
    )


def _evaluate(
    model: TemporalTaskModel,
    loader: DataLoader,
    task: TaskDefinition,
    device: torch.device,
) -> Evaluation:
    model.eval()
    outputs = []
    targets = []
    participants = []
    with torch.inference_mode():
        for item in loader:
            output = model(
                item["daily_embeddings"].to(device, non_blocking=True),
                item["day_present"].to(device, non_blocking=True),
            )
            outputs.append(output.float().cpu().numpy())
            targets.append(item["target"].numpy())
            participants.extend(item["participant_id"])
    return _metrics(
        task,
        np.concatenate(targets),
        np.concatenate(outputs),
        np.asarray(participants),
    )


def _constant_evaluation(
    training: McPhasesEmbeddingHistoryDataset,
    evaluation: McPhasesEmbeddingHistoryDataset,
    task: TaskDefinition,
) -> Evaluation:
    train_target = np.asarray(
        [training.targets[item["sample_index"]] for item in training.examples]
    )
    target = np.asarray(
        [evaluation.targets[item["sample_index"]] for item in evaluation.examples]
    )
    participant = evaluation.participants
    if task.kind == "regression":
        output = np.full(target.shape, np.median(train_target), dtype=np.float64)
    elif task.kind == "ordinal":
        constant_class = int(np.rint(np.median(train_target)))
        output = np.full((target.size, task.classes or 2), -20.0, dtype=np.float64)
        output[:, constant_class] = 20.0
    else:
        counts = np.bincount(
            train_target.astype(int), minlength=task.classes or 2
        ).astype(np.float64)
        probability = (counts + 1e-6) / (counts.sum() + 1e-6 * len(counts))
        output = np.tile(np.log(probability), (target.size, 1))
    return _metrics(task, target, output, participant)


def _oriented(task: TaskDefinition, value: float) -> float:
    return -value if task.kind in {"ordinal", "regression"} else value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--architecture", choices=tuple(TEMPORAL_ENCODERS), required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-days", type=int, default=60)
    parser.add_argument("--minimum-history-days", type=int, default=7)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(
        args.history_days,
        args.minimum_history_days,
        args.hidden_dim,
        args.modes,
        args.batch_size,
        args.max_epochs,
        args.patience,
    ) <= 0:
        raise ValueError("history, dimensions, batch size, and epochs must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    task = _task(args.task)
    datasets = {
        split: McPhasesEmbeddingHistoryDataset(
            args.processed_dir,
            args.embeddings,
            task=task,
            history_days=args.history_days,
            minimum_history_days=args.minimum_history_days,
            split=split,
        )
        for split in ("train", "validation", "test")
    }
    if any(len(dataset) == 0 for dataset in datasets.values()):
        raise ValueError("every participant split needs at least one history example")
    device = torch.device(args.device)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            generator=generator if split == "train" else None,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        for split, dataset in datasets.items()
    }
    model = TemporalTaskModel(
        args.architecture,
        task,
        input_dim=int(datasets["train"].embeddings.shape[1]),
        hidden_dim=args.hidden_dim,
        history_days=args.history_days,
        modes=args.modes,
        dropout=args.dropout,
    ).to(device)
    class_weights = (
        _class_weights(datasets["train"], task.classes or 2).to(device)
        if task.kind != "regression"
        else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_oriented = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        total_loss = 0.0
        samples = 0
        for item in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            output = model(
                item["daily_embeddings"].to(device, non_blocking=True),
                item["day_present"].to(device, non_blocking=True),
            )
            target = item["target"].to(device, non_blocking=True)
            if task.kind == "regression":
                loss = F.smooth_l1_loss(output, target.float())
            elif task.kind == "ordinal":
                probabilities = torch.softmax(output, dim=-1)
                levels = torch.arange(
                    task.classes or output.shape[-1],
                    device=device,
                    dtype=probabilities.dtype,
                )
                expected = probabilities @ levels
                loss = F.smooth_l1_loss(expected, target.float()) + 0.2 * F.cross_entropy(
                    output, target.long()
                )
            else:
                loss = F.cross_entropy(output, target.long(), weight=class_weights)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * target.shape[0]
            samples += target.shape[0]
        validation = _evaluate(model, loaders["validation"], task, device)
        record = {
            "epoch": epoch,
            "training_loss": total_loss / max(samples, 1),
            "validation": asdict(validation),
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        oriented = _oriented(task, validation.primary_value)
        if oriented > best_oriented + 1e-8:
            best_oriented = oriented
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "format_version": 1,
                    "architecture": args.architecture,
                    "task": task.name,
                    "seed": args.seed,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                },
                args.output,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break

    checkpoint = torch.load(args.output, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation = _evaluate(model, loaders["validation"], task, device)
    test = _evaluate(model, loaders["test"], task, device)
    report = {
        "format_version": 1,
        "stage": "mcphases_temporal_development_v1",
        "temporal_model_version": "selective_cycle_ssm_v2",
        "architecture": args.architecture,
        "task": task.name,
        "task_chinese": task.chinese_name,
        "seed": args.seed,
        "history_days": args.history_days,
        "minimum_history_days": args.minimum_history_days,
        "hidden_dim": args.hidden_dim,
        "modes": args.modes if args.architecture == "cyclessm" else None,
        "trainable_parameters": count_trainable_parameters(model),
        "embedding_path": str(args.embeddings.resolve()),
        "samples": {split: len(dataset) for split, dataset in datasets.items()},
        "participants": {
            split: int(np.unique(dataset.participants).size)
            for split, dataset in datasets.items()
        },
        "best_epoch": best_epoch,
        "validation": asdict(validation),
        "test": asdict(test),
        "constant_baseline": {
            "validation": asdict(
                _constant_evaluation(datasets["train"], datasets["validation"], task)
            ),
            "test": asdict(
                _constant_evaluation(datasets["train"], datasets["test"], task)
            ),
        },
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    for dataset in datasets.values():
        dataset.close()


if __name__ == "__main__":
    main()
