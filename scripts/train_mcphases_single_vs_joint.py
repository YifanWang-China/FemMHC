#!/usr/bin/env python
"""Train one controlled mcPHASES joint or task-isolated FemMHC model.

The experiment intentionally uses only mcPHASES and a fixed training budget.
The 24 h and 72 h onset targets remain one isolated task family because their
shared nested head is part of the deployable probability definition.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from femmhc import JOINT_TASKS, FemMHCJointModel, ProbabilisticOutput
from femmhc import partial_multitask_loss
from femmhc.data import McPhasesJointEmbeddingDataset


TASK_GROUPS: dict[str, tuple[str, ...]] = {
    "cycle": ("mcphases/cycle_phase",),
    "onset": (
        "mcphases/menstrual_onset_24h",
        "mcphases/menstrual_onset_72h",
    ),
    "cramps": ("mcphases/cramps",),
    "mood": ("mcphases/mood_swing",),
    "sleep": ("mcphases/sleep_issue",),
}
JOINT_TASK_IDS = tuple(task for group in TASK_GROUPS.values() for task in group)


def _task_specs(task_ids: Sequence[str]):
    by_id = {task.task_id: task for task in JOINT_TASKS if task.trainable}
    missing = set(task_ids) - set(by_id)
    if missing:
        raise KeyError(f"unknown tasks: {sorted(missing)}")
    return tuple(by_id[task_id] for task_id in task_ids)


def _is_observed(value: torch.Tensor, *, kind: str) -> bool:
    if kind == "regression":
        return bool(torch.isfinite(value).item())
    return bool((torch.isfinite(value) & (value >= 0)).item())


class TaskFilteredDataset(Dataset[dict[str, Any]]):
    """Keep examples with at least one selected target and expose a stable key."""

    def __init__(self, base: McPhasesJointEmbeddingDataset, task_ids: Sequence[str]):
        self.base = base
        self.task_ids = tuple(task_ids)
        self.task_by_id = {task.task_id: task for task in _task_specs(task_ids)}
        self.indices: list[int] = []
        for index in range(len(base)):
            targets = base[index]["targets"]
            if any(
                _is_observed(targets[task_id], kind=self.task_by_id[task_id].kind)
                for task_id in self.task_ids
            ):
                self.indices.append(index)
        self.participant_ids = [base.participant_ids[index] for index in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        base_index = self.indices[index]
        item = dict(self.base[base_index])
        item["targets"] = {
            task_id: item["targets"][task_id] for task_id in self.task_ids
        }
        item["example_index"] = base_index
        return item


def _loader(
    dataset: Dataset[dict[str, Any]],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed),
    )


def _to_device(batch: dict[str, Any], device: torch.device):
    embeddings = batch["daily_embeddings"].to(device, non_blocking=True)
    present = batch["day_present"].to(device, non_blocking=True)
    targets = {
        task_id: value.to(device, non_blocking=True)
        for task_id, value in batch["targets"].items()
    }
    return embeddings, present, targets


def _cycle_prediction(output: ProbabilisticOutput) -> tuple[np.ndarray, np.ndarray]:
    probabilities = output.probabilities.float().cpu().numpy()
    return probabilities.argmax(axis=1).astype(float), probabilities.max(axis=1)


def _binary_prediction(output: ProbabilisticOutput) -> tuple[np.ndarray, np.ndarray]:
    probabilities = output.probabilities.float().cpu().numpy()
    return probabilities[:, 1], probabilities[:, 1]


def _ordinal_prediction(output: ProbabilisticOutput) -> tuple[np.ndarray, np.ndarray]:
    probabilities = output.probabilities.float().cpu().numpy()
    classes = np.arange(probabilities.shape[1], dtype=np.float64)
    expected = probabilities @ classes
    return expected, probabilities.max(axis=1)


def _predict(
    model: FemMHCJointModel,
    loader: DataLoader,
    *,
    device: torch.device,
    task_ids: Sequence[str],
    seed: int,
    mode: str,
    task_group: str,
) -> pd.DataFrame:
    task_by_id = {task.task_id: task for task in _task_specs(task_ids)}
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            embeddings, present, targets = _to_device(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(embeddings, present, task_ids=tuple(task_ids))
            participants = list(batch["participant_id"])
            example_indices = batch["example_index"].cpu().numpy()
            for task_id in task_ids:
                target = targets[task_id].float().cpu().numpy()
                observed = np.isfinite(target)
                if task_by_id[task_id].kind != "regression":
                    observed &= target >= 0
                if not observed.any():
                    continue
                prediction_output = output.predictions[task_id]
                if not isinstance(prediction_output, ProbabilisticOutput):
                    raise TypeError("the six-task experiment expects probabilistic outputs")
                if task_by_id[task_id].kind == "multiclass":
                    prediction, confidence = _cycle_prediction(prediction_output)
                elif task_by_id[task_id].kind == "binary":
                    prediction, confidence = _binary_prediction(prediction_output)
                elif task_by_id[task_id].kind == "ordinal":
                    prediction, confidence = _ordinal_prediction(prediction_output)
                else:
                    raise ValueError(f"unsupported task kind: {task_by_id[task_id].kind}")
                for offset in np.flatnonzero(observed):
                    rows.append(
                        {
                            "mode": mode,
                            "task_group": task_group,
                            "seed": seed,
                            "task_id": task_id,
                            "participant_id": participants[offset],
                            "example_index": int(example_indices[offset]),
                            "target": float(target[offset]),
                            "prediction": float(prediction[offset]),
                            "confidence": float(confidence[offset]),
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("joint", "single"), required=True)
    parser.add_argument("--task-group", choices=tuple(TASK_GROUPS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--processed-dir", type=Path, default=Path("processed/mcphases"))
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("artifacts/embeddings/mcphases/dual-v4-seed42/femmhc-dual.npy"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--architecture",
        choices=("dual_path_router", "history_conditioned_adapter", "shared_backbone", "last_day_shared", "mmoe"),
        default="dual_path_router",
        help="Daily representation architecture used for the controlled six-task run.",
    )
    parser.add_argument("--maximum-days", type=int, default=60)
    parser.add_argument("--minimum-history-days", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.mode == "single" and args.task_group is None:
        parser.error("--task-group is required for --mode single")
    if args.mode == "joint" and args.task_group is not None:
        parser.error("--task-group is not used for --mode joint")
    if min(
        args.max_steps,
        args.batch_size,
        args.hidden_dim,
        args.maximum_days,
        args.minimum_history_days,
        args.log_every,
    ) <= 0:
        raise ValueError("training sizes must be positive")

    task_group = "joint6" if args.mode == "joint" else str(args.task_group)
    task_ids = JOINT_TASK_IDS if args.mode == "joint" else TASK_GROUPS[task_group]
    tasks = _task_specs(task_ids)
    # Every arm instantiates the exact same six-task parameterization.  The
    # isolated arms differ only in which targets are exposed to the loss.  This
    # keeps the state encoder, task heads, and routing tensors bitwise matched
    # at initialization for a given seed.
    model_tasks = _task_specs(JOINT_TASK_IDS)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)

    train_base = McPhasesJointEmbeddingDataset(
        args.processed_dir,
        args.embeddings,
        split="train",
        history_days=args.maximum_days,
        minimum_history_days=args.minimum_history_days,
    )
    validation_base = McPhasesJointEmbeddingDataset(
        args.processed_dir,
        args.embeddings,
        split="validation",
        history_days=args.maximum_days,
        minimum_history_days=args.minimum_history_days,
    )
    train_data = TaskFilteredDataset(train_base, task_ids)
    validation_data = TaskFilteredDataset(validation_base, task_ids)
    train_loader = _loader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    validation_loader = _loader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed + 10_000,
    )

    model = FemMHCJointModel(
        input_dim=768,
        hidden_dim=args.hidden_dim,
        tasks=model_tasks,
        maximum_days=args.maximum_days,
        architecture=args.architecture,
        dropout=0.0,
        initialization_seed=args.seed,
        routing_initial_logit=-2.0,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    iterator = iter(train_loader)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    model.train()
    for step in range(1, args.max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        embeddings, present, targets = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(embeddings, present, task_ids=tuple(task_ids))
            loss = partial_multitask_loss(output, targets, tasks=tasks).total
        loss.backward()
        gradient_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.max_steps:
            event = {
                "step": step,
                "loss": float(loss.detach().float().cpu()),
                "gradient_norm": float(gradient_norm.detach().float().cpu()),
            }
            history.append(event)
            print(json.dumps(event, ensure_ascii=False), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "checkpoint.pt"
    torch.save(
        {
            "format_version": 1,
            "experiment": "mcphases_single_vs_joint",
            "mode": args.mode,
            "task_group": task_group,
            "task_ids": list(task_ids),
            "active_tasks": [asdict(task) for task in tasks],
            "instantiated_tasks": [asdict(task) for task in model_tasks],
            "seed": args.seed,
            "step": args.max_steps,
            "input_dim": 768,
            "hidden_dim": args.hidden_dim,
            "maximum_days": args.maximum_days,
            "architecture": args.architecture,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        checkpoint,
    )
    predictions = _predict(
        model,
        validation_loader,
        device=device,
        task_ids=task_ids,
        seed=args.seed,
        mode=args.mode,
        task_group=task_group,
    )
    predictions.to_csv(args.output_dir / "validation_predictions.csv", index=False)
    summary = {
        "format_version": 1,
        "mode": args.mode,
        "task_group": task_group,
        "task_ids": list(task_ids),
        "seed": args.seed,
        "steps": args.max_steps,
        "train_examples": len(train_data),
        "validation_examples": len(validation_data),
        "validation_prediction_rows": len(predictions),
        "participants": len(set(validation_data.participant_ids)),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "elapsed_seconds": time.perf_counter() - started,
        "fixed_protocol": {
            "input_dimension": 768,
            "history_days": args.maximum_days,
            "hidden_dimension": args.hidden_dim,
            "architecture": args.architecture,
            "dropout": 0.0,
            "routing_initial_logit": -2.0,
            "checkpoint_selection": "fixed_final_step",
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
