#!/usr/bin/env python
"""Train FemMHC's shared female-health state on heterogeneous cohorts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
from typing import Any, Iterable

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from femmhc import JOINT_TASKS, FemMHCJointModel, partial_multitask_loss
from femmhc.data import (
    AffectiveJointEmbeddingDataset,
    HRVMentalJointEmbeddingDataset,
    McPhasesJointEmbeddingDataset,
    OpenMHCAuxiliaryEmbeddingDataset,
    PregnancyJointEmbeddingDataset,
)


def _datasets(args: argparse.Namespace, split: str) -> dict[str, Dataset]:
    return {
        "openmhc": OpenMHCAuxiliaryEmbeddingDataset(
            args.openmhc_data_dir,
            args.openmhc_native_cache,
            args.openmhc_adapted_cache,
            split=split,
            history_days=args.openmhc_history_days,
        ),
        "mcphases": McPhasesJointEmbeddingDataset(
            args.mcphases_dir,
            args.mcphases_embeddings,
            split=split,
            history_days=args.maximum_days,
            minimum_history_days=args.minimum_history_days,
        ),
        "depress_fitbit": AffectiveJointEmbeddingDataset(
            "depress_fitbit",
            args.depress_dir,
            args.depress_embeddings,
            split=split,
            history_days=28,
            minimum_history_days=args.minimum_history_days,
        ),
        "inphrsym": AffectiveJointEmbeddingDataset(
            "inphrsym",
            args.inphrsym_dir,
            args.inphrsym_embeddings,
            split=split,
            history_days=28,
            minimum_history_days=args.minimum_history_days,
        ),
        "wearable_hrv_sleep": HRVMentalJointEmbeddingDataset(
            args.hrv_mental_dir,
            args.hrv_mental_embeddings,
            split=split,
        ),
        "pregnancy_ga_clock": PregnancyJointEmbeddingDataset(
            args.pregnancy_dir,
            args.pregnancy_embeddings,
            split=split,
        ),
    }


def _loaders(
    datasets: dict[str, Dataset],
    *,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    seed: int,
) -> dict[str, DataLoader]:
    loaders = {}
    for index, (name, dataset) in enumerate(datasets.items()):
        if len(dataset):
            loaders[name] = _loader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                device=device,
                seed=seed + index,
            )
    return loaders


def _loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    seed: int,
) -> DataLoader:
    sampler = None
    if shuffle:
        participant_ids = list(getattr(dataset, "participant_ids"))
        counts = Counter(participant_ids)
        weights = torch.tensor(
            [1.0 / counts[participant] for participant in participant_ids],
            dtype=torch.double,
        )
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def _fit_regression_statistics(
    datasets: Iterable[Dataset],
) -> dict[str, dict[str, float]]:
    regression = {
        task.task_id for task in JOINT_TASKS if task.trainable and task.kind == "regression"
    }
    values: dict[str, list[float]] = defaultdict(list)
    for dataset in datasets:
        target_iterator = (
            dataset.iter_targets()
            if hasattr(dataset, "iter_targets")
            else (dataset[index]["targets"] for index in range(len(dataset)))
        )
        for targets in target_iterator:
            for task_id, target in targets.items():
                if task_id not in regression:
                    continue
                number = float(target)
                if np.isfinite(number):
                    values[task_id].append(number)
    statistics = {}
    for task_id, samples in values.items():
        array = np.asarray(samples, dtype=np.float64)
        standard_deviation = float(array.std())
        statistics[task_id] = {
            "mean": float(array.mean()),
            "std": max(standard_deviation, 1e-6),
            "count": int(len(array)),
        }
    return statistics


def _to_device(
    batch: dict[str, Any],
    device: torch.device,
    statistics: dict[str, dict[str, float]],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    embeddings = batch["daily_embeddings"].to(device, non_blocking=True)
    present = batch["day_present"].to(device, non_blocking=True)
    targets = {}
    for task_id, value in batch["targets"].items():
        target = value.to(device, non_blocking=True)
        if task_id in statistics:
            stats = statistics[task_id]
            target = (target - stats["mean"]) / stats["std"]
        targets[task_id] = target
    return embeddings, present, targets


def _next_batch(
    name: str,
    loaders: dict[str, DataLoader],
    iterators: dict[str, Any],
) -> dict[str, Any]:
    try:
        return next(iterators[name])
    except StopIteration:
        iterators[name] = iter(loaders[name])
        return next(iterators[name])


@torch.no_grad()
def _validate(
    model: FemMHCJointModel,
    loaders: dict[str, DataLoader],
    *,
    device: torch.device,
    statistics: dict[str, dict[str, float]],
    maximum_batches: int,
) -> tuple[float, dict[str, float]]:
    model.eval()
    cohort_scores = {}
    for cohort, loader in loaders.items():
        losses = []
        for index, batch in enumerate(loader):
            if index >= maximum_batches:
                break
            embeddings, present, targets = _to_device(batch, device, statistics)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(
                    embeddings,
                    present,
                    task_ids=tuple(targets),
                )
                loss = partial_multitask_loss(output, targets).total
            losses.append(float(loss.float().cpu()))
        cohort_scores[cohort] = float(np.mean(losses))
    model.train()
    return float(np.mean(list(cohort_scores.values()))), cohort_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--openmhc-data-dir", type=Path, default=Path("datasets/openmhc-xs")
    )
    parser.add_argument(
        "--openmhc-native-cache",
        type=Path,
        default=Path("artifacts/embeddings/openmhc-xs/openmhc-lsm2"),
    )
    parser.add_argument(
        "--openmhc-adapted-cache",
        type=Path,
        default=Path("artifacts/embeddings/openmhc-xs/femmhc-stage1-v4"),
    )
    parser.add_argument("--openmhc-history-days", type=int, default=7)
    parser.add_argument("--mcphases-dir", type=Path, default=Path("processed/mcphases"))
    parser.add_argument(
        "--mcphases-embeddings",
        type=Path,
        default=Path("artifacts/embeddings/mcphases/dual-v4-seed42/femmhc-dual.npy"),
    )
    parser.add_argument("--depress-dir", type=Path, default=Path("processed/depress_fitbit"))
    parser.add_argument(
        "--depress-embeddings",
        type=Path,
        default=Path("artifacts/embeddings/depress-fitbit-affective-dynamics-step100.npz"),
    )
    parser.add_argument("--inphrsym-dir", type=Path, default=Path("processed/inphrsym"))
    parser.add_argument(
        "--inphrsym-embeddings",
        type=Path,
        default=Path("artifacts/embeddings/inphrsym-affective-dynamics-step100.npz"),
    )
    parser.add_argument(
        "--hrv-mental-dir",
        type=Path,
        default=Path("processed/wearable_hrv_mental_female"),
    )
    parser.add_argument(
        "--hrv-mental-embeddings",
        type=Path,
        default=Path("artifacts/embeddings/hrv-mental-female/femmhc-stage1-seed42.npz"),
    )
    parser.add_argument(
        "--pregnancy-dir",
        type=Path,
        default=Path("processed/pregnancy_ga_clock_official"),
    )
    parser.add_argument(
        "--pregnancy-embeddings",
        type=Path,
        default=Path("artifacts/embeddings/pregnancy-ga-official/progression-v4-best.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/checkpoints/femmhc-joint-v1.pt"),
    )
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--routing-initial-logit", type=float, default=-4.0)
    parser.add_argument(
        "--architecture",
        choices=(
            "last_day_shared",
            "history_conditioned_adapter",
            "shared_backbone",
            "factorized_no_graph",
            "full",
            "gated_graph",
            "task_router",
            "dual_path_router",
            "dual_view_residual_router",
            "dual_path_no_cycle",
            "dual_path_own_domain",
            "dual_path_fixed_gate",
            "dual_path_timescale_router",
            "dual_path_source_aware",
            "dual_path_cycle_aware",
            "dual_path_cycle_direct",
            "dual_path_task_selected",
            "dual_path_task_selected_soft",
            "dual_path_phase_geometry",
            "dual_path_circular_phase_head",
            "mmoe",
            "ple",
        ),
        default="dual_path_router",
    )
    parser.add_argument("--maximum-days", type=int, default=60)
    parser.add_argument("--minimum-history-days", type=int, default=3)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--cohort-sampling-temperature", type=float, default=0.5)
    parser.add_argument("--phase-geometry-weight", type=float, default=0.25)
    parser.add_argument(
        "--checkpoint-selection",
        choices=("best_validation", "final_step"),
        default="best_validation",
        help="Use final_step for fixed-budget architecture comparisons.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(
        args.max_steps,
        args.batch_size,
        args.hidden_dim,
        args.maximum_days,
        args.minimum_history_days,
        args.validate_every,
        args.validation_batches,
        args.log_every,
    ) <= 0:
        raise ValueError("training sizes and intervals must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if not 0 <= args.cohort_sampling_temperature <= 1:
        raise ValueError("cohort-sampling-temperature must be between 0 and 1")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if args.phase_geometry_weight < 0:
        raise ValueError("phase-geometry-weight must be non-negative")

    train_datasets = _datasets(args, "train")
    validation_datasets = _datasets(args, "validation")
    statistics = _fit_regression_statistics(train_datasets.values())
    train_loaders = _loaders(
        train_datasets,
        batch_size=args.batch_size,
        shuffle=True,
        device=device,
        seed=args.seed,
    )
    validation_loaders = _loaders(
        validation_datasets,
        batch_size=args.batch_size,
        shuffle=False,
        device=device,
        seed=args.seed + 10_000,
    )
    model = FemMHCJointModel(
        input_dim=768,
        hidden_dim=args.hidden_dim,
        maximum_days=args.maximum_days,
        architecture=args.architecture,
        dropout=args.dropout,
        initialization_seed=args.seed,
        routing_initial_logit=args.routing_initial_logit,
    ).to(device)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    cohort_names = tuple(train_loaders)
    cohort_sampling_weights = [
        len(train_datasets[name]) ** args.cohort_sampling_temperature
        for name in cohort_names
    ]
    total_sampling_weight = sum(cohort_sampling_weights)
    cohort_sampling_probabilities = {
        name: weight / total_sampling_weight
        for name, weight in zip(cohort_names, cohort_sampling_weights)
    }
    randomizer = random.Random(args.seed)
    iterators = {name: iter(loader) for name, loader in train_loaders.items()}
    history: list[dict[str, Any]] = []
    best_validation = float("inf")
    best_step = 0
    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.max_steps + 1):
        cohort = randomizer.choices(
            cohort_names,
            weights=cohort_sampling_weights,
            k=1,
        )[0]
        batch = _next_batch(cohort, train_loaders, iterators)
        embeddings, present, targets = _to_device(batch, device, statistics)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(embeddings, present, task_ids=tuple(targets))
            losses = partial_multitask_loss(
                output,
                targets,
                phase_geometry_weight=(
                    args.phase_geometry_weight
                    if args.architecture
                    in {
                        "dual_path_phase_geometry",
                        "dual_path_circular_phase_head",
                    }
                    else 0.0
                ),
            )
        losses.total.backward()
        gradient_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            event = {
                "step": step,
                "cohort": cohort,
                "train_loss": float(losses.total.detach().float().cpu()),
                "gradient_norm": float(gradient_norm.detach().float().cpu()),
                "active_domains": sorted(losses.per_domain),
            }
            print(json.dumps(event, ensure_ascii=False), flush=True)

        if step % args.validate_every == 0 or step == args.max_steps:
            validation, cohort_validation = _validate(
                model,
                validation_loaders,
                device=device,
                statistics=statistics,
                maximum_batches=args.validation_batches,
            )
            record = {
                "step": step,
                "validation_loss": validation,
                "cohort_validation_loss": cohort_validation,
            }
            history.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            is_best = validation < best_validation
            if is_best:
                best_validation = validation
                best_step = step
            should_save = (
                args.checkpoint_selection == "best_validation" and is_best
            ) or (
                args.checkpoint_selection == "final_step" and step == args.max_steps
            )
            if should_save:
                torch.save(
                    {
                        "format_version": 1,
                        "model": "FemMHCJointModel",
                        "stage": "joint_female_health_v1",
                        "architecture": args.architecture,
                        "dropout": args.dropout,
                        "initialization_seed": args.seed,
                        "routing_initial_logit": args.routing_initial_logit,
                        "phase_geometry_weight": args.phase_geometry_weight,
                        "status": (
                            "best"
                            if args.checkpoint_selection == "best_validation"
                            else "final_step"
                        ),
                        "checkpoint_selection": args.checkpoint_selection,
                        "seed": args.seed,
                        "step": step,
                        "input_dim": 768,
                        "hidden_dim": args.hidden_dim,
                        "trainable_parameters": trainable_parameters,
                        "maximum_days": args.maximum_days,
                        "cohort_sizes": {
                            name: {
                                "train": len(train_datasets[name]),
                                "validation": len(validation_datasets[name]),
                            }
                            for name in train_datasets
                        },
                        "cohort_sampling_probabilities": cohort_sampling_probabilities,
                        "regression_target_statistics": statistics,
                        "tasks": [asdict(task) for task in JOINT_TASKS],
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "validation_loss": validation,
                        "cohort_validation_loss": cohort_validation,
                        "history": history,
                    },
                    args.output,
                )

    test_datasets = _datasets(args, "test")
    summary = {
        "status": "complete",
        "device": str(device),
        "steps": args.max_steps,
        "architecture": args.architecture,
        "dropout": args.dropout,
        "routing_initial_logit": args.routing_initial_logit,
        "phase_geometry_weight": args.phase_geometry_weight,
        "checkpoint_selection": args.checkpoint_selection,
        "trainable_parameters": trainable_parameters,
        "best_step": best_step,
        "best_validation_loss": best_validation,
        "checkpoint": str(args.output.resolve()),
        "elapsed_seconds": time.perf_counter() - started,
        "cohort_sizes": {
            name: {
                "train": len(train_datasets[name]),
                "validation": len(validation_datasets[name]),
                "test": len(test_datasets[name]),
            }
            for name in train_datasets
        },
        "cohort_sampling_probabilities": cohort_sampling_probabilities,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
