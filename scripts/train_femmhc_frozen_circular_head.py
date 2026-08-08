#!/usr/bin/env python
"""Train only FemMHC's circular menstrual-phase head on a frozen joint model."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import f1_score
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, WeightedRandomSampler

from femmhc import JOINT_TASKS, FemMHCJointModel, partial_multitask_loss
from femmhc.data import McPhasesJointEmbeddingDataset


PHASE_TASK_ID = "mcphases/cycle_phase"
PROJECTOR_PREFIX = "cycle_phase_projector."
EXPECTED_NEW_STATE = {
    "cycle_phase_prototypes",
    "cycle_phase_projector.0.weight",
    "cycle_phase_projector.0.bias",
    "cycle_phase_projector.1.weight",
    "cycle_phase_projector.1.bias",
}


def build_frozen_circular_model(
    base_artifact: Mapping[str, Any],
    *,
    initialization_seed: int,
) -> tuple[FemMHCJointModel, dict[str, Any]]:
    """Transplant a trained dual-path model and expose only the new 2-D head."""

    base_architecture = str(base_artifact.get("architecture", ""))
    if not base_architecture.startswith("dual_path"):
        raise ValueError("base checkpoint must use a dual_path architecture")
    model = FemMHCJointModel(
        input_dim=int(base_artifact["input_dim"]),
        hidden_dim=int(base_artifact["hidden_dim"]),
        maximum_days=int(base_artifact["maximum_days"]),
        architecture="dual_path_circular_phase_head",
        dropout=float(base_artifact.get("dropout", 0.0)),
        initialization_seed=initialization_seed,
        routing_initial_logit=float(base_artifact.get("routing_initial_logit", -2.0)),
    )
    incompatible = model.load_state_dict(base_artifact["model_state_dict"], strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing != EXPECTED_NEW_STATE or unexpected:
        raise RuntimeError(
            "unexpected checkpoint transplant mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.cycle_phase_projector.parameters():
        parameter.requires_grad_(True)
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not trainable or any(not name.startswith(PROJECTOR_PREFIX) for name in trainable):
        raise RuntimeError("only the circular phase projector may remain trainable")
    audit = {
        "base_architecture": base_architecture,
        "missing_new_state": sorted(missing),
        "trainable_parameter_names": sorted(trainable),
        "trainable_parameters": int(sum(value.numel() for value in trainable.values())),
        "total_parameters": int(sum(value.numel() for value in model.parameters())),
    }
    return model, audit


def snapshot_frozen_state(model: FemMHCJointModel) -> dict[str, torch.Tensor]:
    """Copy every tensor that must remain bitwise unchanged during head training."""

    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not name.startswith(PROJECTOR_PREFIX)
    }


def assert_frozen_state_unchanged(
    model: FemMHCJointModel,
    reference: Mapping[str, torch.Tensor],
) -> None:
    current = model.state_dict()
    changed = [
        name
        for name, expected in reference.items()
        if name not in current or not torch.equal(current[name].detach().cpu(), expected)
    ]
    if changed:
        raise RuntimeError(f"frozen model state changed: {changed[:10]}")


def _loader(
    dataset: McPhasesJointEmbeddingDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    sampler = None
    if shuffle:
        participants = list(dataset.participant_ids)
        counts = Counter(participants)
        weights = torch.tensor(
            [1.0 / counts[participant] for participant in participants],
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


def _phase_batch(
    batch: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    return (
        batch["daily_embeddings"].to(device, non_blocking=True),
        batch["day_present"].to(device, non_blocking=True),
        {PHASE_TASK_ID: batch["targets"][PHASE_TASK_ID].to(device, non_blocking=True)},
    )


@torch.no_grad()
def evaluate_phase(
    model: FemMHCJointModel,
    loader: DataLoader,
    *,
    device: torch.device,
    geometry_weight: float,
) -> dict[str, float | int]:
    model.eval()
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    losses: list[float] = []
    for batch in loader:
        embeddings, present, targets = _phase_batch(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(embeddings, present, task_ids=(PHASE_TASK_ID,))
            loss = partial_multitask_loss(
                output,
                targets,
                phase_geometry_weight=geometry_weight,
            ).total
        target = targets[PHASE_TASK_ID]
        observed = torch.isfinite(target) & (target >= 0)
        if bool(observed.any()):
            probability = output.predictions[PHASE_TASK_ID].probabilities
            labels.append(target[observed].long().cpu().numpy())
            predictions.append(probability[observed].argmax(dim=-1).cpu().numpy())
            losses.append(float(loss.detach().float().cpu()))
    if not labels:
        raise RuntimeError("validation split contains no observed cycle-phase labels")
    target = np.concatenate(labels)
    prediction = np.concatenate(predictions)
    return {
        "macro_f1": float(f1_score(target, prediction, average="macro")),
        "loss": float(np.mean(losses)),
        "samples": int(len(target)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mcphases-dir", type=Path, default=Path("processed/mcphases"))
    parser.add_argument(
        "--mcphases-embeddings",
        type=Path,
        default=Path("artifacts/embeddings/mcphases/dual-v4-seed42/femmhc-dual.npy"),
    )
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--phase-geometry-weight", type=float, default=0.1)
    parser.add_argument("--validate-every", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--checkpoint-selection",
        choices=("validation_macro_f1", "final_step"),
        default="validation_macro_f1",
    )
    parser.add_argument("--minimum-history-days", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(
        args.max_steps,
        args.batch_size,
        args.validate_every,
        args.log_every,
        args.minimum_history_days,
    ) <= 0:
        raise ValueError("training sizes and intervals must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.phase_geometry_weight < 0:
        raise ValueError("optimizer settings and geometry weight must be non-negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    base_artifact = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    model, transplant_audit = build_frozen_circular_model(
        base_artifact,
        initialization_seed=args.seed,
    )
    model = model.to(device)
    frozen_reference = snapshot_frozen_state(model)
    maximum_days = int(base_artifact["maximum_days"])
    train_dataset = McPhasesJointEmbeddingDataset(
        args.mcphases_dir,
        args.mcphases_embeddings,
        split="train",
        history_days=maximum_days,
        minimum_history_days=args.minimum_history_days,
    )
    validation_dataset = McPhasesJointEmbeddingDataset(
        args.mcphases_dir,
        args.mcphases_embeddings,
        split="validation",
        history_days=maximum_days,
        minimum_history_days=args.minimum_history_days,
    )
    train_loader = _loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        device=device,
    )
    validation_loader = _loader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed + 10_000,
        device=device,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    iterator = iter(train_loader)
    history: list[dict[str, Any]] = []
    best_f1 = float("-inf")
    best_loss = float("inf")
    best_step = 0
    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        embeddings, present, targets = _phase_batch(batch, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(embeddings, present, task_ids=(PHASE_TASK_ID,))
            loss = partial_multitask_loss(
                output,
                targets,
                phase_geometry_weight=args.phase_geometry_weight,
            ).total
        loss.backward()
        gradient_norm = clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "train_loss": float(loss.detach().float().cpu()),
                        "gradient_norm": float(gradient_norm.detach().float().cpu()),
                    }
                ),
                flush=True,
            )
        should_validate = step == args.max_steps or (
            args.checkpoint_selection == "validation_macro_f1"
            and step % args.validate_every == 0
        )
        if should_validate:
            validation = evaluate_phase(
                model,
                validation_loader,
                device=device,
                geometry_weight=args.phase_geometry_weight,
            )
            record = {"step": step, **validation}
            history.append(record)
            print(json.dumps(record), flush=True)
            improved = args.checkpoint_selection == "final_step" or (
                validation["macro_f1"] > best_f1 + 1e-12
                or (
                    abs(validation["macro_f1"] - best_f1) <= 1e-12
                    and validation["loss"] < best_loss
                )
            )
            if improved:
                best_f1 = float(validation["macro_f1"])
                best_loss = float(validation["loss"])
                best_step = step
                torch.save(
                    {
                        "format_version": 1,
                        "model": "FemMHCJointModel",
                        "stage": "frozen_circular_phase_head_v1",
                        "architecture": "dual_path_circular_phase_head",
                        "base_architecture": transplant_audit["base_architecture"],
                        "base_checkpoint": str(args.base_checkpoint.resolve()),
                        "trainable_scope": "circular_phase_head_only",
                        "checkpoint_selection": args.checkpoint_selection,
                        "trainable_parameters": transplant_audit["trainable_parameters"],
                        "dropout": float(base_artifact.get("dropout", 0.0)),
                        "initialization_seed": args.seed,
                        "routing_initial_logit": float(
                            base_artifact.get("routing_initial_logit", -2.0)
                        ),
                        "phase_geometry_weight": args.phase_geometry_weight,
                        "learning_rate": args.learning_rate,
                        "weight_decay": args.weight_decay,
                        "status": "best",
                        "seed": args.seed,
                        "step": step,
                        "input_dim": int(base_artifact["input_dim"]),
                        "hidden_dim": int(base_artifact["hidden_dim"]),
                        "maximum_days": maximum_days,
                        "cohort_sizes": {
                            "mcphases": {
                                "train": len(train_dataset),
                                "validation": len(validation_dataset),
                            }
                        },
                        "regression_target_statistics": base_artifact.get(
                            "regression_target_statistics", {}
                        ),
                        "tasks": base_artifact.get(
                            "tasks", [asdict(task) for task in JOINT_TASKS]
                        ),
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "validation_phase_macro_f1": best_f1,
                        "validation_phase_loss": best_loss,
                        "history": history,
                    },
                    args.output,
                )

    assert_frozen_state_unchanged(model, frozen_reference)
    summary = {
        "status": "complete",
        "device": str(device),
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "checkpoint": str(args.output.resolve()),
        "trainable_scope": "circular_phase_head_only",
        "checkpoint_selection": args.checkpoint_selection,
        **transplant_audit,
        "phase_geometry_weight": args.phase_geometry_weight,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "steps": args.max_steps,
        "best_step": best_step,
        "best_validation_phase_macro_f1": best_f1,
        "best_validation_phase_loss": best_loss,
        "frozen_state_bitwise_unchanged": True,
        "elapsed_seconds": time.perf_counter() - started,
        "cohort_sizes": {
            "train": len(train_dataset),
            "validation": len(validation_dataset),
        },
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
