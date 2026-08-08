#!/usr/bin/env python
"""Train one frozen-state phase-head control and evaluate a fixed validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score
import torch

from femmhc.data import McPhasesJointEmbeddingDataset
try:
    from scripts.select_frozen_circular_head_train_cv import (
        HEAD_FAMILIES,
        REPRESENTATION_SOURCES,
        extract_phase_representations,
        fit_phase_head,
        predict_phase_head,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from select_frozen_circular_head_train_cv import (  # type: ignore[no-redef]
        HEAD_FAMILIES,
        REPRESENTATION_SOURCES,
        extract_phase_representations,
        fit_phase_head,
        predict_phase_head,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path)
    parser.add_argument("--head-family", choices=HEAD_FAMILIES, required=True)
    parser.add_argument(
        "--representation-source",
        choices=REPRESENTATION_SOURCES,
        default="cycle",
    )
    parser.add_argument("--geometry-weight", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--minimum-history-days", type=int, default=3)
    parser.add_argument("--mcphases-dir", type=Path, default=Path("processed/mcphases"))
    parser.add_argument(
        "--mcphases-embeddings",
        type=Path,
        default=Path("artifacts/embeddings/mcphases/dual-v4-seed42/femmhc-dual.npy"),
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.learning_rate, args.steps, args.batch_size) <= 0:
        raise ValueError("training settings must be positive")
    if args.geometry_weight < 0:
        raise ValueError("geometry weight must be non-negative")

    device = torch.device(args.device)
    artifact = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    dataset_arguments = {
        "processed_dir": args.mcphases_dir,
        "embeddings_path": args.mcphases_embeddings,
        "history_days": int(artifact["maximum_days"]),
        "minimum_history_days": args.minimum_history_days,
    }
    train_dataset = McPhasesJointEmbeddingDataset(split="train", **dataset_arguments)
    validation_dataset = McPhasesJointEmbeddingDataset(
        split="validation", **dataset_arguments
    )
    x_train, y_train, participant_train = extract_phase_representations(
        artifact,
        train_dataset,
        batch_size=128,
        device=device,
        representation_source=args.representation_source,
    )
    x_validation, y_validation, participant_validation = extract_phase_representations(
        artifact,
        validation_dataset,
        batch_size=128,
        device=device,
        representation_source=args.representation_source,
    )
    model = fit_phase_head(
        x_train,
        y_train,
        participant_train,
        hidden_dim=int(artifact["hidden_dim"]),
        learning_rate=args.learning_rate,
        steps=args.steps,
        batch_size=args.batch_size,
        geometry_weight=args.geometry_weight,
        seed=args.seed,
        device=device,
        head_family=args.head_family,
    )
    prediction = predict_phase_head(
        model,
        x_validation,
        device=device,
    )
    if args.head_checkpoint is not None:
        args.head_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "artifact_type": "frozen_phase_head",
                "base_checkpoint": str(args.base_checkpoint.resolve()),
                "head_family": args.head_family,
                "representation_source": args.representation_source,
                "hidden_dim": int(artifact["hidden_dim"]),
                "head_state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                "geometry_weight": args.geometry_weight,
                "learning_rate": args.learning_rate,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "train_participants": int(len(np.unique(participant_train))),
                "train_samples": int(len(y_train)),
                "validation_used_for_selection": False,
                "test_used": False,
            },
            args.head_checkpoint,
        )
    summary = {
        "format_version": 1,
        "split": "validation",
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "head_checkpoint": (
            None
            if args.head_checkpoint is None
            else str(args.head_checkpoint.resolve())
        ),
        "head_family": args.head_family,
        "representation_source": args.representation_source,
        "trainable_parameters": int(sum(value.numel() for value in model.parameters())),
        "geometry_weight": args.geometry_weight,
        "learning_rate": args.learning_rate,
        "steps": args.steps,
        "seed": args.seed,
        "train_participants": int(len(np.unique(participant_train))),
        "train_samples": int(len(y_train)),
        "validation_participants": int(len(np.unique(participant_validation))),
        "validation_samples": int(len(y_validation)),
        "macro_f1": float(
            f1_score(
                y_validation,
                prediction,
                labels=np.arange(4),
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_validation, prediction)
        ),
        "validation_used_for_selection": False,
        "test_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
