#!/usr/bin/env python
"""Paired participant bootstrap for two train-selected frozen phase heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
import torch

from femmhc.data import McPhasesJointEmbeddingDataset
from femmhc.statistics import paired_cluster_bootstrap
try:
    from scripts.select_frozen_circular_head_train_cv import (
        HEAD_FAMILIES,
        REPRESENTATION_SOURCES,
        extract_phase_representations,
        fit_predict,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from select_frozen_circular_head_train_cv import (  # type: ignore[no-redef]
        HEAD_FAMILIES,
        REPRESENTATION_SOURCES,
        extract_phase_representations,
        fit_predict,
    )


def parse_head(value: str) -> tuple[str, float, float, int]:
    """Parse FAMILY,GEOMETRY_WEIGHT,LEARNING_RATE,STEPS."""

    parts = value.split(",")
    if len(parts) != 4 or parts[0] not in HEAD_FAMILIES:
        raise argparse.ArgumentTypeError(
            "head must be FAMILY,GEOMETRY_WEIGHT,LEARNING_RATE,STEPS"
        )
    family, weight, learning_rate, steps = parts
    parsed = (family, float(weight), float(learning_rate), int(steps))
    if parsed[1] < 0 or min(parsed[2], parsed[3]) <= 0:
        raise argparse.ArgumentTypeError("head training settings are invalid")
    return parsed


def macro_f1(target: np.ndarray, prediction: np.ndarray, indices: np.ndarray) -> float:
    return float(
        f1_score(
            target[indices],
            prediction[indices],
            labels=np.arange(4),
            average="macro",
            zero_division=0,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-head", type=parse_head, required=True)
    parser.add_argument("--candidate-head", type=parse_head, required=True)
    parser.add_argument(
        "--baseline-representation-source",
        choices=REPRESENTATION_SOURCES,
        default="cycle",
    )
    parser.add_argument(
        "--candidate-representation-source",
        choices=REPRESENTATION_SOURCES,
        default="cycle",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--minimum-history-days", type=int, default=3)
    parser.add_argument("--mcphases-dir", type=Path, default=Path("processed/mcphases"))
    parser.add_argument(
        "--mcphases-embeddings",
        type=Path,
        default=Path("artifacts/embeddings/mcphases/dual-v4-seed42/femmhc-dual.npy"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.replicates <= 0 or args.batch_size <= 0:
        raise ValueError("replicates and batch size must be positive")

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
    extracted: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, source in {
        "baseline": args.baseline_representation_source,
        "candidate": args.candidate_representation_source,
    }.items():
        x_train, y_train, participant_train = extract_phase_representations(
            artifact,
            train_dataset,
            batch_size=128,
            device=device,
            representation_source=source,
        )
        x_validation, y_validation, participant_validation = (
            extract_phase_representations(
                artifact,
                validation_dataset,
                batch_size=128,
                device=device,
                representation_source=source,
            )
        )
        extracted[name] = (
            x_train,
            y_train,
            participant_train,
            x_validation,
        )
        if name == "baseline":
            shared_y_validation = y_validation
            shared_participant_validation = participant_validation
        else:
            if not np.array_equal(shared_y_validation, y_validation):
                raise RuntimeError("representation sources produced different targets")
            if not np.array_equal(
                shared_participant_validation,
                participant_validation,
            ):
                raise RuntimeError(
                    "representation sources produced different participant order"
                )
    y_validation = shared_y_validation
    participant_validation = shared_participant_validation

    predictions: dict[str, np.ndarray] = {}
    specifications = {
        "baseline": (
            args.baseline_head,
            args.baseline_representation_source,
        ),
        "candidate": (
            args.candidate_head,
            args.candidate_representation_source,
        ),
    }
    for name, (head, _) in specifications.items():
        family, weight, learning_rate, steps = head
        x_train, y_train, participant_train, x_validation = extracted[name]
        predictions[name] = fit_predict(
            x_train,
            y_train,
            participant_train,
            x_validation,
            hidden_dim=int(artifact["hidden_dim"]),
            learning_rate=learning_rate,
            steps=steps,
            batch_size=args.batch_size,
            geometry_weight=weight,
            seed=args.seed,
            device=device,
            head_family=family,
        )

    def score_pair(indices: np.ndarray) -> tuple[float, float]:
        return (
            macro_f1(y_validation, predictions["baseline"], indices),
            macro_f1(y_validation, predictions["candidate"], indices),
        )

    all_indices = np.arange(len(y_validation))
    baseline_score, candidate_score = score_pair(all_indices)
    bootstrap = paired_cluster_bootstrap(
        participant_validation,
        score_pair,
        lower_is_better=False,
        replicates=args.replicates,
        confidence=0.95,
        seed=args.seed,
        minimum_clusters=5,
    )
    summary = {
        "format_version": 1,
        "split": "validation",
        "selection_split": "train_only",
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "seed": args.seed,
        "participants": int(len(np.unique(participant_validation))),
        "samples": int(len(y_validation)),
        "baseline_head": {
            "family": args.baseline_head[0],
            "representation_source": args.baseline_representation_source,
            "geometry_weight": args.baseline_head[1],
            "learning_rate": args.baseline_head[2],
            "steps": args.baseline_head[3],
            "macro_f1": baseline_score,
        },
        "candidate_head": {
            "family": args.candidate_head[0],
            "representation_source": args.candidate_representation_source,
            "geometry_weight": args.candidate_head[1],
            "learning_rate": args.candidate_head[2],
            "steps": args.candidate_head[3],
            "macro_f1": candidate_score,
        },
        "participant_cluster_bootstrap_candidate_minus_baseline": {
            "estimate": bootstrap.estimate,
            "confidence_low": bootstrap.confidence_low,
            "confidence_high": bootstrap.confidence_high,
            "probability_candidate_better": bootstrap.probability_candidate_better,
            "p_value_two_sided": bootstrap.p_value_two_sided,
            "clusters": bootstrap.clusters,
            "requested_replicates": bootstrap.requested_replicates,
            "valid_replicates": bootstrap.valid_replicates,
            "eligible": bootstrap.eligible,
            "reason": bootstrap.reason,
        },
        "validation_used_for_selection": False,
        "test_used": False,
        "limitations": [
            "Only six validation participants are available.",
            "The bootstrap quantifies validation-participant uncertainty, not external generalization.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
