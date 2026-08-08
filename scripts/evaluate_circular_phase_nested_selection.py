#!/usr/bin/env python
"""Participant-nested weight selection for the circular cycle-phase head."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader

from femmhc import FemMHCJointModel
from femmhc.data import McPhasesJointEmbeddingDataset
from femmhc.statistics import paired_cluster_bootstrap


TASK_ID = "mcphases/cycle_phase"


def _parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be NAME=CHECKPOINT")
    name, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not name or not path.is_file():
        raise argparse.ArgumentTypeError(f"invalid candidate {value!r}")
    return name, path


def _load_model(path: Path, device: torch.device) -> FemMHCJointModel:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    model = FemMHCJointModel(
        input_dim=int(artifact["input_dim"]),
        hidden_dim=int(artifact["hidden_dim"]),
        maximum_days=int(artifact["maximum_days"]),
        architecture=str(artifact.get("architecture", "full")),
        dropout=float(artifact.get("dropout", 0.0)),
        initialization_seed=artifact.get("initialization_seed"),
        routing_initial_logit=float(artifact.get("routing_initial_logit", -4.0)),
    )
    model.load_state_dict(artifact["model_state_dict"])
    return model.to(device).eval()


def leave_one_participant_out_selection(
    participant: np.ndarray,
    target: np.ndarray,
    candidate_predictions: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Select one candidate on all non-held participants for every fold."""

    participant = np.asarray(participant)
    target = np.asarray(target)
    if participant.shape != target.shape or participant.ndim != 1:
        raise ValueError("participant and target must share one-dimensional shape")
    if not candidate_predictions:
        raise ValueError("at least one candidate is required")
    candidates = list(candidate_predictions)
    for prediction in candidate_predictions.values():
        if np.asarray(prediction).shape != target.shape:
            raise ValueError("candidate predictions must match target shape")

    selected_prediction = np.empty_like(target, dtype=np.int64)
    records: list[dict[str, Any]] = []
    for fold, held_participant in enumerate(np.unique(participant), start=1):
        held = participant == held_participant
        inner = ~held
        scores = {
            name: float(
                f1_score(
                    target[inner],
                    np.asarray(candidate_predictions[name])[inner],
                    average="macro",
                )
            )
            for name in candidates
        }
        # Dictionary order is the declared low-complexity tie breaker.
        selected = max(candidates, key=lambda name: scores[name])
        selected_prediction[held] = np.asarray(candidate_predictions[selected])[held]
        records.append(
            {
                "fold": fold,
                "held_samples": int(held.sum()),
                "selected_candidate": selected,
                "inner_scores": scores,
            }
        )
    return selected_prediction, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        type=_parse_candidate,
        required=True,
        help="NAME=CHECKPOINT in low-complexity tie-break order",
    )
    parser.add_argument("--mcphases-dir", type=Path, default=Path("processed/mcphases"))
    parser.add_argument(
        "--mcphases-embeddings",
        type=Path,
        default=Path("artifacts/embeddings/mcphases/dual-v4-seed42/femmhc-dual.npy"),
    )
    parser.add_argument("--history-days", type=int, default=60)
    parser.add_argument("--minimum-history-days", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(dict(args.candidate)) != len(args.candidate):
        raise ValueError("candidate names must be unique")

    device = torch.device(args.device)
    models = {"baseline": _load_model(args.baseline_checkpoint, device)}
    models.update({name: _load_model(path, device) for name, path in args.candidate})
    dataset = McPhasesJointEmbeddingDataset(
        args.mcphases_dir,
        args.mcphases_embeddings,
        split="validation",
        history_days=args.history_days,
        minimum_history_days=args.minimum_history_days,
    )
    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=False,
        num_workers=0,
    )
    target_parts: list[np.ndarray] = []
    participant_parts: list[np.ndarray] = []
    prediction_parts: dict[str, list[np.ndarray]] = {name: [] for name in models}
    with torch.inference_mode():
        for batch in loader:
            target = batch["targets"][TASK_ID].numpy()
            observed = np.isfinite(target) & (target >= 0)
            embeddings = batch["daily_embeddings"].to(device)
            present = batch["day_present"].to(device)
            target_parts.append(target[observed].astype(np.int64))
            participant_parts.append(
                np.asarray(batch["participant_id"], dtype=str)[observed]
            )
            for name, model in models.items():
                output = model(embeddings, present, task_ids=(TASK_ID,))
                prediction_parts[name].append(
                    output.predictions[TASK_ID]
                    .probabilities.argmax(dim=-1).cpu().numpy()[observed]
                )

    target = np.concatenate(target_parts)
    participant = np.concatenate(participant_parts)
    predictions = {
        name: np.concatenate(parts) for name, parts in prediction_parts.items()
    }
    selected, folds = leave_one_participant_out_selection(
        participant,
        target,
        {name: predictions[name] for name, _ in args.candidate},
    )
    baseline = predictions["baseline"]

    def macro_f1(values: np.ndarray) -> float:
        return float(f1_score(target[values], selected[values], average="macro"))

    def score_pair(indices: np.ndarray) -> tuple[float, float]:
        return (
            float(f1_score(target[indices], baseline[indices], average="macro")),
            float(f1_score(target[indices], selected[indices], average="macro")),
        )

    all_indices = np.arange(len(target))
    baseline_score, selected_score = score_pair(all_indices)
    bootstrap = paired_cluster_bootstrap(
        participant,
        score_pair,
        lower_is_better=False,
        replicates=args.replicates,
        confidence=0.95,
        seed=args.seed,
        minimum_clusters=5,
    )
    fixed_scores = {
        name: float(f1_score(target, predictions[name], average="macro"))
        for name, _ in args.candidate
    }
    fixed_candidate = len(args.candidate) == 1
    summary = {
        "format_version": 1,
        "split": "validation",
        "selection_protocol": (
            "fixed_training_selected_candidate"
            if fixed_candidate
            else "leave_one_validation_participant_out"
        ),
        "task_id": TASK_ID,
        "participants": int(len(np.unique(participant))),
        "samples": int(len(target)),
        "candidate_order_for_ties": [name for name, _ in args.candidate],
        "selected_candidate_counts": dict(
            Counter(record["selected_candidate"] for record in folds)
        ),
        "baseline_macro_f1": baseline_score,
        "nested_selected_macro_f1": selected_score,
        "absolute_delta": selected_score - baseline_score,
        "fixed_candidate_macro_f1": fixed_scores,
        "participant_bootstrap": {
            "estimate": bootstrap.estimate,
            "confidence_low": bootstrap.confidence_low,
            "confidence_high": bootstrap.confidence_high,
            "p_value_two_sided": bootstrap.p_value_two_sided,
            "requested_replicates": bootstrap.requested_replicates,
            "valid_replicates": bootstrap.valid_replicates,
        },
        "folds": folds,
        "limitations": [
            "The candidate models were trained only on the training split.",
            (
                "The sole candidate and its hyperparameters were fixed before validation."
                if fixed_candidate
                else "Each held participant was excluded from weight selection."
            ),
            "Only six validation participants are available; this remains development evidence.",
            "The sealed test split is not used.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
