#!/usr/bin/env python
"""Participant-safe probes for inPHRsym and DEPRESS affective tasks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any
import warnings

import numpy as np
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


INPHRSYM_TASKS = {
    "next_anxiety_severity": "regression",
    "next_high_anxiety": "classification",
    "next_irritability_severity": "regression",
    "next_high_irritability": "classification",
    "next_negative_mood_severity": "regression",
    "next_high_negative_mood": "classification",
    "next_negative_energy_severity": "regression",
    "next_high_negative_energy": "classification",
    "next_reported_panic": "classification",
    "next_menstruation_state": "classification",
}
DEPRESS_TASKS = {
    "cesd": "regression",
    "stai_state": "regression",
    "perceived_stress": "regression",
    "positive_affect": "regression",
    "negative_affect": "regression",
}
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
LOGISTIC_C = (0.001, 0.01, 0.1, 1.0, 10.0)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _regression_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    participant: np.ndarray,
) -> dict[str, float | None]:
    correlation = (
        np.nan
        if np.std(target) < 1e-12 or np.std(prediction) < 1e-12
        else float(spearmanr(target, prediction).statistic)
    )
    participant_mae = [
        mean_absolute_error(target[participant == item], prediction[participant == item])
        for item in np.unique(participant)
    ]
    return {
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(mean_squared_error(target, prediction) ** 0.5),
        "spearman": correlation if np.isfinite(correlation) else None,
        "participant_macro_mae": float(np.mean(participant_mae)),
    }


def _classification_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    participant: np.ndarray,
) -> dict[str, float | None]:
    prediction = probability >= 0.5
    has_both = np.unique(target).size == 2
    participant_brier = [
        brier_score_loss(target[participant == item], probability[participant == item])
        for item in np.unique(participant)
    ]
    participant_auprc = []
    for item in np.unique(participant):
        observed = participant == item
        if np.unique(target[observed]).size == 2:
            participant_auprc.append(
                average_precision_score(target[observed], probability[observed])
            )
    return {
        "auprc": float(average_precision_score(target, probability)),
        "auroc": float(roc_auc_score(target, probability)) if has_both else None,
        "balanced_accuracy": (
            float(balanced_accuracy_score(target, prediction)) if has_both else None
        ),
        "brier": float(brier_score_loss(target, probability)),
        "participant_macro_brier": float(np.mean(participant_brier)),
        "participant_macro_auprc": (
            float(np.mean(participant_auprc)) if participant_auprc else None
        ),
    }


def _inphrsym_examples(
    embeddings: np.ndarray,
    rows: list[dict[str, str]],
    task: str,
    *,
    history_days: int,
    minimum_history_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_participant: dict[str, list[tuple[np.datetime64, int]]] = {}
    for row in rows:
        by_participant.setdefault(row["participant_id"], []).append(
            (np.datetime64(row["date"], "D"), int(row["day_index"]))
        )
    for values in by_participant.values():
        values.sort()
    selected = []
    features = []
    for row in rows:
        if row[task] == "":
            continue
        current = np.datetime64(row["date"], "D")
        first = current - np.timedelta64(history_days - 1, "D")
        indices = [
            day_index
            for day, day_index in by_participant[row["participant_id"]]
            if first <= day <= current
        ]
        if len(indices) < minimum_history_days:
            continue
        history = embeddings[indices]
        features.append(
            np.concatenate(
                [history.mean(axis=0), history.std(axis=0), history[-1], history[-1] - history[0]]
            )
        )
        selected.append(row)
    target = np.asarray([float(row[task]) for row in selected], dtype=np.float64)
    participant = np.asarray([row["participant_id"] for row in selected])
    return np.asarray(features), target, participant


def _depress_examples(
    embeddings: np.ndarray,
    assessments: list[dict[str, str]],
    task: str,
    *,
    minimum_history_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [
        row
        for row in assessments
        if row[task] != "" and int(row["history_days_available"]) >= minimum_history_days
    ]
    features = []
    for row in selected:
        indices = [int(value) for value in row["history_indices"].split(";") if value]
        history = embeddings[indices]
        features.append(
            np.concatenate(
                [history.mean(axis=0), history.std(axis=0), history[-1], history[-1] - history[0]]
            )
        )
    return (
        np.asarray(features),
        np.asarray([float(row[task]) for row in selected], dtype=np.float64),
        np.asarray([row["participant_id"] for row in selected]),
    )


def _split_masks(
    participant: np.ndarray,
    splits: dict[str, list[str]],
) -> dict[str, np.ndarray]:
    return {
        name: np.isin(participant, np.asarray(ids)) for name, ids in splits.items()
    }


def _fit_regression(
    features: np.ndarray,
    target: np.ndarray,
    participant: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    scaler = StandardScaler().fit(features[masks["train"]])
    transformed = scaler.transform(features)
    candidates = []
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha, solver="lsqr").fit(
            transformed[masks["train"]], target[masks["train"]]
        )
        prediction = model.predict(transformed[masks["validation"]])
        candidates.append(
            (mean_absolute_error(target[masks["validation"]], prediction), alpha)
        )
    validation_mae, alpha = min(candidates)
    development = masks["train"] | masks["validation"]
    scaler = StandardScaler().fit(features[development])
    transformed = scaler.transform(features)
    model = Ridge(alpha=alpha, solver="lsqr").fit(
        transformed[development], target[development]
    )
    observed = masks["test"]
    prediction = model.predict(transformed[observed])
    baseline = np.full(observed.sum(), np.median(target[development]))
    return {
        "selected_alpha": float(alpha),
        "validation_mae": float(validation_mae),
        "constant_baseline": _regression_metrics(
            target[observed], baseline, participant[observed]
        ),
        "embedding_probe": _regression_metrics(
            target[observed], prediction, participant[observed]
        ),
    }


def _fit_classification(
    features: np.ndarray,
    target: np.ndarray,
    participant: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    train_target = target[masks["train"]].astype(int)
    if np.unique(train_target).size < 2:
        raise ValueError("classification training split contains only one class")
    scaler = StandardScaler().fit(features[masks["train"]])
    transformed = scaler.transform(features)
    validation_target = target[masks["validation"]].astype(int)
    candidates = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for c_value in LOGISTIC_C:
            model = LogisticRegression(
                C=c_value,
                class_weight="balanced",
                max_iter=2000,
                solver="liblinear",
                random_state=42,
            ).fit(transformed[masks["train"]], train_target)
            probability = model.predict_proba(transformed[masks["validation"]])[:, 1]
            score = (
                average_precision_score(validation_target, probability)
                if np.unique(validation_target).size == 2
                else -brier_score_loss(validation_target, probability)
            )
            candidates.append((float(score), c_value))
    validation_score, c_value = max(candidates)
    development = masks["train"] | masks["validation"]
    scaler = StandardScaler().fit(features[development])
    transformed = scaler.transform(features)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=42,
        ).fit(transformed[development], target[development].astype(int))
    observed = masks["test"]
    probability = model.predict_proba(transformed[observed])[:, 1]
    prevalence = float(target[development].mean())
    baseline = np.full(observed.sum(), prevalence)
    return {
        "selected_c": float(c_value),
        "validation_selection_score": float(validation_score),
        "development_prevalence": prevalence,
        "constant_baseline": _classification_metrics(
            target[observed].astype(int), baseline, participant[observed]
        ),
        "embedding_probe": _classification_metrics(
            target[observed].astype(int), probability, participant[observed]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("inphrsym", "depress_fitbit"), required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-history-days", type=int, default=3)
    parser.add_argument("--inphrsym-history-days", type=int, default=14)
    parser.add_argument("--adapted-name", default="adapted_representation")
    args = parser.parse_args()

    cache = np.load(args.embeddings)
    splits = json.loads(
        (args.processed_dir / "participant_splits.json").read_text(encoding="utf-8")
    )
    if args.cohort == "inphrsym":
        tasks = INPHRSYM_TASKS
        source_rows = _read_rows(args.processed_dir / "index.csv")
    else:
        tasks = DEPRESS_TASKS
        source_rows = _read_rows(args.processed_dir / "assessments.csv")

    representations = {
        "native_openmhc": "native_embeddings",
        args.adapted_name: "adapted_embeddings",
    }
    results = []
    for representation, key in representations.items():
        embeddings = cache[key]
        for task, kind in tasks.items():
            if args.cohort == "inphrsym":
                features, target, participant = _inphrsym_examples(
                    embeddings,
                    source_rows,
                    task,
                    history_days=args.inphrsym_history_days,
                    minimum_history_days=args.minimum_history_days,
                )
            else:
                features, target, participant = _depress_examples(
                    embeddings,
                    source_rows,
                    task,
                    minimum_history_days=args.minimum_history_days,
                )
            masks = _split_masks(participant, splits)
            if any(not bool(mask.any()) for mask in masks.values()):
                raise ValueError(f"{task} has an empty participant split")
            evaluation = (
                _fit_classification(features, target, participant, masks)
                if kind == "classification"
                else _fit_regression(features, target, participant, masks)
            )
            results.append(
                {
                    "representation": representation,
                    "task": task,
                    "kind": kind,
                    "samples": {name: int(mask.sum()) for name, mask in masks.items()},
                    "participants": {
                        name: int(np.unique(participant[mask]).size)
                        for name, mask in masks.items()
                    },
                    **evaluation,
                }
            )
    report = {
        "format_version": 1,
        "cohort": args.cohort,
        "split_unit": "participant_id",
        "selection_split": "validation",
        "final_fit_split": "train+validation",
        "test_split": "held-out participants",
        "history": (
            {
                "calendar_days": args.inphrsym_history_days,
                "minimum_observed_days": args.minimum_history_days,
                "aggregation": "mean+std+latest+latest-minus-first",
            }
            if args.cohort == "inphrsym"
            else {
                "calendar_days": 28,
                "minimum_observed_days": args.minimum_history_days,
                "aggregation": "mean+std+latest+latest-minus-first",
            }
        ),
        "representations": list(representations),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
