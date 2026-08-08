#!/usr/bin/env python
"""Nested participant-grouped evaluation for female affective-health probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings
import zlib

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

try:
    from evaluate_affective_embeddings import (
        DEPRESS_TASKS,
        INPHRSYM_TASKS,
        LOGISTIC_C,
        RIDGE_ALPHAS,
        _classification_metrics,
        _depress_examples,
        _inphrsym_examples,
        _read_rows,
        _regression_metrics,
    )
except ModuleNotFoundError:  # Enables importing this script in repository tests.
    from scripts.evaluate_affective_embeddings import (
        DEPRESS_TASKS,
        INPHRSYM_TASKS,
        LOGISTIC_C,
        RIDGE_ALPHAS,
        _classification_metrics,
        _depress_examples,
        _inphrsym_examples,
        _read_rows,
        _regression_metrics,
    )


def _splitter(kind: str, folds: int, seed: int):
    if kind == "classification":
        return StratifiedGroupKFold(
            n_splits=folds,
            shuffle=True,
            random_state=seed,
        )
    return GroupKFold(n_splits=folds, shuffle=True, random_state=seed)


def _fit_predict(
    features: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    *,
    kind: str,
    parameter: float,
    seed: int,
) -> np.ndarray:
    scaler = StandardScaler().fit(features[train])
    train_features = scaler.transform(features[train])
    test_features = scaler.transform(features[test])
    if kind == "regression":
        model = Ridge(alpha=parameter, solver="lsqr").fit(
            train_features,
            target[train],
        )
        return model.predict(test_features)
    train_target = target[train].astype(int)
    if np.unique(train_target).size < 2:
        raise ValueError("classification training fold contains only one class")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = LogisticRegression(
            C=parameter,
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=seed,
        ).fit(train_features, train_target)
    return model.predict_proba(test_features)[:, 1]


def _selection_score(kind: str, target: np.ndarray, prediction: np.ndarray) -> float:
    if kind == "regression":
        return -float(mean_absolute_error(target, prediction))
    observed = target.astype(int)
    if np.unique(observed).size == 2:
        return float(average_precision_score(observed, prediction))
    return -float(brier_score_loss(observed, prediction))


def _nested_predictions(
    features: np.ndarray,
    target: np.ndarray,
    participant: np.ndarray,
    *,
    kind: str,
    outer_folds: int,
    inner_folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    unique_participants = np.unique(participant)
    if unique_participants.size < outer_folds:
        raise ValueError("fewer participants than outer folds")
    prediction = np.full(target.shape, np.nan, dtype=np.float64)
    baseline = np.full(target.shape, np.nan, dtype=np.float64)
    fold_records: list[dict[str, object]] = []
    outer = _splitter(kind, outer_folds, seed)
    outer_y = target.astype(int) if kind == "classification" else target
    for fold, (development, test) in enumerate(
        outer.split(features, outer_y, groups=participant)
    ):
        development_groups = np.unique(participant[development])
        actual_inner_folds = min(inner_folds, development_groups.size)
        if actual_inner_folds < 2:
            raise ValueError("nested evaluation needs at least two inner folds")
        parameters = RIDGE_ALPHAS if kind == "regression" else LOGISTIC_C
        candidate_scores: list[tuple[float, float]] = []
        inner = _splitter(kind, actual_inner_folds, seed + fold + 1)
        inner_y = (
            target[development].astype(int)
            if kind == "classification"
            else target[development]
        )
        for parameter in parameters:
            scores = []
            for inner_train_local, inner_validation_local in inner.split(
                features[development],
                inner_y,
                groups=participant[development],
            ):
                inner_train = development[inner_train_local]
                inner_validation = development[inner_validation_local]
                if (
                    kind == "classification"
                    and np.unique(target[inner_train].astype(int)).size < 2
                ):
                    continue
                inner_prediction = _fit_predict(
                    features,
                    target,
                    inner_train,
                    inner_validation,
                    kind=kind,
                    parameter=float(parameter),
                    seed=seed,
                )
                scores.append(
                    _selection_score(
                        kind,
                        target[inner_validation],
                        inner_prediction,
                    )
                )
            if scores:
                candidate_scores.append((float(np.mean(scores)), float(parameter)))
        if not candidate_scores:
            raise ValueError(f"outer fold {fold} has no valid inner model")
        selection_score, selected_parameter = max(candidate_scores)
        prediction[test] = _fit_predict(
            features,
            target,
            development,
            test,
            kind=kind,
            parameter=selected_parameter,
            seed=seed,
        )
        if kind == "regression":
            baseline[test] = float(np.median(target[development]))
        else:
            baseline[test] = float(np.mean(target[development]))
        fold_records.append(
            {
                "outer_fold": fold,
                "development_participants": int(development_groups.size),
                "test_participants": int(np.unique(participant[test]).size),
                "development_samples": int(development.size),
                "test_samples": int(test.size),
                "selected_parameter": selected_parameter,
                "inner_selection_score": selection_score,
            }
        )
    if not np.isfinite(prediction).all() or not np.isfinite(baseline).all():
        raise RuntimeError("not every sample received an out-of-fold prediction")
    return prediction, baseline, fold_records


def _metric_bundle(
    kind: str,
    target: np.ndarray,
    prediction: np.ndarray,
    participant: np.ndarray,
) -> dict[str, float | None]:
    if kind == "classification":
        return _classification_metrics(
            target.astype(int),
            prediction,
            participant,
        )
    return _regression_metrics(target, prediction, participant)


def _paired_participant_bootstrap(
    target: np.ndarray,
    participant: np.ndarray,
    native_prediction: np.ndarray,
    adapted_prediction: np.ndarray,
    *,
    kind: str,
    resamples: int,
    seed: int,
) -> dict[str, float | int]:
    groups = np.unique(participant)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(resamples):
        sampled = rng.choice(groups, size=groups.size, replace=True)
        indices = np.concatenate([np.flatnonzero(participant == group) for group in sampled])
        observed = target[indices]
        if kind == "classification":
            if np.unique(observed.astype(int)).size < 2:
                continue
            difference = average_precision_score(
                observed.astype(int), adapted_prediction[indices]
            ) - average_precision_score(observed.astype(int), native_prediction[indices])
        else:
            difference = mean_absolute_error(
                observed, native_prediction[indices]
            ) - mean_absolute_error(observed, adapted_prediction[indices])
        differences.append(float(difference))
    if not differences:
        raise ValueError("all participant bootstrap resamples were degenerate")
    values = np.asarray(differences)
    return {
        "valid_resamples": int(values.size),
        "improvement_mean": float(values.mean()),
        "improvement_ci_low": float(np.quantile(values, 0.025)),
        "improvement_ci_high": float(np.quantile(values, 0.975)),
        "probability_improved": float(np.mean(values > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("inphrsym", "depress_fitbit"), required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapted-name", default="adapted_representation")
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--minimum-history-days", type=int, default=3)
    parser.add_argument("--inphrsym-history-days", type=int, default=14)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(
        args.minimum_history_days,
        args.inphrsym_history_days,
        args.outer_folds,
        args.inner_folds,
        args.bootstrap_resamples,
    ) <= 0:
        raise ValueError("history, folds, and bootstrap resamples must be positive")

    cache = np.load(args.embeddings)
    if args.cohort == "inphrsym":
        all_tasks = INPHRSYM_TASKS
        rows = _read_rows(args.processed_dir / "index.csv")
    else:
        all_tasks = DEPRESS_TASKS
        rows = _read_rows(args.processed_dir / "assessments.csv")
    tasks = args.tasks or list(all_tasks)
    unknown = sorted(set(tasks) - set(all_tasks))
    if unknown:
        raise ValueError(f"unknown tasks for {args.cohort}: {unknown}")
    representations = {
        "native_openmhc": cache["native_embeddings"],
        args.adapted_name: cache["adapted_embeddings"],
    }

    results = []
    prediction_cache: dict[tuple[str, str], np.ndarray] = {}
    task_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for task in tasks:
        kind = all_tasks[task]
        expected_target = None
        expected_participant = None
        for representation, embeddings in representations.items():
            if args.cohort == "inphrsym":
                features, target, participant = _inphrsym_examples(
                    embeddings,
                    rows,
                    task,
                    history_days=args.inphrsym_history_days,
                    minimum_history_days=args.minimum_history_days,
                )
            else:
                features, target, participant = _depress_examples(
                    embeddings,
                    rows,
                    task,
                    minimum_history_days=args.minimum_history_days,
                )
            if expected_target is None:
                expected_target = target
                expected_participant = participant
            elif not (
                np.array_equal(expected_target, target)
                and np.array_equal(expected_participant, participant)
            ):
                raise RuntimeError("representations do not use identical evaluation samples")
            prediction, baseline, folds = _nested_predictions(
                features,
                target,
                participant,
                kind=kind,
                outer_folds=args.outer_folds,
                inner_folds=args.inner_folds,
                seed=args.seed,
            )
            prediction_cache[(task, representation)] = prediction
            task_arrays[task] = (target, participant)
            results.append(
                {
                    "representation": representation,
                    "task": task,
                    "kind": kind,
                    "samples": int(target.size),
                    "participants": int(np.unique(participant).size),
                    "constant_baseline": _metric_bundle(
                        kind, target, baseline, participant
                    ),
                    "embedding_probe": _metric_bundle(
                        kind, target, prediction, participant
                    ),
                    "folds": folds,
                }
            )

    comparisons = []
    for task in tasks:
        target, participant = task_arrays[task]
        task_seed = args.seed + zlib.crc32(task.encode("utf-8"))
        comparisons.append(
            {
                "task": task,
                "kind": all_tasks[task],
                "positive_improvement_means": (
                    "adapted AUPRC minus native AUPRC"
                    if all_tasks[task] == "classification"
                    else "native MAE minus adapted MAE"
                ),
                "participant_paired_bootstrap": _paired_participant_bootstrap(
                    target,
                    participant,
                    prediction_cache[(task, "native_openmhc")],
                    prediction_cache[(task, args.adapted_name)],
                    kind=all_tasks[task],
                    resamples=args.bootstrap_resamples,
                    seed=task_seed,
                ),
            }
        )

    report = {
        "format_version": 1,
        "cohort": args.cohort,
        "protocol": "nested participant-grouped cross-validation",
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "fold_seed": args.seed,
        "bootstrap_unit": "participant_id",
        "bootstrap_resamples": args.bootstrap_resamples,
        "history": {
            "calendar_days": (
                args.inphrsym_history_days if args.cohort == "inphrsym" else 28
            ),
            "minimum_observed_days": args.minimum_history_days,
            "aggregation": "mean+std+latest+latest-minus-first",
        },
        "representations": list(representations),
        "results": results,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
