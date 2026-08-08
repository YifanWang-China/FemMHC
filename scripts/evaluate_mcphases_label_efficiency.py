#!/usr/bin/env python
"""Participant-disjoint low-label transfer curves for mcPHASES representations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from femmhc.tasks import MCPHASES_TASKS, TaskDefinition
try:
    from evaluate_femmhc_mcphases import parse_embedding, targets_for_task
    from evaluate_mcphases_representation_routing import (
        ProbePrediction,
        fit_probe,
        participant_weights,
        task_metric,
    )
except ModuleNotFoundError:  # Imported as scripts.* in unit tests.
    from scripts.evaluate_femmhc_mcphases import parse_embedding, targets_for_task
    from scripts.evaluate_mcphases_representation_routing import (
        ProbePrediction,
        fit_probe,
        participant_weights,
        task_metric,
    )


DEFAULT_FRACTIONS = (0.01, 0.05, 0.10, 0.25, 1.0)
DEFAULT_STRENGTHS = (0.01, 0.1, 1.0, 10.0)


def _kind(task: TaskDefinition) -> str:
    if task.kind == "classification":
        return "binary" if task.classes == 2 else "multiclass"
    return task.kind


def _fit_label_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    participants: np.ndarray,
    x_evaluation: np.ndarray,
    *,
    kind: str,
    strength: float,
    seed: int,
) -> ProbePrediction:
    if kind != "regression":
        return fit_probe(
            x_train,
            y_train,
            participants,
            x_evaluation,
            kind=kind,
            strength=strength,
            seed=seed,
        )
    weights = participant_weights(participants)
    model = make_pipeline(
        StandardScaler(), Ridge(alpha=strength, solver="lsqr", tol=1e-4)
    )
    model.fit(x_train, y_train, ridge__sample_weight=weights)
    return ProbePrediction(model.predict(x_evaluation).astype(np.float64))


def stratified_label_subset(
    target: np.ndarray,
    fraction: float,
    *,
    kind: str,
    seed: int,
) -> np.ndarray:
    """Select a fixed label budget; categorical budgets contain every observed class."""

    if not 0 < fraction <= 1:
        raise ValueError("label fraction must be in (0, 1]")
    count = len(target)
    if fraction == 1:
        return np.arange(count, dtype=np.int64)
    generator = np.random.default_rng(seed)
    budget = max(1, int(round(count * fraction)))
    if kind in {"binary", "multiclass", "ordinal"}:
        labels = target.astype(np.int64)
        classes = np.unique(labels)
        budget = max(budget, len(classes))
        selected = [int(generator.choice(np.flatnonzero(labels == item))) for item in classes]
        remaining = np.setdiff1d(np.arange(count), np.asarray(selected), assume_unique=False)
        if budget > len(selected):
            selected.extend(
                generator.choice(
                    remaining,
                    size=min(budget - len(selected), len(remaining)),
                    replace=False,
                ).astype(int).tolist()
            )
        return np.asarray(sorted(set(selected)), dtype=np.int64)
    budget = max(10, budget)
    return np.sort(
        generator.choice(np.arange(count), size=min(budget, count), replace=False)
    ).astype(np.int64)


def _select_strength(
    x: np.ndarray,
    target: np.ndarray,
    participants: np.ndarray,
    task: TaskDefinition,
    *,
    strengths: tuple[float, ...],
    inner_folds: int,
    seed: int,
) -> float:
    kind = _kind(task)
    groups = np.unique(participants)
    folds = min(inner_folds, len(groups))
    if folds < 2:
        return 1.0
    splitter = GroupKFold(n_splits=folds)
    best_strength = 1.0
    best_oriented = -np.inf
    for strength in strengths:
        hard = np.empty(len(target), dtype=np.float64)
        positive = np.empty(len(target), dtype=np.float64) if kind == "binary" else None
        for fold, (train, held) in enumerate(
            splitter.split(x, target, groups=participants), start=1
        ):
            prediction = _fit_label_probe(
                x[train],
                target[train],
                participants[train],
                x[held],
                kind=kind,
                strength=strength,
                seed=seed + fold,
            )
            hard[held] = prediction.hard
            if positive is not None and prediction.positive_probability is not None:
                positive[held] = prediction.positive_probability
        prediction = ProbePrediction(hard, positive)
        metric = task_metric(
            target,
            prediction,
            kind=kind,
            classes=task.classes,
        )
        if metric is None:
            continue
        oriented = -metric if kind in {"ordinal", "regression"} else metric
        if oriented > best_oriented:
            best_oriented = oriented
            best_strength = strength
    return float(best_strength)


def _evaluate_model(
    embedding: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    target: np.ndarray,
    participants: np.ndarray,
    task: TaskDefinition,
    *,
    strengths: tuple[float, ...],
    inner_folds: int,
    seed: int,
) -> tuple[float | None, float]:
    kind = _kind(task)
    selected_strength = _select_strength(
        embedding[train_indices],
        target[train_indices],
        participants[train_indices],
        task,
        strengths=strengths,
        inner_folds=inner_folds,
        seed=seed,
    )
    prediction = _fit_label_probe(
        embedding[train_indices],
        target[train_indices],
        participants[train_indices],
        embedding[validation_indices],
        kind=kind,
        strength=selected_strength,
        seed=seed,
    )
    metric = task_metric(
        target[validation_indices],
        prediction,
        kind=kind,
        classes=task.classes,
    )
    return metric, selected_strength


def _evaluate_run(
    *,
    embedding: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    target: np.ndarray,
    participants: np.ndarray,
    task: TaskDefinition,
    strengths: tuple[float, ...],
    inner_folds: int,
    seed: int,
) -> tuple[float | None, float]:
    """Evaluate one model/budget/repeat unit without nested BLAS threads."""

    with threadpool_limits(limits=1):
        return _evaluate_model(
            embedding,
            train,
            validation,
            target,
            participants,
            task,
            strengths=strengths,
            inner_folds=inner_folds,
            seed=seed,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=parse_embedding, required=True)
    parser.add_argument("--candidate", action="append", type=parse_embedding, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--task",
        action="append",
        choices=[task.name for task in MCPHASES_TASKS],
    )
    parser.add_argument("--fraction", type=float, action="append")
    parser.add_argument("--strength", type=float, action="append")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    fractions = tuple(args.fraction or DEFAULT_FRACTIONS)
    strengths = tuple(args.strength or DEFAULT_STRENGTHS)
    if args.repeats <= 0 or args.inner_folds < 2:
        raise ValueError("repeats and inner folds must be positive")
    if any(not 0 < value <= 1 for value in fractions):
        raise ValueError("fractions must lie in (0, 1]")

    labels = np.load(args.processed_dir / "labels.npy")
    hormones = np.load(args.processed_dir / "hormones.npy")
    with (args.processed_dir / "index.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    participants = np.asarray([row["participant_id"] for row in rows], dtype=str)
    splits = json.loads(
        (args.processed_dir / "participant_splits.json").read_text(encoding="utf-8")
    )
    train_participants = set(map(str, splits["train"]))
    validation_participants = set(map(str, splits["validation"]))
    train_participant_mask = np.asarray([item in train_participants for item in participants])
    validation_participant_mask = np.asarray(
        [item in validation_participants for item in participants]
    )

    models = [args.baseline, *args.candidate]
    embeddings: dict[str, np.ndarray] = {}
    for name, path in models:
        values = np.load(path, mmap_mode="r")
        if values.ndim != 2 or len(values) != len(rows):
            raise ValueError(f"unexpected embedding shape for {path}: {values.shape}")
        embeddings[name] = values
    if len(embeddings) != len(models):
        raise ValueError("model names must be unique")
    all_finite = np.ones(len(rows), dtype=bool)
    for values in embeddings.values():
        all_finite &= np.isfinite(values).all(axis=1)

    selected_tasks = [
        task
        for task in MCPHASES_TASKS
        if args.task is None or task.name in set(args.task)
    ]
    result_rows: list[dict[str, Any]] = []
    for task_index, task in enumerate(selected_tasks):
        target = targets_for_task(task, labels, hormones, rows)
        observed = all_finite & np.isfinite(target)
        train_all = np.flatnonzero(observed & train_participant_mask)
        validation = np.flatnonzero(observed & validation_participant_mask)
        if not len(train_all) or not len(validation):
            continue
        train_target = target[train_all]
        kind = _kind(task)
        run_specs: list[dict[str, Any]] = []
        for fraction_index, fraction in enumerate(fractions):
            for repeat in range(args.repeats):
                subset_local = stratified_label_subset(
                    train_target,
                    fraction,
                    kind=kind,
                    seed=args.seed + task_index * 100_000 + fraction_index * 1_000 + repeat,
                )
                train = train_all[subset_local]
                for model_index, (name, _) in enumerate(models):
                    run_specs.append(
                        {
                            "fraction": fraction,
                            "repeat": repeat,
                            "train": train,
                            "name": name,
                            "seed": (
                            args.seed
                            + task_index * 100_000
                            + fraction_index * 1_000
                            + repeat * 10
                            + model_index
                            ),
                        }
                    )
        evaluated = Parallel(n_jobs=args.jobs, prefer="threads")(
            delayed(_evaluate_run)(
                embedding=embeddings[spec["name"]],
                train=spec["train"],
                validation=validation,
                target=target,
                participants=participants,
                task=task,
                strengths=strengths,
                inner_folds=args.inner_folds,
                seed=spec["seed"],
            )
            for spec in run_specs
        )
        for spec, (metric, selected_strength) in zip(run_specs, evaluated):
            train = spec["train"]
            name = spec["name"]
            fraction = spec["fraction"]
            repeat = spec["repeat"]
            result_rows.append(
                {
                    "task": task.name,
                    "task_chinese": task.chinese_name,
                    "kind": kind,
                    "primary_metric": task.primary_metric,
                    "model": name,
                    "fraction_requested": fraction,
                    "repeat": repeat,
                    "train_labels": int(len(train)),
                    "train_labels_total": int(len(train_all)),
                    "fraction_realized": float(len(train) / len(train_all)),
                    "train_participants": int(len(np.unique(participants[train]))),
                    "validation_samples": int(len(validation)),
                    "validation_participants": int(len(np.unique(participants[validation]))),
                    "selected_strength": selected_strength,
                    "primary_value": metric,
                }
            )
        print(json.dumps({"task": task.name, "status": "complete"}), flush=True)

    frame = pd.DataFrame(result_rows)
    baseline_name = args.baseline[0]
    candidate_names = [name for name, _ in args.candidate]
    summaries: list[dict[str, Any]] = []
    for (task_name, fraction), group in frame.groupby(
        ["task", "fraction_requested"], sort=False
    ):
        task = next(item for item in MCPHASES_TASKS if item.name == task_name)
        baseline_values = group[group["model"] == baseline_name]["primary_value"].dropna().to_numpy(float)
        candidate_values = group[group["model"].isin(candidate_names)]["primary_value"].dropna().to_numpy(float)
        if not len(baseline_values) or not len(candidate_values):
            continue
        baseline_mean = float(np.mean(baseline_values))
        candidate_mean = float(np.mean(candidate_values))
        lower = _kind(task) in {"ordinal", "regression"}
        delta = baseline_mean - candidate_mean if lower else candidate_mean - baseline_mean
        summaries.append(
            {
                "task": task_name,
                "task_chinese": task.chinese_name,
                "primary_metric": task.primary_metric,
                "fraction": fraction,
                "baseline_mean": baseline_mean,
                "candidate_mean": candidate_mean,
                "candidate_sample_sd": float(np.std(candidate_values, ddof=1)),
                "oriented_delta": delta,
                "relative_improvement_percent": float(
                    100.0 * delta / max(abs(baseline_mean), 1e-6)
                ),
                "baseline_runs": int(len(baseline_values)),
                "candidate_runs": int(len(candidate_values)),
            }
        )
    summary_frame = pd.DataFrame(summaries)
    fraction_summary: list[dict[str, Any]] = []
    for fraction, group in summary_frame.groupby("fraction", sort=True):
        deltas = group["oriented_delta"].to_numpy(float)
        relative = group["relative_improvement_percent"].to_numpy(float)
        fraction_summary.append(
            {
                "fraction": fraction,
                "tasks": int(len(group)),
                "candidate_wins": int(np.count_nonzero(deltas > 0)),
                "ties": int(np.count_nonzero(np.isclose(deltas, 0.0))),
                "candidate_losses": int(np.count_nonzero(deltas < 0)),
                "relative_improvement_mean_percent": float(np.mean(relative)),
                "relative_improvement_median_percent": float(np.median(relative)),
                "relative_improvement_p25_percent": float(np.quantile(relative, 0.25)),
                "relative_improvement_p75_percent": float(np.quantile(relative, 0.75)),
            }
        )

    output = {
        "format_version": 1,
        "split": "validation",
        "selection_split": "train_only",
        "test_used": False,
        "fractions": fractions,
        "repeats": args.repeats,
        "strengths": strengths,
        "baseline": baseline_name,
        "candidates": candidate_names,
        "fraction_summary": fraction_summary,
        "limitations": [
            "Categorical low-label samples contain at least one example from every observed training class.",
            "The fixed validation split contains six participants.",
            "The sealed test split is not used.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "all_runs.csv", index=False)
    summary_frame.to_csv(args.output_dir / "per_task_summary.csv", index=False)
    pd.DataFrame(fraction_summary).to_csv(args.output_dir / "fraction_summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# mcPHASES少标签迁移",
        "",
        "> 标签子集仅来自29名训练参与者，6名验证参与者只评估；测试集未使用。",
        "",
        "| 标签比例 | 任务数 | FemMHC胜/平/负 | 相对改善中位数 | 四分位区间 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in fraction_summary:
        lines.append(
            f"| {100*row['fraction']:.0f}% | {row['tasks']} | "
            f"{row['candidate_wins']}/{row['ties']}/{row['candidate_losses']} | "
            f"{row['relative_improvement_median_percent']:+.2f}% | "
            f"[{row['relative_improvement_p25_percent']:+.2f}%, "
            f"{row['relative_improvement_p75_percent']:+.2f}%] |"
        )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output_dir": str(args.output_dir.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
