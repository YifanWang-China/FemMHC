"""Nested leave-one-participant-out evaluation for the key mcPHASES tasks.

The outer loop holds out one participant at a time. Probe regularization is
selected only inside the remaining participants with grouped cross-validation.
All reported uncertainty is paired and clustered by participant.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import NamedTuple

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from threadpoolctl import threadpool_limits

try:
    from evaluate_femmhc_mcphases import parse_embedding, targets_for_task
except ModuleNotFoundError:  # Imported as scripts.evaluate_mcphases_nested_loso in tests.
    from scripts.evaluate_femmhc_mcphases import parse_embedding, targets_for_task
from femmhc.tasks import MCPHASES_TASKS, TaskDefinition


DEFAULT_TASKS = ("cycle_phase", "flow_volume")
# This matches the frozen-probe grid used by the fixed participant split.
# A wider-grid sensitivity analysis is stored separately in the benchmark artifacts.
REGULARIZATION_GRID = (0.01, 0.1, 1.0, 10.0)


class FoldPrediction(NamedTuple):
    indices: np.ndarray
    prediction: np.ndarray
    score: np.ndarray
    selected_c: float


def _probe(task: TaskDefinition, c: float, seed: int) -> object:
    if task.kind == "regression":
        # Iterative LSQR avoids repeatedly forming a 768x768 dense factorization
        # in every inner/outer LOSO fold while optimizing the same L2 objective.
        return make_pipeline(
            StandardScaler(), Ridge(alpha=1.0 / c, solver="lsqr", tol=1e-4)
        )
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c,
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
            tol=1e-3,
        ),
    )


def _predict(
    task: TaskDefinition,
    model: object,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if task.kind == "regression":
        prediction = model.predict(x).astype(np.float64)
        return prediction, prediction
    probabilities = model.predict_proba(x)
    classes = model[-1].classes_
    if task.kind == "ordinal":
        continuous = probabilities @ classes.astype(np.float64)
        hard = np.clip(
            np.rint(continuous), 0, (task.classes or 2) - 1
        ).astype(np.int64)
        return hard, continuous
    hard = model.predict(x).astype(np.int64)
    if task.classes == 2:
        positive = np.flatnonzero(classes == 1)
        score = probabilities[:, int(positive[0])] if len(positive) else np.zeros(len(x))
        return hard, score
    aligned = np.zeros((len(x), task.classes or probabilities.shape[1]), dtype=np.float64)
    aligned[:, classes.astype(int)] = probabilities
    return hard, aligned


def _primary_metric(
    task: TaskDefinition,
    y: np.ndarray,
    prediction: np.ndarray,
    score: np.ndarray,
) -> float:
    if task.kind in {"ordinal", "regression"}:
        return float(mean_absolute_error(y, score))
    if task.classes == 2:
        return float(average_precision_score(y, score))
    return float(
        f1_score(
            y,
            prediction,
            labels=np.arange(task.classes or 2),
            average="macro",
            zero_division=0,
        )
    )


def _select_regularization(
    task: TaskDefinition,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    inner_folds: int,
    seed: int,
) -> float:
    unique_groups = np.unique(groups)
    folds = min(inner_folds, len(unique_groups))
    if folds < 2:
        return 1.0
    splitter = GroupKFold(n_splits=folds)
    best_c = 1.0
    best_oriented = -np.inf
    for c in REGULARIZATION_GRID:
        predictions = np.full(len(y), np.nan, dtype=np.float64)
        if task.classes and task.classes > 2 and task.kind != "ordinal":
            scores = np.full((len(y), task.classes), np.nan, dtype=np.float64)
        else:
            scores = np.full(len(y), np.nan, dtype=np.float64)
        for fold_index, (train, validation) in enumerate(
            splitter.split(x, y, groups=groups)
        ):
            if task.kind != "regression" and np.unique(y[train]).size < 2:
                continue
            model = _probe(task, c, seed + fold_index)
            model.fit(x[train], y[train])
            fold_prediction, fold_score = _predict(task, model, x[validation])
            predictions[validation] = fold_prediction
            scores[validation] = fold_score
        valid = np.isfinite(predictions)
        if scores.ndim == 1:
            valid &= np.isfinite(scores)
        else:
            valid &= np.isfinite(scores).all(axis=1)
        if not valid.any():
            continue
        value = _primary_metric(
            task, y[valid], predictions[valid], scores[valid]
        )
        oriented = -value if task.kind in {"ordinal", "regression"} else value
        if oriented > best_oriented:
            best_oriented = oriented
            best_c = c
    return best_c


def _outer_fold(
    task: TaskDefinition,
    embeddings: np.ndarray,
    target: np.ndarray,
    participants: np.ndarray,
    common_mask: np.ndarray,
    held_out: str,
    *,
    inner_folds: int,
    seed: int,
) -> FoldPrediction:
    train = common_mask & (participants != held_out)
    test = common_mask & (participants == held_out)
    train_indices = np.flatnonzero(train)
    test_indices = np.flatnonzero(test)
    y_train = target[train_indices].astype(
        np.float64 if task.kind == "regression" else np.int64
    )
    if task.kind != "regression" and np.unique(y_train).size < 2:
        raise ValueError(f"outer fold {held_out} has fewer than two training classes")
    selected_c = _select_regularization(
        task,
        embeddings[train_indices],
        y_train,
        participants[train_indices],
        inner_folds=inner_folds,
        seed=seed,
    )
    model = _probe(task, selected_c, seed)
    model.fit(embeddings[train_indices], y_train)
    prediction, score = _predict(task, model, embeddings[test_indices])
    return FoldPrediction(test_indices, prediction, score, selected_c)


def nested_loso_predictions(
    task: TaskDefinition,
    embeddings: np.ndarray,
    target: np.ndarray,
    participants: np.ndarray,
    common_mask: np.ndarray,
    *,
    inner_folds: int,
    jobs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, Counter[float]]:
    held_out_participants = sorted(np.unique(participants[common_mask]))
    with threadpool_limits(limits=1):
        folds = Parallel(n_jobs=jobs, prefer="threads")(
            delayed(_outer_fold)(
                task,
                embeddings,
                target,
                participants,
                common_mask,
                held_out,
                inner_folds=inner_folds,
                seed=seed + fold_index * 101,
            )
            for fold_index, held_out in enumerate(held_out_participants)
        )
    prediction = np.full(len(target), np.nan, dtype=np.float64)
    if task.classes and task.classes > 2 and task.kind != "ordinal":
        score = np.full((len(target), task.classes), np.nan, dtype=np.float64)
    else:
        score = np.full(len(target), np.nan, dtype=np.float64)
    selected: Counter[float] = Counter()
    for fold in folds:
        prediction[fold.indices] = fold.prediction
        score[fold.indices] = fold.score
        selected[fold.selected_c] += 1
    return prediction, score, selected


def _all_metrics(
    task: TaskDefinition,
    y: np.ndarray,
    prediction: np.ndarray,
    score: np.ndarray,
) -> dict[str, float]:
    if task.kind == "regression":
        return {
            "mae": float(mean_absolute_error(y, score)),
            "rmse": float(np.sqrt(np.mean((y - score) ** 2))),
        }
    if task.kind == "ordinal":
        return {
            "mae": float(mean_absolute_error(y, score)),
            "macro_f1": float(
                f1_score(
                    y,
                    prediction,
                    labels=np.arange(task.classes or 2),
                    average="macro",
                    zero_division=0,
                )
            ),
            "quadratic_kappa": float(
                cohen_kappa_score(y, prediction, weights="quadratic")
            ),
        }
    if task.classes == 2:
        return {
            "auprc": float(average_precision_score(y, score)),
            "macro_f1": float(
                f1_score(y, prediction, average="macro", zero_division=0)
            ),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        }
    one_hot = label_binarize(y, classes=np.arange(task.classes or 2))
    return {
        "macro_f1": float(
            f1_score(
                y,
                prediction,
                labels=np.arange(task.classes or 2),
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_auprc": float(
            average_precision_score(one_hot, score, average="macro")
        ),
    }


def _paired_bootstrap(
    task: TaskDefinition,
    y: np.ndarray,
    baseline_prediction: np.ndarray,
    baseline_score: np.ndarray,
    candidate_prediction: np.ndarray,
    candidate_score: np.ndarray,
    participants: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    baseline_value = _primary_metric(
        task, y, baseline_prediction, baseline_score
    )
    candidate_value = _primary_metric(
        task, y, candidate_prediction, candidate_score
    )
    sign = -1.0 if task.kind in {"ordinal", "regression"} else 1.0
    point = (
        100.0
        * sign
        * (candidate_value - baseline_value)
        / max(abs(baseline_value), 1e-12)
    )
    unique = np.unique(participants)
    grouped_indices = [np.flatnonzero(participants == item) for item in unique]
    generator = np.random.default_rng(seed)
    improvements: list[float] = []
    for _ in range(draws):
        sampled_groups = generator.integers(0, len(unique), size=len(unique))
        sampled = np.concatenate([grouped_indices[index] for index in sampled_groups])
        sampled_y = y[sampled]
        if task.classes == 2 and np.unique(sampled_y).size < 2:
            continue
        baseline_draw = _primary_metric(
            task,
            sampled_y,
            baseline_prediction[sampled],
            baseline_score[sampled],
        )
        candidate_draw = _primary_metric(
            task,
            sampled_y,
            candidate_prediction[sampled],
            candidate_score[sampled],
        )
        improvements.append(
            100.0
            * sign
            * (candidate_draw - baseline_draw)
            / max(abs(baseline_draw), 1e-12)
        )
    sampled_improvements = np.asarray(improvements, dtype=np.float64)
    low, high = np.percentile(sampled_improvements, [2.5, 97.5])
    return {
        "baseline_value": baseline_value,
        "candidate_value": candidate_value,
        "relative_improvement_percent": point,
        "paired_bootstrap_ci_low": float(low),
        "paired_bootstrap_ci_high": float(high),
        "paired_bootstrap_probability_improved": float(
            np.mean(sampled_improvements > 0)
        ),
        "bootstrap_draws_used": int(len(sampled_improvements)),
    }


def _write_summary(
    output_dir: Path,
    comparison: pd.DataFrame,
    task_lookup: dict[str, TaskDefinition],
) -> None:
    aggregate_rows: list[dict[str, object]] = []
    for task_name, group in comparison.groupby("task", sort=False):
        task = task_lookup[task_name]
        candidate = group["candidate_value"].to_numpy(dtype=float)
        baseline = float(group["baseline_value"].iloc[0])
        improvement = group["relative_improvement_percent"].to_numpy(dtype=float)
        aggregate_rows.append(
            {
                "task": task_name,
                "task_chinese": task.chinese_name,
                "primary_metric": task.primary_metric,
                "baseline_value": baseline,
                "candidate_mean": float(candidate.mean()),
                "candidate_std": float(candidate.std(ddof=1)) if len(candidate) > 1 else 0.0,
                "mean_relative_improvement_percent": float(improvement.mean()),
                "seeds_improved": int((improvement > 0).sum()),
                "seeds_total": int(len(improvement)),
                "strict_positive_ci_seeds": int(
                    (group["paired_bootstrap_ci_low"] > 0).sum()
                ),
                "mean_bootstrap_probability_improved": float(
                    group["paired_bootstrap_probability_improved"].mean()
                ),
                "samples": int(group["samples"].iloc[0]),
                "participants": int(group["participants"].iloc[0]),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(
        output_dir / "nested_loso_three_seed_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lines = [
        "# mcPHASES 嵌套留一参与者评估",
        "",
        "外层每次留出 1 名参与者；正则化系数仅在其余参与者的分组内层交叉验证中选择。置信区间为参与者级配对 Bootstrap。",
        "",
        "| 任务 | 指标 | OpenMHC | FemMHC-Dual | 相对提升 | 改善种子 | 严格正区间种子 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"| {row.task_chinese} | {row.primary_metric} | "
            f"{row.baseline_value:.4f} | {row.candidate_mean:.4f} ± {row.candidate_std:.4f} | "
            f"{row.mean_relative_improvement_percent:+.2f}% | "
            f"{row.seeds_improved}/{row.seeds_total} | "
            f"{row.strict_positive_ci_seeds}/{row.seeds_total} |"
        )
    lines.extend(
        [
            "",
            "注：经量任务的 MAE 越低越好；周期阶段的 macro-F1 越高越好。",
            "",
        ]
    )
    (output_dir / "nested_loso_three_seed_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=parse_embedding, required=True)
    parser.add_argument("--candidate", action="append", type=parse_embedding, required=True)
    parser.add_argument("--task", action="append", choices=[task.name for task in MCPHASES_TASKS])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels = np.load(args.processed_dir / "labels.npy")
    hormones = np.load(args.processed_dir / "hormones.npy")
    with (args.processed_dir / "index.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    participants = np.asarray([row["participant_id"] for row in rows])
    task_lookup = {task.name: task for task in MCPHASES_TASKS}
    task_names = args.task or list(DEFAULT_TASKS)
    models = [args.baseline, *args.candidate]
    loaded: dict[str, np.ndarray] = {}
    for name, path in models:
        embedding = np.load(path, mmap_mode="r")
        if embedding.ndim != 2 or embedding.shape[0] != len(rows):
            raise ValueError(f"{path} has unexpected shape {embedding.shape}")
        loaded[name] = embedding
    if len(loaded) != len(models):
        raise ValueError("embedding names must be unique")

    all_finite = np.ones(len(rows), dtype=bool)
    for embedding in loaded.values():
        all_finite &= np.isfinite(embedding).all(axis=1)

    result_rows: list[dict[str, object]] = []
    prediction_records: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    hyperparameter_rows: list[dict[str, object]] = []
    task_masks: dict[str, np.ndarray] = {}
    for task_index, task_name in enumerate(task_names):
        task = task_lookup[task_name]
        target = targets_for_task(task, labels, hormones, rows)
        common_mask = all_finite & np.isfinite(target)
        task_masks[task_name] = common_mask
        y = target[common_mask].astype(
            np.float64 if task.kind == "regression" else np.int64
        )
        for model_index, (model_name, _) in enumerate(models):
            print(
                json.dumps(
                    {
                        "task": task_name,
                        "model": model_name,
                        "samples": int(common_mask.sum()),
                        "participants": int(np.unique(participants[common_mask]).size),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            prediction, score, selected = nested_loso_predictions(
                task,
                loaded[model_name],
                target,
                participants,
                common_mask,
                inner_folds=args.inner_folds,
                jobs=args.jobs,
                seed=args.seed + task_index * 10_000 + model_index * 1_000,
            )
            task_prediction = prediction[common_mask].astype(
                np.float64 if task.kind == "regression" else np.int64
            )
            metrics = _all_metrics(
                task,
                y,
                task_prediction,
                score[common_mask],
            )
            prediction_records[(model_name, task_name)] = (
                task_prediction,
                score[common_mask],
                y,
            )
            for metric_name, value in metrics.items():
                result_rows.append(
                    {
                        "model": model_name,
                        "task": task_name,
                        "task_chinese": task.chinese_name,
                        "metric": metric_name,
                        "value": value,
                        "is_primary": metric_name == task.primary_metric,
                        "samples": int(common_mask.sum()),
                        "participants": int(np.unique(participants[common_mask]).size),
                        "outer_protocol": "leave_one_participant_out",
                        "inner_folds": args.inner_folds,
                    }
                )
            for c, count in sorted(selected.items()):
                hyperparameter_rows.append(
                    {
                        "model": model_name,
                        "task": task_name,
                        "C": c,
                        "outer_folds_selected": count,
                    }
                )

    results = pd.DataFrame(result_rows)
    results.to_csv(
        args.output_dir / "nested_loso_results.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(hyperparameter_rows).to_csv(
        args.output_dir / "nested_loso_hyperparameter_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    baseline_name = args.baseline[0]
    comparison_rows: list[dict[str, object]] = []
    for task_index, task_name in enumerate(task_names):
        task = task_lookup[task_name]
        mask = task_masks[task_name]
        y = targets_for_task(task, labels, hormones, rows)[mask].astype(
            np.float64 if task.kind == "regression" else np.int64
        )
        baseline_prediction, baseline_score, _ = prediction_records[
            (baseline_name, task_name)
        ]
        for candidate_index, (candidate_name, _) in enumerate(args.candidate):
            candidate_prediction, candidate_score, _ = prediction_records[
                (candidate_name, task_name)
            ]
            bootstrap = _paired_bootstrap(
                task,
                y,
                baseline_prediction,
                baseline_score,
                candidate_prediction,
                candidate_score,
                participants[mask],
                draws=args.bootstrap_draws,
                seed=args.seed + 100_000 + task_index * 1_000 + candidate_index,
            )
            comparison_rows.append(
                {
                    "baseline": baseline_name,
                    "candidate": candidate_name,
                    "task": task_name,
                    "task_chinese": task.chinese_name,
                    "primary_metric": task.primary_metric,
                    "samples": int(mask.sum()),
                    "participants": int(np.unique(participants[mask]).size),
                    **bootstrap,
                }
            )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(
        args.output_dir / "nested_loso_paired_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _write_summary(args.output_dir, comparison, task_lookup)
    manifest = {
        "format_version": 1,
        "protocol": "nested_leave_one_participant_out",
        "outer_folds": int(np.unique(participants[all_finite]).size),
        "inner_folds": args.inner_folds,
        "regularization_grid": list(REGULARIZATION_GRID),
        "common_finite_embedding_mask": True,
        "bootstrap_unit": "participant",
        "bootstrap_draws": args.bootstrap_draws,
        "tasks": task_names,
        "models": [
            {"name": name, "embedding": str(path.resolve())} for name, path in models
        ],
        "individual_predictions_written": False,
    }
    (args.output_dir / "nested_loso_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
