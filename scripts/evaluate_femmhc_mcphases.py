"""Participant-safe frozen-probe benchmark for FemMHC mcPHASES tasks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

from femmhc.tasks import MCPHASES_TASKS, TaskDefinition


def parse_embedding(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise argparse.ArgumentTypeError("embedding must be NAME=PATH")
    name, path = specification.split("=", 1)
    if not name.strip():
        raise argparse.ArgumentTypeError("embedding name cannot be empty")
    return name.strip(), Path(path)


def primary_value(task: TaskDefinition, y: np.ndarray, prediction: np.ndarray, score: np.ndarray | None) -> float:
    if task.kind == "regression" or task.kind == "ordinal":
        return float(mean_absolute_error(y, prediction))
    if task.classes == 2:
        assert score is not None
        return float(average_precision_score(y, score))
    return float(f1_score(y, prediction, average="macro", zero_division=0))


def bootstrap_interval(
    task: TaskDefinition,
    y: np.ndarray,
    prediction: np.ndarray,
    score: np.ndarray | None,
    participants: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    unique = np.unique(participants)
    estimates: list[float] = []
    for _ in range(draws):
        sampled = generator.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(participants == item) for item in sampled])
        try:
            value = primary_value(
                task,
                y[indices],
                prediction[indices],
                score[indices] if score is not None else None,
            )
        except ValueError:
            continue
        if np.isfinite(value):
            estimates.append(value)
    if not estimates:
        return float("nan"), float("nan")
    return tuple(float(item) for item in np.percentile(estimates, [2.5, 97.5]))


def targets_for_task(
    task: TaskDefinition,
    labels: np.ndarray,
    hormones: np.ndarray,
    rows: list[dict[str, str]],
) -> np.ndarray:
    target = np.full(len(rows), np.nan, dtype=np.float32)
    if task.kind == "regression":
        hormone_index = {"lh": 0, "estrogen": 1, "pdg": 2}[task.name]
        target[:] = hormones[:, hormone_index]
        return target
    assert task.label_column is not None
    if task.target_offset_days == 0:
        observed = labels[:, task.label_column]
        target[observed >= 0] = observed[observed >= 0]
        return target
    lookup = {
        (row["participant_id"], row["study_interval"], int(row["day_in_study"])): int(row["sample_index"])
        for row in rows
    }
    for row in rows:
        source_index = int(row["sample_index"])
        future = lookup.get(
            (
                row["participant_id"],
                row["study_interval"],
                int(row["day_in_study"]) + task.target_offset_days,
            )
        )
        if future is not None and labels[future, task.label_column] >= 0:
            target[source_index] = labels[future, task.label_column]
    return target


def classification_probe(
    task: TaskDefinition,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    best_c = 1.0
    best_value = -np.inf
    for c in (0.01, 0.1, 1.0, 10.0):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, max_iter=2000, class_weight="balanced", random_state=42),
        )
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_validation)
        classes = model[-1].classes_
        if task.kind == "ordinal":
            continuous = probabilities @ classes.astype(np.float64)
            value = -mean_absolute_error(y_validation, continuous)
        elif task.classes == 2:
            positive = int(np.flatnonzero(classes == 1)[0])
            value = average_precision_score(y_validation, probabilities[:, positive])
        else:
            value = f1_score(y_validation, model.predict(x_validation), average="macro", zero_division=0)
        if value > best_value:
            best_value = float(value)
            best_c = c
    final = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=best_c, max_iter=2000, class_weight="balanced", random_state=42),
    )
    final.fit(np.concatenate([x_train, x_validation]), np.concatenate([y_train, y_validation]))
    probabilities = final.predict_proba(x_test)
    classes = final[-1].classes_
    if task.kind == "ordinal":
        continuous = probabilities @ classes.astype(np.float64)
        prediction = np.clip(np.rint(continuous), 0, (task.classes or 2) - 1).astype(int)
        return prediction, continuous, best_c
    prediction = final.predict(x_test)
    if task.classes == 2:
        positive = int(np.flatnonzero(classes == 1)[0])
        return prediction, probabilities[:, positive], best_c
    return prediction, probabilities, best_c


def regression_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, float]:
    best_alpha = 1.0
    best_mae = np.inf
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(x_train, y_train)
        mae = mean_absolute_error(y_validation, model.predict(x_validation))
        if mae < best_mae:
            best_mae = float(mae)
            best_alpha = alpha
    final = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha))
    final.fit(np.concatenate([x_train, x_validation]), np.concatenate([y_train, y_validation]))
    return final.predict(x_test), best_alpha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--embedding", action="append", type=parse_embedding, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels = np.load(args.processed_dir / "labels.npy")
    hormones = np.load(args.processed_dir / "hormones.npy")
    with (args.processed_dir / "index.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    splits = json.loads((args.processed_dir / "participant_splits.json").read_text(encoding="utf-8"))
    split_by_user = {user: split for split, users in splits.items() for user in users}
    participant = np.asarray([row["participant_id"] for row in rows])
    sample_split = np.asarray([split_by_user[item] for item in participant])

    results: list[dict[str, object]] = []
    for model_name, embedding_path in args.embedding:
        embeddings = np.load(embedding_path)
        if embeddings.shape != (len(rows), 384):
            raise ValueError(f"{embedding_path} has unexpected shape {embeddings.shape}")
        usable = np.isfinite(embeddings).all(axis=1)
        for task_index, task in enumerate(MCPHASES_TASKS):
            target = targets_for_task(task, labels, hormones, rows)
            masks = {
                split: usable & np.isfinite(target) & (sample_split == split)
                for split in ("train", "validation", "test")
            }
            if int(masks["train"].sum()) == 0 or int(masks["test"].sum()) == 0:
                continue
            validation_source = "official_validation_participants"
            if int(masks["validation"].sum()) == 0:
                train_users = sorted(np.unique(participant[masks["train"]]))
                if len(train_users) < 2:
                    continue
                tuning_users = set(train_users[:: max(2, len(train_users) // 3)])
                original_train = masks["train"].copy()
                masks["validation"] = original_train & np.isin(participant, list(tuning_users))
                masks["train"] = original_train & ~np.isin(participant, list(tuning_users))
                validation_source = "training_participant_holdout"
            x_train, y_train = embeddings[masks["train"]], target[masks["train"]]
            x_validation, y_validation = embeddings[masks["validation"]], target[masks["validation"]]
            x_test, y_test = embeddings[masks["test"]], target[masks["test"]]
            if task.kind == "regression":
                prediction, hyperparameter = regression_probe(
                    x_train, y_train, x_validation, y_validation, x_test
                )
                score = None
                metrics = {
                    "mae": mean_absolute_error(y_test, prediction),
                    "r2": r2_score(y_test, prediction),
                    "spearman": spearmanr(y_test, prediction).statistic,
                }
            else:
                prediction, score, hyperparameter = classification_probe(
                    task, x_train, y_train.astype(int), x_validation, y_validation.astype(int), x_test
                )
                if task.kind == "ordinal":
                    metrics = {
                        "mae": mean_absolute_error(y_test, score),
                        "macro_f1": f1_score(y_test, prediction, average="macro", zero_division=0),
                        "quadratic_kappa": cohen_kappa_score(y_test, prediction, weights="quadratic"),
                    }
                elif task.classes == 2:
                    metrics = {
                        "auprc": average_precision_score(y_test, score),
                        "auroc": roc_auc_score(y_test, score),
                        "macro_f1": f1_score(y_test, prediction, average="macro", zero_division=0),
                        "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
                    }
                else:
                    probabilities = score
                    one_hot = label_binarize(y_test.astype(int), classes=np.arange(task.classes or 2))
                    metrics = {
                        "macro_f1": f1_score(y_test, prediction, average="macro", zero_division=0),
                        "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
                        "macro_auprc": average_precision_score(one_hot, probabilities, average="macro"),
                    }
            primary = float(metrics[task.primary_metric])
            ci_low, ci_high = bootstrap_interval(
                task,
                y_test,
                prediction if task.kind != "ordinal" else score,
                score if task.kind == "classification" and task.classes == 2 else None,
                participant[masks["test"]],
                draws=args.bootstrap_draws,
                seed=args.seed + task_index,
            )
            for metric, value in metrics.items():
                results.append(
                    {
                        "model": model_name,
                        "task": task.name,
                        "task_chinese": task.chinese_name,
                        "kind": task.kind,
                        "target_offset_days": task.target_offset_days,
                        "metric": metric,
                        "value": float(value),
                        "is_primary": metric == task.primary_metric,
                        "primary_ci_low": ci_low if metric == task.primary_metric else np.nan,
                        "primary_ci_high": ci_high if metric == task.primary_metric else np.nan,
                        "train_samples": int(masks["train"].sum()),
                        "validation_samples": int(masks["validation"].sum()),
                        "test_samples": int(masks["test"].sum()),
                        "test_participants": int(np.unique(participant[masks["test"]]).size),
                        "selected_hyperparameter": hyperparameter,
                        "validation_source": validation_source,
                    }
                )
            print(json.dumps({"model": model_name, "task": task.name, "primary": primary}), flush=True)

    frame = pd.DataFrame(results)
    frame.to_csv(args.output_dir / "frozen_probe_results.csv", index=False, encoding="utf-8-sig")
    primary = frame[frame["is_primary"]].copy()
    model_order = [name for name, _ in args.embedding]
    pivot = primary.pivot(index="task_chinese", columns="model", values="value").reindex(columns=model_order)
    columns = list(pivot.columns)
    markdown = [
        "| 任务 | " + " | ".join(columns) + " |",
        "|---|" + "|".join("---:" for _ in columns) + "|",
    ]
    for task_name, row in pivot.iterrows():
        markdown.append(
            "| " + str(task_name) + " | "
            + " | ".join(f"{float(row[column]):.4f}" for column in columns)
            + " |"
        )
    lines = [
        "# FemMHC mcPHASES 冻结探测结果",
        "",
        "所有超参数只在验证参与者上选择，最终结果来自未参与训练的测试参与者。区间为参与者级 bootstrap 95% 置信区间。",
        "",
        "\n".join(markdown),
        "",
    ]
    (args.output_dir / "frozen_probe_summary.md").write_text("\n".join(lines), encoding="utf-8")
    if len(model_order) >= 2:
        baseline_name = model_order[0]
        candidate_name = model_order[-1]
        comparison = primary[primary["model"].isin([baseline_name, candidate_name])].pivot(
            index=["task", "task_chinese", "kind", "metric"],
            columns="model",
            values="value",
        ).reset_index()
        higher_is_better = comparison["kind"].eq("classification")
        raw_delta = comparison[candidate_name] - comparison[baseline_name]
        comparison["higher_is_better"] = higher_is_better
        comparison["oriented_absolute_improvement"] = np.where(
            higher_is_better,
            raw_delta,
            -raw_delta,
        )
        comparison["relative_improvement_percent"] = (
            comparison["oriented_absolute_improvement"]
            / comparison[baseline_name].abs().clip(lower=1e-12)
            * 100.0
        )
        comparison.to_csv(
            args.output_dir / "primary_improvements.csv",
            index=False,
            encoding="utf-8-sig",
        )
    print(pivot.to_string())


if __name__ == "__main__":
    main()
