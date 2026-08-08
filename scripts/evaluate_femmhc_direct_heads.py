"""Evaluate trained FemMHC female-task heads without fitting on test data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)

from femmhc import MCPHASES_TASKS, McPhasesTaskHeads, McPhasesV2TaskHeads


def expected_calibration_error(
    target: np.ndarray,
    probability: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Fixed-width binary expected calibration error."""

    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for index in range(bins):
        selected = assignments == index
        if selected.any():
            error += selected.mean() * abs(target[selected].mean() - probability[selected].mean())
    return float(error)


def parse_model(value: str) -> tuple[str, Path, Path]:
    parts = value.split("|", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("model must be NAME|CHECKPOINT|EMBEDDINGS")
    return parts[0], Path(parts[1]), Path(parts[2])


def task_target(task, labels, hormones, rows):
    target = np.full(len(rows), np.nan, dtype=np.float32)
    if task.kind == "regression":
        target[:] = hormones[:, {"lh": 0, "estrogen": 1, "pdg": 2}[task.name]]
    elif task.target_offset_days == 0:
        source = labels[:, task.label_column]
        target[source >= 0] = source[source >= 0]
    else:
        lookup = {
            (row["participant_id"], row["study_interval"], int(row["day_in_study"])): int(row["sample_index"])
            for row in rows
        }
        for row in rows:
            source_index = int(row["sample_index"])
            future = lookup.get(
                (row["participant_id"], row["study_interval"], int(row["day_in_study"]) + 1)
            )
            if future is not None and labels[future, task.label_column] >= 0:
                target[source_index] = labels[future, task.label_column]
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--selection-task",
        action="append",
        help="Optional task name(s) used for validation checkpoint selection.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels = np.load(args.processed_dir / "labels.npy")
    hormones = np.load(args.processed_dir / "hormones.npy")
    with (args.processed_dir / "index.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    splits = json.loads((args.processed_dir / "participant_splits.json").read_text(encoding="utf-8"))
    split_by_user = {user: split for split, users in splits.items() for user in users}
    sample_split = np.asarray([split_by_user[row["participant_id"]] for row in rows])

    result_rows: list[dict[str, object]] = []
    for model_name, checkpoint_path, embedding_path in args.model:
        artifact = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if artifact.get("task_head_version") == "v2":
            heads = McPhasesV2TaskHeads(
                384,
                linear_cycle_head=bool(artifact.get("linear_cycle_head", False)),
            ).eval()
        else:
            heads = McPhasesTaskHeads(384).eval()
        heads.load_state_dict(artifact["task_heads_state_dict"])
        embedding = np.load(embedding_path)
        usable = np.isfinite(embedding).all(axis=1)
        outputs: dict[str, list[np.ndarray]] = {task.name: [] for task in MCPHASES_TASKS}
        with torch.inference_mode():
            for start in range(0, len(embedding), 256):
                batch = torch.from_numpy(np.nan_to_num(embedding[start : start + 256])).float()
                predicted = heads(batch)
                for task in MCPHASES_TASKS:
                    output = predicted[task.name]
                    if task.kind == "regression":
                        outputs[task.name].append(output.numpy())
                    else:
                        outputs[task.name].append(output.probabilities.numpy())
        hormone_means = np.asarray(artifact["hormone_log_means"], dtype=np.float32)
        hormone_stds = np.asarray(artifact["hormone_log_stds"], dtype=np.float32)

        for task in MCPHASES_TASKS:
            target = task_target(task, labels, hormones, rows)
            raw = np.concatenate(outputs[task.name])
            if task.kind == "regression":
                index = {"lh": 0, "estrogen": 1, "pdg": 2}[task.name]
                continuous = np.expm1(raw * hormone_stds[index] + hormone_means[index])
                prediction = continuous
                score = None
            elif task.kind == "ordinal":
                classes = np.arange(task.classes or 2)
                continuous = raw @ classes
                prediction = np.clip(np.rint(continuous), 0, len(classes) - 1).astype(int)
                score = continuous
            else:
                prediction = raw.argmax(axis=1)
                score = raw[:, 1] if task.classes == 2 else raw

            for split in ("validation", "test"):
                mask = usable & np.isfinite(target) & (sample_split == split)
                if not bool(mask.any()):
                    continue
                y = target[mask]
                p = prediction[mask]
                if task.kind == "regression":
                    metrics = {
                        "mae": mean_absolute_error(y, p),
                        "r2": r2_score(y, p),
                        "spearman": spearmanr(y, p).statistic,
                    }
                elif task.kind == "ordinal":
                    metrics = {
                        "mae": mean_absolute_error(y, score[mask]),
                        "macro_f1": f1_score(y, p, average="macro", zero_division=0),
                        "quadratic_kappa": cohen_kappa_score(y, p, weights="quadratic"),
                    }
                elif task.classes == 2:
                    metrics = {
                        "auprc": average_precision_score(y, score[mask]),
                        "auroc": roc_auc_score(y, score[mask]),
                        "brier": brier_score_loss(y, score[mask]),
                        "ece_10": expected_calibration_error(y, score[mask], bins=10),
                        "macro_f1": f1_score(y, p, average="macro", zero_division=0),
                        "balanced_accuracy": balanced_accuracy_score(y, p),
                    }
                else:
                    metrics = {
                        "macro_f1": f1_score(y, p, average="macro", zero_division=0),
                        "balanced_accuracy": balanced_accuracy_score(y, p),
                    }
                for metric, value in metrics.items():
                    result_rows.append(
                        {
                            "model": model_name,
                            "checkpoint_step": artifact["steps"],
                            "split": split,
                            "task": task.name,
                            "task_chinese": task.chinese_name,
                            "kind": task.kind,
                            "metric": metric,
                            "value": float(value),
                            "is_primary": metric == task.primary_metric,
                            "samples": int(mask.sum()),
                        }
                    )

    frame = pd.DataFrame(result_rows)
    frame.to_csv(args.output_dir / "direct_head_results.csv", index=False, encoding="utf-8-sig")
    validation = frame[(frame["split"] == "validation") & frame["is_primary"]].copy()
    if args.selection_task:
        validation = validation[validation["task"].isin(args.selection_task)].copy()
        if validation.empty:
            raise ValueError("none of the requested --selection-task values were evaluated")
    validation["oriented"] = np.where(
        validation["kind"].eq("classification"),
        validation["value"],
        -validation["value"],
    )
    validation["task_rank"] = validation.groupby("task")["oriented"].rank(ascending=False)
    ranking = validation.groupby("model", as_index=False)["task_rank"].mean().sort_values("task_rank")
    selected = str(ranking.iloc[0]["model"])
    test = frame[(frame["split"] == "test") & frame["is_primary"] & frame["model"].eq(selected)]
    validation_tasks_used = int(validation["task"].nunique())
    summary = {
        "selection_rule": (
            "lowest mean validation-set rank across available primary metrics; "
            "the test split is not used for checkpoint selection"
        ),
        "validation_tasks_used": validation_tasks_used,
        "selected_model": selected,
        "validation_mean_ranks": dict(zip(ranking["model"], ranking["task_rank"])),
        "selected_test_primary_metrics": dict(zip(test["task_chinese"], test["value"])),
    }
    (args.output_dir / "direct_head_selection.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
