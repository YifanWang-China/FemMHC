#!/usr/bin/env python
"""Outer-LOSO constant baselines for every mcPHASES task."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from femmhc.tasks import MCPHASES_TASKS

try:
    from evaluate_femmhc_mcphases import targets_for_task
    from evaluate_mcphases_nested_loso import _all_metrics
except ModuleNotFoundError:
    from scripts.evaluate_femmhc_mcphases import targets_for_task
    from scripts.evaluate_mcphases_nested_loso import _all_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument(
        "--embedding",
        type=Path,
        action="append",
        help="Optional embeddings whose shared finite-row mask defines evaluation samples.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    labels = np.load(args.processed_dir / "labels.npy")
    hormones = np.load(args.processed_dir / "hormones.npy")
    with (args.processed_dir / "index.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    participants = np.asarray([row["participant_id"] for row in rows], dtype=str)
    common_finite = np.ones(len(rows), dtype=bool)
    for path in args.embedding or []:
        embedding = np.load(path, mmap_mode="r")
        if embedding.ndim != 2 or len(embedding) != len(rows):
            raise ValueError(f"unexpected embedding shape for {path}: {embedding.shape}")
        common_finite &= np.isfinite(embedding).all(axis=1)
    records = []
    for task in MCPHASES_TASKS:
        target = targets_for_task(task, labels, hormones, rows)
        observed = common_finite & np.isfinite(target)
        indices = np.flatnonzero(observed)
        y = target[indices].astype(
            np.float64 if task.kind == "regression" else np.int64
        )
        groups = participants[indices]
        prediction = np.empty(len(y), dtype=np.float64)
        if task.kind == "classification" and task.classes and task.classes > 2:
            score = np.empty((len(y), task.classes), dtype=np.float64)
        else:
            score = np.empty(len(y), dtype=np.float64)
        for held in np.unique(groups):
            train = groups != held
            test = groups == held
            train_target = y[train]
            if task.kind == "regression":
                value = float(np.mean(train_target))
                prediction[test] = value
                score[test] = value
            elif task.kind == "ordinal":
                value = float(np.median(train_target))
                score[test] = value
                prediction[test] = int(np.clip(round(value), 0, (task.classes or 2) - 1))
            elif task.classes == 2:
                probability = float(np.mean(train_target == 1))
                score[test] = probability
                prediction[test] = int(probability >= 0.5)
            else:
                counts = np.bincount(
                    train_target.astype(int), minlength=task.classes or 2
                ).astype(np.float64)
                probabilities = counts / counts.sum()
                score[test] = probabilities
                prediction[test] = int(np.argmax(probabilities))
        metrics = _all_metrics(
            task,
            y,
            prediction.astype(
                np.float64 if task.kind == "regression" else np.int64
            ),
            score,
        )
        primary = metrics[task.primary_metric]
        records.append(
            {
                "task": task.name,
                "task_chinese": task.chinese_name,
                "kind": task.kind,
                "primary_metric": task.primary_metric,
                "primary_value": primary,
                "samples": int(len(y)),
                "participants": int(len(np.unique(groups))),
                "metrics": metrics,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{key: value for key, value in item.items() if key != "metrics"} for item in records]
    ).to_csv(args.output_dir / "simple_baselines.csv", index=False)
    summary = {
        "format_version": 1,
        "protocol": "leave_one_participant_out_train_constant",
        "common_finite_embedding_mask": bool(args.embedding),
        "embeddings": [str(path.resolve()) for path in args.embedding or []],
        "test_used_as_independent_holdout": False,
        "tasks": records,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "tasks": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
