"""Participant-safe probes for female sleep and mental-health scores."""

from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


TASKS = (
    ("isi_middle", "isi_2", 14),
    ("phq9_middle", "phq9_2", 14),
    ("gad7_middle", "gad7_2", 14),
    ("isi_final", "isi_f", 28),
    ("phq9_final", "phq9_f", 28),
    ("gad7_final", "gad7_f", 28),
)
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def _participant_features(
    embeddings: np.ndarray,
    rows: list[dict[str, str]],
    participants: list[str],
    *,
    window_days: int,
) -> np.ndarray:
    features = []
    for participant in participants:
        indices_and_dates = sorted(
            (
                (int(row["day_index"]), date.fromisoformat(row["date"]))
                for row in rows
                if row["participant_id"] == participant
            ),
            key=lambda item: item[1],
        )
        if not indices_and_dates:
            raise ValueError(f"participant {participant} has no wearable days")
        cutoff = indices_and_dates[0][1] + timedelta(days=window_days)
        indices = [index for index, day in indices_and_dates if day < cutoff]
        if not indices:
            indices = [indices_and_dates[0][0]]
        values = embeddings[indices]
        features.append(
            np.concatenate(
                [values.mean(axis=0), values.std(axis=0), values[0], values[-1]]
            )
        )
    return np.asarray(features)


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    correlation = (
        np.nan
        if np.std(target) < 1e-12 or np.std(prediction) < 1e-12
        else float(spearmanr(target, prediction).statistic)
    )
    return {
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(mean_squared_error(target, prediction) ** 0.5),
        "spearman": correlation if np.isfinite(correlation) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    embeddings = np.load(args.embeddings)["embeddings"]
    with (args.processed_dir / "index.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (args.processed_dir / "labels.csv").open(encoding="utf-8", newline="") as handle:
        label_rows = list(csv.DictReader(handle))
    labels = {row["participant_id"]: row for row in label_rows}
    splits = json.loads(
        (args.processed_dir / "participant_splits.json").read_text(encoding="utf-8")
    )
    participants = splits["train"] + splits["validation"] + splits["test"]
    split_masks = {
        name: np.asarray([participant in set(ids) for participant in participants])
        for name, ids in splits.items()
    }
    results = []
    for task_name, label_name, window_days in TASKS:
        features = _participant_features(
            embeddings, rows, participants, window_days=window_days
        )
        target = np.asarray([float(labels[item][label_name]) for item in participants])
        train_scaler = StandardScaler().fit(features[split_masks["train"]])
        train_features = train_scaler.transform(features)
        candidates = []
        for alpha in ALPHAS:
            probe = Ridge(alpha=alpha, solver="lsqr").fit(
                train_features[split_masks["train"]], target[split_masks["train"]]
            )
            prediction = probe.predict(train_features[split_masks["validation"]])
            candidates.append(
                (mean_absolute_error(target[split_masks["validation"]], prediction), alpha)
            )
        validation_mae, alpha = min(candidates)
        development = split_masks["train"] | split_masks["validation"]
        scaler = StandardScaler().fit(features[development])
        transformed = scaler.transform(features)
        probe = Ridge(alpha=alpha, solver="lsqr").fit(
            transformed[development], target[development]
        )
        test_target = target[split_masks["test"]]
        test_prediction = probe.predict(transformed[split_masks["test"]])
        median_prediction = np.full_like(test_target, np.median(target[development]))
        results.append(
            {
                "task": task_name,
                "label": label_name,
                "input_window_days": window_days,
                "selected_alpha": float(alpha),
                "validation_mae": float(validation_mae),
                "training_median": _metrics(test_target, median_prediction),
                "embedding_ridge": _metrics(test_target, test_prediction),
            }
        )
    report = {
        "format_version": 1,
        "female_participants": len(participants),
        "split_participants": {name: len(ids) for name, ids in splits.items()},
        "alignment": "first 14/28 calendar days approximate released middle/final study timepoints",
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
