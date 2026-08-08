from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from femmhc.data import (
    McPhasesEmbeddingHistoryDataset,
    McPhasesHistoryAdapterDataset,
    mcphases_task_targets,
)
from femmhc.data.mcphases import MCPHASES_CONTEXT_FEATURES, MCPHASES_LABEL_FIELDS


def _write_cohort(root: Path) -> Path:
    rows = [
        (0, "p1", "A", 1),
        (1, "p1", "A", 2),
        (2, "p1", "A", 4),
        (3, "p1", "B", 1),
        (4, "p1", "B", 2),
        (5, "p2", "A", 1),
        (6, "p2", "A", 2),
        (7, "p2", "A", 3),
    ]
    with (root / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["sample_index", "participant_id", "study_interval", "day_in_study"]
        )
        writer.writerows(rows)
    labels = np.full((len(rows), len(MCPHASES_LABEL_FIELDS)), -1, dtype=np.int16)
    labels[:, 0] = np.arange(len(rows)) % 4
    labels[:, 7] = np.arange(len(rows)) % 7
    np.save(root / "labels.npy", labels)
    np.save(root / "hormones.npy", np.zeros((len(rows), 3), dtype=np.float32))
    np.save(root / "sensor_values.npy", np.ones((len(rows), 6, 20), dtype=np.float32))
    np.save(root / "daily_context.npy", np.zeros((len(rows), 11), dtype=np.float32))
    (root / "normalization.json").write_text(
        json.dumps(
            {
                "sensors": {
                    str(index): {"mean": 0.0, "std": 1.0}
                    for index in range(6)
                },
                "daily_context": {
                    name: {"mean": 0.0, "std": 1.0}
                    for name in MCPHASES_CONTEXT_FEATURES
                },
            }
        ),
        encoding="utf-8",
    )
    embeddings = np.arange(len(rows) * 4, dtype=np.float32).reshape(len(rows), 4)
    path = root / "embeddings.npy"
    np.save(path, embeddings)
    (root / "participant_splits.json").write_text(
        json.dumps({"train": ["p1"], "validation": [], "test": ["p2"]}),
        encoding="utf-8",
    )
    return path


def test_next_day_target_does_not_cross_interval(tmp_path: Path) -> None:
    _write_cohort(tmp_path)
    target = mcphases_task_targets(tmp_path, "flow_volume")
    labels = np.load(tmp_path / "labels.npy")
    assert target[0] == labels[1, 7]
    assert np.isnan(target[1])  # Day 3 is absent; interval B day 1 is not future day 3.
    assert target[3] == labels[4, 7]


def test_history_window_is_dense_causal_calendar_time(tmp_path: Path) -> None:
    embeddings_path = _write_cohort(tmp_path)
    dataset = McPhasesEmbeddingHistoryDataset(
        tmp_path,
        embeddings_path,
        task="cycle_phase",
        history_days=4,
        minimum_history_days=3,
        participant_ids=["p1"],
    )
    example = next(item for item in dataset if item["day_in_study"] == 4)
    assert example["history_count"] == 3
    assert example["day_present"].tolist() == [True, True, False, True]
    source = np.load(embeddings_path)
    assert np.array_equal(example["daily_embeddings"][0].numpy(), source[0])
    assert np.array_equal(example["daily_embeddings"][1].numpy(), source[1])
    assert np.array_equal(example["daily_embeddings"][3].numpy(), source[2])


def test_raw_history_adapter_excludes_current_prediction_day(tmp_path: Path) -> None:
    embeddings_path = _write_cohort(tmp_path)
    dataset = McPhasesHistoryAdapterDataset(
        tmp_path,
        embeddings_path,
        split="train",
        history_days=4,
        minimum_history_days=0,
    )
    example = next(item for item in dataset if item["day_in_study"] == 4)
    source = np.load(embeddings_path)
    # Current day index 2 is never present in the causal controller.
    assert example["history_count"] == 2
    assert example["history_present"].tolist() == [False, True, True, False]
    assert np.array_equal(example["history_embeddings"][1].numpy(), source[0])
    assert np.array_equal(example["history_embeddings"][2].numpy(), source[1])
    assert not np.any(np.all(example["history_embeddings"].numpy() == source[2], axis=1))
