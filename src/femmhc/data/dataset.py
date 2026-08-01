"""Memory-mapped mcPHASES datasets with participant-safe normalization."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .mcphases import MCPHASES_CONTEXT_FEATURES, MCPHASES_LABEL_FIELDS


def _index_rows(processed_dir: Path) -> list[dict[str, str]]:
    with (processed_dir / "index.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _split_ids(processed_dir: Path, split: str) -> set[str]:
    splits = json.loads(
        (processed_dir / "participant_splits.json").read_text(encoding="utf-8")
    )
    if split not in splits:
        raise ValueError(f"unknown split {split!r}; choose from {sorted(splits)}")
    return {str(item) for item in splits[split]}


def fit_mcphases_normalization(
    processed_dir: Path,
    *,
    chunk_size: int = 256,
) -> dict[str, Any]:
    """Fit sensor/context z-score statistics on training participants only."""

    processed_dir = Path(processed_dir).resolve()
    rows = _index_rows(processed_dir)
    train_ids = _split_ids(processed_dir, "train")
    train_indices = np.asarray(
        [int(row["sample_index"]) for row in rows if row["participant_id"] in train_ids],
        dtype=np.int64,
    )
    if train_indices.size == 0:
        raise ValueError("the training split contains no samples")

    sensors = np.load(processed_dir / "sensor_values.npy", mmap_mode="r")
    sensor_sum = np.zeros(sensors.shape[1], dtype=np.float64)
    sensor_square_sum = np.zeros(sensors.shape[1], dtype=np.float64)
    sensor_count = np.zeros(sensors.shape[1], dtype=np.int64)
    for start in range(0, len(train_indices), chunk_size):
        batch = np.asarray(sensors[train_indices[start : start + chunk_size]], dtype=np.float64)
        finite = np.isfinite(batch)
        clean = np.where(finite, batch, 0.0)
        sensor_sum += clean.sum(axis=(0, 2))
        sensor_square_sum += np.square(clean).sum(axis=(0, 2))
        sensor_count += finite.sum(axis=(0, 2))
    if (sensor_count == 0).any():
        raise ValueError("at least one sensor has no observations in the training split")
    sensor_mean = sensor_sum / sensor_count
    sensor_variance = sensor_square_sum / sensor_count - np.square(sensor_mean)
    sensor_std = np.sqrt(np.maximum(sensor_variance, 1e-12))

    context = np.load(processed_dir / "daily_context.npy", mmap_mode="r")
    train_context = np.asarray(context[train_indices], dtype=np.float64)
    context_mean = np.nanmean(train_context, axis=0)
    context_std = np.nanstd(train_context, axis=0)
    context_std = np.where(context_std < 1e-6, 1.0, context_std)

    schema = json.loads((processed_dir / "schema.json").read_text(encoding="utf-8"))
    sensor_names = [item["name"] for item in schema["sensor_descriptors"]]
    statistics = {
        "format_version": 1,
        "fit_split": "train",
        "fit_participants": len(train_ids),
        "fit_samples": int(len(train_indices)),
        "sensors": {
            name: {
                "mean": float(sensor_mean[index]),
                "std": float(sensor_std[index]),
                "observations": int(sensor_count[index]),
            }
            for index, name in enumerate(sensor_names)
        },
        "daily_context": {
            name: {
                "mean": float(context_mean[index]),
                "std": float(context_std[index]),
                "observations": int(np.isfinite(train_context[:, index]).sum()),
            }
            for index, name in enumerate(MCPHASES_CONTEXT_FEATURES)
        },
    }
    (processed_dir / "normalization.json").write_text(
        json.dumps(statistics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return statistics


class McPhasesDataset(Dataset[dict[str, Any]]):
    """One day per item, read lazily from the generated NumPy memmaps."""

    def __init__(
        self,
        processed_dir: Path,
        *,
        split: str,
        normalize: bool = True,
        require_usable: bool = False,
        patch_size: int = 10,
        min_observed_fraction: float = 0.5,
    ) -> None:
        self.processed_dir = Path(processed_dir).resolve()
        self.rows = _index_rows(self.processed_dir)
        split_ids = _split_ids(self.processed_dir, split)
        self.sample_indices = [
            int(row["sample_index"])
            for row in self.rows
            if row["participant_id"] in split_ids
        ]
        self.rows_by_index = {int(row["sample_index"]): row for row in self.rows}
        self.sensors = np.load(self.processed_dir / "sensor_values.npy", mmap_mode="r")
        if require_usable:
            retained: list[int] = []
            for sample_index in self.sample_indices:
                values = self.sensors[sample_index]
                patches = np.isfinite(values).reshape(
                    values.shape[0],
                    values.shape[1] // patch_size,
                    patch_size,
                )
                if bool((patches.mean(axis=-1) >= min_observed_fraction).any()):
                    retained.append(sample_index)
            self.sample_indices = retained
        self.labels = np.load(self.processed_dir / "labels.npy", mmap_mode="r")
        self.context = np.load(self.processed_dir / "daily_context.npy", mmap_mode="r")
        self.hormones = np.load(self.processed_dir / "hormones.npy", mmap_mode="r")
        self.normalize = bool(normalize)
        if self.normalize:
            normalization_path = self.processed_dir / "normalization.json"
            if not normalization_path.is_file():
                fit_mcphases_normalization(self.processed_dir)
            statistics = json.loads(normalization_path.read_text(encoding="utf-8"))
            sensor_stats = list(statistics["sensors"].values())
            context_stats = list(statistics["daily_context"].values())
            self.sensor_mean = np.asarray([item["mean"] for item in sensor_stats], dtype=np.float32)
            self.sensor_std = np.asarray([item["std"] for item in sensor_stats], dtype=np.float32)
            self.context_mean = np.asarray([item["mean"] for item in context_stats], dtype=np.float32)
            self.context_std = np.asarray([item["std"] for item in context_stats], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        sample_index = self.sample_indices[item]
        sensor = np.array(self.sensors[sample_index], dtype=np.float32, copy=True)
        context = np.array(self.context[sample_index], dtype=np.float32, copy=True)
        if self.normalize:
            sensor = (sensor - self.sensor_mean[:, None]) / self.sensor_std[:, None]
            context = (context - self.context_mean) / self.context_std
        row = self.rows_by_index[sample_index]
        return {
            "sensor_values": torch.from_numpy(sensor),
            "channel_present": torch.from_numpy(np.isfinite(sensor).any(axis=1)),
            "labels": torch.from_numpy(
                np.array(self.labels[sample_index], dtype=np.int64, copy=True)
            ),
            "daily_context": torch.from_numpy(context),
            "hormones": torch.from_numpy(
                np.array(self.hormones[sample_index], dtype=np.float32, copy=True)
            ),
            "sample_index": sample_index,
            "participant_id": row["participant_id"],
            "study_interval": row["study_interval"],
            "day_in_study": int(row["day_in_study"]),
        }


class McPhasesTemporalPairDataset(Dataset[dict[str, Any]]):
    """Adjacent within-participant days for the trajectory-order objective."""

    def __init__(self, processed_dir: Path, *, split: str, normalize: bool = True) -> None:
        self.days = McPhasesDataset(
            processed_dir,
            split=split,
            normalize=normalize,
            require_usable=True,
        )
        positions: dict[tuple[str, str, int], int] = {}
        for position, sample_index in enumerate(self.days.sample_indices):
            row = self.days.rows_by_index[sample_index]
            positions[(row["participant_id"], row["study_interval"], int(row["day_in_study"]))] = position
        self.pairs: list[tuple[int, int]] = []
        for key, position in positions.items():
            following = positions.get((key[0], key[1], key[2] + 1))
            if following is not None:
                self.pairs.append((position, following))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, item: int) -> dict[str, Any]:
        earlier, later = self.pairs[item]
        earlier_day = self.days[earlier]
        later_day = self.days[later]
        if item % 2:
            first, second, target = later_day, earlier_day, 0.0
        else:
            first, second, target = earlier_day, later_day, 1.0
        return {
            "first": first,
            "second": second,
            "earlier": earlier_day,
            "later": later_day,
            "second_is_later": torch.tensor(target, dtype=torch.float32),
        }


__all__ = [
    "MCPHASES_CONTEXT_FEATURES",
    "MCPHASES_LABEL_FIELDS",
    "McPhasesDataset",
    "McPhasesTemporalPairDataset",
    "fit_mcphases_normalization",
]
