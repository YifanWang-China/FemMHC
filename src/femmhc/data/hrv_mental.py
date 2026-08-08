"""Female-only adapter for the Wearable HRV and Sleep public cohort."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from femmhc.sensors import WEARABLE_HRV_MENTAL_SENSOR_DESCRIPTORS


SENSOR_COLUMNS = ("steps", "HR", "rmssd", "light_avg")
LABEL_COLUMNS = (
    "ISI_1",
    "PHQ9_1",
    "GAD7_1",
    "ISI_2",
    "PHQ9_2",
    "GAD7_2",
    "ISI_F",
    "PHQ9_F",
    "GAD7_F",
)


@dataclass(frozen=True)
class WearableHRVMentalPreparationSummary:
    female_participants: int
    participant_days: int
    sensor_shape: tuple[int, int, int]
    label_observations: dict[str, int]
    split_participants: dict[str, int]
    excluded_days_below_minimum_windows: int
    output_dir: str


def _stable_split(
    participants: Iterable[str],
    *,
    seed: int,
    train_fraction: float = 0.68,
    validation_fraction: float = 0.16,
) -> dict[str, list[str]]:
    values = sorted(set(participants))
    random.Random(seed).shuffle(values)
    n_train = round(len(values) * train_fraction)
    n_validation = round(len(values) * validation_fraction)
    return {
        "train": sorted(values[:n_train]),
        "validation": sorted(values[n_train : n_train + n_validation]),
        "test": sorted(values[n_train + n_validation :]),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_wearable_hrv_mental(
    source_dir: Path,
    output_dir: Path,
    *,
    female_code: int = 2,
    minimum_windows_per_day: int = 6,
    seed: int = 42,
) -> WearableHRVMentalPreparationSummary:
    """Convert five-minute HRV summaries to minute grids for female participants."""

    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if minimum_windows_per_day <= 0:
        raise ValueError("minimum_windows_per_day must be positive")
    survey_path = source_dir / "survey.csv"
    sensor_path = source_dir / "sensor_hrv_filtered.csv"
    if not survey_path.is_file() or not sensor_path.is_file():
        raise FileNotFoundError("survey.csv and sensor_hrv_filtered.csv are required")

    survey = pd.read_csv(survey_path, dtype={"deviceId": str})
    required_survey = {"deviceId", "sex", *LABEL_COLUMNS}
    if missing := required_survey.difference(survey.columns):
        raise ValueError(f"survey.csv is missing columns: {sorted(missing)}")
    female = survey.loc[survey["sex"].eq(female_code)].copy()
    female["deviceId"] = female["deviceId"].astype(str)
    female_ids = sorted(female["deviceId"].unique())
    if not female_ids:
        raise ValueError(f"no participants have sex code {female_code}")

    sensor = pd.read_csv(
        sensor_path,
        usecols=["deviceId", "ts_start", *SENSOR_COLUMNS],
        dtype={"deviceId": str},
        low_memory=False,
    )
    sensor = sensor.loc[sensor["deviceId"].isin(female_ids)].copy()
    timestamp = pd.to_datetime(sensor["ts_start"], unit="ms", utc=True, errors="coerce")
    sensor = sensor.loc[timestamp.notna()].copy()
    timestamp = timestamp.loc[timestamp.notna()]
    sensor["date"] = timestamp.dt.strftime("%Y-%m-%d").to_numpy()
    sensor["minute"] = (timestamp.dt.hour * 60 + timestamp.dt.minute).to_numpy()
    sensor.sort_values(["deviceId", "date", "minute"], inplace=True)

    groups: list[tuple[str, str, pd.DataFrame]] = []
    excluded = 0
    for (participant, date), frame in sensor.groupby(["deviceId", "date"], sort=True):
        usable = frame["HR"].notna() & frame["rmssd"].notna()
        if int(usable.sum()) < minimum_windows_per_day:
            excluded += 1
            continue
        groups.append((str(participant), str(date), frame))
    if not groups:
        raise ValueError("no participant-days satisfy the minimum-window criterion")

    output_dir.mkdir(parents=True, exist_ok=True)
    values_path = output_dir / "sensor_values.npy"
    values = np.lib.format.open_memmap(
        values_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(groups), len(SENSOR_COLUMNS), 1440),
    )
    values[:] = np.nan
    index_rows: list[dict[str, Any]] = []
    for day_index, (participant, date, frame) in enumerate(groups):
        for item in frame.itertuples(index=False):
            start = int(item.minute)
            stop = min(start + 5, 1440)
            width = stop - start
            row_values = (
                getattr(item, "steps"),
                getattr(item, "HR"),
                getattr(item, "rmssd"),
                getattr(item, "light_avg"),
            )
            for channel, value in enumerate(row_values):
                if pd.isna(value):
                    continue
                minute_value = float(value) / width if channel == 0 else float(value)
                values[day_index, channel, start:stop] = minute_value
        index_rows.append(
            {
                "day_index": day_index,
                "participant_id": participant,
                "date": date,
                "valid_5min_windows": int(frame["HR"].notna().sum()),
            }
        )
    values.flush()
    del values

    label_rows = []
    for item in female.itertuples(index=False):
        row: dict[str, Any] = {"participant_id": str(item.deviceId)}
        for column in LABEL_COLUMNS:
            value = getattr(item, column)
            row[column.lower()] = "" if pd.isna(value) else float(value)
        label_rows.append(row)
    splits = _stable_split(female_ids, seed=seed)
    _write_csv(
        output_dir / "index.csv",
        index_rows,
        ("day_index", "participant_id", "date", "valid_5min_windows"),
    )
    _write_csv(
        output_dir / "labels.csv",
        label_rows,
        ("participant_id", *(column.lower() for column in LABEL_COLUMNS)),
    )
    (output_dir / "participant_splits.json").write_text(
        json.dumps(splits, indent=2), encoding="utf-8"
    )
    schema = {
        "format_version": 1,
        "source": "Wearable HRV and Sleep (Figshare article 28509740)",
        "selection": {"sex_field": "sex", "female_code": female_code},
        "timestamp_interpretation": "UTC day from Unix millisecond timestamp",
        "sensor_columns": [
            {
                "name": descriptor.name,
                "source": source,
                "transform": "log1p" if source in {"steps", "rmssd", "light_avg"} else "identity",
            }
            for descriptor, source in zip(
                WEARABLE_HRV_MENTAL_SENSOR_DESCRIPTORS, SENSOR_COLUMNS
            )
        ],
        "labels": list(LABEL_COLUMNS),
        "label_timing": {"_1": "beginning", "_2": "middle", "_F": "end"},
    }
    (output_dir / "schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )
    summary = WearableHRVMentalPreparationSummary(
        female_participants=len(female_ids),
        participant_days=len(groups),
        sensor_shape=(len(groups), len(SENSOR_COLUMNS), 1440),
        label_observations={
            column.lower(): int(female[column].notna().sum()) for column in LABEL_COLUMNS
        },
        split_participants={name: len(ids) for name, ids in splits.items()},
        excluded_days_below_minimum_windows=excluded,
        output_dir=str(output_dir),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(asdict(summary), indent=2), encoding="utf-8"
    )
    fit_wearable_hrv_mental_normalization(output_dir)
    return summary


def fit_wearable_hrv_mental_normalization(processed_dir: Path) -> dict[str, Any]:
    processed_dir = Path(processed_dir)
    with (processed_dir / "index.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    splits = json.loads(
        (processed_dir / "participant_splits.json").read_text(encoding="utf-8")
    )
    train_ids = set(splits["train"])
    train_indices = [int(row["day_index"]) for row in rows if row["participant_id"] in train_ids]
    values = np.load(processed_dir / "sensor_values.npy", mmap_mode="r")
    statistics = []
    for channel, source in enumerate(SENSOR_COLUMNS):
        observed = np.asarray(values[train_indices, channel]).reshape(-1)
        observed = observed[np.isfinite(observed)]
        if source in {"steps", "rmssd", "light_avg"}:
            observed = np.log1p(np.clip(observed, 0.0, None))
        mean = float(observed.mean())
        std = float(observed.std())
        statistics.append({"source": source, "mean": mean, "std": max(std, 1e-6)})
    report = {"fit_split": "train", "channels": statistics}
    (processed_dir / "normalization.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


class WearableHRVMentalDailyDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Minute-grid female participant-days with training-only normalization."""

    def __init__(self, processed_dir: Path, *, split: str, normalize: bool = True) -> None:
        self.processed_dir = Path(processed_dir)
        with (self.processed_dir / "index.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        splits = json.loads(
            (self.processed_dir / "participant_splits.json").read_text(encoding="utf-8")
        )
        if split not in splits:
            raise ValueError(f"unknown split: {split}")
        allowed = set(splits[split])
        self.rows = [row for row in rows if row["participant_id"] in allowed]
        self.values = np.load(self.processed_dir / "sensor_values.npy", mmap_mode="r")
        self.normalize = normalize
        self.normalization = json.loads(
            (self.processed_dir / "normalization.json").read_text(encoding="utf-8")
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        values = np.asarray(self.values[int(row["day_index"])]).copy()
        for channel, source in enumerate(SENSOR_COLUMNS):
            observed = np.isfinite(values[channel])
            if source in {"steps", "rmssd", "light_avg"}:
                values[channel, observed] = np.log1p(
                    np.clip(values[channel, observed], 0.0, None)
                )
            if self.normalize:
                stats = self.normalization["channels"][channel]
                values[channel, observed] = (
                    values[channel, observed] - stats["mean"]
                ) / stats["std"]
        return {
            "sensor_values": torch.from_numpy(values),
            "channel_present": torch.from_numpy(np.isfinite(values).any(axis=1)),
            "day_index": torch.tensor(int(row["day_index"]), dtype=torch.long),
            "participant_id": row["participant_id"],
            "date": row["date"],
        }

    def close(self) -> None:
        mmap = getattr(self.values, "_mmap", None)
        if mmap is not None:
            mmap.close()
