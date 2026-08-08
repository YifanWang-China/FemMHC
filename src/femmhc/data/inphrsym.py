"""Leakage-safe female adapter for the public inPHRsym cohort.

The release contains both raw minute streams and author-generated Feather
tables.  The merged Feather table fills missing diary entries with zero, which
would turn non-response into a false negative.  This adapter therefore uses
the raw emotion diary as the observation mask and predicts day ``t+1`` only
from wearable signals recorded on day ``t``.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from femmhc.sensors import AFFECTIVE_DAILY_SENSOR_DESCRIPTORS


SENSOR_SOURCES: tuple[str, ...] = ("steps", "heart_rate", "sleep_inbed")
SENSOR_TRANSFORMS: tuple[str, ...] = ("log1p", "identity", "identity")
TARGET_COLUMNS: tuple[str, ...] = (
    "next_anxiety_severity",
    "next_high_anxiety",
    "next_irritability_severity",
    "next_high_irritability",
    "next_negative_mood_severity",
    "next_high_negative_mood",
    "next_negative_energy_severity",
    "next_high_negative_energy",
    "next_reported_panic",
    "next_menstruation_state",
)

_ID_COLUMN = "Non-identifying keys"
_DEVICE_PRIORITY = {"Fitbit": 3, "MI-BAND": 2, "User": 1}
_EMOTION_COLUMNS = (
    "Anxiety",
    "Irritability",
    "Negative Mood",
    "Negative Energy",
)


@dataclass(frozen=True)
class InPHRSymPreparationSummary:
    female_participants_in_release: int
    female_participants_with_sensor_data: int
    participant_days: int
    date_min: str
    date_max: str
    sensor_shape: tuple[int, int, int]
    sensor_observations: dict[str, int]
    target_observations: dict[str, int]
    target_positives: dict[str, int]
    duplicate_sensor_days_resolved: dict[str, int]
    excluded_days_below_minimum_minutes: int
    split_participants: dict[str, int]
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


def _minute_vector(raw: Any, *, length: int = 1440) -> np.ndarray:
    result = np.full(length, np.nan, dtype=np.float32)
    if pd.isna(raw):
        return result
    values = np.fromstring(str(raw), sep=",", dtype=np.float32)
    if not values.size:
        return result
    values = values[:length]
    values[values == -1.0] = np.nan
    result[: values.size] = values
    return result


def _daily_vectors(
    path: Path,
    *,
    female_ids: set[str],
    value_column: str,
) -> tuple[dict[tuple[str, pd.Timestamp], np.ndarray], dict[tuple[str, pd.Timestamp], str], int]:
    frame = pd.read_excel(
        path,
        usecols=[_ID_COLUMN, "Date", "Measurement types", value_column],
    )
    frame[_ID_COLUMN] = frame[_ID_COLUMN].astype(str)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
    frame = frame.loc[
        frame[_ID_COLUMN].isin(female_ids) & frame["Date"].notna()
    ].copy()
    duplicate_count = int(frame.duplicated([_ID_COLUMN, "Date"]).sum())
    selected: dict[tuple[str, pd.Timestamp], np.ndarray] = {}
    sources: dict[tuple[str, pd.Timestamp], str] = {}
    for (participant, date), group in frame.groupby([_ID_COLUMN, "Date"], sort=True):
        best_vector: np.ndarray | None = None
        best_source = "unknown"
        best_score = (-1, -1)
        for _, item in group.iterrows():
            vector = _minute_vector(item[value_column])
            source = str(item["Measurement types"])
            score = (int(np.isfinite(vector).sum()), _DEVICE_PRIORITY.get(source, 0))
            if score > best_score:
                best_vector = vector
                best_source = source
                best_score = score
        if best_vector is not None:
            key = (str(participant), pd.Timestamp(date))
            selected[key] = best_vector
            sources[key] = best_source
    return selected, sources, duplicate_count


def _sleep_inbed_vectors(
    path: Path,
    *,
    female_ids: set[str],
) -> dict[tuple[str, pd.Timestamp], np.ndarray]:
    frame = pd.read_excel(
        path,
        usecols=[_ID_COLUMN, "Bedtime", "Wake up time"],
    )
    frame[_ID_COLUMN] = frame[_ID_COLUMN].astype(str)
    frame = frame.loc[frame[_ID_COLUMN].isin(female_ids)].copy()
    frame["Bedtime"] = pd.to_datetime(frame["Bedtime"], errors="coerce")
    frame["Wake up time"] = pd.to_datetime(frame["Wake up time"], errors="coerce")
    frame = frame.loc[frame["Bedtime"].notna() & frame["Wake up time"].notna()]
    result: dict[tuple[str, pd.Timestamp], np.ndarray] = {}
    for participant, start, stop in frame[
        [_ID_COLUMN, "Bedtime", "Wake up time"]
    ].itertuples(index=False, name=None):
        start = pd.Timestamp(start)
        stop = pd.Timestamp(stop)
        if stop <= start:
            stop += pd.Timedelta(days=1)
        # Implausible records should not label an entire multi-day interval as sleep.
        if stop - start > pd.Timedelta(hours=24):
            continue
        day = start.normalize()
        final_day = (stop - pd.Timedelta(microseconds=1)).normalize()
        while day <= final_day:
            overlap_start = max(start, day)
            overlap_stop = min(stop, day + pd.Timedelta(days=1))
            first = max(0, int(math.floor((overlap_start - day).total_seconds() / 60.0)))
            last = min(1440, int(math.ceil((overlap_stop - day).total_seconds() / 60.0)))
            key = (str(participant), day)
            values = result.setdefault(key, np.zeros(1440, dtype=np.float32))
            if last > first:
                values[first:last] = 1.0
            day += pd.Timedelta(days=1)
    return result


def _emotion_targets(
    source_dir: Path,
    *,
    female_ids: set[str],
) -> dict[tuple[str, pd.Timestamp], dict[str, float]]:
    emotion = pd.read_excel(source_dir / "Emotion Diary.xlsx")
    emotion[_ID_COLUMN] = emotion[_ID_COLUMN].astype(str)
    emotion["Date"] = pd.to_datetime(emotion["Date"], errors="coerce").dt.normalize()
    emotion = emotion.loc[
        emotion[_ID_COLUMN].isin(female_ids) & emotion["Date"].notna()
    ]
    emotion = emotion.groupby([_ID_COLUMN, "Date"], as_index=False)[
        list(_EMOTION_COLUMNS)
    ].mean()

    panic = pd.read_excel(
        source_dir / "Panic Diary.xlsx",
        usecols=[_ID_COLUMN, "Date"],
    )
    panic[_ID_COLUMN] = panic[_ID_COLUMN].astype(str)
    panic["Date"] = pd.to_datetime(panic["Date"], errors="coerce").dt.normalize()
    panic_days = {
        (str(participant), pd.Timestamp(date))
        for participant, date in panic.loc[
            panic[_ID_COLUMN].isin(female_ids) & panic["Date"].notna(),
            [_ID_COLUMN, "Date"],
        ].itertuples(index=False, name=None)
    }

    targets: dict[tuple[str, pd.Timestamp], dict[str, float]] = {}
    for participant, date, anxiety_raw, irritability_raw, mood_raw, energy_raw in emotion[
        [_ID_COLUMN, "Date", *_EMOTION_COLUMNS]
    ].itertuples(index=False, name=None):
        key = (str(participant), pd.Timestamp(date))
        anxiety = float(anxiety_raw)
        irritability = float(irritability_raw)
        negative_mood = float(-mood_raw)
        negative_energy = float(-energy_raw)
        targets[key] = {
            "next_anxiety_severity": anxiety,
            "next_high_anxiety": float(anxiety >= 2.0),
            "next_irritability_severity": irritability,
            "next_high_irritability": float(irritability >= 2.0),
            "next_negative_mood_severity": negative_mood,
            "next_high_negative_mood": float(negative_mood >= 2.0),
            "next_negative_energy_severity": negative_energy,
            "next_high_negative_energy": float(negative_energy >= 2.0),
            # A negative is defined only on a day with an explicit emotion diary.
            "next_reported_panic": float(key in panic_days),
        }
    for key in panic_days:
        targets.setdefault(key, {})["next_reported_panic"] = 1.0
    return targets


def _menstruation_targets(
    source_dir: Path,
    *,
    female_ids: set[str],
) -> dict[tuple[str, pd.Timestamp], float]:
    frame = pd.read_excel(
        source_dir / "Lifestyle - Smoking, Eating, Menstruation.xlsx",
        usecols=[_ID_COLUMN, "Date", "Menstruation"],
    )
    frame[_ID_COLUMN] = frame[_ID_COLUMN].astype(str)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()

    def parse(value: Any) -> float | None:
        normalized = str(value).strip().lower()
        if normalized in {"y", "yes", "1", "true"}:
            return 1.0
        if normalized in {"n", "no", "0", "false"}:
            return 0.0
        return None

    observations: dict[tuple[str, pd.Timestamp], list[float]] = {}
    for participant, date, value in frame.loc[
        frame[_ID_COLUMN].isin(female_ids) & frame["Date"].notna(),
        [_ID_COLUMN, "Date", "Menstruation"],
    ].itertuples(index=False, name=None):
        parsed = parse(value)
        if parsed is not None:
            observations.setdefault((str(participant), pd.Timestamp(date)), []).append(parsed)
    return {key: float(max(values)) for key, values in observations.items()}


def prepare_inphrsym(
    source_dir: Path,
    output_dir: Path,
    *,
    minimum_observed_minutes: int = 60,
    seed: int = 42,
) -> InPHRSymPreparationSummary:
    """Prepare female participant-days and strictly next-day targets."""

    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if minimum_observed_minutes <= 0:
        raise ValueError("minimum_observed_minutes must be positive")
    required = (
        "Basic research participation information.xlsx",
        "Lifestyle - Heart rate.xlsx",
        "Lifestyle - Step count.xlsx",
        "Lifestyle - Sleep.xlsx",
        "Lifestyle - Smoking, Eating, Menstruation.xlsx",
        "Emotion Diary.xlsx",
        "Panic Diary.xlsx",
    )
    if missing := [name for name in required if not (source_dir / name).is_file()]:
        raise FileNotFoundError(f"inPHRsym release is missing files: {missing}")

    demographics = pd.read_excel(
        source_dir / "Basic research participation information.xlsx",
        usecols=[_ID_COLUMN, "Gender"],
    )
    female_ids = set(
        demographics.loc[demographics["Gender"].eq("F"), _ID_COLUMN].astype(str)
    )
    if not female_ids:
        raise ValueError("the release contains no participants with Gender='F'")

    heart, heart_sources, heart_duplicates = _daily_vectors(
        source_dir / "Lifestyle - Heart rate.xlsx",
        female_ids=female_ids,
        value_column="Measure (-1: no value)",
    )
    steps, step_sources, step_duplicates = _daily_vectors(
        source_dir / "Lifestyle - Step count.xlsx",
        female_ids=female_ids,
        value_column="Measure (-1: no value)",
    )
    sleep = _sleep_inbed_vectors(
        source_dir / "Lifestyle - Sleep.xlsx",
        female_ids=female_ids,
    )
    emotion_targets = _emotion_targets(source_dir, female_ids=female_ids)
    menstruation_targets = _menstruation_targets(source_dir, female_ids=female_ids)

    all_keys = sorted(set(heart) | set(steps) | set(sleep))
    retained: list[tuple[str, pd.Timestamp]] = []
    excluded = 0
    for key in all_keys:
        observed = int(np.isfinite(heart.get(key, np.empty(0))).sum())
        observed += int(np.isfinite(steps.get(key, np.empty(0))).sum())
        if observed < minimum_observed_minutes:
            excluded += 1
            continue
        retained.append(key)
    if not retained:
        raise ValueError("no female participant-days satisfy the sensor coverage criterion")

    output_dir.mkdir(parents=True, exist_ok=True)
    values = np.lib.format.open_memmap(
        output_dir / "sensor_values.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(retained), len(SENSOR_SOURCES), 1440),
    )
    values[:] = np.nan
    index_rows: list[dict[str, Any]] = []
    target_observations = {target: 0 for target in TARGET_COLUMNS}
    target_positives = {target: 0 for target in TARGET_COLUMNS}
    for day_index, key in enumerate(retained):
        participant, date = key
        if key in steps:
            values[day_index, 0] = steps[key]
        if key in heart:
            values[day_index, 1] = heart[key]
        if key in sleep:
            values[day_index, 2] = sleep[key]
        target_date = date + pd.Timedelta(days=1)
        target_key = (participant, target_date)
        target_values = dict(emotion_targets.get(target_key, {}))
        if target_key in menstruation_targets:
            target_values["next_menstruation_state"] = menstruation_targets[target_key]
        row: dict[str, Any] = {
            "day_index": day_index,
            "participant_id": participant,
            "date": date.strftime("%Y-%m-%d"),
            "target_date": target_date.strftime("%Y-%m-%d"),
            "heart_minutes": int(np.isfinite(values[day_index, 1]).sum()),
            "step_minutes": int(np.isfinite(values[day_index, 0]).sum()),
            "sleep_available": int(key in sleep),
            "heart_source": heart_sources.get(key, ""),
            "step_source": step_sources.get(key, ""),
        }
        for target in TARGET_COLUMNS:
            value = target_values.get(target)
            row[target] = "" if value is None or not np.isfinite(value) else float(value)
            if row[target] != "":
                target_observations[target] += 1
                target_positives[target] += int(float(row[target]) > 0.0)
        index_rows.append(row)
    values.flush()
    sensor_observations = {
        descriptor.name: int(np.isfinite(values[:, channel]).sum())
        for channel, descriptor in enumerate(AFFECTIVE_DAILY_SENSOR_DESCRIPTORS)
    }
    del values

    participants = sorted({participant for participant, _ in retained})
    splits = _stable_split(participants, seed=seed)
    index_fields = (
        "day_index",
        "participant_id",
        "date",
        "target_date",
        "heart_minutes",
        "step_minutes",
        "sleep_available",
        "heart_source",
        "step_source",
        *TARGET_COLUMNS,
    )
    _write_csv(output_dir / "index.csv", index_rows, index_fields)
    (output_dir / "participant_splits.json").write_text(
        json.dumps(splits, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    schema = {
        "format_version": 1,
        "source": "inPHRsym",
        "selection": {"gender_field": "Gender", "female_value": "F"},
        "sample_grain": "participant-day",
        "input_timing": "wearable day t",
        "target_timing": "raw diary/event observation on calendar day t+1",
        "sensor_columns": [
            {
                "name": descriptor.name,
                "source": source,
                "transform": transform,
            }
            for descriptor, source, transform in zip(
                AFFECTIVE_DAILY_SENSOR_DESCRIPTORS,
                SENSOR_SOURCES,
                SENSOR_TRANSFORMS,
            )
        ],
        "targets": list(TARGET_COLUMNS),
        "missing_target": "empty CSV field; never imputed as a negative",
        "panic_negative_definition": (
            "no panic event on a target day with an observed emotion diary"
        ),
        "split_unit": "participant_id",
    }
    (output_dir / "schema.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = InPHRSymPreparationSummary(
        female_participants_in_release=len(female_ids),
        female_participants_with_sensor_data=len(participants),
        participant_days=len(retained),
        date_min=min(date for _, date in retained).strftime("%Y-%m-%d"),
        date_max=max(date for _, date in retained).strftime("%Y-%m-%d"),
        sensor_shape=(len(retained), len(SENSOR_SOURCES), 1440),
        sensor_observations=sensor_observations,
        target_observations=target_observations,
        target_positives=target_positives,
        duplicate_sensor_days_resolved={
            "heart_rate": heart_duplicates,
            "steps": step_duplicates,
        },
        excluded_days_below_minimum_minutes=excluded,
        split_participants={name: len(ids) for name, ids in splits.items()},
        output_dir=str(output_dir),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(asdict(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fit_inphrsym_normalization(output_dir)
    return summary


def fit_inphrsym_normalization(processed_dir: Path) -> dict[str, Any]:
    """Fit channel statistics on training participants only."""

    processed_dir = Path(processed_dir).resolve()
    with (processed_dir / "index.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    splits = json.loads(
        (processed_dir / "participant_splits.json").read_text(encoding="utf-8")
    )
    train_ids = set(splits["train"])
    train_indices = [
        int(row["day_index"]) for row in rows if row["participant_id"] in train_ids
    ]
    if not train_indices:
        raise ValueError("the training split contains no participant-days")
    values = np.load(processed_dir / "sensor_values.npy", mmap_mode="r")
    statistics: list[dict[str, Any]] = []
    for channel, (source, transform) in enumerate(
        zip(SENSOR_SOURCES, SENSOR_TRANSFORMS)
    ):
        observed = np.asarray(values[train_indices, channel]).reshape(-1)
        observed = observed[np.isfinite(observed)]
        if transform == "log1p":
            observed = np.log1p(np.clip(observed, 0.0, None))
        mean = float(observed.mean()) if observed.size else 0.0
        std = float(observed.std()) if observed.size else 1.0
        statistics.append(
            {
                "source": source,
                "transform": transform,
                "mean": mean,
                "std": max(std, 1e-6),
                "observations": int(observed.size),
            }
        )
    report = {
        "fit_split": "train",
        "fit_participants": len(train_ids),
        "channels": statistics,
    }
    (processed_dir / "normalization.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


class InPHRSymDailyDataset(Dataset[dict[str, Any]]):
    """Female inPHRsym participant-days for continual pretraining."""

    def __init__(
        self,
        processed_dir: Path,
        *,
        split: str,
        normalize: bool = True,
    ) -> None:
        self.processed_dir = Path(processed_dir).resolve()
        with (self.processed_dir / "index.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        splits = json.loads(
            (self.processed_dir / "participant_splits.json").read_text(
                encoding="utf-8"
            )
        )
        if split not in splits:
            raise ValueError(f"unknown split {split!r}; choose from {sorted(splits)}")
        allowed = set(splits[split])
        self.rows = [row for row in rows if row["participant_id"] in allowed]
        self.values = np.load(self.processed_dir / "sensor_values.npy", mmap_mode="r")
        self.normalize = bool(normalize)
        self.normalization = json.loads(
            (self.processed_dir / "normalization.json").read_text(encoding="utf-8")
        )

    def __len__(self) -> int:
        return len(self.rows)

    def _sensor_values(self, row: dict[str, str]) -> np.ndarray:
        values = np.array(self.values[int(row["day_index"])], copy=True)
        for channel, transform in enumerate(SENSOR_TRANSFORMS):
            observed = np.isfinite(values[channel])
            if transform == "log1p":
                values[channel, observed] = np.log1p(
                    np.clip(values[channel, observed], 0.0, None)
                )
            if self.normalize:
                stats = self.normalization["channels"][channel]
                values[channel, observed] = (
                    values[channel, observed] - stats["mean"]
                ) / stats["std"]
        return values.astype(np.float32, copy=False)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        values = self._sensor_values(row)
        return {
            "sensor_values": torch.from_numpy(values),
            "channel_present": torch.from_numpy(np.isfinite(values).any(axis=1)),
            "day_index": torch.tensor(int(row["day_index"]), dtype=torch.long),
            "participant_id": row["participant_id"],
            "date": row["date"],
        }

    def close(self) -> None:
        mmap = getattr(getattr(self, "values", None), "_mmap", None)
        if mmap is not None:
            mmap.close()

    def __del__(self) -> None:
        self.close()


class InPHRSymNextDayDataset(InPHRSymDailyDataset):
    """Observed next-day affective targets with an explicit missingness mask."""

    def __init__(
        self,
        processed_dir: Path,
        *,
        split: str,
        task: str | None = None,
        normalize: bool = True,
    ) -> None:
        if task is not None and task not in TARGET_COLUMNS:
            raise ValueError(f"unknown task {task!r}; choose from {TARGET_COLUMNS}")
        super().__init__(processed_dir, split=split, normalize=normalize)
        self.task = task
        if task is not None:
            self.rows = [row for row in self.rows if row[task] != ""]

    def __getitem__(self, index: int) -> dict[str, Any]:
        result = super().__getitem__(index)
        row = self.rows[index]
        targets = np.asarray(
            [float(row[name]) if row[name] != "" else np.nan for name in TARGET_COLUMNS],
            dtype=np.float32,
        )
        result.update(
            {
                "target_date": row["target_date"],
                "targets": torch.from_numpy(targets),
                "target_present": torch.from_numpy(np.isfinite(targets)),
            }
        )
        if self.task is not None:
            result["target"] = torch.tensor(float(row[self.task]), dtype=torch.float32)
        return result


__all__ = [
    "InPHRSymDailyDataset",
    "InPHRSymNextDayDataset",
    "InPHRSymPreparationSummary",
    "SENSOR_SOURCES",
    "TARGET_COLUMNS",
    "fit_inphrsym_normalization",
    "prepare_inphrsym",
]
