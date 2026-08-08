"""Leakage-safe adapter for the public pregnancy gestational-age clock data.

The raw release stores one MotionWatch ``.mtn`` XML document per longitudinal
measurement.  Each document contains minute-level wrist actigraphy and light.
This module streams members directly from the ZIP, keeps the first seven full
midnight-aligned days, and always splits by participant rather than recording.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import pickle
import random
import re
from typing import Any, Iterable
import zipfile

import numpy as np
import torch
from torch.utils.data import Dataset

from femmhc.sensors import PREGNANCY_GA_SENSOR_DESCRIPTORS


_MEASUREMENT_RE = re.compile(
    r"^(?P<participant>\d{4})\s*[_-]?\s*GA\s*[_-]?\s*(?P<ga>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_START_TIME_RE = re.compile(
    rb"<name>\s*=StartTime\s*</name>\s*<content>\s*([^<]+?)\s*</content>",
    re.IGNORECASE,
)
_CHANNEL_RE = re.compile(
    rb"<channel>\s*<name>\s*([^<]+?)\s*</name>.*?"
    rb"<data[^>]*>\s*(.*?)\s*</data>\s*</channel>",
    re.IGNORECASE | re.DOTALL,
)
_PROCESSED_KEY_RE = re.compile(
    r"^(?P<participant>\d{4})_+\s*(?P<ga>\d+(?:\.\d+)?)\s*$"
)


@dataclass(frozen=True)
class PregnancyGAPreparationSummary:
    source: str
    measurements: int
    participants: int
    measurements_per_participant: dict[str, float]
    gestational_age_min: float
    gestational_age_max: float
    days_per_measurement: int
    sensor_shape: tuple[int, int, int, int]
    sensor_observations: dict[str, int]
    excluded: dict[str, int]
    split_participants: dict[str, int]
    output_dir: str


def parse_measurement_name(name: str) -> tuple[str, float] | None:
    """Parse participant and gestational age from a public ``.mtn`` name."""

    stem = Path(name).stem.strip()
    matched = _MEASUREMENT_RE.match(stem)
    if matched is None:
        return None
    participant = matched.group("participant")
    gestational_age = float(matched.group("ga"))
    if not 0.0 < gestational_age < 45.0:
        return None
    return participant, gestational_age


def _parse_start_time(raw: bytes) -> datetime | None:
    matched = _START_TIME_RE.search(raw)
    if matched is None:
        return None
    value = matched.group(1).decode("utf-8", errors="replace").strip()
    for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(value, pattern)
        except ValueError:
            continue
    return None


def _parse_channels(raw: bytes) -> dict[str, np.ndarray]:
    channels: dict[str, np.ndarray] = {}
    for matched in _CHANNEL_RE.finditer(raw):
        name = matched.group(1).decode("utf-8", errors="replace").strip().lower()
        payload = matched.group(2).decode("ascii", errors="ignore")
        values = np.fromstring(payload, sep=",", dtype=np.float32)
        if values.size:
            channels[name] = values
    return channels


def _measurement_window(
    raw: bytes,
    *,
    days: int,
    minutes_per_day: int,
) -> tuple[np.ndarray | None, str | None]:
    start = _parse_start_time(raw)
    if start is None:
        return None, "missing_start_time"
    channels = _parse_channels(raw)
    activity = channels.get("motion")
    if activity is None:
        return None, "missing_activity"

    minute_of_day = start.hour * 60 + start.minute
    midnight_offset = (minutes_per_day - minute_of_day) % minutes_per_day
    required = midnight_offset + days * minutes_per_day
    if activity.size < required:
        return None, "short_activity"

    values = np.full((2, days, minutes_per_day), np.nan, dtype=np.float32)
    values[0] = activity[midnight_offset:required].reshape(days, minutes_per_day)
    light = channels.get("light")
    if light is not None and light.size >= required:
        values[1] = light[midnight_offset:required].reshape(days, minutes_per_day)
    return values, None


def _stable_participant_split(
    participants: Iterable[str],
    *,
    seed: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
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


def _write_index(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "measurement_index",
        "participant_id",
        "gestational_age_weeks",
        "source_member",
    )
    with (output_dir / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_pregnancy_ga_clock(
    archive_path: Path,
    output_dir: Path,
    *,
    days: int = 7,
    minutes_per_day: int = 1440,
    seed: int = 42,
) -> PregnancyGAPreparationSummary:
    """Stream the raw archive into a compact memory-mapped tensor."""

    archive_path = Path(archive_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if days <= 0 or minutes_per_day <= 0:
        raise ValueError("days and minutes_per_day must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path) as archive:
        candidates = [
            (info, parsed)
            for info in archive.infolist()
            if info.filename.lower().endswith(".mtn")
            if (parsed := parse_measurement_name(info.filename)) is not None
        ]
        if not candidates:
            raise ValueError("no gestational-age-labelled .mtn members were found")

        sensor_path = output_dir / "sensor_values.npy"
        sensor_values = np.lib.format.open_memmap(
            sensor_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(candidates), 2, days, minutes_per_day),
        )
        sensor_values[:] = np.nan
        rows: list[dict[str, Any]] = []
        excluded: dict[str, int] = {"unlabelled_filename": 0}
        all_mtn = sum(info.filename.lower().endswith(".mtn") for info in archive.infolist())
        excluded["unlabelled_filename"] = all_mtn - len(candidates)

        for info, (participant, gestational_age) in candidates:
            try:
                raw = archive.read(info)
                values, reason = _measurement_window(
                    raw,
                    days=days,
                    minutes_per_day=minutes_per_day,
                )
            except (OSError, ValueError) as error:
                values, reason = None, f"parse_error:{type(error).__name__}"
            if values is None:
                reason = reason or "unknown"
                excluded[reason] = excluded.get(reason, 0) + 1
                continue
            measurement_index = len(rows)
            sensor_values[measurement_index] = values
            rows.append(
                {
                    "measurement_index": measurement_index,
                    "participant_id": participant,
                    "gestational_age_weeks": f"{gestational_age:g}",
                    "source_member": info.filename,
                }
            )
            if len(rows) % 100 == 0:
                sensor_values.flush()
                print(
                    f"[pregnancy-ga] retained={len(rows):,} "
                    f"scanned={sum(excluded.values()) + len(rows):,}",
                    flush=True,
                )
        sensor_values.flush()
        retained_shape = (len(rows), 2, days, minutes_per_day)
        sensor_observations = {
            descriptor.name: int(
                np.isfinite(sensor_values[: len(rows), index]).sum()
            )
            for index, descriptor in enumerate(PREGNANCY_GA_SENSOR_DESCRIPTORS)
        }
        if len(rows) < len(candidates):
            trimmed_path = sensor_path.with_name("sensor_values.trim.npy")
            trimmed = np.lib.format.open_memmap(
                trimmed_path,
                mode="w+",
                dtype=np.float32,
                shape=retained_shape,
            )
            for start in range(0, len(rows), 64):
                end = min(start + 64, len(rows))
                trimmed[start:end] = sensor_values[start:end]
            trimmed.flush()
            trimmed._mmap.close()
            sensor_values._mmap.close()
            trimmed_path.replace(sensor_path)
        else:
            sensor_values._mmap.close()

    if not rows:
        raise ValueError("all labelled pregnancy measurements were unusable")
    _write_index(output_dir, rows)
    splits = _stable_participant_split(
        (str(row["participant_id"]) for row in rows),
        seed=seed,
    )
    (output_dir / "participant_splits.json").write_text(
        json.dumps(splits, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    schema = {
        "format_version": 1,
        "sensor_descriptors": [
            asdict(item) for item in PREGNANCY_GA_SENSOR_DESCRIPTORS
        ],
        "days_per_measurement": days,
        "minutes_per_day": minutes_per_day,
        "target": "gestational_age_weeks",
        "split_unit": "participant_id",
        "transform": "log1p then training-participant z-score",
        "source_values_are_log1p": False,
        "missing_sensor_value": "NaN",
    }
    (output_dir / "schema.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    ages = [float(row["gestational_age_weeks"]) for row in rows]
    participant_measurements: dict[str, int] = {}
    for row in rows:
        participant = str(row["participant_id"])
        participant_measurements[participant] = participant_measurements.get(participant, 0) + 1
    measurement_counts = np.asarray(list(participant_measurements.values()), dtype=np.float64)
    summary = PregnancyGAPreparationSummary(
        source="raw_mtn_archive",
        measurements=len(rows),
        participants=len(participant_measurements),
        measurements_per_participant={
            "minimum": float(measurement_counts.min()),
            "median": float(np.median(measurement_counts)),
            "maximum": float(measurement_counts.max()),
        },
        gestational_age_min=min(ages),
        gestational_age_max=max(ages),
        days_per_measurement=days,
        sensor_shape=retained_shape,
        sensor_observations=sensor_observations,
        excluded=excluded,
        split_participants={name: len(ids) for name, ids in splits.items()},
        output_dir=str(output_dir),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(asdict(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def prepare_pregnancy_ga_processed_pickle(
    pickle_path: Path,
    output_dir: Path,
    *,
    seed: int = 42,
) -> PregnancyGAPreparationSummary:
    """Convert the official Zenodo processed pickle to safe array artifacts.

    Pickle can execute code while loading.  This function is intended only for
    the official ``Ravindra_s2sGAclock_processed_nomd_public.pkl`` release.  The
    generated NumPy/CSV/JSON files should be used for all subsequent training.
    """

    pickle_path = Path(pickle_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not pickle_path.is_file():
        raise FileNotFoundError(pickle_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    with pickle_path.open("rb") as handle:
        source = pickle.load(handle)
    if not isinstance(source, dict):
        raise TypeError("official processed pregnancy pickle must contain a dictionary")

    parsed: list[tuple[str, str, float, dict[str, Any]]] = []
    excluded: dict[str, int] = {}
    for key, record in source.items():
        matched = _PROCESSED_KEY_RE.match(str(key))
        if matched is None:
            excluded["invalid_measurement_key"] = excluded.get(
                "invalid_measurement_key", 0
            ) + 1
            continue
        if not isinstance(record, dict):
            excluded["invalid_record"] = excluded.get("invalid_record", 0) + 1
            continue
        participant = matched.group("participant")
        gestational_age = float(matched.group("ga"))
        if not 0.0 < gestational_age < 45.0:
            excluded["invalid_gestational_age"] = excluded.get(
                "invalid_gestational_age", 0
            ) + 1
            continue
        parsed.append((str(key), participant, gestational_age, record))
    if not parsed:
        raise ValueError("no valid measurements in processed pregnancy pickle")

    days = 7
    minutes_per_day = 1440
    values = np.lib.format.open_memmap(
        output_dir / "sensor_values.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(parsed), 2, days, minutes_per_day),
    )
    values[:] = np.nan
    rows: list[dict[str, Any]] = []
    for source_key, participant, gestational_age, record in parsed:
        activity = np.asarray(record.get("activity", []), dtype=np.float32)
        light = np.asarray(record.get("light", []), dtype=np.float32)
        if activity.size < days * minutes_per_day or light.size < days * minutes_per_day:
            excluded["short_or_missing_channel"] = excluded.get(
                "short_or_missing_channel", 0
            ) + 1
            continue
        activity = activity[: days * minutes_per_day]
        light = light[: days * minutes_per_day]
        if not np.isfinite(activity).all() or not np.isfinite(light).all():
            excluded["nonfinite_channel"] = excluded.get("nonfinite_channel", 0) + 1
            continue
        measurement_index = len(rows)
        values[measurement_index, 0] = activity.reshape(days, minutes_per_day)
        values[measurement_index, 1] = light.reshape(days, minutes_per_day)
        rows.append(
            {
                "measurement_index": measurement_index,
                "participant_id": participant,
                "gestational_age_weeks": f"{gestational_age:g}",
                "source_member": source_key,
            }
        )
    values.flush()
    if len(rows) < len(parsed):
        trimmed_path = output_dir / "sensor_values.trim.npy"
        trimmed = np.lib.format.open_memmap(
            trimmed_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(rows), 2, days, minutes_per_day),
        )
        for start in range(0, len(rows), 64):
            end = min(start + 64, len(rows))
            trimmed[start:end] = values[start:end]
        trimmed.flush()
        trimmed._mmap.close()
        values._mmap.close()
        trimmed_path.replace(output_dir / "sensor_values.npy")
    else:
        values._mmap.close()
    del source

    _write_index(output_dir, rows)
    splits = _stable_participant_split(
        (str(row["participant_id"]) for row in rows), seed=seed
    )
    (output_dir / "participant_splits.json").write_text(
        json.dumps(splits, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    schema = {
        "format_version": 1,
        "sensor_descriptors": [
            asdict(item) for item in PREGNANCY_GA_SENSOR_DESCRIPTORS
        ],
        "days_per_measurement": days,
        "minutes_per_day": minutes_per_day,
        "target": "gestational_age_weeks",
        "split_unit": "participant_id",
        "transform": "source log1p then training-participant z-score",
        "source_values_are_log1p": True,
        "missing_sensor_value": "NaN",
    }
    (output_dir / "schema.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    ages = np.asarray(
        [float(row["gestational_age_weeks"]) for row in rows], dtype=np.float64
    )
    participant_measurements: dict[str, int] = {}
    for row in rows:
        participant = str(row["participant_id"])
        participant_measurements[participant] = participant_measurements.get(participant, 0) + 1
    counts = np.asarray(list(participant_measurements.values()), dtype=np.float64)
    observations_per_sensor = len(rows) * days * minutes_per_day
    summary = PregnancyGAPreparationSummary(
        source="official_processed_nomd_public_pickle",
        measurements=len(rows),
        participants=len(participant_measurements),
        measurements_per_participant={
            "minimum": float(counts.min()),
            "median": float(np.median(counts)),
            "maximum": float(counts.max()),
        },
        gestational_age_min=float(ages.min()),
        gestational_age_max=float(ages.max()),
        days_per_measurement=days,
        sensor_shape=(len(rows), 2, days, minutes_per_day),
        sensor_observations={
            descriptor.name: observations_per_sensor
            for descriptor in PREGNANCY_GA_SENSOR_DESCRIPTORS
        },
        excluded=excluded,
        split_participants={name: len(ids) for name, ids in splits.items()},
        output_dir=str(output_dir),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(asdict(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


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


def fit_pregnancy_ga_normalization(
    processed_dir: Path,
    *,
    chunk_size: int = 64,
) -> dict[str, Any]:
    """Fit log1p z-score statistics on training participants only."""

    processed_dir = Path(processed_dir).resolve()
    rows = _index_rows(processed_dir)
    train_ids = _split_ids(processed_dir, "train")
    train_indices = np.asarray(
        [
            int(row["measurement_index"])
            for row in rows
            if row["participant_id"] in train_ids
        ],
        dtype=np.int64,
    )
    if train_indices.size == 0:
        raise ValueError("the training split contains no measurements")
    values = np.load(processed_dir / "sensor_values.npy", mmap_mode="r")
    schema = json.loads((processed_dir / "schema.json").read_text(encoding="utf-8"))
    source_values_are_log1p = bool(schema.get("source_values_are_log1p", False))
    sums = np.zeros(values.shape[1], dtype=np.float64)
    square_sums = np.zeros(values.shape[1], dtype=np.float64)
    counts = np.zeros(values.shape[1], dtype=np.int64)
    for start in range(0, len(train_indices), chunk_size):
        batch = np.asarray(values[train_indices[start : start + chunk_size]], dtype=np.float64)
        if not source_values_are_log1p:
            batch = np.log1p(np.clip(batch, 0.0, None))
        finite = np.isfinite(batch)
        clean = np.where(finite, batch, 0.0)
        sums += clean.sum(axis=(0, 2, 3))
        square_sums += np.square(clean).sum(axis=(0, 2, 3))
        counts += finite.sum(axis=(0, 2, 3))
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    variances = np.divide(
        square_sums,
        counts,
        out=np.ones_like(square_sums),
        where=counts > 0,
    ) - np.square(means)
    stds = np.sqrt(np.maximum(variances, 1e-12))
    descriptors = PREGNANCY_GA_SENSOR_DESCRIPTORS
    statistics = {
        "format_version": 1,
        "fit_split": "train",
        "fit_participants": len(train_ids),
        "fit_measurements": int(len(train_indices)),
        "transform": "source_log1p" if source_values_are_log1p else "log1p",
        "sensors": {
            descriptor.name: {
                "mean": float(means[index]),
                "std": float(stds[index]),
                "observations": int(counts[index]),
            }
            for index, descriptor in enumerate(descriptors)
        },
    }
    (processed_dir / "normalization.json").write_text(
        json.dumps(statistics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return statistics


class _PregnancyGABase(Dataset[dict[str, Any]]):
    def __init__(
        self,
        processed_dir: Path,
        *,
        split: str,
        normalize: bool,
        include_light: bool,
    ) -> None:
        self.processed_dir = Path(processed_dir).resolve()
        split_ids = _split_ids(self.processed_dir, split)
        self.rows = [
            row
            for row in _index_rows(self.processed_dir)
            if row["participant_id"] in split_ids
        ]
        self.values = np.load(self.processed_dir / "sensor_values.npy", mmap_mode="r")
        schema = json.loads((self.processed_dir / "schema.json").read_text(encoding="utf-8"))
        self.days = int(schema["days_per_measurement"])
        self.source_values_are_log1p = bool(
            schema.get("source_values_are_log1p", False)
        )
        self.include_light = bool(include_light)
        self.descriptors = (
            PREGNANCY_GA_SENSOR_DESCRIPTORS
            if self.include_light
            else PREGNANCY_GA_SENSOR_DESCRIPTORS[:1]
        )
        self.normalize = bool(normalize)
        if self.normalize:
            normalization_path = self.processed_dir / "normalization.json"
            if not normalization_path.is_file():
                fit_pregnancy_ga_normalization(self.processed_dir)
            statistics = json.loads(normalization_path.read_text(encoding="utf-8"))
            sensor_stats = [statistics["sensors"][item.name] for item in self.descriptors]
            self.means = np.asarray([item["mean"] for item in sensor_stats], dtype=np.float32)
            self.stds = np.asarray([item["std"] for item in sensor_stats], dtype=np.float32)

    def _measurement(self, row: dict[str, str]) -> np.ndarray:
        index = int(row["measurement_index"])
        channels = len(self.descriptors)
        values = np.array(self.values[index, :channels], dtype=np.float32, copy=True)
        if self.normalize:
            if not self.source_values_are_log1p:
                values = np.log1p(np.clip(values, 0.0, None))
            values = (values - self.means[:, None, None]) / self.stds[:, None, None]
        return values

    def close(self) -> None:
        """Release the NumPy memory map, which is required on Windows."""

        memory_map = getattr(getattr(self, "values", None), "_mmap", None)
        if memory_map is not None:
            memory_map.close()

    def __del__(self) -> None:
        self.close()


class PregnancyGADailyDataset(_PregnancyGABase):
    """One midnight-aligned day per item for continual pretraining."""

    def __init__(
        self,
        processed_dir: Path,
        *,
        split: str,
        normalize: bool = True,
        include_light: bool = True,
    ) -> None:
        super().__init__(
            processed_dir,
            split=split,
            normalize=normalize,
            include_light=include_light,
        )
        self.day_indices = [
            (row_index, day)
            for row_index in range(len(self.rows))
            for day in range(self.days)
        ]

    def __len__(self) -> int:
        return len(self.day_indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        row_index, day = self.day_indices[item]
        row = self.rows[row_index]
        values = self._measurement(row)[:, day]
        return {
            "sensor_values": torch.from_numpy(values),
            "channel_present": torch.from_numpy(np.isfinite(values).any(axis=1)),
            "gestational_age_weeks": torch.tensor(
                float(row["gestational_age_weeks"]), dtype=torch.float32
            ),
            "participant_id": row["participant_id"],
            "measurement_index": int(row["measurement_index"]),
            "day_index": day,
        }


class PregnancyGAWindowDataset(_PregnancyGABase):
    """Seven daily sensor sets per item for gestational-age evaluation."""

    def __init__(
        self,
        processed_dir: Path,
        *,
        split: str,
        normalize: bool = True,
        include_light: bool = True,
    ) -> None:
        super().__init__(
            processed_dir,
            split=split,
            normalize=normalize,
            include_light=include_light,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = self.rows[item]
        values = self._measurement(row).transpose(1, 0, 2)
        return {
            "sensor_values": torch.from_numpy(values.copy()),
            "channel_present": torch.from_numpy(np.isfinite(values).any(axis=2)),
            "gestational_age_weeks": torch.tensor(
                float(row["gestational_age_weeks"]), dtype=torch.float32
            ),
            "participant_id": row["participant_id"],
            "measurement_index": int(row["measurement_index"]),
        }


class PregnancyGAProgressionPairDataset(Dataset[dict[str, Any]]):
    """Within-participant pregnancy visits with a relative-time target."""

    def __init__(
        self,
        processed_dir: Path,
        *,
        split: str,
        normalize: bool = True,
        include_light: bool = True,
    ) -> None:
        self.windows = PregnancyGAWindowDataset(
            processed_dir,
            split=split,
            normalize=normalize,
            include_light=include_light,
        )
        by_participant: dict[str, list[int]] = {}
        for index, row in enumerate(self.windows.rows):
            by_participant.setdefault(row["participant_id"], []).append(index)
        chronological: list[tuple[int, int]] = []
        for indices in by_participant.values():
            ordered = sorted(
                indices,
                key=lambda item: float(
                    self.windows.rows[item]["gestational_age_weeks"]
                ),
            )
            for earlier, later in zip(ordered, ordered[1:]):
                earlier_age = float(
                    self.windows.rows[earlier]["gestational_age_weeks"]
                )
                later_age = float(self.windows.rows[later]["gestational_age_weeks"])
                if later_age > earlier_age:
                    chronological.append((earlier, later))
        self.pairs = [
            directed
            for earlier, later in chronological
            for directed in ((earlier, later, 1.0), (later, earlier, 0.0))
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, item: int) -> dict[str, Any]:
        first, second, second_is_later = self.pairs[item]
        return {
            "first": self.windows[first],
            "second": self.windows[second],
            "second_is_later": torch.tensor(second_is_later, dtype=torch.float32),
        }

    def close(self) -> None:
        self.windows.close()

    def __del__(self) -> None:
        self.close()


__all__ = [
    "PregnancyGADailyDataset",
    "PregnancyGAProgressionPairDataset",
    "PregnancyGAPreparationSummary",
    "PregnancyGAWindowDataset",
    "fit_pregnancy_ga_normalization",
    "parse_measurement_name",
    "prepare_pregnancy_ga_clock",
    "prepare_pregnancy_ga_processed_pickle",
]
