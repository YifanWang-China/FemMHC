"""Female lifecycle adapter for minute-level NHANES wrist accelerometry."""

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

from femmhc.sensors import NHANES_FEMALE_SENSOR_DESCRIPTORS


MINUTE_COLUMNS = tuple(f"min_{minute:04d}" for minute in range(1, 1441))
KEY_COLUMNS = ("SEQN", "PAXDAYM", "PAXDAYWM")


@dataclass(frozen=True)
class NHANESFemalePreparationSummary:
    female_participants: int
    participant_days: int
    age_min: float
    age_max: float
    sensor_shape: tuple[int, int, int]
    valid_minute_fraction: float
    excluded_low_coverage_days: int
    split_participants: dict[str, int]
    output_dir: str


def _stable_split(
    participants: Iterable[str],
    *,
    seed: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict[str, list[str]]:
    values = sorted(set(participants))
    random.Random(seed).shuffle(values)
    train_count = round(len(values) * train_fraction)
    validation_count = round(len(values) * validation_fraction)
    return {
        "train": sorted(values[:train_count]),
        "validation": sorted(values[train_count : train_count + validation_count]),
        "test": sorted(values[train_count + validation_count :]),
    }


def _keys(frame: pd.DataFrame) -> np.ndarray:
    return frame.loc[:, KEY_COLUMNS].astype(str).to_numpy()


def _assert_aligned(*frames: pd.DataFrame) -> None:
    first = _keys(frames[0])
    if any(not np.array_equal(first, _keys(frame)) for frame in frames[1:]):
        raise ValueError("NHANES activity, wear-state and quality rows are not aligned")


def _flag_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = frame.loc[:, MINUTE_COLUMNS]
    missing = values.isna().to_numpy()
    flagged = values.eq(True).to_numpy() | values.eq("TRUE").to_numpy()
    return flagged, missing


def _write_index(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "sample_index",
        "participant_id",
        "measurement_day",
        "day_of_week",
        "age_years",
        "release_cycle",
        "valid_minutes",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_nhanes_female(
    source_dir: Path,
    output_dir: Path,
    *,
    minimum_age: float = 12.0,
    minimum_valid_minutes: int = 600,
    seed: int = 42,
    chunk_size: int = 128,
) -> NHANESFemalePreparationSummary:
    """Create two-channel female NHANES days using quality-safe minute masks."""

    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not 0 <= minimum_age <= 120:
        raise ValueError("minimum_age must be between 0 and 120")
    if not 1 <= minimum_valid_minutes <= 1440:
        raise ValueError("minimum_valid_minutes must be between 1 and 1440")
    csv_dir = source_dir / "csv"
    paths = {
        "activity": csv_dir / "nhanes_1440_log10PAXMTSM.csv.xz",
        "wear_state": csv_dir / "nhanes_1440_PAXPREDM.csv.xz",
        "quality": csv_dir / "nhanes_1440_PAXFLGSM.csv.xz",
    }
    if not (source_dir / "subject-info.csv").is_file() or not all(
        path.is_file() for path in paths.values()
    ):
        raise FileNotFoundError("NHANES subject info and three minute-level CSV.XZ files are required")

    demographics = pd.read_csv(
        source_dir / "subject-info.csv", dtype={"SEQN": str}
    )
    required = {
        "SEQN",
        "gender",
        "age_in_years_at_screening",
        "data_release_cycle",
    }
    if missing := required.difference(demographics.columns):
        raise ValueError(f"subject-info.csv is missing columns: {sorted(missing)}")
    female = demographics.loc[
        demographics["gender"].eq("Female")
        & demographics["age_in_years_at_screening"].ge(minimum_age)
    ].copy()
    female["SEQN"] = female["SEQN"].astype(str)
    metadata = female.set_index("SEQN").to_dict(orient="index")
    female_ids = set(metadata)
    if not female_ids:
        raise ValueError("no female participants satisfy the age threshold")

    candidate_rows = pd.read_csv(
        paths["activity"],
        usecols=list(KEY_COLUMNS),
        dtype={"SEQN": str},
    )
    candidate_rows = candidate_rows.loc[
        candidate_rows["SEQN"].isin(female_ids)
        & candidate_rows["PAXDAYM"].between(2, 8)
    ]
    candidate_count = len(candidate_rows)
    if candidate_count == 0:
        raise ValueError("no complete female participant-days were found")

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = output_dir / "sensor_values.incomplete.npy"
    values = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=np.float32,
        shape=(candidate_count, len(NHANES_FEMALE_SENSOR_DESCRIPTORS), 1440),
    )
    values[:] = np.nan
    index_rows: list[dict[str, Any]] = []
    excluded = 0
    valid_minute_total = 0

    reader_arguments = {
        "dtype": {"SEQN": str},
        "chunksize": chunk_size,
        "low_memory": False,
    }
    activity_reader = pd.read_csv(paths["activity"], **reader_arguments)
    state_reader = pd.read_csv(paths["wear_state"], **reader_arguments)
    flag_reader = pd.read_csv(paths["quality"], **reader_arguments)
    for activity_chunk, state_chunk, flag_chunk in zip(
        activity_reader, state_reader, flag_reader, strict=True
    ):
        _assert_aligned(activity_chunk, state_chunk, flag_chunk)
        selected = activity_chunk["SEQN"].isin(female_ids) & activity_chunk[
            "PAXDAYM"
        ].between(2, 8)
        if not bool(selected.any()):
            continue
        activity_chunk = activity_chunk.loc[selected]
        state_chunk = state_chunk.loc[selected]
        flag_chunk = flag_chunk.loc[selected]
        activity = activity_chunk.loc[:, MINUTE_COLUMNS].to_numpy(dtype=np.float32)
        states = state_chunk.loc[:, MINUTE_COLUMNS].to_numpy(dtype=np.float32)
        flagged, flag_missing = _flag_arrays(flag_chunk)
        valid = (
            np.isfinite(activity)
            & np.isfinite(states)
            & np.isin(states, (1.0, 2.0))
            & ~flagged
            & ~flag_missing
        )
        for row_position in range(len(activity_chunk)):
            valid_minutes = int(valid[row_position].sum())
            if valid_minutes < minimum_valid_minutes:
                excluded += 1
                continue
            sample_index = len(index_rows)
            activity_values = np.clip(activity[row_position], 0.0, None)
            values[sample_index, 0, valid[row_position]] = activity_values[
                valid[row_position]
            ]
            values[sample_index, 1, valid[row_position]] = (
                states[row_position, valid[row_position]] == 2.0
            ).astype(np.float32)
            participant = str(activity_chunk.iloc[row_position]["SEQN"])
            participant_metadata = metadata[participant]
            index_rows.append(
                {
                    "sample_index": sample_index,
                    "participant_id": participant,
                    "measurement_day": int(
                        activity_chunk.iloc[row_position]["PAXDAYM"]
                    ),
                    "day_of_week": int(
                        activity_chunk.iloc[row_position]["PAXDAYWM"]
                    ),
                    "age_years": float(
                        participant_metadata["age_in_years_at_screening"]
                    ),
                    "release_cycle": participant_metadata["data_release_cycle"],
                    "valid_minutes": valid_minutes,
                }
            )
            valid_minute_total += valid_minutes
    values.flush()
    del values

    if not index_rows:
        temporary_path.unlink(missing_ok=True)
        raise ValueError("all NHANES candidate days failed the coverage threshold")
    final_path = output_dir / "sensor_values.npy"
    if len(index_rows) == candidate_count:
        temporary_path.replace(final_path)
    else:
        source_values = np.load(temporary_path, mmap_mode="r")
        final_values = np.lib.format.open_memmap(
            final_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(index_rows), len(NHANES_FEMALE_SENSOR_DESCRIPTORS), 1440),
        )
        for start in range(0, len(index_rows), 256):
            stop = min(start + 256, len(index_rows))
            final_values[start:stop] = source_values[start:stop]
        final_values.flush()
        del final_values, source_values
        temporary_path.unlink()

    retained_participants = sorted({row["participant_id"] for row in index_rows})
    splits = _stable_split(retained_participants, seed=seed)
    _write_index(output_dir / "index.csv", index_rows)
    (output_dir / "participant_splits.json").write_text(
        json.dumps(splits, indent=2), encoding="utf-8"
    )
    schema = {
        "format_version": 1,
        "source": "PhysioNet minute-level NHANES 2011-2014 v1.0.1",
        "selection": {
            "gender": "Female",
            "minimum_age": minimum_age,
            "measurement_days": [2, 3, 4, 5, 6, 7, 8],
            "minimum_valid_minutes": minimum_valid_minutes,
        },
        "sensor_descriptors": [
            asdict(descriptor) for descriptor in NHANES_FEMALE_SENSOR_DESCRIPTORS
        ],
        "activity_source": "log10(1 + MIMS)",
        "sleep_source": "PAXPREDM == 2",
        "valid_minute_rule": "state in {wake wear, sleep wear}; finite MIMS; no PAXFLGSM flag",
        "split_unit": "participant_id",
    }
    (output_dir / "schema.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    retained_ages = np.asarray([float(row["age_years"]) for row in index_rows])
    summary = NHANESFemalePreparationSummary(
        female_participants=len(retained_participants),
        participant_days=len(index_rows),
        age_min=float(retained_ages.min()),
        age_max=float(retained_ages.max()),
        sensor_shape=(len(index_rows), len(NHANES_FEMALE_SENSOR_DESCRIPTORS), 1440),
        valid_minute_fraction=valid_minute_total / (len(index_rows) * 1440),
        excluded_low_coverage_days=excluded,
        split_participants={name: len(ids) for name, ids in splits.items()},
        output_dir=str(output_dir),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(asdict(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fit_nhanes_female_normalization(output_dir)
    return summary


def fit_nhanes_female_normalization(
    processed_dir: Path,
    *,
    chunk_size: int = 256,
) -> dict[str, Any]:
    processed_dir = Path(processed_dir)
    with (processed_dir / "index.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    splits = json.loads(
        (processed_dir / "participant_splits.json").read_text(encoding="utf-8")
    )
    train_ids = set(splits["train"])
    indices = np.asarray(
        [int(row["sample_index"]) for row in rows if row["participant_id"] in train_ids]
    )
    values = np.load(processed_dir / "sensor_values.npy", mmap_mode="r")
    sums = np.zeros(values.shape[1], dtype=np.float64)
    square_sums = np.zeros(values.shape[1], dtype=np.float64)
    counts = np.zeros(values.shape[1], dtype=np.int64)
    for start in range(0, len(indices), chunk_size):
        batch = np.asarray(values[indices[start : start + chunk_size]], dtype=np.float64)
        finite = np.isfinite(batch)
        clean = np.where(finite, batch, 0.0)
        sums += clean.sum(axis=(0, 2))
        square_sums += np.square(clean).sum(axis=(0, 2))
        counts += finite.sum(axis=(0, 2))
    means = sums / counts
    standard_deviations = np.sqrt(
        np.maximum(square_sums / counts - np.square(means), 1e-12)
    )
    report = {
        "format_version": 1,
        "fit_split": "train",
        "fit_participants": len(train_ids),
        "fit_days": len(indices),
        "sensors": {
            descriptor.name: {
                "mean": float(means[index]),
                "std": float(standard_deviations[index]),
                "observations": int(counts[index]),
            }
            for index, descriptor in enumerate(NHANES_FEMALE_SENSOR_DESCRIPTORS)
        },
    }
    (processed_dir / "normalization.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


class NHANESFemaleDailyDataset(Dataset[dict[str, Any]]):
    """Memory-mapped participant-days for female NHANES pretraining."""

    def __init__(self, processed_dir: Path, *, split: str, normalize: bool = True) -> None:
        self.processed_dir = Path(processed_dir)
        with (self.processed_dir / "index.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            all_rows = list(csv.DictReader(handle))
        splits = json.loads(
            (self.processed_dir / "participant_splits.json").read_text(encoding="utf-8")
        )
        if split not in splits:
            raise ValueError(f"unknown split: {split}")
        allowed = set(splits[split])
        self.rows = [row for row in all_rows if row["participant_id"] in allowed]
        self.values = np.load(self.processed_dir / "sensor_values.npy", mmap_mode="r")
        self.normalize = normalize
        statistics = json.loads(
            (self.processed_dir / "normalization.json").read_text(encoding="utf-8")
        )
        sensor_statistics = list(statistics["sensors"].values())
        self.mean = np.asarray([item["mean"] for item in sensor_statistics], dtype=np.float32)
        self.std = np.asarray([item["std"] for item in sensor_statistics], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = self.rows[item]
        sample_index = int(row["sample_index"])
        values = np.asarray(self.values[sample_index]).copy()
        if self.normalize:
            values = (values - self.mean[:, None]) / self.std[:, None]
        return {
            "sensor_values": torch.from_numpy(values),
            "channel_present": torch.from_numpy(np.isfinite(values).any(axis=1)),
            "sample_index": sample_index,
            "participant_id": row["participant_id"],
            "measurement_day": int(row["measurement_day"]),
            "day_of_week": int(row["day_of_week"]),
            "age_years": float(row["age_years"]),
        }

    def close(self) -> None:
        mmap = getattr(self.values, "_mmap", None)
        if mmap is not None:
            mmap.close()


class NHANESFemaleTemporalPairDataset(Dataset[dict[str, Any]]):
    """Adjacent within-participant NHANES days for temporal-order training."""

    def __init__(self, processed_dir: Path, *, split: str, normalize: bool = True) -> None:
        self.days = NHANESFemaleDailyDataset(
            processed_dir, split=split, normalize=normalize
        )
        positions = {
            (row["participant_id"], int(row["measurement_day"])): position
            for position, row in enumerate(self.days.rows)
        }
        self.pairs = []
        for (participant, day), position in positions.items():
            following = positions.get((participant, day + 1))
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
            "second_is_later": torch.tensor(target, dtype=torch.float32),
        }

    def close(self) -> None:
        self.days.close()


__all__ = [
    "NHANESFemaleDailyDataset",
    "NHANESFemalePreparationSummary",
    "NHANESFemaleTemporalPairDataset",
    "fit_nhanes_female_normalization",
    "prepare_nhanes_female",
]
