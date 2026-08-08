"""Female-only adapter for the longitudinal DEPRESS Fitbit release.

Daily Fitbit streams are kept at minute resolution.  Questionnaire targets
are paired only with wearable days strictly before each assessment, so the
assessment date and all future measurements are excluded from the input.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from femmhc.sensors import AFFECTIVE_DAILY_SENSOR_DESCRIPTORS


SENSOR_SOURCES: tuple[str, ...] = ("steps", "heart_rate", "sleep_asleep")
SENSOR_TRANSFORMS: tuple[str, ...] = ("log1p", "identity", "identity")
TARGET_COLUMNS: tuple[str, ...] = (
    "cesd",
    "stai_state",
    "perceived_stress",
    "positive_affect",
    "negative_affect",
)

_RAW_FILE_RE = re.compile(r"^(heart|step|sleep)(\d{8})\.csv$", re.IGNORECASE)


@dataclass(frozen=True)
class DEPRESSFitbitPreparationSummary:
    female_participants_in_demographics: int
    female_participants_with_daily_streams: int
    participant_days: int
    date_min: str
    date_max: str
    sensor_shape: tuple[int, int, int]
    sensor_observations: dict[str, int]
    selected_raw_files: dict[str, int]
    duplicate_raw_files_resolved: int
    assessment_observations: dict[str, int]
    assessments_with_minimum_history: int
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


def _folder_id(participant_id: Any) -> str:
    return str(participant_id).strip().replace(" ", "_")


def _discover_raw_files(
    fitbit_root: Path,
    *,
    female_ids: set[str],
) -> tuple[dict[tuple[str, pd.Timestamp, str], Path], int]:
    candidates: dict[tuple[str, pd.Timestamp, str], list[Path]] = {}
    for path in fitbit_root.rglob("*.csv"):
        matched = _RAW_FILE_RE.match(path.name)
        if matched is None:
            continue
        relative = path.relative_to(fitbit_root)
        if len(relative.parts) < 2:
            continue
        participant = relative.parts[0]
        if participant not in female_ids:
            continue
        date = pd.to_datetime(matched.group(2), format="%Y%m%d", errors="coerce")
        if pd.isna(date):
            continue
        sensor = matched.group(1).lower()
        candidates.setdefault((participant, pd.Timestamp(date), sensor), []).append(path)
    selected: dict[tuple[str, pd.Timestamp, str], Path] = {}
    duplicates = 0
    for key, paths in candidates.items():
        duplicates += len(paths) - 1
        # Repeated export periods overlap.  Prefer the largest copy because it
        # normally has the most samples; path order is the deterministic tie-break.
        selected[key] = max(paths, key=lambda item: (item.stat().st_size, str(item)))
    return selected, duplicates


def _minute_index(time_values: pd.Series) -> pd.Series:
    parsed = pd.to_timedelta(time_values.astype(str), errors="coerce")
    return (parsed.dt.total_seconds() // 60).astype("Int64")


def _read_minute_stream(path: Path, sensor: str) -> np.ndarray:
    result = np.full(1440, np.nan, dtype=np.float32)
    frame = pd.read_csv(path)
    if "Time" not in frame.columns:
        return result
    minute = _minute_index(frame["Time"])
    valid_minute = minute.notna() & minute.between(0, 1439)
    if sensor == "heart":
        if "Heart Rate" not in frame.columns:
            return result
        values = pd.to_numeric(frame["Heart Rate"], errors="coerce")
        aggregate = pd.DataFrame({"minute": minute, "value": values}).loc[
            valid_minute & values.notna()
        ].groupby("minute")["value"].mean()
    elif sensor == "step":
        if "Step" not in frame.columns:
            return result
        values = pd.to_numeric(frame["Step"], errors="coerce")
        aggregate = pd.DataFrame({"minute": minute, "value": values}).loc[
            valid_minute & values.notna()
        ].groupby("minute")["value"].sum()
    elif sensor == "sleep":
        if "Interpreted" in frame.columns:
            interpreted = frame["Interpreted"].fillna("").astype(str).str.lower()
            values = interpreted.str.contains("asleep|light|deep|rem", regex=True).astype(float)
        elif "State" in frame.columns:
            # Fallback for releases without the interpretation column.
            values = pd.to_numeric(frame["State"], errors="coerce").eq(1).astype(float)
        else:
            return result
        aggregate = pd.DataFrame({"minute": minute, "value": values}).loc[
            valid_minute
        ].groupby("minute")["value"].max()
    else:
        raise ValueError(f"unknown Fitbit stream: {sensor}")
    if len(aggregate):
        indices = aggregate.index.to_numpy(dtype=np.int64)
        result[indices] = aggregate.to_numpy(dtype=np.float32)
    return result


def _assessment_table(
    source_dir: Path,
    *,
    female_ids: set[str],
) -> pd.DataFrame:
    cesd = pd.read_csv(source_dir / "CES-D_STAI.csv")
    cesd = cesd.loc[cesd["ID"].notna() & cesd["StartDate"].notna()].copy()
    cesd["participant_id"] = cesd["ID"].map(_folder_id)
    cesd["date"] = pd.to_datetime(cesd["StartDate"], errors="coerce").dt.normalize()
    cesd = cesd.loc[cesd["participant_id"].isin(female_ids) & cesd["date"].notna()]
    cesd = cesd.groupby(["participant_id", "date"], as_index=False)[
        ["CESD", "STAI_st"]
    ].mean()
    cesd.rename(columns={"CESD": "cesd", "STAI_st": "stai_state"}, inplace=True)

    panas = pd.read_csv(source_dir / "PANAS_PSS.csv")
    panas = panas.loc[panas["ID"].notna() & panas["StartDate"].notna()].copy()
    panas["participant_id"] = panas["ID"].map(_folder_id)
    panas["date"] = pd.to_datetime(panas["StartDate"], errors="coerce").dt.normalize()
    panas = panas.loc[
        panas["participant_id"].isin(female_ids) & panas["date"].notna()
    ]
    panas = panas.groupby(["participant_id", "date"], as_index=False)[
        ["PSS", "Positive emotion", "Negative emotion"]
    ].mean()
    panas.rename(
        columns={
            "PSS": "perceived_stress",
            "Positive emotion": "positive_affect",
            "Negative emotion": "negative_affect",
        },
        inplace=True,
    )
    return cesd.merge(panas, on=["participant_id", "date"], how="outer").sort_values(
        ["participant_id", "date"]
    )


def prepare_depress_fitbit(
    source_dir: Path,
    output_dir: Path,
    *,
    fitbit_root: Path | None = None,
    history_days: int = 28,
    minimum_history_days: int = 3,
    minimum_observed_minutes: int = 60,
    seed: int = 42,
) -> DEPRESSFitbitPreparationSummary:
    """Build daily streams and pre-assessment windows for female participants."""

    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    fitbit_root = (
        Path(fitbit_root).resolve()
        if fitbit_root is not None
        else source_dir / "Fitbit_extracted" / "Fitbit"
    )
    if history_days <= 0 or minimum_history_days <= 0:
        raise ValueError("history_days and minimum_history_days must be positive")
    if minimum_history_days > history_days:
        raise ValueError("minimum_history_days cannot exceed history_days")
    required = ("demographics.xlsx", "CES-D_STAI.csv", "PANAS_PSS.csv")
    if missing := [name for name in required if not (source_dir / name).is_file()]:
        raise FileNotFoundError(f"DEPRESS release is missing files: {missing}")
    if not fitbit_root.is_dir():
        raise FileNotFoundError(
            f"extracted Fitbit directory not found: {fitbit_root}; extract Fitbit.rar first"
        )

    demographics = pd.read_excel(source_dir / "demographics.xlsx", usecols=["ID", "Sex"])
    female_ids = set(
        demographics.loc[demographics["Sex"].eq("Female"), "ID"].map(_folder_id)
    )
    selected, duplicate_files = _discover_raw_files(
        fitbit_root, female_ids=female_ids
    )
    if not selected:
        raise ValueError("no female raw heart, step, or sleep files were discovered")

    keys = sorted({(participant, date) for participant, date, _ in selected})
    retained: list[tuple[str, pd.Timestamp, dict[str, np.ndarray]]] = []
    selected_counts = {sensor: 0 for sensor in ("heart", "step", "sleep")}
    for item_index, (participant, date) in enumerate(keys, start=1):
        streams: dict[str, np.ndarray] = {}
        for sensor in ("heart", "step", "sleep"):
            path = selected.get((participant, date, sensor))
            if path is not None:
                streams[sensor] = _read_minute_stream(path, sensor)
                selected_counts[sensor] += 1
        observed = int(np.isfinite(streams.get("heart", np.empty(0))).sum())
        observed += int(np.isfinite(streams.get("step", np.empty(0))).sum())
        if observed >= minimum_observed_minutes:
            retained.append((participant, date, streams))
        if item_index % 500 == 0:
            print(
                f"processed {item_index}/{len(keys)} DEPRESS participant-days",
                flush=True,
            )
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
    participant_days: dict[str, list[tuple[pd.Timestamp, int]]] = {}
    for day_index, (participant, date, streams) in enumerate(retained):
        if "step" in streams:
            values[day_index, 0] = streams["step"]
        if "heart" in streams:
            values[day_index, 1] = streams["heart"]
        if "sleep" in streams:
            values[day_index, 2] = streams["sleep"]
        index_rows.append(
            {
                "day_index": day_index,
                "participant_id": participant,
                "date": date.strftime("%Y-%m-%d"),
                "heart_minutes": int(np.isfinite(values[day_index, 1]).sum()),
                "step_minutes": int(np.isfinite(values[day_index, 0]).sum()),
                "sleep_minutes": int(np.isfinite(values[day_index, 2]).sum()),
            }
        )
        participant_days.setdefault(participant, []).append((date, day_index))
    values.flush()
    sensor_observations = {
        descriptor.name: int(np.isfinite(values[:, channel]).sum())
        for channel, descriptor in enumerate(AFFECTIVE_DAILY_SENSOR_DESCRIPTORS)
    }
    del values

    assessments = _assessment_table(source_dir, female_ids=female_ids)
    sensor_participants = set(participant_days)
    assessments = assessments.loc[assessments["participant_id"].isin(sensor_participants)]
    assessment_rows: list[dict[str, Any]] = []
    usable_assessments = 0
    for assessment_index, item in enumerate(assessments.itertuples(index=False)):
        participant = str(item.participant_id)
        date = pd.Timestamp(item.date)
        history = [
            (day, day_index)
            for day, day_index in participant_days[participant]
            if date - pd.Timedelta(days=history_days) <= day < date
        ]
        history.sort()
        indices = [day_index for _, day_index in history]
        usable_assessments += int(len(indices) >= minimum_history_days)
        row: dict[str, Any] = {
            "assessment_index": assessment_index,
            "participant_id": participant,
            "date": date.strftime("%Y-%m-%d"),
            "history_indices": ";".join(str(index) for index in indices),
            "history_days_available": len(indices),
        }
        for target in TARGET_COLUMNS:
            value = getattr(item, target)
            row[target] = "" if pd.isna(value) else float(value)
        assessment_rows.append(row)

    participants = sorted(sensor_participants)
    splits = _stable_split(participants, seed=seed)
    _write_csv(
        output_dir / "index.csv",
        index_rows,
        (
            "day_index",
            "participant_id",
            "date",
            "heart_minutes",
            "step_minutes",
            "sleep_minutes",
        ),
    )
    _write_csv(
        output_dir / "assessments.csv",
        assessment_rows,
        (
            "assessment_index",
            "participant_id",
            "date",
            "history_indices",
            "history_days_available",
            *TARGET_COLUMNS,
        ),
    )
    (output_dir / "participant_splits.json").write_text(
        json.dumps(splits, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    schema = {
        "format_version": 1,
        "source": "DEPRESS Fitbit",
        "selection": {"sex_field": "Sex", "female_value": "Female"},
        "sample_grain": "participant-day",
        "assessment_input": f"up to {history_days} calendar days strictly before assessment",
        "minimum_history_days": minimum_history_days,
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
        "split_unit": "participant_id",
    }
    (output_dir / "schema.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = DEPRESSFitbitPreparationSummary(
        female_participants_in_demographics=len(female_ids),
        female_participants_with_daily_streams=len(participants),
        participant_days=len(retained),
        date_min=min(date for _, date, _ in retained).strftime("%Y-%m-%d"),
        date_max=max(date for _, date, _ in retained).strftime("%Y-%m-%d"),
        sensor_shape=(len(retained), len(SENSOR_SOURCES), 1440),
        sensor_observations=sensor_observations,
        selected_raw_files=selected_counts,
        duplicate_raw_files_resolved=duplicate_files,
        assessment_observations={
            target: int(assessments[target].notna().sum()) for target in TARGET_COLUMNS
        },
        assessments_with_minimum_history=usable_assessments,
        split_participants={name: len(ids) for name, ids in splits.items()},
        output_dir=str(output_dir),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(asdict(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fit_depress_fitbit_normalization(output_dir)
    return summary


def fit_depress_fitbit_normalization(processed_dir: Path) -> dict[str, Any]:
    """Fit per-channel transforms and moments using training participants."""

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


class DEPRESSFitbitDailyDataset(Dataset[dict[str, Any]]):
    """Female DEPRESS Fitbit days for continual pretraining."""

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

    def sensor_values(self, day_index: int) -> np.ndarray:
        values = np.array(self.values[day_index], copy=True)
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
        values = self.sensor_values(int(row["day_index"]))
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


class DEPRESSAssessmentWindowDataset(DEPRESSFitbitDailyDataset):
    """Padded pre-assessment windows with questionnaire targets and masks."""

    def __init__(
        self,
        processed_dir: Path,
        *,
        split: str,
        history_days: int = 28,
        minimum_history_days: int = 3,
        task: str | None = None,
        normalize: bool = True,
    ) -> None:
        if history_days <= 0 or minimum_history_days <= 0:
            raise ValueError("history_days and minimum_history_days must be positive")
        if task is not None and task not in TARGET_COLUMNS:
            raise ValueError(f"unknown task {task!r}; choose from {TARGET_COLUMNS}")
        super().__init__(processed_dir, split=split, normalize=normalize)
        splits = json.loads(
            (self.processed_dir / "participant_splits.json").read_text(
                encoding="utf-8"
            )
        )
        allowed = set(splits[split])
        with (self.processed_dir / "assessments.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            assessments = list(csv.DictReader(handle))
        self.assessments = [
            row
            for row in assessments
            if row["participant_id"] in allowed
            and int(row["history_days_available"]) >= minimum_history_days
            and (task is None or row[task] != "")
        ]
        self.history_days = int(history_days)
        self.task = task

    def __len__(self) -> int:
        return len(self.assessments)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.assessments[index]
        indices = [
            int(value) for value in row["history_indices"].split(";") if value != ""
        ][-self.history_days :]
        values = np.full(
            (self.history_days, len(SENSOR_SOURCES), 1440),
            np.nan,
            dtype=np.float32,
        )
        first = self.history_days - len(indices)
        for offset, day_index in enumerate(indices, start=first):
            values[offset] = self.sensor_values(day_index)
        targets = np.asarray(
            [float(row[name]) if row[name] != "" else np.nan for name in TARGET_COLUMNS],
            dtype=np.float32,
        )
        result: dict[str, Any] = {
            "sensor_values": torch.from_numpy(values),
            "channel_present": torch.from_numpy(np.isfinite(values).any(axis=2)),
            "day_present": torch.from_numpy(np.isfinite(values).any(axis=(1, 2))),
            "targets": torch.from_numpy(targets),
            "target_present": torch.from_numpy(np.isfinite(targets)),
            "participant_id": row["participant_id"],
            "assessment_date": row["date"],
            "assessment_index": torch.tensor(
                int(row["assessment_index"]), dtype=torch.long
            ),
        }
        if self.task is not None:
            result["target"] = torch.tensor(float(row[self.task]), dtype=torch.float32)
        return result


__all__ = [
    "DEPRESSAssessmentWindowDataset",
    "DEPRESSFitbitDailyDataset",
    "DEPRESSFitbitPreparationSummary",
    "SENSOR_SOURCES",
    "TARGET_COLUMNS",
    "fit_depress_fitbit_normalization",
    "prepare_depress_fitbit",
]
