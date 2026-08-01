"""Build a leakage-safe daily FemMHC substrate from the mcPHASES archive."""

from __future__ import annotations

import ast
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import io
import json
import math
from pathlib import Path
import random
from typing import Callable, Iterable
import zipfile

import numpy as np
import pandas as pd

from femmhc.sensors import MCPHASES_SENSOR_DESCRIPTORS


MCPHASES_LABEL_FIELDS: tuple[str, ...] = (
    "phase",
    "cramps",
    "moodswing",
    "fatigue",
    "sleepissue",
    "stress",
    "bloating",
    "flow_volume",
    "menstrual_onset_24h",
    "menstrual_onset_72h",
)

MCPHASES_CONTEXT_FEATURES: tuple[str, ...] = (
    "resting_heart_rate",
    "stress_score",
    "sleep_overall_score",
    "sleep_composition_score",
    "sleep_revitalization_score",
    "sleep_duration_score",
    "nightly_temperature",
    "sedentary_minutes",
    "lightly_active_minutes",
    "moderately_active_minutes",
    "very_active_minutes",
)

PHASE_VALUES = {
    "Follicular": 0,
    "Fertility": 1,
    "Ovulation": 1,
    "Luteal": 2,
    "Menstrual": 3,
}

SEVERITY_VALUES = {
    "Not at all": 0,
    "Very Low/Little": 1,
    "Very Low": 1,
    "Low": 2,
    "Moderate": 3,
    "High": 4,
    "Very High": 5,
}

FLOW_VALUES = {
    "Not at all": 0,
    "Spotting / Very Light": 1,
    "Light": 2,
    "Somewhat Light": 2,
    "Moderate": 3,
    "Somewhat Heavy": 4,
    "Heavy": 5,
    "Very Heavy": 6,
}

SLEEP_STATE_VALUES = {
    "wake": 0.0,
    "awake": 0.0,
    "restless": 0.5,
    "asleep": 1.0,
    "light": 1.0,
    "deep": 2.0,
    "rem": 3.0,
}


@dataclass(frozen=True)
class McPhasesPreparationSummary:
    samples: int
    participants: int
    participant_intervals: int
    sensor_shape: tuple[int, int, int] | None
    sensor_observed_minutes: dict[str, int]
    context_observed: dict[str, int]
    label_observed: dict[str, int]
    split_participants: dict[str, int]
    output_dir: str


def _open_csv(
    archive: zipfile.ZipFile,
    names: dict[str, str],
    basename: str,
) -> csv.DictReader:
    raw = archive.open(names[basename])
    text = io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace", newline="")
    return csv.DictReader(text)


def _archive_names(archive: zipfile.ZipFile) -> dict[str, str]:
    names = {
        Path(name).name: name
        for name in archive.namelist()
        if name.lower().endswith((".csv", ".txt"))
    }
    required = {"hormones_and_selfreport.csv", "subject-info.csv"}
    missing = sorted(required - names.keys())
    if missing:
        raise ValueError(f"mcPHASES archive is missing required files: {missing}")
    return names


def _sample_key(row: dict[str, str], day_field: str = "day_in_study") -> tuple[str, str, int]:
    return row["id"], row["study_interval"], int(float(row[day_field]))


def _minute(timestamp: str) -> int | None:
    if not timestamp:
        return None
    try:
        parsed = datetime.strptime(timestamp.split(".")[0], "%H:%M:%S")
    except ValueError:
        return None
    return parsed.hour * 60 + parsed.minute


def _float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _stable_participant_split(
    participants: Iterable[str],
    *,
    seed: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict[str, list[str]]:
    participants = sorted(set(participants), key=lambda x: (len(x), x))
    random.Random(seed).shuffle(participants)
    n = len(participants)
    n_train = round(n * train_fraction)
    n_validation = round(n * validation_fraction)
    return {
        "train": sorted(participants[:n_train]),
        "validation": sorted(participants[n_train : n_train + n_validation]),
        "test": sorted(participants[n_train + n_validation :]),
    }


def _encode_label(field: str, value: str) -> int:
    if not value:
        return -1
    if field == "phase":
        return PHASE_VALUES.get(value, -1)
    if field == "flow_volume":
        return FLOW_VALUES.get(value, -1)
    if field in SEVERITY_VALUES:
        return SEVERITY_VALUES[field]
    if field in {"cramps", "moodswing", "fatigue", "sleepissue", "stress", "bloating"}:
        if value in SEVERITY_VALUES:
            return SEVERITY_VALUES[value]
        # A handful of source cells use numeric strings despite the documented
        # categorical scale.  Preserve their ordinal meaning explicitly.
        try:
            numeric = int(float(value))
        except ValueError:
            return -1
        return numeric if 0 <= numeric <= 5 else -1
    return -1


def _write_index(
    output_dir: Path,
    keys: list[tuple[str, str, int]],
    labels: np.ndarray,
) -> None:
    with (output_dir / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "participant_id", "study_interval", "day_in_study", "has_symptom_labels"])
        symptom_columns = [MCPHASES_LABEL_FIELDS.index(x) for x in ("cramps", "moodswing", "fatigue")]
        for index, key in enumerate(keys):
            has_symptoms = bool((labels[index, symptom_columns] >= 0).any())
            writer.writerow([index, key[0], key[1], key[2], int(has_symptoms)])


def _populate_point_sensor(
    archive: zipfile.ZipFile,
    names: dict[str, str],
    *,
    basename: str,
    value_field: str,
    sensor_index: int,
    values: np.ndarray,
    key_to_index: dict[tuple[str, str, int], int],
    expansion_minutes: int = 1,
    chunk_size: int = 1_000_000,
) -> None:
    # Dense per-day accumulators are far smaller than Python dictionaries for
    # heart-rate data (millions of observed minutes, tens of millions of raw
    # rows) and keep peak memory predictable.  Pandas' C parser plus vectorized
    # indexing is also an order of magnitude faster than csv.DictReader here.
    sums = np.zeros((values.shape[0], values.shape[-1]), dtype=np.float32)
    counts = np.zeros((values.shape[0], values.shape[-1]), dtype=np.uint16)
    encoded_to_index = {
        int(participant) * 10**12 + int(interval) * 10**6 + int(day): sample_index
        for (participant, interval, day), sample_index in key_to_index.items()
    }
    encoded_keys = np.asarray(sorted(encoded_to_index), dtype=np.int64)
    encoded_indices = np.asarray(
        [encoded_to_index[int(key)] for key in encoded_keys],
        dtype=np.int64,
    )

    usecols = ["id", "study_interval", "day_in_study", "timestamp", value_field]
    with archive.open(names[basename]) as source:
        chunks = pd.read_csv(
            source,
            usecols=usecols,
            chunksize=chunk_size,
            dtype={
                "id": "int64",
                "study_interval": "int64",
                "day_in_study": "int64",
                "timestamp": "string",
                value_field: "float32",
            },
        )
        for chunk in chunks:
            encoded = (
                chunk["id"].to_numpy(dtype=np.int64) * 10**12
                + chunk["study_interval"].to_numpy(dtype=np.int64) * 10**6
                + chunk["day_in_study"].to_numpy(dtype=np.int64)
            )
            positions = np.searchsorted(encoded_keys, encoded)
            positions_in_bounds = positions < len(encoded_keys)
            valid_key = np.zeros(len(chunk), dtype=bool)
            valid_key[positions_in_bounds] = (
                encoded_keys[positions[positions_in_bounds]]
                == encoded[positions_in_bounds]
            )
            timestamp = chunk["timestamp"].fillna("").astype(str)
            hour = pd.to_numeric(timestamp.str.slice(0, 2), errors="coerce").to_numpy()
            minute_part = pd.to_numeric(timestamp.str.slice(3, 5), errors="coerce").to_numpy()
            minute = hour * 60 + minute_part
            measurement = chunk[value_field].to_numpy(dtype=np.float32)
            valid = valid_key & np.isfinite(minute) & np.isfinite(measurement)
            if not valid.any():
                continue
            sample_index = encoded_indices[positions[valid]]
            minute = minute[valid].astype(np.int64, copy=False)
            measurement = measurement[valid]
            for offset in range(expansion_minutes):
                target_minute = minute + offset
                within_day = target_minute < values.shape[-1]
                flat_index = (
                    sample_index[within_day] * values.shape[-1]
                    + target_minute[within_day]
                )
                np.add.at(sums.ravel(), flat_index, measurement[within_day])
                np.add.at(counts.ravel(), flat_index, 1)
    observed = counts > 0
    channel = values[:, sensor_index, :]
    channel[observed] = sums[observed] / counts[observed]


def _load_sensor_progress(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"format_version": 1, "completed": []}
    progress = json.loads(path.read_text(encoding="utf-8"))
    completed = progress.get("completed", [])
    if not isinstance(completed, list):
        raise ValueError(f"Invalid sensor progress file: {path}")
    return progress


def _save_sensor_progress(path: Path, progress: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(progress, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_iso(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _populate_sleep_state(
    archive: zipfile.ZipFile,
    names: dict[str, str],
    *,
    sensor_index: int,
    values: np.ndarray,
    key_to_index: dict[tuple[str, str, int], int],
) -> None:
    reader = _open_csv(archive, names, "sleep.csv")
    for row in reader:
        try:
            levels = ast.literal_eval(row.get("levels", ""))
        except (ValueError, SyntaxError):
            continue
        if not isinstance(levels, dict):
            continue
        segments = levels.get("data", [])
        if not isinstance(segments, list):
            continue
        session_start = _parse_iso(
            f"2000-01-01T{row.get('sleep_start_timestamp', '').split('.')[0]}"
        )
        if session_start is None:
            continue
        start_day = int(float(row["sleep_start_day_in_study"]))
        participant = row["id"]
        interval = row["study_interval"]
        first_segment = _parse_iso(segments[0].get("dateTime")) if segments else None
        if first_segment is None:
            continue
        anchor = first_segment.replace(
            hour=session_start.hour,
            minute=session_start.minute,
            second=session_start.second,
            microsecond=0,
        )
        for segment in segments:
            start = _parse_iso(segment.get("dateTime"))
            state = SLEEP_STATE_VALUES.get(str(segment.get("level", "")).lower())
            seconds = _float(str(segment.get("seconds", "")))
            if start is None or state is None or seconds is None:
                continue
            offset_minutes = max(0, int((start - anchor).total_seconds() // 60))
            duration_minutes = max(1, int(math.ceil(seconds / 60.0)))
            for offset in range(duration_minutes):
                absolute_minute = (
                    session_start.hour * 60
                    + session_start.minute
                    + offset_minutes
                    + offset
                )
                day_offset, minute = divmod(absolute_minute, 1440)
                key = (participant, interval, start_day + day_offset)
                sample_index = key_to_index.get(key)
                if sample_index is not None:
                    values[sample_index, sensor_index, minute] = state


def _populate_context(
    archive: zipfile.ZipFile,
    names: dict[str, str],
    context: np.ndarray,
    key_to_index: dict[tuple[str, str, int], int],
) -> None:
    field_to_index = {name: index for index, name in enumerate(MCPHASES_CONTEXT_FEATURES)}

    def assign(
        basename: str,
        day_field: str,
        mappings: dict[str, str],
        valid: Callable[[dict[str, str]], bool] | None = None,
    ) -> None:
        accumulators: dict[tuple[int, int], list[float]] = {}
        for row in _open_csv(archive, names, basename):
            if valid is not None and not valid(row):
                continue
            key = _sample_key(row, day_field)
            sample_index = key_to_index.get(key)
            if sample_index is None:
                continue
            for source, target in mappings.items():
                value = _float(row.get(source))
                if value is None:
                    continue
                location = (sample_index, field_to_index[target])
                accumulators.setdefault(location, []).append(value)
        for (sample_index, feature_index), observed in accumulators.items():
            context[sample_index, feature_index] = float(np.mean(observed))

    assign("resting_heart_rate.csv", "day_in_study", {"value": "resting_heart_rate"})
    assign(
        "stress_score.csv",
        "day_in_study",
        {"stress_score": "stress_score"},
        lambda row: row.get("status") == "READY" and row.get("calculation_failed") != "True",
    )
    assign(
        "sleep_score.csv",
        "day_in_study",
        {
            "overall_score": "sleep_overall_score",
            "composition_score": "sleep_composition_score",
            "revitalization_score": "sleep_revitalization_score",
            "duration_score": "sleep_duration_score",
        },
    )
    assign(
        "computed_temperature.csv",
        "sleep_end_day_in_study",
        {"nightly_temperature": "nightly_temperature"},
    )
    assign(
        "active_minutes.csv",
        "day_in_study",
        {
            "sedentary": "sedentary_minutes",
            "lightly": "lightly_active_minutes",
            "moderately": "moderately_active_minutes",
            "very": "very_active_minutes",
        },
    )


def prepare_mcphases(
    archive_path: Path,
    output_dir: Path,
    *,
    include_sensors: bool = True,
    seed: int = 42,
    resume: bool = True,
) -> McPhasesPreparationSummary:
    """Materialize mcPHASES without extracting its 3.5 GB archive tree."""

    archive_path = Path(archive_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path) as archive:
        names = _archive_names(archive)
        source_rows = list(_open_csv(archive, names, "hormones_and_selfreport.csv"))
        keys = sorted(
            {_sample_key(row) for row in source_rows},
            key=lambda key: (int(key[0]), key[1], key[2]),
        )
        key_to_index = {key: index for index, key in enumerate(keys)}
        rows_by_key = {_sample_key(row): row for row in source_rows}

        labels = np.full((len(keys), len(MCPHASES_LABEL_FIELDS)), -1, dtype=np.int16)
        hormones = np.full((len(keys), 3), np.nan, dtype=np.float32)
        hormone_fields = ("lh", "estrogen", "pdg")
        for sample_index, key in enumerate(keys):
            row = rows_by_key[key]
            for field in MCPHASES_LABEL_FIELDS[:8]:
                labels[sample_index, MCPHASES_LABEL_FIELDS.index(field)] = _encode_label(
                    field,
                    row.get(field, ""),
                )
            for hormone_index, field in enumerate(hormone_fields):
                value = _float(row.get(field))
                if value is not None:
                    hormones[sample_index, hormone_index] = value

        onset_24_index = MCPHASES_LABEL_FIELDS.index("menstrual_onset_24h")
        onset_72_index = MCPHASES_LABEL_FIELDS.index("menstrual_onset_72h")
        for sample_index, key in enumerate(keys):
            current_phase = rows_by_key[key].get("phase")
            future_phases = []
            for offset in (1, 2, 3):
                future = rows_by_key.get((key[0], key[1], key[2] + offset))
                future_phases.append(future.get("phase") if future else None)
            if future_phases[0] is not None:
                labels[sample_index, onset_24_index] = int(
                    current_phase != "Menstrual" and future_phases[0] == "Menstrual"
                )
            observed_future = [phase for phase in future_phases if phase is not None]
            if observed_future:
                labels[sample_index, onset_72_index] = int(
                    current_phase != "Menstrual" and "Menstrual" in observed_future
                )

        context = np.full(
            (len(keys), len(MCPHASES_CONTEXT_FEATURES)),
            np.nan,
            dtype=np.float32,
        )
        _populate_context(archive, names, context, key_to_index)

        sensor_observed: dict[str, int] = {}
        sensor_shape: tuple[int, int, int] | None = None
        if include_sensors:
            sensor_path = output_dir / "sensor_values.npy"
            progress_path = output_dir / "sensor_build_progress.json"
            expected_shape = (len(keys), len(MCPHASES_SENSOR_DESCRIPTORS), 1440)
            can_resume = resume and sensor_path.is_file() and progress_path.is_file()
            if can_resume:
                sensor_values = np.load(sensor_path, mmap_mode="r+")
                if tuple(sensor_values.shape) != expected_shape:
                    raise ValueError(
                        f"Existing sensor matrix has shape {sensor_values.shape}; "
                        f"expected {expected_shape}. Use --no-resume to rebuild it."
                    )
                progress = _load_sensor_progress(progress_path)
            else:
                sensor_values = np.lib.format.open_memmap(
                    sensor_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=expected_shape,
                )
                sensor_values[:] = np.nan
                sensor_values.flush()
                progress = {"format_version": 1, "completed": []}
                _save_sensor_progress(progress_path, progress)

            completed = set(str(item) for item in progress["completed"])
            point_tasks = (
                ("steps", "steps.csv", "steps", 0, 1),
                ("heart_rate", "heart_rate.csv", "bpm", 1, 1),
                ("hrv_rmssd", "heart_rate_variability_details.csv", "rmssd", 2, 5),
                (
                    "wrist_temperature",
                    "wrist_temperature.csv",
                    "temperature_diff_from_baseline",
                    3,
                    1,
                ),
                (
                    "oxygen_variation",
                    "estimated_oxygen_variation.csv",
                    "infrared_to_red_signal_ratio",
                    4,
                    1,
                ),
            )
            for sensor_name, basename, value_field, sensor_index, expansion in point_tasks:
                if sensor_name in completed:
                    print(f"[mcPHASES] resume: {sensor_name} already complete", flush=True)
                    continue
                print(f"[mcPHASES] building sensor: {sensor_name}", flush=True)
                sensor_values[:, sensor_index, :] = np.nan
                _populate_point_sensor(
                    archive,
                    names,
                    basename=basename,
                    value_field=value_field,
                    sensor_index=sensor_index,
                    values=sensor_values,
                    key_to_index=key_to_index,
                    expansion_minutes=expansion,
                )
                sensor_values.flush()
                completed.add(sensor_name)
                progress["completed"] = sorted(completed)
                _save_sensor_progress(progress_path, progress)
                observed = int(np.isfinite(sensor_values[:, sensor_index, :]).sum())
                print(f"[mcPHASES] completed {sensor_name}: {observed:,} minutes", flush=True)

            sensor_name = "sleep_state"
            sensor_index = 5
            if sensor_name not in completed:
                print(f"[mcPHASES] building sensor: {sensor_name}", flush=True)
                sensor_values[:, sensor_index, :] = np.nan
                _populate_sleep_state(
                    archive,
                    names,
                    sensor_index=sensor_index,
                    values=sensor_values,
                    key_to_index=key_to_index,
                )
                sensor_values.flush()
                completed.add(sensor_name)
                progress["completed"] = sorted(completed)
                _save_sensor_progress(progress_path, progress)
                observed = int(np.isfinite(sensor_values[:, sensor_index, :]).sum())
                print(f"[mcPHASES] completed {sensor_name}: {observed:,} minutes", flush=True)
            sensor_values.flush()
            sensor_observed = {
                descriptor.name: int(np.isfinite(sensor_values[:, index]).sum())
                for index, descriptor in enumerate(MCPHASES_SENSOR_DESCRIPTORS)
            }
            sensor_shape = tuple(int(x) for x in sensor_values.shape)
            del sensor_values

    np.save(output_dir / "labels.npy", labels)
    np.save(output_dir / "hormones.npy", hormones)
    np.save(output_dir / "daily_context.npy", context)
    _write_index(output_dir, keys, labels)

    splits = _stable_participant_split((key[0] for key in keys), seed=seed)
    (output_dir / "participant_splits.json").write_text(
        json.dumps(splits, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    schema = {
        "format_version": 1,
        "sensor_descriptors": [asdict(item) for item in MCPHASES_SENSOR_DESCRIPTORS],
        "context_features": list(MCPHASES_CONTEXT_FEATURES),
        "label_fields": list(MCPHASES_LABEL_FIELDS),
        "hormone_fields": ["lh", "estrogen", "pdg"],
        "missing_sensor_value": "NaN",
        "missing_label_value": -1,
        "split_unit": "participant_id",
        "normalization": "fit on training participants only",
    }
    (output_dir / "schema.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = McPhasesPreparationSummary(
        samples=len(keys),
        participants=len({key[0] for key in keys}),
        participant_intervals=len({(key[0], key[1]) for key in keys}),
        sensor_shape=sensor_shape,
        sensor_observed_minutes=sensor_observed,
        context_observed={
            field: int(np.isfinite(context[:, index]).sum())
            for index, field in enumerate(MCPHASES_CONTEXT_FEATURES)
        },
        label_observed={
            field: int((labels[:, index] >= 0).sum())
            for index, field in enumerate(MCPHASES_LABEL_FIELDS)
        },
        split_participants={name: len(ids) for name, ids in splits.items()},
        output_dir=str(output_dir),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(asdict(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary
