"""Embedding-level datasets for cross-cohort FemMHC joint training."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..tasks import MCPHASES_TASKS
from .mcphases_history import mcphases_task_targets


MCPHASES_CANONICAL = {
    "cycle_phase": "mcphases/cycle_phase",
    "menstrual_onset_24h": "mcphases/menstrual_onset_24h",
    "menstrual_onset_72h": "mcphases/menstrual_onset_72h",
    "cramps": "mcphases/cramps",
    "mood_swing": "mcphases/mood_swing",
    "fatigue": "mcphases/fatigue",
    "sleep_issue": "mcphases/sleep_issue",
    "perceived_stress": "mcphases/perceived_stress",
    "bloating": "mcphases/bloating",
    "flow_volume": "mcphases/flow_volume",
    "lh": "mcphases/lh",
    "estrogen": "mcphases/estrogen_metabolite",
    "pdg": "mcphases/pdg",
}

DEPRESS_COLUMNS = {
    "cesd": "depress/cesd",
    "stai_state": "depress/stai_state",
    "perceived_stress": "depress/perceived_stress",
    "positive_affect": "depress/positive_affect",
    "negative_affect": "depress/negative_affect",
}

INPHRSYM_COLUMNS = {
    "next_anxiety_severity": "inphrsym/next_anxiety_severity",
    "next_high_anxiety": "inphrsym/next_high_anxiety",
    "next_irritability_severity": "inphrsym/next_irritability_severity",
    "next_high_irritability": "inphrsym/next_high_irritability",
    "next_negative_mood_severity": "inphrsym/next_negative_mood_severity",
    "next_high_negative_mood": "inphrsym/next_high_negative_mood",
    "next_negative_energy_severity": "inphrsym/next_low_energy_severity",
    "next_high_negative_energy": "inphrsym/next_low_energy",
    "next_reported_panic": "inphrsym/next_reported_panic",
    "next_menstruation_state": "inphrsym/next_menstruation_state",
}

HRV_MENTAL_WINDOWS = {
    "middle": (
        14,
        {
            "phq9_2": "hrv_mental/phq9_middle",
            "gad7_2": "hrv_mental/gad7_middle",
            "isi_2": "hrv_mental/isi_middle",
        },
    ),
    "final": (
        28,
        {
            "phq9_f": "hrv_mental/phq9_final",
            "gad7_f": "hrv_mental/gad7_final",
            "isi_f": "hrv_mental/isi_final",
        },
    ),
}

OPENMHC_COLUMNS = {
    "Atrial fibrillation (Afib)": "openmhc/atrial_fibrillation",
    "BMI_categories": "openmhc/bmi_categories",
    "BMI_values": "openmhc/bmi_values",
    "BiologicalSex": "openmhc/biological_sex",
    "CAD": "openmhc/cad",
    "Cerebrovascular Disease": "openmhc/cerebrovascular_disease",
    "Diabetes": "openmhc/diabetes",
    "GoSleepTime_categories": "openmhc/go_sleep_time",
    "Hdl": "openmhc/hdl",
    "Hypertension": "openmhc/hypertension",
    "Ldl": "openmhc/ldl",
    "SystolicBloodPressure": "openmhc/systolic_blood_pressure",
    "TotalCholesterol": "openmhc/total_cholesterol",
    "WakeUpTime_categories": "openmhc/wake_up_time",
    "WeightKilograms": "openmhc/weight_kg",
    "age": "openmhc/age",
    "blood_pressure_categories": "openmhc/blood_pressure_categories",
    "cardiovascular_disease": "openmhc/cardiovascular_disease",
    "feel_worthwhile1": "openmhc/feel_worthwhile",
    "feel_worthwhile2": "openmhc/happiness",
    "feel_worthwhile3": "openmhc/worry",
    "feel_worthwhile4": "openmhc/depressed_feeling",
    "framingham_risk": "openmhc/framingham_risk",
    "satisfiedwith_life": "openmhc/life_satisfaction",
    "sleep_diagnosis1": "openmhc/sleep_diagnosis",
    "sleep_time_categories": "openmhc/sleep_duration",
    "vigorous_act": "openmhc/vigorous_activity",
    "work": "openmhc/work_status",
    "Watch_RestingHeartRate": "openmhc/watch_resting_heart_rate",
    "Watch_HeartRateVariabilitySDNN": "openmhc/watch_hrv_sdnn",
    "Watch_RespiratoryRate": "openmhc/watch_respiratory_rate",
    "Watch_WalkingHeartRateAverage": "openmhc/watch_walking_heart_rate",
    "Watch_VO2Max": "openmhc/watch_vo2max",
    "Watch_StandTime": "openmhc/watch_stand_time",
    "Watch_BasalEnergyBurned": "openmhc/watch_basal_energy",
}

OPENMHC_CATEGORICAL = {
    "Atrial fibrillation (Afib)",
    "BMI_categories",
    "BiologicalSex",
    "CAD",
    "Cerebrovascular Disease",
    "Diabetes",
    "GoSleepTime_categories",
    "Hypertension",
    "WakeUpTime_categories",
    "blood_pressure_categories",
    "cardiovascular_disease",
    "feel_worthwhile1",
    "feel_worthwhile2",
    "feel_worthwhile3",
    "feel_worthwhile4",
    "satisfiedwith_life",
    "sleep_diagnosis1",
    "sleep_time_categories",
    "work",
}


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _split_ids(processed_dir: Path, split: str) -> set[str]:
    values = json.loads(
        (Path(processed_dir) / "participant_splits.json").read_text(encoding="utf-8")
    )
    if split not in values:
        raise ValueError(f"unknown split {split!r}; choose from {sorted(values)}")
    return {str(item) for item in values[split]}


def load_aligned_embeddings(
    path: Path,
    *,
    key: str | None = None,
    output_dim: int = 768,
    single_view: str = "adapted",
) -> np.ndarray:
    """Load .npy/.npz embeddings and align single views to a dual-view width."""

    path = Path(path)
    if path.suffix.lower() == ".npy":
        values = np.load(path, mmap_mode="r")
    else:
        archive = np.load(path)
        if key is None:
            if "dual_embeddings" in archive:
                key = "dual_embeddings"
            elif "embeddings" in archive:
                key = "embeddings"
            else:
                raise ValueError(f"cannot choose an embedding array from {path}")
        values = archive[key]
    if values.ndim not in {2, 3}:
        raise ValueError("embeddings must have shape (samples, dim) or (samples, days, dim)")
    dimension = values.shape[-1]
    if dimension == output_dim:
        return values
    if dimension * 2 != output_dim:
        raise ValueError(f"cannot align embedding dimension {dimension} to {output_dim}")
    aligned = np.zeros((*values.shape[:-1], output_dim), dtype=np.float32)
    if single_view == "adapted":
        aligned[..., dimension:] = values
    elif single_view == "native":
        aligned[..., :dimension] = values
    else:
        raise ValueError("single_view must be 'native' or 'adapted'")
    return aligned


def _target_tensor(value: float, categorical: bool) -> torch.Tensor:
    if categorical:
        return torch.tensor(-1 if not np.isfinite(value) else int(value), dtype=torch.long)
    return torch.tensor(value, dtype=torch.float32)


class McPhasesJointEmbeddingDataset(Dataset[dict[str, Any]]):
    """All available menstrual targets on one causal calendar history."""

    def __init__(
        self,
        processed_dir: Path,
        embeddings_path: Path,
        *,
        split: str,
        history_days: int = 60,
        minimum_history_days: int = 7,
        output_dim: int = 768,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.history_days = int(history_days)
        self.rows = _csv_rows(self.processed_dir / "index.csv")
        self.embeddings = load_aligned_embeddings(
            embeddings_path, output_dim=output_dim
        )
        if len(self.rows) != len(self.embeddings):
            raise ValueError("mcPHASES rows and embeddings are misaligned")
        allowed = _split_ids(self.processed_dir, split)
        self.definitions = tuple(MCPHASES_TASKS)
        self.target_arrays = {
            MCPHASES_CANONICAL[task.name]: mcphases_task_targets(
                self.processed_dir, task
            )
            for task in self.definitions
        }

        by_interval: dict[tuple[str, str], list[tuple[int, int]]] = {}
        for row in self.rows:
            participant = str(row["participant_id"])
            if participant not in allowed:
                continue
            by_interval.setdefault(
                (participant, row["study_interval"]), []
            ).append((int(row["day_in_study"]), int(row["sample_index"])))
        for days in by_interval.values():
            days.sort()

        self.examples: list[dict[str, Any]] = []
        for (participant, interval), days in by_interval.items():
            for current_day, current_index in days:
                if not any(
                    np.isfinite(values[current_index])
                    for values in self.target_arrays.values()
                ):
                    continue
                first_day = current_day - self.history_days + 1
                history = [
                    (day - first_day, sample_index)
                    for day, sample_index in days
                    if first_day <= day <= current_day
                    and np.isfinite(self.embeddings[sample_index]).all()
                ]
                if len(history) < minimum_history_days:
                    continue
                self.examples.append(
                    {
                        "participant_id": participant,
                        "study_interval": interval,
                        "sample_index": current_index,
                        "history": history,
                    }
                )
        self.participant_ids = [item["participant_id"] for item in self.examples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        values = np.zeros(
            (self.history_days, self.embeddings.shape[-1]), dtype=np.float32
        )
        present = np.zeros(self.history_days, dtype=bool)
        for slot, sample_index in example["history"]:
            values[slot] = self.embeddings[sample_index]
            present[slot] = True
        target_index = example["sample_index"]
        targets = {}
        for task in self.definitions:
            task_id = MCPHASES_CANONICAL[task.name]
            targets[task_id] = _target_tensor(
                float(self.target_arrays[task_id][target_index]),
                task.kind != "regression",
            )
        return {
            "daily_embeddings": torch.from_numpy(values),
            "day_present": torch.from_numpy(present),
            "targets": targets,
            "participant_id": example["participant_id"],
            "cohort": "mcphases",
        }


class AffectiveJointEmbeddingDataset(Dataset[dict[str, Any]]):
    """DEPRESS assessment windows or inPHRsym next-day histories."""

    def __init__(
        self,
        cohort: str,
        processed_dir: Path,
        embeddings_path: Path,
        *,
        split: str,
        history_days: int = 28,
        minimum_history_days: int = 3,
        output_dim: int = 768,
    ) -> None:
        if cohort not in {"depress_fitbit", "inphrsym"}:
            raise ValueError("cohort must be depress_fitbit or inphrsym")
        self.cohort = cohort
        self.processed_dir = Path(processed_dir)
        self.history_days = int(history_days)
        self.embeddings = load_aligned_embeddings(
            embeddings_path, output_dim=output_dim
        )
        allowed = _split_ids(self.processed_dir, split)
        self.examples: list[dict[str, Any]] = []
        if cohort == "depress_fitbit":
            self.columns = DEPRESS_COLUMNS
            rows = _csv_rows(self.processed_dir / "assessments.csv")
            for row in rows:
                indices = [
                    int(value) for value in row["history_indices"].split(";") if value
                ][-self.history_days :]
                if row["participant_id"] not in allowed or len(indices) < minimum_history_days:
                    continue
                if not any(row[column] != "" for column in self.columns):
                    continue
                self.examples.append({"row": row, "indices": indices})
        else:
            self.columns = INPHRSYM_COLUMNS
            rows = _csv_rows(self.processed_dir / "index.csv")
            by_participant: dict[str, list[tuple[np.datetime64, int]]] = {}
            for row in rows:
                by_participant.setdefault(row["participant_id"], []).append(
                    (np.datetime64(row["date"], "D"), int(row["day_index"]))
                )
            for days in by_participant.values():
                days.sort()
            for row in rows:
                if row["participant_id"] not in allowed:
                    continue
                if not any(row[column] != "" for column in self.columns):
                    continue
                current = np.datetime64(row["date"], "D")
                first = current - np.timedelta64(self.history_days - 1, "D")
                history = [
                    (int((day - first).astype(int)), day_index)
                    for day, day_index in by_participant[row["participant_id"]]
                    if first <= day <= current
                ]
                if len(history) < minimum_history_days:
                    continue
                self.examples.append({"row": row, "history": history})
        self.participant_ids = [
            item["row"]["participant_id"] for item in self.examples
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        row = example["row"]
        values = np.zeros(
            (self.history_days, self.embeddings.shape[-1]), dtype=np.float32
        )
        present = np.zeros(self.history_days, dtype=bool)
        if self.cohort == "depress_fitbit":
            indices = example["indices"]
            start = self.history_days - len(indices)
            for slot, day_index in enumerate(indices, start=start):
                values[slot] = self.embeddings[day_index]
                present[slot] = True
        else:
            for slot, day_index in example["history"]:
                values[slot] = self.embeddings[day_index]
                present[slot] = True
        categorical = self.cohort == "inphrsym"
        targets = {
            task_id: _target_tensor(
                float(row[column]) if row[column] != "" else float("nan"),
                categorical,
            )
            for column, task_id in self.columns.items()
        }
        return {
            "daily_embeddings": torch.from_numpy(values),
            "day_present": torch.from_numpy(present),
            "targets": targets,
            "participant_id": row["participant_id"],
            "cohort": self.cohort,
        }


class HRVMentalJointEmbeddingDataset(Dataset[dict[str, Any]]):
    """Participant-level 14/28-day windows for female mental-health scales."""

    def __init__(
        self,
        processed_dir: Path,
        embeddings_path: Path,
        *,
        split: str,
        output_dim: int = 768,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.history_days = 28
        self.embeddings = load_aligned_embeddings(
            embeddings_path, output_dim=output_dim
        )
        rows = _csv_rows(self.processed_dir / "index.csv")
        labels = {
            row["participant_id"]: row
            for row in _csv_rows(self.processed_dir / "labels.csv")
        }
        allowed = _split_ids(self.processed_dir, split)
        by_participant: dict[str, list[tuple[np.datetime64, int]]] = {}
        for row in rows:
            by_participant.setdefault(row["participant_id"], []).append(
                (np.datetime64(row["date"], "D"), int(row["day_index"]))
            )
        self.examples = []
        for participant in sorted(allowed):
            if participant not in labels or participant not in by_participant:
                continue
            days = sorted(by_participant[participant])
            for timepoint, (window_days, columns) in HRV_MENTAL_WINDOWS.items():
                cutoff = days[0][0] + np.timedelta64(window_days, "D")
                indices = [index for day, index in days if day < cutoff]
                if not indices:
                    continue
                self.examples.append(
                    {
                        "participant_id": participant,
                        "timepoint": timepoint,
                        "indices": indices[-window_days:],
                        "columns": columns,
                        "labels": labels[participant],
                    }
                )
        self.participant_ids = [item["participant_id"] for item in self.examples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        values = np.zeros(
            (self.history_days, self.embeddings.shape[-1]), dtype=np.float32
        )
        present = np.zeros(self.history_days, dtype=bool)
        indices = example["indices"]
        for slot, day_index in enumerate(indices, start=self.history_days - len(indices)):
            values[slot] = self.embeddings[day_index]
            present[slot] = True
        all_columns = {
            column: task_id
            for _, columns in HRV_MENTAL_WINDOWS.values()
            for column, task_id in columns.items()
        }
        targets = {
            task_id: torch.tensor(float("nan"), dtype=torch.float32)
            for task_id in all_columns.values()
        }
        targets.update({
            task_id: torch.tensor(float(example["labels"][column]), dtype=torch.float32)
            for column, task_id in example["columns"].items()
        })
        return {
            "daily_embeddings": torch.from_numpy(values),
            "day_present": torch.from_numpy(present),
            "targets": targets,
            "participant_id": example["participant_id"],
            "cohort": "wearable_hrv_sleep",
        }


class PregnancyJointEmbeddingDataset(Dataset[dict[str, Any]]):
    """Seven-day pregnancy windows with gestational-age supervision."""

    def __init__(
        self,
        processed_dir: Path,
        embeddings_path: Path,
        *,
        split: str,
        output_dim: int = 768,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        archive = np.load(embeddings_path)
        self.embeddings = load_aligned_embeddings(
            embeddings_path,
            key="embeddings",
            output_dim=output_dim,
        )
        self.day_present = archive["day_present"]
        allowed = _split_ids(self.processed_dir, split)
        self.rows = [
            row
            for row in _csv_rows(self.processed_dir / "index.csv")
            if row["participant_id"] in allowed
        ]
        self.participant_ids = [row["participant_id"] for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        measurement = int(row["measurement_index"])
        return {
            "daily_embeddings": torch.from_numpy(
                np.asarray(self.embeddings[measurement], dtype=np.float32)
            ),
            "day_present": torch.from_numpy(
                np.asarray(self.day_present[measurement], dtype=bool)
            ),
            "targets": {
                "pregnancy/gestational_age": torch.tensor(
                    float(row["gestational_age_weeks"]), dtype=torch.float32
                )
            },
            "participant_id": row["participant_id"],
            "cohort": "pregnancy_ga_clock",
        }


class OpenMHCAuxiliaryEmbeddingDataset(Dataset[dict[str, Any]]):
    """Causal OpenMHC histories with all 28 currently trainable XS targets."""

    def __init__(
        self,
        data_dir: Path,
        native_cache_dir: Path,
        adapted_cache_dir: Path,
        *,
        split: str,
        history_days: int = 7,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.history_days = int(history_days)
        native_cache_dir = Path(native_cache_dir)
        adapted_cache_dir = Path(adapted_cache_dir)
        self.native = np.load(native_cache_dir / "embeddings.npy", mmap_mode="r")
        self.adapted = np.load(adapted_cache_dir / "embeddings.npy", mmap_mode="r")
        user_ids = np.load(native_cache_dir / "user_ids.npy", allow_pickle=True)
        dates = np.load(native_cache_dir / "dates.npy", allow_pickle=True)
        adapted_user_ids = np.load(adapted_cache_dir / "user_ids.npy", allow_pickle=True)
        adapted_dates = np.load(adapted_cache_dir / "dates.npy", allow_pickle=True)
        if (
            self.native.shape != self.adapted.shape
            or not np.array_equal(user_ids, adapted_user_ids)
            or not np.array_equal(dates, adapted_dates)
        ):
            raise ValueError("native and adapted OpenMHC caches are not aligned")
        self.embedding_dim = self.native.shape[-1] + self.adapted.shape[-1]
        split_path = (
            self.data_dir / "splits" / "sharable_users_seed42_2026_xs.json"
        )
        split_values = json.loads(split_path.read_text(encoding="utf-8"))
        if split not in split_values:
            raise ValueError(f"unknown OpenMHC split {split!r}")
        allowed = {str(item) for item in split_values[split]}
        embedding_lookup = {
            (str(participant), str(day)): index
            for index, (participant, day) in enumerate(zip(user_ids, dates))
        }
        by_participant: dict[str, list[tuple[np.datetime64, int]]] = {}
        for index, (participant, day) in enumerate(zip(user_ids, dates)):
            participant = str(participant)
            if participant in allowed:
                by_participant.setdefault(participant, []).append(
                    (np.datetime64(str(day), "D"), index)
                )
        for values in by_participant.values():
            values.sort()
        participant_dates = {
            participant: [day for day, _ in values]
            for participant, values in by_participant.items()
        }

        label_rows = pd.read_parquet(
            self.data_dir / "processed" / "daily_labels_lookup.parquet"
        ).to_dict("records")
        self.examples = []
        for row in label_rows:
            participant = str(row["user_id"])
            key = (participant, str(row["date"]))
            if participant not in allowed or key not in embedding_lookup:
                continue
            if not any(float(row[column]) >= 0 for column in OPENMHC_COLUMNS):
                continue
            current = np.datetime64(str(row["date"]), "D")
            first = current - np.timedelta64(self.history_days - 1, "D")
            dates_for_participant = participant_dates[participant]
            left = bisect_left(dates_for_participant, first)
            right = bisect_right(dates_for_participant, current)
            history = [
                (int((day - first).astype(int)), index)
                for day, index in by_participant[participant][left:right]
            ]
            if not history:
                continue
            self.examples.append(
                {"participant_id": participant, "row": row, "history": history}
            )
        self.participant_ids = [item["participant_id"] for item in self.examples]

    def __len__(self) -> int:
        return len(self.examples)

    @staticmethod
    def _targets(row: dict[str, Any]) -> dict[str, torch.Tensor]:
        targets = {}
        for column, task_id in OPENMHC_COLUMNS.items():
            value = float(row[column])
            targets[task_id] = _target_tensor(
                value if value >= 0 else float("nan"),
                column in OPENMHC_CATEGORICAL,
            )
        return targets

    def iter_targets(self) -> Iterable[dict[str, torch.Tensor]]:
        for example in self.examples:
            yield self._targets(example["row"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        values = np.zeros((self.history_days, self.embedding_dim), dtype=np.float32)
        present = np.zeros(self.history_days, dtype=bool)
        for slot, row_index in example["history"]:
            width = self.native.shape[-1]
            values[slot, :width] = self.native[row_index]
            values[slot, width:] = self.adapted[row_index]
            present[slot] = True
        return {
            "daily_embeddings": torch.from_numpy(values),
            "day_present": torch.from_numpy(present),
            "targets": self._targets(example["row"]),
            "participant_id": example["participant_id"],
            "cohort": "openmhc",
        }


__all__ = [
    "AffectiveJointEmbeddingDataset",
    "HRVMentalJointEmbeddingDataset",
    "McPhasesJointEmbeddingDataset",
    "OpenMHCAuxiliaryEmbeddingDataset",
    "PregnancyJointEmbeddingDataset",
    "load_aligned_embeddings",
]
