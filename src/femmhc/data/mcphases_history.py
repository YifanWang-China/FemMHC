"""Leakage-safe causal history windows over cached mcPHASES day embeddings."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from femmhc.tasks import MCPHASES_TASKS, TaskDefinition


def _rows(processed_dir: Path) -> list[dict[str, str]]:
    with (processed_dir / "index.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _task_definition(task: str | TaskDefinition) -> TaskDefinition:
    if isinstance(task, TaskDefinition):
        return task
    try:
        return next(item for item in MCPHASES_TASKS if item.name == task)
    except StopIteration as error:
        raise ValueError(f"unknown mcPHASES task: {task}") from error


def mcphases_task_targets(
    processed_dir: Path,
    task: str | TaskDefinition,
) -> np.ndarray:
    """Build same-day or future-day targets without crossing study intervals."""

    processed_dir = Path(processed_dir).resolve()
    definition = _task_definition(task)
    rows = _rows(processed_dir)
    target = np.full(len(rows), np.nan, dtype=np.float32)
    if definition.kind == "regression":
        hormone_index = {"lh": 0, "estrogen": 1, "pdg": 2}[definition.name]
        target[:] = np.load(
            processed_dir / "hormones.npy", mmap_mode="r"
        )[:, hormone_index]
        return target
    if definition.label_column is None:
        raise ValueError(f"task {definition.name} has no label column")
    labels = np.load(processed_dir / "labels.npy", mmap_mode="r")
    if definition.target_offset_days == 0:
        observed = np.asarray(labels[:, definition.label_column])
        target[observed >= 0] = observed[observed >= 0]
        return target
    lookup = {
        (
            row["participant_id"],
            row["study_interval"],
            int(row["day_in_study"]),
        ): int(row["sample_index"])
        for row in rows
    }
    for row in rows:
        source_index = int(row["sample_index"])
        future_index = lookup.get(
            (
                row["participant_id"],
                row["study_interval"],
                int(row["day_in_study"]) + definition.target_offset_days,
            )
        )
        if future_index is None:
            continue
        value = int(labels[future_index, definition.label_column])
        if value >= 0:
            target[source_index] = value
    return target


class McPhasesEmbeddingHistoryDataset(Dataset[dict[str, Any]]):
    """Dense calendar-day history ending at each eligible prediction day."""

    def __init__(
        self,
        processed_dir: Path,
        embeddings_path: Path,
        *,
        task: str | TaskDefinition,
        history_days: int = 60,
        minimum_history_days: int = 7,
        split: str | None = None,
        participant_ids: Iterable[str] | None = None,
    ) -> None:
        if history_days <= 0 or minimum_history_days <= 0:
            raise ValueError("history lengths must be positive")
        if minimum_history_days > history_days:
            raise ValueError("minimum_history_days cannot exceed history_days")
        if split is not None and participant_ids is not None:
            raise ValueError("choose split or participant_ids, not both")
        self.processed_dir = Path(processed_dir).resolve()
        self.embeddings_path = Path(embeddings_path).resolve()
        self.task = _task_definition(task)
        self.history_days = int(history_days)
        self.minimum_history_days = int(minimum_history_days)
        self.rows = _rows(self.processed_dir)
        self.embeddings = np.load(self.embeddings_path, mmap_mode="r")
        if self.embeddings.ndim != 2 or self.embeddings.shape[0] != len(self.rows):
            raise ValueError(
                f"embedding shape {self.embeddings.shape} does not match {len(self.rows)} rows"
            )
        self.targets = mcphases_task_targets(self.processed_dir, self.task)

        allowed: set[str] | None = None
        if split is not None:
            splits = json.loads(
                (self.processed_dir / "participant_splits.json").read_text(
                    encoding="utf-8"
                )
            )
            if split not in splits:
                raise ValueError(f"unknown split {split!r}")
            allowed = {str(item) for item in splits[split]}
        elif participant_ids is not None:
            allowed = {str(item) for item in participant_ids}

        by_interval: dict[tuple[str, str], list[tuple[int, int]]] = {}
        for row in self.rows:
            participant = row["participant_id"]
            if allowed is not None and participant not in allowed:
                continue
            sample_index = int(row["sample_index"])
            by_interval.setdefault(
                (participant, row["study_interval"]), []
            ).append((int(row["day_in_study"]), sample_index))
        for values in by_interval.values():
            values.sort()

        self.examples: list[dict[str, Any]] = []
        for (participant, interval), days in by_interval.items():
            for current_day, current_index in days:
                if not np.isfinite(self.targets[current_index]):
                    continue
                if not np.isfinite(self.embeddings[current_index]).all():
                    continue
                first_day = current_day - self.history_days + 1
                history = [
                    (day - first_day, sample_index)
                    for day, sample_index in days
                    if first_day <= day <= current_day
                    and np.isfinite(self.embeddings[sample_index]).all()
                ]
                if len(history) < self.minimum_history_days:
                    continue
                if not history or history[-1][1] != current_index:
                    raise RuntimeError("the current prediction day is missing from its history")
                self.examples.append(
                    {
                        "participant_id": participant,
                        "study_interval": interval,
                        "day_in_study": current_day,
                        "sample_index": current_index,
                        "history": history,
                    }
                )

    @property
    def participants(self) -> np.ndarray:
        return np.asarray([item["participant_id"] for item in self.examples])

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        values = np.zeros(
            (self.history_days, self.embeddings.shape[1]), dtype=np.float32
        )
        present = np.zeros(self.history_days, dtype=bool)
        for slot, sample_index in example["history"]:
            values[slot] = self.embeddings[sample_index]
            present[slot] = True
        target_value = self.targets[example["sample_index"]]
        target = (
            torch.tensor(float(target_value), dtype=torch.float32)
            if self.task.kind == "regression"
            else torch.tensor(int(target_value), dtype=torch.long)
        )
        return {
            "daily_embeddings": torch.from_numpy(values),
            "day_present": torch.from_numpy(present),
            "target": target,
            "participant_id": example["participant_id"],
            "study_interval": example["study_interval"],
            "day_in_study": int(example["day_in_study"]),
            "sample_index": int(example["sample_index"]),
            "history_count": int(present.sum()),
        }

    def close(self) -> None:
        memory_map = getattr(self.embeddings, "_mmap", None)
        if memory_map is not None:
            memory_map.close()

    def __del__(self) -> None:
        self.close()


__all__ = ["McPhasesEmbeddingHistoryDataset", "mcphases_task_targets"]
