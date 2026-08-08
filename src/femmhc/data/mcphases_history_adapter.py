"""Raw mcPHASES days paired with strictly previous frozen daily embeddings.

The history vectors are an immutable reference representation.  They are used
only to create a personal state for the current raw wearable day; they never
include the current day or future wearable observations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ..tasks import MCPHASES_TASKS, TaskDefinition
from .dataset import McPhasesDataset
from .mcphases_history import mcphases_task_targets


def _split_ids(processed_dir: Path, split: str) -> set[str]:
    splits = json.loads(
        (Path(processed_dir) / "participant_splits.json").read_text(encoding="utf-8")
    )
    if split not in splits:
        raise ValueError(f"unknown split {split!r}; choose from {sorted(splits)}")
    return {str(participant) for participant in splits[split]}


class McPhasesHistoryAdapterDataset(Dataset[dict[str, Any]]):
    """Current raw day plus an immutable, causal history representation.

    ``history_embeddings[s]`` must be a wearable-only representation for day
    ``s`` produced by a model that was not trained on this dataset's validation
    or test labels.  Windows use calendar days ``[t-history_days, t-1]``;
    importantly, the current day ``t`` cannot enter its own control state.
    """

    def __init__(
        self,
        processed_dir: Path,
        history_embeddings: Path,
        *,
        split: str,
        history_days: int = 60,
        minimum_history_days: int = 0,
        normalize: bool = True,
        require_usable: bool = True,
        require_target: bool = True,
        tasks: tuple[TaskDefinition, ...] = MCPHASES_TASKS,
    ) -> None:
        if history_days <= 0:
            raise ValueError("history_days must be positive")
        if not 0 <= minimum_history_days <= history_days:
            raise ValueError("minimum_history_days must be in [0, history_days]")
        self.processed_dir = Path(processed_dir).resolve()
        self.history_days = int(history_days)
        self.minimum_history_days = int(minimum_history_days)
        self.tasks = tuple(tasks)
        self.days = McPhasesDataset(
            self.processed_dir,
            split=split,
            normalize=normalize,
            require_usable=require_usable,
        )
        self.history_embeddings = np.load(Path(history_embeddings), mmap_mode="r")
        if self.history_embeddings.ndim != 2:
            raise ValueError("history_embeddings must have shape (samples, embed_dim)")
        if self.history_embeddings.shape[0] != len(self.days.rows):
            raise ValueError("history_embeddings and mcPHASES index are misaligned")

        allowed = _split_ids(self.processed_dir, split)
        self.target_arrays = {
            task.name: mcphases_task_targets(self.processed_dir, task)
            for task in self.tasks
        }
        allowed_indices = set(self.days.sample_indices)
        position_by_index = {
            sample_index: position
            for position, sample_index in enumerate(self.days.sample_indices)
        }
        by_interval: dict[tuple[str, str], list[tuple[int, int]]] = {}
        for row in self.days.rows:
            participant = str(row["participant_id"])
            sample_index = int(row["sample_index"])
            if participant in allowed and sample_index in allowed_indices:
                by_interval.setdefault(
                    (participant, str(row["study_interval"])),
                    [],
                ).append((int(row["day_in_study"]), sample_index))
        for values in by_interval.values():
            values.sort()

        self.examples: list[dict[str, Any]] = []
        for (participant, interval), days in by_interval.items():
            for current_day, current_index in days:
                if require_target and not any(
                    np.isfinite(target[current_index])
                    for target in self.target_arrays.values()
                ):
                    continue
                first_history_day = current_day - self.history_days
                history = [
                    (day - first_history_day, sample_index)
                    for day, sample_index in days
                    if first_history_day <= day < current_day
                    and np.isfinite(self.history_embeddings[sample_index]).all()
                ]
                if len(history) < self.minimum_history_days:
                    continue
                # This guards against accidental same-day or future leakage if
                # the window logic changes later.
                if any(slot < 0 or slot >= self.history_days for slot, _ in history):
                    raise RuntimeError("history slot is outside the requested causal window")
                self.examples.append(
                    {
                        "participant_id": participant,
                        "study_interval": interval,
                        "day_in_study": current_day,
                        "sample_index": current_index,
                        "dataset_position": position_by_index[current_index],
                        "history": history,
                    }
                )
        self.participant_ids = [item["participant_id"] for item in self.examples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        item = dict(self.days[example["dataset_position"]])
        history = np.zeros(
            (self.history_days, self.history_embeddings.shape[1]), dtype=np.float32
        )
        present = np.zeros(self.history_days, dtype=bool)
        for slot, sample_index in example["history"]:
            history[slot] = self.history_embeddings[sample_index]
            present[slot] = True
        if int(item["sample_index"]) != int(example["sample_index"]):
            raise RuntimeError("current raw day and history target are misaligned")
        targets = {}
        for task in self.tasks:
            value = float(self.target_arrays[task.name][example["sample_index"]])
            targets[task.name] = (
                torch.tensor(value, dtype=torch.float32)
                if task.kind == "regression"
                else torch.tensor(-1 if not np.isfinite(value) else int(value), dtype=torch.long)
            )
        item.update(
            {
                "history_embeddings": torch.from_numpy(history),
                "history_present": torch.from_numpy(present),
                "history_count": int(present.sum()),
                "targets": targets,
            }
        )
        return item

    def close(self) -> None:
        memory_map = getattr(self.history_embeddings, "_mmap", None)
        if memory_map is not None:
            memory_map.close()

    def __del__(self) -> None:
        self.close()


__all__ = ["McPhasesHistoryAdapterDataset"]
