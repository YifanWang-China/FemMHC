"""Participant-safe female cohort view over the public OpenMHC-XS release."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class OpenMHCFemaleDataset(Dataset[dict[str, Any]]):
    """Daily OpenMHC records restricted by source sex label and official split."""

    def __init__(
        self,
        root: Path,
        *,
        split: str,
        sex: str = "Female",
        normalize: bool = True,
    ) -> None:
        from datasets import load_from_disk

        self.root = Path(root).resolve()
        split_path = self.root / "splits" / "sharable_users_seed42_2026_xs.json"
        splits = json.loads(split_path.read_text(encoding="utf-8"))
        if split not in splits:
            raise ValueError(f"unknown split {split!r}; choose from {sorted(splits)}")
        label_path = self.root / "labels" / "last_labels.json"
        labels = json.loads(label_path.read_text(encoding="utf-8"))["BiologicalSex"]
        sex_by_user = {
            user: item["values"][-1]
            for user, item in labels.items()
            if item.get("values")
        }
        allowed = {
            user for user in splits[split] if sex_by_user.get(user) == sex
        }
        source = load_from_disk(str(self.root / "processed" / "daily_hf"))
        self.indices = [
            index for index, user in enumerate(source["user_id"]) if user in allowed
        ]
        self.source = source
        self.participants = allowed
        self.sex = sex
        self.normalize = bool(normalize)
        if self.normalize:
            stats = json.loads(
                (self.root / "processed" / "normalization_stats.json").read_text(
                    encoding="utf-8"
                )
            )
            self.normalization_channels = np.asarray(stats["channels"], dtype=np.int64)
            self.means = np.asarray(stats["means"], dtype=np.float32)
            self.stds = np.asarray(stats["stds"], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = self.source[self.indices[item]]
        values = np.asarray(row["values"], dtype=np.float32)
        if self.normalize:
            values[self.normalization_channels] = (
                values[self.normalization_channels]
                - self.means[self.normalization_channels, None]
            ) / self.stds[self.normalization_channels, None]
        return {
            "sensor_values": torch.from_numpy(values.copy()),
            "channel_present": torch.from_numpy(np.isfinite(values).any(axis=1)),
            "participant_id": row["user_id"],
            "date": row["date"],
            "source_index": self.indices[item],
        }


__all__ = ["OpenMHCFemaleDataset"]
