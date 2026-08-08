"""Participant-safe female cohort view over the public OpenMHC-XS release."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


def preprocess_openmhc_day(
    values: np.ndarray,
    *,
    means: np.ndarray | None = None,
    stds: np.ndarray | None = None,
    normalization_channels: np.ndarray | None = None,
) -> np.ndarray:
    """Restore OpenMHC missingness and optionally apply its global priors.

    ``daily_hf`` stores missing values as zeros.  The official LSM2 evaluation
    first applies ``ZeroToNaNTransform`` and only then z-scores channels 0--6;
    training must use the same input semantics.
    """

    from data.transforms.nan_transforms import ZeroToNaNTransform

    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32).copy())
    tensor = ZeroToNaNTransform()(tensor)
    output = tensor.numpy()
    if normalization_channels is None:
        return output
    if means is None or stds is None:
        raise ValueError("means and stds are required when normalization channels are set")
    output[normalization_channels] = (
        output[normalization_channels] - means[normalization_channels, None]
    ) / stds[normalization_channels, None]
    return output


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
        source_users = source["user_id"]
        self.indices = [
            index for index, user in enumerate(source_users) if user in allowed
        ]
        self.participant_ids = [str(source_users[index]) for index in self.indices]
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

    def balanced_indices(self, max_examples: int, *, seed: int = 0) -> list[int]:
        """Choose days round-robin across participants for stable validation.

        The underlying Arrow table is participant ordered, so taking its first
        ``N`` rows can cover only a few people.  This sampler first shuffles days
        within each participant and then takes one day per participant per round.
        """

        if max_examples <= 0:
            raise ValueError("max_examples must be positive")
        groups: dict[str, list[int]] = defaultdict(list)
        for local_index, participant_id in enumerate(self.participant_ids):
            groups[participant_id].append(local_index)
        generator = np.random.default_rng(seed)
        participants = sorted(groups)
        generator.shuffle(participants)
        for indices in groups.values():
            generator.shuffle(indices)
        selected: list[int] = []
        round_index = 0
        while len(selected) < min(max_examples, len(self)):
            added = False
            for participant_id in participants:
                indices = groups[participant_id]
                if round_index < len(indices):
                    selected.append(indices[round_index])
                    added = True
                    if len(selected) == min(max_examples, len(self)):
                        break
            if not added:
                break
            round_index += 1
        return selected

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = self.source[self.indices[item]]
        values = np.asarray(row["values"], dtype=np.float32)
        if self.normalize:
            values = preprocess_openmhc_day(
                values,
                means=self.means,
                stds=self.stds,
                normalization_channels=self.normalization_channels,
            )
        else:
            values = preprocess_openmhc_day(values)
        return {
            "sensor_values": torch.from_numpy(values.copy()),
            "channel_present": torch.from_numpy(np.isfinite(values).any(axis=1)),
            "participant_id": row["user_id"],
            "date": row["date"],
            "source_index": self.indices[item],
        }


__all__ = ["OpenMHCFemaleDataset", "preprocess_openmhc_day"]
