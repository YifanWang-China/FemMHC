"""Generic participant-safe temporal views over processed daily datasets."""

from __future__ import annotations

from datetime import date
from typing import Any

import torch
from torch.utils.data import Dataset


class AdjacentDayPairDataset(Dataset[dict[str, Any]]):
    """Chronological within-participant pairs without crossing identities."""

    def __init__(
        self,
        daily_dataset: Dataset[dict[str, Any]],
        *,
        maximum_gap_days: int = 1,
    ) -> None:
        if maximum_gap_days <= 0:
            raise ValueError("maximum_gap_days must be positive")
        if not hasattr(daily_dataset, "rows"):
            raise TypeError("daily_dataset must expose participant/date rows")
        self.daily = daily_dataset
        by_participant: dict[str, list[tuple[date, int]]] = {}
        for local_index, row in enumerate(daily_dataset.rows):
            by_participant.setdefault(str(row["participant_id"]), []).append(
                (date.fromisoformat(str(row["date"])), local_index)
            )
        pairs: list[tuple[int, int, int]] = []
        for values in by_participant.values():
            ordered = sorted(values)
            for (earlier_date, earlier), (later_date, later) in zip(
                ordered, ordered[1:]
            ):
                gap = (later_date - earlier_date).days
                if 0 < gap <= maximum_gap_days:
                    pairs.append((earlier, later, gap))
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        earlier, later, gap = self.pairs[index]
        return {
            "earlier": self.daily[earlier],
            "later": self.daily[later],
            "gap_days": torch.tensor(gap, dtype=torch.long),
        }

    def close(self) -> None:
        close = getattr(self.daily, "close", None)
        if close is not None:
            close()

    def __del__(self) -> None:
        self.close()


__all__ = ["AdjacentDayPairDataset"]
