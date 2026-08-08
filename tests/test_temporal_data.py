from __future__ import annotations

import torch

from femmhc.data import AdjacentDayPairDataset
from femmhc.objectives import (
    PhysiologyChangeHead,
    adjacent_day_contrastive_loss,
    daily_sensor_statistics,
    physiology_change_loss,
)


class _Daily:
    def __init__(self) -> None:
        self.rows = [
            {"participant_id": "a", "date": "2025-01-01"},
            {"participant_id": "a", "date": "2025-01-02"},
            {"participant_id": "a", "date": "2025-01-05"},
            {"participant_id": "b", "date": "2025-01-01"},
            {"participant_id": "b", "date": "2025-01-02"},
        ]

    def __getitem__(self, index: int):
        return {"index": index}

    def __len__(self) -> int:
        return len(self.rows)


def test_adjacent_day_pairs_do_not_cross_participants_or_gaps() -> None:
    pairs = AdjacentDayPairDataset(_Daily())
    assert len(pairs) == 2
    assert pairs[0]["earlier"]["index"] == 0
    assert pairs[0]["later"]["index"] == 1
    assert pairs[1]["earlier"]["index"] == 3
    assert pairs[1]["later"]["index"] == 4


def test_adjacent_day_contrastive_loss_prefers_matching_pairs() -> None:
    matching = torch.eye(4)
    shuffled = matching.roll(1, dims=0)
    good = adjacent_day_contrastive_loss(matching, matching, temperature=0.1)
    bad = adjacent_day_contrastive_loss(matching, shuffled, temperature=0.1)
    assert good < bad


def test_physiology_change_target_ignores_missing_samples() -> None:
    values = torch.tensor(
        [
            [[1.0, 3.0, torch.nan], [2.0, 2.0, 2.0]],
            [[4.0, 4.0, 4.0], [torch.nan, 1.0, 3.0]],
        ]
    )
    statistics = daily_sensor_statistics(values)
    assert statistics.shape == (2, 4)
    assert torch.allclose(statistics[:, :2], torch.tensor([[2.0, 2.0], [4.0, 2.0]]))
    assert torch.isfinite(statistics).all()


def test_physiology_change_loss_is_finite() -> None:
    torch.manual_seed(11)
    head = PhysiologyChangeHead(embed_dim=8, channels=3, hidden_dim=4)
    earlier_embedding = torch.randn(2, 8)
    later_embedding = torch.randn(2, 8)
    earlier_values = torch.randn(2, 3, 10)
    later_values = torch.randn(2, 3, 10)
    earlier_values[0, 1, :3] = torch.nan
    loss = physiology_change_loss(
        head,
        earlier_embedding,
        later_embedding,
        earlier_values,
        later_values,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
