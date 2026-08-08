from __future__ import annotations

import torch

from femmhc.affective import (
    AffectiveTaskHeads,
    INPHRSYM_TASKS,
    PersonalBaselineTemporalEncoder,
)
from femmhc.heads import ProbabilisticOutput


def test_personal_baseline_temporal_encoder_ignores_padded_values() -> None:
    torch.manual_seed(7)
    model = PersonalBaselineTemporalEncoder(
        8,
        maximum_days=4,
        temporal_layers=1,
        temporal_heads=2,
        dropout=0.0,
    ).eval()
    values = torch.randn(2, 4, 8)
    present = torch.tensor([[False, True, True, True], [False, False, True, True]])
    changed = values.clone()
    changed[~present] = torch.nan
    with torch.inference_mode():
        first = model(values, present)
        second = model(changed, present)
    assert first.representation.shape == (2, 8)
    assert torch.allclose(first.representation, second.representation, atol=1e-5)
    assert torch.all(first.day_attention[~present] == 0)
    assert torch.allclose(first.day_attention.sum(dim=1), torch.ones(2))


def test_affective_task_heads_match_task_kinds() -> None:
    model = AffectiveTaskHeads(8, INPHRSYM_TASKS, adapter_bottleneck=4)
    output = model(torch.randn(3, 8))
    assert set(output) == {task.name for task in INPHRSYM_TASKS}
    assert isinstance(output["next_high_anxiety"], ProbabilisticOutput)
    assert output["next_high_anxiety"].probabilities.shape == (3, 2)
    assert output["next_anxiety_severity"].probabilities.shape == (3, 4)
