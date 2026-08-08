from __future__ import annotations

import torch

from femmhc.cyclessm import (
    CausalGRUEncoder,
    CausalTransformerEncoder,
    CycleSSMEncoder,
    LastDayMLPEncoder,
    PersonalCausalMemory,
    count_trainable_parameters,
)


def test_personal_memory_is_strictly_causal() -> None:
    torch.manual_seed(2)
    memory = PersonalCausalMemory(8).eval()
    values = torch.randn(2, 5, 8)
    changed = values.clone()
    changed[:, 4] = 1e4
    with torch.inference_mode():
        original = memory(values)
        modified = memory(changed)
    assert torch.allclose(
        original.memory_states[:, :4], modified.memory_states[:, :4], atol=1e-6
    )
    assert torch.allclose(
        original.deviations[:, :4], modified.deviations[:, :4], atol=1e-6
    )


def test_cyclessm_is_causal_and_padding_safe() -> None:
    torch.manual_seed(3)
    model = CycleSSMEncoder(
        input_dim=16,
        hidden_dim=32,
        modes=4,
        maximum_days=6,
        dropout=0.0,
    ).eval()
    values = torch.randn(2, 6, 16)
    present = torch.tensor(
        [[False, True, True, True, True, True], [False, False, True, True, True, True]]
    )
    changed_padding = values.clone()
    changed_padding[~present] = torch.nan
    changed_future = values.clone()
    changed_future[:, 5] = 1000.0
    with torch.inference_mode():
        original = model(values, present)
        padded = model(changed_padding, present)
        future = model(changed_future, present)
    assert original.representation.shape == (2, 32)
    assert torch.allclose(original.representation, padded.representation, atol=1e-5)
    assert torch.allclose(
        original.sequence_states[:, :5], future.sequence_states[:, :5], atol=1e-5
    )
    periods = original.auxiliary["cycle_period_days"]
    assert torch.all(periods >= 18.0)
    assert torch.all(periods <= 45.0)


def test_temporal_controls_have_matching_interfaces_and_similar_budgets() -> None:
    constructors = (
        LastDayMLPEncoder,
        CausalGRUEncoder,
        CausalTransformerEncoder,
        CycleSSMEncoder,
    )
    values = torch.randn(3, 12, 64)
    present = torch.ones(3, 12, dtype=torch.bool)
    counts = []
    for constructor in constructors:
        model = constructor(
            input_dim=64,
            hidden_dim=32,
            maximum_days=12,
            modes=4,
            dropout=0.0,
        ).eval()
        with torch.inference_mode():
            output = model(values, present)
        assert output.representation.shape == (3, 32)
        assert output.sequence_states.shape == (3, 12, 32)
        counts.append(count_trainable_parameters(model))
    # The controls are deliberately close, not mathematically identical; the
    # exact trainable parameter count is always recorded in experiment output.
    assert max(counts) / min(counts) < 1.8
