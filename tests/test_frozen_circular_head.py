from __future__ import annotations

import torch

from femmhc import FemMHCJointModel, partial_multitask_loss
from scripts.train_femmhc_frozen_circular_head import (
    PHASE_TASK_ID,
    assert_frozen_state_unchanged,
    build_frozen_circular_model,
    snapshot_frozen_state,
)
from scripts.select_frozen_circular_head_train_cv import (
    HEAD_FAMILIES,
    CircularProjector,
    build_phase_head,
)


def _base_artifact() -> dict[str, object]:
    model = FemMHCJointModel(
        input_dim=16,
        hidden_dim=16,
        maximum_days=5,
        architecture="dual_path_router",
        dropout=0.0,
        initialization_seed=7,
        routing_initial_logit=-2.0,
    )
    return {
        "architecture": "dual_path_router",
        "input_dim": 16,
        "hidden_dim": 16,
        "maximum_days": 5,
        "dropout": 0.0,
        "routing_initial_logit": -2.0,
        "model_state_dict": model.state_dict(),
    }


def test_frozen_circular_transplant_loads_base_and_exposes_only_projector() -> None:
    artifact = _base_artifact()
    model, audit = build_frozen_circular_model(artifact, initialization_seed=11)
    assert audit["trainable_parameters"] == 66
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    assert trainable == [
        "cycle_phase_projector.0.weight",
        "cycle_phase_projector.0.bias",
        "cycle_phase_projector.1.weight",
        "cycle_phase_projector.1.bias",
    ]
    for name, expected in artifact["model_state_dict"].items():
        assert torch.equal(model.state_dict()[name], expected)


def test_optimizer_step_changes_head_but_not_frozen_model() -> None:
    model, _ = build_frozen_circular_model(_base_artifact(), initialization_seed=11)
    frozen = snapshot_frozen_state(model)
    initial_head = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if value.requires_grad
    }
    optimizer = torch.optim.AdamW(
        [value for value in model.parameters() if value.requires_grad],
        lr=1e-2,
    )
    embeddings = torch.randn(8, 5, 16)
    present = torch.ones(8, 5, dtype=torch.bool)
    targets = {PHASE_TASK_ID: torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])}
    output = model(embeddings, present, task_ids=(PHASE_TASK_ID,))
    loss = partial_multitask_loss(
        output,
        targets,
        phase_geometry_weight=0.1,
    ).total
    loss.backward()
    optimizer.step()
    assert_frozen_state_unchanged(model, frozen)
    assert any(
        not torch.equal(initial_head[name], value.detach())
        for name, value in model.named_parameters()
        if value.requires_grad
    )


def test_cached_circular_projector_outputs_four_probabilistic_logits() -> None:
    model = CircularProjector(16, initialization_seed=5)
    vector, logits = model(torch.randn(7, 16))
    assert vector.shape == (7, 2)
    assert logits.shape == (7, 4)
    assert torch.allclose(logits.softmax(dim=-1).sum(dim=-1), torch.ones(7))


def test_all_phase_control_heads_have_expected_parameter_budget_and_shape() -> None:
    expected = {
        "circular_fixed": 514,
        "circular_permuted": 514,
        "learnable_prototype": 522,
        "bottleneck_softmax": 526,
        "linear_matched": 516,
        "linear_softmax": 772,
    }
    for family in HEAD_FAMILIES:
        model = build_phase_head(128, head_family=family, initialization_seed=5)
        _, logits = model(torch.randn(7, 128))
        assert logits.shape == (7, 4)
        assert sum(value.numel() for value in model.parameters()) == expected[family]
