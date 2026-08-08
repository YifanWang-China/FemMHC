from __future__ import annotations

from femmhc import FemMHCJointModel
from femmhc.cyclessm import CausalGRUEncoder, LastDayMLPEncoder


ARMS = {
    "last_day_shared": 168,
    "shared_backbone": 166,
    "mmoe": 136,
    "dual_path_router": 128,
}


def _parameters(model: FemMHCJointModel) -> int:
    return sum(value.numel() for value in model.parameters() if value.requires_grad)


def test_capacity_matched_arms_are_within_one_percent() -> None:
    models = {
        architecture: FemMHCJointModel(
            768,
            hidden,
            architecture=architecture,
            maximum_days=60,
            dropout=0.0,
            initialization_seed=17,
            routing_initial_logit=-2.0,
        )
        for architecture, hidden in ARMS.items()
    }
    target = _parameters(models["dual_path_router"])
    assert target == 1_983_696
    for architecture, model in models.items():
        relative_gap = abs(_parameters(model) - target) / target
        assert relative_gap <= 0.01, (architecture, _parameters(model), relative_gap)


def test_shared_controls_separate_non_temporal_and_gru_history() -> None:
    non_temporal = FemMHCJointModel(
        32,
        32,
        architecture="last_day_shared",
        maximum_days=14,
        dropout=0.0,
    )
    temporal = FemMHCJointModel(
        32,
        32,
        architecture="shared_backbone",
        maximum_days=14,
        dropout=0.0,
    )
    assert isinstance(non_temporal.state_encoder.general_temporal, LastDayMLPEncoder)
    assert isinstance(temporal.state_encoder.general_temporal, CausalGRUEncoder)
