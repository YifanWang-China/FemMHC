import numpy as np
import torch

from femmhc import FemMHCJointModel
from scripts.aggregate_mcphases_single_vs_joint import _score
from scripts.train_mcphases_single_vs_joint import (
    JOINT_TASK_IDS,
    TASK_GROUPS,
    _task_specs,
)


def test_single_task_groups_cover_six_prespecified_outputs() -> None:
    task_ids = [task_id for group in TASK_GROUPS.values() for task_id in group]
    assert len(task_ids) == 6
    assert len(set(task_ids)) == 6
    assert len(TASK_GROUPS["onset"]) == 2


def test_subset_models_emit_phase_and_nested_onset_outputs() -> None:
    embeddings = torch.randn(2, 7, 768)
    present = torch.ones(2, 7, dtype=torch.bool)

    phase_tasks = _task_specs(TASK_GROUPS["cycle"])
    phase_model = FemMHCJointModel(
        768,
        hidden_dim=16,
        tasks=phase_tasks,
        maximum_days=7,
        architecture="dual_path_router",
        dropout=0.0,
        initialization_seed=17,
    )
    phase = phase_model(embeddings, present, task_ids=TASK_GROUPS["cycle"])
    assert set(phase.predictions) == {"mcphases/cycle_phase"}

    onset_tasks = _task_specs(TASK_GROUPS["onset"])
    onset_model = FemMHCJointModel(
        768,
        hidden_dim=16,
        tasks=onset_tasks,
        maximum_days=7,
        architecture="dual_path_router",
        dropout=0.0,
        initialization_seed=17,
    )
    onset = onset_model(embeddings, present, task_ids=TASK_GROUPS["onset"])
    assert set(onset.predictions) == set(TASK_GROUPS["onset"])
    probability_24 = onset.predictions[TASK_GROUPS["onset"][0]].probabilities[:, 1]
    probability_72 = onset.predictions[TASK_GROUPS["onset"][1]].probabilities[:, 1]
    assert torch.all(probability_24 <= probability_72)


def test_task_metrics_use_expected_orientation() -> None:
    target = np.array([0, 1, 2, 3])
    assert _score("mcphases/cycle_phase", target, target) == 1.0
    ordinal = _score("mcphases/cramps", target, target + 0.5)
    assert ordinal == 0.5
    onset = _score(
        "mcphases/menstrual_onset_24h",
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.2, 0.8, 0.9]),
    )
    assert onset == 1.0


def test_all_experiment_arms_can_share_bitwise_initialization() -> None:
    tasks = _task_specs(JOINT_TASK_IDS)
    torch.manual_seed(42)
    first = FemMHCJointModel(
        768,
        hidden_dim=16,
        tasks=tasks,
        maximum_days=7,
        architecture="dual_path_router",
        dropout=0.0,
        initialization_seed=42,
    )
    torch.manual_seed(42)
    second = FemMHCJointModel(
        768,
        hidden_dim=16,
        tasks=tasks,
        maximum_days=7,
        architecture="dual_path_router",
        dropout=0.0,
        initialization_seed=42,
    )
    assert first.state_dict().keys() == second.state_dict().keys()
    assert all(
        torch.equal(first.state_dict()[key], second.state_dict()[key])
        for key in first.state_dict()
    )
