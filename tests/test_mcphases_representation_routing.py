from __future__ import annotations

import numpy as np

from scripts.evaluate_mcphases_representation_routing import (
    ProbePrediction,
    observed_mask,
    oriented,
    participant_weights,
    task_metric,
)


def test_observed_mask_keeps_negative_regression_values_only() -> None:
    target = np.asarray([-2.0, -1.0, 0.0, np.nan])
    assert observed_mask("regression", target).tolist() == [True, True, True, False]
    assert observed_mask("ordinal", target).tolist() == [False, False, True, False]


def test_participant_weights_balance_total_mass() -> None:
    participants = np.asarray(["a", "a", "a", "b"])
    weights = participant_weights(participants)
    assert np.isclose(weights[:3].sum(), weights[3])


def test_binary_metric_requires_both_classes() -> None:
    prediction = ProbePrediction(
        hard=np.asarray([0, 0]),
        positive_probability=np.asarray([0.1, 0.2]),
    )
    assert task_metric(
        np.asarray([0, 0]),
        prediction,
        kind="binary",
        classes=2,
    ) is None


def test_metric_orientation_matches_task_direction() -> None:
    assert oriented(0.8, kind="binary") == 0.8
    assert oriented(0.8, kind="multiclass") == 0.8
    assert oriented(0.8, kind="ordinal") == -0.8
    assert oriented(0.8, kind="regression") == -0.8
