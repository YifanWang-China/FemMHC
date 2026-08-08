from __future__ import annotations

import numpy as np

from scripts.evaluate_affective_nested_group_cv import _nested_predictions


def test_nested_group_cv_produces_one_prediction_per_sample() -> None:
    rng = np.random.default_rng(4)
    participant = np.repeat([f"p{index:02d}" for index in range(12)], 4)
    features = rng.normal(size=(participant.size, 8))
    target = np.tile([0.0, 1.0, 0.0, 1.0], 12)
    prediction, baseline, folds = _nested_predictions(
        features,
        target,
        participant,
        kind="classification",
        outer_folds=3,
        inner_folds=2,
        seed=42,
    )
    assert len(folds) == 3
    assert np.isfinite(prediction).all()
    assert np.isfinite(baseline).all()
    assert all(record["test_participants"] == 4 for record in folds)
