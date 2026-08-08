from __future__ import annotations

import numpy as np

from femmhc.tasks import MCPHASES_TASKS
from scripts.evaluate_mcphases_nested_loso import (
    _paired_bootstrap,
    nested_loso_predictions,
)


def _task(name: str):
    return next(task for task in MCPHASES_TASKS if task.name == name)


def test_nested_loso_writes_prediction_for_every_held_out_sample() -> None:
    generator = np.random.default_rng(7)
    participants = np.repeat(np.asarray(["a", "b", "c", "d", "e", "f"]), 8)
    target = np.tile(np.asarray([0, 1, 2, 3, 0, 1, 2, 3]), 6).astype(float)
    embeddings = generator.normal(size=(len(target), 6)).astype(np.float32)
    embeddings[:, :4] += np.eye(4, dtype=np.float32)[target.astype(int)]
    usable = np.ones(len(target), dtype=bool)

    prediction, score, selected = nested_loso_predictions(
        _task("cycle_phase"),
        embeddings,
        target,
        participants,
        usable,
        inner_folds=2,
        jobs=1,
        seed=11,
    )

    assert np.isfinite(prediction).all()
    assert np.isfinite(score).all()
    assert score.shape == (len(target), 4)
    assert sum(selected.values()) == 6


def test_paired_bootstrap_orients_lower_ordinal_error_as_improvement() -> None:
    task = _task("flow_volume")
    y = np.tile(np.asarray([0, 1, 2, 3]), 6)
    participants = np.repeat(np.asarray(["a", "b", "c", "d", "e", "f"]), 4)
    baseline_score = y.astype(float) + 1.0
    candidate_score = y.astype(float) + 0.2
    baseline_prediction = np.rint(baseline_score).astype(int)
    candidate_prediction = np.rint(candidate_score).astype(int)

    result = _paired_bootstrap(
        task,
        y,
        baseline_prediction,
        baseline_score,
        candidate_prediction,
        candidate_score,
        participants,
        draws=100,
        seed=17,
    )

    assert result["relative_improvement_percent"] > 0
    assert result["paired_bootstrap_ci_low"] > 0
    assert result["paired_bootstrap_probability_improved"] == 1.0


def test_nested_loso_supports_one_dimensional_ordinal_scores() -> None:
    generator = np.random.default_rng(19)
    participants = np.repeat(np.asarray(["a", "b", "c", "d", "e", "f"]), 7)
    target = np.tile(np.arange(7), 6).astype(float)
    embeddings = generator.normal(size=(len(target), 5)).astype(np.float32)
    embeddings[:, 0] += target
    usable = np.ones(len(target), dtype=bool)

    prediction, score, selected = nested_loso_predictions(
        _task("flow_volume"),
        embeddings,
        target,
        participants,
        usable,
        inner_folds=2,
        jobs=1,
        seed=23,
    )

    assert prediction.shape == target.shape
    assert score.shape == target.shape
    assert np.isfinite(score).all()
    assert sum(selected.values()) == 6


def test_nested_loso_supports_continuous_hormone_regression() -> None:
    generator = np.random.default_rng(29)
    participants = np.repeat(np.asarray(["a", "b", "c", "d", "e", "f"]), 10)
    embeddings = generator.normal(size=(len(participants), 6)).astype(np.float32)
    target = 3.0 * embeddings[:, 0] - 1.5 * embeddings[:, 1] + 0.2
    usable = np.ones(len(target), dtype=bool)

    prediction, score, selected = nested_loso_predictions(
        _task("lh"),
        embeddings,
        target,
        participants,
        usable,
        inner_folds=2,
        jobs=1,
        seed=31,
    )

    assert prediction.shape == target.shape
    np.testing.assert_allclose(prediction, score)
    assert np.isfinite(score).all()
    assert np.mean(np.abs(score - target)) < 0.2
    assert sum(selected.values()) == 6
