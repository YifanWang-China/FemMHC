import numpy as np

from scripts.evaluate_mcphases_label_efficiency import stratified_label_subset


def test_categorical_low_label_subset_contains_every_class() -> None:
    target = np.tile(np.arange(4), 25)
    indices = stratified_label_subset(target, 0.01, kind="multiclass", seed=7)
    np.testing.assert_array_equal(np.unique(target[indices]), np.arange(4))
    assert len(indices) == 4


def test_regression_low_label_subset_has_minimum_stable_budget() -> None:
    target = np.linspace(0, 1, 100)
    indices = stratified_label_subset(target, 0.01, kind="regression", seed=11)
    assert len(indices) == 10
    assert len(np.unique(indices)) == len(indices)


def test_full_fraction_returns_every_label() -> None:
    target = np.arange(13)
    indices = stratified_label_subset(target, 1.0, kind="regression", seed=13)
    np.testing.assert_array_equal(indices, np.arange(13))
