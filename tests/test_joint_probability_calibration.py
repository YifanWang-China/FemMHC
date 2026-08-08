import numpy as np

from scripts.evaluate_joint_probability_calibration import (
    categorical_nll,
    fit_temperature,
    onset_bins,
    onset_probabilities,
    softmax_temperature,
)


def test_temperature_fit_reduces_overconfident_nll() -> None:
    logits = np.asarray([[8.0, 0.0], [8.0, 0.0], [0.0, 8.0], [0.0, 8.0]])
    target = np.asarray([0, 1, 1, 0])
    temperature = fit_temperature(logits, target)
    assert temperature > 1.0
    assert categorical_nll(logits, target, temperature) < categorical_nll(logits, target, 1.0)


def test_onset_temperature_preserves_nested_probabilities() -> None:
    logits = np.asarray([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
    probabilities = softmax_temperature(logits, 2.5)
    within_24h, within_72h = onset_probabilities(probabilities)
    assert np.all(within_24h <= within_72h)


def test_onset_bins_encode_three_mutually_exclusive_windows() -> None:
    y24 = np.asarray([1, 0, 0])
    y72 = np.asarray([1, 1, 0])
    np.testing.assert_array_equal(onset_bins(y24, y72), np.asarray([0, 1, 2]))
