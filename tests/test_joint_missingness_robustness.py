import numpy as np

from scripts.evaluate_joint_missingness_robustness import (
    MissingnessScenario,
    perturb_present,
)


def test_random_missingness_is_deterministic_and_keeps_one_day() -> None:
    present = np.ones((3, 10), dtype=bool)
    scenario = MissingnessScenario("random", "random_fraction", 0.40)
    first = perturb_present(present, scenario, seed=7)
    second = perturb_present(present, scenario, seed=7)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first.sum(axis=1), np.asarray([6, 6, 6]))


def test_latest_missingness_removes_most_recent_observed_days() -> None:
    present = np.asarray([[0, 1, 0, 1, 1]], dtype=bool)
    scenario = MissingnessScenario("latest", "latest_observed", 2)
    result = perturb_present(present, scenario, seed=11)
    np.testing.assert_array_equal(result, np.asarray([[0, 1, 0, 0, 0]], dtype=bool))


def test_recent_calendar_window_preserves_only_tail_slots() -> None:
    present = np.asarray([[1, 1, 0, 1, 0, 1]], dtype=bool)
    scenario = MissingnessScenario("recent", "recent_calendar", 3)
    result = perturb_present(present, scenario, seed=13)
    np.testing.assert_array_equal(result, np.asarray([[0, 0, 0, 1, 0, 1]], dtype=bool))


def test_contiguous_mask_does_not_remove_only_observation() -> None:
    present = np.asarray([[0, 0, 1, 0]], dtype=bool)
    scenario = MissingnessScenario("gap", "contiguous_observed", 7)
    result = perturb_present(present, scenario, seed=17)
    np.testing.assert_array_equal(result, present)
