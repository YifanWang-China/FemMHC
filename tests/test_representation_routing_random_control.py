import unittest

import numpy as np

from scripts.evaluate_representation_routing_random_control import (
    best_rank,
    normalized_utility,
)


class RepresentationRoutingRandomControlTests(unittest.TestCase):
    def test_normalized_utility_preserves_order(self) -> None:
        values = normalized_utility(np.asarray([-3.0, 1.0, 5.0]))
        np.testing.assert_allclose(values, np.asarray([0.0, 0.5, 1.0]))

    def test_constant_metrics_receive_neutral_utility(self) -> None:
        values = normalized_utility(np.asarray([2.0, 2.0, 2.0]))
        np.testing.assert_allclose(values, np.asarray([0.5, 0.5, 0.5]))

    def test_best_rank_averages_ties(self) -> None:
        values = np.asarray([3.0, 2.0, 2.0, 1.0])
        self.assertEqual(best_rank(values, 0), 1.0)
        self.assertEqual(best_rank(values, 1), 2.5)
        self.assertEqual(best_rank(values, 3), 4.0)


if __name__ == "__main__":
    unittest.main()
