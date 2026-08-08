from __future__ import annotations

import numpy as np
import pytest

from femmhc.statistics import holm_adjust, paired_cluster_bootstrap


def test_paired_cluster_bootstrap_preserves_candidate_direction() -> None:
    participants = np.repeat(["a", "b", "c", "d", "e"], 3)
    baseline = np.arange(len(participants), dtype=np.float64)
    candidate = baseline - 2.0

    result = paired_cluster_bootstrap(
        participants,
        lambda indices: (
            float(np.mean(baseline[indices])),
            float(np.mean(candidate[indices])),
        ),
        lower_is_better=True,
        replicates=200,
        seed=7,
    )

    assert result.eligible
    assert result.estimate == 2.0
    assert result.confidence_low == pytest.approx(2.0)
    assert result.confidence_high == pytest.approx(2.0)
    assert result.probability_candidate_better == 1.0


def test_paired_cluster_bootstrap_marks_too_few_participants() -> None:
    result = paired_cluster_bootstrap(
        ["a", "b", "c", "d"],
        lambda indices: (1.0, 0.0),
        lower_is_better=True,
        replicates=100,
    )

    assert not result.eligible
    assert result.clusters == 4
    assert result.confidence_low is None


def test_holm_adjust_is_monotone_in_rank_order() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.04})

    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.06}
