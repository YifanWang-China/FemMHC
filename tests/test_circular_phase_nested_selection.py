from __future__ import annotations

import numpy as np

from scripts.evaluate_circular_phase_nested_selection import (
    leave_one_participant_out_selection,
)


def test_leave_one_participant_out_selection_never_scores_held_rows() -> None:
    participant = np.asarray(["a", "a", "b", "b", "c", "c"])
    target = np.asarray([0, 1, 0, 1, 0, 1])
    candidates = {
        "simple": np.asarray([0, 1, 0, 1, 1, 0]),
        "complex": np.asarray([1, 0, 1, 0, 0, 1]),
    }
    selected, folds = leave_one_participant_out_selection(
        participant,
        target,
        candidates,
    )
    assert selected.shape == target.shape
    assert len(folds) == 3
    assert all(record["held_samples"] == 2 for record in folds)
    assert {record["selected_candidate"] for record in folds} <= set(candidates)
