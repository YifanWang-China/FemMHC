from __future__ import annotations

from femmhc.multicohort import FemaleCohort, square_root_sampling_probabilities
from femmhc.sensors import SensorDescriptor


class _SizedDataset:
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size


def _cohort(name: str, size: int) -> FemaleCohort:
    return FemaleCohort(
        name=name,
        dataset=_SizedDataset(size),
        descriptors=(SensorDescriptor(name),),
    )


def test_square_root_sampling_balances_unequal_cohorts() -> None:
    probabilities = square_root_sampling_probabilities(
        [_cohort("small", 100), _cohort("large", 400)]
    )
    assert probabilities == (1.0 / 3.0, 2.0 / 3.0)
    assert sum(probabilities) == 1.0


def test_square_root_sampling_rejects_empty_cohort() -> None:
    try:
        square_root_sampling_probabilities([_cohort("empty", 0)])
    except ValueError as error:
        assert "at least one training sample" in str(error)
    else:
        raise AssertionError("empty cohorts must be rejected")
