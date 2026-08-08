"""Utilities for locked multi-cohort female continual pretraining."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .sensors import SensorDescriptor


@dataclass(frozen=True)
class FemaleCohort:
    """One homogeneous sensor-set cohort used by the pretraining mixture."""

    name: str
    dataset: object
    descriptors: tuple[SensorDescriptor, ...]

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]


def square_root_sampling_probabilities(
    cohorts: Sequence[FemaleCohort],
) -> tuple[float, ...]:
    """Balance cohort diversity without discarding the larger pregnancy set."""

    if not cohorts:
        raise ValueError("at least one cohort is required")
    weights = [math.sqrt(len(cohort)) for cohort in cohorts]
    if any(weight <= 0 for weight in weights):
        raise ValueError("every cohort must contain at least one training sample")
    total = sum(weights)
    return tuple(weight / total for weight in weights)
