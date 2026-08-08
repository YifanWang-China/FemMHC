"""Participant-level statistical inference for longitudinal benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np


ScorePair = Callable[[np.ndarray], tuple[float | None, float | None]]


@dataclass(frozen=True)
class PairedClusterBootstrapResult:
    """Paired cluster-bootstrap result with positive values favoring candidate."""

    estimate: float | None
    confidence_low: float | None
    confidence_high: float | None
    probability_candidate_better: float | None
    p_value_two_sided: float | None
    clusters: int
    valid_replicates: int
    requested_replicates: int
    eligible: bool
    reason: str | None = None


def _oriented_delta(
    baseline: float, candidate: float, *, lower_is_better: bool
) -> float:
    return baseline - candidate if lower_is_better else candidate - baseline


def paired_cluster_bootstrap(
    cluster_ids: Sequence[object],
    score_pair: ScorePair,
    *,
    lower_is_better: bool,
    replicates: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    minimum_clusters: int = 5,
) -> PairedClusterBootstrapResult:
    """Resample participants and estimate a paired candidate-minus-baseline effect.

    ``score_pair`` receives row indices and must return baseline and candidate
    scores computed on exactly those rows. Repeatedly sampled clusters repeat all
    of their rows, preserving the longitudinal dependence within a participant.
    """

    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if minimum_clusters <= 1:
        raise ValueError("minimum_clusters must be greater than one")

    cluster_array = np.asarray(cluster_ids, dtype=str)
    if cluster_array.ndim != 1 or len(cluster_array) == 0:
        raise ValueError("cluster_ids must be a non-empty one-dimensional sequence")
    clusters = np.unique(cluster_array)
    baseline, candidate = score_pair(np.arange(len(cluster_array)))
    estimate = (
        None
        if baseline is None or candidate is None
        else _oriented_delta(
            float(baseline), float(candidate), lower_is_better=lower_is_better
        )
    )
    if len(clusters) < minimum_clusters:
        return PairedClusterBootstrapResult(
            estimate=estimate,
            confidence_low=None,
            confidence_high=None,
            probability_candidate_better=None,
            p_value_two_sided=None,
            clusters=int(len(clusters)),
            valid_replicates=0,
            requested_replicates=replicates,
            eligible=False,
            reason=f"fewer than {minimum_clusters} participant clusters",
        )
    if estimate is None or not np.isfinite(estimate):
        return PairedClusterBootstrapResult(
            estimate=None,
            confidence_low=None,
            confidence_high=None,
            probability_candidate_better=None,
            p_value_two_sided=None,
            clusters=int(len(clusters)),
            valid_replicates=0,
            requested_replicates=replicates,
            eligible=False,
            reason="primary metric is undefined on the observed sample",
        )

    indices_by_cluster = {
        cluster: np.flatnonzero(cluster_array == cluster) for cluster in clusters
    }
    generator = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(replicates):
        sampled = generator.choice(clusters, size=len(clusters), replace=True)
        indices = np.concatenate([indices_by_cluster[cluster] for cluster in sampled])
        baseline, candidate = score_pair(indices)
        if baseline is None or candidate is None:
            continue
        delta = _oriented_delta(
            float(baseline), float(candidate), lower_is_better=lower_is_better
        )
        if np.isfinite(delta):
            deltas.append(delta)

    minimum_valid = min(replicates, max(100, int(np.ceil(0.8 * replicates))))
    if len(deltas) < minimum_valid:
        return PairedClusterBootstrapResult(
            estimate=estimate,
            confidence_low=None,
            confidence_high=None,
            probability_candidate_better=None,
            p_value_two_sided=None,
            clusters=int(len(clusters)),
            valid_replicates=len(deltas),
            requested_replicates=replicates,
            eligible=False,
            reason="too many bootstrap replicates had an undefined primary metric",
        )

    values = np.asarray(deltas, dtype=np.float64)
    alpha = 1.0 - confidence
    low, high = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    left = (float(np.count_nonzero(values <= 0.0)) + 1.0) / (len(values) + 1.0)
    right = (float(np.count_nonzero(values >= 0.0)) + 1.0) / (len(values) + 1.0)
    return PairedClusterBootstrapResult(
        estimate=estimate,
        confidence_low=float(low),
        confidence_high=float(high),
        probability_candidate_better=float(np.mean(values > 0.0)),
        p_value_two_sided=min(1.0, 2.0 * min(left, right)),
        clusters=int(len(clusters)),
        valid_replicates=len(values),
        requested_replicates=replicates,
        eligible=True,
    )


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm-adjust a named family of finite p-values."""

    finite = {
        key: float(value)
        for key, value in p_values.items()
        if np.isfinite(value) and 0.0 <= value <= 1.0
    }
    ordered = sorted(finite, key=finite.get)
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, key in enumerate(ordered):
        running = max(running, (count - rank) * finite[key])
        adjusted[key] = min(1.0, running)
    return adjusted
