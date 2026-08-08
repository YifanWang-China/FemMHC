#!/usr/bin/env python
"""Verify that a FemMHC native-branch cache equals an OpenMHC cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_cache(directory: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    embeddings = np.load(directory / "embeddings.npy").astype(np.float32)
    user_ids = np.load(directory / "user_ids.npy", allow_pickle=True).astype(str)
    dates = np.load(directory / "dates.npy", allow_pickle=True).astype(str)
    if not (len(embeddings) == len(user_ids) == len(dates)):
        raise ValueError(f"inconsistent cache lengths in {directory}")
    return embeddings, user_ids, dates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openmhc-cache", type=Path, required=True)
    parser.add_argument("--native-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()

    baseline, baseline_users, baseline_dates = _load_cache(args.openmhc_cache)
    native, native_users, native_dates = _load_cache(args.native_cache)
    if not np.array_equal(baseline_users, native_users) or not np.array_equal(
        baseline_dates, native_dates
    ):
        raise ValueError("cache rows are not aligned by participant and date")
    if baseline.shape != native.shape:
        raise ValueError(f"embedding shape mismatch: {baseline.shape} != {native.shape}")

    absolute = np.abs(native - baseline)
    baseline_norm = np.linalg.norm(baseline, axis=1)
    native_norm = np.linalg.norm(native, axis=1)
    denominator = np.maximum(baseline_norm * native_norm, np.finfo(np.float32).tiny)
    cosine = np.sum(baseline * native, axis=1) / denominator
    exact = bool(np.array_equal(baseline, native))
    report = {
        "format_version": 1,
        "openmhc_cache": str(args.openmhc_cache),
        "native_cache": str(args.native_cache),
        "participant_days": int(len(native)),
        "embedding_dimension": int(native.shape[1]),
        "compared_values": int(native.size),
        "different_values": int(np.count_nonzero(absolute)),
        "maximum_absolute_difference": float(absolute.max(initial=0.0)),
        "mean_absolute_difference": float(absolute.mean()),
        "minimum_cosine_similarity": float(cosine.min(initial=1.0)),
        "mean_cosine_similarity": float(cosine.mean()),
        "exactly_equal": exact,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.require_exact and not exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
