#!/usr/bin/env python
"""Aggregate the three-seed internal-adapter representation comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--run-template",
        default="mcphases-internal-adapter-rank32-onset2-six-seed{seed}",
        help="Run directory name relative to --root; {seed} is substituted.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 42, 73))
    args = parser.parse_args()

    frames = []
    for seed in args.seeds:
        path = (
            args.root
            / args.run_template.format(seed=seed)
            / "comparison"
            / "representation_comparison.csv"
        )
        frame = pd.read_csv(path)
        frame.insert(0, "seed", seed)
        frames.append(frame)
    per_seed = pd.concat(frames, ignore_index=True)
    summary = (
        per_seed.groupby(["task_id", "task_name", "metric"], as_index=False)
        .agg(
            baseline_mean=("baseline", "mean"),
            candidate_mean=("candidate", "mean"),
            candidate_std=("candidate", "std"),
            improvement_mean=("oriented_improvement", "mean"),
            improvement_std=("oriented_improvement", "std"),
            relative_improvement_mean=("relative_improvement", "mean"),
            seed_wins=("oriented_improvement", lambda x: int((x > 0).sum())),
            positive_ci_seeds=("ci_low", lambda x: int((x > 0).sum())),
        )
    )
    args.root.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(args.root / "per_seed.csv", index=False)
    summary.to_csv(args.root / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
