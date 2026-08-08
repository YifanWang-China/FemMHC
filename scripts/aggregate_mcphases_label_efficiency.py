#!/usr/bin/env python
"""Aggregate task-level mcPHASES label-efficiency runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TASKS = (
    "cycle_phase",
    "cramps",
    "menstrual_onset_24h",
    "mood_swing",
    "estrogen",
    "menstrual_onset_72h",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-task-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    missing = [task for task in TASKS if not (args.per_task_dir / task / "summary.json").exists()]
    if missing:
        raise FileNotFoundError(f"missing completed task runs: {', '.join(missing)}")

    all_runs = pd.concat(
        [pd.read_csv(args.per_task_dir / task / "all_runs.csv") for task in TASKS],
        ignore_index=True,
    )
    task_summary = pd.concat(
        [
            pd.read_csv(args.per_task_dir / task / "per_task_summary.csv")
            for task in TASKS
        ],
        ignore_index=True,
    )
    fraction_rows: list[dict[str, object]] = []
    for fraction, group in task_summary.groupby("fraction", sort=True):
        relative = group["relative_improvement_percent"].to_numpy(float)
        fraction_rows.append(
            {
                "fraction": float(fraction),
                "tasks": int(len(group)),
                "candidate_wins": int(np.count_nonzero(relative > 0)),
                "ties": int(np.count_nonzero(np.isclose(relative, 0.0))),
                "candidate_losses": int(np.count_nonzero(relative < 0)),
                "relative_improvement_mean_percent": float(np.mean(relative)),
                "relative_improvement_median_percent": float(np.median(relative)),
                "relative_improvement_p25_percent": float(np.quantile(relative, 0.25)),
                "relative_improvement_p75_percent": float(np.quantile(relative, 0.75)),
            }
        )
    fraction_summary = pd.DataFrame(fraction_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_runs.to_csv(args.output_dir / "all_runs.csv", index=False)
    task_summary.to_csv(args.output_dir / "per_task_summary.csv", index=False)
    fraction_summary.to_csv(args.output_dir / "fraction_summary.csv", index=False)
    manifest = {
        "format_version": 1,
        "tasks": list(TASKS),
        "split": "validation",
        "selection_split": "train_only",
        "test_used": False,
        "baseline": "OpenMHC-dual",
        "candidate_seeds": [42, 43, 44],
        "fractions": fraction_summary["fraction"].tolist(),
        "repeats": 5,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# mcPHASES少标签迁移汇总",
        "",
        "> 标签子集仅来自29名训练参与者；6名验证参与者只评估；测试集未使用。每个标签预算独立在训练参与者内选择探针强度。",
        "",
        "| 标签比例 | 任务数 | FemMHC胜/平/负 | 相对改善中位数 | 四分位区间 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in fraction_summary.itertuples(index=False):
        lines.append(
            f"| {100*row.fraction:.0f}% | {row.tasks} | "
            f"{row.candidate_wins}/{row.ties}/{row.candidate_losses} | "
            f"{row.relative_improvement_median_percent:+.2f}% | "
            f"[{row.relative_improvement_p25_percent:+.2f}%, "
            f"{row.relative_improvement_p75_percent:+.2f}%] |"
        )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
