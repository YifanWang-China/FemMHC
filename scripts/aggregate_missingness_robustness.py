#!/usr/bin/env python
"""Aggregate missing-history robustness by task family after seed averaging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _scope(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "all":
        return frame
    if name == "female_specific":
        return frame[frame["source"] != "openmhc"]
    if name == "mcphases":
        return frame[frame["source"] == "mcphases"]
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.input_dir / "per_task_primary_metrics.csv")
    history = pd.read_csv(args.input_dir / "history_realization.csv")
    mean_tasks = (
        frame.groupby(
            [
                "scenario",
                "task_id",
                "source",
                "domain",
                "kind",
                "primary_metric",
            ],
            as_index=False,
        )[
            [
                "baseline_value",
                "scenario_value",
                "oriented_delta",
                "relative_change_percent",
            ]
        ]
        .mean()
    )
    summary_rows: list[dict[str, object]] = []
    for scope_name in ("all", "female_specific", "mcphases"):
        scoped = _scope(mean_tasks, scope_name)
        for scenario, group in scoped.groupby("scenario", sort=False):
            finite = group[np.isfinite(group["oriented_delta"])].copy()
            delta = finite["oriented_delta"].to_numpy(float)
            relative = finite["relative_change_percent"].to_numpy(float)
            summary_rows.append(
                {
                    "scope": scope_name,
                    "scenario": scenario,
                    "tasks": int(len(finite)),
                    "improved_tasks": int(np.count_nonzero(delta > 1e-12)),
                    "ties": int(np.count_nonzero(np.abs(delta) <= 1e-12)),
                    "worsened_tasks": int(np.count_nonzero(delta < -1e-12)),
                    "relative_change_mean_percent": float(np.mean(relative)),
                    "relative_change_median_percent": float(np.median(relative)),
                    "relative_change_p25_percent": float(np.quantile(relative, 0.25)),
                    "relative_change_p75_percent": float(np.quantile(relative, 0.75)),
                }
            )
    summary = pd.DataFrame(summary_rows)
    mcphases_history = (
        history[history["cohort"] == "mcphases"]
        .groupby("scenario", as_index=False)[
            [
                "observed_days_before_mean",
                "observed_days_after_mean",
                "observed_days_after_median",
                "retained_fraction_mean",
            ]
        ]
        .mean()
    )
    output = {
        "format_version": 1,
        "split": "validation",
        "test_used": False,
        "scopes": {
            scope: summary[summary["scope"] == scope].drop(columns="scope").to_dict("records")
            for scope in ("all", "female_specific", "mcphases")
        },
        "mcphases_history": mcphases_history.to_dict("records"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mean_tasks.to_csv(args.output_dir / "seed_averaged_tasks.csv", index=False)
    summary.to_csv(args.output_dir / "scope_summary.csv", index=False)
    mcphases_history.to_csv(args.output_dir / "mcphases_history.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# FemMHC时序缺失鲁棒性分任务族汇总",
        "",
        "> 先对三个模型种子逐任务取均值，再统计任务胜负；测试集未使用。",
        "",
        "| 范围 | 场景 | 任务数 | 改善/平/下降 | 相对变化中位数 | 四分位区间 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.scope} | {row.scenario} | {row.tasks} | "
            f"{row.improved_tasks}/{row.ties}/{row.worsened_tasks} | "
            f"{row.relative_change_median_percent:+.2f}% | "
            f"[{row.relative_change_p25_percent:+.2f}%, {row.relative_change_p75_percent:+.2f}%] |"
        )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output_dir": str(args.output_dir.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
