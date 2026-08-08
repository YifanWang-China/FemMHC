#!/usr/bin/env python
"""Aggregate candidate missing-history robustness across adapter seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-template", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 42, 73))
    args = parser.parse_args()
    task_frames = []
    history_frames = []
    for seed in args.seeds:
        root = Path(args.run_template.format(seed=seed))
        task = pd.read_csv(root / "per_task_primary_metrics.csv")
        history = pd.read_csv(root / "history_realization.csv")
        if "seed" not in task:
            task.insert(0, "seed", seed)
        if "seed" not in history:
            history.insert(0, "seed", seed)
        task_frames.append(task)
        history_frames.append(history)
    tasks = pd.concat(task_frames, ignore_index=True)
    history = pd.concat(history_frames, ignore_index=True)
    mean_tasks = tasks.groupby(
        ["scenario", "task_id", "source", "domain", "kind", "primary_metric"],
        as_index=False,
    )[["baseline_value", "scenario_value", "oriented_delta", "relative_change_percent"]].mean()
    summary_rows = []
    for scenario, group in mean_tasks.groupby("scenario", sort=False):
        delta = group.oriented_delta.to_numpy(float)
        relative = group.relative_change_percent.to_numpy(float)
        summary_rows.append(
            {
                "scenario": scenario,
                "tasks": len(group),
                "improved": int((delta > 1e-12).sum()),
                "ties": int((np.abs(delta) <= 1e-12).sum()),
                "worsened": int((delta < -1e-12).sum()),
                "relative_change_mean_percent": float(relative.mean()),
                "relative_change_median_percent": float(np.median(relative)),
                "relative_change_p25_percent": float(np.quantile(relative, 0.25)),
                "relative_change_p75_percent": float(np.quantile(relative, 0.75)),
            }
        )
    summary = pd.DataFrame(summary_rows)
    mc_history = history[history.cohort == "mcphases"].groupby(
        "scenario", as_index=False
    )[[
        "observed_days_before_mean",
        "observed_days_after_mean",
        "observed_days_after_median",
        "retained_fraction_mean",
    ]].mean()
    output = {
        "format_version": 1,
        "split": "validation",
        "test_used": False,
        "seeds": list(args.seeds),
        "scenarios": summary.to_dict("records"),
        "mcphases_history": mc_history.to_dict("records"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks.to_csv(args.output_dir / "per_seed_tasks.csv", index=False)
    mean_tasks.to_csv(args.output_dir / "seed_averaged_tasks.csv", index=False)
    history.to_csv(args.output_dir / "per_seed_history.csv", index=False)
    summary.to_csv(args.output_dir / "scenario_summary.csv", index=False)
    mc_history.to_csv(args.output_dir / "mcphases_history.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# 主干内 Adapter 三种子时序/缺失鲁棒性",
        "",
        "仅使用 validation；测试集未使用。相对变化为相对各 seed 的 baseline 场景。",
        "",
        "| 场景 | 任务数 | 改善/持平/下降 | 相对变化中位数 |",
        "|---|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.scenario} | {row.tasks} | {row.improved}/{row.ties}/{row.worsened} | "
            f"{row.relative_change_median_percent:+.2f}% |"
        )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
