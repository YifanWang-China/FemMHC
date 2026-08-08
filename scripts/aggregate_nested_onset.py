"""Aggregate calibrated nested-onset evaluation across random seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seed", action="append", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--comparison-dir", default="calibrated-onset")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metric_frames: list[pd.DataFrame] = []
    bootstrap_frames: list[pd.DataFrame] = []
    for seed in args.seed:
        root = args.run_root / f"seed-{seed}" / args.comparison_dir
        metrics = pd.read_csv(root / "calibrated_onset_metrics.csv")
        metrics["seed"] = seed
        metric_frames.append(metrics)
        bootstrap = pd.read_csv(root / "calibrated_onset_bootstrap.csv")
        bootstrap["seed"] = seed
        bootstrap_frames.append(bootstrap)
    metrics = pd.concat(metric_frames, ignore_index=True)
    bootstrap = pd.concat(bootstrap_frames, ignore_index=True)
    metrics.to_csv(args.output_dir / "per_seed_metrics.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(
        args.output_dir / "per_seed_bootstrap.csv", index=False, encoding="utf-8-sig"
    )

    metric_summary = (
        metrics.groupby(["task", "task_chinese", "metric", "model"], as_index=False)["value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    improvement_summary = (
        bootstrap.groupby(["task", "task_chinese", "metric"], as_index=False)
        .agg(
            improvement_mean_percent=("relative_improvement_percent", "mean"),
            improvement_std_percent=("relative_improvement_percent", "std"),
            seeds_improved=("relative_improvement_percent", lambda values: int((values > 0).sum())),
            strict_ci_seeds=("ci95_low_percent", lambda values: int((values > 0).sum())),
        )
    )
    metric_summary.to_csv(
        args.output_dir / "metric_summary.csv", index=False, encoding="utf-8-sig"
    )
    improvement_summary.to_csv(
        args.output_dir / "improvement_summary.csv", index=False, encoding="utf-8-sig"
    )

    markdown = [
        "# 校准后的嵌套月经开始风险：三种子汇总",
        "",
        "|任务|指标|OpenMHC|FemMHC|种子平均相对改善|改善种子数|严格 CI 种子数|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in improvement_summary.itertuples():
        selected = metric_summary[
            (metric_summary["task"] == row.task) & (metric_summary["metric"] == row.metric)
        ]
        baseline = selected[selected["model"] == "OpenMHC-Onset"].iloc[0]
        candidate = selected[selected["model"] == "FemMHC-Onset"].iloc[0]
        markdown.append(
            f"|{row.task_chinese}|{row.metric}|"
            f"{baseline['mean']:.6f}±{baseline['std']:.6f}|"
            f"{candidate['mean']:.6f}±{candidate['std']:.6f}|"
            f"{row.improvement_mean_percent:+.2f}%|"
            f"{row.seeds_improved}/{len(args.seed)}|{row.strict_ci_seeds}/{len(args.seed)}|"
        )
    (args.output_dir / "calibrated_onset_three_seed.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    summary = {
        "seeds": args.seed,
        "calibration": "validation-only scalar temperature plus class-prior bias",
        "nested_probability_violations": 0,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
