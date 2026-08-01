"""Aggregate selected FemMHC-vs-OpenMHC comparisons across random seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seed", action="append", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--comparison-dir", default="paired-bootstrap")
    parser.add_argument("--baseline", default="OpenMHC")
    parser.add_argument("--task", action="append")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for seed in args.seed:
        path = args.run_root / f"seed-{seed}" / args.comparison_dir / "paired_participant_bootstrap.csv"
        frame = pd.read_csv(path)
        frame = frame[frame["baseline"] == args.baseline].copy()
        if args.task:
            frame = frame[frame["task"].isin(args.task)].copy()
        if frame.empty:
            raise ValueError(f"no matching comparison rows in {path}")
        frame["seed"] = seed
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(args.output_dir / "per_seed_results.csv", index=False, encoding="utf-8-sig")

    rows: list[dict[str, object]] = []
    for task, group in combined.groupby("task", sort=False):
        direction = str(group.iloc[0]["direction"])
        baseline_mean = float(group["baseline_value"].mean())
        candidate_mean = float(group["candidate_value"].mean())
        oriented_absolute = (
            candidate_mean - baseline_mean
            if direction == "higher"
            else baseline_mean - candidate_mean
        )
        relative_from_means = 100.0 * oriented_absolute / max(abs(baseline_mean), 1e-12)
        rows.append(
            {
                "task": task,
                "task_chinese": group.iloc[0]["task_chinese"],
                "primary_metric": group.iloc[0]["primary_metric"],
                "direction": direction,
                "seeds": len(group),
                "baseline_mean": baseline_mean,
                "baseline_std": float(group["baseline_value"].std(ddof=1)),
                "candidate_mean": candidate_mean,
                "candidate_std": float(group["candidate_value"].std(ddof=1)),
                "relative_improvement_from_means_percent": relative_from_means,
                "per_seed_relative_improvement_mean_percent": float(
                    group["relative_improvement_percent"].mean()
                ),
                "per_seed_relative_improvement_std_percent": float(
                    group["relative_improvement_percent"].std(ddof=1)
                ),
                "seeds_improved": int((group["relative_improvement_percent"] > 0).sum()),
                "seeds_ci_strictly_above_zero": int((group["ci95_low_percent"] > 0).sum()),
            }
        )
    aggregate = pd.DataFrame(rows)
    aggregate.to_csv(args.output_dir / "three_seed_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "seeds": args.seed,
        "tasks_improved_by_mean": int(
            (aggregate["relative_improvement_from_means_percent"] > 0).sum()
        ),
        "tasks_improved_in_all_seeds": int((aggregate["seeds_improved"] == len(args.seed)).sum()),
        "tasks_with_strict_ci_in_all_seeds": int(
            (aggregate["seeds_ci_strictly_above_zero"] == len(args.seed)).sum()
        ),
        "tasks_total": int(len(aggregate)),
    }
    (args.output_dir / "three_seed_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    markdown = [
        "# FemMHC 三随机种子汇总",
        "",
        f"随机种子：{', '.join(map(str, args.seed))}",
        "",
        "|任务|指标|OpenMHC 均值±标准差|FemMHC 均值±标准差|按均值相对提升|改善种子数|严格 CI 种子数|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.itertuples():
        markdown.append(
            f"|{row.task_chinese}|{row.primary_metric}|"
            f"{row.baseline_mean:.4f}±{row.baseline_std:.4f}|"
            f"{row.candidate_mean:.4f}±{row.candidate_std:.4f}|"
            f"{row.relative_improvement_from_means_percent:+.2f}%|"
            f"{row.seeds_improved}/{len(args.seed)}|"
            f"{row.seeds_ci_strictly_above_zero}/{len(args.seed)}|"
        )
    markdown.extend(
        [
            "",
            "“严格 CI 种子数”表示该种子内的参与者级 bootstrap 95% 区间严格高于零。",
        ]
    )
    (args.output_dir / "three_seed_summary.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
