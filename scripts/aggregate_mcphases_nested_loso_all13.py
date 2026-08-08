#!/usr/bin/env python
"""Aggregate independently persisted mcPHASES nested-LOSO task runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from femmhc.statistics import holm_adjust
from femmhc.tasks import MCPHASES_TASKS


def bootstrap_p_value(probability_improved: float, draws: int) -> float:
    """Recover the finite-sample two-sided sign-tail p-value from saved draws."""

    if draws <= 0 or not np.isfinite(probability_improved):
        return float("nan")
    improved = int(round(probability_improved * draws))
    other = draws - improved
    return float(min(1.0, 2.0 * (min(improved, other) + 1.0) / (draws + 1.0)))


def _fmt(value: float, metric: str) -> str:
    return f"{value:.4f}" if metric != "mae" or abs(value) < 10 else f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-task-dir", type=Path, required=True)
    parser.add_argument("--simple-baselines", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    expected = [task.name for task in MCPHASES_TASKS]
    missing = [
        task
        for task in expected
        if not (args.per_task_dir / task / "nested_loso_paired_comparison.csv").exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing completed task runs: {', '.join(missing)}")

    results = pd.concat(
        [
            pd.read_csv(args.per_task_dir / task / "nested_loso_results.csv")
            for task in expected
        ],
        ignore_index=True,
    )
    comparisons = pd.concat(
        [
            pd.read_csv(args.per_task_dir / task / "nested_loso_paired_comparison.csv")
            for task in expected
        ],
        ignore_index=True,
    )
    hyperparameters = pd.concat(
        [
            pd.read_csv(
                args.per_task_dir
                / task
                / "nested_loso_hyperparameter_distribution.csv"
            )
            for task in expected
        ],
        ignore_index=True,
    )
    comparisons["p_value_two_sided"] = comparisons.apply(
        lambda row: bootstrap_p_value(
            float(row["paired_bootstrap_probability_improved"]),
            int(row["bootstrap_draws_used"]),
        ),
        axis=1,
    )
    comparisons["p_value_holm_13_tasks"] = np.nan
    for candidate, group in comparisons.groupby("candidate", sort=False):
        adjusted = holm_adjust(
            {
                str(row.task): float(row.p_value_two_sided)
                for row in group.itertuples(index=False)
            }
        )
        mask = comparisons["candidate"] == candidate
        comparisons.loc[mask, "p_value_holm_13_tasks"] = comparisons.loc[
            mask, "task"
        ].map(adjusted)

    simple = pd.read_csv(args.simple_baselines)
    simple_lookup = simple.set_index("task")["primary_value"].to_dict()
    summary_rows: list[dict[str, object]] = []
    for task_name, group in comparisons.groupby("task", sort=False):
        metric = str(group["primary_metric"].iloc[0])
        lower = metric == "mae"
        baseline = float(group["baseline_value"].iloc[0])
        candidates = group["candidate_value"].to_numpy(float)
        candidate_mean = float(np.mean(candidates))
        simple_value = float(simple_lookup[task_name])
        summary_rows.append(
            {
                "task": task_name,
                "task_chinese": str(group["task_chinese"].iloc[0]),
                "primary_metric": metric,
                "samples": int(group["samples"].iloc[0]),
                "participants": int(group["participants"].iloc[0]),
                "simple_baseline": simple_value,
                "openmhc": baseline,
                "femmhc_mean": candidate_mean,
                "femmhc_sd": float(np.std(candidates, ddof=1)),
                "relative_improvement_vs_openmhc_percent": float(
                    np.mean(group["relative_improvement_percent"].to_numpy(float))
                ),
                "femmhc_seeds_beating_openmhc": int(
                    np.count_nonzero(
                        candidates < baseline if lower else candidates > baseline
                    )
                ),
                "femmhc_mean_beats_simple_baseline": bool(
                    candidate_mean < simple_value
                    if lower
                    else candidate_mean > simple_value
                ),
                "strict_positive_ci_seeds": int(
                    np.count_nonzero(
                        group["paired_bootstrap_ci_low"].to_numpy(float) > 0
                    )
                ),
                "holm_significant_seeds": int(
                    np.count_nonzero(
                        group["p_value_holm_13_tasks"].to_numpy(float) < 0.05
                    )
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "nested_loso_results.csv", index=False)
    comparisons.to_csv(
        args.output_dir / "nested_loso_paired_comparison.csv", index=False
    )
    hyperparameters.to_csv(
        args.output_dir / "nested_loso_hyperparameter_distribution.csv", index=False
    )
    summary.to_csv(args.output_dir / "nested_loso_three_seed_summary.csv", index=False)

    wins_openmhc = int(
        np.count_nonzero(summary["femmhc_seeds_beating_openmhc"].to_numpy(int) >= 2)
    )
    wins_simple = int(
        np.count_nonzero(summary["femmhc_mean_beats_simple_baseline"].to_numpy(bool))
    )
    lines = [
        "# mcPHASES 42人嵌套留一结果",
        "",
        "外层每次留出1名参与者；正则化仅在其余参与者的三折分组内层交叉验证中选择。置信区间为参与者级配对Bootstrap；Holm校正在每个FemMHC种子的13项任务族内进行。",
        "",
        f"FemMHC三种子多数胜过OpenMHC：**{wins_openmhc}/13项任务**；FemMHC种子均值胜过无穿戴简单基线：**{wins_simple}/13项任务**。",
        "",
        "| 任务 | 指标 | 人数 | 简单基线 | OpenMHC | FemMHC（三种子） | 对OpenMHC提升 | 胜出种子 | 正置信区间 | Holm显著 | 胜简单基线 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.task_chinese} | {row.primary_metric} | {row.participants} | "
            f"{_fmt(row.simple_baseline, row.primary_metric)} | "
            f"{_fmt(row.openmhc, row.primary_metric)} | "
            f"{_fmt(row.femmhc_mean, row.primary_metric)} ± "
            f"{_fmt(row.femmhc_sd, row.primary_metric)} | "
            f"{row.relative_improvement_vs_openmhc_percent:+.2f}% | "
            f"{row.femmhc_seeds_beating_openmhc}/3 | "
            f"{row.strict_positive_ci_seeds}/3 | "
            f"{row.holm_significant_seeds}/3 | "
            f"{'是' if row.femmhc_mean_beats_simple_baseline else '否'} |"
        )
    lines.extend(
        [
            "",
            "注：MAE越低越好，其余主指标越高越好。简单基线分别为训练折常数/多数分布；它是防止将数据先验误当作表征能力的必要对照。",
            "",
        ]
    )
    (args.output_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    manifest = {
        "format_version": 1,
        "protocol": "nested_leave_one_participant_out",
        "outer_participants": 42,
        "tasks": expected,
        "models": sorted(results["model"].unique().tolist()),
        "multiple_testing": "Holm within each FemMHC seed across 13 tasks",
        "simple_baseline_file": str(args.simple_baselines.resolve()),
        "individual_predictions_written": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
