#!/usr/bin/env python
"""Aggregate internal-adapter runs against the baseline with the same seed."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error

from femmhc.statistics import paired_cluster_bootstrap


TASKS = {
    "mcphases/cycle_phase": ("周期阶段", "宏 F1", False),
    "mcphases/menstrual_onset_24h": ("24 小时月经开始", "AUPRC", False),
    "mcphases/menstrual_onset_72h": ("72 小时月经开始", "AUPRC", False),
    "mcphases/cramps": ("经期痉挛", "MAE", True),
    "mcphases/mood_swing": ("情绪波动", "MAE", True),
    "mcphases/sleep_issue": ("睡眠问题", "MAE", True),
}


def _score(task_id: str, target: np.ndarray, prediction: np.ndarray) -> float:
    if task_id.endswith("cycle_phase"):
        return float(
            f1_score(
                target.astype(int),
                prediction.astype(int),
                labels=np.arange(4),
                average="macro",
                zero_division=0,
            )
        )
    if "onset" in task_id:
        return float(average_precision_score(target.astype(int), prediction))
    return float(mean_absolute_error(target, prediction))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-template", type=str, required=True)
    parser.add_argument("--candidate-template", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 42, 73))
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        baseline_path = Path(args.baseline_template.format(seed=seed))
        candidate_path = Path(args.candidate_template.format(seed=seed))
        baseline = pd.read_csv(baseline_path)
        candidate = pd.read_csv(candidate_path)
        keys = ["participant_id", "example_index", "task_id"]
        paired = baseline.merge(
            candidate,
            on=keys,
            suffixes=("_baseline", "_candidate"),
            validate="one_to_one",
        )
        if len(paired) != len(baseline) or len(paired) != len(candidate):
            raise ValueError(f"unmatched prediction rows for seed={seed}")
        if not np.allclose(paired.target_baseline, paired.target_candidate):
            raise ValueError(f"targets differ for seed={seed}")
        for task_id, (name, metric, lower_is_better) in TASKS.items():
            frame = paired[paired.task_id == task_id]
            target = frame.target_baseline.to_numpy(float)
            baseline_prediction = frame.prediction_baseline.to_numpy(float)
            candidate_prediction = frame.prediction_candidate.to_numpy(float)

            def score_pair(indices: np.ndarray):
                return (
                    _score(task_id, target[indices], baseline_prediction[indices]),
                    _score(task_id, target[indices], candidate_prediction[indices]),
                )

            bootstrap = paired_cluster_bootstrap(
                frame.participant_id,
                score_pair,
                lower_is_better=lower_is_better,
                replicates=args.bootstrap_draws,
                seed=args.seed + seed * 100 + len(rows),
                minimum_clusters=5,
            )
            baseline_metric = _score(task_id, target, baseline_prediction)
            candidate_metric = _score(task_id, target, candidate_prediction)
            oriented = (
                baseline_metric - candidate_metric
                if lower_is_better
                else candidate_metric - baseline_metric
            )
            rows.append(
                {
                    "seed": seed,
                    "task_id": task_id,
                    "task_name": name,
                    "metric": metric,
                    "participants": int(frame.participant_id.nunique()),
                    "samples": int(len(frame)),
                    "baseline": baseline_metric,
                    "candidate": candidate_metric,
                    "oriented_improvement": oriented,
                    "relative_improvement": oriented / max(abs(baseline_metric), 1e-12),
                    "ci_low": bootstrap.confidence_low,
                    "ci_high": bootstrap.confidence_high,
                    "p_value": bootstrap.p_value_two_sided,
                    "candidate_better_probability": bootstrap.probability_candidate_better,
                    "valid_bootstrap": bootstrap.valid_replicates,
                }
            )

    per_seed = pd.DataFrame(rows)
    summary = (
        per_seed.groupby(["task_id", "task_name", "metric"], as_index=False)
        .agg(
            baseline_mean=("baseline", "mean"),
            baseline_std=("baseline", "std"),
            candidate_mean=("candidate", "mean"),
            candidate_std=("candidate", "std"),
            improvement_mean=("oriented_improvement", "mean"),
            improvement_std=("oriented_improvement", "std"),
            relative_improvement_mean=("relative_improvement", "mean"),
            seed_wins=("oriented_improvement", lambda x: int((x > 0).sum())),
            positive_ci_seeds=("ci_low", lambda x: int((x > 0).sum())),
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(args.output_dir / "matched_per_seed.csv", index=False)
    summary.to_csv(args.output_dir / "matched_summary.csv", index=False)
    lines = [
        "# FemMHC 主干内 Adapter：同 seed 配对三种子结果",
        "",
        "每个候选 seed 只与相同 seed 的旧 FemMHC-Dual 基线比较；验证集为6名参与者，test未使用。",
        "",
        "| 任务 | 指标 | 基线均值 | 候选均值 | 平均有向改善 | 种子胜率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.task_name} | {row.metric} | {row.baseline_mean:.4f} | "
            f"{row.candidate_mean:.4f} | {row.improvement_mean:+.4f} | "
            f"{row.seed_wins}/{len(args.seeds)} |"
        )
    (args.output_dir / "matched_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
