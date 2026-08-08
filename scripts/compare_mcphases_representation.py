#!/usr/bin/env python
"""Compare two fixed six-task mcPHASES validation prediction files."""

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
    if task_id == "mcphases/cycle_phase":
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
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    baseline = pd.read_csv(args.baseline)
    candidate = pd.read_csv(args.candidate)
    keys = ["participant_id", "example_index", "task_id"]
    paired = baseline.merge(
        candidate,
        on=keys,
        how="inner",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    if len(paired) != len(baseline) or len(paired) != len(candidate):
        raise ValueError("prediction files are not exactly aligned")
    if not np.allclose(paired.target_baseline, paired.target_candidate):
        raise ValueError("target values differ between prediction files")

    rows: list[dict[str, object]] = []
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

        result = paired_cluster_bootstrap(
            frame.participant_id,
            score_pair,
            lower_is_better=lower_is_better,
            replicates=args.bootstrap_draws,
            seed=args.seed + len(rows),
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
                "task_id": task_id,
                "task_name": name,
                "metric": metric,
                "participants": int(frame.participant_id.nunique()),
                "samples": int(len(frame)),
                "baseline": baseline_metric,
                "candidate": candidate_metric,
                "oriented_improvement": oriented,
                "relative_improvement": oriented / max(abs(baseline_metric), 1e-12),
                "ci_low": result.confidence_low,
                "ci_high": result.confidence_high,
                "p_value": result.p_value_two_sided,
                "candidate_better_probability": result.probability_candidate_better,
                "valid_bootstrap": result.valid_replicates,
            }
        )

    result_frame = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(args.output_dir / "representation_comparison.csv", index=False)
    lines = [
        "# mcPHASES 表征对比（验证集）",
        "",
        "仅使用 validation；按 participant_id 聚类 bootstrap；未读取 test 标签。",
        "",
        "| 任务 | 指标 | 基线 | 候选 | 有向改善 | 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task_name']} | {row['metric']} | {row['baseline']:.4f} | "
            f"{row['candidate']:.4f} | {row['oriented_improvement']:+.4f} | "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] |"
        )
    (args.output_dir / "representation_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(result_frame.to_string(index=False))


if __name__ == "__main__":
    main()
