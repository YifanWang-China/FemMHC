#!/usr/bin/env python
"""Aggregate the controlled six-task joint-versus-isolated experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error

from femmhc.statistics import holm_adjust, paired_cluster_bootstrap
try:
    from train_mcphases_single_vs_joint import TASK_GROUPS
except ImportError:  # pragma: no cover - package import used by tests
    from scripts.train_mcphases_single_vs_joint import TASK_GROUPS


TASK_META = {
    "mcphases/cycle_phase": ("月经周期阶段", "宏平均F1", False),
    "mcphases/menstrual_onset_24h": ("24小时内月经开始", "AUPRC", False),
    "mcphases/menstrual_onset_72h": ("72小时内月经开始", "AUPRC", False),
    "mcphases/cramps": ("次日经期痉挛严重度", "MAE", True),
    "mcphases/mood_swing": ("次日情绪波动严重度", "MAE", True),
    "mcphases/sleep_issue": ("次日睡眠问题严重度", "MAE", True),
}


def _score(task_id: str, target: np.ndarray, prediction: np.ndarray) -> float | None:
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
    if task_id in {
        "mcphases/menstrual_onset_24h",
        "mcphases/menstrual_onset_72h",
    }:
        labels = target.astype(int)
        if len(np.unique(labels)) < 2:
            return None
        return float(average_precision_score(labels, prediction))
    return float(mean_absolute_error(target, prediction))


def _load_predictions(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {
        "seed",
        "task_id",
        "participant_id",
        "example_index",
        "target",
        "prediction",
    }
    if required - set(frame):
        raise ValueError(f"missing columns in {path}: {sorted(required - set(frame))}")
    return frame


def _read_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 42, 73))
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    args = parser.parse_args()

    per_seed_rows: list[dict] = []
    parameter_rows: list[dict] = []
    for seed in args.seeds:
        seed_dir = args.root / f"seed-{seed}"
        joint_dir = seed_dir / "joint6"
        joint = _load_predictions(joint_dir / "validation_predictions.csv")
        joint_summary = _read_summary(joint_dir / "summary.json")
        isolated_parameters = 0
        for group_name, group_tasks in TASK_GROUPS.items():
            single_dir = seed_dir / f"single-{group_name}"
            single = _load_predictions(single_dir / "validation_predictions.csv")
            isolated_parameters += int(
                _read_summary(single_dir / "summary.json")["trainable_parameters"]
            )
            for task_id in group_tasks:
                joint_task = joint[joint.task_id == task_id].copy()
                single_task = single[single.task_id == task_id].copy()
                keys = ["participant_id", "example_index", "task_id"]
                paired = single_task.merge(
                    joint_task,
                    on=keys,
                    how="inner",
                    suffixes=("_single", "_joint"),
                    validate="one_to_one",
                )
                if len(paired) != len(single_task) or len(paired) != len(joint_task):
                    raise ValueError(f"prediction alignment failed for seed={seed}, task={task_id}")
                if not np.allclose(
                    paired.target_single.to_numpy(),
                    paired.target_joint.to_numpy(),
                ):
                    raise ValueError(f"target mismatch for seed={seed}, task={task_id}")
                target = paired.target_single.to_numpy(dtype=float)
                single_prediction = paired.prediction_single.to_numpy(dtype=float)
                joint_prediction = paired.prediction_joint.to_numpy(dtype=float)

                def score_pair(indices: np.ndarray):
                    return (
                        _score(task_id, target[indices], single_prediction[indices]),
                        _score(task_id, target[indices], joint_prediction[indices]),
                    )

                lower = TASK_META[task_id][2]
                result = paired_cluster_bootstrap(
                    paired.participant_id,
                    score_pair,
                    lower_is_better=lower,
                    replicates=args.bootstrap_draws,
                    seed=args.bootstrap_seed + seed * 100 + len(per_seed_rows),
                    minimum_clusters=5,
                )
                single_metric = _score(task_id, target, single_prediction)
                joint_metric = _score(task_id, target, joint_prediction)
                if single_metric is None or joint_metric is None:
                    relative = None
                elif lower:
                    relative = (single_metric - joint_metric) / max(abs(single_metric), 1e-12)
                else:
                    relative = (joint_metric - single_metric) / max(abs(single_metric), 1e-12)
                per_seed_rows.append(
                    {
                        "seed": seed,
                        "task_group": group_name,
                        "task_id": task_id,
                        "task_name_cn": TASK_META[task_id][0],
                        "metric": TASK_META[task_id][1],
                        "lower_is_better": lower,
                        "participants": paired.participant_id.nunique(),
                        "samples": len(paired),
                        "single_metric": single_metric,
                        "joint_metric": joint_metric,
                        "oriented_improvement": result.estimate,
                        "relative_improvement": relative,
                        "ci_low": result.confidence_low,
                        "ci_high": result.confidence_high,
                        "p_value": result.p_value_two_sided,
                        "probability_joint_better": result.probability_candidate_better,
                        "valid_bootstrap": result.valid_replicates,
                    }
                )
        parameter_rows.append(
            {
                "seed": seed,
                "joint_parameters": int(joint_summary["trainable_parameters"]),
                "isolated_ensemble_parameters": isolated_parameters,
                "parameter_ratio_isolated_over_joint": isolated_parameters
                / int(joint_summary["trainable_parameters"]),
            }
        )

    per_seed = pd.DataFrame(per_seed_rows)
    adjusted_values: dict[tuple[int, str], float] = {}
    for seed, frame in per_seed.groupby("seed"):
        adjusted = holm_adjust(
            {
                str(row.task_id): float(row.p_value)
                for row in frame.itertuples()
                if pd.notna(row.p_value)
            }
        )
        for task_id, value in adjusted.items():
            adjusted_values[(int(seed), task_id)] = value
    per_seed["p_holm"] = [
        adjusted_values.get((int(row.seed), str(row.task_id)), np.nan)
        for row in per_seed.itertuples()
    ]
    per_seed["joint_wins"] = per_seed.oriented_improvement > 0
    per_seed["strict_positive_ci"] = per_seed.ci_low > 0

    summary = (
        per_seed.groupby(
            ["task_id", "task_name_cn", "metric", "lower_is_better"],
            as_index=False,
        )
        .agg(
            participants=("participants", "min"),
            samples=("samples", "min"),
            single_mean=("single_metric", "mean"),
            single_std=("single_metric", "std"),
            joint_mean=("joint_metric", "mean"),
            joint_std=("joint_metric", "std"),
            oriented_improvement_mean=("oriented_improvement", "mean"),
            relative_improvement_mean=("relative_improvement", "mean"),
            joint_seed_wins=("joint_wins", "sum"),
            strict_positive_ci_seeds=("strict_positive_ci", "sum"),
            holm_significant_seeds=("p_holm", lambda x: int((x < 0.05).sum())),
        )
        .sort_values("task_id")
    )
    parameters = pd.DataFrame(parameter_rows)
    args.root.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(args.root / "per_seed_task_results.csv", index=False)
    summary.to_csv(args.root / "three_seed_summary.csv", index=False)
    parameters.to_csv(args.root / "parameter_summary.csv", index=False)

    majority_positive = summary.joint_seed_wins >= 2
    mean_positive = summary.oriented_improvement_mean > 0
    joint_majority_wins = int(majority_positive.sum())
    joint_mean_wins = int(mean_positive.sum())
    joint_consensus_wins = int((majority_positive & mean_positive).sum())
    mixed_tasks = int((majority_positive != mean_positive).sum())
    joint_consensus_losses = int((~majority_positive & ~mean_positive).sum())
    strict_tasks = int((summary.strict_positive_ci_seeds >= 2).sum())
    manifest = {
        "format_version": 1,
        "experiment": "mcphases_six_task_joint_vs_task_isolated",
        "seeds": list(args.seeds),
        "split": "validation",
        "test_opened": False,
        "bootstrap_draws": args.bootstrap_draws,
        "tasks": list(TASK_META),
        "joint_majority_wins": joint_majority_wins,
        "joint_mean_wins": joint_mean_wins,
        "joint_consensus_wins": joint_consensus_wins,
        "mixed_direction_tasks": mixed_tasks,
        "joint_consensus_losses": joint_consensus_losses,
        "tasks_with_positive_ci_in_at_least_two_seeds": strict_tasks,
        "onset_definition": "24h and 72h are one isolated nested-probability family",
        "checkpoint_selection": "fixed final step; no validation early stopping",
    }
    (args.root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# mcPHASES六任务：联合训练与任务隔离训练",
        "",
        "固定768维输入、60天历史、128维主干、1,000步训练和参与者划分。",
        "24/72小时月经开始作为一个嵌套概率任务族隔离训练；其余任务各训练一个独立模型。",
        "检查点固定取最后一步，不使用验证集早停；置信区间为2,000次参与者级配对Bootstrap。",
        "",
        "| 任务 | 指标 | 隔离模型 | 六任务联合 | 相对改善 | 联合胜种子 | CI严格为正种子 | Holm显著种子 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.task_name_cn} | {row.metric} | "
            f"{row.single_mean:.4f} ± {row.single_std:.4f} | "
            f"{row.joint_mean:.4f} ± {row.joint_std:.4f} | "
            f"{100.0 * row.relative_improvement_mean:+.2f}% | "
            f"{int(row.joint_seed_wins)}/3 | "
            f"{int(row.strict_positive_ci_seeds)}/3 | "
            f"{int(row.holm_significant_seeds)}/3 |"
        )
    ratio = parameters.parameter_ratio_isolated_over_joint.mean()
    lines.extend(
        [
            "",
            (
                f"联合模型在{joint_consensus_wins}/6项上同时满足三种子均值改善和"
                "至少2/3种子胜出；"
                f"{mixed_tasks}/6项方向混合，{joint_consensus_losses}/6项两种判据均为负。"
            ),
            (
                f"仅按多数种子计为{joint_majority_wins}/6项，仅按三种子均值计为"
                f"{joint_mean_wins}/6项；正文采用更严格的双判据。"
            ),
            f"五个隔离模型的总参数量约为一个联合模型的{ratio:.2f}倍。",
            "没有任何任务在至少两个种子中获得严格为正的参与者Bootstrap区间。",
            "该实验只使用6名固定验证参与者，不是独立外部验证。",
        ]
    )
    (args.root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
