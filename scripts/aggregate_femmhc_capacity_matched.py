#!/usr/bin/env python
"""Aggregate the preregistered capacity-matched four-model comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


MODELS = (
    "last_day_shared",
    "shared_backbone",
    "mmoe",
    "dual_path_router",
)
BASELINES = MODELS[:-1]
CANDIDATE = MODELS[-1]
SEEDS = (17, 42, 73)
LOWER_IS_BETTER = {"mae", "mae_weeks", "rmse", "brier", "ece"}


def _primary_metrics(path: Path, model: str, seed: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    primary = frame[frame["is_primary"].astype(str).str.lower().eq("true")].copy()
    primary = primary[np.isfinite(primary["value"])].copy()
    primary["model"] = model
    primary["seed"] = seed
    primary["oriented_value"] = np.where(
        primary["metric"].isin(LOWER_IS_BETTER),
        -primary["value"],
        primary["value"],
    )
    return primary


def _mean_std(values: pd.Series) -> str:
    return f"{values.mean():.4f} ± {values.std(ddof=1):.4f}"


def _load(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, Any]] = []
    metric_frames = []
    for model in MODELS:
        for seed in SEEDS:
            checkpoint = root / "checkpoints" / f"{model}-seed{seed}.pt"
            evaluation = (
                root
                / "evaluations"
                / f"{model}-seed{seed}-validation"
                / "per_task_metrics.csv"
            )
            if not checkpoint.is_file() or not evaluation.is_file():
                raise FileNotFoundError(f"incomplete run: {model} seed={seed}")
            artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if int(artifact["step"]) != 1000:
                raise ValueError(f"{checkpoint} is not the fixed 1,000-step checkpoint")
            if artifact.get("checkpoint_selection") != "final_step":
                raise ValueError(f"{checkpoint} was not selected by final_step")
            run_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "hidden_dim": int(artifact["hidden_dim"]),
                    "trainable_parameters": int(artifact["trainable_parameters"]),
                    "step": int(artifact["step"]),
                    "final_validation_loss": float(artifact["validation_loss"]),
                }
            )
            metric_frames.append(_primary_metrics(evaluation, model, seed))
    runs = pd.DataFrame(run_rows)
    metrics = pd.concat(metric_frames, ignore_index=True)
    duplicates = metrics.duplicated(["model", "seed", "task_id"], keep=False)
    if duplicates.any():
        raise ValueError("each model/seed/task must have exactly one primary metric")
    complete = metrics.groupby(["model", "seed"])["task_id"].nunique()
    if set(complete) != {63}:
        raise ValueError(f"expected 63 comparable primary tasks per run, got {complete}")
    return runs, metrics


def _pairwise(metrics: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for baseline in BASELINES:
        pair = metrics[metrics["model"].isin([baseline, CANDIDATE])]
        wide = pair.pivot(
            index=["task_id", "source", "domain", "seed"],
            columns="model",
            values="oriented_value",
        ).reset_index()
        wide["delta"] = wide[CANDIDATE] - wide[baseline]
        task_mean = wide.groupby(["task_id", "source"], as_index=False)["delta"].mean()
        baseline_loss = runs[runs["model"].eq(baseline)].set_index("seed")
        candidate_loss = runs[runs["model"].eq(CANDIDATE)].set_index("seed")
        losses = baseline_loss[["final_validation_loss"]].join(
            candidate_loss[["final_validation_loss"]],
            lsuffix="_baseline",
            rsuffix="_candidate",
        )
        loss_delta = (
            losses["final_validation_loss_baseline"]
            - losses["final_validation_loss_candidate"]
        )
        all_wins = int((task_mean["delta"] > 0).sum())
        female = task_mean[task_mean["source"].ne("openmhc")]
        female_wins = int((female["delta"] > 0).sum())
        criteria = {
            "lower_mean_loss": bool(loss_delta.mean() > 0),
            "loss_seed_wins_at_least_two": bool((loss_delta > 0).sum() >= 2),
            "majority_all_tasks": bool(all_wins > len(task_mean) / 2),
            "majority_female_tasks": bool(female_wins > len(female) / 2),
        }
        rows.append(
            {
                "baseline": baseline,
                "baseline_loss_mean": losses["final_validation_loss_baseline"].mean(),
                "baseline_loss_std": losses["final_validation_loss_baseline"].std(ddof=1),
                "candidate_loss_mean": losses["final_validation_loss_candidate"].mean(),
                "candidate_loss_std": losses["final_validation_loss_candidate"].std(ddof=1),
                "relative_loss_improvement_percent": 100
                * loss_delta.mean()
                / losses["final_validation_loss_baseline"].mean(),
                "candidate_loss_seed_wins": int((loss_delta > 0).sum()),
                "all_task_wins": all_wins,
                "all_tasks": int(len(task_mean)),
                "female_task_wins": female_wins,
                "female_tasks": int(len(female)),
                **criteria,
                "passes_all_preregistered_criteria": bool(all(criteria.values())),
            }
        )
    return pd.DataFrame(rows)


def _model_summary(metrics: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    normalized = metrics.copy()
    group = normalized.groupby(["task_id", "seed"])["oriented_value"]
    minimum = group.transform("min")
    span = group.transform("max") - minimum
    normalized["normalized_utility"] = np.where(
        span > 1e-12,
        (normalized["oriented_value"] - minimum) / span,
        0.5,
    )
    utilities = normalized.groupby("model")["normalized_utility"].agg(
        normalized_utility_mean="mean",
        normalized_utility_std="std",
    )
    losses = runs.groupby("model").agg(
        hidden_dim=("hidden_dim", "first"),
        trainable_parameters=("trainable_parameters", "first"),
        final_validation_loss_mean=("final_validation_loss", "mean"),
        final_validation_loss_std=("final_validation_loss", "std"),
    )
    best = runs.loc[runs.groupby("seed")["final_validation_loss"].idxmin()]
    losses["best_loss_seeds"] = best["model"].value_counts().reindex(losses.index, fill_value=0)
    result = losses.join(utilities).reset_index()
    target = int(
        result.loc[result["model"].eq(CANDIDATE), "trainable_parameters"].iloc[0]
    )
    result["parameter_gap_percent_vs_femmhc"] = (
        100 * (result["trainable_parameters"] - target) / target
    )
    return result


def _bootstrap_summary(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for baseline in BASELINES:
        for seed in SEEDS:
            path = (
                root
                / "bootstrap"
                / f"dual-vs-{baseline}"
                / f"seed{seed}"
                / "paired_participant_bootstrap.csv"
            )
            if not path.is_file():
                raise FileNotFoundError(f"missing bootstrap output: {path}")
            frame = pd.read_csv(path)
            frame["baseline"] = baseline
            frame["seed"] = seed
            frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["ci_positive"] = combined["eligible"] & (combined["confidence_low"] > 0)
    combined["ci_negative"] = combined["eligible"] & (combined["confidence_high"] < 0)
    combined["holm_positive"] = (
        combined["eligible"]
        & (combined["oriented_delta"] > 0)
        & (combined["holm_adjusted_p"] < 0.05)
    )
    combined["holm_negative"] = (
        combined["eligible"]
        & (combined["oriented_delta"] < 0)
        & (combined["holm_adjusted_p"] < 0.05)
    )
    per_task = (
        combined.groupby(
            ["baseline", "task_id", "source", "domain"], as_index=False
        )
        .agg(
            eligible_seeds=("eligible", "sum"),
            point_win_seeds=("oriented_delta", lambda x: int((x > 0).sum())),
            ci_positive_seeds=("ci_positive", "sum"),
            ci_negative_seeds=("ci_negative", "sum"),
            holm_positive_seeds=("holm_positive", "sum"),
            holm_negative_seeds=("holm_negative", "sum"),
        )
    )
    rows = []
    for baseline, frame in per_task.groupby("baseline"):
        eligible = frame[frame["eligible_seeds"].eq(len(SEEDS))]
        rows.append(
            {
                "baseline": baseline,
                "tasks": int(len(frame)),
                "eligible_all_three_seeds": int(len(eligible)),
                "ci_positive_at_least_two_seeds": int(
                    (eligible["ci_positive_seeds"] >= 2).sum()
                ),
                "ci_negative_at_least_two_seeds": int(
                    (eligible["ci_negative_seeds"] >= 2).sum()
                ),
                "holm_positive_at_least_two_seeds": int(
                    (eligible["holm_positive_seeds"] >= 2).sum()
                ),
                "holm_negative_at_least_two_seeds": int(
                    (eligible["holm_negative_seeds"] >= 2).sum()
                ),
            }
        )
    return per_task, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/benchmark/femmhc-capacity-matched-v1"),
    )
    args = parser.parse_args()
    root = args.root
    runs, metrics = _load(root)
    models = _model_summary(metrics, runs)
    pairwise = _pairwise(metrics, runs)
    bootstrap_tasks, bootstrap_pairs = _bootstrap_summary(root)

    runs.to_csv(root / "run_audit.csv", index=False)
    models.to_csv(root / "model_summary.csv", index=False)
    pairwise.to_csv(root / "pairwise_summary.csv", index=False)
    bootstrap_tasks.to_csv(root / "bootstrap_multiseed_tasks.csv", index=False)
    bootstrap_pairs.to_csv(root / "bootstrap_pair_summary.csv", index=False)

    all_pass = bool(pairwise["passes_all_preregistered_criteria"].all())
    manifest = {
        "format_version": 1,
        "status": "complete",
        "candidate": CANDIDATE,
        "baselines": list(BASELINES),
        "seeds": list(SEEDS),
        "checkpoint_selection": "final_step",
        "training_steps": 1000,
        "primary_tasks": 63,
        "female_specific_primary_tasks": 33,
        "test_split_opened": False,
        "all_preregistered_criteria_passed_against_every_baseline": all_pass,
        "baselines_fully_passed": pairwise.loc[
            pairwise["passes_all_preregistered_criteria"], "baseline"
        ].tolist(),
        "baselines_not_fully_passed": pairwise.loc[
            ~pairwise["passes_all_preregistered_criteria"], "baseline"
        ].tolist(),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    labels = {
        "last_day_shared": "最后一天共享MLP",
        "shared_backbone": "GRU时序适配器",
        "mmoe": "MMoE（8专家）",
        "dual_path_router": "FemMHC双路径",
    }
    lines = [
        "# FemMHC四模型参数匹配对照",
        "",
        "固定1,000步、固定最终检查点、3个随机种子、相同768维OpenMHC输入与相同参与者划分。测试集保持封存。",
        "",
        "## 系统级结果",
        "",
        "| 模型 | 隐层 | 参数量 | 参数差 | 第1,000步验证损失 | 单种子最优 | 归一化任务效用 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in models.itertuples():
        lines.append(
            f"| {labels[row.model]} | {row.hidden_dim} | {row.trainable_parameters:,} | "
            f"{row.parameter_gap_percent_vs_femmhc:+.2f}% | "
            f"{row.final_validation_loss_mean:.4f} ± {row.final_validation_loss_std:.4f} | "
            f"{row.best_loss_seeds}/3 | {row.normalized_utility_mean:.4f} |"
        )
    lines.extend(
        [
            "",
            "## FemMHC成对结果",
            "",
            "| 基线 | 损失相对改善 | 损失胜种子 | 63任务胜 | 33女性任务胜 | 全部预注册判据 |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in pairwise.itertuples():
        criterion_status = "通过" if row.passes_all_preregistered_criteria else "未通过"
        lines.append(
            f"| {labels[row.baseline]} | {row.relative_loss_improvement_percent:+.2f}% | "
            f"{row.candidate_loss_seed_wins}/3 | {row.all_task_wins}/63 | "
            f"{row.female_task_wins}/33 | "
            f"{criterion_status} |"
        )
    lines.extend(["", "## 参与者Bootstrap", ""])
    lines.append(
        "| 基线 | 三种子均可推断任务 | CI正向≥2种子 | CI负向≥2种子 | Holm正向≥2种子 | Holm负向≥2种子 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in bootstrap_pairs.itertuples():
        lines.append(
            f"| {labels[row.baseline]} | {row.eligible_all_three_seeds} | "
            f"{row.ci_positive_at_least_two_seeds} | {row.ci_negative_at_least_two_seeds} | "
            f"{row.holm_positive_at_least_two_seeds} | {row.holm_negative_at_least_two_seeds} |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            (
                "FemMHC通过了对所有三个基线的预注册稳定性判据。"
                if all_pass
                else "FemMHC没有通过对所有三个基线的预注册稳定性判据，不能声称参数匹配后稳定优于普通GRU和MMoE。"
            ),
            "本表只使用验证集，没有打开已封存测试集。",
        ]
    )
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
