#!/usr/bin/env python
"""Aggregate the single-seed FemMHC-HCA gate without opening the test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


MODELS = (
    "last_day_shared",
    "shared_backbone",
    "mmoe",
    "dual_path_router",
    "history_conditioned_adapter",
)
BASELINES = MODELS[:-1]
SEED = 42
LOWER_IS_BETTER = {"mae", "mae_weeks", "rmse", "brier", "ece"}
SIX_TASKS = (
    "mcphases/cycle_phase",
    "mcphases/menstrual_onset_24h",
    "mcphases/menstrual_onset_72h",
    "mcphases/cramps",
    "mcphases/mood_swing",
    "mcphases/sleep_issue",
)


def load(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = []
    metrics = []
    for model in MODELS:
        checkpoint = root / "checkpoints" / f"{model}-seed{SEED}.pt"
        evaluation = (
            root / "evaluations" / f"{model}-seed{SEED}-validation" / "per_task_metrics.csv"
        )
        artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
        frame = pd.read_csv(evaluation)
        frame = frame[frame["is_primary"].astype(str).str.lower().eq("true")].copy()
        frame["model"] = model
        frame["oriented_value"] = np.where(
            frame["metric"].isin(LOWER_IS_BETTER), -frame["value"], frame["value"]
        )
        metrics.append(frame)
        runs.append(
            {
                "model": model,
                "seed": SEED,
                "hidden_dim": int(artifact["hidden_dim"]),
                "trainable_parameters": int(artifact["trainable_parameters"]),
                "steps": int(artifact["step"]),
                "validation_loss": float(artifact["validation_loss"]),
            }
        )
    return pd.DataFrame(runs), pd.concat(metrics, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/benchmark/femmhc-hca-gate-seed42"))
    args = parser.parse_args()
    root = args.root
    runs, metrics = load(root)
    wide = metrics.pivot(
        index=["task_id", "source", "domain"],
        columns="model",
        values="oriented_value",
    ).reset_index()
    rows = []
    for baseline in BASELINES:
        delta = wide["history_conditioned_adapter"] - wide[baseline]
        female = wide["source"].ne("openmhc")
        rows.append(
            {
                "baseline": baseline,
                "candidate": "history_conditioned_adapter",
                "candidate_loss": float(runs.loc[runs.model.eq("history_conditioned_adapter"), "validation_loss"].iloc[0]),
                "baseline_loss": float(runs.loc[runs.model.eq(baseline), "validation_loss"].iloc[0]),
                "relative_loss_change_percent": float(
                    100.0
                    * (runs.loc[runs.model.eq(baseline), "validation_loss"].iloc[0]
                       - runs.loc[runs.model.eq("history_conditioned_adapter"), "validation_loss"].iloc[0])
                    / runs.loc[runs.model.eq(baseline), "validation_loss"].iloc[0]
                ),
                "all_task_wins": int((delta > 0).sum()),
                "all_tasks": int(len(delta)),
                "female_task_wins": int((delta[female] > 0).sum()),
                "female_tasks": int(female.sum()),
                "six_task_wins": int((delta[wide.task_id.isin(SIX_TASKS)] > 0).sum()),
                "six_tasks": len(SIX_TASKS),
                "mean_oriented_delta": float(delta.mean()),
                "female_mean_oriented_delta": float(delta[female].mean()),
            }
        )
    pairwise = pd.DataFrame(rows)
    six = wide[wide.task_id.isin(SIX_TASKS)].copy()
    six.to_csv(root / "six_task_primary_metrics.csv", index=False)
    runs.to_csv(root / "run_audit.csv", index=False)
    pairwise.to_csv(root / "pairwise_summary.csv", index=False)
    metrics.to_csv(root / "all_primary_metrics_long.csv", index=False)
    report = {
        "format_version": 1,
        "status": "complete",
        "seed": SEED,
        "test_split_opened": False,
        "models": list(MODELS),
        "female_task_definition": "source != openmhc",
        "six_tasks": list(SIX_TASKS),
        "pairwise": pairwise.to_dict(orient="records"),
    }
    (root / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# FemMHC-HCA 单种子门控实验",
        "",
        "协议：seed=42，固定1000步，最终检查点，五个模型使用相同768维缓存OpenMHC输入、参与者划分、采样和优化器；测试集未打开。",
        "",
        "## 系统结果",
        "",
        "| 模型 | 隐藏维度 | 可训练参数 | 验证损失 |",
        "|---|---:|---:|---:|",
    ]
    for row in runs.itertuples():
        lines.append(
            f"| {row.model} | {row.hidden_dim} | {row.trainable_parameters:,} | {row.validation_loss:.4f} |"
        )
    lines += [
        "",
        "## 对照结果",
        "",
        "| 对照模型 | HCA相对损失变化 | 全部任务胜出 | 女性任务胜出 | 六项核心任务胜出 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in pairwise.itertuples():
        lines.append(
            f"| {row.baseline} | {row.relative_loss_change_percent:+.2f}% | {row.all_task_wins}/{row.all_tasks} | {row.female_task_wins}/{row.female_tasks} | {row.six_task_wins}/{row.six_tasks} |"
        )
    lines += [
        "",
        "## 结论",
        "",
        "当前HCA在周期阶段任务上取得最高macro-F1，但系统级验证损失和多数任务不优于GRU/MMoE。因此本次单种子门控未通过，不能直接进入三种子正式实验。下一步先检查HCA的身份：当前版本是在缓存OpenMHC日表征上进行的历史条件化，而不是OpenMHC LSM2主干内部适配；需要先做简化结构和训练目标诊断，再决定是否实现主干内版本。",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "root": str(root.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
