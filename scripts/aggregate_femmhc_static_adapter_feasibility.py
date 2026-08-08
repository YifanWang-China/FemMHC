#!/usr/bin/env python
"""Aggregate the locked single-seed static-adapter multicohort feasibility run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ARMS = (
    "openmhc_gru",
    "static_adapter_gru",
    "static_adapter_mmoe",
    "static_adapter_dual_path",
)
LOWER_IS_BETTER = {"mae", "mae_weeks", "rmse", "brier", "ece"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_rows = []
    metric_frames = []
    for arm in ARMS:
        checkpoint = args.root / "checkpoints" / f"{arm}-seed{args.seed}.pt"
        metrics_path = (
            args.root
            / "evaluations"
            / f"{arm}-seed{args.seed}-validation"
            / "per_task_metrics.csv"
        )
        artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if artifact.get("checkpoint_selection") != "final_step":
            raise ValueError(f"{checkpoint} is not a fixed-final-step checkpoint")
        run_rows.append(
            {
                "arm": arm,
                "architecture": artifact["architecture"],
                "parameters": int(artifact["trainable_parameters"]),
                "validation_loss": float(artifact["validation_loss"]),
            }
        )
        frame = pd.read_csv(metrics_path)
        frame = frame[
            frame["is_primary"].astype(str).str.lower().eq("true")
            & np.isfinite(frame["value"])
        ].copy()
        frame["arm"] = arm
        frame["oriented_value"] = np.where(
            frame["metric"].isin(LOWER_IS_BETTER), -frame["value"], frame["value"]
        )
        metric_frames.append(frame)

    runs = pd.DataFrame(run_rows)
    metrics = pd.concat(metric_frames, ignore_index=True)
    comparisons = (
        ("openmhc_gru", "static_adapter_gru", "representation_transfer"),
        ("static_adapter_gru", "static_adapter_dual_path", "dual_vs_gru"),
        ("static_adapter_mmoe", "static_adapter_dual_path", "dual_vs_mmoe"),
    )
    rows = []
    for baseline, candidate, comparison in comparisons:
        pair = metrics[metrics["arm"].isin((baseline, candidate))].pivot(
            index=["task_id", "source", "domain"],
            columns="arm",
            values="oriented_value",
        ).dropna()
        delta = pair[candidate] - pair[baseline]
        female = pair.index.get_level_values("source") != "openmhc"
        baseline_loss = float(runs.loc[runs["arm"].eq(baseline), "validation_loss"].iloc[0])
        candidate_loss = float(runs.loc[runs["arm"].eq(candidate), "validation_loss"].iloc[0])
        rows.append(
            {
                "comparison": comparison,
                "baseline": baseline,
                "candidate": candidate,
                "baseline_loss": baseline_loss,
                "candidate_loss": candidate_loss,
                "relative_loss_improvement_percent": 100.0 * (baseline_loss - candidate_loss) / baseline_loss,
                "all_task_wins": int((delta > 0).sum()),
                "all_tasks": int(len(delta)),
                "female_task_wins": int((delta[female] > 0).sum()),
                "female_tasks": int(female.sum()),
            }
        )
    summary = pd.DataFrame(rows)
    representation_pass = bool(
        (summary.loc[summary["comparison"].eq("representation_transfer"), "candidate_loss"].iloc[0]
         < summary.loc[summary["comparison"].eq("representation_transfer"), "baseline_loss"].iloc[0])
        and (
            summary.loc[summary["comparison"].eq("representation_transfer"), "female_task_wins"].iloc[0]
            > summary.loc[summary["comparison"].eq("representation_transfer"), "female_tasks"].iloc[0] / 2
        )
    )
    architecture_rows = summary[summary["comparison"].isin(("dual_vs_gru", "dual_vs_mmoe"))]
    architecture_pass = bool(
        (architecture_rows["candidate_loss"] < architecture_rows["baseline_loss"]).all()
        and (architecture_rows["female_task_wins"] > architecture_rows["female_tasks"] / 2).all()
    )

    bootstrap = {}
    for name in ("dual-vs-gru", "dual-vs-mmoe"):
        path = args.root / "bootstrap" / name / f"seed{args.seed}" / "summary.json"
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            bootstrap[name] = {
                "replicates": value["replicates"],
                "female_specific_tasks": value["female_specific_tasks"],
            }

    args.root.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.root / "run_summary.csv", index=False)
    summary.to_csv(args.root / "comparison_summary.csv", index=False)
    manifest = {
        "format_version": 1,
        "status": "complete",
        "seed": args.seed,
        "split": "validation",
        "test_split_opened": False,
        "representation_transfer_passed": representation_pass,
        "architecture_transfer_passed": architecture_pass,
        "launch_additional_seeds": representation_pass and architecture_pass,
        "participant_bootstrap": bootstrap,
    }
    (args.root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# 静态女性Adapter多队列可行性实验",
        "",
        "固定seed=42、固定1,000步最终检查点；测试集保持封存。",
        "",
        "## 模型",
        "",
        "| 实验臂 | 架构 | 参数量 | 验证损失 |",
        "|---|---|---:|---:|",
    ]
    for row in runs.itertuples():
        lines.append(
            f"| {row.arm} | {row.architecture} | {row.parameters:,} | {row.validation_loss:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 预注册比较",
            "",
            "| 比较 | 损失相对改善 | 全部任务胜 | 女性任务胜 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in summary.itertuples():
        lines.append(
            f"| {row.comparison} | {row.relative_loss_improvement_percent:+.2f}% | "
            f"{row.all_task_wins}/{row.all_tasks} | {row.female_task_wins}/{row.female_tasks} |"
        )
    lines.extend(
        [
            "",
            "## 决策",
            "",
            f"- 表征迁移门槛：{'通过' if representation_pass else '未通过'}。",
            f"- 架构迁移门槛：{'通过' if architecture_pass else '未通过'}。",
            f"- 是否启动额外种子：{'是' if representation_pass and architecture_pass else '否'}。",
            "- 本实验没有打开测试集。",
        ]
    )
    (args.root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
