#!/usr/bin/env python
"""Aggregate the locked multi-cohort continual-pretraining feasibility gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


LOWER_IS_BETTER = {"mae", "mae_weeks", "rmse", "brier", "ece"}


def _primary_metrics(path: Path, model: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[
        frame["is_primary"].astype(str).str.lower().eq("true")
        & np.isfinite(frame["value"])
    ].copy()
    frame["model"] = model
    frame["oriented_value"] = np.where(
        frame["metric"].isin(LOWER_IS_BETTER), -frame["value"], frame["value"]
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    baseline_checkpoint = (
        args.baseline_root / "checkpoints" / f"static_adapter_gru-seed{args.seed}.pt"
    )
    candidate_checkpoint = (
        args.root
        / "checkpoints"
        / f"multicohort_continual_gru-seed{args.seed}.pt"
    )
    baseline_metrics_path = (
        args.baseline_root
        / "evaluations"
        / f"static_adapter_gru-seed{args.seed}-validation"
        / "per_task_metrics.csv"
    )
    candidate_metrics_path = (
        args.root
        / "evaluations"
        / f"multicohort_continual_gru-seed{args.seed}-validation"
        / "per_task_metrics.csv"
    )
    baseline_artifact = torch.load(
        baseline_checkpoint, map_location="cpu", weights_only=False
    )
    candidate_artifact = torch.load(
        candidate_checkpoint, map_location="cpu", weights_only=False
    )
    for name, artifact in (
        ("baseline", baseline_artifact),
        ("candidate", candidate_artifact),
    ):
        if artifact.get("checkpoint_selection") != "final_step":
            raise ValueError(f"{name} was not selected at the fixed final step")

    baseline = _primary_metrics(baseline_metrics_path, "baseline")
    candidate = _primary_metrics(candidate_metrics_path, "candidate")
    identifiers = ["task_id", "source", "domain", "kind", "metric", "is_primary"]
    pair = baseline.merge(
        candidate,
        on=identifiers,
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    pair["oriented_delta"] = (
        pair["oriented_value_candidate"] - pair["oriented_value_baseline"]
    )
    pair["candidate_wins"] = pair["oriented_delta"] > 0
    pair["female_specific"] = pair["source"] != "openmhc"
    pair.sort_values(["source", "task_id"]).to_csv(
        args.root / "per_task_comparison.csv", index=False
    )

    source_summary = (
        pair.groupby("source", as_index=False)
        .agg(
            tasks=("task_id", "count"),
            wins=("candidate_wins", "sum"),
            mean_oriented_delta=("oriented_delta", "mean"),
        )
        .sort_values("source")
    )
    source_summary.to_csv(args.root / "source_task_summary.csv", index=False)

    baseline_cohorts = baseline_artifact["cohort_validation_loss"]
    candidate_cohorts = candidate_artifact["cohort_validation_loss"]
    female_domains = {
        "menstrual": ("mcphases",),
        "affective": ("depress_fitbit", "inphrsym"),
        "hrv_sleep": ("wearable_hrv_sleep",),
        "pregnancy": ("pregnancy_ga_clock",),
    }
    domain_rows = []
    for domain, cohorts in female_domains.items():
        baseline_loss = float(np.mean([baseline_cohorts[name] for name in cohorts]))
        candidate_loss = float(np.mean([candidate_cohorts[name] for name in cohorts]))
        domain_rows.append(
            {
                "female_domain": domain,
                "cohorts": "+".join(cohorts),
                "baseline_loss": baseline_loss,
                "candidate_loss": candidate_loss,
                "relative_loss_improvement_percent": 100.0
                * (baseline_loss - candidate_loss)
                / baseline_loss,
                "improved": candidate_loss < baseline_loss,
            }
        )
    domains = pd.DataFrame(domain_rows)
    domains.to_csv(args.root / "female_domain_summary.csv", index=False)

    baseline_loss = float(baseline_artifact["validation_loss"])
    candidate_loss = float(candidate_artifact["validation_loss"])
    all_wins = int(pair["candidate_wins"].sum())
    all_tasks = int(len(pair))
    female = pair["female_specific"]
    female_wins = int(pair.loc[female, "candidate_wins"].sum())
    female_tasks = int(female.sum())
    improved_domains = int(domains["improved"].sum())
    gates = {
        "lower_final_validation_loss": candidate_loss < baseline_loss,
        "majority_of_63_primary_metrics": all_wins > all_tasks / 2,
        "majority_of_33_female_metrics": female_wins > female_tasks / 2,
        "at_least_three_of_four_female_domains": improved_domains >= 3,
    }
    passed = all(gates.values())
    summary = {
        "format_version": 1,
        "status": "complete",
        "seed": args.seed,
        "split": "validation",
        "checkpoint_selection": "final_step",
        "test_split_opened": False,
        "baseline_validation_loss": baseline_loss,
        "candidate_validation_loss": candidate_loss,
        "relative_loss_improvement_percent": 100.0
        * (baseline_loss - candidate_loss)
        / baseline_loss,
        "all_task_wins": all_wins,
        "all_tasks": all_tasks,
        "female_task_wins": female_wins,
        "female_tasks": female_tasks,
        "improved_female_domains": improved_domains,
        "female_domains": len(domains),
        "gates": gates,
        "stage_a_passed": passed,
        "launch_additional_seeds": passed,
        "decision": (
            "run_locked_multiseed_confirmation"
            if passed
            else "stop_architecture_expansion_and_report_not_iclr_ready"
        ),
    }
    (args.root / "manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    stage_status = "通过" if passed else "未通过"
    lines = [
        "# 多队列女性持续预训练：Stage A 收口",
        "",
        "固定 seed=42、编码器 1,000 步、下游 1,000 步，均使用最终步检查点；测试标签未打开。",
        "",
        "## 总门槛",
        "",
        f"- 验证损失：{baseline_loss:.6f} → {candidate_loss:.6f}（{summary['relative_loss_improvement_percent']:+.2f}%）。",
        f"- 63 项主指标：{all_wins}/{all_tasks} 提升。",
        f"- 33 项女性指标：{female_wins}/{female_tasks} 提升。",
        f"- 女性领域：{improved_domains}/{len(domains)} 队列聚合损失降低。",
        f"- Stage A：{stage_status}。",
        "",
        "## 女性领域",
        "",
        "| 领域 | 基线损失 | 候选损失 | 相对改善 |",
        "|---|---:|---:|---:|",
    ]
    for row in domains.itertuples():
        lines.append(
            f"| {row.female_domain} | {row.baseline_loss:.6f} | "
            f"{row.candidate_loss:.6f} | {row.relative_loss_improvement_percent:+.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 决策",
            "",
            "候选表征降低了联合损失，但任务级改善不广泛，因此不启动额外种子，不打开测试集，不再调节训练步数或保持损失权重。",
        ]
    )
    (args.root / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
