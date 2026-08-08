"""Summarize matched mcPHASES frozen probes with metric-aware directions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LOWER_IS_BETTER = {"mae", "brier", "ece_10"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.results)
    primary = frame[frame["is_primary"].astype(str).str.lower().eq("true")].copy()
    models = list(dict.fromkeys(primary["model"].astype(str)))
    if args.baseline not in models:
        raise ValueError(f"baseline {args.baseline!r} is absent from {args.results}")
    baseline = primary[primary["model"].eq(args.baseline)][
        ["task", "task_chinese", "metric", "value"]
    ].rename(columns={"value": "baseline_value"})

    comparisons: list[pd.DataFrame] = []
    summary: dict[str, object] = {
        "format_version": 1,
        "baseline": args.baseline,
        "tasks": int(baseline["task"].nunique()),
        "models": {},
    }
    for model in models:
        if model == args.baseline:
            continue
        candidate = primary[primary["model"].eq(model)][["task", "value"]].rename(
            columns={"value": "candidate_value"}
        )
        comparison = baseline.merge(candidate, on="task", validate="one_to_one")
        comparison.insert(0, "candidate", model)
        higher = ~comparison["metric"].isin(LOWER_IS_BETTER)
        comparison["oriented_absolute_change"] = np.where(
            higher,
            comparison["candidate_value"] - comparison["baseline_value"],
            comparison["baseline_value"] - comparison["candidate_value"],
        )
        comparison["relative_improvement_percent"] = (
            100.0
            * comparison["oriented_absolute_change"]
            / comparison["baseline_value"].abs().replace(0, np.nan)
        )
        comparison["result"] = np.select(
            [
                comparison["oriented_absolute_change"] > 0,
                comparison["oriented_absolute_change"] < 0,
            ],
            ["win", "loss"],
            default="tie",
        )
        comparisons.append(comparison)
        summary["models"][model] = {
            "wins": int(comparison["result"].eq("win").sum()),
            "losses": int(comparison["result"].eq("loss").sum()),
            "ties": int(comparison["result"].eq("tie").sum()),
            "median_oriented_absolute_change": float(
                comparison["oriented_absolute_change"].median()
            ),
        }

    detailed = pd.concat(comparisons, ignore_index=True)
    detailed = detailed.sort_values(
        ["candidate", "oriented_absolute_change"], ascending=[True, False]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(
        args.output_dir / "女性任务冻结探针逐项变化.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.output_dir / "女性任务冻结探针汇总.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# mcPHASES 女性任务冻结探针消融",
        "",
        f"基线：{args.baseline}。正值表示候选模型更好；MAE 按越低越好统一方向。",
        "",
        "| 候选模型 | 任务 | 指标 | 基线 | 候选 | 相对改善 | 结果 |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in detailed.itertuples(index=False):
        lines.append(
            f"| {row.candidate} | {row.task_chinese} | {row.metric} | "
            f"{row.baseline_value:.4f} | {row.candidate_value:.4f} | "
            f"{row.relative_improvement_percent:+.2f}% | {row.result} |"
        )
    lines.extend(["", "## 汇总", ""])
    for model, item in summary["models"].items():
        lines.append(
            f"- {model}：{item['wins']} 胜/{item['losses']} 负/"
            f"{item['ties']} 平。"
        )
    (args.output_dir / "女性任务冻结探针逐项变化.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
