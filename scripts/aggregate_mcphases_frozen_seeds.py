"""Aggregate matched mcPHASES frozen-probe results across training seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LOWER_IS_BETTER = {"mae", "brier", "ece_10"}


def parse_result(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("result must be SEED=PATH")
    seed, path = value.split("=", 1)
    return int(seed), Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", type=parse_result, required=True)
    parser.add_argument("--baseline", default="OpenMHC")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frames: list[pd.DataFrame] = []
    paired_frames: list[pd.DataFrame] = []
    for seed, path in args.result:
        frame = pd.read_csv(path)
        frame = frame[frame["is_primary"].astype(str).str.lower().eq("true")].copy()
        frame.insert(0, "seed", seed)
        frames.append(frame)
        paired_path = path.parent / "paired_primary_bootstrap.csv"
        if paired_path.is_file():
            paired = pd.read_csv(paired_path)
            paired.insert(0, "seed", seed)
            paired_frames.append(paired)
    combined = pd.concat(frames, ignore_index=True)
    seeds = sorted(combined["seed"].unique().tolist())
    models = list(dict.fromkeys(combined["model"].astype(str)))
    if args.baseline not in models:
        raise ValueError(f"baseline {args.baseline!r} is absent")

    baseline = combined[combined["model"].eq(args.baseline)][
        ["seed", "task", "value"]
    ].rename(columns={"value": "baseline_value"})
    candidates = combined[~combined["model"].eq(args.baseline)].merge(
        baseline,
        on=["seed", "task"],
        validate="many_to_one",
    )
    higher = ~candidates["metric"].isin(LOWER_IS_BETTER)
    candidates["oriented_absolute_change"] = np.where(
        higher,
        candidates["value"] - candidates["baseline_value"],
        candidates["baseline_value"] - candidates["value"],
    )
    candidates["relative_improvement_percent"] = (
        100.0
        * candidates["oriented_absolute_change"]
        / candidates["baseline_value"].abs().replace(0, np.nan)
    )

    rows: list[dict[str, object]] = []
    for (model, task), group in candidates.groupby(["model", "task"], sort=False):
        first = group.iloc[0]
        rows.append(
            {
                "model": model,
                "task": task,
                "task_chinese": first["task_chinese"],
                "metric": first["metric"],
                "baseline_mean": float(group["baseline_value"].mean()),
                "candidate_mean": float(group["value"].mean()),
                "candidate_std": float(group["value"].std(ddof=1)),
                "relative_improvement_mean_percent": float(
                    group["relative_improvement_percent"].mean()
                ),
                "relative_improvement_std_percent": float(
                    group["relative_improvement_percent"].std(ddof=1)
                ),
                "improved_seeds": int((group["oriented_absolute_change"] > 0).sum()),
                "total_seeds": int(len(group)),
            }
        )
    summary_frame = pd.DataFrame(rows).sort_values(
        ["model", "relative_improvement_mean_percent"], ascending=[True, False]
    )
    model_summary: dict[str, object] = {}
    for model, group in summary_frame.groupby("model", sort=False):
        model_summary[model] = {
            "tasks_improved_on_mean": int(
                (group["relative_improvement_mean_percent"] > 0).sum()
            ),
            "tasks_worse_on_mean": int(
                (group["relative_improvement_mean_percent"] < 0).sum()
            ),
            "tasks_improved_all_seeds": int(
                group["improved_seeds"].eq(group["total_seeds"]).sum()
            ),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(
        args.output_dir / "三种子女性任务冻结探针.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report = {
        "format_version": 1,
        "baseline": args.baseline,
        "seeds": seeds,
        "seed_count": len(seeds),
        "test_participants": 7,
        "models": model_summary,
    }
    paired_summary = pd.DataFrame()
    if paired_frames:
        if len(paired_frames) != len(frames):
            raise ValueError("paired bootstrap files must be present for every seed or none")
        paired_combined = pd.concat(paired_frames, ignore_index=True)
        paired_combined.to_csv(
            args.output_dir / "三种子配对Bootstrap逐种子.csv",
            index=False,
            encoding="utf-8-sig",
        )
        paired_summary = (
            paired_combined.groupby(
                ["candidate", "task", "task_chinese", "metric"],
                as_index=False,
            )
            .agg(
                baseline_mean=("baseline_value", "mean"),
                candidate_mean=("candidate_value", "mean"),
                relative_improvement_mean_percent=(
                    "relative_improvement_percent",
                    "mean",
                ),
                improved_seeds=(
                    "relative_improvement_percent",
                    lambda values: int((values > 0).sum()),
                ),
                strict_ci_seeds=(
                    "ci95_low_percent",
                    lambda values: int((values > 0).sum()),
                ),
                bootstrap_probability_improved_mean=(
                    "bootstrap_probability_improved",
                    "mean",
                ),
            )
            .sort_values(
                ["candidate", "relative_improvement_mean_percent"],
                ascending=[True, False],
            )
        )
        paired_summary.to_csv(
            args.output_dir / "三种子配对Bootstrap汇总.csv",
            index=False,
            encoding="utf-8-sig",
        )
        report["paired_bootstrap"] = {
            "draws_per_seed": int(paired_combined["valid_draws"].max()),
            "tasks_strict_ci_all_seeds": int(
                paired_summary["strict_ci_seeds"].eq(len(seeds)).sum()
            ),
        }
    (args.output_dir / "三种子女性任务冻结探针汇总.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# mcPHASES 女性任务冻结探针三种子汇总",
        "",
        "训练种子：" + "、".join(str(seed) for seed in seeds) + "。正的相对改善表示候选更好。",
        "",
        "| 模型 | 任务 | 指标 | OpenMHC | 候选（均值±标准差） | 相对改善 | 改善种子 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in summary_frame.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.task_chinese} | {row.metric} | "
            f"{row.baseline_mean:.4f} | {row.candidate_mean:.4f} ± {row.candidate_std:.4f} | "
            f"{row.relative_improvement_mean_percent:+.2f}% | "
            f"{row.improved_seeds}/{row.total_seeds} |"
        )
    if not paired_summary.empty:
        lines.extend(
            [
                "",
                "## 参与者级配对 Bootstrap",
                "",
                "| 模型 | 任务 | 平均相对改善 | 改善种子 | 95% CI 严格为正种子 | 平均改善概率 |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in paired_summary.itertuples(index=False):
            lines.append(
                f"| {row.candidate} | {row.task_chinese} | "
                f"{row.relative_improvement_mean_percent:+.2f}% | "
                f"{row.improved_seeds}/{len(seeds)} | "
                f"{row.strict_ci_seeds}/{len(seeds)} | "
                f"{row.bootstrap_probability_improved_mean:.3f} |"
            )
    (args.output_dir / "三种子女性任务冻结探针.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
