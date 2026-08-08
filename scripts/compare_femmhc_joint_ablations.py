#!/usr/bin/env python
"""Compare validation-only FemMHC joint-architecture ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LOWER_IS_BETTER = {"mae", "mae_weeks", "rmse", "brier", "ece"}


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be NAME=OUTPUT_DIR")
    name, raw_path = value.split("=", 1)
    if not name.strip():
        raise argparse.ArgumentTypeError("run name cannot be empty")
    path = Path(raw_path)
    if not (path / "per_task_metrics.csv").is_file():
        raise argparse.ArgumentTypeError(
            f"missing per_task_metrics.csv under {path}"
        )
    return name.strip(), path


def _load_primary(name: str, directory: Path) -> pd.DataFrame:
    frame = pd.read_csv(directory / "per_task_metrics.csv")
    primary = frame[frame["is_primary"].astype(str).str.lower() == "true"].copy()
    primary["run"] = name
    primary["oriented_value"] = np.where(
        primary["metric"].isin(LOWER_IS_BETTER),
        -primary["value"],
        primary["value"],
    )
    return primary


def _group_summary(frame: pd.DataFrame, run_names: list[str]) -> dict[str, object]:
    frame = frame[np.isfinite(frame["oriented_value"])].copy()
    complete = frame.groupby("task_id")["run"].nunique()
    comparable_ids = complete[complete == len(run_names)].index
    comparable = frame[frame["task_id"].isin(comparable_ids)].copy()
    comparable["rank"] = comparable.groupby("task_id")["oriented_value"].rank(
        method="average", ascending=False
    )
    summary: dict[str, object] = {
        "comparable_tasks": int(len(comparable_ids)),
        "runs": {},
    }
    for name in run_names:
        values = comparable[comparable["run"] == name]
        summary["runs"][name] = {
            "mean_task_rank": float(values["rank"].mean()),
            "first_place_tasks": int((values["rank"] == 1.0).sum()),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        type=_parse_run,
        required=True,
        help="NAME=directory containing per_task_metrics.csv; repeat per run",
    )
    parser.add_argument("--reference", default="full")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    runs = dict(args.run)
    if len(runs) != len(args.run):
        raise ValueError("run names must be unique")
    if args.reference not in runs:
        raise ValueError("reference must match one --run name")

    combined = pd.concat(
        [_load_primary(name, path) for name, path in runs.items()],
        ignore_index=True,
    )
    run_names = list(runs)
    summary = {
        "format_version": 1,
        "split": "validation",
        "reference": args.reference,
        "all_tasks": _group_summary(combined, run_names),
        "openmhc_tasks": _group_summary(
            combined[combined["source"] == "openmhc"], run_names
        ),
        "female_specific_tasks": _group_summary(
            combined[combined["source"] != "openmhc"], run_names
        ),
        "domains": {
            domain: _group_summary(group, run_names)
            for domain, group in combined.groupby("domain", sort=True)
        },
        "sources": {
            source: _group_summary(group, run_names)
            for source, group in combined.groupby("source", sort=True)
        },
    }

    index_columns = ["task_id", "source", "domain", "metric"]
    wide = combined.pivot_table(
        index=index_columns,
        columns="run",
        values="value",
        aggfunc="first",
    ).reset_index()
    reference_oriented = combined[combined["run"] == args.reference][
        ["task_id", "oriented_value"]
    ].rename(columns={"oriented_value": "reference_oriented_value"})
    pairwise = []
    for name in run_names:
        if name == args.reference:
            continue
        candidate = combined[combined["run"] == name][
            ["task_id", "oriented_value"]
        ].rename(columns={"oriented_value": "baseline_oriented_value"})
        comparison = reference_oriented.merge(candidate, on="task_id", how="inner")
        comparison = comparison[
            np.isfinite(comparison["reference_oriented_value"])
            & np.isfinite(comparison["baseline_oriented_value"])
        ]
        delta = (
            comparison["reference_oriented_value"]
            - comparison["baseline_oriented_value"]
        )
        pairwise.append(
            {
                "baseline": name,
                "common_tasks": int(len(comparison)),
                "reference_wins": int((delta > 1e-12).sum()),
                "ties": int((delta.abs() <= 1e-12).sum()),
                "reference_losses": int((delta < -1e-12).sum()),
            }
        )
    summary["reference_pairwise"] = pairwise

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_dir / "primary_metrics_long.csv", index=False)
    wide.to_csv(args.output_dir / "primary_metrics_wide.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# FemMHC 联合架构验证集消融",
        "",
        "> 仅使用验证集；测试集保持锁定。不同量纲任务通过逐任务排名汇总。",
        "",
        "| 范围 | 可比任务数 | "
        + " | ".join(f"{name} 平均排名 / 第一名数" for name in run_names)
        + " |",
        "|---|---:|" + "---:|" * len(run_names),
    ]
    for key, label in (
        ("all_tasks", "全部任务"),
        ("openmhc_tasks", "OpenMHC任务"),
        ("female_specific_tasks", "女性特异任务"),
    ):
        item = summary[key]
        cells = [
            f"{item['runs'][name]['mean_task_rank']:.3f} / "
            f"{item['runs'][name]['first_place_tasks']}"
            for name in run_names
        ]
        lines.append(
            f"| {label} | {item['comparable_tasks']} | " + " | ".join(cells) + " |"
        )
    lines.extend(["", f"完整模型（{args.reference}）逐任务胜负：", ""])
    for item in pairwise:
        lines.append(
            f"- 对 {item['baseline']}："
            f"{item['reference_wins']} 胜 / {item['ties']} 平 / "
            f"{item['reference_losses']} 负（{item['common_tasks']} 个共同任务）。"
        )
    (args.output_dir / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
