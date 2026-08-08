"""Create a compact matched report for the female HRV mental-health probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openmhc", type=Path, required=True)
    parser.add_argument("--femmhc", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.openmhc.read_text(encoding="utf-8"))
    femmhc = json.loads(args.femmhc.read_text(encoding="utf-8"))
    baseline_by_task = {row["task"]: row for row in baseline["results"]}
    femmhc_by_task = {row["task"]: row for row in femmhc["results"]}
    if baseline_by_task.keys() != femmhc_by_task.keys():
        raise ValueError("OpenMHC and FemMHC task sets differ")

    rows = []
    for task, baseline_row in baseline_by_task.items():
        femmhc_row = femmhc_by_task[task]
        openmhc_mae = baseline_row["embedding_ridge"]["mae"]
        femmhc_mae = femmhc_row["embedding_ridge"]["mae"]
        rows.append(
            {
                "task": task,
                "label": baseline_row["label"],
                "input_window_days": baseline_row["input_window_days"],
                "openmhc_mae": openmhc_mae,
                "femmhc_mae": femmhc_mae,
                "relative_mae_improvement_percent": 100.0
                * (openmhc_mae - femmhc_mae)
                / openmhc_mae,
                "openmhc_spearman": baseline_row["embedding_ridge"]["spearman"],
                "femmhc_spearman": femmhc_row["embedding_ridge"]["spearman"],
                "training_median_mae": baseline_row["training_median"]["mae"],
            }
        )
    report = {
        "format_version": 1,
        "female_participants": baseline["female_participants"],
        "split_participants": baseline["split_participants"],
        "alignment": baseline["alignment"],
        "improved_tasks": sum(
            row["femmhc_mae"] < row["openmhc_mae"] for row in rows
        ),
        "task_count": len(rows),
        "headline_eligible": False,
        "reason": "测试集只有4名女性，且量表仅有粗粒度时间点对齐",
        "results": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 女性 HRV 心理健康迁移探针",
        "",
        "女性参与者划分：17 人训练 / 4 人验证 / 4 人测试。",
        "",
        "| 任务 | OpenMHC MAE ↓ | FemMHC MAE ↓ | 相对变化 | 训练集中位数 MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['openmhc_mae']:.3f} | {row['femmhc_mae']:.3f} | "
            f"{row['relative_mae_improvement_percent']:+.2f}% | {row['training_median_mae']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"FemMHC 在 {report['improved_tasks']}/{report['task_count']} 个匹配 MAE 探针上改善。",
            "这是支持性结果，不是主结果：测试集只有 4 名女性，公开量表仅提供开始/中期/末期的粗粒度时间标记。",
        ]
    )
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
