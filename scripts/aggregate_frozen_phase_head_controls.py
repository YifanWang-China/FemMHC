#!/usr/bin/env python
"""Aggregate train-selected frozen phase-head controls across random seeds."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


DISPLAY_NAMES = {
    "dual_default": "原始联合头",
    "circular_fixed": "正确圆形头",
    "circular_permuted": "打乱圆形头",
    "learnable_prototype": "可学习原型头",
    "bottleneck_softmax": "瓶颈 Softmax",
    "linear_matched": "等参数线性 Softmax",
    "linear_softmax": "线性 Softmax",
}


def sample_standard_deviation(values: list[float]) -> float:
    return float(stdev(values)) if len(values) > 1 else 0.0


def load_default_baseline(path: Path, seeds: list[int]) -> dict[int, float]:
    result: dict[int, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["task_id"] != "mcphases/cycle_phase":
                continue
            seed = int(row["seed"])
            if seed in seeds:
                result[seed] = float(row["baseline_value"])
    missing = sorted(set(seeds) - set(result))
    if missing:
        raise ValueError(f"default baseline is missing seeds: {missing}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("artifacts/benchmark/femmhc-joint-phase-head-controls"),
    )
    parser.add_argument(
        "--default-baseline-csv",
        type=Path,
        default=Path(
            "artifacts/benchmark/femmhc-joint-frozen-circular-head/"
            "final-vs-dual-multiseed-validation/paired_seed_metrics.csv"
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 73])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    expected_seeds = sorted(args.seeds)
    by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
    for path in sorted(args.input_dir.glob("*-seed*-validation.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        family = str(record["head_family"])
        if int(record["seed"]) in expected_seeds:
            by_family[family].append(record)
    if not by_family:
        raise ValueError(f"no validation records found in {args.input_dir}")

    default_by_seed = load_default_baseline(
        args.default_baseline_csv,
        expected_seeds,
    )
    default_values = [default_by_seed[seed] for seed in expected_seeds]
    default_mean = mean(default_values)

    family_rows: list[dict[str, object]] = []
    for family, records in sorted(by_family.items()):
        records = sorted(records, key=lambda item: int(item["seed"]))
        seeds = [int(item["seed"]) for item in records]
        if seeds != expected_seeds:
            raise ValueError(f"{family} has seeds {seeds}, expected {expected_seeds}")
        if any(bool(item["validation_used_for_selection"]) for item in records):
            raise ValueError(f"{family} used validation for selection")
        if any(bool(item["test_used"]) for item in records):
            raise ValueError(f"{family} used the test split")

        macro_f1 = [float(item["macro_f1"]) for item in records]
        balanced_accuracy = [float(item["balanced_accuracy"]) for item in records]
        parameter_counts = {int(item["trainable_parameters"]) for item in records}
        if len(parameter_counts) != 1:
            raise ValueError(f"{family} has inconsistent parameter counts")
        family_rows.append(
            {
                "head_family": family,
                "display_name": DISPLAY_NAMES.get(family, family),
                "trainable_parameters": parameter_counts.pop(),
                "geometry_weight": float(records[0]["geometry_weight"]),
                "learning_rate": float(records[0]["learning_rate"]),
                "steps": int(records[0]["steps"]),
                "macro_f1_by_seed": dict(zip(expected_seeds, macro_f1, strict=True)),
                "macro_f1_mean": mean(macro_f1),
                "macro_f1_sample_sd": sample_standard_deviation(macro_f1),
                "balanced_accuracy_mean": mean(balanced_accuracy),
                "balanced_accuracy_sample_sd": sample_standard_deviation(
                    balanced_accuracy
                ),
                "absolute_delta_vs_default": mean(macro_f1) - default_mean,
                "relative_delta_vs_default_percent": (
                    (mean(macro_f1) / default_mean - 1.0) * 100.0
                ),
            }
        )

    circular = next(
        row for row in family_rows if row["head_family"] == "circular_fixed"
    )
    circular_mean = float(circular["macro_f1_mean"])
    for row in family_rows:
        row["absolute_delta_vs_circular_fixed"] = (
            float(row["macro_f1_mean"]) - circular_mean
        )
    family_rows.sort(key=lambda row: float(row["macro_f1_mean"]), reverse=True)
    for rank, row in enumerate(family_rows, start=1):
        row["validation_rank"] = rank

    default_row = {
        "head_family": "dual_default",
        "display_name": DISPLAY_NAMES["dual_default"],
        "trainable_parameters": 0,
        "macro_f1_by_seed": default_by_seed,
        "macro_f1_mean": default_mean,
        "macro_f1_sample_sd": sample_standard_deviation(default_values),
        "absolute_delta_vs_default": 0.0,
        "relative_delta_vs_default_percent": 0.0,
        "absolute_delta_vs_circular_fixed": default_mean - circular_mean,
    }
    summary = {
        "format_version": 1,
        "split": "validation",
        "selection_split": "train_only",
        "test_used": False,
        "seeds": expected_seeds,
        "validation_participants": 6,
        "validation_samples": 528,
        "default_baseline": default_row,
        "phase_head_controls": family_rows,
        "best_control": family_rows[0]["head_family"],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_fields = [
        "validation_rank",
        "head_family",
        "display_name",
        "trainable_parameters",
        "geometry_weight",
        "learning_rate",
        "steps",
        "macro_f1_mean",
        "macro_f1_sample_sd",
        "balanced_accuracy_mean",
        "balanced_accuracy_sample_sd",
        "absolute_delta_vs_default",
        "relative_delta_vs_default_percent",
        "absolute_delta_vs_circular_fixed",
    ]
    with (args.output_dir / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(family_rows)

    lines = [
        "# 冻结周期任务头同协议对照",
        "",
        "所有配置仅由 29 名训练参与者的分组交叉验证选择；验证集固定为 6 名参与者、528 个样本；测试集未使用。",
        "",
        "| 排名 | 周期头 | 可训练参数 | 宏平均 F1（均值±样本标准差） | 相对原始头 | 相对正确圆形头 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in family_rows:
        lines.append(
            "| {rank} | {name} | {params} | {mean:.4f} ± {sd:.4f} | "
            "{default:+.4f} | {circle:+.4f} |".format(
                rank=row["validation_rank"],
                name=row["display_name"],
                params=row["trainable_parameters"],
                mean=row["macro_f1_mean"],
                sd=row["macro_f1_sample_sd"],
                default=row["absolute_delta_vs_default"],
                circle=row["absolute_delta_vs_circular_fixed"],
            )
        )
    lines.extend(
        [
            "",
            "原始联合头宏平均 F1："
            f"{default_mean:.4f} ± {sample_standard_deviation(default_values):.4f}。",
        ]
    )
    (args.output_dir / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
