"""Aggregate matched OpenMHC/FemMHC pregnancy probes across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = {
    "mae_weeks": "MAE weeks ↓",
    "rmse_weeks": "RMSE weeks ↓",
    "r2": "R² ↑",
    "spearman": "Spearman ↑",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    seeds = args.seed or [42, 43, 44]

    reports = []
    for seed in seeds:
        directory = args.run_root / f"pregnancy-ga-seed{seed}-summary"
        report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        reports.append((seed, report))

    aggregates = {}
    for metric in METRICS:
        baseline = np.asarray(
            [report["models"]["OpenMHC"][metric] for _, report in reports]
        )
        femmhc = np.asarray(
            [report["models"]["FemMHC"][metric] for _, report in reports]
        )
        lower_is_better = metric in {"mae_weeks", "rmse_weeks"}
        improvement = baseline - femmhc if lower_is_better else femmhc - baseline
        aggregates[metric] = {
            "openmhc_mean": float(baseline.mean()),
            "openmhc_std": float(baseline.std(ddof=1)),
            "femmhc_mean": float(femmhc.mean()),
            "femmhc_std": float(femmhc.std(ddof=1)),
            "absolute_improvement_mean": float(improvement.mean()),
            "relative_improvement_percent": float(
                100.0 * improvement.mean() / abs(baseline.mean())
            ),
            "improved_seeds": int((improvement > 0).sum()),
        }

    output = {
        "format_version": 1,
        "seeds": seeds,
        "cohort": reports[0][1]["cohort"],
        "probe_protocol": reports[0][1]["probe_protocol"],
        "seed_results": [
            {
                "seed": seed,
                "OpenMHC": report["models"]["OpenMHC"],
                "FemMHC": report["models"]["FemMHC"],
                "improvement": report["improvement"],
            }
            for seed, report in reports
        ],
        "aggregate": aggregates,
        "conclusion": "孕期专门化在不同初始化种子上不稳定",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    lines = [
        "# 孕期孕周表征基准",
        "",
        "按参与者划分，在验证集选择 Ridge 探针，OpenMHC 与 FemMHC 使用匹配初始化种子。",
        "",
        "| 种子 | OpenMHC MAE ↓ | FemMHC MAE ↓ | 相对变化 | OpenMHC Spearman ↑ | FemMHC Spearman ↑ |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for seed, report in reports:
        baseline = report["models"]["OpenMHC"]
        femmhc = report["models"]["FemMHC"]
        change = report["improvement"]["mae_relative_percent"]
        lines.append(
            f"| {seed} | {baseline['mae_weeks']:.4f} | {femmhc['mae_weeks']:.4f} | "
            f"{change:+.2f}% | {baseline['spearman']:.4f} | {femmhc['spearman']:.4f} |"
        )
    mae = aggregates["mae_weeks"]
    spearman = aggregates["spearman"]
    lines.extend(
        [
            f"| 均值 ± 标准差 | {mae['openmhc_mean']:.4f} ± {mae['openmhc_std']:.4f} | "
            f"{mae['femmhc_mean']:.4f} ± {mae['femmhc_std']:.4f} | "
            f"{mae['relative_improvement_percent']:+.2f}% | "
            f"{spearman['openmhc_mean']:.4f} | {spearman['femmhc_mean']:.4f} |",
            "",
            f"MAE 在 {mae['improved_seeds']}/{len(seeds)} 个种子上改善。聚合 MAE 变化太小且不稳定，不作为主结果。",
        ]
    )
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(output["aggregate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
