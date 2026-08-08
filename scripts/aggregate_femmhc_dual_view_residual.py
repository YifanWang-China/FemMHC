#!/usr/bin/env python
"""Aggregate the preregistered dual-view residual feasibility comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


LOWER_IS_BETTER = {"mae", "mae_weeks", "rmse", "brier", "ece"}
BASELINES = ("openmhc_gru", "static_adapter_gru", "static_adapter_mmoe")
CANDIDATE = "dual_view_residual"


def _metrics(path: Path, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[
        frame["is_primary"].astype(str).str.lower().eq("true")
        & np.isfinite(frame["value"])
    ].copy()
    frame["model"] = name
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

    candidate_checkpoint = args.root / "checkpoints" / f"{CANDIDATE}-seed{args.seed}.pt"
    candidate_evaluation = (
        args.root / "evaluations" / f"{CANDIDATE}-seed{args.seed}-validation" / "per_task_metrics.csv"
    )
    candidate_artifact = torch.load(
        candidate_checkpoint, map_location="cpu", weights_only=False
    )
    candidate_metrics = _metrics(candidate_evaluation, CANDIDATE)
    rows = []
    for baseline in BASELINES:
        checkpoint = args.baseline_root / "checkpoints" / f"{baseline}-seed{args.seed}.pt"
        evaluation = (
            args.baseline_root
            / "evaluations"
            / f"{baseline}-seed{args.seed}-validation"
            / "per_task_metrics.csv"
        )
        artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
        baseline_metrics = _metrics(evaluation, baseline)
        pair = pd.concat([baseline_metrics, candidate_metrics]).pivot(
            index=["task_id", "source", "domain"],
            columns="model",
            values="oriented_value",
        ).dropna()
        delta = pair[CANDIDATE] - pair[baseline]
        female = pair.index.get_level_values("source") != "openmhc"
        baseline_loss = float(artifact["validation_loss"])
        candidate_loss = float(candidate_artifact["validation_loss"])
        rows.append(
            {
                "baseline": baseline,
                "baseline_loss": baseline_loss,
                "candidate_loss": candidate_loss,
                "relative_loss_improvement_percent": 100.0 * (baseline_loss - candidate_loss) / baseline_loss,
                "all_task_wins": int((delta > 0).sum()),
                "all_tasks": int(len(delta)),
                "female_task_wins": int((delta[female] > 0).sum()),
                "female_tasks": int(female.sum()),
            }
        )
    result = pd.DataFrame(rows)
    result["loss_pass"] = result["candidate_loss"] < result["baseline_loss"]
    result["all_task_pass"] = result["all_task_wins"] > result["all_tasks"] / 2
    result["female_task_pass"] = result["female_task_wins"] > result["female_tasks"] / 2
    passed = bool(
        result[["loss_pass", "all_task_pass", "female_task_pass"]].to_numpy().all()
    )
    result.to_csv(args.root / "comparison_summary.csv", index=False)
    manifest = {
        "format_version": 1,
        "status": "complete",
        "seed": args.seed,
        "candidate_parameters": int(candidate_artifact["trainable_parameters"]),
        "candidate_validation_loss": float(candidate_artifact["validation_loss"]),
        "stage_a_passed": passed,
        "launch_additional_seeds": passed,
        "test_split_opened": False,
    }
    (args.root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
