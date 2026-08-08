#!/usr/bin/env python
"""Validation-only robustness to missing and shortened wearable histories."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from femmhc import JOINT_TASKS, FemMHCJointModel, ProbabilisticOutput
try:
    from evaluate_femmhc_joint import _datasets, _metrics
except ModuleNotFoundError:  # Imported as scripts.* in unit tests.
    from scripts.evaluate_femmhc_joint import _datasets, _metrics


@dataclass(frozen=True)
class MissingnessScenario:
    name: str
    kind: str
    value: float


SCENARIOS: tuple[MissingnessScenario, ...] = (
    MissingnessScenario("baseline", "baseline", 0),
    MissingnessScenario("history_7d", "recent_calendar", 7),
    MissingnessScenario("history_14d", "recent_calendar", 14),
    MissingnessScenario("history_30d", "recent_calendar", 30),
    MissingnessScenario("random_drop_10pct", "random_fraction", 0.10),
    MissingnessScenario("random_drop_25pct", "random_fraction", 0.25),
    MissingnessScenario("random_drop_40pct", "random_fraction", 0.40),
    MissingnessScenario("contiguous_drop_1d", "contiguous_observed", 1),
    MissingnessScenario("contiguous_drop_3d", "contiguous_observed", 3),
    MissingnessScenario("contiguous_drop_7d", "contiguous_observed", 7),
    MissingnessScenario("latest_drop_1d", "latest_observed", 1),
    MissingnessScenario("latest_drop_3d", "latest_observed", 3),
    MissingnessScenario("latest_drop_7d", "latest_observed", 7),
)


LOWER_IS_BETTER = {"mae", "mae_weeks", "rmse", "brier", "ece"}


def _parse_checkpoint(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be SEED=PATH")
    raw_seed, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"checkpoint does not exist: {path}")
    return int(raw_seed), path


def perturb_present(
    present: np.ndarray,
    scenario: MissingnessScenario,
    *,
    seed: int,
    row_offset: int = 0,
) -> np.ndarray:
    """Apply a deterministic per-history mask while retaining at least one day."""

    original = np.asarray(present, dtype=bool)
    if original.ndim != 2:
        raise ValueError("present must have shape (batch, days)")
    result = original.copy()
    if scenario.kind == "baseline":
        return result
    for row in range(len(result)):
        observed = np.flatnonzero(original[row])
        if len(observed) <= 1:
            continue
        if scenario.kind == "recent_calendar":
            days = int(scenario.value)
            if days <= 0:
                raise ValueError("recent history length must be positive")
            cutoff = max(0, original.shape[1] - days)
            result[row, :cutoff] = False
        elif scenario.kind == "random_fraction":
            if not 0 < scenario.value < 1:
                raise ValueError("random missing fraction must be in (0, 1)")
            count = min(len(observed) - 1, max(1, int(round(len(observed) * scenario.value))))
            generator = np.random.default_rng(seed + (row_offset + row) * 1_000_003)
            removed = generator.choice(observed, size=count, replace=False)
            result[row, removed] = False
        elif scenario.kind == "contiguous_observed":
            count = min(len(observed) - 1, int(scenario.value))
            if count <= 0:
                raise ValueError("contiguous missing length must be positive")
            generator = np.random.default_rng(seed + (row_offset + row) * 1_000_003)
            start = int(generator.integers(0, len(observed) - count + 1))
            result[row, observed[start : start + count]] = False
        elif scenario.kind == "latest_observed":
            count = min(len(observed) - 1, int(scenario.value))
            if count <= 0:
                raise ValueError("latest missing length must be positive")
            result[row, observed[-count:]] = False
        else:
            raise ValueError(f"unknown missingness scenario: {scenario.kind}")
        if not result[row].any():
            result[row, observed[-1]] = True
    return result


def _load_model(artifact: dict[str, Any], device: torch.device) -> FemMHCJointModel:
    task_metadata = artifact.get("instantiated_tasks") or artifact.get("active_tasks")
    if task_metadata:
        task_ids = {
            item["task_id"] if isinstance(item, dict) else str(item)
            for item in task_metadata
        }
        tasks = tuple(task for task in JOINT_TASKS if task.task_id in task_ids)
    else:
        tasks = JOINT_TASKS
    model = FemMHCJointModel(
        input_dim=int(artifact["input_dim"]),
        hidden_dim=int(artifact["hidden_dim"]),
        tasks=tasks,
        maximum_days=int(artifact["maximum_days"]),
        architecture=str(artifact["architecture"]),
        dropout=float(artifact.get("dropout", 0.0)),
        routing_initial_logit=float(artifact.get("routing_initial_logit", -2.0)),
    )
    model.load_state_dict(artifact["model_state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def evaluate_scenario(
    model: FemMHCJointModel,
    datasets: dict[str, Any],
    statistics: dict[str, dict[str, float]],
    scenario: MissingnessScenario,
    *,
    batch_size: int,
    mask_seed: int,
    device: torch.device,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    task_by_id = {
        task.task_id: task
        for task in getattr(model.task_heads, "tasks", JOINT_TASKS)
        if task.trainable
    }
    collected: dict[str, dict[str, list[Any]]] = {}
    history_records: list[dict[str, Any]] = []
    for cohort, dataset in datasets.items():
        loader = DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        offset = 0
        before_counts: list[int] = []
        after_counts: list[int] = []
        for batch in loader:
            original_present = batch["day_present"].numpy().astype(bool)
            changed_present = perturb_present(
                original_present,
                scenario,
                seed=mask_seed,
                row_offset=offset,
            )
            offset += len(original_present)
            before_counts.extend(original_present.sum(axis=1).tolist())
            after_counts.extend(changed_present.sum(axis=1).tolist())
            embeddings = batch["daily_embeddings"].to(device, non_blocking=True)
            present = torch.from_numpy(changed_present).to(device, non_blocking=True)
            embeddings = embeddings.masked_fill(~present.unsqueeze(-1), 0.0)
            task_ids = tuple(task_id for task_id in batch["targets"] if task_id in task_by_id)
            if not task_ids:
                continue
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(embeddings, present, task_ids=task_ids)
            participants = np.asarray(batch["participant_id"], dtype=str)
            for task_id in task_ids:
                target_tensor = batch["targets"][task_id]
                task = task_by_id[task_id]
                target = target_tensor.numpy()
                observed = np.isfinite(target)
                if task.kind != "regression":
                    observed &= target >= 0
                if not observed.any():
                    continue
                prediction_output = output.predictions[task_id]
                record = collected.setdefault(
                    task_id,
                    {
                        "target": [],
                        "prediction": [],
                        "probabilities": [],
                        "participant": [],
                        "cohort": cohort,
                    },
                )
                if isinstance(prediction_output, ProbabilisticOutput):
                    probabilities = prediction_output.probabilities.float().cpu().numpy()
                    prediction = probabilities.argmax(axis=1)
                    record["probabilities"].append(probabilities[observed])
                else:
                    prediction = prediction_output.float().cpu().numpy()
                    if task_id in statistics:
                        prediction = (
                            prediction * statistics[task_id]["std"]
                            + statistics[task_id]["mean"]
                        )
                record["target"].append(target[observed])
                record["prediction"].append(prediction[observed])
                record["participant"].extend(participants[observed].tolist())
        before = np.asarray(before_counts, dtype=np.float64)
        after = np.asarray(after_counts, dtype=np.float64)
        history_records.append(
            {
                "cohort": cohort,
                "scenario": scenario.name,
                "histories": int(len(after)),
                "observed_days_before_mean": float(before.mean()),
                "observed_days_after_mean": float(after.mean()),
                "observed_days_after_median": float(np.median(after)),
                "observed_days_after_p25": float(np.quantile(after, 0.25)),
                "observed_days_after_p75": float(np.quantile(after, 0.75)),
                "retained_fraction_mean": float(np.mean(after / before)),
            }
        )

    results: dict[str, dict[str, Any]] = {}
    for task_id, record in sorted(collected.items()):
        task = task_by_id[task_id]
        target = np.concatenate(record["target"])
        prediction = np.concatenate(record["prediction"])
        probabilities = (
            np.concatenate(record["probabilities"])
            if record["probabilities"]
            else None
        )
        metrics = _metrics(
            kind=task.kind,
            target=target,
            prediction=prediction,
            probabilities=probabilities,
        )
        primary_name = "mae" if task.primary_metric == "mae_weeks" else task.primary_metric
        results[task_id] = {
            "source": task.source,
            "domain": task.domain,
            "kind": task.kind,
            "primary_metric": primary_name,
            "primary_value": metrics.get(primary_name),
            "samples": int(len(target)),
            "participants": int(len(set(record["participant"]))),
            "metrics": metrics,
        }
    return results, history_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", type=_parse_checkpoint, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mask-seed", type=int, default=20260803)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--openmhc-data-dir", type=Path, default=Path("datasets/openmhc-xs"))
    parser.add_argument("--openmhc-native-cache", type=Path, default=Path("artifacts/embeddings/openmhc-xs/openmhc-lsm2"))
    parser.add_argument("--openmhc-adapted-cache", type=Path, default=Path("artifacts/embeddings/openmhc-xs/femmhc-stage1-v4"))
    parser.add_argument("--openmhc-history-days", type=int, default=7)
    parser.add_argument("--mcphases-dir", type=Path, default=Path("processed/mcphases"))
    parser.add_argument("--mcphases-embeddings", type=Path, default=Path("artifacts/embeddings/mcphases/dual-v4-seed42/femmhc-dual.npy"))
    parser.add_argument("--depress-dir", type=Path, default=Path("processed/depress_fitbit"))
    parser.add_argument("--depress-embeddings", type=Path, default=Path("artifacts/embeddings/depress-fitbit-affective-dynamics-step100.npz"))
    parser.add_argument("--inphrsym-dir", type=Path, default=Path("processed/inphrsym"))
    parser.add_argument("--inphrsym-embeddings", type=Path, default=Path("artifacts/embeddings/inphrsym-affective-dynamics-step100.npz"))
    parser.add_argument("--hrv-mental-dir", type=Path, default=Path("processed/wearable_hrv_mental_female"))
    parser.add_argument("--hrv-mental-embeddings", type=Path, default=Path("artifacts/embeddings/hrv-mental-female/femmhc-stage1-seed42.npz"))
    parser.add_argument("--pregnancy-dir", type=Path, default=Path("processed/pregnancy_ga_clock_official"))
    parser.add_argument("--pregnancy-embeddings", type=Path, default=Path("artifacts/embeddings/pregnancy-ga-official/progression-v4-best.npz"))
    parser.add_argument("--maximum-days", type=int, default=60)
    parser.add_argument("--minimum-history-days", type=int, default=3)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    checkpoints = dict(args.checkpoint)
    if len(checkpoints) != len(args.checkpoint):
        raise ValueError("checkpoint seeds must be unique")
    args.split = "validation"
    args.allow_test = False
    device = torch.device(args.device)
    datasets = _datasets(args)
    task_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    for seed, path in sorted(checkpoints.items()):
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        model = _load_model(artifact, device)
        task_metadata = artifact.get("instantiated_tasks") or artifact.get("active_tasks")
        if task_metadata:
            task_ids = {
                item["task_id"] if isinstance(item, dict) else str(item)
                for item in task_metadata
            }
            allowed_sources = {
                task.source for task in JOINT_TASKS if task.task_id in task_ids
            }
            datasets = {
                name: dataset
                for name, dataset in datasets.items()
                if name in allowed_sources
            }
        baseline: dict[str, dict[str, Any]] | None = None
        for scenario in SCENARIOS:
            print(json.dumps({"seed": seed, "scenario": scenario.name}), flush=True)
            results, history = evaluate_scenario(
                model,
                datasets,
                artifact.get("regression_target_statistics", {}),
                scenario,
                batch_size=args.batch_size,
                mask_seed=args.mask_seed,
                device=device,
            )
            history_rows.extend({"seed": seed, **row} for row in history)
            if scenario.name == "baseline":
                baseline = results
            if baseline is None:
                raise RuntimeError("baseline scenario must run first")
            for task_id, result in results.items():
                baseline_value = baseline[task_id]["primary_value"]
                candidate_value = result["primary_value"]
                metric = result["primary_metric"]
                if baseline_value is None or candidate_value is None:
                    oriented_delta = None
                    relative_change = None
                else:
                    sign = -1.0 if metric in LOWER_IS_BETTER else 1.0
                    oriented_delta = sign * (float(candidate_value) - float(baseline_value))
                    relative_change = 100.0 * oriented_delta / max(abs(float(baseline_value)), 1e-6)
                task_rows.append(
                    {
                        "seed": seed,
                        "scenario": scenario.name,
                        "task_id": task_id,
                        "source": result["source"],
                        "domain": result["domain"],
                        "kind": result["kind"],
                        "primary_metric": metric,
                        "baseline_value": baseline_value,
                        "scenario_value": candidate_value,
                        "oriented_delta": oriented_delta,
                        "relative_change_percent": relative_change,
                        "samples": result["samples"],
                        "participants": result["participants"],
                    }
                )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    task_frame = pd.DataFrame(task_rows)
    history_frame = pd.DataFrame(history_rows)
    summaries: list[dict[str, Any]] = []
    for scenario, group in task_frame.groupby("scenario", sort=False):
        finite = group[np.isfinite(group["oriented_delta"].astype(float))].copy()
        deltas = finite["oriented_delta"].to_numpy(dtype=float)
        relative = finite["relative_change_percent"].to_numpy(dtype=float)
        tolerance = 1e-12
        summaries.append(
            {
                "scenario": scenario,
                "task_seed_comparisons": int(len(finite)),
                "improved": int(np.count_nonzero(deltas > tolerance)),
                "ties": int(np.count_nonzero(np.abs(deltas) <= tolerance)),
                "worsened": int(np.count_nonzero(deltas < -tolerance)),
                "relative_change_mean_percent": float(np.mean(relative)),
                "relative_change_median_percent": float(np.median(relative)),
                "relative_change_p25_percent": float(np.quantile(relative, 0.25)),
                "relative_change_p75_percent": float(np.quantile(relative, 0.75)),
            }
        )
    summary = {
        "format_version": 1,
        "split": "validation",
        "test_used": False,
        "mask_seed": args.mask_seed,
        "seeds": sorted(checkpoints),
        "scenarios": summaries,
        "limitations": [
            "This evaluates missing daily histories, not device-specific sensor noise.",
            "Random and contiguous masks are deterministic and shared across model seeds.",
            "The formal test split is not used.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_frame.to_csv(args.output_dir / "per_task_primary_metrics.csv", index=False)
    history_frame.to_csv(args.output_dir / "history_realization.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.output_dir / "scenario_summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# FemMHC时序缺失鲁棒性",
        "",
        "> 仅使用现有验证数据进行日级遮蔽；测试集未使用。",
        "",
        "| 场景 | 任务×种子 | 改善/持平/下降 | 相对变化中位数 | 四分位区间 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['scenario']} | {row['task_seed_comparisons']} | "
            f"{row['improved']}/{row['ties']}/{row['worsened']} | "
            f"{row['relative_change_median_percent']:+.2f}% | "
            f"[{row['relative_change_p25_percent']:+.2f}%, {row['relative_change_p75_percent']:+.2f}%] |"
        )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output_dir": str(args.output_dir.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
