"""Paired participant-bootstrap comparison for pregnancy representations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)


def _features(embeddings: np.ndarray, day_present: np.ndarray) -> np.ndarray:
    masked = np.where(day_present[:, :, None], embeddings, np.nan)
    mean = np.nanmean(masked, axis=1)
    std = np.nanstd(masked, axis=1)
    first = np.take_along_axis(
        masked,
        np.nanargmax(day_present, axis=1)[:, None, None],
        axis=1,
    ).squeeze(1)
    return np.concatenate([mean, std, first], axis=1)


def _fit_probe(
    features: np.ndarray,
    target: np.ndarray,
    masks: dict[str, np.ndarray],
) -> tuple[np.ndarray, float, float]:
    train_scaler = StandardScaler().fit(features[masks["train"]])
    train_features = train_scaler.transform(features)
    candidates: list[tuple[float, float]] = []
    for alpha in RIDGE_ALPHAS:
        probe = Ridge(alpha=alpha, solver="lsqr", tol=1e-6).fit(
            train_features[masks["train"]], target[masks["train"]]
        )
        validation_prediction = probe.predict(train_features[masks["validation"]])
        candidates.append(
            (
                float(
                    mean_absolute_error(
                        target[masks["validation"]], validation_prediction
                    )
                ),
                alpha,
            )
        )
    validation_mae, selected_alpha = min(candidates)

    development = masks["train"] | masks["validation"]
    final_scaler = StandardScaler().fit(features[development])
    final_features = final_scaler.transform(features)
    probe = Ridge(alpha=selected_alpha, solver="lsqr", tol=1e-6).fit(
        final_features[development], target[development]
    )
    return (
        probe.predict(final_features[masks["test"]]),
        float(selected_alpha),
        validation_mae,
    )


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae_weeks": float(mean_absolute_error(target, prediction)),
        "rmse_weeks": float(mean_squared_error(target, prediction) ** 0.5),
        "r2": float(r2_score(target, prediction)),
        "spearman": float(spearmanr(target, prediction).statistic),
    }


def _paired_bootstrap(
    target: np.ndarray,
    baseline_prediction: np.ndarray,
    femmhc_prediction: np.ndarray,
    participants: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    unique = np.unique(participants)
    by_participant = {item: np.flatnonzero(participants == item) for item in unique}
    mae_improvements = np.empty(draws, dtype=np.float64)
    spearman_improvements = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([by_participant[item] for item in sampled])
        draw_target = target[indices]
        baseline = baseline_prediction[indices]
        femmhc = femmhc_prediction[indices]
        mae_improvements[draw] = mean_absolute_error(
            draw_target, baseline
        ) - mean_absolute_error(draw_target, femmhc)
        spearman_improvements[draw] = float(
            spearmanr(draw_target, femmhc).statistic
            - spearmanr(draw_target, baseline).statistic
        )
    return {
        "draws": draws,
        "mae_improvement_ci_low_weeks": float(np.quantile(mae_improvements, 0.025)),
        "mae_improvement_ci_high_weeks": float(np.quantile(mae_improvements, 0.975)),
        "mae_improvement_bootstrap_support": float(np.mean(mae_improvements > 0.0)),
        "spearman_improvement_ci_low": float(
            np.quantile(spearman_improvements, 0.025)
        ),
        "spearman_improvement_ci_high": float(
            np.quantile(spearman_improvements, 0.975)
        ),
        "spearman_improvement_bootstrap_support": float(
            np.mean(spearman_improvements > 0.0)
        ),
    }


def _load_features(path: Path) -> np.ndarray:
    cached = np.load(path)
    return _features(cached["embeddings"], cached["day_present"].astype(bool))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-embeddings", type=Path, required=True)
    parser.add_argument("--femmhc-embeddings", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    baseline_features = _load_features(args.baseline_embeddings)
    femmhc_features = _load_features(args.femmhc_embeddings)
    if baseline_features.shape != femmhc_features.shape:
        raise ValueError("baseline and FemMHC feature matrices must have identical shape")

    with (args.processed_dir / "index.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    target = np.asarray([float(row["gestational_age_weeks"]) for row in rows])
    participants = np.asarray([row["participant_id"] for row in rows])
    splits = json.loads(
        (args.processed_dir / "participant_splits.json").read_text(encoding="utf-8")
    )
    masks = {
        name: np.isin(participants, np.asarray(ids).astype(str))
        for name, ids in splits.items()
    }

    baseline_prediction, baseline_alpha, baseline_validation_mae = _fit_probe(
        baseline_features, target, masks
    )
    femmhc_prediction, femmhc_alpha, femmhc_validation_mae = _fit_probe(
        femmhc_features, target, masks
    )
    test_target = target[masks["test"]]
    test_participants = participants[masks["test"]]
    baseline_metrics = _metrics(test_target, baseline_prediction)
    femmhc_metrics = _metrics(test_target, femmhc_prediction)
    mae_gain = baseline_metrics["mae_weeks"] - femmhc_metrics["mae_weeks"]
    spearman_gain = femmhc_metrics["spearman"] - baseline_metrics["spearman"]

    report = {
        "format_version": 1,
        "initialization_seed": args.seed,
        "cohort": {
            "measurements": len(rows),
            "participants": len(np.unique(participants)),
            "train_participants": len(splits["train"]),
            "validation_participants": len(splits["validation"]),
            "test_participants": len(splits["test"]),
            "split_unit": "participant_id",
        },
        "probe_protocol": "validation-selected Ridge; refit on train+validation",
        "models": {
            "OpenMHC": {
                **baseline_metrics,
                "selected_ridge_alpha": baseline_alpha,
                "validation_mae_weeks": baseline_validation_mae,
            },
            "FemMHC": {
                **femmhc_metrics,
                "selected_ridge_alpha": femmhc_alpha,
                "validation_mae_weeks": femmhc_validation_mae,
            },
        },
        "improvement": {
            "mae_absolute_weeks": mae_gain,
            "mae_relative_percent": 100.0
            * mae_gain
            / baseline_metrics["mae_weeks"],
            "spearman_absolute": spearman_gain,
            "spearman_relative_percent": 100.0
            * spearman_gain
            / baseline_metrics["spearman"],
            **_paired_bootstrap(
                test_target,
                baseline_prediction,
                femmhc_prediction,
                test_participants,
                draws=args.bootstrap_draws,
                seed=args.seed,
            ),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    markdown = f"""# Pregnancy gestational-age representation benchmark

Participant-level split: {len(splits['train'])}/{len(splits['validation'])}/{len(splits['test'])} train/validation/test participants. Ridge hyperparameters are selected on validation participants and refit on train+validation participants.

| Representation | MAE weeks ↓ | RMSE weeks ↓ | R² ↑ | Spearman ↑ |
|---|---:|---:|---:|---:|
| OpenMHC | {baseline_metrics['mae_weeks']:.4f} | {baseline_metrics['rmse_weeks']:.4f} | {baseline_metrics['r2']:.4f} | {baseline_metrics['spearman']:.4f} |
| FemMHC | **{femmhc_metrics['mae_weeks']:.4f}** | **{femmhc_metrics['rmse_weeks']:.4f}** | **{femmhc_metrics['r2']:.4f}** | **{femmhc_metrics['spearman']:.4f}** |

- MAE relative improvement: {report['improvement']['mae_relative_percent']:.2f}%.
- Spearman relative improvement: {report['improvement']['spearman_relative_percent']:.2f}%.
- Paired participant-bootstrap MAE improvement 95% CI: [{report['improvement']['mae_improvement_ci_low_weeks']:.4f}, {report['improvement']['mae_improvement_ci_high_weeks']:.4f}] weeks.
- Bootstrap support for positive MAE improvement: {100.0 * report['improvement']['mae_improvement_bootstrap_support']:.1f}%.
"""
    (args.output_dir / "README.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
