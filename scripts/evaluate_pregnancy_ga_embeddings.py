"""Participant-safe frozen-probe evaluation for pregnancy gestational age."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    correlation_value = (
        np.nan
        if np.std(prediction) < 1e-12
        else float(spearmanr(target, prediction).statistic)
    )
    correlation = float(correlation_value) if np.isfinite(correlation_value) else None
    return {
        "mae_weeks": float(mean_absolute_error(target, prediction)),
        "rmse_weeks": float(mean_squared_error(target, prediction) ** 0.5),
        "r2": float(r2_score(target, prediction)),
        "spearman": correlation,
    }


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


def _participant_bootstrap(
    target: np.ndarray,
    prediction: np.ndarray,
    participants: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    unique = np.unique(participants)
    values = np.empty(draws, dtype=np.float64)
    by_participant = {item: np.flatnonzero(participants == item) for item in unique}
    for draw in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([by_participant[item] for item in sampled])
        values[draw] = mean_absolute_error(target[indices], prediction[indices])
    return {
        "mae_ci_low": float(np.quantile(values, 0.025)),
        "mae_ci_high": float(np.quantile(values, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--femmhc-checkpoint", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cached = np.load(args.embeddings)
    embeddings = cached["embeddings"]
    day_present = cached["day_present"].astype(bool)
    features = _features(embeddings, day_present)
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
    if not all(mask.any() for mask in masks.values()):
        raise ValueError("every participant split must contain measurements")

    scaler = StandardScaler().fit(features[masks["train"]])
    transformed = scaler.transform(features)
    alphas = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)
    validation = []
    for alpha in alphas:
        model = Ridge(alpha=alpha, solver="lsqr", tol=1e-6).fit(
            transformed[masks["train"]], target[masks["train"]]
        )
        prediction = model.predict(transformed[masks["validation"]])
        validation.append((mean_absolute_error(target[masks["validation"]], prediction), alpha))
    validation_mae, selected_alpha = min(validation)

    development = masks["train"] | masks["validation"]
    final_scaler = StandardScaler().fit(features[development])
    final_features = final_scaler.transform(features)
    model = Ridge(alpha=selected_alpha, solver="lsqr", tol=1e-6).fit(
        final_features[development], target[development]
    )
    test_prediction = model.predict(final_features[masks["test"]])
    test_target = target[masks["test"]]
    test_participants = participants[masks["test"]]

    median_prediction = np.full_like(test_target, np.median(target[development]))
    result_rows = []
    evaluated_predictions: list[tuple[str, np.ndarray]] = [
        ("training_median", median_prediction),
        ("frozen_embedding_ridge", test_prediction),
    ]
    if args.femmhc_checkpoint is not None:
        import torch

        from femmhc import PregnancyGAHead

        checkpoint = torch.load(
            args.femmhc_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if "gestational_age_head_state_dict" not in checkpoint:
            raise ValueError("checkpoint has no trained gestational-age head")
        head = PregnancyGAHead(embeddings.shape[-1]).eval()
        head.load_state_dict(checkpoint["gestational_age_head_state_dict"])
        test_embeddings = torch.from_numpy(embeddings[masks["test"]]).float()
        test_days = torch.from_numpy(day_present[masks["test"]]).bool()
        with torch.inference_mode():
            normalized_prediction = head(
                test_embeddings,
                day_present=test_days,
            ).prediction.numpy()
        direct_prediction = (
            normalized_prediction * float(checkpoint["gestational_age_target_std"])
            + float(checkpoint["gestational_age_target_mean"])
        )
        evaluated_predictions.append(("trained_progression_head", direct_prediction))

    for name, prediction in evaluated_predictions:
        result = {"model": name, **metrics(test_target, prediction)}
        result.update(
            _participant_bootstrap(
                test_target,
                prediction,
                test_participants,
                draws=args.bootstrap_draws,
                seed=args.seed,
            )
        )
        result_rows.append(result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result_rows).to_csv(args.output_dir / "metrics.csv", index=False)
    report = {
        "format_version": 1,
        "split_unit": "participant_id",
        "train_participants": len(splits["train"]),
        "validation_participants": len(splits["validation"]),
        "test_participants": len(splits["test"]),
        "selected_ridge_alpha": selected_alpha,
        "selection_validation_mae_weeks": float(validation_mae),
        "bootstrap_draws": args.bootstrap_draws,
        "results": result_rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
