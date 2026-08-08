#!/usr/bin/env python
"""Cache native, adapted, and dual embeddings for affective daily cohorts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from femmhc import (
    AFFECTIVE_DAILY_SENSOR_DESCRIPTORS,
    SensorBatch,
    build_femmhc_encoder_from_artifact,
)
from femmhc.data import DEPRESSFitbitDailyDataset, InPHRSymDailyDataset
from openmhc.models.lsm2.modules import LSM2Module


DATASETS = {
    "inphrsym": InPHRSymDailyDataset,
    "depress_fitbit": DEPRESSFitbitDailyDataset,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=sorted(DATASETS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--femmhc-checkpoint", type=Path)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    stage = "openmhc_initialization"
    step = 0
    artifact = None
    if args.femmhc_checkpoint is not None:
        artifact = torch.load(args.femmhc_checkpoint, map_location="cpu", weights_only=False)
        stage = str(artifact.get("stage", "unknown"))
        step = int(artifact.get("steps", 0))
    source = LSM2Module.load_from_checkpoint(str(args.checkpoint), map_location="cpu")
    model = build_femmhc_encoder_from_artifact(
        source.model,
        artifact,
        freeze_backbone=True,
    )
    del source
    model = model.to(device).eval()

    with (args.processed_dir / "index.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    native = np.full((len(rows), model.embed_dim), np.nan, dtype=np.float32)
    adapted = np.full_like(native, np.nan)
    dual = np.full((len(rows), model.embed_dim * 2), np.nan, dtype=np.float32)
    native_available = np.zeros(len(rows), dtype=bool)
    dataset_class = DATASETS[args.cohort]
    started = time.perf_counter()
    with torch.inference_mode():
        for split in ("train", "validation", "test"):
            dataset = dataset_class(args.processed_dir, split=split, normalize=True)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=device.type == "cuda",
            )
            for item in loader:
                batch = SensorBatch(
                    item["sensor_values"].to(device, non_blocking=True),
                    AFFECTIVE_DAILY_SENSOR_DESCRIPTORS,
                    item["channel_present"].to(device, non_blocking=True),
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    encoded = model.forward_dual(batch)
                indices = item["day_index"].numpy()
                native[indices] = encoded.native_pooled.float().cpu().numpy()
                adapted[indices] = encoded.adapted.pooled.float().cpu().numpy()
                dual[indices] = encoded.pooled.float().cpu().numpy()
                native_available[indices] = encoded.native_available.cpu().numpy()
            dataset.close()
    if not np.isfinite(adapted).all() or not np.isfinite(dual).all():
        raise ValueError("embedding cache contains unfilled participant-days")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        native_embeddings=native,
        adapted_embeddings=adapted,
        dual_embeddings=dual,
        native_available=native_available,
    )
    report = {
        "format_version": 1,
        "cohort": args.cohort,
        "model": "FemMHC",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "femmhc_checkpoint": (
            str(args.femmhc_checkpoint.resolve()) if args.femmhc_checkpoint else None
        ),
        "checkpoint_stage": stage,
        "checkpoint_step": step,
        "initialization_seed": args.seed,
        "device": str(device),
        "daily_samples": len(rows),
        "native_available": int(native_available.sum()),
        "native_shape": list(native.shape),
        "adapted_shape": list(adapted.shape),
        "dual_shape": list(dual.shape),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
