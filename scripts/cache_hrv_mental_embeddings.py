"""Cache female Wearable HRV and Sleep daily embeddings."""

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
    SensorBatch,
    WEARABLE_HRV_MENTAL_SENSOR_DESCRIPTORS,
    build_femmhc_encoder_from_artifact,
)
from femmhc.data import WearableHRVMentalDailyDataset
from openmhc.models.lsm2.modules import LSM2Module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--femmhc-checkpoint", type=Path)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
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
    embeddings = np.full((len(rows), model.embed_dim), np.nan, dtype=np.float32)
    started = time.perf_counter()
    with torch.inference_mode():
        for split in ("train", "validation", "test"):
            dataset = WearableHRVMentalDailyDataset(
                args.processed_dir, split=split, normalize=True
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                pin_memory=device.type == "cuda",
            )
            for item in loader:
                batch = SensorBatch(
                    item["sensor_values"].to(device, non_blocking=True),
                    WEARABLE_HRV_MENTAL_SENSOR_DESCRIPTORS,
                    item["channel_present"].to(device, non_blocking=True),
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    output = model(batch)
                embeddings[item["day_index"].numpy()] = output.pooled.float().cpu().numpy()
            dataset.close()
    if not np.isfinite(embeddings).all():
        raise ValueError("embedding cache contains unfilled participant-days")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, embeddings=embeddings)
    report = {
        "format_version": 1,
        "model": "FemMHC",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "femmhc_checkpoint": (
            str(args.femmhc_checkpoint.resolve()) if args.femmhc_checkpoint else None
        ),
        "checkpoint_stage": stage,
        "checkpoint_step": step,
        "initialization_seed": args.seed,
        "shape": list(embeddings.shape),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
