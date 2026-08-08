"""Cache seven-day pregnancy embeddings from OpenMHC or FemMHC."""

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
    PREGNANCY_GA_SENSOR_DESCRIPTORS,
    SensorBatch,
    build_femmhc_encoder_from_artifact,
)
from femmhc.data import PregnancyGAWindowDataset
from openmhc.models.lsm2.modules import LSM2Module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--femmhc-checkpoint", type=Path)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--activity-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    checkpoint_step = 0
    checkpoint_stage = "openmhc_initialization"
    artifact = None
    if args.femmhc_checkpoint is not None:
        artifact = torch.load(args.femmhc_checkpoint, map_location="cpu", weights_only=False)
        checkpoint_step = int(artifact.get("steps", 0))
        checkpoint_stage = str(artifact.get("stage", "unknown"))
    source = LSM2Module.load_from_checkpoint(str(args.checkpoint), map_location="cpu")
    model = build_femmhc_encoder_from_artifact(
        source.model,
        artifact,
        freeze_backbone=True,
    )
    del source
    model = model.to(device).eval()

    with (args.processed_dir / "index.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    schema = json.loads((args.processed_dir / "schema.json").read_text(encoding="utf-8"))
    measurements = len(rows)
    days = int(schema["days_per_measurement"])
    embeddings = np.full(
        (measurements, days, model.embed_dim),
        np.nan,
        dtype=np.float32,
    )
    day_present = np.zeros((measurements, days), dtype=bool)
    descriptors = (
        PREGNANCY_GA_SENSOR_DESCRIPTORS[:1]
        if args.activity_only
        else PREGNANCY_GA_SENSOR_DESCRIPTORS
    )

    started = time.perf_counter()
    encoded = 0
    with torch.inference_mode():
        for split in ("train", "validation", "test"):
            dataset = PregnancyGAWindowDataset(
                args.processed_dir,
                split=split,
                normalize=True,
                include_light=not args.activity_only,
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                pin_memory=device.type == "cuda",
            )
            for item in loader:
                values = item["sensor_values"]
                present = item["channel_present"]
                batch_size, n_days, channels, minutes = values.shape
                flat_values = values.reshape(batch_size * n_days, channels, minutes)
                flat_present = present.reshape(batch_size * n_days, channels)
                usable = flat_present.any(dim=-1)
                if not bool(usable.all()):
                    raise ValueError("pregnancy preprocessing produced an empty day")
                batch = SensorBatch(
                    flat_values.to(device, non_blocking=True),
                    descriptors,
                    flat_present.to(device, non_blocking=True),
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    output = model(batch)
                encoded_batch = output.pooled.float().cpu().numpy().reshape(
                    batch_size, n_days, model.embed_dim
                )
                indices = item["measurement_index"].numpy()
                embeddings[indices] = encoded_batch
                day_present[indices] = present.any(dim=-1).numpy()
                encoded += batch_size * n_days
            print(
                json.dumps(
                    {"split": split, "measurements": len(dataset), "encoded_days": encoded}
                ),
                flush=True,
            )
            dataset.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        embeddings=embeddings,
        day_present=day_present,
    )
    report = {
        "format_version": 1,
        "model": "FemMHC",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "femmhc_checkpoint": (
            str(args.femmhc_checkpoint.resolve()) if args.femmhc_checkpoint else None
        ),
        "checkpoint_stage": checkpoint_stage,
        "checkpoint_step": checkpoint_step,
        "activity_only": bool(args.activity_only),
        "initialization_seed": args.seed,
        "shape": list(embeddings.shape),
        "encoded_days": encoded,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
