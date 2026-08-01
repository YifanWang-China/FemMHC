"""Cache participant-aligned mcPHASES embeddings for frozen probing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from femmhc import FemMHCEncoder, MCPHASES_SENSOR_DESCRIPTORS, SensorBatch
from femmhc.data import McPhasesDataset
from openmhc.models.lsm2.modules import LSM2Module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--femmhc-checkpoint", type=Path)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    source = LSM2Module.load_from_checkpoint(str(args.checkpoint), map_location="cpu")
    model = FemMHCEncoder(source.model, freeze_backbone=True)
    del source
    checkpoint_step = 0
    checkpoint_stage = "openmhc_initialization"
    if args.femmhc_checkpoint is not None:
        artifact = torch.load(args.femmhc_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(artifact["student_state_dict"])
        checkpoint_step = int(artifact.get("steps", 0))
        checkpoint_stage = str(artifact.get("stage", "unknown"))
    model = model.to(device).eval()

    total_samples = len(np.load(args.processed_dir / "labels.npy", mmap_mode="r"))
    embeddings = np.full((total_samples, model.embed_dim), np.nan, dtype=np.float32)
    started = time.perf_counter()
    encoded = 0
    with torch.inference_mode():
        for split in ("train", "validation", "test"):
            dataset = McPhasesDataset(
                args.processed_dir,
                split=split,
                normalize=True,
                require_usable=True,
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
                    MCPHASES_SENSOR_DESCRIPTORS,
                    item["channel_present"].to(device, non_blocking=True),
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    output = model(batch)
                indices = item["sample_index"].numpy()
                embeddings[indices] = output.pooled.float().cpu().numpy()
                encoded += len(indices)
            print(json.dumps({"split": split, "encoded": len(dataset)}), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, embeddings)
    report = {
        "format_version": 1,
        "model": "FemMHC",
        "checkpoint": str(args.femmhc_checkpoint.resolve()) if args.femmhc_checkpoint else None,
        "checkpoint_stage": checkpoint_stage,
        "checkpoint_step": checkpoint_step,
        "shape": list(embeddings.shape),
        "encoded_samples": encoded,
        "excluded_unusable_samples": int(total_samples - encoded),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
