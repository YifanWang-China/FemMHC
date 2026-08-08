#!/usr/bin/env python
"""Cache mcPHASES day embeddings with causal history-gated internal adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from femmhc import FemMHCEncoder, MCPHASES_SENSOR_DESCRIPTORS, SensorBatch
from femmhc.data import McPhasesHistoryAdapterDataset
from openmhc.models.lsm2.modules import LSM2Module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--femmhc-checkpoint", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument(
        "--history-embeddings",
        type=Path,
        required=True,
        help="Frozen static day embeddings used only for days strictly preceding t.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--representation", choices=("adapted", "dual"), default="dual")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    device = torch.device(args.device)
    artifact = torch.load(args.femmhc_checkpoint, map_location="cpu", weights_only=False)
    if not artifact.get("history_conditioned_internal_adapters", False):
        raise ValueError("the supplied FemMHC checkpoint is not history-conditioned")
    source = LSM2Module.load_from_checkpoint(str(args.checkpoint), map_location="cpu")
    model = FemMHCEncoder(
        source.model,
        freeze_backbone=True,
        internal_adapter_rank=int(artifact["internal_adapter_rank"]),
        internal_adapter_layers=int(artifact["internal_adapter_layers"]),
        history_conditioned_internal_adapters=True,
        history_context_dim=int(artifact["history_context_dim"]),
        history_maximum_days=int(artifact["history_days"]),
        history_cycle_modes=int(artifact["history_cycle_modes"]),
    )
    model.load_state_dict(artifact["student_state_dict"])
    model = model.to(device).eval()
    del source

    total_samples = len(np.load(args.processed_dir / "labels.npy", mmap_mode="r"))
    output_dim = model.embed_dim * (2 if args.representation == "dual" else 1)
    embeddings = np.full((total_samples, output_dim), np.nan, dtype=np.float32)
    encoded = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for split in ("train", "validation", "test"):
            dataset = McPhasesHistoryAdapterDataset(
                args.processed_dir,
                args.history_embeddings,
                split=split,
                history_days=int(artifact["history_days"]),
                minimum_history_days=0,
                require_target=False,
            )
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
                    MCPHASES_SENSOR_DESCRIPTORS,
                    item["channel_present"].to(device, non_blocking=True),
                )
                history_embeddings = item["history_embeddings"].to(device, non_blocking=True)
                history_present = item["history_present"].to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    output = (
                        model.forward_dual(
                            batch,
                            history_embeddings=history_embeddings,
                            history_present=history_present,
                        )
                        if args.representation == "dual"
                        else model(
                            batch,
                            history_embeddings=history_embeddings,
                            history_present=history_present,
                        )
                    )
                indices = item["sample_index"].numpy()
                embeddings[indices] = output.pooled.float().cpu().numpy()
                encoded += len(indices)
            print(json.dumps({"split": split, "encoded": len(dataset)}), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, embeddings)
    report = {
        "format_version": 1,
        "model": "FemMHC",
        "checkpoint": str(args.femmhc_checkpoint.resolve()),
        "history_embeddings": str(args.history_embeddings.resolve()),
        "history_days": int(artifact["history_days"]),
        "history_excludes_current_day": True,
        "representation": args.representation,
        "shape": list(embeddings.shape),
        "encoded_samples": encoded,
        "excluded_unusable_samples": int(total_samples - encoded),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
