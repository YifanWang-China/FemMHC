"""End-to-end FemMHC smoke test on real mcPHASES days and OpenMHC weights."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from femmhc import (
    FemMHCEncoder,
    MCPHASES_SENSOR_DESCRIPTORS,
    PatchReconstructionHead,
    SensorBatch,
    TemporalOrderHead,
    combine_losses,
    drop_sensor_channels,
    mask_sensor_patches,
    masked_patch_reconstruction_loss,
    preservation_loss,
    sensor_set_consistency_loss,
    temporal_order_loss,
)
from femmhc.data import McPhasesTemporalPairDataset
from openmhc.models.lsm2.modules import LSM2Module


def parameter_counts(module: torch.nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in module.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in module.parameters() if parameter.requires_grad
        ),
    }


def batch_from_item(item: dict[str, object], device: torch.device) -> SensorBatch:
    return SensorBatch(
        item["sensor_values"].unsqueeze(0).to(device),
        MCPHASES_SENSOR_DESCRIPTORS,
        item["channel_present"].unsqueeze(0).to(device),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    started = time.perf_counter()
    module = LSM2Module.load_from_checkpoint(str(args.checkpoint), map_location="cpu")
    encoder = FemMHCEncoder(module.model, freeze_backbone=True).to(device).eval()
    teacher = copy.deepcopy(encoder).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    del module

    reconstruction_head = PatchReconstructionHead(encoder.embed_dim, encoder.patch_size).to(device).eval()
    order_head = TemporalOrderHead(encoder.embed_dim).to(device).eval()
    dataset = McPhasesTemporalPairDataset(args.processed_dir, split="train")
    item = dataset[0]
    first = batch_from_item(item["first"], device)
    second = batch_from_item(item["second"], device)
    masked, artificial_mask = mask_sensor_patches(
        first,
        patch_size=encoder.patch_size,
        mask_probability=0.15,
    )
    subset = drop_sensor_channels(
        first,
        drop_probability=0.35,
        patch_size=encoder.patch_size,
        min_observed_fraction=encoder.min_observed_fraction,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        first_output = encoder(first)
        second_output = encoder(second)
        masked_output = encoder(masked)
        subset_output = encoder(subset)
        teacher_output = teacher(first)
        prediction = reconstruction_head(masked_output.latent)
        reconstruction = masked_patch_reconstruction_loss(
            prediction,
            first.values,
            artificial_mask,
            patch_size=encoder.patch_size,
        )
        consistency = sensor_set_consistency_loss(
            first_output.pooled,
            subset_output.pooled,
        )
        preservation = preservation_loss(first_output.pooled, teacher_output.pooled)
        trajectory = temporal_order_loss(
            order_head,
            first_output.pooled,
            second_output.pooled,
            item["second_is_later"].reshape(1).to(device),
        )
        losses = combine_losses(
            reconstruction=reconstruction,
            sensor_consistency=consistency,
            preservation=preservation,
            trajectory=trajectory,
        )

    report = {
        "status": "passed" if bool(torch.isfinite(losses.total)) else "failed",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "checkpoint": str(args.checkpoint.resolve()),
        "processed_dir": str(args.processed_dir.resolve()),
        "real_pair": {
            "participant_id": item["first"]["participant_id"],
            "study_interval": item["first"]["study_interval"],
            "first_day": item["first"]["day_in_study"],
            "second_day": item["second"]["day_in_study"],
        },
        "input_shape": list(first.values.shape),
        "latent_shape": list(first_output.latent.shape),
        "pooled_shape": list(first_output.pooled.shape),
        "artificially_masked_patches": int(artificial_mask.sum()),
        "losses": {
            name: float(value.float().cpu())
            for name, value in asdict(losses).items()
        },
        "adapter_weights": [float(value) for value in first_output.adapter_weights[0].float().cpu()],
        "encoder_parameters": parameter_counts(encoder),
        "peak_gpu_memory_gb": (
            round(torch.cuda.max_memory_allocated(device) / 1024**3, 3)
            if device.type == "cuda"
            else None
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
