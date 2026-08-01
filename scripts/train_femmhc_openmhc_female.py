"""Stage-1 FemMHC specialization on the female OpenMHC cohort."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from femmhc import (
    FemMHCEncoder,
    OPENMHC_SENSOR_DESCRIPTORS,
    PatchReconstructionHead,
    SensorBatch,
    drop_sensor_channels,
    mask_sensor_patches,
    masked_patch_reconstruction_loss,
    preservation_loss,
    sensor_set_consistency_loss,
)
from femmhc.data import OpenMHCFemaleDataset
from femmhc.checkpointing import (
    capture_rng_state,
    restore_rng_state,
    save_training_checkpoint,
)
from openmhc.models.lsm2.modules import LSM2Module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--openmhc-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-channels", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.max_steps <= 0 or args.max_channels <= 0 or args.save_every <= 0:
        raise ValueError("steps and channel count must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    source = LSM2Module.load_from_checkpoint(str(args.checkpoint), map_location="cpu")
    student = FemMHCEncoder(source.model, freeze_backbone=True).to(device).train()
    teacher = copy.deepcopy(student).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    del source
    reconstruction_head = PatchReconstructionHead(student.embed_dim, student.patch_size).to(device).train()
    trainable = [
        parameter
        for module in (student, reconstruction_head)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    dataset = OpenMHCFemaleDataset(args.openmhc_root, split="train")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=device.type == "cuda")

    history: list[dict[str, float | int]] = []
    step = 0
    elapsed_offset = 0.0
    if args.resume and args.output.is_file():
        checkpoint = torch.load(args.output, map_location="cpu", weights_only=False)
        if checkpoint.get("stage") != "openmhc_female_specialization":
            raise ValueError(f"Not an OpenMHC-female checkpoint: {args.output}")
        student.load_state_dict(checkpoint["student_state_dict"])
        reconstruction_head.load_state_dict(checkpoint["reconstruction_head_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        step = int(checkpoint["steps"])
        history = list(checkpoint.get("history", []))
        elapsed_offset = float(checkpoint.get("elapsed_seconds", 0.0))
        restore_rng_state(checkpoint)
        print(json.dumps({"event": "resumed", "step": step}), flush=True)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    def save(status: str) -> None:
        save_training_checkpoint(
            args.output,
            {
                "format_version": 1,
                "model": "FemMHC",
                "stage": "openmhc_female_specialization",
                "status": status,
                "source_checkpoint": str(args.checkpoint.resolve()),
                "seed": args.seed,
                "steps": step,
                "max_steps": args.max_steps,
                "history": history,
                "elapsed_seconds": elapsed_offset + time.perf_counter() - started,
                "peak_gpu_memory_gb": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None,
                "student_state_dict": student.state_dict(),
                "reconstruction_head_state_dict": reconstruction_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                **capture_rng_state(),
            },
        )

    while step < args.max_steps:
        for item in loader:
            all_values = item["sensor_values"].to(device, non_blocking=True)
            usable = torch.isfinite(
                all_values.reshape(all_values.shape[0], all_values.shape[1], -1, student.patch_size)
            ).float().mean(dim=-1).ge(student.min_observed_fraction).any(dim=-1)
            if not bool(usable.any(dim=1).all()):
                continue
            required = {int(row.nonzero(as_tuple=False)[0]) for row in usable if bool(row.any())}
            candidates = [index for index in range(all_values.shape[1]) if index not in required]
            random.shuffle(candidates)
            selected = sorted(required | set(candidates[: max(0, args.max_channels - len(required))]))
            if not selected:
                continue
            values = all_values[:, selected]
            present = item["channel_present"].to(device, non_blocking=True)[:, selected]
            descriptors = tuple(OPENMHC_SENSOR_DESCRIPTORS[index] for index in selected)
            full = SensorBatch(values, descriptors, present)
            masked, artificial_mask = mask_sensor_patches(
                full,
                patch_size=student.patch_size,
                mask_probability=0.15,
            )
            subset = drop_sensor_channels(
                full,
                drop_probability=0.35,
                patch_size=student.patch_size,
                min_observed_fraction=student.min_observed_fraction,
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                full_output = student(full)
                subset_output = student(subset)
                masked_output = student(masked)
                with torch.no_grad():
                    teacher_output = teacher(full)
                reconstruction = masked_patch_reconstruction_loss(
                    reconstruction_head(masked_output.latent),
                    full.values,
                    artificial_mask,
                    patch_size=student.patch_size,
                )
                consistency = sensor_set_consistency_loss(full_output.pooled, subset_output.pooled)
                preservation = preservation_loss(full_output.pooled, teacher_output.pooled)
                total = reconstruction + consistency + preservation
            if not bool(torch.isfinite(total)):
                raise FloatingPointError(f"non-finite loss at step {step}")
            total.backward()
            gradient_norm = clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            step += 1
            record = {
                "step": step,
                "total": float(total.detach().float().cpu()),
                "reconstruction": float(reconstruction.detach().float().cpu()),
                "sensor_consistency": float(consistency.detach().float().cpu()),
                "preservation": float(preservation.detach().float().cpu()),
                "gradient_norm": float(gradient_norm.detach().float().cpu()),
                "channels": len(selected),
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            if step % args.save_every == 0:
                save("running")
                print(json.dumps({"event": "checkpoint", "step": step}), flush=True)
            if step >= args.max_steps:
                break

    save("complete")


if __name__ == "__main__":
    main()
