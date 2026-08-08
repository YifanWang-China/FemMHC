#!/usr/bin/env python
"""Continual pretraining on female longitudinal affective-health cohorts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import ConcatDataset, DataLoader, Subset

from femmhc import (
    AFFECTIVE_DAILY_SENSOR_DESCRIPTORS,
    FemMHCEncoder,
    PatchReconstructionHead,
    PhysiologyChangeHead,
    SensorBatch,
    TemporalOrderHead,
    adjacent_day_contrastive_loss,
    drop_sensor_channels,
    mask_sensor_patches,
    masked_patch_reconstruction_loss,
    preservation_loss,
    physiology_change_loss,
    sensor_set_consistency_loss,
    temporal_order_loss,
)
from femmhc.checkpointing import (
    capture_rng_state,
    restore_rng_state,
    save_training_checkpoint,
)
from femmhc.data import (
    AdjacentDayPairDataset,
    DEPRESSFitbitDailyDataset,
    InPHRSymDailyDataset,
)
from openmhc.models.lsm2.modules import LSM2Module


def _sensor_batch(item: dict[str, Any], device: torch.device) -> SensorBatch:
    return SensorBatch(
        item["sensor_values"].to(device, non_blocking=True),
        AFFECTIVE_DAILY_SENSOR_DESCRIPTORS,
        item["channel_present"].to(device, non_blocking=True),
    )


def _build_pairs(
    inphrsym_dir: Path,
    depress_dir: Path,
    *,
    split: str,
) -> tuple[AdjacentDayPairDataset, AdjacentDayPairDataset]:
    return (
        AdjacentDayPairDataset(
            InPHRSymDailyDataset(inphrsym_dir, split=split, normalize=True)
        ),
        AdjacentDayPairDataset(
            DEPRESSFitbitDailyDataset(depress_dir, split=split, normalize=True)
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initial-femmhc-checkpoint", type=Path)
    parser.add_argument("--inphrsym-dir", type=Path, required=True)
    parser.add_argument("--depress-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    parser.add_argument("--preservation-weight", type=float, default=10.0)
    parser.add_argument("--temporal-order-weight", type=float, default=0.5)
    parser.add_argument("--physiology-change-weight", type=float, default=0.5)
    parser.add_argument("--contrastive-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--validation-pairs", type=int, default=256)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(
        args.max_steps,
        args.batch_size,
        args.validation_pairs,
        args.validate_every,
        args.save_every,
        args.log_every,
    ) <= 0:
        raise ValueError("steps, batch size, validation size, and intervals must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    source = LSM2Module.load_from_checkpoint(str(args.checkpoint), map_location="cpu")
    student = FemMHCEncoder(source.model, freeze_backbone=True)
    del source
    initial_stage = "openmhc_initialization"
    if args.initial_femmhc_checkpoint is not None:
        artifact = torch.load(
            args.initial_femmhc_checkpoint, map_location="cpu", weights_only=False
        )
        student.load_state_dict(artifact["student_state_dict"])
        initial_stage = str(artifact.get("stage", "unknown"))
    student = student.to(device).train()
    reconstruction_head = PatchReconstructionHead(
        student.embed_dim, student.patch_size
    ).to(device).train()
    temporal_head = TemporalOrderHead(student.embed_dim).to(device).train()
    physiology_change_head = PhysiologyChangeHead(
        student.embed_dim,
        channels=len(AFFECTIVE_DAILY_SENSOR_DESCRIPTORS),
    ).to(device).train()
    contrastive_projection = nn.Sequential(
        nn.LayerNorm(student.embed_dim),
        nn.Linear(student.embed_dim, 128),
    ).to(device).train()
    trainable = [
        parameter
        for module in (
            student,
            reconstruction_head,
            temporal_head,
            physiology_change_head,
            contrastive_projection,
        )
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)

    train_pairs = _build_pairs(
        args.inphrsym_dir, args.depress_dir, split="train"
    )
    validation_pairs = _build_pairs(
        args.inphrsym_dir, args.depress_dir, split="validation"
    )
    train_dataset = ConcatDataset(train_pairs)
    validation_dataset = ConcatDataset(validation_pairs)
    if len(train_dataset) == 0 or len(validation_dataset) == 0:
        raise ValueError("both training and validation need adjacent-day pairs")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    validation_generator = torch.Generator().manual_seed(args.seed)
    validation_indices = torch.randperm(
        len(validation_dataset), generator=validation_generator
    )[: min(args.validation_pairs, len(validation_dataset))].tolist()
    validation_loader = DataLoader(
        Subset(validation_dataset, validation_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    history: list[dict[str, float | int | str]] = []
    step = 0
    best_score = float("inf")
    best_step = 0
    elapsed_offset = 0.0
    if args.resume and args.output.is_file():
        artifact = torch.load(args.output, map_location="cpu", weights_only=False)
        if artifact.get("stage") != "affective_female_continual_pretraining_v2":
            raise ValueError(f"not an affective FemMHC checkpoint: {args.output}")
        student.load_state_dict(artifact["student_state_dict"])
        reconstruction_head.load_state_dict(artifact["reconstruction_head_state_dict"])
        temporal_head.load_state_dict(artifact["temporal_head_state_dict"])
        physiology_change_head.load_state_dict(
            artifact["physiology_change_head_state_dict"]
        )
        contrastive_projection.load_state_dict(
            artifact["contrastive_projection_state_dict"]
        )
        optimizer.load_state_dict(artifact["optimizer_state_dict"])
        step = int(artifact["steps"])
        best_score = float(artifact.get("best_validation_score", float("inf")))
        best_step = int(artifact.get("best_step", 0))
        history = list(artifact.get("history", []))
        elapsed_offset = float(artifact.get("elapsed_seconds", 0.0))
        restore_rng_state(artifact)
        print(json.dumps({"event": "resumed", "step": step}), flush=True)

    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def save(status: str, path: Path | None = None) -> None:
        save_training_checkpoint(
            path or args.output,
            {
                "format_version": 1,
                "model": "FemMHC",
                "stage": "affective_female_continual_pretraining_v2",
                "status": status,
                "source_checkpoint": str(args.checkpoint.resolve()),
                "initial_femmhc_checkpoint": (
                    str(args.initial_femmhc_checkpoint.resolve())
                    if args.initial_femmhc_checkpoint
                    else None
                ),
                "initial_stage": initial_stage,
                "seed": args.seed,
                "steps": step,
                "max_steps": args.max_steps,
                "training_pairs": len(train_dataset),
                "validation_pairs": len(validation_indices),
                "loss_weights": {
                    "reconstruction": args.reconstruction_weight,
                    "sensor_consistency": args.consistency_weight,
                    "native_preservation": args.preservation_weight,
                    "temporal_order": args.temporal_order_weight,
                    "physiology_change": args.physiology_change_weight,
                    "adjacent_day_contrastive": args.contrastive_weight,
                },
                "contrastive_temperature": args.contrastive_temperature,
                "best_validation_score": best_score,
                "best_step": best_step,
                "history": history,
                "elapsed_seconds": elapsed_offset + time.perf_counter() - started,
                "peak_gpu_memory_gb": (
                    torch.cuda.max_memory_allocated(device) / 1024**3
                    if device.type == "cuda"
                    else None
                ),
                "student_state_dict": student.state_dict(),
                "reconstruction_head_state_dict": reconstruction_head.state_dict(),
                "temporal_head_state_dict": temporal_head.state_dict(),
                "physiology_change_head_state_dict": physiology_change_head.state_dict(),
                "contrastive_projection_state_dict": contrastive_projection.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                **capture_rng_state(),
            },
        )

    def validation_score() -> dict[str, float]:
        student.eval()
        temporal_head.eval()
        physiology_change_head.eval()
        contrastive_projection.eval()
        totals = {
            "preservation": 0.0,
            "temporal_order": 0.0,
            "physiology_change": 0.0,
            "contrastive": 0.0,
        }
        batches = 0
        with torch.inference_mode():
            for item in validation_loader:
                earlier = _sensor_batch(item["earlier"], device)
                later = _sensor_batch(item["later"], device)
                first = student(earlier).pooled
                second = student(later).pooled
                native = student.forward_native(earlier).pooled
                preservation = preservation_loss(first, native)
                order = temporal_order_loss(
                    temporal_head,
                    first,
                    second,
                    torch.ones(first.shape[0], device=device),
                )
                change = physiology_change_loss(
                    physiology_change_head,
                    first,
                    second,
                    earlier.values,
                    later.values,
                )
                contrastive = adjacent_day_contrastive_loss(
                    contrastive_projection(first),
                    contrastive_projection(second),
                    temperature=args.contrastive_temperature,
                )
                totals["preservation"] += float(preservation.cpu())
                totals["temporal_order"] += float(order.cpu())
                totals["physiology_change"] += float(change.cpu())
                totals["contrastive"] += float(contrastive.cpu())
                batches += 1
        student.train()
        temporal_head.train()
        physiology_change_head.train()
        contrastive_projection.train()
        result = {name: value / max(batches, 1) for name, value in totals.items()}
        result["score"] = (
            args.preservation_weight * result["preservation"]
            + args.temporal_order_weight * result["temporal_order"]
            + args.physiology_change_weight * result["physiology_change"]
            + args.contrastive_weight * result["contrastive"]
        )
        return result

    while step < args.max_steps:
        for item in train_loader:
            optimizer.zero_grad(set_to_none=True)
            earlier = _sensor_batch(item["earlier"], device)
            later = _sensor_batch(item["later"], device)
            subset = drop_sensor_channels(
                earlier,
                drop_probability=0.35,
                patch_size=student.patch_size,
                min_observed_fraction=student.min_observed_fraction,
            )
            masked, artificial_mask = mask_sensor_patches(
                earlier,
                patch_size=student.patch_size,
                mask_probability=0.15,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                first = student(earlier)
                second = student(later)
                subset_output = student(subset)
                masked_output = student(masked)
                with torch.no_grad():
                    native_first = student.forward_native(earlier).pooled
                    native_second = student.forward_native(later).pooled
                reconstruction = masked_patch_reconstruction_loss(
                    reconstruction_head(masked_output.latent),
                    earlier.values,
                    artificial_mask,
                    patch_size=student.patch_size,
                )
                consistency = sensor_set_consistency_loss(
                    first.pooled, subset_output.pooled
                )
                preservation = 0.5 * (
                    preservation_loss(first.pooled, native_first)
                    + preservation_loss(second.pooled, native_second)
                )
                swap = torch.rand(first.pooled.shape[0], device=device) < 0.5
                order_first = torch.where(swap[:, None], second.pooled, first.pooled)
                order_second = torch.where(swap[:, None], first.pooled, second.pooled)
                order_target = (~swap).to(first.pooled.dtype)
                order = temporal_order_loss(
                    temporal_head, order_first, order_second, order_target
                )
                change = physiology_change_loss(
                    physiology_change_head,
                    first.pooled,
                    second.pooled,
                    earlier.values,
                    later.values,
                )
                contrastive = adjacent_day_contrastive_loss(
                    contrastive_projection(first.pooled),
                    contrastive_projection(second.pooled),
                    temperature=args.contrastive_temperature,
                )
                total = (
                    args.reconstruction_weight * reconstruction
                    + args.consistency_weight * consistency
                    + args.preservation_weight * preservation
                    + args.temporal_order_weight * order
                    + args.physiology_change_weight * change
                    + args.contrastive_weight * contrastive
                )
            if not bool(torch.isfinite(total)):
                raise FloatingPointError(f"non-finite loss at step {step}")
            total.backward()
            gradient_norm = clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            step += 1
            record: dict[str, float | int | str] = {
                "event": "train",
                "step": step,
                "total": float(total.detach().float().cpu()),
                "reconstruction": float(reconstruction.detach().float().cpu()),
                "sensor_consistency": float(consistency.detach().float().cpu()),
                "native_preservation": float(preservation.detach().float().cpu()),
                "temporal_order": float(order.detach().float().cpu()),
                "physiology_change": float(change.detach().float().cpu()),
                "adjacent_day_contrastive": float(contrastive.detach().float().cpu()),
                "gradient_norm": float(gradient_norm.detach().float().cpu()),
            }
            history.append(record)
            if step == 1 or step % args.log_every == 0:
                print(json.dumps(record), flush=True)

            if step % args.validate_every == 0 or step == args.max_steps:
                validation = validation_score()
                history.append({"event": "validation", "step": step, **validation})
                print(
                    json.dumps({"event": "validation", "step": step, **validation}),
                    flush=True,
                )
                if validation["score"] < best_score:
                    best_score = validation["score"]
                    best_step = step
                    save("best", args.output.with_name(f"{args.output.stem}-best.ckpt"))
            if step % args.save_every == 0:
                save("running")
                print(json.dumps({"event": "checkpoint", "step": step}), flush=True)
            if step >= args.max_steps:
                break

    save("complete")
    for dataset in (*train_pairs, *validation_pairs):
        dataset.close()


if __name__ == "__main__":
    main()
