"""Pregnancy specialization with within-woman physiological progression."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from femmhc import (
    FemMHCEncoder,
    PREGNANCY_GA_SENSOR_DESCRIPTORS,
    PatchReconstructionHead,
    PregnancyGAHead,
    SensorBatch,
    TemporalOrderHead,
    drop_sensor_channels,
    mask_sensor_patches,
    masked_patch_reconstruction_loss,
    preservation_loss,
    sensor_set_consistency_loss,
    temporal_order_loss,
)
from femmhc.checkpointing import (
    capture_rng_state,
    restore_rng_state,
    save_training_checkpoint,
)
from femmhc.data import PregnancyGAProgressionPairDataset, PregnancyGAWindowDataset
from openmhc.models.lsm2.modules import LSM2Module


def _flatten_window(item: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    values = item["sensor_values"]
    present = item["channel_present"]
    batch, days, channels, minutes = values.shape
    return (
        values.reshape(batch * days, channels, minutes),
        present.reshape(batch * days, channels),
    )


def _measurement_embedding(
    daily: torch.Tensor,
    head: PregnancyGAHead,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = head(daily)
    pooled = torch.sum(daily * output.day_attention.unsqueeze(-1), dim=1)
    return pooled, output.prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initial-femmhc-checkpoint", type=Path)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ga-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--unfreeze-last-blocks", type=int, default=2)
    parser.add_argument("--progression-weight", type=float, default=1.0)
    parser.add_argument("--gestational-age-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if (
        args.max_steps <= 0
        or args.batch_size <= 0
        or args.ga_batch_size <= 0
        or args.save_every <= 0
    ):
        raise ValueError("steps, batch size and checkpoint interval must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    source = LSM2Module.load_from_checkpoint(str(args.checkpoint), map_location="cpu")
    student = FemMHCEncoder(source.model, freeze_backbone=True)
    del source
    initial_stage = "openmhc_initialization"
    if args.initial_femmhc_checkpoint is not None:
        initial = torch.load(
            args.initial_femmhc_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        student.load_state_dict(initial["student_state_dict"])
        initial_stage = str(initial.get("stage", "unknown"))
    student = student.to(device).train()
    if not 0 <= args.unfreeze_last_blocks <= len(student.encoder.blocks):
        raise ValueError("unfreeze-last-blocks exceeds the OpenMHC encoder depth")
    backbone_parameters: list[torch.nn.Parameter] = []
    if args.unfreeze_last_blocks:
        for block in student.encoder.blocks[-args.unfreeze_last_blocks :]:
            for parameter in block.parameters():
                parameter.requires_grad = True
                backbone_parameters.append(parameter)
        for parameter in student.encoder.norm.parameters():
            parameter.requires_grad = True
            backbone_parameters.append(parameter)
    teacher = copy.deepcopy(student).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    reconstruction_head = PatchReconstructionHead(
        student.embed_dim, student.patch_size
    ).to(device).train()
    progression_head = TemporalOrderHead(student.embed_dim).to(device).train()
    gestational_age_head = PregnancyGAHead(student.embed_dim).to(device).train()
    backbone_parameter_ids = {id(parameter) for parameter in backbone_parameters}
    primary_parameters = [
        parameter
        for module in (
            student,
            reconstruction_head,
            progression_head,
            gestational_age_head,
        )
        for parameter in module.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_parameter_ids
    ]
    trainable = primary_parameters + backbone_parameters
    optimizer_groups: list[dict[str, Any]] = [
        {"params": primary_parameters, "lr": args.learning_rate}
    ]
    if backbone_parameters:
        optimizer_groups.append(
            {"params": backbone_parameters, "lr": args.backbone_learning_rate}
        )
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=0.01)
    dataset = PregnancyGAProgressionPairDataset(
        args.processed_dir,
        split="train",
        normalize=True,
        include_light=True,
    )
    if len(dataset) == 0:
        raise ValueError("the training split has no longitudinal pregnancy pairs")
    ga_dataset = PregnancyGAWindowDataset(
        args.processed_dir,
        split="train",
        normalize=True,
        include_light=True,
    )
    train_ages = np.asarray(
        [float(row["gestational_age_weeks"]) for row in ga_dataset.rows],
        dtype=np.float32,
    )
    gestational_age_mean = float(train_ages.mean())
    gestational_age_std = float(train_ages.std())
    if gestational_age_std < 1e-6:
        raise ValueError("gestational-age training targets have zero variance")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    ga_loader = DataLoader(
        ga_dataset,
        batch_size=args.ga_batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    ga_iterator = iter(ga_loader)
    validation_dataset = PregnancyGAWindowDataset(
        args.processed_dir,
        split="validation",
        normalize=True,
        include_light=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.ga_batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    history: list[dict[str, float | int]] = []
    step = 0
    elapsed_offset = 0.0
    best_validation_mae = float("inf")
    best_step = 0
    if args.resume and args.output.is_file():
        checkpoint = torch.load(args.output, map_location="cpu", weights_only=False)
        if checkpoint.get("stage") != "pregnancy_progression_specialization":
            raise ValueError(f"not a pregnancy-progression checkpoint: {args.output}")
        student.load_state_dict(checkpoint["student_state_dict"])
        reconstruction_head.load_state_dict(checkpoint["reconstruction_head_state_dict"])
        progression_head.load_state_dict(checkpoint["progression_head_state_dict"])
        gestational_age_head.load_state_dict(checkpoint["gestational_age_head_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        step = int(checkpoint["steps"])
        history = list(checkpoint.get("history", []))
        elapsed_offset = float(checkpoint.get("elapsed_seconds", 0.0))
        best_validation_mae = float(checkpoint.get("best_validation_mae_weeks", float("inf")))
        best_step = int(checkpoint.get("best_step", 0))
        restore_rng_state(checkpoint)
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
                "stage": "pregnancy_progression_specialization",
                "status": status,
                "source_checkpoint": str(args.checkpoint.resolve()),
                "initial_femmhc_checkpoint": (
                    str(args.initial_femmhc_checkpoint.resolve())
                    if args.initial_femmhc_checkpoint
                    else None
                ),
                "initial_stage": initial_stage,
                "unfreeze_last_blocks": args.unfreeze_last_blocks,
                "learning_rate": args.learning_rate,
                "backbone_learning_rate": args.backbone_learning_rate,
                "seed": args.seed,
                "steps": step,
                "max_steps": args.max_steps,
                "training_pairs": len(dataset),
                "training_measurements": len(ga_dataset),
                "best_validation_mae_weeks": best_validation_mae,
                "best_step": best_step,
                "gestational_age_target_mean": gestational_age_mean,
                "gestational_age_target_std": gestational_age_std,
                "history": history,
                "elapsed_seconds": elapsed_offset + time.perf_counter() - started,
                "peak_gpu_memory_gb": (
                    torch.cuda.max_memory_allocated(device) / 1024**3
                    if device.type == "cuda"
                    else None
                ),
                "student_state_dict": student.state_dict(),
                "reconstruction_head_state_dict": reconstruction_head.state_dict(),
                "progression_head_state_dict": progression_head.state_dict(),
                "gestational_age_head_state_dict": gestational_age_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                **capture_rng_state(),
            },
        )

    def validation_mae() -> float:
        student.eval()
        gestational_age_head.eval()
        absolute_errors: list[torch.Tensor] = []
        with torch.inference_mode():
            for validation_item in validation_loader:
                values, present = _flatten_window(validation_item)
                batch_size = validation_item["gestational_age_weeks"].shape[0]
                days = validation_item["sensor_values"].shape[1]
                sensor_batch = SensorBatch(
                    values.to(device, non_blocking=True),
                    PREGNANCY_GA_SENSOR_DESCRIPTORS,
                    present.to(device, non_blocking=True),
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    daily = student(sensor_batch).pooled.reshape(batch_size, days, -1)
                    _, normalized = _measurement_embedding(
                        daily, gestational_age_head
                    )
                prediction = normalized.float() * gestational_age_std + gestational_age_mean
                target = validation_item["gestational_age_weeks"].to(
                    device, non_blocking=True
                )
                absolute_errors.append((prediction - target).abs().cpu())
        student.train()
        gestational_age_head.train()
        return float(torch.cat(absolute_errors).mean())

    while step < args.max_steps:
        for item in loader:
            try:
                ga_item = next(ga_iterator)
            except StopIteration:
                ga_iterator = iter(ga_loader)
                ga_item = next(ga_iterator)
            first_values, first_present = _flatten_window(item["first"])
            second_values, second_present = _flatten_window(item["second"])
            batch_size = item["second_is_later"].shape[0]
            days = item["first"]["sensor_values"].shape[1]
            first = SensorBatch(
                first_values.to(device, non_blocking=True),
                PREGNANCY_GA_SENSOR_DESCRIPTORS,
                first_present.to(device, non_blocking=True),
            )
            second = SensorBatch(
                second_values.to(device, non_blocking=True),
                PREGNANCY_GA_SENSOR_DESCRIPTORS,
                second_present.to(device, non_blocking=True),
            )
            subset = drop_sensor_channels(
                first,
                drop_probability=0.35,
                patch_size=student.patch_size,
                min_observed_fraction=student.min_observed_fraction,
            )
            masked, artificial_mask = mask_sensor_patches(
                first,
                patch_size=student.patch_size,
                mask_probability=0.15,
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                combined = SensorBatch(
                    torch.cat([first.values, second.values], dim=0),
                    PREGNANCY_GA_SENSOR_DESCRIPTORS,
                    torch.cat([first.present_mask(), second.present_mask()], dim=0),
                )
                combined_output = student(combined)
                first_daily, second_daily = combined_output.pooled.split(
                    [batch_size * days, batch_size * days]
                )
                first_daily = first_daily.reshape(batch_size, days, -1)
                second_daily = second_daily.reshape(batch_size, days, -1)
                first_embedding, _ = _measurement_embedding(
                    first_daily, gestational_age_head
                )
                second_embedding, _ = _measurement_embedding(
                    second_daily, gestational_age_head
                )
                progression = temporal_order_loss(
                    progression_head,
                    first_embedding,
                    second_embedding,
                    item["second_is_later"].to(device, non_blocking=True),
                )
                ga_values, ga_present = _flatten_window(ga_item)
                ga_batch_size = ga_item["gestational_age_weeks"].shape[0]
                ga_days = ga_item["sensor_values"].shape[1]
                ga_sensor_batch = SensorBatch(
                    ga_values.to(device, non_blocking=True),
                    PREGNANCY_GA_SENSOR_DESCRIPTORS,
                    ga_present.to(device, non_blocking=True),
                )
                ga_daily = student(ga_sensor_batch).pooled.reshape(
                    ga_batch_size, ga_days, -1
                )
                _, ga_prediction = _measurement_embedding(
                    ga_daily, gestational_age_head
                )
                ga_target = ga_item["gestational_age_weeks"].to(
                    device, non_blocking=True
                )
                ga_target = (ga_target - gestational_age_mean) / gestational_age_std
                ga_regression = F.smooth_l1_loss(ga_prediction, ga_target)

                subset_output = student(subset)
                masked_output = student(masked)
                with torch.no_grad():
                    teacher_output = teacher(first)
                reconstruction = masked_patch_reconstruction_loss(
                    reconstruction_head(masked_output.latent),
                    first.values,
                    artificial_mask,
                    patch_size=student.patch_size,
                )
                consistency = sensor_set_consistency_loss(
                    first_daily.reshape(batch_size * days, -1),
                    subset_output.pooled,
                )
                preservation = preservation_loss(
                    first_daily.reshape(batch_size * days, -1),
                    teacher_output.pooled,
                )
                total = (
                    reconstruction
                    + consistency
                    + preservation
                    + args.progression_weight * progression
                    + args.gestational_age_weight * ga_regression
                )
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
                "progression": float(progression.detach().float().cpu()),
                "gestational_age": float(ga_regression.detach().float().cpu()),
                "gradient_norm": float(gradient_norm.detach().float().cpu()),
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            if step % args.save_every == 0:
                current_validation_mae = validation_mae()
                improved = current_validation_mae < best_validation_mae
                if improved:
                    best_validation_mae = current_validation_mae
                    best_step = step
                save("running")
                if improved:
                    best_path = args.output.with_name(
                        f"{args.output.stem}-best{args.output.suffix}"
                    )
                    save("best_validation", best_path)
                print(
                    json.dumps(
                        {
                            "event": "checkpoint",
                            "step": step,
                            "validation_mae_weeks": current_validation_mae,
                            "best_validation_mae_weeks": best_validation_mae,
                            "best_step": best_step,
                        }
                    ),
                    flush=True,
                )
            if step >= args.max_steps:
                break

    dataset.close()
    ga_dataset.close()
    validation_dataset.close()
    save("complete")


if __name__ == "__main__":
    main()
