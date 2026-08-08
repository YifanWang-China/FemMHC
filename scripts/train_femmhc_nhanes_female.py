"""Continual female pretraining on NHANES wrist activity and sleep-wear."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import time
from typing import Any
from collections import defaultdict

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset

from femmhc import (
    FemMHCEncoder,
    NHANES_FEMALE_SENSOR_DESCRIPTORS,
    OPENMHC_SENSOR_DESCRIPTORS,
    PatchReconstructionHead,
    SensorBatch,
    TemporalOrderHead,
    drop_sensor_channels,
    mask_sensor_patches,
    masked_patch_reconstruction_loss,
    preservation_loss,
    preservation_distance,
    pool_native_openmhc,
    sensor_set_consistency_loss,
    temporal_order_loss,
)
from femmhc.checkpointing import (
    capture_rng_state,
    restore_rng_state,
    save_training_checkpoint,
)
from femmhc.data import NHANESFemaleTemporalPairDataset, OpenMHCFemaleDataset
from openmhc.models.lsm2.modules import LSM2Module


def _sensor_batch(item: dict[str, Any], device: torch.device) -> SensorBatch:
    return SensorBatch(
        item["sensor_values"].to(device, non_blocking=True),
        NHANES_FEMALE_SENSOR_DESCRIPTORS,
        item["channel_present"].to(device, non_blocking=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initial-femmhc-checkpoint", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--unfreeze-last-blocks", type=int, default=0)
    parser.add_argument("--temporal-weight", type=float, default=0.5)
    parser.add_argument("--preservation-weight", type=float, default=10.0)
    parser.add_argument("--openmhc-root", type=Path)
    parser.add_argument("--replay-every", type=int, default=4)
    parser.add_argument("--replay-weight", type=float, default=10.0)
    parser.add_argument("--validation-pairs", type=int, default=512)
    parser.add_argument("--validation-openmhc-days", type=int, default=64)
    parser.add_argument("--selection-preservation-weight", type=float, default=10.0)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(
        args.max_steps,
        args.batch_size,
        args.save_every,
        args.validation_pairs,
        args.validation_openmhc_days,
    ) <= 0:
        raise ValueError("step, batch, validation and checkpoint values must be positive")
    if args.replay_every <= 0 or args.replay_weight < 0:
        raise ValueError("replay interval must be positive and replay weight non-negative")
    if args.selection_preservation_weight < 0:
        raise ValueError("selection preservation weight must be non-negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    source = LSM2Module.load_from_checkpoint(str(args.checkpoint), map_location="cpu")
    student = FemMHCEncoder(source.model, freeze_backbone=True)
    native_teacher = source.model.to(device).eval()
    for parameter in native_teacher.parameters():
        parameter.requires_grad = False
    del source
    initial = torch.load(
        args.initial_femmhc_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    student.load_state_dict(initial["student_state_dict"])
    if not 0 <= args.unfreeze_last_blocks <= len(student.encoder.blocks):
        raise ValueError("unfreeze-last-blocks exceeds encoder depth")
    backbone_parameters: list[torch.nn.Parameter] = []
    if args.unfreeze_last_blocks:
        for block in student.encoder.blocks[-args.unfreeze_last_blocks :]:
            for parameter in block.parameters():
                parameter.requires_grad = True
                backbone_parameters.append(parameter)
        for parameter in student.encoder.norm.parameters():
            parameter.requires_grad = True
            backbone_parameters.append(parameter)
    student = student.to(device).train()
    teacher = copy.deepcopy(student).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    reconstruction_head = PatchReconstructionHead(
        student.embed_dim, student.patch_size
    ).to(device).train()
    temporal_head = TemporalOrderHead(student.embed_dim).to(device).train()
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    primary_parameters = [
        parameter
        for module in (student, reconstruction_head, temporal_head)
        for parameter in module.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    optimizer_groups: list[dict[str, Any]] = [
        {"params": primary_parameters, "lr": args.learning_rate}
    ]
    if backbone_parameters:
        optimizer_groups.append(
            {"params": backbone_parameters, "lr": args.backbone_learning_rate}
        )
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=0.01)
    train_dataset = NHANESFemaleTemporalPairDataset(
        args.processed_dir, split="train", normalize=True
    )
    validation_dataset = NHANESFemaleTemporalPairDataset(
        args.processed_dir, split="validation", normalize=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    replay_loader = None
    replay_iterator = None
    openmhc_validation_loader = None
    if args.openmhc_root is not None:
        replay_dataset = OpenMHCFemaleDataset(args.openmhc_root, split="train")
        replay_loader = DataLoader(
            replay_dataset,
            batch_size=1,
            shuffle=True,
            pin_memory=device.type == "cuda",
        )
        replay_iterator = iter(replay_loader)
        openmhc_validation_dataset = OpenMHCFemaleDataset(
            args.openmhc_root, split="validation"
        )
        openmhc_validation_indices = openmhc_validation_dataset.balanced_indices(
            min(
                len(openmhc_validation_dataset),
                args.validation_openmhc_days * 2,
            ),
            seed=args.seed,
        )
        openmhc_validation_loader = DataLoader(
            Subset(openmhc_validation_dataset, openmhc_validation_indices),
            batch_size=min(4, args.validation_openmhc_days),
            shuffle=False,
            pin_memory=device.type == "cuda",
        )

    history: list[dict[str, float | int]] = []
    step = 0
    elapsed_offset = 0.0
    best_validation_loss = float("inf")
    best_validation_temporal_loss = float("inf")
    best_validation_openmhc_preservation = float("inf")
    best_step = 0
    if args.resume and args.output.is_file():
        artifact = torch.load(args.output, map_location="cpu", weights_only=False)
        if artifact.get("stage") != "nhanes_female_continual_pretraining_v3":
            raise ValueError(f"not an NHANES-female checkpoint: {args.output}")
        student.load_state_dict(artifact["student_state_dict"])
        reconstruction_head.load_state_dict(artifact["reconstruction_head_state_dict"])
        temporal_head.load_state_dict(artifact["temporal_head_state_dict"])
        optimizer.load_state_dict(artifact["optimizer_state_dict"])
        step = int(artifact["steps"])
        history = list(artifact.get("history", []))
        elapsed_offset = float(artifact.get("elapsed_seconds", 0.0))
        best_validation_loss = float(artifact.get("best_validation_score", float("inf")))
        best_validation_temporal_loss = float(
            artifact.get("best_validation_temporal_loss", float("inf"))
        )
        best_validation_openmhc_preservation = float(
            artifact.get("best_validation_openmhc_preservation", float("inf"))
        )
        best_step = int(artifact.get("best_step", 0))
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
                "stage": "nhanes_female_continual_pretraining_v3",
                "status": status,
                "source_checkpoint": str(args.checkpoint.resolve()),
                "initial_femmhc_checkpoint": str(
                    args.initial_femmhc_checkpoint.resolve()
                ),
                "initial_stage": str(initial.get("stage", "unknown")),
                "seed": args.seed,
                "steps": step,
                "max_steps": args.max_steps,
                "training_pairs": len(train_dataset),
                "validation_pairs": len(validation_dataset),
                "unfreeze_last_blocks": args.unfreeze_last_blocks,
                "temporal_weight": args.temporal_weight,
                "preservation_weight": args.preservation_weight,
                "openmhc_replay_root": (
                    str(args.openmhc_root.resolve()) if args.openmhc_root else None
                ),
                "openmhc_replay_every": args.replay_every,
                "openmhc_replay_weight": args.replay_weight,
                "validation_openmhc_days": args.validation_openmhc_days,
                "validation_openmhc_candidate_days": (
                    len(openmhc_validation_indices)
                    if openmhc_validation_loader is not None
                    else 0
                ),
                "validation_openmhc_participants": (
                    len(
                        {
                            openmhc_validation_dataset.participant_ids[index]
                            for index in openmhc_validation_indices
                        }
                    )
                    if openmhc_validation_loader is not None
                    else 0
                ),
                "validation_openmhc_sampling": (
                    "participant_balanced_round_robin"
                    if openmhc_validation_loader is not None
                    else None
                ),
                "validation_openmhc_aggregation": (
                    "participant_mean_then_cohort_mean"
                    if openmhc_validation_loader is not None
                    else None
                ),
                "selection_preservation_weight": args.selection_preservation_weight,
                "best_validation_score": best_validation_loss,
                "best_validation_temporal_loss": best_validation_temporal_loss,
                "best_validation_openmhc_preservation": (
                    best_validation_openmhc_preservation
                ),
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
                "optimizer_state_dict": optimizer.state_dict(),
                **capture_rng_state(),
            },
        )

    def validation_temporal_loss() -> float:
        student.eval()
        temporal_head.eval()
        losses: list[torch.Tensor] = []
        examples = 0
        with torch.inference_mode():
            for item in validation_loader:
                first = _sensor_batch(item["first"], device)
                second = _sensor_batch(item["second"], device)
                combined = SensorBatch(
                    torch.cat([first.values, second.values]),
                    NHANES_FEMALE_SENSOR_DESCRIPTORS,
                    torch.cat([first.present_mask(), second.present_mask()]),
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    embedding = student(combined).pooled
                    first_embedding, second_embedding = embedding.chunk(2)
                    loss = temporal_order_loss(
                        temporal_head,
                        first_embedding,
                        second_embedding,
                        item["second_is_later"].to(device, non_blocking=True),
                    )
                batch_examples = len(item["second_is_later"])
                losses.append(loss.float().cpu() * batch_examples)
                examples += batch_examples
                if examples >= args.validation_pairs:
                    break
        student.train()
        temporal_head.train()
        return float(torch.stack(losses).sum() / examples)

    def validation_openmhc_preservation() -> float:
        if openmhc_validation_loader is None:
            return 0.0
        student.eval()
        participant_losses: dict[str, list[float]] = defaultdict(list)
        examples = 0
        with torch.inference_mode():
            for item in openmhc_validation_loader:
                values = item["sensor_values"].to(device, non_blocking=True)
                present = item["channel_present"].to(device, non_blocking=True)
                usable = torch.isfinite(
                    values.reshape(
                        values.shape[0],
                        values.shape[1],
                        -1,
                        student.patch_size,
                    )
                ).float().mean(dim=-1).ge(student.min_observed_fraction).any(dim=(1, 2))
                if not bool(usable.any()):
                    continue
                participant_ids = [
                    participant_id
                    for participant_id, keep in zip(
                        item["participant_id"], usable.cpu().tolist()
                    )
                    if keep
                ]
                values = values[usable]
                present = present[usable]
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    student_embedding = student(
                        SensorBatch(values, OPENMHC_SENSOR_DESCRIPTORS, present)
                    ).pooled
                    teacher_embedding = pool_native_openmhc(native_teacher, values)
                    distances = preservation_distance(
                        student_embedding, teacher_embedding
                    )
                batch_examples = len(values)
                for participant_id, distance in zip(
                    participant_ids,
                    distances.float().cpu().tolist(),
                ):
                    participant_losses[str(participant_id)].append(float(distance))
                examples += batch_examples
                if examples >= args.validation_openmhc_days:
                    break
        student.train()
        return float(
            np.mean(
                [np.mean(losses) for losses in participant_losses.values()]
            )
        )

    while step < args.max_steps:
        for item in train_loader:
            first = _sensor_batch(item["first"], device)
            second = _sensor_batch(item["second"], device)
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
            replay_item = None
            if replay_loader is not None and step % args.replay_every == 0:
                try:
                    replay_item = next(replay_iterator)
                except StopIteration:
                    replay_iterator = iter(replay_loader)
                    replay_item = next(replay_iterator)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                combined = SensorBatch(
                    torch.cat([first.values, second.values]),
                    NHANES_FEMALE_SENSOR_DESCRIPTORS,
                    torch.cat([first.present_mask(), second.present_mask()]),
                )
                combined_output = student(combined)
                first_embedding, second_embedding = combined_output.pooled.chunk(2)
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
                    first_embedding, subset_output.pooled
                )
                preservation = preservation_loss(
                    first_embedding, teacher_output.pooled
                )
                temporal = temporal_order_loss(
                    temporal_head,
                    first_embedding,
                    second_embedding,
                    item["second_is_later"].to(device, non_blocking=True),
                )
                replay = first_embedding.sum() * 0.0
                if replay_item is not None:
                    replay_values = replay_item["sensor_values"].to(
                        device, non_blocking=True
                    )
                    replay_present = replay_item["channel_present"].to(
                        device, non_blocking=True
                    )
                    replay_batch = SensorBatch(
                        replay_values,
                        OPENMHC_SENSOR_DESCRIPTORS,
                        replay_present,
                    )
                    replay_student = student(replay_batch).pooled
                    with torch.no_grad():
                        replay_teacher = pool_native_openmhc(
                            native_teacher,
                            replay_values,
                        )
                    replay = preservation_loss(replay_student, replay_teacher)
                total = (
                    reconstruction
                    + consistency
                    + args.preservation_weight * preservation
                    + args.temporal_weight * temporal
                    + args.replay_weight * replay
                )
            if not bool(torch.isfinite(total)):
                raise FloatingPointError(f"non-finite loss at step {step}")
            total.backward()
            gradient_norm = clip_grad_norm_(
                primary_parameters + backbone_parameters, 1.0
            )
            optimizer.step()
            step += 1
            record = {
                "step": step,
                "total": float(total.detach().float().cpu()),
                "reconstruction": float(reconstruction.detach().float().cpu()),
                "sensor_consistency": float(consistency.detach().float().cpu()),
                "preservation": float(preservation.detach().float().cpu()),
                "temporal": float(temporal.detach().float().cpu()),
                "openmhc_replay": float(replay.detach().float().cpu()),
                "gradient_norm": float(gradient_norm.detach().float().cpu()),
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            if step % args.save_every == 0:
                validation_temporal = validation_temporal_loss()
                validation_openmhc = validation_openmhc_preservation()
                validation_score = (
                    validation_temporal
                    + args.selection_preservation_weight * validation_openmhc
                )
                if validation_score < best_validation_loss:
                    best_validation_loss = validation_score
                    best_validation_temporal_loss = validation_temporal
                    best_validation_openmhc_preservation = validation_openmhc
                    best_step = step
                    save("best", args.output.with_name(args.output.stem + "-best.ckpt"))
                save("running")
                print(
                    json.dumps(
                        {
                            "event": "checkpoint",
                            "step": step,
                            "validation_score": validation_score,
                            "validation_temporal_loss": validation_temporal,
                            "validation_openmhc_preservation": validation_openmhc,
                            "best_validation_score": best_validation_loss,
                            "best_validation_temporal_loss": (
                                best_validation_temporal_loss
                            ),
                            "best_validation_openmhc_preservation": (
                                best_validation_openmhc_preservation
                            ),
                            "best_step": best_step,
                        }
                    ),
                    flush=True,
                )
            if step >= args.max_steps:
                break

    save("complete")
    train_dataset.close()
    validation_dataset.close()


if __name__ == "__main__":
    main()
