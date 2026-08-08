"""Minimal reproducible FemMHC continual-pretraining entry point."""

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
    MCPHASES_TASKS,
    MCPHASES_SENSOR_DESCRIPTORS,
    McPhasesTaskHeads,
    McPhasesV2TaskHeads,
    PatchReconstructionHead,
    SensorBatch,
    TemporalOrderHead,
    combine_losses,
    cyclic_phase_loss,
    drop_sensor_channels,
    mask_sensor_patches,
    masked_patch_reconstruction_loss,
    masked_task_loss,
    nested_onset_loss,
    preservation_loss,
    sensor_set_consistency_loss,
    temporal_order_loss,
)
from femmhc.data import McPhasesTemporalPairDataset
from femmhc.checkpointing import (
    capture_rng_state,
    restore_rng_state,
    save_training_checkpoint,
)
from openmhc.models.lsm2.modules import LSM2Module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--femmhc-init",
        type=Path,
        help="Optional FemMHC checkpoint from the female OpenMHC specialization stage.",
    )
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--self-supervised-weight",
        type=float,
        default=1.0,
        help="Weight for the reconstruction, consistency, preservation, and trajectory objectives.",
    )
    parser.add_argument("--supervised-weight", type=float, default=0.5)
    parser.add_argument(
        "--cycle-loss-weight",
        type=float,
        default=1.0,
        help="Relative weight of the cycle-phase group in v2 supervised loss.",
    )
    parser.add_argument(
        "--symptom-loss-weight",
        type=float,
        default=1.0,
        help="Relative weight of the symptom group in v2 supervised loss.",
    )
    parser.add_argument(
        "--onset-loss-weight",
        type=float,
        default=1.0,
        help="Relative weight of the nested onset group in v2 supervised loss.",
    )
    parser.add_argument(
        "--hormone-loss-weight",
        type=float,
        default=1.0,
        help="Relative weight of the hormone group in v2 supervised loss.",
    )
    parser.add_argument(
        "--internal-adapter-rank",
        type=int,
        default=0,
        help="Bottleneck rank for adapters inserted inside the final LSM2 blocks.",
    )
    parser.add_argument(
        "--internal-adapter-layers",
        type=int,
        default=0,
        help="Number of final LSM2 Transformer blocks receiving internal adapters.",
    )
    parser.add_argument(
        "--task-head-version",
        choices=("v1", "v2"),
        default="v1",
        help="v2 enables grouped adapters, cyclic phase loss, and nested onset risk.",
    )
    parser.add_argument(
        "--task-group",
        choices=("all", "female_six", "cycle", "symptoms", "onset", "hormones"),
        default="all",
        help="Train all v2 task families or isolate one family for task-specific probing.",
    )
    parser.add_argument(
        "--freeze-student",
        action="store_true",
        help="Freeze the entire FemMHC encoder and train task heads only (matched-head baseline).",
    )
    parser.add_argument(
        "--linear-cycle-head",
        action="store_true",
        help="Use a normalized linear cycle head so supervision must reshape the encoder.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument(
        "--keep-periodic-checkpoints",
        action="store_true",
        help="Retain step-numbered checkpoints in addition to the resumable latest checkpoint.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def sensor_batch(item: dict[str, torch.Tensor], device: torch.device) -> SensorBatch:
    return SensorBatch(
        item["sensor_values"].to(device, non_blocking=True),
        MCPHASES_SENSOR_DESCRIPTORS,
        item["channel_present"].to(device, non_blocking=True),
    )


def main() -> None:
    args = parse_args()
    if args.max_steps <= 0 or args.save_every <= 0:
        raise ValueError("--max-steps must be positive")
    if min(
        args.cycle_loss_weight,
        args.symptom_loss_weight,
        args.onset_loss_weight,
        args.hormone_loss_weight,
    ) <= 0:
        raise ValueError("supervised group weights must be positive")
    if args.task_head_version == "v1" and args.task_group != "all":
        raise ValueError("--task-group is only supported by v2 task heads")
    if args.task_head_version == "v1" and args.linear_cycle_head:
        raise ValueError("--linear-cycle-head requires v2 task heads")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    source = LSM2Module.load_from_checkpoint(str(args.checkpoint), map_location="cpu")
    student = FemMHCEncoder(
        source.model,
        freeze_backbone=True,
        internal_adapter_rank=args.internal_adapter_rank,
        internal_adapter_layers=args.internal_adapter_layers,
    ).to(device).train()
    if args.femmhc_init is not None:
        initialization = torch.load(args.femmhc_init, map_location="cpu", weights_only=False)
        student.load_state_dict(initialization["student_state_dict"])
    if args.freeze_student:
        for parameter in student.parameters():
            parameter.requires_grad = False
        student.eval()
    teacher = copy.deepcopy(student).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    del source
    reconstruction_head = PatchReconstructionHead(student.embed_dim, student.patch_size).to(device).train()
    order_head = TemporalOrderHead(student.embed_dim).to(device).train()
    if args.task_head_version == "v2":
        task_heads = McPhasesV2TaskHeads(
            student.embed_dim,
            linear_cycle_head=args.linear_cycle_head,
        ).to(device).train()
    else:
        task_heads = McPhasesTaskHeads(student.embed_dim).to(device).train()
    trainable = [
        parameter
        for module in (student, reconstruction_head, order_head, task_heads)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)

    dataset = McPhasesTemporalPairDataset(args.processed_dir, split="train")
    training_labels = np.asarray(
        dataset.days.labels[dataset.days.sample_indices],
        dtype=np.int64,
    )
    training_hormones = np.asarray(
        dataset.days.hormones[dataset.days.sample_indices],
        dtype=np.float32,
    )
    logged_hormones = np.log1p(training_hormones)
    hormone_means = np.nanmean(logged_hormones, axis=0).astype(np.float32)
    hormone_stds = np.nanstd(logged_hormones, axis=0).astype(np.float32)
    hormone_stds = np.maximum(hormone_stds, 1e-6)
    task_class_weights: dict[str, torch.Tensor] = {}
    for task in MCPHASES_TASKS:
        if task.kind == "regression" or task.label_column is None:
            continue
        observed = training_labels[:, task.label_column]
        observed = observed[observed >= 0]
        counts = np.bincount(observed, minlength=task.classes or 2).astype(np.float64)
        exponent = 1.0 if args.task_head_version == "v2" else 0.5
        weights = (observed.size / np.maximum(counts * len(counts), 1.0)) ** exponent
        weights = weights / weights.mean()
        task_class_weights[task.name] = torch.tensor(weights, dtype=torch.float32)

    onset_observed = (training_labels[:, 8] >= 0) & (training_labels[:, 9] >= 0)
    onset_24h = training_labels[onset_observed, 8]
    onset_72h = training_labels[onset_observed, 9]
    onset_bins = np.where(onset_24h == 1, 0, np.where(onset_72h == 1, 1, 2))
    onset_counts = np.bincount(onset_bins, minlength=3).astype(np.float64)
    onset_weights = onset_bins.size / np.maximum(onset_counts * 3, 1.0)
    onset_weights = onset_weights / onset_weights.mean()
    onset_class_weights = torch.tensor(onset_weights, dtype=torch.float32)

    hormone_index = {"lh": 0, "estrogen": 1, "pdg": 2}

    def compute_supervised_loss(
        task_outputs: dict[str, torch.Tensor],
        item: dict[str, object],
        onset_output=None,
    ) -> torch.Tensor:
        supervised_losses: list[torch.Tensor] = []
        grouped_losses: dict[str, list[torch.Tensor]] = {
            "cycle": [],
            "symptoms": [],
            "onset": [],
            "hormones": [],
        }
        for task in MCPHASES_TASKS:
            if args.task_head_version == "v2" and task.name in {
                "menstrual_onset_24h",
                "menstrual_onset_72h",
            }:
                continue
            if task.kind == "regression":
                index = hormone_index[task.name]
                target = item["earlier"]["hormones"][:, index]
                target = (torch.log1p(target) - hormone_means[index]) / hormone_stds[index]
            elif task.target_offset_days == 1:
                target = item["later"]["labels"][:, task.label_column]
            else:
                target = item["earlier"]["labels"][:, task.label_column]
            if args.task_head_version == "v2" and task.name == "cycle_phase":
                loss = cyclic_phase_loss(
                    task_outputs[task.name],
                    target.to(device),
                    class_weights=task_class_weights.get(task.name),
                )
            else:
                loss = masked_task_loss(
                    task_outputs[task.name],
                    target.to(device),
                    kind=task.kind,
                    class_weights=task_class_weights.get(task.name),
                )
            supervised_losses.append(loss)
            if args.task_head_version == "v2":
                if task.kind == "regression":
                    group = "hormones"
                elif task.name == "cycle_phase":
                    group = "cycle"
                else:
                    group = "symptoms"
                grouped_losses[group].append(loss)
        if args.task_head_version == "v2":
            if onset_output is None:
                raise ValueError("v2 task heads must provide nested onset output")
            onset_loss = nested_onset_loss(
                onset_output,
                item["earlier"]["labels"][:, 8].to(device),
                item["earlier"]["labels"][:, 9].to(device),
                class_weights=onset_class_weights,
            )
            grouped_losses["onset"].append(onset_loss)
            if args.task_group == "all":
                selected_names = ("cycle", "symptoms", "onset", "hormones")
            elif args.task_group == "female_six":
                selected_names = ("cycle", "symptoms", "onset")
            else:
                selected_names = (args.task_group,)
            group_weights = {
                "cycle": args.cycle_loss_weight,
                "symptoms": args.symptom_loss_weight,
                "onset": args.onset_loss_weight,
                "hormones": args.hormone_loss_weight,
            }
            weighted_means = [
                (torch.stack(grouped_losses[name]).mean(), group_weights[name])
                for name in selected_names
                if grouped_losses[name]
            ]
            numerator = sum(loss * weight for loss, weight in weighted_means)
            denominator = sum(weight for _, weight in weighted_means)
            return numerator / denominator
        return torch.stack(supervised_losses).mean()

    def forward_task_heads(embedding: torch.Tensor):
        if args.task_head_version == "v2":
            return task_heads.forward_with_aux(embedding)
        return task_heads(embedding), None

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    history: list[dict[str, float | int]] = []
    step = 0
    elapsed_offset = 0.0
    if args.resume and args.output.is_file():
        checkpoint = torch.load(args.output, map_location="cpu", weights_only=False)
        if checkpoint.get("stage") not in {None, "mcphases_specialization"}:
            raise ValueError(f"Not an mcPHASES checkpoint: {args.output}")
        student.load_state_dict(checkpoint["student_state_dict"])
        reconstruction_head.load_state_dict(checkpoint["reconstruction_head_state_dict"])
        order_head.load_state_dict(checkpoint["order_head_state_dict"])
        if "task_heads_state_dict" in checkpoint:
            task_heads.load_state_dict(checkpoint["task_heads_state_dict"])
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
        payload = {
                "format_version": 1,
                "model": "FemMHC",
                "stage": "mcphases_specialization",
                "status": status,
                "source_checkpoint": str(args.checkpoint.resolve()),
                "femmhc_initialization": str(args.femmhc_init.resolve()) if args.femmhc_init else None,
                "seed": args.seed,
                "self_supervised_weight": args.self_supervised_weight,
                "supervised_weight": args.supervised_weight,
                "cycle_loss_weight": args.cycle_loss_weight,
                "symptom_loss_weight": args.symptom_loss_weight,
                "onset_loss_weight": args.onset_loss_weight,
                "hormone_loss_weight": args.hormone_loss_weight,
                "internal_adapter_rank": args.internal_adapter_rank,
                "internal_adapter_layers": args.internal_adapter_layers,
                "task_head_version": args.task_head_version,
                "task_group": args.task_group,
                "linear_cycle_head": args.linear_cycle_head,
                "freeze_student": args.freeze_student,
                "hormone_log_means": hormone_means.tolist(),
                "hormone_log_stds": hormone_stds.tolist(),
                "steps": step,
                "max_steps": args.max_steps,
                "history": history,
                "elapsed_seconds": elapsed_offset + time.perf_counter() - started,
                "peak_gpu_memory_gb": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None,
                "student_state_dict": student.state_dict(),
                "reconstruction_head_state_dict": reconstruction_head.state_dict(),
                "order_head_state_dict": order_head.state_dict(),
                "task_heads_state_dict": task_heads.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                **capture_rng_state(),
            }
        save_training_checkpoint(args.output, payload)
        if args.keep_periodic_checkpoints and status == "running":
            periodic_output = args.output.with_name(
                f"{args.output.stem}-step{step:04d}{args.output.suffix}"
            )
            save_training_checkpoint(periodic_output, payload)

    while step < args.max_steps:
        for item in loader:
            optimizer.zero_grad(set_to_none=True)
            if args.freeze_student and args.self_supervised_weight == 0:
                earlier = sensor_batch(item["earlier"], device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    with torch.no_grad():
                        earlier_embedding = student(earlier).pooled
                    task_outputs, onset_output = forward_task_heads(earlier_embedding)
                    supervised = compute_supervised_loss(
                        task_outputs, item, onset_output
                    )
                    zero = supervised.new_zeros(())
                    reconstruction = consistency = preservation = trajectory = zero
                    losses = combine_losses(
                        reconstruction=zero,
                        sensor_consistency=zero,
                        preservation=zero,
                        trajectory=zero,
                    )
            else:
                first = sensor_batch(item["first"], device)
                second = sensor_batch(item["second"], device)
                masked, artificial_mask = mask_sensor_patches(
                    first,
                    patch_size=student.patch_size,
                    mask_probability=0.15,
                )
                subset = drop_sensor_channels(
                    first,
                    drop_probability=0.35,
                    patch_size=student.patch_size,
                    min_observed_fraction=student.min_observed_fraction,
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    first_output = student(first)
                    second_output = student(second)
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
                        first_output.pooled,
                        subset_output.pooled,
                    )
                    preservation = preservation_loss(
                        first_output.pooled,
                        teacher_output.pooled,
                    )
                    trajectory = temporal_order_loss(
                        order_head,
                        first_output.pooled,
                        second_output.pooled,
                        item["second_is_later"].to(device),
                    )
                    chronological = item["second_is_later"].to(device).bool().unsqueeze(-1)
                    earlier_embedding = torch.where(
                        chronological,
                        first_output.pooled,
                        second_output.pooled,
                    )
                    task_outputs, onset_output = forward_task_heads(earlier_embedding)
                    supervised = compute_supervised_loss(
                        task_outputs, item, onset_output
                    )
                    losses = combine_losses(
                        reconstruction=reconstruction,
                        sensor_consistency=consistency,
                        preservation=preservation,
                        trajectory=trajectory,
                    )
            total = (
                args.self_supervised_weight * losses.total
                + args.supervised_weight * supervised
            )
            if not bool(torch.isfinite(total)):
                raise FloatingPointError(f"non-finite loss at step {step}")
            total.backward()
            gradient_norm = clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            step += 1
            record = {
                "step": step,
                "total": float(total.detach().float().cpu()),
                "self_supervised_total": float(losses.total.detach().float().cpu()),
                "supervised": float(supervised.detach().float().cpu()),
                "reconstruction": float(reconstruction.detach().float().cpu()),
                "sensor_consistency": float(consistency.detach().float().cpu()),
                "preservation": float(preservation.detach().float().cpu()),
                "trajectory": float(trajectory.detach().float().cpu()),
                "gradient_norm": float(gradient_norm.detach().float().cpu()),
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
