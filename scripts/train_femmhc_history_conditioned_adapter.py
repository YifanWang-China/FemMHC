#!/usr/bin/env python
"""Fine-tune causal personal-history gates inside frozen OpenMHC adapters.

This entry point starts from a completed static FemMHC internal-adapter
checkpoint.  Only newly introduced causal-history parameters are trainable:
the personal-memory/CycleSSM controller and per-layer history gates.  The
reference history embedding file is wearable-only and every window ends at
``t-1`` for prediction day ``t``.
"""

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
    MCPHASES_SENSOR_DESCRIPTORS,
    McPhasesV2TaskHeads,
    PatchReconstructionHead,
    SensorBatch,
    combine_losses,
    cyclic_phase_loss,
    drop_sensor_channels,
    mask_sensor_patches,
    masked_patch_reconstruction_loss,
    masked_task_loss,
    nested_onset_loss,
    preservation_loss,
    sensor_set_consistency_loss,
)
from femmhc.checkpointing import capture_rng_state, save_training_checkpoint
from femmhc.data import McPhasesHistoryAdapterDataset
from openmhc.models.lsm2.modules import LSM2Module


CORE_SYMPTOMS = ("cramps", "mood_swing", "sleep_issue")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--femmhc-init",
        type=Path,
        required=True,
        help="Completed static internal-adapter checkpoint used as the identity-preserving start.",
    )
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument(
        "--history-embeddings",
        type=Path,
        required=True,
        help="Static, adapted per-day embeddings aligned to processed mcPHASES rows.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--self-supervised-weight", type=float, default=0.25)
    parser.add_argument("--supervised-weight", type=float, default=1.0)
    parser.add_argument("--onset-loss-weight", type=float, default=2.0)
    parser.add_argument("--history-days", type=int, default=60)
    parser.add_argument("--minimum-history-days", type=int, default=7)
    parser.add_argument("--history-context-dim", type=int, default=96)
    parser.add_argument("--history-cycle-modes", type=int, default=8)
    parser.add_argument("--internal-adapter-rank", type=int)
    parser.add_argument("--internal-adapter-layers", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def sensor_batch(item: dict[str, torch.Tensor], device: torch.device) -> SensorBatch:
    return SensorBatch(
        item["sensor_values"].to(device, non_blocking=True),
        MCPHASES_SENSOR_DESCRIPTORS,
        item["channel_present"].to(device, non_blocking=True),
    )


def class_weights(
    dataset: McPhasesHistoryAdapterDataset,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    labels = np.load(dataset.processed_dir / "labels.npy", mmap_mode="r")
    indices = np.asarray(dataset.days.sample_indices, dtype=np.int64)
    columns = {"cycle_phase": 0, "cramps": 1, "mood_swing": 2, "sleep_issue": 4}
    classes = {"cycle_phase": 4, "cramps": 6, "mood_swing": 6, "sleep_issue": 6}
    weights: dict[str, torch.Tensor] = {}
    for name, column in columns.items():
        observed = np.asarray(labels[indices, column])
        observed = observed[observed >= 0]
        counts = np.bincount(observed, minlength=classes[name]).astype(np.float64)
        value = observed.size / np.maximum(counts * len(counts), 1.0)
        value = value / value.mean()
        weights[name] = torch.tensor(value, dtype=torch.float32)
    observed = (labels[indices, 8] >= 0) & (labels[indices, 9] >= 0)
    bins = np.where(
        labels[indices, 8][observed] == 1,
        0,
        np.where(labels[indices, 9][observed] == 1, 1, 2),
    )
    counts = np.bincount(bins, minlength=3).astype(np.float64)
    onset = bins.size / np.maximum(counts * 3, 1.0)
    onset = onset / onset.mean()
    return weights, torch.tensor(onset, dtype=torch.float32)


def load_static_state(
    source_model: torch.nn.Module,
    artifact: dict[str, object],
    *,
    internal_adapter_rank: int,
    internal_adapter_layers: int,
    history_context_dim: int,
    history_days: int,
    history_cycle_modes: int,
    device: torch.device,
) -> tuple[FemMHCEncoder, FemMHCEncoder]:
    static = FemMHCEncoder(
        source_model,
        freeze_backbone=True,
        internal_adapter_rank=internal_adapter_rank,
        internal_adapter_layers=internal_adapter_layers,
    )
    static.load_state_dict(artifact["student_state_dict"])
    static = static.to(device).eval()
    for parameter in static.parameters():
        parameter.requires_grad = False

    student = FemMHCEncoder(
        source_model,
        freeze_backbone=True,
        internal_adapter_rank=internal_adapter_rank,
        internal_adapter_layers=internal_adapter_layers,
        history_conditioned_internal_adapters=True,
        history_context_dim=history_context_dim,
        history_maximum_days=history_days,
        history_cycle_modes=history_cycle_modes,
    )
    loaded = student.load_state_dict(artifact["student_state_dict"], strict=False)
    permitted_missing = {
        key
        for key in student.state_dict()
        if key.startswith("history_encoder.") or key.endswith(".history_gate.0.weight")
        or key.endswith(".history_gate.0.bias")
        or key.endswith(".history_gate.1.weight")
        or key.endswith(".history_gate.1.bias")
    }
    if set(loaded.missing_keys) - permitted_missing or loaded.unexpected_keys:
        raise RuntimeError(
            "static initialization is incompatible with the history-conditioned model: "
            f"missing={loaded.missing_keys}, unexpected={loaded.unexpected_keys}"
        )
    # Attribution is clean: all pre-existing FemMHC parameters remain fixed.
    for parameter in student.parameters():
        parameter.requires_grad = False
    if student.history_encoder is None:
        raise RuntimeError("history-conditioned student is missing its controller")
    for parameter in student.history_encoder.parameters():
        parameter.requires_grad = True
    for adapter in student.internal_adapters.values():
        if adapter.history_gate is None:
            raise RuntimeError("history-conditioned adapter is missing its gate")
        for parameter in adapter.history_gate.parameters():
            parameter.requires_grad = True
    return student.to(device).train(), static


def main() -> None:
    args = parse_args()
    if min(
        args.max_steps,
        args.batch_size,
        args.history_days,
        args.history_context_dim,
        args.history_cycle_modes,
        args.save_every,
    ) <= 0:
        raise ValueError("training, history, and save sizes must be positive")
    if args.minimum_history_days < 0 or args.minimum_history_days > args.history_days:
        raise ValueError("minimum history must be in [0, history_days]")
    if min(args.learning_rate, args.self_supervised_weight, args.supervised_weight, args.onset_loss_weight) < 0:
        raise ValueError("loss weights and learning rate must be non-negative")
    if args.history_context_dim % (2 * args.history_cycle_modes):
        raise ValueError("history-context-dim must be divisible by 2 * history-cycle-modes")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    artifact = torch.load(args.femmhc_init, map_location="cpu", weights_only=False)
    if artifact.get("task_head_version") != "v2":
        raise ValueError("history-conditioned training requires a v2 static FemMHC checkpoint")
    rank = int(
        args.internal_adapter_rank
        if args.internal_adapter_rank is not None
        else artifact.get("internal_adapter_rank", 0)
    )
    layers = int(
        args.internal_adapter_layers
        if args.internal_adapter_layers is not None
        else artifact.get("internal_adapter_layers", 0)
    )
    if rank <= 0 or layers <= 0:
        raise ValueError("the static checkpoint must contain internal adapters")

    source = LSM2Module.load_from_checkpoint(str(args.checkpoint), map_location="cpu")
    student, teacher = load_static_state(
        source.model,
        artifact,
        internal_adapter_rank=rank,
        internal_adapter_layers=layers,
        history_context_dim=args.history_context_dim,
        history_days=args.history_days,
        history_cycle_modes=args.history_cycle_modes,
        device=device,
    )
    del source
    task_heads = McPhasesV2TaskHeads(student.embed_dim).to(device).eval()
    task_heads.load_state_dict(artifact["task_heads_state_dict"])
    reconstruction_head = PatchReconstructionHead(student.embed_dim, student.patch_size).to(device).eval()
    reconstruction_head.load_state_dict(artifact["reconstruction_head_state_dict"])
    for module in (task_heads, reconstruction_head):
        for parameter in module.parameters():
            parameter.requires_grad = False

    dataset = McPhasesHistoryAdapterDataset(
        args.processed_dir,
        args.history_embeddings,
        split="train",
        history_days=args.history_days,
        minimum_history_days=args.minimum_history_days,
    )
    if not len(dataset):
        raise ValueError("history dataset has no eligible training examples")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    task_weights, onset_weights = class_weights(dataset)
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)

    def supervised_loss(
        outputs: dict[str, object],
        onset: object,
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        cycle = cyclic_phase_loss(
            outputs["cycle_phase"],
            targets["cycle_phase"].to(device),
            class_weights=task_weights["cycle_phase"],
        )
        symptom = torch.stack(
            [
                masked_task_loss(
                    outputs[name],
                    targets[name].to(device),
                    kind="ordinal",
                    class_weights=task_weights[name],
                )
                for name in CORE_SYMPTOMS
            ]
        ).mean()
        onset_value = nested_onset_loss(
            onset,
            targets["menstrual_onset_24h"].to(device),
            targets["menstrual_onset_72h"].to(device),
            class_weights=onset_weights,
        )
        return (cycle + symptom + args.onset_loss_weight * onset_value) / (
            2.0 + args.onset_loss_weight
        )

    history: list[dict[str, float | int]] = []
    iterator = iter(loader)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, args.max_steps + 1):
        try:
            item = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            item = next(iterator)
        batch = sensor_batch(item, device)
        history_embeddings = item["history_embeddings"].to(device, non_blocking=True)
        history_present = item["history_present"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = student(
                batch,
                history_embeddings=history_embeddings,
                history_present=history_present,
            )
            task_output, onset_output = task_heads.forward_with_aux(output.pooled)
            supervised = supervised_loss(task_output, onset_output, item["targets"])
            masked, artificial_mask = mask_sensor_patches(
                batch,
                patch_size=student.patch_size,
                mask_probability=0.15,
            )
            subset = drop_sensor_channels(
                batch,
                drop_probability=0.35,
                patch_size=student.patch_size,
                min_observed_fraction=student.min_observed_fraction,
            )
            masked_output = student(
                masked,
                history_embeddings=history_embeddings,
                history_present=history_present,
            )
            subset_output = student(
                subset,
                history_embeddings=history_embeddings,
                history_present=history_present,
            )
            with torch.no_grad():
                teacher_output = teacher(batch)
            reconstruction = masked_patch_reconstruction_loss(
                reconstruction_head(masked_output.latent),
                batch.values,
                artificial_mask,
                patch_size=student.patch_size,
            )
            consistency = sensor_set_consistency_loss(output.pooled, subset_output.pooled)
            preservation = preservation_loss(output.pooled, teacher_output.pooled)
            zero = supervised.new_zeros(())
            ssl = combine_losses(
                reconstruction=reconstruction,
                sensor_consistency=consistency,
                preservation=preservation,
                trajectory=zero,
            )
            total = args.supervised_weight * supervised + args.self_supervised_weight * ssl.total
        if not bool(torch.isfinite(total)):
            raise FloatingPointError(f"non-finite loss at step {step}")
        total.backward()
        gradient_norm = clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        record = {
            "step": step,
            "total": float(total.detach().float().cpu()),
            "supervised": float(supervised.detach().float().cpu()),
            "reconstruction": float(reconstruction.detach().float().cpu()),
            "sensor_consistency": float(consistency.detach().float().cpu()),
            "preservation": float(preservation.detach().float().cpu()),
            "gradient_norm": float(gradient_norm.detach().float().cpu()),
        }
        history.append(record)
        print(json.dumps(record), flush=True)

        if step % args.save_every == 0 or step == args.max_steps:
            payload = {
                "format_version": 1,
                "model": "FemMHC",
                "stage": "mcphases_history_conditioned_internal_adapter",
                "status": "complete" if step == args.max_steps else "running",
                "source_checkpoint": str(args.checkpoint.resolve()),
                "femmhc_initialization": str(args.femmhc_init.resolve()),
                "history_embeddings": str(args.history_embeddings.resolve()),
                "seed": args.seed,
                "self_supervised_weight": args.self_supervised_weight,
                "supervised_weight": args.supervised_weight,
                "onset_loss_weight": args.onset_loss_weight,
                "internal_adapter_rank": rank,
                "internal_adapter_layers": layers,
                "history_conditioned_internal_adapters": True,
                "history_context_dim": args.history_context_dim,
                "history_days": args.history_days,
                "history_cycle_modes": args.history_cycle_modes,
                "history_embedding_dim": int(dataset.history_embeddings.shape[1]),
                "minimum_history_days": args.minimum_history_days,
                "trainable_parameters": sum(parameter.numel() for parameter in trainable),
                "frozen_static_adapter": True,
                "task_head_version": "v2",
                "task_group": "female_six",
                "steps": step,
                "max_steps": args.max_steps,
                "history": history,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_gpu_memory_gb": (
                    torch.cuda.max_memory_allocated(device) / 1024**3
                    if device.type == "cuda"
                    else None
                ),
                "student_state_dict": student.state_dict(),
                "reconstruction_head_state_dict": reconstruction_head.state_dict(),
                "task_heads_state_dict": task_heads.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                **capture_rng_state(),
            }
            save_training_checkpoint(args.output, payload)
            print(json.dumps({"event": "checkpoint", "step": step}), flush=True)


if __name__ == "__main__":
    main()
