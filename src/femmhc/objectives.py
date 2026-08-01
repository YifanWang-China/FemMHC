"""Self-supervised and specialization objectives used by FemMHC."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .sensors import SensorBatch


def sensor_set_consistency_loss(
    full_embedding: torch.Tensor,
    subset_embedding: torch.Tensor,
) -> torch.Tensor:
    """Cosine distance between two sensor-set views of the same recording."""

    if full_embedding.shape != subset_embedding.shape:
        raise ValueError("sensor-set views must have identical embedding shapes")
    distance = 1.0 - F.cosine_similarity(full_embedding, subset_embedding, dim=-1)
    return distance.clamp_min(0.0).mean()


def preservation_loss(
    student_embedding: torch.Tensor,
    teacher_embedding: torch.Tensor,
) -> torch.Tensor:
    """Preserve the frozen general OpenMHC representation during specialization."""

    if student_embedding.shape != teacher_embedding.shape:
        raise ValueError("student and teacher embeddings must have identical shapes")
    student = F.normalize(student_embedding, dim=-1)
    teacher = F.normalize(teacher_embedding.detach(), dim=-1)
    return F.smooth_l1_loss(student, teacher)


class TemporalOrderHead(nn.Module):
    """Predict whether the second representation occurs after the first."""

    def __init__(self, embed_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        if first.shape != second.shape:
            raise ValueError("temporal pair embeddings must have identical shapes")
        features = torch.cat([first, second, second - first], dim=-1)
        return self.network(features).squeeze(-1)


def temporal_order_loss(
    head: TemporalOrderHead,
    first: torch.Tensor,
    second: torch.Tensor,
    second_is_later: torch.Tensor,
) -> torch.Tensor:
    logits = head(first, second)
    target = second_is_later.to(device=logits.device, dtype=logits.dtype)
    if target.shape != logits.shape:
        raise ValueError(f"expected temporal labels shaped {tuple(logits.shape)}")
    return F.binary_cross_entropy_with_logits(logits, target)


def masked_patch_reconstruction_loss(
    prediction: torch.Tensor,
    target_values: torch.Tensor,
    artificial_patch_mask: torch.Tensor,
    *,
    patch_size: int,
) -> torch.Tensor:
    """MSE on artificially hidden patches that existed in the source data."""

    if target_values.ndim != 3:
        raise ValueError("target_values must have shape (B,C,L)")
    batch, channels, minutes = target_values.shape
    if minutes % patch_size:
        raise ValueError("target length must be divisible by patch_size")
    target = target_values.reshape(batch, channels, minutes // patch_size, patch_size)
    target = target.reshape(batch, -1, patch_size)
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction shape {tuple(prediction.shape)} does not match {tuple(target.shape)}"
        )
    if artificial_patch_mask.shape != target.shape[:2]:
        raise ValueError("artificial_patch_mask has the wrong shape")
    originally_observed = torch.isfinite(target).all(dim=-1)
    selected = artificial_patch_mask.bool() & originally_observed
    if not bool(selected.any()):
        return prediction.sum() * 0.0
    return F.mse_loss(prediction[selected], target[selected])


def drop_sensor_channels(
    batch: SensorBatch,
    *,
    drop_probability: float = 0.35,
    generator: torch.Generator | None = None,
    patch_size: int | None = None,
    min_observed_fraction: float = 0.5,
) -> SensorBatch:
    """Create a sensor-subset view while retaining at least one channel per sample."""

    batch.validate()
    if not 0 <= drop_probability < 1:
        raise ValueError("drop_probability must be in [0, 1)")
    present = batch.present_mask().clone()
    if patch_size is not None:
        if batch.values.shape[-1] % patch_size:
            raise ValueError("sensor length must be divisible by patch_size")
        patches = batch.values.reshape(
            batch.values.shape[0],
            batch.values.shape[1],
            -1,
            patch_size,
        )
        usable = torch.isfinite(patches).float().mean(dim=-1)
        usable = usable.ge(min_observed_fraction).any(dim=-1)
        present = present & usable
    random_values = torch.rand(
        present.shape,
        device=present.device,
        generator=generator,
    )
    retained = present & (random_values >= drop_probability)
    for row in range(retained.shape[0]):
        if not bool(retained[row].any()):
            candidates = present[row].nonzero(as_tuple=False).flatten()
            if candidates.numel() == 0:
                continue
            retained[row, candidates[0]] = True
    values = batch.values.clone()
    values = values.masked_fill(~retained.unsqueeze(-1), torch.nan)
    return SensorBatch(values, batch.descriptors, retained)


def mask_sensor_patches(
    batch: SensorBatch,
    *,
    patch_size: int,
    mask_probability: float = 0.15,
    generator: torch.Generator | None = None,
) -> tuple[SensorBatch, torch.Tensor]:
    """Hide fully observed patches and return the exact reconstruction target mask."""

    batch.validate()
    if not 0 < mask_probability < 1:
        raise ValueError("mask_probability must be in (0, 1)")
    batch_size, channels, minutes = batch.values.shape
    if minutes % patch_size:
        raise ValueError("sensor length must be divisible by patch_size")
    patches = batch.values.reshape(
        batch_size,
        channels,
        minutes // patch_size,
        patch_size,
    )
    eligible = torch.isfinite(patches).all(dim=-1)
    eligible = eligible & batch.present_mask().unsqueeze(-1)
    selected = eligible & (
        torch.rand(eligible.shape, device=eligible.device, generator=generator)
        < mask_probability
    )
    for row in range(batch_size):
        eligible_count = int(eligible[row].sum())
        if eligible_count > 1 and not bool(selected[row].any()):
            first = eligible[row].nonzero(as_tuple=False)[0]
            selected[row, first[0], first[1]] = True
        if eligible_count > 0 and int(selected[row].sum()) >= eligible_count:
            retained = selected[row].nonzero(as_tuple=False)[0]
            selected[row, retained[0], retained[1]] = False
    masked_patches = patches.clone().masked_fill(selected.unsqueeze(-1), torch.nan)
    return (
        SensorBatch(
            masked_patches.reshape_as(batch.values),
            batch.descriptors,
            batch.channel_present,
        ),
        selected.reshape(batch_size, -1),
    )


@dataclass(frozen=True)
class FemMHCLosses:
    total: torch.Tensor
    reconstruction: torch.Tensor
    sensor_consistency: torch.Tensor
    preservation: torch.Tensor
    trajectory: torch.Tensor


def combine_losses(
    *,
    reconstruction: torch.Tensor,
    sensor_consistency: torch.Tensor,
    preservation: torch.Tensor,
    trajectory: torch.Tensor,
    consistency_weight: float = 1.0,
    preservation_weight: float = 1.0,
    trajectory_weight: float = 0.5,
) -> FemMHCLosses:
    total = (
        reconstruction
        + consistency_weight * sensor_consistency
        + preservation_weight * preservation
        + trajectory_weight * trajectory
    )
    return FemMHCLosses(
        total=total,
        reconstruction=reconstruction,
        sensor_consistency=sensor_consistency,
        preservation=preservation,
        trajectory=trajectory,
    )
