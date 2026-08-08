"""OpenMHC-initialized variable-sensor encoder for FemMHC."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from openmhc.models.lsm2.positional import get_1d_sincos_pos_embed_from_grid

from .cyclessm import CycleSSMEncoder
from .sensors import SensorBatch, SensorMetadataEncoder


def build_patch_missing_mask(
    batch: SensorBatch,
    *,
    patch_size: int,
    min_observed_fraction: float,
) -> torch.Tensor:
    """Return a boolean ``(B, C*P)`` mask for missing or padded patches."""

    batch.validate()
    if not 0 < min_observed_fraction <= 1:
        raise ValueError("min_observed_fraction must be in (0, 1]")
    values = batch.values
    batch_size, channels, minutes = values.shape
    if minutes % patch_size:
        raise ValueError(f"{minutes=} must be divisible by {patch_size=}")
    patches = values.reshape(batch_size, channels, minutes // patch_size, patch_size)
    observed = torch.isfinite(patches).float().mean(dim=-1)
    missing = observed < min_observed_fraction
    missing = missing | ~batch.present_mask().unsqueeze(-1)
    return missing.reshape(batch_size, -1)


def _time_position_embedding(
    *,
    channels: int,
    patches_per_channel: int,
    embed_dim: int,
) -> torch.Tensor:
    time_positions = np.tile(
        np.arange(patches_per_channel, dtype=np.float32),
        channels,
    )
    values = get_1d_sincos_pos_embed_from_grid(embed_dim, time_positions)
    return torch.from_numpy(values).float().unsqueeze(0)


OPENMHC_SENSOR_ANCHORS: dict[str, int] = {
    # OpenMHC channel indices: watch_steps=3, watch_hr=5, sleep_asleep=7.
    # Novel female-wearable channels intentionally have no numeric-channel anchor.
    "steps": 3,
    "heart_rate": 5,
    "sleep_state": 7,
    "iphone_steps": 0,
    "iphone_distance": 1,
    "iphone_flights": 2,
    "watch_steps": 3,
    "watch_distance": 4,
    "watch_hr": 5,
    "watch_energy": 6,
    "sleep_asleep": 7,
    "sleep_inbed": 8,
    "workout_walking": 9,
    "workout_cycling": 10,
    "workout_running": 11,
    "workout_other": 12,
    "workout_mixed_cardio": 13,
    "workout_strength": 14,
    "workout_elliptical": 15,
    "workout_hiit": 16,
    "workout_functional": 17,
    "workout_yoga": 18,
}


class LowRankAdapter(nn.Module):
    """A residual bottleneck initialized as an identity mapping."""

    def __init__(self, embed_dim: int, rank: int, dropout: float = 0.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("adapter rank must be positive")
        self.normalizer = nn.LayerNorm(embed_dim)
        self.down = nn.Linear(embed_dim, rank, bias=False)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(rank, embed_dim, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.dropout(self.activation(self.down(self.normalizer(x)))))


class TransformerResidualAdapter(nn.Module):
    """Zero-initialized bottleneck inserted after an OpenMHC LSM2 block.

    When ``history_context_dim`` is enabled, the residual update is modulated
    by a strictly causal personal-history state.  The gate is initialized to
    exactly one, so loading a pre-existing static adapter checkpoint preserves
    its behavior before history-conditioned fine-tuning starts.
    """

    def __init__(
        self,
        embed_dim: int,
        rank: int,
        dropout: float = 0.0,
        *,
        history_context_dim: int = 0,
        history_gate_range: float = 0.25,
    ) -> None:
        super().__init__()
        if embed_dim <= 0 or rank <= 0:
            raise ValueError("embed_dim and rank must be positive")
        if history_context_dim < 0:
            raise ValueError("history_context_dim must be non-negative")
        if not 0 < history_gate_range <= 1:
            raise ValueError("history_gate_range must be in (0, 1]")
        self.normalizer = nn.LayerNorm(embed_dim)
        self.down = nn.Linear(embed_dim, rank, bias=False)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(rank, embed_dim, bias=False)
        self.history_context_dim = int(history_context_dim)
        self.history_gate_range = float(history_gate_range)
        self.history_gate = (
            nn.Sequential(
                nn.LayerNorm(self.history_context_dim),
                nn.Linear(self.history_context_dim, embed_dim),
            )
            if self.history_context_dim
            else None
        )
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)
        if self.history_gate is not None:
            # tanh(0) = 0, therefore 1 + range * tanh(.) starts as exactly 1.
            nn.init.zeros_(self.history_gate[-1].weight)
            nn.init.zeros_(self.history_gate[-1].bias)

    def forward(
        self,
        tokens: torch.Tensor,
        history_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        delta = self.up(
            self.dropout(self.activation(self.down(self.normalizer(tokens))))
        )
        if history_context is not None:
            if self.history_gate is None:
                raise ValueError("history_context was supplied to a static adapter")
            if history_context.shape != (tokens.shape[0], self.history_context_dim):
                raise ValueError(
                    "history_context must have shape "
                    f"({tokens.shape[0]}, {self.history_context_dim})"
                )
            gate = 1.0 + self.history_gate_range * torch.tanh(
                self.history_gate(history_context)
            )
            delta = delta * gate.unsqueeze(1)
        return tokens + delta


class PhysiologicalRegimeAdapterBank(nn.Module):
    """Softly route tokens across unlabeled physiological-regime experts."""

    def __init__(
        self,
        embed_dim: int,
        *,
        n_experts: int = 3,
        rank: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_experts <= 0:
            raise ValueError("n_experts must be positive")
        self.experts = nn.ModuleList(
            LowRankAdapter(embed_dim, rank, dropout) for _ in range(n_experts)
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, n_experts),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        pooled: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.gate(pooled), dim=-1)
        deltas = torch.stack([expert.delta(tokens) for expert in self.experts], dim=1)
        mixed = torch.sum(deltas * weights[:, :, None, None], dim=1)
        return tokens + mixed, weights


@dataclass(frozen=True)
class FemMHCOutput:
    pooled: torch.Tensor
    latent: torch.Tensor
    patch_missing_mask: torch.Tensor
    adapter_weights: torch.Tensor
    history_context: torch.Tensor | None = None


@dataclass(frozen=True)
class FemMHCDualOutput:
    """Source-preserving and female-adapted views of the same sensor day."""

    pooled: torch.Tensor
    native_pooled: torch.Tensor
    adapted: FemMHCOutput
    native_available: torch.Tensor


class FemMHCEncoder(nn.Module):
    """Variable-sensor encoder initialized from an OpenMHC LSM2 checkpoint.

    The OpenMHC patch projection is shared across channels.  FemMHC reuses it
    and the pretrained Transformer, replaces device-specific channel indices
    with semantic sensor descriptors, and adds a small gated adapter bank.
    """

    def __init__(
        self,
        pretrained_lsm2: nn.Module,
        *,
        min_observed_fraction: float = 0.5,
        adapter_rank: int = 32,
        n_adapter_experts: int = 3,
        internal_adapter_rank: int = 0,
        internal_adapter_layers: int = 0,
        history_conditioned_internal_adapters: bool = False,
        history_context_dim: int = 96,
        history_maximum_days: int = 60,
        history_cycle_modes: int = 8,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.seq_length = int(pretrained_lsm2.seq_length)
        self.patch_size = int(pretrained_lsm2.patch_size)
        self.embed_dim = int(pretrained_lsm2.embed_dim)
        self.native_in_channels = int(pretrained_lsm2.in_channels)
        self.patches_per_channel = self.seq_length // self.patch_size
        self.min_observed_fraction = float(min_observed_fraction)
        self.internal_adapter_rank = int(internal_adapter_rank)
        self.internal_adapter_layers = int(internal_adapter_layers)
        self.history_conditioned_internal_adapters = bool(
            history_conditioned_internal_adapters
        )
        self.history_context_dim = int(history_context_dim)
        self.history_maximum_days = int(history_maximum_days)

        self.patch_projection = copy.deepcopy(pretrained_lsm2.patch_embed.proj)
        self.encoder = copy.deepcopy(pretrained_lsm2.encoder)
        if self.internal_adapter_rank < 0 or self.internal_adapter_layers < 0:
            raise ValueError("internal adapter settings must be non-negative")
        depth = len(self.encoder.blocks)
        if self.internal_adapter_layers > depth:
            raise ValueError(
                f"internal_adapter_layers={self.internal_adapter_layers} exceeds encoder depth {depth}"
            )
        if self.internal_adapter_layers and not self.internal_adapter_rank:
            raise ValueError("internal_adapter_rank is required when layers are enabled")
        if self.history_conditioned_internal_adapters and not self.internal_adapter_layers:
            raise ValueError(
                "history-conditioned adapters require at least one internal adapter layer"
            )
        if self.history_conditioned_internal_adapters and self.history_context_dim <= 0:
            raise ValueError("history_context_dim must be positive when history conditioning is enabled")
        if self.history_conditioned_internal_adapters and self.history_maximum_days <= 0:
            raise ValueError("history_maximum_days must be positive")
        selected_layers = tuple(
            range(max(0, depth - self.internal_adapter_layers), depth)
        )
        self.internal_adapter_indices = selected_layers
        self.internal_adapters = nn.ModuleDict(
            {
                str(index): TransformerResidualAdapter(
                    self.embed_dim,
                    self.internal_adapter_rank,
                    history_context_dim=(
                        self.history_context_dim
                        if self.history_conditioned_internal_adapters
                        else 0
                    ),
                )
                for index in selected_layers
            }
        )
        self.history_encoder = (
            CycleSSMEncoder(
                self.embed_dim,
                hidden_dim=self.history_context_dim,
                modes=history_cycle_modes,
                maximum_days=self.history_maximum_days,
            )
            if self.history_conditioned_internal_adapters
            else None
        )
        # Keep OpenMHC's original positional tensor outside checkpoints.  It is
        # deterministic, is restored whenever an OpenMHC backbone is built,
        # and using the source tensor here makes the native branch numerically
        # identical instead of relying on a second reconstruction of sin/cos.
        self.register_buffer(
            "_native_position_embedding",
            pretrained_lsm2.pos_embed.detach().clone(),
            persistent=False,
        )
        if self.embed_dim % 2:
            raise ValueError("OpenMHC positional transfer requires an even embedding dimension")
        self.position_dim = self.embed_dim // 2
        self.sensor_metadata = SensorMetadataEncoder(self.position_dim)
        # Start close to OpenMHC's original channel position for shared sensors,
        # while allowing semantics to replace the arbitrary device channel ID.
        # Known OpenMHC sensors start on the native channel-position manifold
        # (sigmoid(-8) ~= 0.0003), while novel sensors still use their semantic
        # embedding directly.  Training may relax this gate when semantics help.
        self.anchor_mix_logit = nn.Parameter(torch.tensor(-8.0))
        self.adapters = PhysiologicalRegimeAdapterBank(
            self.embed_dim,
            n_experts=n_adapter_experts,
            rank=adapter_rank,
            dropout=0.1,
        )
        self.set_backbone_trainable(not freeze_backbone)

    def _channel_position_embedding(self, batch: SensorBatch) -> torch.Tensor:
        semantic = self.sensor_metadata(batch.descriptors)
        anchor_indices = [OPENMHC_SENSOR_ANCHORS.get(item.name) for item in batch.descriptors]
        known = torch.tensor(
            [item is not None for item in anchor_indices],
            device=semantic.device,
            dtype=torch.bool,
        )
        if not bool(known.any()):
            return semantic
        grid = np.asarray([item or 0 for item in anchor_indices], dtype=np.float32)
        anchors = torch.from_numpy(
            get_1d_sincos_pos_embed_from_grid(self.position_dim, grid)
        ).to(device=semantic.device, dtype=semantic.dtype)
        mixture = torch.sigmoid(self.anchor_mix_logit)
        blended = anchors + mixture * (semantic - anchors)
        return torch.where(known.unsqueeze(-1), blended, semantic)

    def _native_dense_values(self, batch: SensorBatch) -> torch.Tensor:
        """Restore registered source sensors to OpenMHC's fixed channel order."""

        anchor_indices = [OPENMHC_SENSOR_ANCHORS.get(item.name) for item in batch.descriptors]
        if any(index is None for index in anchor_indices):
            unknown = [
                descriptor.name
                for descriptor, index in zip(batch.descriptors, anchor_indices)
                if index is None
            ]
            raise ValueError(
                "native OpenMHC branch only supports registered source sensors: "
                + ", ".join(unknown)
            )
        if len(set(anchor_indices)) != len(anchor_indices):
            raise ValueError("native OpenMHC branch received duplicate channel anchors")
        if any(index >= self.native_in_channels for index in anchor_indices):
            raise ValueError("native OpenMHC channel anchor is outside the source model")

        batch_size, _, minutes = batch.values.shape
        dense = torch.full(
            (batch_size, self.native_in_channels, minutes),
            torch.nan,
            device=batch.values.device,
            dtype=batch.values.dtype,
        )
        present = batch.present_mask()
        for local_index, native_index in enumerate(anchor_indices):
            source = batch.values[:, local_index]
            source = torch.where(
                present[:, local_index, None],
                source,
                torch.full_like(source, torch.nan),
            )
            dense[:, native_index] = source
        return dense

    def _native_patch_missing_mask(self, values: torch.Tensor) -> torch.Tensor:
        """Reproduce OpenMHC's inherited missingness mask exactly."""

        if self.native_in_channels <= 5:
            raise ValueError("native OpenMHC branch requires heart-rate channel index 5")
        patches = values.reshape(
            values.shape[0],
            self.native_in_channels,
            self.patches_per_channel,
            self.patch_size,
        )
        missing = torch.isnan(patches).any(dim=-1)
        # OpenMHC deliberately treats sparse HR differently from every other
        # channel: a HR patch is missing only when all samples are absent.
        missing[:, 5] = torch.isnan(patches[:, 5]).all(dim=-1)
        return missing.reshape(values.shape[0], -1)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.patch_projection.parameters():
            parameter.requires_grad = trainable
        for parameter in self.encoder.parameters():
            parameter.requires_grad = trainable

    def _encode_adapted(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        history_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the OpenMHC encoder with optional internal residual adapters."""

        for index, block in enumerate(self.encoder.blocks):
            tokens = block(tokens, attention_mask)
            key = str(index)
            if key in self.internal_adapters:
                tokens = self.internal_adapters[key](tokens, history_context)
        return self.encoder.norm(tokens)

    def _history_context(
        self,
        history_embeddings: torch.Tensor | None,
        history_present: torch.Tensor | None,
        *,
        batch_size: int,
    ) -> torch.Tensor | None:
        """Encode only days strictly preceding the current wearable day.

        The caller owns the temporal contract: ``history_embeddings`` must
        exclude the current day.  Rows without prior history receive a zero
        context, which keeps the initialized residual gate equal to its static
        behavior and provides a well-defined cold-start path.
        """

        if history_embeddings is None and history_present is None:
            return None
        if history_embeddings is None or history_present is None:
            raise ValueError("history_embeddings and history_present must be supplied together")
        if self.history_encoder is None:
            raise ValueError("history was supplied but history conditioning is disabled")
        if history_embeddings.ndim != 3:
            raise ValueError("history_embeddings must have shape (batch, days, embed_dim)")
        if history_embeddings.shape[0] != batch_size or history_embeddings.shape[2] != self.embed_dim:
            raise ValueError(
                "history_embeddings must match the current batch and encoder embedding dimension"
            )
        if history_present.shape != history_embeddings.shape[:2]:
            raise ValueError("history_present must have shape (batch, days)")
        if history_embeddings.shape[1] > self.history_maximum_days:
            raise ValueError("history exceeds history_maximum_days")
        present = history_present.to(
            device=history_embeddings.device,
            dtype=torch.bool,
        )
        valid = present.any(dim=1)
        context = torch.zeros(
            batch_size,
            self.history_context_dim,
            device=history_embeddings.device,
            dtype=history_embeddings.dtype,
        )
        if bool(valid.any()):
            encoded = self.history_encoder(
                history_embeddings[valid],
                present[valid],
            ).representation
            context[valid] = encoded.to(context.dtype)
        return context

    def _patch_tokens(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, channels, minutes = values.shape
        clean = torch.nan_to_num(values, nan=0.0)
        tokens = self.patch_projection(clean.reshape(batch_size * channels, 1, minutes))
        tokens = tokens.reshape(
            batch_size,
            channels,
            self.embed_dim,
            self.patches_per_channel,
        )
        return tokens.permute(0, 1, 3, 2).reshape(batch_size, -1, self.embed_dim)

    def forward(
        self,
        batch: SensorBatch,
        *,
        history_embeddings: torch.Tensor | None = None,
        history_present: torch.Tensor | None = None,
    ) -> FemMHCOutput:
        batch.validate()
        values = batch.values
        batch_size, channels, minutes = values.shape
        if minutes != self.seq_length:
            raise ValueError(f"expected {self.seq_length} samples per day, got {minutes}")

        missing = build_patch_missing_mask(
            batch,
            patch_size=self.patch_size,
            min_observed_fraction=self.min_observed_fraction,
        )
        empty = (~missing).sum(dim=1) == 0
        if bool(empty.any()):
            indices = empty.nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(f"samples have no usable sensor patches: {indices}")

        tokens = self._patch_tokens(values)
        sensor_embed = self._channel_position_embedding(batch)
        sensor_embed = sensor_embed[:, None, :].expand(
            channels,
            self.patches_per_channel,
            self.position_dim,
        )
        time_embed = _time_position_embedding(
            channels=channels,
            patches_per_channel=self.patches_per_channel,
            embed_dim=self.position_dim,
        ).to(device=tokens.device, dtype=tokens.dtype)
        position_embed = torch.cat(
            [sensor_embed.reshape(1, -1, self.position_dim), time_embed],
            dim=-1,
        )
        tokens = tokens + position_embed

        attention_mask = torch.zeros(
            batch_size,
            1,
            1,
            tokens.shape[1],
            device=tokens.device,
            dtype=tokens.dtype,
        )
        attention_mask.masked_fill_(missing[:, None, None, :], float("-inf"))
        history_context = self._history_context(
            history_embeddings,
            history_present,
            batch_size=batch_size,
        )
        latent = self._encode_adapted(tokens, attention_mask, history_context)

        observed = (~missing).to(latent.dtype).unsqueeze(-1)
        pooled = (latent * observed).sum(dim=1) / observed.sum(dim=1).clamp_min(1.0)
        latent, adapter_weights = self.adapters(latent, pooled)
        pooled = (latent * observed).sum(dim=1) / observed.sum(dim=1).clamp_min(1.0)
        return FemMHCOutput(
            pooled=pooled,
            latent=latent,
            patch_missing_mask=missing,
            adapter_weights=adapter_weights,
            history_context=history_context,
        )

    def forward_native(self, batch: SensorBatch) -> FemMHCOutput:
        """Encode on a frozen, non-interfering OpenMHC-compatible branch.

        This path uses exact source channel positions and bypasses every FemMHC
        adapter.  Because the shared patch projection and Transformer remain
        frozen, registered OpenMHC inputs retain their original representation.
        """

        batch.validate()
        batch_size, _, minutes = batch.values.shape
        if minutes != self.seq_length:
            raise ValueError(f"expected {self.seq_length} samples per day, got {minutes}")
        values = self._native_dense_values(batch)
        missing = self._native_patch_missing_mask(values)
        empty = (~missing).sum(dim=1) == 0
        if bool(empty.any()):
            indices = empty.nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(f"samples have no usable sensor patches: {indices}")

        tokens = self._patch_tokens(values)
        tokens = tokens + self._native_position_embedding.to(
            device=tokens.device,
            dtype=tokens.dtype,
        )
        attention_mask = torch.zeros(
            batch_size,
            1,
            1,
            tokens.shape[1],
            device=tokens.device,
            dtype=tokens.dtype,
        )
        attention_mask.masked_fill_(missing[:, None, None, :], float("-inf"))
        latent = self.encoder(tokens, attn_mask=attention_mask)
        observed = (~missing).to(latent.dtype).unsqueeze(-1)
        pooled = (latent * observed).sum(dim=1) / observed.sum(dim=1).clamp_min(1.0)
        adapter_weights = torch.zeros(
            batch_size,
            len(self.adapters.experts),
            device=pooled.device,
            dtype=pooled.dtype,
        )
        return FemMHCOutput(
            pooled=pooled,
            latent=latent,
            patch_missing_mask=missing,
            adapter_weights=adapter_weights,
            history_context=None,
        )

    def forward_dual(
        self,
        batch: SensorBatch,
        *,
        history_embeddings: torch.Tensor | None = None,
        history_present: torch.Tensor | None = None,
    ) -> FemMHCDualOutput:
        """Concatenate native source evidence with the adapted representation.

        Novel sensors are consumed by the adapted path.  The native path sees
        only descriptors that can be placed on OpenMHC's original channel
        grid; samples without a usable source-compatible patch receive a zero
        native view and are marked by ``native_available``.
        """

        batch.validate()
        adapted = self(
            batch,
            history_embeddings=history_embeddings,
            history_present=history_present,
        )
        source_indices = [
            index
            for index, descriptor in enumerate(batch.descriptors)
            if descriptor.name in OPENMHC_SENSOR_ANCHORS
        ]
        native_pooled = torch.zeros_like(adapted.pooled)
        native_available = torch.zeros(
            batch.values.shape[0],
            dtype=torch.bool,
            device=batch.values.device,
        )
        if source_indices:
            source_batch = SensorBatch(
                batch.values[:, source_indices],
                tuple(batch.descriptors[index] for index in source_indices),
                (
                    batch.present_mask()[:, source_indices]
                    if batch.channel_present is not None
                    else None
                ),
            )
            dense = self._native_dense_values(source_batch)
            missing = self._native_patch_missing_mask(dense)
            native_available = (~missing).any(dim=1)
            if bool(native_available.any()):
                usable_batch = SensorBatch(
                    source_batch.values[native_available],
                    source_batch.descriptors,
                    (
                        source_batch.present_mask()[native_available]
                        if source_batch.channel_present is not None
                        else None
                    ),
                )
                native = self.forward_native(usable_batch).pooled
                indices = native_available.nonzero(as_tuple=False).flatten()
                native_pooled = native_pooled.index_copy(0, indices, native)
        return FemMHCDualOutput(
            pooled=torch.cat([native_pooled, adapted.pooled], dim=-1),
            native_pooled=native_pooled,
            adapted=adapted,
            native_available=native_available,
        )


class PatchReconstructionHead(nn.Module):
    """Decode each FemMHC token back to one normalized sensor patch."""

    def __init__(self, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.normalizer = nn.LayerNorm(embed_dim)
        self.projection = nn.Linear(embed_dim, patch_size)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.projection(self.normalizer(latent))


# Backward-compatible name for early experiment checkpoints.
LifecycleAdapterBank = PhysiologicalRegimeAdapterBank
