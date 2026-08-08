"""Causal personal memory and oscillatory state-space models for menstrual wearables."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
from torch import nn


@dataclass(frozen=True)
class CausalMemoryOutput:
    memory_states: torch.Tensor
    deviations: torch.Tensor
    retention: torch.Tensor


@dataclass(frozen=True)
class TemporalRepresentationOutput:
    representation: torch.Tensor
    sequence_states: torch.Tensor
    auxiliary: dict[str, torch.Tensor] = field(default_factory=dict)


def _validate_sequence(
    values: torch.Tensor,
    day_present: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 3:
        raise ValueError("daily_embeddings must have shape (batch, days, embed_dim)")
    batch, days, _ = values.shape
    if day_present is None:
        present = torch.ones(batch, days, dtype=torch.bool, device=values.device)
    else:
        if day_present.shape != (batch, days):
            raise ValueError("day_present must have shape (batch, days)")
        present = day_present.to(device=values.device, dtype=torch.bool)
    if not bool(present.any(dim=1).all()):
        raise ValueError("each history needs at least one observed day")
    clean = torch.where(present.unsqueeze(-1), values, torch.zeros_like(values))
    return clean, present


def _last_observed(values: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(values.shape[1], device=values.device).unsqueeze(0)
    last = positions.masked_fill(~present, -1).max(dim=1).values
    return values[torch.arange(values.shape[0], device=values.device), last]


class PersonalCausalMemory(nn.Module):
    """Online personal baseline that never reads a future day.

    The representation for day ``t`` is compared with memory from days strictly
    before ``t``.  The current observation updates memory only after its
    deviation has been emitted.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        self.initial_memory = nn.Parameter(torch.zeros(embed_dim))
        self.value = nn.Linear(embed_dim, embed_dim)
        self.retention = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        self.deviation_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        daily_embeddings: torch.Tensor,
        day_present: torch.Tensor | None = None,
    ) -> CausalMemoryOutput:
        values, present = _validate_sequence(daily_embeddings, day_present)
        batch, days, embed_dim = values.shape
        memory = self.initial_memory.unsqueeze(0).expand(batch, -1)
        seen = torch.zeros(batch, dtype=torch.bool, device=values.device)
        memories = []
        deviations = []
        retentions = []
        for index in range(days):
            current = values[:, index]
            observed = present[:, index]
            # The first observed day initializes a participant-specific level;
            # later days are measured against strictly historical memory.
            reference = torch.where(seen.unsqueeze(-1), memory, current)
            deviation = self.deviation_norm(current - reference)
            keep = self.retention(torch.cat([current, reference], dim=-1))
            candidate = keep * reference + (1.0 - keep) * self.value(current)
            memory = torch.where(observed.unsqueeze(-1), candidate, memory)
            seen = seen | observed
            memories.append(reference)
            deviations.append(
                torch.where(observed.unsqueeze(-1), deviation, torch.zeros_like(deviation))
            )
            retentions.append(
                torch.where(observed.unsqueeze(-1), keep, torch.ones_like(keep))
            )
        return CausalMemoryOutput(
            memory_states=torch.stack(memories, dim=1),
            deviations=torch.stack(deviations, dim=1),
            retention=torch.stack(retentions, dim=1),
        )


class CyclicStateSpaceLayer(nn.Module):
    """Stable complex-diagonal SSM with learnable menstrual-range periods."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        modes: int = 8,
        minimum_period_days: float = 18.0,
        maximum_period_days: float = 45.0,
        maximum_velocity_change: float = 0.15,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or modes <= 0 or hidden_dim % (2 * modes):
            raise ValueError("hidden_dim must be divisible by 2 * modes")
        if not 0 < minimum_period_days < maximum_period_days:
            raise ValueError("period bounds must satisfy 0 < minimum < maximum")
        self.hidden_dim = int(hidden_dim)
        self.modes = int(modes)
        self.mode_width = hidden_dim // (2 * modes)
        self.minimum_period_days = float(minimum_period_days)
        self.maximum_period_days = float(maximum_period_days)
        self.maximum_velocity_change = float(maximum_velocity_change)

        initial_periods = torch.linspace(21.0, 38.0, modes).clamp(
            minimum_period_days + 1e-3,
            maximum_period_days - 1e-3,
        )
        scaled = (initial_periods - minimum_period_days) / (
            maximum_period_days - minimum_period_days
        )
        self.period_logits = nn.Parameter(torch.logit(scaled))
        self.decay_logits = nn.Parameter(torch.full((modes,), math.log(0.98 / 0.02)))
        self.input_projection = nn.Linear(hidden_dim, hidden_dim)
        self.input_gate = nn.Linear(hidden_dim, hidden_dim)
        self.decay_adjustment = nn.Linear(hidden_dim, modes)
        self.velocity = nn.Linear(hidden_dim, modes)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))

    @property
    def periods(self) -> torch.Tensor:
        return self.minimum_period_days + (
            self.maximum_period_days - self.minimum_period_days
        ) * torch.sigmoid(self.period_logits)

    def forward(
        self,
        inputs: torch.Tensor,
        day_present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3 or day_present.shape != inputs.shape[:2]:
            raise ValueError("inputs/day_present have incompatible shapes")
        batch, days, _ = inputs.shape
        state = torch.zeros(
            batch,
            self.modes,
            2,
            self.mode_width,
            device=inputs.device,
            dtype=inputs.dtype,
        )
        base_angular_velocity = (2.0 * math.pi / self.periods).to(dtype=inputs.dtype)
        outputs = []
        velocities = []
        for index in range(days):
            current = inputs[:, index]
            observed = day_present[:, index]
            relative_velocity = self.maximum_velocity_change * torch.tanh(
                self.velocity(current)
            )
            angular_velocity = base_angular_velocity.unsqueeze(0) * (
                1.0 + relative_velocity
            )
            cosine = torch.cos(angular_velocity).unsqueeze(-1)
            sine = torch.sin(angular_velocity).unsqueeze(-1)
            real, imaginary = state[:, :, 0], state[:, :, 1]
            rotated_real = cosine * real - sine * imaginary
            rotated_imaginary = sine * real + cosine * imaginary
            injection = self.input_projection(current).reshape(
                batch,
                self.modes,
                2,
                self.mode_width,
            )
            input_gate = torch.sigmoid(self.input_gate(current)).reshape(
                batch,
                self.modes,
                2,
                self.mode_width,
            )
            injection = input_gate * injection * observed[:, None, None, None]
            decay_shift = 0.5 * torch.tanh(self.decay_adjustment(current))
            decay_shift = decay_shift * observed.unsqueeze(-1)
            decay = torch.sigmoid(self.decay_logits.unsqueeze(0) + decay_shift)
            state = torch.stack([rotated_real, rotated_imaginary], dim=2)
            state = decay[:, :, None, None] * state + injection
            output = self.output_projection(state.reshape(batch, self.hidden_dim))
            output = output + self.skip_scale * current
            outputs.append(output)
            velocities.append(angular_velocity)
        return torch.stack(outputs, dim=1), torch.stack(velocities, dim=1)


class CycleSSMEncoder(nn.Module):
    """Personal baseline deviations drive an oscillatory causal state space."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        *,
        modes: int = 8,
        maximum_days: int = 60,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.maximum_days = int(maximum_days)
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        self.personal_memory = PersonalCausalMemory(hidden_dim)
        self.cycle_ssm = CyclicStateSpaceLayer(hidden_dim, modes=modes)
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        daily_embeddings: torch.Tensor,
        day_present: torch.Tensor | None = None,
    ) -> TemporalRepresentationOutput:
        values, present = _validate_sequence(daily_embeddings, day_present)
        if values.shape[1] > self.maximum_days:
            raise ValueError("history exceeds maximum_days")
        projected = self.input_projection(values)
        projected = torch.where(present.unsqueeze(-1), projected, torch.zeros_like(projected))
        memory = self.personal_memory(projected, present)
        cycle, velocity = self.cycle_ssm(memory.deviations, present)
        sequence = self.output_projection(
            torch.cat(
                [projected, memory.memory_states, memory.deviations, cycle],
                dim=-1,
            )
        )
        sequence = torch.where(present.unsqueeze(-1), sequence, torch.zeros_like(sequence))
        return TemporalRepresentationOutput(
            representation=_last_observed(sequence, present),
            sequence_states=sequence,
            auxiliary={
                "personal_memory": memory.memory_states,
                "personal_deviation": memory.deviations,
                "memory_retention": memory.retention,
                "cycle_velocity": velocity,
                "cycle_period_days": self.cycle_ssm.periods,
            },
        )


class LastDayMLPEncoder(nn.Module):
    """Equal-interface non-temporal control using only the latest observed day."""

    def __init__(self, input_dim: int, hidden_dim: int = 96, **_: object) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        daily_embeddings: torch.Tensor,
        day_present: torch.Tensor | None = None,
    ) -> TemporalRepresentationOutput:
        values, present = _validate_sequence(daily_embeddings, day_present)
        sequence = self.network(values)
        sequence = torch.where(present.unsqueeze(-1), sequence, torch.zeros_like(sequence))
        return TemporalRepresentationOutput(
            representation=_last_observed(sequence, present),
            sequence_states=sequence,
        )


class CausalGRUEncoder(nn.Module):
    """GRU history baseline with a parameter budget close to CycleSSM."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        *,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 3 // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 3 // 2, hidden_dim),
        )

    def forward(
        self,
        daily_embeddings: torch.Tensor,
        day_present: torch.Tensor | None = None,
    ) -> TemporalRepresentationOutput:
        values, present = _validate_sequence(daily_embeddings, day_present)
        projected = self.input_projection(values)
        projected = torch.where(present.unsqueeze(-1), projected, torch.zeros_like(projected))
        sequence, _ = self.gru(projected)
        sequence = self.output_projection(sequence)
        sequence = torch.where(present.unsqueeze(-1), sequence, torch.zeros_like(sequence))
        return TemporalRepresentationOutput(
            representation=_last_observed(sequence, present),
            sequence_states=sequence,
        )


class HistoryConditionedAdapterEncoder(nn.Module):
    """Causal personal-history adapter over daily OpenMHC representations.

    This is the fast gate for the planned FemMHC-HCA model.  It keeps a
    participant-specific baseline and three causal exponential memories.  The
    current day's representation is changed by a bottleneck adapter whose gate
    is generated only from observations up to the previous day.  The class is
    intentionally implemented at the daily-embedding level first so the
    history-conditioned mechanism can be compared cheaply before inserting the
    same adapter inside the LSM2 Transformer blocks.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        *,
        adapter_rank: int | None = None,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        rank = int(adapter_rank or max(8, hidden_dim // 4))
        if rank <= 0:
            raise ValueError("adapter_rank must be positive")
        self.hidden_dim = int(hidden_dim)
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        # These logits are constrained to (0, 1), giving a stable causal EMA
        # rather than an unconstrained recurrent state.
        self.timescale_logits = nn.Parameter(
            torch.tensor([-1.5, 0.5, 2.5], dtype=torch.float32)
        )
        self.baseline_update = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.context_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim * 5),
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.GELU(),
        )
        self.adapter_norm = nn.LayerNorm(hidden_dim)
        self.adapter_down = nn.Linear(hidden_dim, rank, bias=False)
        self.adapter_up = nn.Linear(rank, hidden_dim, bias=False)
        self.adapter_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.output_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        nn.init.normal_(self.adapter_down.weight, std=0.02)
        # Start as an identity over the projected OpenMHC representation.
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.constant_(self.adapter_gate[-2].bias, -4.0)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        with torch.no_grad():
            self.output_projection.weight[:, :hidden_dim].copy_(
                torch.eye(hidden_dim)
            )

    def forward(
        self,
        daily_embeddings: torch.Tensor,
        day_present: torch.Tensor | None = None,
    ) -> TemporalRepresentationOutput:
        values, present = _validate_sequence(daily_embeddings, day_present)
        projected = self.input_projection(values)
        batch, days, hidden = projected.shape
        timescales = torch.sigmoid(self.timescale_logits).to(
            device=projected.device, dtype=projected.dtype
        )
        memories = torch.zeros(
            batch, 3, hidden, device=projected.device, dtype=projected.dtype
        )
        baseline = torch.zeros(
            batch, hidden, device=projected.device, dtype=projected.dtype
        )
        seen = torch.zeros(batch, dtype=torch.bool, device=projected.device)
        sequence = []
        history_states = []
        deviations = []
        gates = []
        for index in range(days):
            current = projected[:, index]
            observed = present[:, index]
            # The context is read before the current observation updates the
            # memories, making the adapter strictly causal.
            reference = torch.where(seen.unsqueeze(-1), baseline, current)
            deviation = self.adapter_norm(current - reference)
            context = self.context_projection(
                torch.cat([current, reference, deviation, memories[:, 0], memories[:, 2]], dim=-1)
            )
            delta = self.adapter_up(
                torch.nn.functional.gelu(self.adapter_down(self.adapter_norm(current)))
            )
            gate = self.adapter_gate(context)
            adapted = self.output_projection(torch.cat([current + gate * delta, context], dim=-1))
            adapted = torch.where(observed.unsqueeze(-1), adapted, torch.zeros_like(adapted))
            sequence.append(adapted)
            history_states.append(context)
            deviations.append(torch.where(observed.unsqueeze(-1), deviation, torch.zeros_like(deviation)))
            gates.append(torch.where(observed.unsqueeze(-1), gate, torch.zeros_like(gate)))

            observed_current = torch.where(observed.unsqueeze(-1), current, baseline)
            keep = self.baseline_update(torch.cat([observed_current, reference], dim=-1))
            baseline_candidate = keep * reference + (1.0 - keep) * observed_current
            baseline = torch.where(observed.unsqueeze(-1), baseline_candidate, baseline)
            memories = timescales.view(1, 3, 1) * memories + (
                1.0 - timescales.view(1, 3, 1)
            ) * observed_current.unsqueeze(1)
            seen = seen | observed

        states = torch.stack(sequence, dim=1)
        states = torch.where(present.unsqueeze(-1), states, torch.zeros_like(states))
        return TemporalRepresentationOutput(
            representation=_last_observed(states, present),
            sequence_states=states,
            auxiliary={
                "personal_history": torch.stack(history_states, dim=1),
                "personal_deviation": torch.stack(deviations, dim=1),
                "adapter_gate": torch.stack(gates, dim=1),
                "timescales": timescales,
            },
        )


class CausalTransformerEncoder(nn.Module):
    """Causally masked Transformer history baseline."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        *,
        maximum_days: int = 60,
        temporal_heads: int = 4,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__()
        if hidden_dim % temporal_heads:
            raise ValueError("hidden_dim must be divisible by temporal_heads")
        self.maximum_days = int(maximum_days)
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        self.day_position = nn.Parameter(torch.zeros(maximum_days, hidden_dim))
        nn.init.normal_(self.day_position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=temporal_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            layer,
            num_layers=1,
            enable_nested_tensor=False,
        )

    def forward(
        self,
        daily_embeddings: torch.Tensor,
        day_present: torch.Tensor | None = None,
    ) -> TemporalRepresentationOutput:
        values, present = _validate_sequence(daily_embeddings, day_present)
        days = values.shape[1]
        if days > self.maximum_days:
            raise ValueError("history exceeds maximum_days")
        projected = self.input_projection(values) + self.day_position[:days].unsqueeze(0)
        projected = torch.where(present.unsqueeze(-1), projected, torch.zeros_like(projected))
        causal_mask = torch.triu(
            torch.ones(days, days, dtype=torch.bool, device=values.device),
            diagonal=1,
        )
        sequence = self.temporal_encoder(
            projected,
            mask=causal_mask,
            src_key_padding_mask=~present,
        )
        sequence = torch.where(present.unsqueeze(-1), sequence, torch.zeros_like(sequence))
        return TemporalRepresentationOutput(
            representation=_last_observed(sequence, present),
            sequence_states=sequence,
        )


TEMPORAL_ENCODERS = {
    "last_day_mlp": LastDayMLPEncoder,
    "gru": CausalGRUEncoder,
    "transformer": CausalTransformerEncoder,
    "cyclessm": CycleSSMEncoder,
    "history_conditioned_adapter": HistoryConditionedAdapterEncoder,
}


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


__all__ = [
    "CausalGRUEncoder",
    "CausalMemoryOutput",
    "CausalTransformerEncoder",
    "CycleSSMEncoder",
    "CyclicStateSpaceLayer",
    "LastDayMLPEncoder",
    "HistoryConditionedAdapterEncoder",
    "PersonalCausalMemory",
    "TEMPORAL_ENCODERS",
    "TemporalRepresentationOutput",
    "count_trainable_parameters",
]
