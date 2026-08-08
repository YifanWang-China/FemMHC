"""Personal-baseline temporal modelling for female affective-health tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .heads import ClassificationHead, OrdinalHead, ProbabilisticOutput, RegressionHead
from .tasks import ResidualTaskAdapter, TaskDefinition


INPHRSYM_TASKS: tuple[TaskDefinition, ...] = (
    TaskDefinition(
        "next_anxiety_severity", "明日焦虑严重度", "ordinal", 4, None, "mae", 1
    ),
    TaskDefinition(
        "next_high_anxiety", "明日明显焦虑", "classification", 2, None, "auprc", 1
    ),
    TaskDefinition(
        "next_irritability_severity", "明日易怒严重度", "ordinal", 4, None, "mae", 1
    ),
    TaskDefinition(
        "next_high_irritability", "明日明显易怒", "classification", 2, None, "auprc", 1
    ),
    TaskDefinition(
        "next_negative_mood_severity", "明日低落严重度", "ordinal", 4, None, "mae", 1
    ),
    TaskDefinition(
        "next_high_negative_mood", "明日明显低落", "classification", 2, None, "auprc", 1
    ),
    TaskDefinition(
        "next_negative_energy_severity", "明日低能量严重度", "ordinal", 4, None, "mae", 1
    ),
    TaskDefinition(
        "next_high_negative_energy", "明日明显低能量", "classification", 2, None, "auprc", 1
    ),
    TaskDefinition(
        "next_reported_panic", "明日报告惊恐发作", "classification", 2, None, "auprc", 1
    ),
    TaskDefinition(
        "next_menstruation_state", "明日经期状态", "classification", 2, None, "auprc", 1
    ),
)

DEPRESS_FITBIT_TASKS: tuple[TaskDefinition, ...] = (
    TaskDefinition("cesd", "抑郁症状评分", "regression", None, None, "mae", 1),
    TaskDefinition("stai_state", "状态焦虑评分", "regression", None, None, "mae", 1),
    TaskDefinition(
        "perceived_stress", "感知压力评分", "regression", None, None, "mae", 1
    ),
    TaskDefinition(
        "positive_affect", "积极情绪评分", "regression", None, None, "mae", 1
    ),
    TaskDefinition(
        "negative_affect", "消极情绪评分", "regression", None, None, "mae", 1
    ),
)

WEARABLE_STRESS_TASK = TaskDefinition(
    "self_reported_stress", "即时主观压力强度", "regression", None, None, "mae", 0
)


@dataclass(frozen=True)
class PersonalBaselineTemporalOutput:
    representation: torch.Tensor
    personal_baseline: torch.Tensor
    temporal_state: torch.Tensor
    day_attention: torch.Tensor


class PersonalBaselineTemporalEncoder(nn.Module):
    """Separate a person's stable baseline from recent within-person changes.

    The module consumes already encoded wearable days.  Missing/padded days are
    excluded from both the personal baseline and attention pool.  The result is
    designed for pre-assessment windows and can also operate on a single day.
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        maximum_days: int = 28,
        temporal_layers: int = 2,
        temporal_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embed_dim <= 0 or maximum_days <= 0:
            raise ValueError("embed_dim and maximum_days must be positive")
        if embed_dim % temporal_heads:
            raise ValueError("embed_dim must be divisible by temporal_heads")
        self.maximum_days = int(maximum_days)
        self.day_position = nn.Parameter(torch.zeros(maximum_days, embed_dim))
        nn.init.normal_(self.day_position, std=0.02)
        self.deviation_projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )
        self.deviation_gate = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=temporal_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            layer,
            num_layers=temporal_layers,
            enable_nested_tensor=False,
        )
        self.day_attention = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, max(embed_dim // 2, 1)),
            nn.Tanh(),
            nn.Linear(max(embed_dim // 2, 1), 1),
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        daily_embeddings: torch.Tensor,
        day_present: torch.Tensor | None = None,
    ) -> PersonalBaselineTemporalOutput:
        if daily_embeddings.ndim != 3:
            raise ValueError("daily_embeddings must have shape (batch, days, embed_dim)")
        batch, days, _ = daily_embeddings.shape
        if days > self.maximum_days:
            raise ValueError(
                f"received {days} days but maximum_days={self.maximum_days}"
            )
        if day_present is None:
            day_present = torch.ones(
                batch,
                days,
                dtype=torch.bool,
                device=daily_embeddings.device,
            )
        elif day_present.shape != (batch, days):
            raise ValueError("day_present must have shape (batch, days)")
        else:
            day_present = day_present.to(
                device=daily_embeddings.device, dtype=torch.bool
            )
        if not bool(day_present.any(dim=1).all()):
            raise ValueError("each sequence needs at least one observed day")

        weights = day_present.to(daily_embeddings.dtype).unsqueeze(-1)
        # Mask before arithmetic.  Multiplying a NaN padding sentinel by zero
        # still produces NaN, so a weighted sum alone is not sufficient.
        observed = torch.where(
            day_present.unsqueeze(-1),
            daily_embeddings,
            torch.zeros_like(daily_embeddings),
        )
        baseline = observed.sum(dim=1) / weights.sum(dim=1)
        baseline_days = baseline.unsqueeze(1).expand(-1, days, -1)
        deviation = observed - baseline_days
        gate = self.deviation_gate(torch.cat([observed, baseline_days], dim=-1))
        fused = observed + gate * self.deviation_projection(deviation)
        fused = fused + self.day_position[:days].unsqueeze(0)
        # Values in padded positions are explicitly zeroed before attention so
        # an arbitrary padding sentinel cannot affect an observed day.
        fused = torch.where(day_present.unsqueeze(-1), fused, torch.zeros_like(fused))
        encoded = self.temporal_encoder(fused, src_key_padding_mask=~day_present)
        logits = self.day_attention(encoded).squeeze(-1)
        logits = logits.masked_fill(~day_present, float("-inf"))
        attention = torch.softmax(logits, dim=-1)
        temporal_state = torch.sum(encoded * attention.unsqueeze(-1), dim=1)
        representation = self.output_projection(
            torch.cat([baseline, temporal_state, temporal_state - baseline], dim=-1)
        )
        return PersonalBaselineTemporalOutput(
            representation=representation,
            personal_baseline=baseline,
            temporal_state=temporal_state,
            day_attention=attention,
        )


class AffectiveTaskHeads(nn.Module):
    """Task-family adapters and calibrated heads for affective-health targets."""

    def __init__(
        self,
        embed_dim: int,
        tasks: Iterable[TaskDefinition] = (*INPHRSYM_TASKS, *DEPRESS_FITBIT_TASKS),
        *,
        adapter_bottleneck: int = 64,
    ) -> None:
        super().__init__()
        self.tasks = tuple(tasks)
        if len({task.name for task in self.tasks}) != len(self.tasks):
            raise ValueError("affective task names must be unique")
        self.adapters = nn.ModuleDict(
            {
                "next_day": ResidualTaskAdapter(embed_dim, adapter_bottleneck),
                "questionnaire": ResidualTaskAdapter(embed_dim, adapter_bottleneck),
                "protocol": ResidualTaskAdapter(embed_dim, adapter_bottleneck),
            }
        )
        heads: dict[str, nn.Module] = {}
        for task in self.tasks:
            if task.kind == "classification":
                heads[task.name] = ClassificationHead(embed_dim, task.classes or 2)
            elif task.kind == "ordinal":
                heads[task.name] = OrdinalHead(embed_dim, task.classes or 2)
            elif task.kind == "regression":
                heads[task.name] = RegressionHead(embed_dim)
            else:
                raise ValueError(f"unsupported affective task kind: {task.kind}")
        self.heads = nn.ModuleDict(heads)

    @staticmethod
    def _family(task: TaskDefinition) -> str:
        if task.name.startswith("next_"):
            return "next_day"
        if task.name == WEARABLE_STRESS_TASK.name:
            return "protocol"
        return "questionnaire"

    def forward(
        self, embedding: torch.Tensor
    ) -> dict[str, ProbabilisticOutput | torch.Tensor]:
        adapted = {
            family: adapter(embedding) for family, adapter in self.adapters.items()
        }
        return {
            task.name: self.heads[task.name](adapted[self._family(task)])
            for task in self.tasks
        }


__all__ = [
    "AffectiveTaskHeads",
    "DEPRESS_FITBIT_TASKS",
    "INPHRSYM_TASKS",
    "PersonalBaselineTemporalEncoder",
    "PersonalBaselineTemporalOutput",
    "WEARABLE_STRESS_TASK",
]
