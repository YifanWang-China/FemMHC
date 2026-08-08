"""Female-health downstream task definitions available in mcPHASES."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .heads import (
    ClassificationHead,
    LinearClassificationHead,
    OrdinalHead,
    ProbabilisticOutput,
    RegressionHead,
)


@dataclass(frozen=True)
class TaskDefinition:
    name: str
    chinese_name: str
    kind: str
    classes: int | None
    label_column: int | None
    primary_metric: str
    target_offset_days: int


MCPHASES_TASKS: tuple[TaskDefinition, ...] = (
    TaskDefinition("cycle_phase", "月经周期阶段", "classification", 4, 0, "macro_f1", 0),
    TaskDefinition("cramps", "次日经期痉挛严重度", "ordinal", 6, 1, "mae", 1),
    TaskDefinition("mood_swing", "次日情绪波动严重度", "ordinal", 6, 2, "mae", 1),
    TaskDefinition("fatigue", "次日疲劳严重度", "ordinal", 6, 3, "mae", 1),
    TaskDefinition("sleep_issue", "次日睡眠问题严重度", "ordinal", 6, 4, "mae", 1),
    TaskDefinition("perceived_stress", "次日主观压力严重度", "ordinal", 6, 5, "mae", 1),
    TaskDefinition("bloating", "次日腹胀严重度", "ordinal", 6, 6, "mae", 1),
    TaskDefinition("flow_volume", "次日经量等级", "ordinal", 7, 7, "mae", 1),
    TaskDefinition("menstrual_onset_24h", "24小时内月经开始", "classification", 2, 8, "auprc", 0),
    TaskDefinition("menstrual_onset_72h", "72小时内月经开始", "classification", 2, 9, "auprc", 0),
    TaskDefinition("lh", "尿液促黄体生成素", "regression", None, None, "mae", 0),
    TaskDefinition("estrogen", "尿液雌激素代谢物", "regression", None, None, "mae", 0),
    TaskDefinition("pdg", "尿液孕二醇葡糖苷酸", "regression", None, None, "mae", 0),
)

PREGNANCY_GA_TASK = TaskDefinition(
    "gestational_age",
    "孕周估计",
    "regression",
    None,
    None,
    "mae_weeks",
    0,
)


@dataclass(frozen=True)
class GestationalAgeOutput:
    prediction: torch.Tensor
    day_attention: torch.Tensor


class PregnancyGAHead(nn.Module):
    """Model ordered day-to-day dynamics and estimate gestational age."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 128,
        *,
        maximum_days: int = 14,
        temporal_heads: int = 4,
    ) -> None:
        super().__init__()
        if embed_dim % temporal_heads:
            raise ValueError("embed_dim must be divisible by temporal_heads")
        self.day_position = nn.Parameter(torch.zeros(maximum_days, embed_dim))
        nn.init.normal_(self.day_position, std=0.02)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=temporal_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer,
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.day_attention = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.regressor = RegressionHead(embed_dim, hidden_dim=hidden_dim)

    def forward(
        self,
        daily_embeddings: torch.Tensor,
        day_present: torch.Tensor | None = None,
    ) -> GestationalAgeOutput:
        if daily_embeddings.ndim != 3:
            raise ValueError("daily_embeddings must have shape (batch, days, embed_dim)")
        days = daily_embeddings.shape[1]
        if days > self.day_position.shape[0]:
            raise ValueError(
                f"received {days} days but maximum_days={self.day_position.shape[0]}"
            )
        encoded = daily_embeddings + self.day_position[:days].unsqueeze(0)
        padding_mask = None
        if day_present is not None:
            if day_present.shape != daily_embeddings.shape[:2]:
                raise ValueError("day_present must have shape (batch, days)")
            if not bool(day_present.any(dim=1).all()):
                raise ValueError("each measurement needs at least one observed day")
            padding_mask = ~day_present.bool()
        encoded = self.temporal_encoder(
            encoded,
            src_key_padding_mask=padding_mask,
        )
        logits = self.day_attention(encoded).squeeze(-1)
        if day_present is not None:
            logits = logits.masked_fill(~day_present.bool(), float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        return GestationalAgeOutput(
            prediction=self.regressor(pooled),
            day_attention=weights,
        )


class McPhasesTaskHeads(nn.Module):
    """A transparent multi-task head bank; every output maps to a source label."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        modules: dict[str, nn.Module] = {}
        for task in MCPHASES_TASKS:
            if task.kind == "ordinal":
                modules[task.name] = OrdinalHead(embed_dim, task.classes or 2)
            elif task.kind == "classification":
                modules[task.name] = ClassificationHead(embed_dim, task.classes or 2)
            else:
                modules[task.name] = RegressionHead(embed_dim)
        self.heads = nn.ModuleDict(modules)

    def forward(self, embedding: torch.Tensor) -> dict[str, ProbabilisticOutput | torch.Tensor]:
        return {name: head(embedding) for name, head in self.heads.items()}


class ResidualTaskAdapter(nn.Module):
    """Small task-family adapter that limits interference between label groups."""

    def __init__(self, embed_dim: int, bottleneck_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, embed_dim),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return embedding + torch.tanh(self.scale) * self.network(embedding)


@dataclass(frozen=True)
class NestedHorizonOutput:
    """Three mutually exclusive onset bins yielding nested 24 h and 72 h risks."""

    bin_logits: torch.Tensor
    bin_probabilities: torch.Tensor
    within_24h: ProbabilisticOutput
    within_72h: ProbabilisticOutput


class NestedOnsetHead(nn.Module):
    """Predict onset in 0-24 h, 24-72 h, or beyond 72 h."""

    def __init__(self, embed_dim: int, hidden_dim: int = 192, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    @staticmethod
    def _binary_output(probability: torch.Tensor) -> ProbabilisticOutput:
        probabilities = torch.stack([1.0 - probability, probability], dim=-1).clamp_min(1e-8)
        return ProbabilisticOutput(probabilities.log(), probabilities)

    def forward(self, embedding: torch.Tensor) -> NestedHorizonOutput:
        logits = self.network(embedding)
        bins = torch.softmax(logits, dim=-1)
        within_24h = bins[:, 0]
        within_72h = bins[:, 0] + bins[:, 1]
        return NestedHorizonOutput(
            bin_logits=logits,
            bin_probabilities=bins,
            within_24h=self._binary_output(within_24h),
            within_72h=self._binary_output(within_72h),
        )


class McPhasesV2TaskHeads(nn.Module):
    """Group-adapted heads with coherent nested menstrual-onset probabilities."""

    ONSET_TASKS = {"menstrual_onset_24h", "menstrual_onset_72h"}
    SYMPTOM_TASKS = {
        "cramps",
        "mood_swing",
        "fatigue",
        "sleep_issue",
        "perceived_stress",
        "bloating",
        "flow_volume",
    }
    HORMONE_TASKS = {"lh", "estrogen", "pdg"}

    def __init__(self, embed_dim: int, *, linear_cycle_head: bool = False) -> None:
        super().__init__()
        self.adapters = nn.ModuleDict(
            {
                name: ResidualTaskAdapter(embed_dim)
                for name in ("cycle", "symptoms", "onset", "hormones")
            }
        )
        modules: dict[str, nn.Module] = {}
        for task in MCPHASES_TASKS:
            if task.name in self.ONSET_TASKS:
                continue
            if task.kind == "ordinal":
                modules[task.name] = OrdinalHead(embed_dim, task.classes or 2)
            elif task.kind == "classification":
                if task.name == "cycle_phase" and linear_cycle_head:
                    modules[task.name] = LinearClassificationHead(
                        embed_dim, task.classes or 2
                    )
                else:
                    modules[task.name] = ClassificationHead(embed_dim, task.classes or 2)
            else:
                modules[task.name] = RegressionHead(embed_dim)
        self.heads = nn.ModuleDict(modules)
        self.onset_head = NestedOnsetHead(embed_dim)

    def _group(self, task_name: str) -> str:
        if task_name in self.SYMPTOM_TASKS:
            return "symptoms"
        if task_name in self.HORMONE_TASKS:
            return "hormones"
        return "cycle"

    def forward_with_aux(
        self, embedding: torch.Tensor
    ) -> tuple[dict[str, ProbabilisticOutput | torch.Tensor], NestedHorizonOutput]:
        adapted = {name: adapter(embedding) for name, adapter in self.adapters.items()}
        outputs = {
            name: head(adapted[self._group(name)]) for name, head in self.heads.items()
        }
        onset = self.onset_head(adapted["onset"])
        outputs["menstrual_onset_24h"] = onset.within_24h
        outputs["menstrual_onset_72h"] = onset.within_72h
        return outputs, onset

    def forward(self, embedding: torch.Tensor) -> dict[str, ProbabilisticOutput | torch.Tensor]:
        outputs, _ = self.forward_with_aux(embedding)
        return outputs


def class_balanced_focal_loss(
    output: ProbabilisticOutput,
    target: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    observed = target >= 0
    if not bool(observed.any()):
        return output.logits.sum() * 0.0
    logits = output.logits[observed]
    labels = target[observed].long()
    weights = class_weights.to(logits.device) if class_weights is not None else None
    cross_entropy = F.cross_entropy(logits, labels, weight=weights, reduction="none")
    true_probability = torch.softmax(logits, dim=-1).gather(1, labels[:, None]).squeeze(1)
    return (((1.0 - true_probability).clamp_min(0.0) ** gamma) * cross_entropy).mean()


def cyclic_phase_loss(
    output: ProbabilisticOutput,
    target: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    distance_weight: float = 0.25,
) -> torch.Tensor:
    """Long-tail focal loss plus expected circular phase distance."""

    observed = target >= 0
    focal = class_balanced_focal_loss(
        output, target, class_weights=class_weights, gamma=2.0
    )
    if not bool(observed.any()):
        return focal
    probabilities = output.probabilities[observed]
    labels = target[observed].long()
    classes = torch.arange(probabilities.shape[-1], device=probabilities.device)
    linear_distance = (classes[None, :] - labels[:, None]).abs()
    circular_distance = torch.minimum(
        linear_distance, probabilities.shape[-1] - linear_distance
    ).to(probabilities.dtype)
    expected_distance = (probabilities * circular_distance).sum(dim=-1).mean()
    return focal + distance_weight * expected_distance


def nested_onset_loss(
    output: NestedHorizonOutput,
    within_24h: torch.Tensor,
    within_72h: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Joint focal likelihood for coherent 0-24 h, 24-72 h, and >72 h bins."""

    observed = (within_24h >= 0) & (within_72h >= 0)
    if not bool(observed.any()):
        return output.bin_logits.sum() * 0.0
    target = torch.where(
        within_24h[observed].bool(),
        torch.zeros_like(within_24h[observed]),
        torch.where(
            within_72h[observed].bool(),
            torch.ones_like(within_72h[observed]),
            torch.full_like(within_72h[observed], 2),
        ),
    ).long()
    logits = output.bin_logits[observed]
    weights = class_weights.to(logits.device) if class_weights is not None else None
    cross_entropy = F.cross_entropy(logits, target, weight=weights, reduction="none")
    true_probability = torch.softmax(logits, dim=-1).gather(1, target[:, None]).squeeze(1)
    return (((1.0 - true_probability).clamp_min(0.0) ** gamma) * cross_entropy).mean()


def masked_task_loss(
    output: ProbabilisticOutput | torch.Tensor,
    target: torch.Tensor,
    *,
    kind: str,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Supervised loss that ignores the documented mcPHASES missing labels."""

    if kind == "regression":
        prediction = output if isinstance(output, torch.Tensor) else output.logits.squeeze(-1)
        observed = torch.isfinite(target)
        if not bool(observed.any()):
            return prediction.sum() * 0.0
        return F.smooth_l1_loss(prediction[observed], target[observed].to(prediction.dtype))
    if not isinstance(output, ProbabilisticOutput):
        raise TypeError("classification and ordinal tasks require ProbabilisticOutput")
    observed = target >= 0
    if not bool(observed.any()):
        return output.logits.sum() * 0.0
    if kind == "classification":
        weights = class_weights.to(output.logits.device) if class_weights is not None else None
        return F.cross_entropy(
            output.logits[observed],
            target[observed].long(),
            weight=weights,
        )
    probabilities = output.probabilities[observed].clamp_min(1e-8)
    weights = class_weights.to(probabilities.device) if class_weights is not None else None
    return F.nll_loss(probabilities.log(), target[observed].long(), weight=weights)


__all__ = [
    "GestationalAgeOutput",
    "MCPHASES_TASKS",
    "McPhasesTaskHeads",
    "McPhasesV2TaskHeads",
    "NestedHorizonOutput",
    "NestedOnsetHead",
    "PREGNANCY_GA_TASK",
    "PregnancyGAHead",
    "ResidualTaskAdapter",
    "TaskDefinition",
    "class_balanced_focal_loss",
    "cyclic_phase_loss",
    "masked_task_loss",
    "nested_onset_loss",
]
