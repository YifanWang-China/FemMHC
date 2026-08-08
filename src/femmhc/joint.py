"""Cross-cohort, partially labelled joint learning for FemMHC.

The module consumes daily OpenMHC/FemMHC embeddings.  A general causal
temporal path models recovery and long-range drift, while CycleSSM is kept as
an explicitly menstrual-domain path.  A learnable health-state graph then
shares evidence between health domains without collapsing them into one
opaque vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .cyclessm import (
    CausalGRUEncoder,
    CycleSSMEncoder,
    HistoryConditionedAdapterEncoder,
    LastDayMLPEncoder,
)
from .heads import ClassificationHead, OrdinalHead, ProbabilisticOutput, RegressionHead
from .tasks import NestedHorizonOutput, NestedOnsetHead, masked_task_loss, nested_onset_loss


HEALTH_DOMAINS: tuple[str, ...] = (
    "menstrual",
    "sleep_recovery",
    "affect_stress",
    "autonomic",
    "activity_load",
    "cardiometabolic",
    "life_stage",
    "context",
)

DUAL_PATH_ARCHITECTURES: tuple[str, ...] = (
    "dual_path_router",
    "dual_view_residual_router",
    "dual_path_no_cycle",
    "dual_path_own_domain",
    "dual_path_fixed_gate",
    "dual_path_timescale_router",
    "dual_path_source_aware",
    "dual_path_cycle_aware",
    "dual_path_cycle_direct",
    "dual_path_task_selected",
    "dual_path_task_selected_soft",
    "dual_path_phase_geometry",
    "dual_path_circular_phase_head",
)


# Representation sources were selected with participant-grouped cross-validation
# on the 29 mcPHASES training participants only.  The two onset horizons share a
# coherent nested-probability head, so both use the 24 h train-selected source;
# no validation or test result enters this mapping.
MCPHASES_TRAIN_SELECTED_REPRESENTATIONS: Mapping[str, str] = {
    "mcphases/cycle_phase": "cycle",
    "mcphases/menstrual_onset_24h": "menstrual_domain",
    "mcphases/menstrual_onset_72h": "menstrual_domain",
    "mcphases/cramps": "menstrual_domain",
    "mcphases/mood_swing": "menstrual_domain",
    "mcphases/fatigue": "general",
    "mcphases/sleep_issue": "general",
    "mcphases/perceived_stress": "task_route",
    "mcphases/bloating": "general",
    "mcphases/flow_volume": "menstrual_domain",
    "mcphases/lh": "task_route",
    "mcphases/estrogen_metabolite": "menstrual_domain",
    "mcphases/pdg": "task_route",
}

FAST_STATE_DOMAINS: tuple[str, ...] = (
    "menstrual",
    "sleep_recovery",
    "affect_stress",
    "autonomic",
    "activity_load",
)

JOINT_ARCHITECTURES: tuple[str, ...] = (
    "last_day_shared",
    "history_conditioned_adapter",
    "shared_backbone",
    "factorized_no_graph",
    "full",
    "gated_graph",
    "task_router",
    *DUAL_PATH_ARCHITECTURES,
    "mmoe",
    "ple",
)


@dataclass(frozen=True)
class JointTaskSpec:
    """One partially observed target in the joint FemMHC objective."""

    task_id: str
    source: str
    domain: str
    kind: str
    classes: int | None
    primary_metric: str
    role: str
    trainable: bool = True
    headline: bool = False


def _task(
    task_id: str,
    source: str,
    domain: str,
    kind: str,
    classes: int | None,
    primary_metric: str,
    role: str,
    *,
    trainable: bool = True,
    headline: bool = False,
) -> JointTaskSpec:
    return JointTaskSpec(
        task_id,
        source,
        domain,
        kind,
        classes,
        primary_metric,
        role,
        trainable,
        headline,
    )


# The complete OpenMHC XS benchmark is retained in the registry.  The four
# rare cardiovascular targets have no scorable positive examples in the
# current XS split and therefore remain auditable but are not optimized.
OPENMHC_AUXILIARY_TASKS: tuple[JointTaskSpec, ...] = (
    _task("openmhc/atrial_fibrillation", "openmhc", "cardiometabolic", "binary", 2, "auprc", "retention"),
    _task("openmhc/bmi_categories", "openmhc", "cardiometabolic", "ordinal", 5, "spearman_r", "background"),
    _task("openmhc/bmi_values", "openmhc", "cardiometabolic", "regression", None, "pearson_r", "background"),
    _task("openmhc/biological_sex", "openmhc", "context", "binary", 2, "auprc", "retention"),
    _task("openmhc/cad", "openmhc", "cardiometabolic", "binary", 2, "auprc", "retention"),
    _task("openmhc/cerebrovascular_disease", "openmhc", "cardiometabolic", "binary", 2, "auprc", "retention"),
    _task("openmhc/congenital_heart", "openmhc", "cardiometabolic", "binary", 2, "auprc", "retention", trainable=False),
    _task("openmhc/diabetes", "openmhc", "cardiometabolic", "binary", 2, "auprc", "background"),
    _task("openmhc/go_sleep_time", "openmhc", "sleep_recovery", "ordinal", 5, "spearman_r", "daily"),
    _task("openmhc/hdl", "openmhc", "cardiometabolic", "regression", None, "pearson_r", "background"),
    _task("openmhc/heart_failure", "openmhc", "cardiometabolic", "binary", 2, "auprc", "retention", trainable=False),
    _task("openmhc/hypertension", "openmhc", "cardiometabolic", "binary", 2, "auprc", "background"),
    _task("openmhc/ldl", "openmhc", "cardiometabolic", "regression", None, "pearson_r", "background"),
    _task("openmhc/pulmonary_hypertension", "openmhc", "cardiometabolic", "binary", 2, "auprc", "retention", trainable=False),
    _task("openmhc/peripheral_vascular_disease", "openmhc", "cardiometabolic", "binary", 2, "auprc", "retention", trainable=False),
    _task("openmhc/systolic_blood_pressure", "openmhc", "cardiometabolic", "regression", None, "pearson_r", "background"),
    _task("openmhc/total_cholesterol", "openmhc", "cardiometabolic", "regression", None, "pearson_r", "background"),
    _task("openmhc/wake_up_time", "openmhc", "sleep_recovery", "ordinal", 4, "spearman_r", "daily"),
    _task("openmhc/weight_kg", "openmhc", "cardiometabolic", "regression", None, "pearson_r", "background"),
    _task("openmhc/age", "openmhc", "context", "regression", None, "pearson_r", "conditioning"),
    _task("openmhc/blood_pressure_categories", "openmhc", "cardiometabolic", "ordinal", 5, "spearman_r", "background"),
    _task("openmhc/cardiovascular_disease", "openmhc", "cardiometabolic", "binary", 2, "auprc", "background"),
    _task("openmhc/feel_worthwhile", "openmhc", "affect_stress", "ordinal", 4, "spearman_r", "daily"),
    _task("openmhc/happiness", "openmhc", "affect_stress", "ordinal", 4, "spearman_r", "daily"),
    _task("openmhc/worry", "openmhc", "affect_stress", "ordinal", 4, "spearman_r", "daily"),
    _task("openmhc/depressed_feeling", "openmhc", "affect_stress", "ordinal", 4, "spearman_r", "daily"),
    _task("openmhc/framingham_risk", "openmhc", "cardiometabolic", "regression", None, "pearson_r", "background"),
    _task("openmhc/life_satisfaction", "openmhc", "affect_stress", "ordinal", 4, "spearman_r", "daily"),
    _task("openmhc/sleep_diagnosis", "openmhc", "sleep_recovery", "binary", 2, "auprc", "background"),
    _task("openmhc/sleep_duration", "openmhc", "sleep_recovery", "ordinal", 4, "spearman_r", "daily"),
    _task("openmhc/vigorous_activity", "openmhc", "activity_load", "regression", None, "pearson_r", "daily"),
    _task("openmhc/work_status", "openmhc", "context", "binary", 2, "auprc", "conditioning"),
)


OPENMHC_PHYSIOLOGY_AUXILIARY_TASKS: tuple[JointTaskSpec, ...] = (
    _task("openmhc/watch_resting_heart_rate", "openmhc", "autonomic", "regression", None, "pearson_r", "daily"),
    _task("openmhc/watch_hrv_sdnn", "openmhc", "autonomic", "regression", None, "pearson_r", "daily"),
    _task("openmhc/watch_respiratory_rate", "openmhc", "autonomic", "regression", None, "pearson_r", "daily"),
    _task("openmhc/watch_walking_heart_rate", "openmhc", "autonomic", "regression", None, "pearson_r", "daily"),
    _task("openmhc/watch_vo2max", "openmhc", "cardiometabolic", "regression", None, "pearson_r", "background"),
    _task("openmhc/watch_stand_time", "openmhc", "activity_load", "regression", None, "pearson_r", "daily"),
    _task("openmhc/watch_basal_energy", "openmhc", "activity_load", "regression", None, "pearson_r", "daily"),
)


FEMALE_HEALTH_TASKS: tuple[JointTaskSpec, ...] = (
    _task("mcphases/cycle_phase", "mcphases", "menstrual", "multiclass", 4, "macro_f1", "daily", headline=True),
    _task("mcphases/menstrual_onset_24h", "mcphases", "menstrual", "binary", 2, "auprc", "daily", headline=True),
    _task("mcphases/menstrual_onset_72h", "mcphases", "menstrual", "binary", 2, "auprc", "daily", headline=True),
    _task("mcphases/cramps", "mcphases", "menstrual", "ordinal", 6, "mae", "daily", headline=True),
    _task("mcphases/mood_swing", "mcphases", "affect_stress", "ordinal", 6, "mae", "daily", headline=True),
    _task("mcphases/fatigue", "mcphases", "sleep_recovery", "ordinal", 6, "mae", "daily", headline=True),
    _task("mcphases/sleep_issue", "mcphases", "sleep_recovery", "ordinal", 6, "mae", "daily", headline=True),
    _task("mcphases/perceived_stress", "mcphases", "affect_stress", "ordinal", 6, "mae", "daily", headline=True),
    _task("mcphases/bloating", "mcphases", "menstrual", "ordinal", 6, "mae", "daily", headline=True),
    _task("mcphases/flow_volume", "mcphases", "menstrual", "ordinal", 7, "mae", "daily", headline=True),
    _task("mcphases/lh", "mcphases", "menstrual", "regression", None, "mae", "transfer"),
    _task("mcphases/estrogen_metabolite", "mcphases", "menstrual", "regression", None, "mae", "transfer"),
    _task("mcphases/pdg", "mcphases", "menstrual", "regression", None, "mae", "transfer"),
    _task("depress/cesd", "depress_fitbit", "affect_stress", "regression", None, "mae", "questionnaire", headline=True),
    _task("depress/stai_state", "depress_fitbit", "affect_stress", "regression", None, "mae", "questionnaire", headline=True),
    _task("depress/perceived_stress", "depress_fitbit", "affect_stress", "regression", None, "mae", "daily", headline=True),
    _task("depress/positive_affect", "depress_fitbit", "affect_stress", "regression", None, "mae", "daily", headline=True),
    _task("depress/negative_affect", "depress_fitbit", "affect_stress", "regression", None, "mae", "daily", headline=True),
    _task("inphrsym/next_anxiety_severity", "inphrsym", "affect_stress", "ordinal", 4, "mae", "daily", headline=True),
    _task("inphrsym/next_high_anxiety", "inphrsym", "affect_stress", "binary", 2, "auprc", "daily", headline=True),
    _task("inphrsym/next_irritability_severity", "inphrsym", "affect_stress", "ordinal", 4, "mae", "daily", headline=True),
    _task("inphrsym/next_high_irritability", "inphrsym", "affect_stress", "binary", 2, "auprc", "daily", headline=True),
    _task("inphrsym/next_negative_mood_severity", "inphrsym", "affect_stress", "ordinal", 4, "mae", "daily", headline=True),
    _task("inphrsym/next_high_negative_mood", "inphrsym", "affect_stress", "binary", 2, "auprc", "daily", headline=True),
    _task("inphrsym/next_low_energy_severity", "inphrsym", "sleep_recovery", "ordinal", 4, "mae", "daily", headline=True),
    _task("inphrsym/next_low_energy", "inphrsym", "sleep_recovery", "binary", 2, "auprc", "daily", headline=True),
    _task("inphrsym/next_reported_panic", "inphrsym", "affect_stress", "binary", 2, "auprc", "daily"),
    _task("inphrsym/next_menstruation_state", "inphrsym", "menstrual", "binary", 2, "auprc", "daily"),
    _task("hrv_mental/phq9_middle", "wearable_hrv_sleep", "affect_stress", "regression", None, "mae", "support"),
    _task("hrv_mental/phq9_final", "wearable_hrv_sleep", "affect_stress", "regression", None, "mae", "support"),
    _task("hrv_mental/gad7_middle", "wearable_hrv_sleep", "affect_stress", "regression", None, "mae", "support"),
    _task("hrv_mental/gad7_final", "wearable_hrv_sleep", "affect_stress", "regression", None, "mae", "support"),
    _task("hrv_mental/isi_middle", "wearable_hrv_sleep", "sleep_recovery", "regression", None, "mae", "support"),
    _task("hrv_mental/isi_final", "wearable_hrv_sleep", "sleep_recovery", "regression", None, "mae", "support"),
    _task("pregnancy/gestational_age", "pregnancy_ga_clock", "life_stage", "regression", None, "mae_weeks", "life_stage", headline=True),
    _task("swan/menopause_stage", "swan", "life_stage", "multiclass", 4, "macro_f1", "future", trainable=False),
    _task("swan/vasomotor_burden", "swan", "life_stage", "ordinal", 4, "mae", "future", trainable=False),
    _task("swan/sleep_difficulty", "swan", "sleep_recovery", "binary", 2, "auprc", "future", trainable=False),
)


JOINT_TASKS: tuple[JointTaskSpec, ...] = (
    *OPENMHC_AUXILIARY_TASKS,
    *OPENMHC_PHYSIOLOGY_AUXILIARY_TASKS,
    *FEMALE_HEALTH_TASKS,
)


def validate_joint_tasks(tasks: Sequence[JointTaskSpec] = JOINT_TASKS) -> None:
    identifiers = [task.task_id for task in tasks]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("joint task identifiers must be unique")
    supported = {"binary", "multiclass", "ordinal", "regression"}
    for task in tasks:
        if task.domain not in HEALTH_DOMAINS:
            raise ValueError(f"unknown health domain for {task.task_id}: {task.domain}")
        if task.kind not in supported:
            raise ValueError(f"unsupported task kind for {task.task_id}: {task.kind}")
        if task.kind != "regression" and (task.classes is None or task.classes < 2):
            raise ValueError(f"task {task.task_id} needs at least two classes")


def _last_observed(values: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(values.shape[1], device=values.device).unsqueeze(0)
    last = positions.masked_fill(~present, -1).max(dim=1).values
    return values[torch.arange(values.shape[0], device=values.device), last]


@dataclass(frozen=True)
class HealthStateOutput:
    shared_state: torch.Tensor
    domain_states: Mapping[str, torch.Tensor]
    relation_attention: torch.Tensor
    general_sequence: torch.Tensor
    cycle_sequence: torch.Tensor
    auxiliary: Mapping[str, torch.Tensor] = field(default_factory=dict)


class HealthStateGraph(nn.Module):
    """Multi-head message passing over interpretable health-domain tokens."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        domains: Sequence[str] = HEALTH_DOMAINS,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by graph heads")
        self.domains = tuple(domains)
        self.heads = int(heads)
        self.head_dim = hidden_dim // heads
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.relation_logits = nn.Parameter(
            self._initial_relation_logits(self.domains, heads)
        )
        self.dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    @staticmethod
    def _initial_relation_logits(domains: Sequence[str], heads: int) -> torch.Tensor:
        size = len(domains)
        prior = torch.full((size, size), -1.0)
        prior.fill_diagonal_(1.0)
        links = {
            ("menstrual", "sleep_recovery"),
            ("menstrual", "affect_stress"),
            ("menstrual", "autonomic"),
            ("sleep_recovery", "affect_stress"),
            ("sleep_recovery", "autonomic"),
            ("sleep_recovery", "activity_load"),
            ("affect_stress", "autonomic"),
            ("activity_load", "autonomic"),
            ("activity_load", "cardiometabolic"),
            ("cardiometabolic", "life_stage"),
            ("life_stage", "menstrual"),
            ("context", "life_stage"),
        }
        index = {name: position for position, name in enumerate(domains)}
        for left, right in links:
            if left in index and right in index:
                prior[index[left], index[right]] = 0.5
                prior[index[right], index[left]] = 0.5
        return prior.unsqueeze(0).repeat(heads, 1, 1)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3 or tokens.shape[1] != len(self.domains):
            raise ValueError("tokens must have shape (batch, domains, hidden_dim)")
        batch, domains, hidden = tokens.shape

        def split_heads(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(batch, domains, self.heads, self.head_dim).transpose(1, 2)

        query = split_heads(self.query(tokens))
        key = split_heads(self.key(tokens))
        value = split_heads(self.value(tokens))
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores + self.relation_logits.unsqueeze(0)
        attention = torch.softmax(scores, dim=-1)
        messages = torch.matmul(self.dropout(attention), value)
        messages = messages.transpose(1, 2).reshape(batch, domains, hidden)
        states = self.attention_norm(tokens + self.output(messages))
        states = states + self.feed_forward(states)
        return states, attention


class FemaleHealthStateEncoder(nn.Module):
    """Factorize multi-day embeddings into interacting female-health states."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        *,
        maximum_days: int = 60,
        cycle_modes: int = 8,
        graph_heads: int = 4,
        context_dim: int = 0,
        dropout: float = 0.1,
        architecture: str = "full",
    ) -> None:
        super().__init__()
        if architecture not in JOINT_ARCHITECTURES:
            raise ValueError(
                f"architecture must be one of {JOINT_ARCHITECTURES}, got {architecture!r}"
            )
        if architecture not in {
            "last_day_shared",
            "history_conditioned_adapter",
            "shared_backbone",
            "mmoe",
            "ple",
        } and hidden_dim % (
            2 * cycle_modes
        ):
            raise ValueError("hidden_dim must be divisible by 2 * cycle_modes")
        self.hidden_dim = int(hidden_dim)
        self.context_dim = int(context_dim)
        self.graph_heads = int(graph_heads)
        self.architecture = architecture
        if architecture == "last_day_shared":
            general_encoder = LastDayMLPEncoder
        elif architecture == "history_conditioned_adapter":
            general_encoder = HistoryConditionedAdapterEncoder
        else:
            general_encoder = CausalGRUEncoder
        self.general_temporal = general_encoder(
            input_dim,
            hidden_dim,
            dropout=dropout,
        )
        self.context_projection = (
            nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, hidden_dim))
            if context_dim > 0
            else None
        )
        if architecture in {
            "last_day_shared",
            "history_conditioned_adapter",
            "shared_backbone",
            "mmoe",
            "ple",
        }:
            self.cycle_temporal = None
            self.domain_embeddings = None
            self.domain_projections = None
            self.menstrual_fusion = None
            self.graph = None
            self.shared_projection = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
        else:
            self.cycle_temporal = CycleSSMEncoder(
                input_dim,
                hidden_dim,
                modes=cycle_modes,
                maximum_days=maximum_days,
                dropout=dropout,
            )
            self.domain_embeddings = nn.Parameter(
                torch.empty(len(HEALTH_DOMAINS), hidden_dim)
            )
            nn.init.normal_(self.domain_embeddings, std=0.02)
            self.domain_projections = nn.ModuleDict(
                {
                    domain: nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                    )
                    for domain in HEALTH_DOMAINS
                }
            )
            self.menstrual_fusion = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
            )
            # Keep common-module initialization independent of whether a graph
            # ablation is present by constructing the shared projection first.
            self.shared_projection = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
            )
            self.graph = (
                HealthStateGraph(hidden_dim, heads=graph_heads, dropout=dropout)
                if architecture in {"full", "gated_graph"}
                else None
            )
            self.graph_residual_scale = (
                nn.Parameter(torch.zeros(len(HEALTH_DOMAINS)))
                if architecture == "gated_graph"
                else None
            )

    def _identity_relations(
        self,
        batch: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.eye(
            len(HEALTH_DOMAINS), device=device, dtype=dtype
        ).reshape(1, 1, len(HEALTH_DOMAINS), len(HEALTH_DOMAINS)).expand(
            batch, self.graph_heads, -1, -1
        )

    def forward(
        self,
        daily_embeddings: torch.Tensor,
        day_present: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> HealthStateOutput:
        if daily_embeddings.ndim != 3:
            raise ValueError("daily_embeddings must have shape (batch, days, input_dim)")
        batch, days, _ = daily_embeddings.shape
        if day_present is None:
            present = torch.ones(
                batch, days, dtype=torch.bool, device=daily_embeddings.device
            )
        else:
            if day_present.shape != (batch, days):
                raise ValueError("day_present must have shape (batch, days)")
            present = day_present.to(device=daily_embeddings.device, dtype=torch.bool)
        if not bool(present.any(dim=1).all()):
            raise ValueError("each history needs at least one observed day")

        general = self.general_temporal(daily_embeddings, present)
        context_state = None
        if context is not None:
            if self.context_projection is None:
                raise ValueError("context was supplied but context_dim=0")
            if context.shape != (batch, self.context_dim):
                raise ValueError("context must have shape (batch, context_dim)")
            context_state = self.context_projection(context)

        if self.architecture in {
            "last_day_shared",
            "history_conditioned_adapter",
            "shared_backbone",
            "mmoe",
            "ple",
        }:
            source = general.representation
            if context_state is not None:
                source = source + context_state
            shared = self.shared_projection(source)
            domain_states = {domain: shared for domain in HEALTH_DOMAINS}
            return HealthStateOutput(
                shared_state=shared,
                domain_states=domain_states,
                relation_attention=self._identity_relations(
                    batch,
                    device=shared.device,
                    dtype=shared.dtype,
                ),
                general_sequence=general.sequence_states,
                cycle_sequence=torch.zeros_like(general.sequence_states),
                auxiliary={
                    **general.auxiliary,
                    "last_general_state": _last_observed(
                        general.sequence_states, present
                    ),
                },
            )

        if self.cycle_temporal is None:
            raise RuntimeError("cycle temporal path is required by this architecture")
        cyclic = self.cycle_temporal(daily_embeddings, present)
        cycle_representation = cyclic.representation
        if self.architecture == "dual_path_no_cycle":
            # Preserve the exact architecture and parameter count while
            # removing CycleSSM information from every supervised prediction.
            cycle_representation = torch.zeros_like(cycle_representation)
        domain_tokens = []
        for index, domain in enumerate(HEALTH_DOMAINS):
            source = general.representation
            if domain == "menstrual":
                source = self.menstrual_fusion(
                    torch.cat([general.representation, cycle_representation], dim=-1)
                )
            token = self.domain_projections[domain](source)
            token = token + self.domain_embeddings[index].unsqueeze(0)
            domain_tokens.append(token)
        tokens = torch.stack(domain_tokens, dim=1)

        if context_state is not None:
            tokens = tokens + context_state.unsqueeze(1)

        if self.graph is None:
            states = tokens
            relation_attention = self._identity_relations(
                batch,
                device=tokens.device,
                dtype=tokens.dtype,
            )
        else:
            graph_states, relation_attention = self.graph(tokens)
            if self.architecture == "gated_graph":
                if self.graph_residual_scale is None:
                    raise RuntimeError("gated graph is missing its residual scale")
                scale = self.graph_residual_scale.reshape(1, -1, 1)
                states = tokens + scale * (graph_states - tokens)
            else:
                states = graph_states
        if self.architecture in DUAL_PATH_ARCHITECTURES:
            shared = general.representation
            if context_state is not None:
                shared = shared + context_state
        else:
            shared = self.shared_projection(
                torch.cat([general.representation, states.mean(dim=1)], dim=-1)
            )
        domain_states = {
            domain: states[:, index] for index, domain in enumerate(HEALTH_DOMAINS)
        }
        auxiliary = {
            key: value for key, value in cyclic.auxiliary.items()
        }
        auxiliary["cycle_representation"] = cyclic.representation
        auxiliary["last_general_state"] = _last_observed(
            general.sequence_states, present
        )
        return HealthStateOutput(
            shared_state=shared,
            domain_states=domain_states,
            relation_attention=relation_attention,
            general_sequence=general.sequence_states,
            cycle_sequence=cyclic.sequence_states,
            auxiliary=auxiliary,
        )


class DualViewResidualHealthStateEncoder(nn.Module):
    """Preserve native OpenMHC state and add a selective female residual view.

    The cached 768-dimensional input is interpreted as two aligned views:
    native OpenMHC followed by the frozen female-adapted representation.  A
    separate causal encoder processes each half.  When a native view exists it
    is the shared base state; otherwise the adapted state is the cold-start
    fallback.  Domain states are zero-initialized residual corrections, so the
    architecture begins as an exact native-preserving model and can learn to
    expose the female view only to tasks that benefit from it.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        graph_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or input_dim % 2:
            raise ValueError("dual-view residual routing requires an even input_dim")
        self.view_dim = input_dim // 2
        self.hidden_dim = int(hidden_dim)
        self.graph_heads = int(graph_heads)
        self.native_temporal = CausalGRUEncoder(
            self.view_dim,
            hidden_dim,
            dropout=dropout,
        )
        self.female_temporal = CausalGRUEncoder(
            self.view_dim,
            hidden_dim,
            dropout=dropout,
        )
        self.domain_residuals = nn.ModuleDict()
        for domain in HEALTH_DOMAINS:
            residual = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            nn.init.zeros_(residual[-1].weight)
            nn.init.zeros_(residual[-1].bias)
            self.domain_residuals[domain] = residual

    def _identity_relations(
        self,
        batch: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.eye(
            len(HEALTH_DOMAINS), device=device, dtype=dtype
        ).reshape(1, 1, len(HEALTH_DOMAINS), len(HEALTH_DOMAINS)).expand(
            batch, self.graph_heads, -1, -1
        )

    def forward(
        self,
        daily_embeddings: torch.Tensor,
        day_present: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> HealthStateOutput:
        if context is not None:
            raise ValueError("dual-view residual routing does not use context")
        if daily_embeddings.ndim != 3 or daily_embeddings.shape[-1] != 2 * self.view_dim:
            raise ValueError(
                "daily_embeddings must have shape (batch, days, 2 * view_dim)"
            )
        batch, days, _ = daily_embeddings.shape
        if day_present is None:
            present = torch.ones(
                batch, days, dtype=torch.bool, device=daily_embeddings.device
            )
        else:
            if day_present.shape != (batch, days):
                raise ValueError("day_present must have shape (batch, days)")
            present = day_present.to(device=daily_embeddings.device, dtype=torch.bool)
        native_values, female_values = daily_embeddings.split(self.view_dim, dim=-1)
        native = self.native_temporal(native_values, present)
        female = self.female_temporal(female_values, present)
        native_day_available = present & native_values.abs().amax(dim=-1).gt(1e-8)
        native_available = native_day_available.any(dim=1)
        use_native = native_available.unsqueeze(-1)
        shared = torch.where(use_native, native.representation, female.representation)
        shared_sequence = torch.where(
            native_available[:, None, None],
            native.sequence_states,
            female.sequence_states,
        )
        female_delta = female.representation - shared
        residual_input = torch.cat([shared, female_delta], dim=-1)
        domain_states = {
            domain: shared + residual(residual_input)
            for domain, residual in self.domain_residuals.items()
        }
        return HealthStateOutput(
            shared_state=shared,
            domain_states=domain_states,
            relation_attention=self._identity_relations(
                batch,
                device=shared.device,
                dtype=shared.dtype,
            ),
            general_sequence=shared_sequence,
            cycle_sequence=female.sequence_states,
            auxiliary={
                "native_view_available": native_available,
                "native_representation": native.representation,
                "female_representation": female.representation,
                "female_residual": female_delta,
            },
        )


def _module_key(task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "__", task_id)


def _mixture_expert(hidden_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


class JointTaskHeadBank(nn.Module):
    """Route every task to its health-domain state and emit calibrated outputs."""

    ONSET_24H = "mcphases/menstrual_onset_24h"
    ONSET_72H = "mcphases/menstrual_onset_72h"

    def __init__(
        self,
        hidden_dim: int,
        tasks: Sequence[JointTaskSpec] = JOINT_TASKS,
        *,
        task_conditioned_routing: bool = False,
        routing_initial_logit: float = -4.0,
        routing_base: str = "domain",
        force_own_domain: bool = False,
        fixed_routing_gate: bool = False,
        timescale_constrained_routing: bool = False,
        mixture_mode: str = "none",
        mmoe_experts: int = 8,
        ple_shared_experts: int = 2,
    ) -> None:
        super().__init__()
        validate_joint_tasks(tasks)
        self.tasks = tuple(task for task in tasks if task.trainable)
        self.task_by_id = {task.task_id: task for task in self.tasks}
        heads: dict[str, nn.Module] = {}
        for task in self.tasks:
            if task.task_id in {self.ONSET_24H, self.ONSET_72H}:
                continue
            if task.kind in {"binary", "multiclass"}:
                head = ClassificationHead(hidden_dim, task.classes or 2)
            elif task.kind == "ordinal":
                head = OrdinalHead(hidden_dim, task.classes or 2)
            else:
                head = RegressionHead(hidden_dim)
            heads[_module_key(task.task_id)] = head
        self.heads = nn.ModuleDict(heads)
        self.onset_head = (
            NestedOnsetHead(hidden_dim)
            if self.ONSET_24H in self.task_by_id and self.ONSET_72H in self.task_by_id
            else None
        )
        self.task_conditioned_routing = bool(task_conditioned_routing)
        if routing_base not in {
            "domain",
            "shared",
            "source_aware",
            "cycle_aware",
            "cycle_direct",
            "task_selected",
            "task_selected_soft",
        }:
            raise ValueError(
                "routing_base must be 'domain', 'shared', 'source_aware', or "
                "'cycle_aware', 'cycle_direct', 'task_selected', or "
                "'task_selected_soft'"
            )
        self.routing_base = routing_base
        self.force_own_domain = bool(force_own_domain)
        self.fixed_routing_gate = bool(fixed_routing_gate)
        self.timescale_constrained_routing = bool(timescale_constrained_routing)
        if mixture_mode not in {"none", "mmoe", "ple"}:
            raise ValueError("mixture_mode must be 'none', 'mmoe', or 'ple'")
        if task_conditioned_routing and mixture_mode != "none":
            raise ValueError("task routing and expert mixtures are separate baselines")
        if mmoe_experts <= 0 or ple_shared_experts <= 0:
            raise ValueError("expert counts must be positive")
        self.mixture_mode = mixture_mode
        self.routing_index = {
            task.task_id: index for index, task in enumerate(self.tasks)
        }
        if self.task_conditioned_routing:
            self.routing_queries = nn.Parameter(
                torch.empty(len(self.tasks), hidden_dim)
            )
            nn.init.normal_(self.routing_queries, std=0.02)
            self.routing_gate_logits = nn.Parameter(
                torch.full((len(self.tasks),), float(routing_initial_logit))
            )
            self.routing_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.routing_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
            nn.init.eye_(self.routing_value.weight)
        else:
            self.routing_queries = None
            self.routing_gate_logits = None
            self.routing_key = None
            self.routing_value = None

        self.shared_experts = nn.ModuleList()
        self.domain_experts = nn.ModuleDict()
        self.mixture_gates = nn.ModuleDict()
        if mixture_mode == "mmoe":
            self.shared_experts.extend(
                _mixture_expert(hidden_dim) for _ in range(mmoe_experts)
            )
            gate_width = mmoe_experts
        elif mixture_mode == "ple":
            self.shared_experts.extend(
                _mixture_expert(hidden_dim) for _ in range(ple_shared_experts)
            )
            self.domain_experts.update(
                {
                    domain: _mixture_expert(hidden_dim)
                    for domain in HEALTH_DOMAINS
                }
            )
            gate_width = ple_shared_experts + 1
        else:
            gate_width = 0
        if gate_width:
            self.mixture_gates.update(
                {
                    _module_key(task.task_id): nn.Linear(hidden_dim, gate_width)
                    for task in self.tasks
                }
            )

    def _prepare_route_cache(self, states: HealthStateOutput) -> dict[str, Any]:
        cache: dict[str, Any] = {}
        if self.task_conditioned_routing:
            if self.routing_key is None or self.routing_value is None:
                raise RuntimeError("task-conditioned router is incomplete")
            domain_tensor = torch.stack(
                [states.domain_states[domain] for domain in HEALTH_DOMAINS], dim=1
            )
            cache["domain_keys"] = self.routing_key(domain_tensor)
            cache["domain_values"] = self.routing_value(domain_tensor)
        if self.mixture_mode in {"mmoe", "ple"}:
            base = states.shared_state
            cache["mixture_base"] = base
            cache["shared_experts"] = torch.stack(
                [expert(base) for expert in self.shared_experts], dim=1
            )
            if self.mixture_mode == "ple":
                cache["domain_experts"] = {
                    domain: expert(base)
                    for domain, expert in self.domain_experts.items()
                }
        return cache

    def _route(
        self,
        task_id: str,
        states: HealthStateOutput,
        cache: Mapping[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        task = self.task_by_id[task_id]
        if self.mixture_mode in {"mmoe", "ple"}:
            base = cache["mixture_base"]
            expert_states = cache["shared_experts"]
            if self.mixture_mode == "ple":
                domain_state = cache["domain_experts"][task.domain].unsqueeze(1)
                expert_states = torch.cat([expert_states, domain_state], dim=1)
            attention = torch.softmax(
                self.mixture_gates[_module_key(task_id)](base), dim=-1
            )
            routed = torch.einsum("be,beh->bh", attention, expert_states)
            return routed, attention
        if self.routing_base == "task_selected":
            source = MCPHASES_TRAIN_SELECTED_REPRESENTATIONS.get(task_id)
            if source == "cycle":
                cycle = states.auxiliary.get("cycle_representation")
                if cycle is None:
                    raise RuntimeError("task-selected routing requires CycleSSM state")
                return cycle, None
            if source == "menstrual_domain":
                return states.domain_states["menstrual"], None
            if source == "general":
                return states.shared_state, None
            # ``task_route`` and all non-mcPHASES tasks retain the original
            # shared-base task-conditioned router below.
            base = states.shared_state
        elif self.routing_base == "task_selected_soft":
            source = MCPHASES_TRAIN_SELECTED_REPRESENTATIONS.get(task_id)
            if source == "cycle":
                cycle = states.auxiliary.get("cycle_representation")
                if cycle is None:
                    raise RuntimeError("task-selected routing requires CycleSSM state")
                base = cycle
            elif source == "menstrual_domain":
                base = states.domain_states["menstrual"]
            else:
                # General, task-route, and non-mcPHASES tasks start from the
                # shared state.  Unlike the hard ablation, every task retains
                # the learnable residual router below.
                base = states.shared_state
        elif self.routing_base == "cycle_direct":
            if task.domain == "menstrual":
                cycle = states.auxiliary.get("cycle_representation")
                if cycle is None:
                    raise RuntimeError("cycle-direct routing requires CycleSSM state")
                base = cycle
            else:
                base = states.shared_state
        elif self.routing_base == "shared":
            base = states.shared_state
        elif self.routing_base == "source_aware" and task.source == "openmhc":
            base = states.shared_state
        elif self.routing_base == "cycle_aware" and task.source != "mcphases":
            base = states.shared_state
        else:
            base = states.domain_states[task.domain]
        if not self.task_conditioned_routing:
            return base, None
        if (
            self.routing_queries is None
            or self.routing_gate_logits is None
            or self.routing_key is None
            or self.routing_value is None
        ):
            raise RuntimeError("task-conditioned router is incomplete")
        index = self.routing_index[task_id]
        own_domain = HEALTH_DOMAINS.index(task.domain)
        keys = cache["domain_keys"]
        if self.force_own_domain or (
            self.timescale_constrained_routing
            and task.domain not in FAST_STATE_DOMAINS
        ):
            attention = keys.new_zeros(keys.shape[:2])
            attention[:, own_domain] = 1.0
        else:
            query = self.routing_queries[index]
            scores = torch.einsum("bdh,h->bd", keys, query) / math.sqrt(
                query.numel()
            )
            own_prior = torch.zeros_like(scores)
            own_prior[:, own_domain] = 2.0
            scores = scores + own_prior
            if self.timescale_constrained_routing:
                allowed = torch.tensor(
                    [domain in FAST_STATE_DOMAINS for domain in HEALTH_DOMAINS],
                    device=scores.device,
                    dtype=torch.bool,
                )
                scores = scores.masked_fill(~allowed.unsqueeze(0), -torch.inf)
            attention = torch.softmax(scores, dim=-1)
        values = cache["domain_values"]
        routed = torch.einsum("bd,bdh->bh", attention, values)
        gate_logit = self.routing_gate_logits[index]
        if self.fixed_routing_gate:
            gate_logit = gate_logit.detach()
        gate = torch.sigmoid(gate_logit)
        return base + gate * (routed - base), attention

    def forward(
        self,
        states: HealthStateOutput,
        task_ids: Sequence[str] | None = None,
    ) -> tuple[
        dict[str, ProbabilisticOutput | torch.Tensor],
        NestedHorizonOutput | None,
        dict[str, torch.Tensor],
    ]:
        requested = set(task_ids) if task_ids is not None else set(self.task_by_id)
        unknown = requested - set(self.task_by_id)
        if unknown:
            raise KeyError(f"unknown or inactive joint tasks: {sorted(unknown)}")
        selected = tuple(
            task.task_id for task in self.tasks if task.task_id in requested
        )
        route_cache = self._prepare_route_cache(states)
        outputs: dict[str, ProbabilisticOutput | torch.Tensor] = {}
        routing_attention: dict[str, torch.Tensor] = {}
        for task_id in selected:
            if task_id in {self.ONSET_24H, self.ONSET_72H}:
                continue
            task = self.task_by_id[task_id]
            task_state, attention = self._route(task_id, states, route_cache)
            if attention is not None:
                routing_attention[task_id] = attention
            outputs[task_id] = self.heads[_module_key(task_id)](
                task_state
            )

        onset = None
        if requested & {self.ONSET_24H, self.ONSET_72H}:
            if self.onset_head is None:
                raise RuntimeError("nested onset head is not configured")
            onset_state, attention = self._route(
                self.ONSET_24H, states, route_cache
            )
            onset = self.onset_head(onset_state)
            if self.ONSET_24H in selected:
                outputs[self.ONSET_24H] = onset.within_24h
                if attention is not None:
                    routing_attention[self.ONSET_24H] = attention
            if self.ONSET_72H in selected:
                outputs[self.ONSET_72H] = onset.within_72h
                if attention is not None:
                    routing_attention[self.ONSET_72H] = attention
        return outputs, onset, routing_attention


@dataclass(frozen=True)
class JointModelOutput:
    states: HealthStateOutput
    predictions: Mapping[str, ProbabilisticOutput | torch.Tensor]
    nested_onset: NestedHorizonOutput | None
    routing_attention: Mapping[str, torch.Tensor] = field(default_factory=dict)
    cycle_phase_geometry: torch.Tensor | None = None


class FemMHCJointModel(nn.Module):
    """Joint female-health model operating on cached daily foundation embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        *,
        tasks: Sequence[JointTaskSpec] = JOINT_TASKS,
        maximum_days: int = 60,
        cycle_modes: int = 8,
        graph_heads: int = 4,
        context_dim: int = 0,
        dropout: float = 0.1,
        architecture: str = "full",
        initialization_seed: int | None = None,
        routing_initial_logit: float = -4.0,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        if architecture == "dual_view_residual_router":
            if context_dim:
                raise ValueError(
                    "dual-view residual routing does not support context_dim"
                )
            self.state_encoder = DualViewResidualHealthStateEncoder(
                input_dim,
                hidden_dim,
                graph_heads=graph_heads,
                dropout=dropout,
            )
        else:
            self.state_encoder = FemaleHealthStateEncoder(
                input_dim,
                hidden_dim,
                maximum_days=maximum_days,
                cycle_modes=cycle_modes,
                graph_heads=graph_heads,
                context_dim=context_dim,
                dropout=dropout,
                architecture=architecture,
            )
        if initialization_seed is None:
            self.task_heads = JointTaskHeadBank(
                hidden_dim,
                tasks,
                task_conditioned_routing=(
                    architecture == "task_router"
                    or architecture in DUAL_PATH_ARCHITECTURES
                ),
                routing_initial_logit=routing_initial_logit,
                routing_base=(
                    "source_aware"
                    if architecture == "dual_path_source_aware"
                    else "cycle_aware"
                    if architecture == "dual_path_cycle_aware"
                    else "cycle_direct"
                    if architecture == "dual_path_cycle_direct"
                    else "task_selected"
                    if architecture == "dual_path_task_selected"
                    else "task_selected_soft"
                    if architecture == "dual_path_task_selected_soft"
                    else "shared"
                    if architecture in DUAL_PATH_ARCHITECTURES
                    else "domain"
                ),
                force_own_domain=architecture == "dual_path_own_domain",
                fixed_routing_gate=architecture == "dual_path_fixed_gate",
                timescale_constrained_routing=(
                    architecture == "dual_path_timescale_router"
                ),
                mixture_mode=(architecture if architecture in {"mmoe", "ple"} else "none"),
            )
        else:
            # Architecture-specific modules consume different amounts of RNG.
            # Forking here makes task-head initialization strictly comparable.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(initialization_seed) + 100_003)
                self.task_heads = JointTaskHeadBank(
                    hidden_dim,
                    tasks,
                    task_conditioned_routing=(
                        architecture == "task_router"
                        or architecture in DUAL_PATH_ARCHITECTURES
                    ),
                    routing_initial_logit=routing_initial_logit,
                    routing_base=(
                        "source_aware"
                        if architecture == "dual_path_source_aware"
                        else "cycle_aware"
                        if architecture == "dual_path_cycle_aware"
                        else "cycle_direct"
                        if architecture == "dual_path_cycle_direct"
                        else "task_selected"
                        if architecture == "dual_path_task_selected"
                        else "task_selected_soft"
                        if architecture == "dual_path_task_selected_soft"
                        else "shared"
                        if architecture in DUAL_PATH_ARCHITECTURES
                        else "domain"
                    ),
                    force_own_domain=architecture == "dual_path_own_domain",
                    fixed_routing_gate=architecture == "dual_path_fixed_gate",
                    timescale_constrained_routing=(
                        architecture == "dual_path_timescale_router"
                    ),
                    mixture_mode=(
                        architecture if architecture in {"mmoe", "ple"} else "none"
                    ),
                )
        if architecture in {
            "dual_path_phase_geometry",
            "dual_path_circular_phase_head",
        }:
            if initialization_seed is None:
                self.cycle_phase_projector = nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, 2),
                )
            else:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(int(initialization_seed) + 200_003)
                    self.cycle_phase_projector = nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, 2),
                    )
        else:
            self.cycle_phase_projector = None
        if architecture == "dual_path_circular_phase_head":
            self.register_buffer(
                "cycle_phase_prototypes",
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
                ),
            )
        else:
            self.cycle_phase_prototypes = None

    def _circular_phase_output(
        self,
        phase_vector: torch.Tensor,
    ) -> ProbabilisticOutput:
        if self.cycle_phase_prototypes is None:
            raise RuntimeError("circular phase prototypes are not configured")
        direction = F.normalize(phase_vector, dim=-1)
        prototypes = self.cycle_phase_prototypes.to(dtype=direction.dtype)
        logits = 4.0 * torch.matmul(direction, prototypes.transpose(0, 1))
        return ProbabilisticOutput(logits, torch.softmax(logits, dim=-1))

    def forward(
        self,
        daily_embeddings: torch.Tensor,
        day_present: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        task_ids: Sequence[str] | None = None,
    ) -> JointModelOutput:
        states = self.state_encoder(daily_embeddings, day_present, context)
        predictions, onset, routing_attention = self.task_heads(states, task_ids)
        phase_geometry = None
        if self.cycle_phase_projector is not None:
            cycle_representation = states.auxiliary.get("cycle_representation")
            if cycle_representation is None:
                raise RuntimeError("phase geometry requires a CycleSSM representation")
            phase_geometry = self.cycle_phase_projector(cycle_representation)
            phase_task_id = "mcphases/cycle_phase"
            if (
                self.architecture == "dual_path_circular_phase_head"
                and phase_task_id in predictions
            ):
                predictions = dict(predictions)
                predictions[phase_task_id] = self._circular_phase_output(
                    phase_geometry
                )
        return JointModelOutput(
            states,
            predictions,
            onset,
            routing_attention,
            phase_geometry,
        )


@dataclass(frozen=True)
class PartialMultiTaskLoss:
    total: torch.Tensor
    per_task: Mapping[str, torch.Tensor]
    per_domain: Mapping[str, torch.Tensor]
    observed_counts: Mapping[str, int]


def _observed_count(task: JointTaskSpec, target: torch.Tensor) -> int:
    if task.kind == "regression":
        return int(torch.isfinite(target).sum().item())
    return int((torch.isfinite(target) & (target >= 0)).sum().item())


def cyclic_phase_geometry_loss(
    phase_vector: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Map four ordered cycle phases onto a closed unit circle."""

    if phase_vector.ndim != 2 or phase_vector.shape[-1] != 2:
        raise ValueError("phase_vector must have shape (batch, 2)")
    if target.shape != phase_vector.shape[:1]:
        raise ValueError("target must have shape (batch,)")
    observed = torch.isfinite(target) & (target >= 0)
    if not bool(observed.any()):
        return phase_vector.sum() * 0.0
    labels = target[observed].long()
    if bool((labels >= 4).any()):
        raise ValueError("cycle phase labels must be in [0, 3]")
    angle = labels.to(phase_vector.dtype) * (2.0 * math.pi / 4.0)
    expected = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
    prediction = F.normalize(phase_vector[observed], dim=-1)
    direction = 1.0 - (prediction * expected).sum(dim=-1)
    unit_norm = (phase_vector[observed].norm(dim=-1) - 1.0).square()
    return direction.mean() + 0.1 * unit_norm.mean()


def partial_multitask_loss(
    output: JointModelOutput,
    targets: Mapping[str, torch.Tensor],
    *,
    tasks: Sequence[JointTaskSpec] = JOINT_TASKS,
    task_weights: Mapping[str, float] | None = None,
    domain_weights: Mapping[str, float] | None = None,
    phase_geometry_weight: float = 0.0,
) -> PartialMultiTaskLoss:
    """Average available tasks within each domain, then average domains.

    Missing tasks may be absent from ``targets``.  Missing values inside a
    present target use NaN for regression and -1 for categorical/ordinal
    labels.  This lets heterogeneous cohorts update one shared state model
    without fabricating labels they do not contain.
    """

    task_by_id = {task.task_id: task for task in tasks if task.trainable}
    task_weights = dict(task_weights or {})
    domain_weights = dict(domain_weights or {})
    per_task: dict[str, torch.Tensor] = {}
    observed_counts: dict[str, int] = {}
    onset_ids = {JointTaskHeadBank.ONSET_24H, JointTaskHeadBank.ONSET_72H}
    handled_onset = False
    if phase_geometry_weight < 0:
        raise ValueError("phase_geometry_weight must be non-negative")

    if (
        output.nested_onset is not None
        and onset_ids.issubset(targets)
        and onset_ids.issubset(output.predictions)
    ):
        target_24 = targets[JointTaskHeadBank.ONSET_24H]
        target_72 = targets[JointTaskHeadBank.ONSET_72H]
        count = int(((target_24 >= 0) & (target_72 >= 0)).sum().item())
        if count:
            key = "mcphases/menstrual_onset_nested"
            per_task[key] = nested_onset_loss(output.nested_onset, target_24, target_72)
            observed_counts[key] = count
            handled_onset = True

    for task_id, prediction in output.predictions.items():
        if handled_onset and task_id in onset_ids:
            continue
        if task_id not in targets:
            continue
        task = task_by_id[task_id]
        count = _observed_count(task, targets[task_id])
        if count == 0:
            continue
        kind = "classification" if task.kind in {"binary", "multiclass"} else task.kind
        loss = masked_task_loss(prediction, targets[task_id], kind=kind)
        per_task[task_id] = loss * float(task_weights.get(task_id, 1.0))
        observed_counts[task_id] = count

    phase_task_id = "mcphases/cycle_phase"
    if (
        phase_geometry_weight > 0
        and output.cycle_phase_geometry is not None
        and phase_task_id in targets
    ):
        phase_target = targets[phase_task_id]
        count = int((torch.isfinite(phase_target) & (phase_target >= 0)).sum().item())
        if count:
            key = "mcphases/cycle_phase_geometry"
            per_task[key] = float(phase_geometry_weight) * cyclic_phase_geometry_loss(
                output.cycle_phase_geometry,
                phase_target,
            )
            observed_counts[key] = count

    domain_members: dict[str, list[torch.Tensor]] = {}
    for task_id, loss in per_task.items():
        domain = (
            "menstrual"
            if task_id
            in {
                "mcphases/menstrual_onset_nested",
                "mcphases/cycle_phase_geometry",
            }
            else task_by_id[task_id].domain
        )
        domain_members.setdefault(domain, []).append(loss)
    per_domain = {
        domain: torch.stack(losses).mean() * float(domain_weights.get(domain, 1.0))
        for domain, losses in domain_members.items()
    }
    if per_domain:
        total = torch.stack(list(per_domain.values())).mean()
    else:
        total = output.states.shared_state.sum() * 0.0
    return PartialMultiTaskLoss(total, per_task, per_domain, observed_counts)


validate_joint_tasks()


__all__ = [
    "FEMALE_HEALTH_TASKS",
    "HEALTH_DOMAINS",
    "JOINT_TASKS",
    "JOINT_ARCHITECTURES",
    "OPENMHC_AUXILIARY_TASKS",
    "OPENMHC_PHYSIOLOGY_AUXILIARY_TASKS",
    "FemaleHealthStateEncoder",
    "FemMHCJointModel",
    "HealthStateGraph",
    "HealthStateOutput",
    "JointModelOutput",
    "JointTaskHeadBank",
    "JointTaskSpec",
    "PartialMultiTaskLoss",
    "cyclic_phase_geometry_loss",
    "partial_multitask_loss",
    "validate_joint_tasks",
]
