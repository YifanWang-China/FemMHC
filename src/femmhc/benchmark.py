"""Auditable task registry for the FemMHC research benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    chinese_name: str
    dataset_id: str
    task_family: str
    target: str
    kind: str
    input_window_days: int
    target_offset_days: int
    primary_metrics: tuple[str, ...]
    protocol: str
    female_evidence: str
    status: str
    headline: bool = True


FEMMHC_BENCHMARK_TASKS: tuple[BenchmarkTask, ...] = (
    BenchmarkTask(
        "mcphases_cycle_phase",
        "月经周期阶段识别",
        "mcphases",
        "menstrual_cycle",
        "phase",
        "multiclass_classification",
        1,
        0,
        ("macro_f1", "balanced_accuracy"),
        "participant_split",
        "female menstrual cohort",
        "implemented",
    ),
    BenchmarkTask(
        "mcphases_onset_24h",
        "24小时内月经开始",
        "mcphases",
        "menstrual_cycle",
        "menstrual_onset_24h",
        "binary_classification",
        1,
        0,
        ("auprc", "auroc", "brier", "ece"),
        "participant_split; nested with 72h risk; validation calibration",
        "female menstrual cohort",
        "implemented",
    ),
    BenchmarkTask(
        "mcphases_onset_72h",
        "72小时内月经开始",
        "mcphases",
        "menstrual_cycle",
        "menstrual_onset_72h",
        "binary_classification",
        1,
        0,
        ("auprc", "auroc", "brier", "ece"),
        "participant_split; nested with 24h risk; validation calibration",
        "female menstrual cohort",
        "implemented",
    ),
    *tuple(
        BenchmarkTask(
            f"mcphases_next_day_{task_id}",
            chinese_name,
            "mcphases",
            "menstrual_symptoms",
            source_target,
            "ordinal_prediction",
            1,
            1,
            ("mae", "quadratic_weighted_kappa", "macro_f1"),
            "use day t wearables only; predict day t+1 label; participant split",
            "female menstrual cohort",
            "implemented",
        )
        for task_id, chinese_name, source_target in (
            ("cramps", "次日经期痉挛严重度", "cramps"),
            ("mood_swing", "次日情绪波动严重度", "moodswing"),
            ("fatigue", "次日疲劳严重度", "fatigue"),
            ("sleep_issue", "次日睡眠问题严重度", "sleepissue"),
            ("stress", "次日主观压力严重度", "stress"),
            ("bloating", "次日腹胀严重度", "bloating"),
            ("flow_volume", "次日经量等级", "flow_volume"),
        )
    ),
    *tuple(
        BenchmarkTask(
            f"mcphases_{target}",
            chinese_name,
            "mcphases",
            "menstrual_hormone_transfer",
            target,
            "regression",
            1,
            0,
            ("mae", "spearman"),
            "participant split; training-participant target transform only",
            "female menstrual cohort",
            "implemented",
            headline=False,
        )
        for target, chinese_name in (
            ("lh", "尿液促黄体生成素回归"),
            ("estrogen", "尿液雌激素代谢物回归"),
            ("pdg", "尿液孕二醇葡糖苷酸回归"),
        )
    ),
    BenchmarkTask(
        "pregnancy_gestational_age",
        "孕周估计",
        "pregnancy_ga_clock",
        "pregnancy",
        "gestational_age_weeks",
        "regression",
        7,
        0,
        ("mae_weeks", "rmse_weeks", "r2", "spearman"),
        "seven midnight-aligned days; participant split across longitudinal visits",
        "pregnant cohort",
        "implemented",
    ),
    *tuple(
        BenchmarkTask(
            task_id,
            chinese_name,
            "loneliness_oura",
            "affect_and_mental_health",
            target,
            "ordinal_regression",
            1,
            1,
            ("mae", "spearman"),
            "day t Oura signals predict day t+1 EMA; participant split",
            "mixed cohort; released archive has no verified sex mapping",
            "adapter_pending",
            headline=False,
        )
        for task_id, chinese_name, target in (
            ("oura_next_day_loneliness", "次日孤独感", "lonely"),
            ("oura_next_day_negative_affect", "次日负面情绪", "negative"),
            ("oura_next_day_positive_affect", "次日积极情绪", "positive"),
        )
    ),
    *tuple(
        BenchmarkTask(
            f"female_{target.lower()}_{timepoint}",
            f"女性{chinese_name}（{timepoint_name}）",
            "wearable_hrv_sleep",
            "sleep_and_mental_health",
            f"{target}_{source_suffix}",
            "regression",
            input_days,
            0,
            ("mae", "spearman"),
            "female subset by survey sex; participant split; coarse study-timepoint alignment",
            "25 women with released sex field and three survey timepoints",
            "evaluated_support",
            headline=False,
        )
        for target, chinese_name in (
            ("PHQ9", "抑郁症状量表分数"),
            ("GAD7", "焦虑症状量表分数"),
            ("ISI", "失眠严重度量表分数"),
        )
        for timepoint, timepoint_name, source_suffix, input_days in (
            ("middle", "中期", "2", 14),
            ("final", "末期", "F", 28),
        )
    ),
    *tuple(
        BenchmarkTask(
            task_id,
            chinese_name,
            "swan",
            "menopause",
            target,
            kind,
            0,
            0,
            metrics,
            "annual visit-level tabular external evaluation only",
            "women-only midlife cohort; no continuous wearable input",
            "label_harmonization_pending",
            headline=False,
        )
        for task_id, chinese_name, target, kind, metrics in (
            (
                "swan_menopausal_stage",
                "更年期阶段",
                "menopausal_stage",
                "multiclass_classification",
                ("macro_f1", "balanced_accuracy"),
            ),
            (
                "swan_vasomotor_burden",
                "潮热盗汗负担",
                "hot_flash_night_sweat",
                "ordinal_prediction",
                ("mae", "quadratic_weighted_kappa"),
            ),
            (
                "swan_sleep_difficulty",
                "中年女性睡眠困难",
                "sleep_difficulty",
                "binary_classification",
                ("auroc", "auprc"),
            ),
        )
    ),
)


def validate_benchmark_tasks(
    tasks: tuple[BenchmarkTask, ...] = FEMMHC_BENCHMARK_TASKS,
) -> None:
    identifiers = [task.task_id for task in tasks]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("benchmark task identifiers must be unique")
    for task in tasks:
        if task.input_window_days < 0 or task.target_offset_days < 0:
            raise ValueError(f"negative temporal window in {task.task_id}")
        if not task.primary_metrics:
            raise ValueError(f"task {task.task_id} has no primary metric")
        if task.dataset_id == "swan" and task.headline:
            raise ValueError("SWAN has no wearable input and cannot be a headline task")


def write_benchmark_manifest(output_dir: Path) -> tuple[Path, Path]:
    validate_benchmark_tasks()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "femmhc_benchmark_tasks.json"
    markdown_path = output_dir / "femmhc_benchmark_tasks.md"
    payload = {
        "format_version": 1,
        "split_unit": "participant",
        "selection_split": "validation",
        "test_policy": "single locked test evaluation plus participant bootstrap",
        "tasks": [asdict(task) for task in FEMMHC_BENCHMARK_TASKS],
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# FemMHC 女性穿戴任务清单",
        "",
        "所有任务按参与者切分；调参与校准只使用验证集；测试集报告参与者聚类 bootstrap 置信区间。",
        "",
        "| 任务 | 数据集 | 输入→目标 | 主指标 | 状态 | 论文主表 |",
        "|---|---|---|---|---|---:|",
    ]
    for task in FEMMHC_BENCHMARK_TASKS:
        timing = f"{task.input_window_days}日输入→+{task.target_offset_days}日"
        lines.append(
            f"| {task.chinese_name} | {task.dataset_id} | {timing} | "
            f"{', '.join(task.primary_metrics)} | {task.status} | "
            f"{'是' if task.headline else '否'} |"
        )
    lines.extend(
        [
            "",
            "## 约束",
            "",
            "- SWAN 没有连续穿戴输入，只用于更年期标签统一和外部临床评价，不进入穿戴模型主结果。",
            "- Oura 孤独感数据公开包尚未确认性别映射，因此暂不作为女性专属提升声明。",
            "- Wearable HRV and Sleep 只有 25 名女性，且量表只标记研究开始/中期/末期，暂作为外部迁移支持任务。",
            "- 所有次日任务严格使用第 t 日信号预测第 t+1 日标签。",
            "- OpenMHC 原始任务作为通用能力保持实验，和女性任务主表分开报告。",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


__all__ = [
    "BenchmarkTask",
    "FEMMHC_BENCHMARK_TASKS",
    "validate_benchmark_tasks",
    "write_benchmark_manifest",
]
