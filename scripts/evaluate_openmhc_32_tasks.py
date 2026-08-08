"""Run the official OpenMHC 32-task frozen-probe protocol on embedding caches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd


PRIMARY_METRICS = {"auprc", "accuracy", "spearman_r", "pearson_r"}

TASK_NAMES_ZH = {
    "Atrial fibrillation (Afib)": "心房颤动",
    "BMI_categories": "体重指数分级",
    "BMI_values": "体重指数",
    "BiologicalSex": "生理性别",
    "CAD": "冠状动脉疾病",
    "Cerebrovascular Disease": "脑血管疾病",
    "Congenital Heart": "先天性心脏病",
    "Diabetes": "糖尿病",
    "GoSleepTime_categories": "入睡时间分级",
    "Hdl": "高密度脂蛋白胆固醇",
    "Heart Failure or CHF": "心力衰竭",
    "Hypertension": "高血压",
    "Ldl": "低密度脂蛋白胆固醇",
    "PH": "肺动脉高压",
    "Peripheral/Systemic Vascular Disease": "外周或全身血管疾病",
    "SystolicBloodPressure": "收缩压",
    "TotalCholesterol": "总胆固醇",
    "WakeUpTime_categories": "起床时间分级",
    "WeightKilograms": "体重",
    "age": "年龄",
    "blood_pressure_categories": "血压分级",
    "cardiovascular_disease": "心血管疾病",
    "feel_worthwhile1": "生活意义感",
    "feel_worthwhile2": "愉快感",
    "feel_worthwhile3": "担忧感",
    "feel_worthwhile4": "低落感",
    "framingham_risk": "弗雷明汉心血管风险",
    "satisfiedwith_life": "生活满意度",
    "sleep_diagnosis1": "睡眠障碍诊断",
    "sleep_time_categories": "睡眠时长分级",
    "vigorous_act": "高强度活动时长",
    "work": "当前就业状态",
}

TASK_TYPE_ZH = {
    "binary": "二分类",
    "multiclass": "多分类",
    "ordinal": "有序分类",
    "regression": "回归",
}

METRIC_NAMES_ZH = {
    "auprc": "精确率-召回率曲线下面积",
    "accuracy": "准确率",
    "spearman_r": "斯皮尔曼等级相关系数",
    "pearson_r": "皮尔逊相关系数",
}


def _markdown_table(frame: pd.DataFrame) -> str:
    def clean(value) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(clean(column) for column in frame.columns) + " |"
    divider = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = [
        "| " + " | ".join(clean(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def _install_openmhc_sklearn_compatibility() -> None:
    """Teach sklearn>=1.9 how to recognize OpenMHC's ordinal wrapper as fitted.

    OpenMHC stores fitted state in ``levels``/``clfs`` rather than attributes
    ending in an underscore.  Newer sklearn therefore rejects the otherwise
    fitted Pipeline in ``Pipeline.predict``.  This changes no estimator,
    hyperparameter, fit or prediction calculation; it only exposes fitted state.
    """

    from downstream_evaluation.models.registry import LogRegOrdinalWrapper

    if not hasattr(LogRegOrdinalWrapper, "__sklearn_is_fitted__"):
        LogRegOrdinalWrapper.__sklearn_is_fitted__ = lambda self: (
            self.levels is not None and len(self.clfs) == max(len(self.levels) - 1, 0)
        )


def _evaluate(
    *,
    name: str,
    cache_dir: Path,
    data_dir: Path,
    output_root: Path,
    seed: int,
) -> pd.DataFrame:
    import openmhc
    from downstream_evaluation.models.lsm2.model import LSM2

    class NamedCachedLSM2(LSM2):
        def __init__(self) -> None:
            super().__init__(
                data_dir=str(data_dir),
                cache_dir=str(cache_dir),
                seed=seed,
            )
            self.name = name
            self._constant_prediction: float | None = None

        def fit(self, data, labels, task_type) -> None:
            unique = np.unique(labels)
            if task_type in ("binary", "multiclass") and len(unique) < 2:
                self._constant_prediction = float(unique[0])
                self._probe = None
                return
            self._constant_prediction = None
            super().fit(data, labels, task_type)

        def predict(self, data) -> np.ndarray:
            if self._constant_prediction is not None:
                return np.full(
                    len(self._ctx.user_ids),
                    self._constant_prediction,
                    dtype=np.float64,
                )
            return super().predict(data)

    method_output = output_root / name
    method_output.mkdir(parents=True, exist_ok=True)
    result = openmhc.evaluate_prediction(
        NamedCachedLSM2(),
        version="xs",
        tasks="all",
        data_dir=data_dir,
        seed=seed,
        output_dir=method_output,
        method_name=name,
    )
    frame = result.to_dataframe()
    frame.to_csv(method_output / "metrics.csv", index=False)
    report = {
        "format_version": 1,
        "method": name,
        "cache_dir": str(cache_dir.resolve()),
        "data_dir": str(data_dir.resolve()),
        "seed": seed,
        "tasks": int(frame["task"].nunique()),
        "overall_fallback_rate": result.overall_fallback_rate,
        "fallback_rate": result.fallback_rate,
    }
    (method_output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return frame


def _primary(frame: pd.DataFrame, value_name: str) -> pd.DataFrame:
    selected = frame[frame["metric"].isin(PRIMARY_METRICS)].copy()
    if selected["task"].duplicated().any():
        raise ValueError("each task must have exactly one primary metric")
    return selected.rename(columns={"value": value_name})[
        ["task", "task_type", "metric", "n_test", value_name]
    ]


def _write_comparison(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    output_root: Path,
) -> None:
    comparison = _primary(baseline, "OpenMHC").merge(
        _primary(candidate, "FemMHC").drop(columns=["task_type", "metric", "n_test"]),
        on="task",
        validate="one_to_one",
    )
    comparison["绝对提升"] = comparison["FemMHC"] - comparison["OpenMHC"]
    comparison["相对变化百分比"] = np.where(
        comparison["OpenMHC"].abs() > 1e-12,
        comparison["绝对提升"] / comparison["OpenMHC"].abs() * 100.0,
        np.nan,
    )
    scorable = np.isfinite(comparison["OpenMHC"]) & np.isfinite(comparison["FemMHC"])
    comparison["结果"] = np.select(
        [
            ~scorable,
            comparison["绝对提升"] > 0,
            comparison["绝对提升"] < 0,
        ],
        ["XS测试集无正例，不可评估", "提升", "下降"],
        default="持平",
    )
    comparison.insert(1, "任务中文", comparison["task"].map(TASK_NAMES_ZH))
    comparison["task_type"] = comparison["task_type"].map(TASK_TYPE_ZH)
    comparison["metric"] = comparison["metric"].map(METRIC_NAMES_ZH)
    comparison = comparison.rename(
        columns={
            "task": "任务代码",
            "task_type": "任务类型",
            "metric": "主要指标",
            "n_test": "测试人数",
        }
    )
    comparison.insert(0, "序号", np.arange(1, len(comparison) + 1))
    comparison.to_csv(output_root / "OpenMHC与FemMHC_32项结果对照.csv", index=False)

    evaluated = comparison.loc[scorable]
    wins = int((evaluated["绝对提升"] > 0).sum())
    losses = int((evaluated["绝对提升"] < 0).sum())
    ties = int((evaluated["绝对提升"] == 0).sum())
    unavailable = int((~scorable).sum())
    summary = {
        "format_version": 1,
        "任务清单数": len(comparison),
        "可评估任务数": len(evaluated),
        "XS测试集无正例不可评估任务数": unavailable,
        "提升任务数": wins,
        "下降任务数": losses,
        "持平任务数": ties,
        "胜率": wins / len(evaluated),
        "平均绝对变化": float(evaluated["绝对提升"].mean()),
        "中位绝对变化": float(evaluated["绝对提升"].median()),
        "OpenMHC平均主要指标": float(evaluated["OpenMHC"].mean()),
        "FemMHC平均主要指标": float(evaluated["FemMHC"].mean()),
    }
    (output_root / "32项汇总.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    table = comparison.copy()
    for column in ("OpenMHC", "FemMHC", "绝对提升"):
        table[column] = table[column].map(lambda value: f"{value:.4f}")
    table["相对变化百分比"] = table["相对变化百分比"].map(
        lambda value: "" if not np.isfinite(value) else f"{value:+.2f}%"
    )
    columns = [
        "序号",
        "任务中文",
        "任务类型",
        "主要指标",
        "测试人数",
        "OpenMHC",
        "FemMHC",
        "绝对提升",
        "相对变化百分比",
        "结果",
    ]
    markdown = [
        "# OpenMHC 与 FemMHC 的 32 项同协议评测",
        "",
        f"32 项清单中有 {len(evaluated)} 项可评估：FemMHC 提升 {wins} 项，下降 {losses} 项，持平 {ties} 项，胜率 {wins / len(evaluated):.1%}。",
        f"另有 {unavailable} 项因 OpenMHC XS 测试集无正例，AUPRC 不可计算；其中 3 项训练集同样无正例。",
        "",
        _markdown_table(table[columns]),
        "",
    ]
    (output_root / "OpenMHC与FemMHC_32项结果对照.md").write_text(
        "\n".join(markdown),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--openmhc-cache", type=Path, required=True)
    parser.add_argument("--femmhc-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-name", default="femmhc-nhanes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--only",
        choices=("openmhc", "femmhc", "both"),
        default="both",
    )
    args = parser.parse_args()
    warnings.filterwarnings(
        "ignore",
        message="'n_jobs' has no effect",
        category=FutureWarning,
    )
    _install_openmhc_sklearn_compatibility()
    os.environ["MHC_DATA_DIR"] = str(args.data_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = args.output_dir / "openmhc-lsm2" / "metrics.csv"
    candidate_path = args.output_dir / args.candidate_name / "metrics.csv"
    if args.only in ("openmhc", "both"):
        baseline = _evaluate(
            name="openmhc-lsm2",
            cache_dir=args.openmhc_cache,
            data_dir=args.data_dir,
            output_root=args.output_dir,
            seed=args.seed,
        )
    elif baseline_path.is_file():
        baseline = pd.read_csv(baseline_path)
    else:
        baseline = None

    if args.only in ("femmhc", "both"):
        candidate = _evaluate(
            name=args.candidate_name,
            cache_dir=args.femmhc_cache,
            data_dir=args.data_dir,
            output_root=args.output_dir,
            seed=args.seed,
        )
    elif candidate_path.is_file():
        candidate = pd.read_csv(candidate_path)
    else:
        candidate = None

    if baseline is not None and candidate is not None:
        _write_comparison(baseline, candidate, args.output_dir)


if __name__ == "__main__":
    main()
