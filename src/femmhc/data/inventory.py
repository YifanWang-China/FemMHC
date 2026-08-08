"""Dataset discovery and lightweight schema profiling for FemMHC.

The profiler avoids extracting archives or deserializing pickle files. It reads
archive central directories and small prefixes of tabular members so that large
collections can be inventoried safely.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import csv
import gzip
import json
import lzma
from pathlib import Path
from typing import BinaryIO, Iterable
import zipfile


TABULAR_SUFFIXES = (".csv", ".tsv", ".txt", ".csv.gz", ".tsv.gz")


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    display_name: str
    path_patterns: tuple[str, ...]
    role: str
    female_scope: str
    time_grain: str
    modalities: tuple[str, ...]
    labels: tuple[str, ...] = ()
    expected_items: int | None = None
    access: str = "public"


@dataclass
class TableSchema:
    container: str
    member: str
    columns: list[str]
    delimiter: str | None
    encoding: str | None


@dataclass
class PathProfile:
    path: str
    kind: str
    bytes: int
    files: int
    extensions: dict[str, int]
    archive_open_ok: bool | None = None
    archive_members: int | None = None
    archive_uncompressed_bytes: int | None = None
    schemas: list[TableSchema] = field(default_factory=list)
    error: str | None = None


@dataclass
class DatasetProfile:
    dataset_id: str
    display_name: str
    present: bool
    readiness: str
    role: str
    female_scope: str
    time_grain: str
    modalities: list[str]
    labels: list[str]
    access: str
    expected_items: int | None
    discovered_items: int
    total_bytes: int
    paths: list[PathProfile]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def default_dataset_specs() -> tuple[DatasetSpec, ...]:
    """Return the versioned FemMHC data catalog.

    ``raw/`` and ``openmhc-xs`` resolve under the W3M data root. Paths prefixed
    with ``curated/`` resolve under the user-facing dataset directory.
    """

    return (
        DatasetSpec(
            "openmhc_xs", "OpenMHC XS",
            ("openmhc-xs", "raw/openmhc_xs_dvn_zymjf6"),
            "foundation_pretraining_and_general_capability_retention",
            "filter female participants using OpenMHC demographics",
            "participant-day, minute sequence",
            ("steps", "heart_rate", "sleep", "activity"),
        ),
        DatasetSpec(
            "mcphases", "mcPHASES 1.0.0", ("curated/mcphases-*.zip",),
            "core_menstrual_supervision_and_causal_next_day_tasks",
            "female menstrual cohort", "participant-day with minute signals",
            ("steps", "heart_rate", "hrv", "temperature", "spo2", "sleep"),
            ("cycle", "symptoms", "menstrual_onset", "urinary_hormones"), 1,
        ),
        DatasetSpec(
            "pregnancy_ga_clock", "Pregnancy Gestational-Age Clock",
            ("curated/Pregnancy_GA_Clock_Zenodo_7689724",),
            "pregnancy_representation_and_gestational_age_transfer",
            "pregnant participants", "seven-day minute-level wearable windows",
            ("activity", "ambient_light"), ("gestational_age",),
        ),
        DatasetSpec(
            "swan", "Study of Women's Health Across the Nation (SWAN)",
            ("curated/ICPSR_*.zip",),
            "menopause_and_midlife_external_tabular_evaluation",
            "women aged approximately 40-55 at enrollment",
            "participant-visit, annual/visit-level",
            ("survey", "clinical", "psychosocial"),
            ("menopause_status", "symptoms", "mood", "sleep_questionnaires"), 13,
        ),
        DatasetSpec(
            "nhanes_activity", "NHANES 2011-2014 Minute-Level Activity",
            ("curated/NHANES_2011_2014",),
            "population_activity_pretraining_and_sex_stratified_evaluation",
            "filter female participants through demographic XPT files",
            "participant-day, minute activity",
            ("accelerometry", "steps", "demographics"),
            ("sex", "age", "population_covariates"),
        ),
        DatasetSpec(
            "loneliness_oura", "Loneliness and Well-being Oura Dataset",
            ("curated/Loneliness_Dataset_Nov10.zip",),
            "affect_stress_sleep_transfer",
            "sex metadata is absent from the public archive; not eligible for a female-only headline result without external participant mapping",
            "participant-day and event-level longitudinal",
            ("oura_sleep", "heart_rate", "hrv", "steps", "samsung_ppg", "imu"),
            ("ema", "phq9", "bdi", "perceived_stress", "loneliness"), 1,
        ),
        DatasetSpec(
            "wearable_hrv_sleep", "Wearable HRV and Sleep",
            ("raw/wearable_hrv_sleep_figshare_28509740",),
            "raw_sensor_pretraining_and_sleep_hrv_transfer",
            "mixed-sex cohort; demographic survey available",
            "raw sensor streams and sleep diary",
            ("ppg", "heart_rate", "hrv", "accelerometry", "sleep"),
            ("sleep_diary", "survey"),
        ),
        DatasetSpec(
            "lifesnaps", "LifeSnaps Fitbit", ("raw/lifesnaps_zenodo_6832242",),
            "longitudinal_lifestyle_pretraining",
            "mixed-sex cohort; retain sex metadata when available",
            "participant-event and participant-day",
            ("fitbit", "heart_rate", "sleep", "steps"),
        ),
        DatasetSpec(
            "lh_surge", "LH Surge Wearable Dataset", ("raw/lh_surge_osf_wzf47",),
            "ovulatory_cycle_external_evaluation", "female cycle cohort",
            "participant-day", ("wearable", "cycle_context"), ("lh_surge",),
        ),
        DatasetSpec(
            "ssaqs", "SSAQ Sleep Dataset", ("raw/ssaqs_zenodo_18706837",),
            "sleep_transfer", "mixed-sex; stratify when demographics permit",
            "sleep episode and participant-day", ("sleep", "wearable"),
        ),
        DatasetSpec(
            "qol_stress", "QoL Stress Fitbit Dataset",
            ("raw/qol_stress_zenodo_20757481",), "stress_and_anxiety_transfer",
            "mixed-sex cohort with sex field", "participant-event and day",
            ("fitbit", "ecg", "eda", "activity"),
            ("pss", "stai", "quality_of_life"),
        ),
        DatasetSpec(
            "crowd_fitbit", "Crowdsourced Fitbit Dataset",
            ("raw/crowdsourced_fitbit_zenodo_53894",), "fitbit_domain_pretraining",
            "sex metadata must be confirmed before female-only use",
            "participant-event", ("heart_rate", "sleep", "activity"),
        ),
        DatasetSpec(
            "weee", "WEEE Wearable Exercise Dataset", ("raw/weee_zenodo_6420886",),
            "cross_device_sensor_alignment", "mixed-sex with demographics",
            "raw sensor streams", ("heart_rate", "hrv", "accelerometry", "fitbit_sense"),
        ),
        DatasetSpec(
            "scientisst_move", "ScientISST-MOVE",
            ("raw/scientisst_move_physionet_1.0.1",), "cross_device_sensor_alignment",
            "mixed-sex cohort with gender field", "raw sensor streams",
            ("ppg", "ecg", "eda", "emg", "temperature", "accelerometry"),
        ),
        DatasetSpec(
            "pulse_transit_ppg", "Pulse Transit Time PPG",
            ("raw/pulse_transit_time_ppg_physionet_1.0.0",),
            "cardiovascular_sensor_alignment", "mixed-sex with gender field",
            "raw sensor streams", ("ppg", "ecg", "spo2", "temperature", "imu"),
        ),
        DatasetSpec(
            "wesad", "WESAD", ("raw/wesad_sciebo",),
            "stress_representation_transfer", "mixed-sex; only three women",
            "raw sensor streams", ("ppg", "ecg", "eda", "temperature", "accelerometry"),
            ("stress", "amusement", "neutral"),
        ),
        DatasetSpec(
            "wisdm", "WISDM Smartphone and Smartwatch", ("raw/wisdm_uci_507",),
            "activity_representation_pretraining",
            "mixed-sex; confirm released demographic metadata", "raw sensor streams",
            ("accelerometry", "gyroscope"), ("activity",),
        ),
        DatasetSpec(
            "ppg_dalia", "PPG-DaLiA", ("raw/ppg_dalia_uci_495",),
            "heart_rate_and_motion_transfer", "mixed-sex with sex metadata",
            "raw sensor streams", ("ppg", "ecg", "eda", "temperature", "accelerometry"),
            ("heart_rate", "activity"),
        ),
        DatasetSpec(
            "capture24", "CAPTURE-24", ("raw/capture24_ora",),
            "free_living_activity_pretraining", "mixed-sex with sex metadata",
            "raw accelerometry and annotated intervals", ("accelerometry",),
            ("activity", "sleep"),
        ),
        DatasetSpec(
            "ifh_affect", "IFH-Affect Oura and Samsung",
            ("curated/ifh_affect.zip", "raw/ifh_affect_dryad_D1WH6T"),
            "longitudinal_affect_transfer",
            "mixed-sex cohort; demographic assessment available",
            "participant-day and event-level longitudinal",
            ("oura", "ppg", "imu", "heart_rate", "hrv", "sleep", "steps"),
            ("panas", "bdi", "gad7", "ema"), access="public_with_browser_antibot",
        ),
        DatasetSpec(
            "wumod", "WUMOD Pregnancy Actigraphy", ("curated/WUMOD",),
            "pregnancy_activity_transfer", "pregnant participants",
            "participant-gestational-week actigraphy", ("actigraphy",),
            ("gestational_age",), access="restricted_nsrr",
        ),
    )


def _suffix(name: str) -> str:
    lower = name.lower()
    for compound in (".csv.gz", ".tsv.gz", ".csv.xz", ".tar.gz"):
        if lower.endswith(compound):
            return compound
    return Path(lower).suffix or "[no_extension]"


def _decode_prefix(data: bytes) -> tuple[str, str] | None:
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return None


def _schema_from_text(
    text: str, *, container: str, member: str, encoding: str
) -> TableSchema | None:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    sample = "\n".join(lines[:20])
    delimiter = "\t" if member.lower().endswith((".tsv", ".tsv.gz")) else ","
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
    except csv.Error:
        pass
    try:
        columns = next(csv.reader([lines[0]], delimiter=delimiter))
    except (csv.Error, StopIteration):
        return None
    columns = [column.strip().lstrip("\ufeff") for column in columns]
    if len(columns) < 2:
        return None
    return TableSchema(container, member, columns[:200], delimiter, encoding)


def _sample_zip_schemas(
    archive: zipfile.ZipFile, archive_path: Path, max_schemas: int = 12
) -> list[TableSchema]:
    schemas: list[TableSchema] = []
    seen_names: set[str] = set()
    candidates = [
        item for item in archive.infolist()
        if not item.is_dir() and item.filename.lower().endswith(TABULAR_SUFFIXES)
    ]
    candidates.sort(key=lambda item: (item.file_size, item.filename))
    for item in candidates:
        basename = Path(item.filename).name.lower()
        if basename in seen_names:
            continue
        seen_names.add(basename)
        try:
            with archive.open(item) as raw:
                if item.filename.lower().endswith(".gz"):
                    with gzip.GzipFile(fileobj=raw) as expanded:
                        prefix = expanded.read(256 * 1024)
                else:
                    prefix = raw.read(256 * 1024)
            decoded = _decode_prefix(prefix)
            if decoded is None:
                continue
            text, encoding = decoded
            schema = _schema_from_text(
                text, container=str(archive_path), member=item.filename, encoding=encoding
            )
            if schema is not None:
                schemas.append(schema)
        except (OSError, RuntimeError, zipfile.BadZipFile):
            continue
        if len(schemas) >= max_schemas:
            break
    return schemas


def _profile_zip(path: Path) -> PathProfile:
    try:
        with zipfile.ZipFile(path) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            return PathProfile(
                path=str(path), kind="zip", bytes=path.stat().st_size, files=1,
                extensions=dict(sorted(Counter(_suffix(i.filename) for i in members).items())),
                archive_open_ok=True, archive_members=len(members),
                archive_uncompressed_bytes=sum(item.file_size for item in members),
                schemas=_sample_zip_schemas(archive, path),
            )
    except (OSError, zipfile.BadZipFile) as error:
        return PathProfile(
            path=str(path), kind="zip", bytes=path.stat().st_size if path.exists() else 0,
            files=1 if path.exists() else 0, extensions={".zip": 1},
            archive_open_ok=False, error=str(error),
        )


def _open_direct_table(path: Path) -> BinaryIO:
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rb")
    if path.name.lower().endswith(".xz"):
        return lzma.open(path, "rb")
    return path.open("rb")


def _sample_direct_schema(path: Path) -> TableSchema | None:
    if not path.name.lower().endswith(TABULAR_SUFFIXES + (".csv.xz",)):
        return None
    try:
        with _open_direct_table(path) as stream:
            decoded = _decode_prefix(stream.read(256 * 1024))
        if decoded is None:
            return None
        text, encoding = decoded
        return _schema_from_text(
            text, container=str(path.parent), member=path.name, encoding=encoding
        )
    except (OSError, EOFError, lzma.LZMAError):
        return None


def _profile_directory(path: Path) -> PathProfile:
    files = [item for item in path.rglob("*") if item.is_file()]
    schemas: list[TableSchema] = []
    for item in sorted(files, key=lambda value: (value.stat().st_size, str(value))):
        schema = _sample_direct_schema(item)
        if schema is not None:
            schemas.append(schema)
        if len(schemas) >= 12:
            break
    return PathProfile(
        path=str(path), kind="directory", bytes=sum(item.stat().st_size for item in files),
        files=len(files),
        extensions=dict(sorted(Counter(_suffix(item.name) for item in files).items())),
        schemas=schemas,
    )


def profile_path(path: Path) -> PathProfile:
    if path.is_dir():
        return _profile_directory(path)
    if path.suffix.lower() == ".zip":
        return _profile_zip(path)
    schema = _sample_direct_schema(path)
    return PathProfile(
        path=str(path), kind="file", bytes=path.stat().st_size, files=1,
        extensions={_suffix(path.name): 1}, schemas=[schema] if schema else [],
    )


def _resolve_pattern(pattern: str, data_root: Path, curated_root: Path) -> list[Path]:
    if pattern.startswith("curated/"):
        root, relative = curated_root, pattern.removeprefix("curated/")
    else:
        root, relative = data_root, pattern
    if any(character in relative for character in "*?["):
        return sorted(root.glob(relative))
    candidate = root / relative
    return [candidate] if candidate.exists() else []


def profile_catalog(
    data_root: Path, curated_root: Path, specs: Iterable[DatasetSpec] | None = None
) -> list[DatasetProfile]:
    profiles: list[DatasetProfile] = []
    for spec in specs or default_dataset_specs():
        discovered: list[Path] = []
        for pattern in spec.path_patterns:
            discovered.extend(_resolve_pattern(pattern, data_root, curated_root))
        paths = sorted({path.resolve() for path in discovered})
        path_profiles = [profile_path(path) for path in paths]
        nonempty = [profile for profile in path_profiles if profile.bytes > 0]
        if nonempty:
            readiness = (
                "partial" if spec.expected_items is not None and len(nonempty) < spec.expected_items
                else "ready"
            )
        elif path_profiles:
            readiness = "empty_or_failed"
        elif spec.access.startswith("restricted"):
            readiness = "restricted_not_available"
        else:
            readiness = "missing"
        profiles.append(DatasetProfile(
            dataset_id=spec.dataset_id, display_name=spec.display_name,
            present=bool(nonempty), readiness=readiness, role=spec.role,
            female_scope=spec.female_scope, time_grain=spec.time_grain,
            modalities=list(spec.modalities), labels=list(spec.labels), access=spec.access,
            expected_items=spec.expected_items, discovered_items=len(nonempty),
            total_bytes=sum(profile.bytes for profile in path_profiles), paths=path_profiles,
        ))
    return profiles


def write_inventory(
    profiles: list[DatasetProfile], json_path: Path, markdown_path: Path
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps({"format_version": 1, "datasets": [p.to_dict() for p in profiles]},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# FemMHC 数据集清单与轻量模式剖析", "",
        "该清单只读取目录、ZIP 中央目录和少量表头；不会反序列化 pickle，也不会抽取个体级数据。", "",
        "| 数据集 | 状态 | 本地规模 | 时间粒度 | 论文角色 | 女性范围 |",
        "|---|---:|---:|---|---|---|",
    ]
    for profile in profiles:
        lines.append(
            f"| {profile.display_name} | {profile.readiness} | "
            f"{profile.total_bytes / 1024**3:.3f} GiB | {profile.time_grain} | "
            f"{profile.role} | {profile.female_scope} |"
        )
    lines.extend(["", "## 发现的表结构", ""])
    for profile in profiles:
        schemas = [schema for path in profile.paths for schema in path.schemas]
        if not schemas:
            continue
        lines.extend([f"### {profile.display_name}", ""])
        for schema in schemas:
            columns = ", ".join(f"`{column}`" for column in schema.columns[:30])
            if len(schema.columns) > 30:
                columns += f", …（共 {len(schema.columns)} 列）"
            lines.append(f"- `{schema.member}`：{columns}")
        lines.append("")
    lines.extend([
        "## 使用边界", "",
        "- SWAN 是访视级临床/问卷数据，不进入分钟级穿戴自监督训练。",
        "- 混合性别数据必须基于参与者级人口学字段筛选或做分层评估。",
        "- 数据切分以参与者为单位；同一参与者不得跨训练、验证、测试集。",
        "- 症状预测使用第 t 天信号预测第 t+1 天标签。",
        "- 受限数据在授权完成前不进入可复现实验主结果。", "",
    ])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "DatasetProfile", "DatasetSpec", "PathProfile", "TableSchema",
    "default_dataset_specs", "profile_catalog", "profile_path", "write_inventory",
]
