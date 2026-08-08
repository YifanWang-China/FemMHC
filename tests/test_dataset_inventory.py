from __future__ import annotations

import json
from pathlib import Path
import zipfile

from femmhc.data.inventory import DatasetSpec, profile_catalog, profile_path, write_inventory


def test_profile_zip_reads_schema_without_extracting(tmp_path: Path) -> None:
    archive = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("participant_1/daily.csv", "date,hr,steps\n2026-01-01,60,1000\n")
        output.writestr("participant_2/daily.csv", "date,hr,steps\n2026-01-02,61,1200\n")

    profile = profile_path(archive)

    assert profile.archive_open_ok is True
    assert profile.archive_members == 2
    assert len(profile.schemas) == 1
    assert profile.schemas[0].columns == ["date", "hr", "steps"]
    assert not (tmp_path / "participant_1").exists()


def test_catalog_marks_expected_group_partial(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    curated_root = tmp_path / "curated"
    curated_root.mkdir(parents=True)
    with zipfile.ZipFile(curated_root / "visit_1.zip", "w") as output:
        output.writestr("data.csv", "id,value\n1,2\n")
    spec = DatasetSpec(
        dataset_id="visits", display_name="Visits",
        path_patterns=("curated/visit_*.zip",), role="evaluation",
        female_scope="women", time_grain="visit", modalities=("survey",),
        expected_items=2,
    )

    profiles = profile_catalog(data_root, curated_root, (spec,))

    assert profiles[0].readiness == "partial"
    assert profiles[0].discovered_items == 1


def test_write_inventory_emits_json_and_markdown(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    curated_root = tmp_path / "curated"
    source = data_root / "raw" / "demo"
    source.mkdir(parents=True)
    (source / "daily.csv").write_text("date,steps\n2026-01-01,100\n", encoding="utf-8")
    spec = DatasetSpec(
        dataset_id="demo", display_name="Demo", path_patterns=("raw/demo",),
        role="pretraining", female_scope="female subset", time_grain="day",
        modalities=("steps",),
    )
    profiles = profile_catalog(data_root, curated_root, (spec,))
    json_path = tmp_path / "inventory.json"
    markdown_path = tmp_path / "inventory.md"

    write_inventory(profiles, json_path, markdown_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["datasets"][0]["dataset_id"] == "demo"
    assert "FemMHC 数据集清单" in markdown_path.read_text(encoding="utf-8")
