from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from femmhc.data import (
    DEPRESSAssessmentWindowDataset,
    InPHRSymNextDayDataset,
    prepare_depress_fitbit,
    prepare_inphrsym,
)


def _write_excel(path: Path, frame: pd.DataFrame) -> None:
    frame.to_excel(path, index=False)


def test_inphrsym_uses_raw_next_day_diary_and_filters_female(tmp_path: Path) -> None:
    source = tmp_path / "inphrsym"
    source.mkdir()
    _write_excel(
        source / "Basic research participation information.xlsx",
        pd.DataFrame(
            {
                "Non-identifying keys": ["female", "male"],
                "Gender": ["F", "M"],
            }
        ),
    )
    minute_values = ",".join(["70", "71", "-1"])
    step_values = ",".join(["2", "3", "0"])
    wearable_rows = {
        "Non-identifying keys": ["female", "female", "male"],
        "Date": ["2025-01-01", "2025-01-02", "2025-01-01"],
        "Measurement types": ["Fitbit", "Fitbit", "Fitbit"],
    }
    _write_excel(
        source / "Lifestyle - Heart rate.xlsx",
        pd.DataFrame({**wearable_rows, "Measure (-1: no value)": [minute_values] * 3}),
    )
    _write_excel(
        source / "Lifestyle - Step count.xlsx",
        pd.DataFrame({**wearable_rows, "Measure (-1: no value)": [step_values] * 3}),
    )
    _write_excel(
        source / "Lifestyle - Sleep.xlsx",
        pd.DataFrame(
            {
                "Non-identifying keys": ["female"],
                "Bedtime": ["2025-01-01 01:00:00"],
                "Wake up time": ["2025-01-01 07:00:00"],
            }
        ),
    )
    _write_excel(
        source / "Emotion Diary.xlsx",
        pd.DataFrame(
            {
                "Non-identifying keys": ["female"],
                "Date": ["2025-01-02"],
                "Positive Mood": [0],
                "Negative Mood": [-2],
                "Positive Energy": [0],
                "Negative Energy": [-1],
                "Anxiety": [2],
                "Irritability": [3],
            }
        ),
    )
    _write_excel(
        source / "Panic Diary.xlsx",
        pd.DataFrame(
            {
                "Non-identifying keys": ["female"],
                "Date": ["2025-01-02"],
            }
        ),
    )
    _write_excel(
        source / "Lifestyle - Smoking, Eating, Menstruation.xlsx",
        pd.DataFrame(
            {
                "Non-identifying keys": ["female"],
                "Date": ["2025-01-02"],
                "Menstruation": ["Y"],
            }
        ),
    )

    output = tmp_path / "processed-inphrsym"
    summary = prepare_inphrsym(
        source,
        output,
        minimum_observed_minutes=1,
        seed=42,
    )
    assert summary.female_participants_with_sensor_data == 1
    assert summary.participant_days == 2
    assert summary.target_observations["next_high_anxiety"] == 1
    assert summary.target_positives["next_reported_panic"] == 1

    dataset = InPHRSymNextDayDataset(
        output,
        split="train",
        task="next_high_anxiety",
        normalize=False,
    )
    assert len(dataset) == 1
    item = dataset[0]
    assert item["date"] == "2025-01-01"
    assert item["target_date"] == "2025-01-02"
    assert float(item["target"]) == 1.0
    assert item["sensor_values"].shape == (3, 1440)
    assert np.isclose(float(item["sensor_values"][0, 0]), np.log1p(2.0))
    dataset.close()


def _write_fitbit_day(folder: Path, date: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"Time": ["00:00:00", "00:00:30", "00:01:00"], "Heart Rate": [70, 72, 74]}
    ).to_csv(folder / f"heart{date}.csv", index=False)
    pd.DataFrame(
        {"Time": ["00:00:00", "00:01:00"], "Step": [2, 3]}
    ).to_csv(folder / f"step{date}.csv", index=False)
    pd.DataFrame(
        {
            "Time": ["00:00:00", "00:01:00"],
            "State": [1, 2],
            "Interpreted": ["Asleep", "Awake"],
        }
    ).to_csv(folder / f"sleep{date}.csv", index=False)


def test_depress_fitbit_excludes_assessment_day_from_history(tmp_path: Path) -> None:
    source = tmp_path / "depress"
    source.mkdir()
    _write_excel(
        source / "demographics.xlsx",
        pd.DataFrame({"ID": ["Fall 1", "Fall 2"], "Sex": ["Female", "Male"]}),
    )
    pd.DataFrame(
        {
            "StartDate": ["1/5/2025", "1/5/2025"],
            "ID": ["Fall 1", "Fall 2"],
            "CESD": [12, 30],
            "STAI_st": [45, 70],
        }
    ).to_csv(source / "CES-D_STAI.csv", index=False)
    pd.DataFrame(
        {
            "StartDate": ["1/5/2025", "1/5/2025"],
            "ID": ["Fall 1", "Fall 2"],
            "PSS": [2.0, 4.0],
            "Positive emotion": [30, 10],
            "Negative emotion": [20, 40],
        }
    ).to_csv(source / "PANAS_PSS.csv", index=False)
    fitbit = source / "Fitbit_extracted" / "Fitbit"
    for participant in ("Fall_1", "Fall_2"):
        for date in ("20250101", "20250102", "20250103", "20250104", "20250105"):
            _write_fitbit_day(fitbit / participant / "export", date)

    output = tmp_path / "processed-depress"
    summary = prepare_depress_fitbit(
        source,
        output,
        history_days=4,
        minimum_history_days=2,
        minimum_observed_minutes=2,
        seed=42,
    )
    assert summary.female_participants_with_daily_streams == 1
    assert summary.participant_days == 5
    assert summary.assessments_with_minimum_history == 1

    assessments = pd.read_csv(output / "assessments.csv")
    history = [int(value) for value in assessments.iloc[0].history_indices.split(";")]
    index = pd.read_csv(output / "index.csv").set_index("day_index")
    assert pd.to_datetime(index.loc[history, "date"]).max() < pd.Timestamp("2025-01-05")

    splits = json.loads((output / "participant_splits.json").read_text())
    split = next(name for name, ids in splits.items() if "Fall_1" in ids)
    dataset = DEPRESSAssessmentWindowDataset(
        output,
        split=split,
        history_days=4,
        minimum_history_days=2,
        task="cesd",
        normalize=False,
    )
    item = dataset[0]
    assert item["sensor_values"].shape == (4, 3, 1440)
    assert int(item["day_present"].sum()) == 4
    assert float(item["target"]) == 12.0
    dataset.close()
