from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from femmhc.data import WearableHRVMentalDailyDataset, prepare_wearable_hrv_mental


def test_prepare_wearable_hrv_mental_filters_female_and_distributes_steps(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    labels = {
        name: [value, value + 1, value + 2]
        for value, name in enumerate(
            (
                "ISI_1",
                "PHQ9_1",
                "GAD7_1",
                "ISI_2",
                "PHQ9_2",
                "GAD7_2",
                "ISI_F",
                "PHQ9_F",
                "GAD7_F",
            )
        )
    }
    pd.DataFrame(
        {
            "deviceId": ["female-a", "female-b", "male"],
            "sex": [2, 2, 1],
            **labels,
        }
    ).to_csv(source / "survey.csv", index=False)
    rows = []
    origin = pd.Timestamp("2025-01-01T00:00:00Z")
    for participant in ("female-a", "female-b", "male"):
        for window in range(6):
            rows.append(
                {
                    "deviceId": participant,
                    "ts_start": int((origin + pd.Timedelta(minutes=5 * window)).timestamp() * 1000),
                    "steps": 10.0,
                    "HR": 70.0,
                    "rmssd": 50.0,
                    "light_avg": 100.0,
                }
            )
    pd.DataFrame(rows).to_csv(source / "sensor_hrv_filtered.csv", index=False)

    output = tmp_path / "processed"
    summary = prepare_wearable_hrv_mental(source, output, seed=42)
    assert summary.female_participants == 2
    assert summary.participant_days == 2
    assert summary.sensor_shape == (2, 4, 1440)
    raw = np.load(output / "sensor_values.npy")
    assert np.allclose(raw[0, 0, :5], 2.0)
    assert np.isnan(raw[:, :, 30:]).all()

    split_sizes = []
    for split in ("train", "validation", "test"):
        dataset = WearableHRVMentalDailyDataset(output, split=split)
        split_sizes.append(len(dataset))
        if len(dataset):
            item = dataset[0]
            assert item["sensor_values"].shape == (4, 1440)
            assert item["channel_present"].all()
        dataset.close()
    assert sum(split_sizes) == 2
