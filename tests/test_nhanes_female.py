from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from femmhc.data import (
    NHANESFemaleDailyDataset,
    NHANESFemaleTemporalPairDataset,
    prepare_nhanes_female,
)


def _minute_frame(rows: list[dict[str, object]], value: object) -> pd.DataFrame:
    minutes = {f"min_{minute:04d}": value for minute in range(1, 1441)}
    return pd.DataFrame([{**row, **minutes} for row in rows])


def test_prepare_nhanes_female_filters_demographics_days_and_quality(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    csv_dir = source / "csv"
    csv_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "SEQN": ["female", "excluded", "young", "male"],
            "data_release_cycle": ["G", "G", "G", "G"],
            "gender": ["Female", "Female", "Female", "Male"],
            "age_in_years_at_screening": [30, 40, 10, 30],
        }
    ).to_csv(source / "subject-info.csv", index=False)
    rows = [
        {"SEQN": participant, "PAXDAYM": day, "PAXDAYWM": day}
        for participant in ("female", "excluded", "young", "male")
        for day in (1, 2, 3, 9)
    ]
    activity = _minute_frame(rows, 1.0)
    state = _minute_frame(rows, 1.0)
    state.loc[:, [f"min_{minute:04d}" for minute in range(721, 1441)]] = 2.0
    flags = _minute_frame(rows, "FALSE")
    flags.loc[flags["SEQN"].eq("female") & flags["PAXDAYM"].eq(2), "min_0001"] = "TRUE"
    flags.loc[
        flags["SEQN"].eq("excluded"),
        [f"min_{minute:04d}" for minute in range(1, 1441)],
    ] = "TRUE"
    activity.to_csv(
        csv_dir / "nhanes_1440_log10PAXMTSM.csv.xz", index=False, compression="xz"
    )
    state.to_csv(
        csv_dir / "nhanes_1440_PAXPREDM.csv.xz", index=False, compression="xz"
    )
    flags.to_csv(
        csv_dir / "nhanes_1440_PAXFLGSM.csv.xz", index=False, compression="xz"
    )

    output = tmp_path / "processed"
    summary = prepare_nhanes_female(
        source,
        output,
        minimum_age=12,
        minimum_valid_minutes=100,
        chunk_size=2,
        seed=42,
    )
    assert summary.female_participants == 1
    assert summary.participant_days == 2
    assert summary.sensor_shape == (2, 2, 1440)
    values = np.load(output / "sensor_values.npy")
    assert np.isnan(values[0, :, 0]).all()
    assert np.allclose(values[:, 1, 720:], 1.0)
    assert np.allclose(values[:, 1, 1:720], 0.0)

    train = NHANESFemaleDailyDataset(output, split="train")
    assert len(train) == 2
    assert train[0]["sensor_values"].shape == (2, 1440)
    pairs = NHANESFemaleTemporalPairDataset(output, split="train")
    assert len(pairs) == 1
    assert pairs[0]["second_is_later"].item() == 1.0
    pairs.close()
    train.close()
