from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path
import zipfile

import numpy as np
import torch

from femmhc.data import (
    PregnancyGADailyDataset,
    PregnancyGAProgressionPairDataset,
    PregnancyGAWindowDataset,
    fit_pregnancy_ga_normalization,
    parse_measurement_name,
    prepare_pregnancy_ga_clock,
    prepare_pregnancy_ga_processed_pickle,
)
from femmhc.tasks import PregnancyGAHead


def _mtn(start_hour: int, values: np.ndarray, light: np.ndarray | None = None) -> bytes:
    channels = [
        "<channel><name>motion</name><units>MW-Counts</units><epoch>60</epoch>"
        f"<data encoding=\"text\">{','.join(map(str, values.tolist()))},</data></channel>"
    ]
    if light is not None:
        channels.append(
            "<channel><name>Light</name><units>lux</units><epoch>60</epoch>"
            f"<data encoding=\"text\">{','.join(map(str, light.tolist()))},</data></channel>"
        )
    return (
        "<?xml version=\"1.0\"?><motionfile><log2>"
        "<change><property><name>=StartTime</name>"
        f"<content>2020-01-01 {start_hour:02d}:00:00</content></property></change>"
        f"<change>{''.join(channels)}</change></log2></motionfile>"
    ).encode("utf-8")


class PregnancyGADataTests(unittest.TestCase):
    def test_measurement_name_parser_accepts_public_variants(self) -> None:
        self.assertEqual(parse_measurement_name("1001_GA25.mtn"), ("1001", 25.0))
        self.assertEqual(parse_measurement_name("1123-GA18.mtn"), ("1123", 18.0))
        self.assertEqual(parse_measurement_name("1278_GA 9.mtn"), ("1278", 9.0))
        self.assertIsNone(parse_measurement_name("1001T1.mtn"))

    def test_prepare_and_dataset_keep_participant_splits_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "pregnancy.zip"
            minutes_per_day = 10
            days = 2
            start_hour = 1
            offset = minutes_per_day - start_hour
            source = np.arange(offset + days * minutes_per_day, dtype=np.float32)
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
                for participant in range(1000, 1010):
                    handle.writestr(
                        f"raw/{participant}_GA{10 + participant - 1000}.mtn",
                        _mtn(start_hour, source, source + 100),
                    )
                handle.writestr("raw/1001T1.mtn", _mtn(start_hour, source))
                handle.writestr("raw/1010_GA20.mtn", _mtn(start_hour, source[:2]))

            output = root / "processed"
            summary = prepare_pregnancy_ga_clock(
                archive,
                output,
                days=days,
                minutes_per_day=minutes_per_day,
                seed=42,
            )
            self.assertEqual(summary.measurements, 10)
            self.assertEqual(summary.participants, 10)
            self.assertEqual(summary.excluded["unlabelled_filename"], 1)
            self.assertEqual(summary.excluded["short_activity"], 1)

            splits = json.loads((output / "participant_splits.json").read_text())
            split_sets = [set(values) for values in splits.values()]
            self.assertFalse(split_sets[0] & split_sets[1])
            self.assertFalse(split_sets[0] & split_sets[2])
            self.assertFalse(split_sets[1] & split_sets[2])

            fit_pregnancy_ga_normalization(output)
            train_days = PregnancyGADailyDataset(output, split="train")
            train_windows = PregnancyGAWindowDataset(output, split="train")
            self.assertEqual(len(train_days), len(train_windows) * days)
            self.assertEqual(tuple(train_days[0]["sensor_values"].shape), (2, minutes_per_day))
            self.assertEqual(
                tuple(train_windows[0]["sensor_values"].shape),
                (days, 2, minutes_per_day),
            )
            self.assertTrue(bool(torch.isfinite(train_days[0]["sensor_values"]).all()))
            train_days.close()
            train_windows.close()

    def test_progression_pairs_never_cross_participants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = np.ones((3, 1, 1, 10), dtype=np.float32)
            np.save(root / "sensor_values.npy", values)
            with (root / "index.csv").open("w", encoding="utf-8", newline="") as handle:
                handle.write(
                    "measurement_index,participant_id,gestational_age_weeks,source_member\n"
                    "0,P1,10,a.mtn\n1,P1,20,b.mtn\n2,P2,30,c.mtn\n"
                )
            (root / "participant_splits.json").write_text(
                json.dumps({"train": ["P1", "P2"], "validation": [], "test": []})
            )
            (root / "schema.json").write_text(
                json.dumps({"days_per_measurement": 1, "minutes_per_day": 10})
            )
            pairs = PregnancyGAProgressionPairDataset(
                root,
                split="train",
                normalize=False,
                include_light=False,
            )
            self.assertEqual(len(pairs), 2)
            self.assertEqual(pairs[0]["first"]["participant_id"], "P1")
            self.assertEqual(pairs[0]["second"]["participant_id"], "P1")
            self.assertEqual(float(pairs[0]["second_is_later"]), 1.0)
            self.assertEqual(float(pairs[1]["second_is_later"]), 0.0)
            pairs.close()

    def test_official_processed_pickle_is_not_log_transformed_twice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "official.pkl"
            processed = {
                f"{participant}_10": {
                    "activity": np.full(10081, np.log1p(9.0), dtype=np.float64),
                    "light": np.full(10081, np.log1p(99.0), dtype=np.float64),
                    "sleep": np.zeros(10081, dtype=np.float64),
                    "t": np.arange(10081),
                }
                for participant in range(1000, 1010)
            }
            with source.open("wb") as handle:
                pickle.dump(processed, handle)
            output = root / "processed"
            summary = prepare_pregnancy_ga_processed_pickle(source, output)
            self.assertEqual(summary.measurements, 10)
            fit_pregnancy_ga_normalization(output)
            dataset = PregnancyGAWindowDataset(output, split="train")
            item = dataset[0]["sensor_values"].numpy()
            self.assertTrue(np.isfinite(item).all())
            schema = json.loads((output / "schema.json").read_text())
            self.assertTrue(schema["source_values_are_log1p"])
            dataset.close()

    def test_gestational_age_head_pools_days(self) -> None:
        head = PregnancyGAHead(embed_dim=16)
        output = head(
            torch.randn(3, 7, 16),
            day_present=torch.tensor(
                [[1, 1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]],
                dtype=torch.bool,
            ),
        )
        self.assertEqual(tuple(output.prediction.shape), (3,))
        self.assertTrue(
            torch.allclose(output.day_attention.sum(dim=-1), torch.ones(3), atol=1e-6)
        )
        self.assertEqual(float(output.day_attention[1, 2:].sum().detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
