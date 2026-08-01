from __future__ import annotations

import csv
import gc
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from femmhc.data.dataset import McPhasesTemporalPairDataset
from femmhc.data.mcphases import MCPHASES_CONTEXT_FEATURES, MCPHASES_LABEL_FIELDS


class McPhasesTemporalPairTests(unittest.TestCase):
    def test_supervised_chronology_is_stable_when_order_input_is_reversed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = np.ones((3, 2, 20), dtype=np.float32)
            labels = np.tile(
                np.arange(len(MCPHASES_LABEL_FIELDS), dtype=np.int64), (3, 1)
            )
            labels[:, 0] = np.asarray([10, 11, 12])
            np.save(root / "sensor_values.npy", values)
            np.save(root / "labels.npy", labels)
            np.save(
                root / "daily_context.npy",
                np.zeros((3, len(MCPHASES_CONTEXT_FEATURES)), dtype=np.float32),
            )
            np.save(root / "hormones.npy", np.zeros((3, 3), dtype=np.float32))
            with (root / "index.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "sample_index",
                        "participant_id",
                        "study_interval",
                        "day_in_study",
                    ],
                )
                writer.writeheader()
                for day in range(3):
                    writer.writerow(
                        {
                            "sample_index": day,
                            "participant_id": "P1",
                            "study_interval": "A",
                            "day_in_study": day,
                        }
                    )
            (root / "participant_splits.json").write_text(
                json.dumps({"train": ["P1"], "validation": [], "test": []}),
                encoding="utf-8",
            )

            dataset = McPhasesTemporalPairDataset(root, split="train", normalize=False)
            forward = dataset[0]
            reversed_input = dataset[1]

            self.assertEqual(float(forward["second_is_later"]), 1.0)
            self.assertEqual(forward["earlier"]["day_in_study"], 0)
            self.assertEqual(forward["later"]["day_in_study"], 1)
            self.assertEqual(float(reversed_input["second_is_later"]), 0.0)
            self.assertEqual(reversed_input["first"]["day_in_study"], 2)
            self.assertEqual(reversed_input["second"]["day_in_study"], 1)
            self.assertEqual(reversed_input["earlier"]["day_in_study"], 1)
            self.assertEqual(reversed_input["later"]["day_in_study"], 2)
            self.assertEqual(int(reversed_input["earlier"]["labels"][0]), 11)
            self.assertEqual(int(reversed_input["later"]["labels"][0]), 12)
            del forward, reversed_input, dataset
            gc.collect()


if __name__ == "__main__":
    unittest.main()
