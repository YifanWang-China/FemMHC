from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from femmhc.benchmark import (
    FEMMHC_BENCHMARK_TASKS,
    validate_benchmark_tasks,
    write_benchmark_manifest,
)


class FemMHCBenchmarkTests(unittest.TestCase):
    def test_registry_has_core_lifecycle_tasks(self) -> None:
        validate_benchmark_tasks()
        identifiers = {task.task_id for task in FEMMHC_BENCHMARK_TASKS}
        self.assertIn("mcphases_onset_24h", identifiers)
        self.assertIn("mcphases_next_day_mood_swing", identifiers)
        self.assertIn("pregnancy_gestational_age", identifiers)

    def test_swan_and_unverified_oura_are_not_headline_tasks(self) -> None:
        for task in FEMMHC_BENCHMARK_TASKS:
            if task.dataset_id in {"swan", "loneliness_oura"}:
                self.assertFalse(task.headline)

    def test_manifest_is_machine_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            json_path, markdown_path = write_benchmark_manifest(Path(directory))
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["tasks"]), len(FEMMHC_BENCHMARK_TASKS))
            self.assertIn("FemMHC 女性穿戴任务清单", markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
