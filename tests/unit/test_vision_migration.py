from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from seedling_vision.migration import legacy_predictions_to_scenes


ROOT = Path(__file__).resolve().parents[2]


class VisionMigrationTests(unittest.TestCase):
    def test_legacy_predictions_convert_to_scene_state_with_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            predictions = Path(tmp) / "predictions.json"
            predictions.write_text(json.dumps(_legacy_payload()), encoding="utf-8")

            scenes = legacy_predictions_to_scenes(
                predictions,
                dataset_version="zks_v0_1",
                ontology_version="ontology_v0_1",
                grid_rows=1,
                grid_cols=1,
            )

            self.assertEqual(len(scenes), 1)
            scene = scenes[0]
            self.assertEqual(scene.tray.bbox_xyxy_px, [0.0, 0.0, 100.0, 100.0])
            self.assertEqual(scene.cells[0].state, "multiple_crop")
            self.assertEqual(len(scene.targets), 1)
            self.assertEqual(scene.targets[0].target_type, "remove_extra_crop")

    def test_migrate_predictions_cli_writes_scene_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            predictions = Path(tmp) / "predictions.json"
            output = Path(tmp) / "scenes.json"
            predictions.write_text(json.dumps(_legacy_payload()), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_vision",
                    "migrate-predictions",
                    "--predictions",
                    str(predictions),
                    "--out",
                    str(output),
                    "--dataset-version",
                    "zks_v0_1",
                    "--grid-rows",
                    "1",
                    "--grid-cols",
                    "1",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(completed.stdout)
            saved = json.loads(output.read_text(encoding="utf-8"))
            registry = json.loads((output.parent / "artifact_registry.json").read_text(encoding="utf-8"))
            snapshot_path = output.parent / "scenes.run_snapshot.json"
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

            self.assertTrue(payload["ok"])
            self.assertTrue(payload["artifact_registry"])
            self.assertEqual(payload["run_snapshot"], str(snapshot_path))
            self.assertEqual(payload["scenes"], 1)
            self.assertEqual(saved["scenes"][0]["scene_id"], "legacy_tray001")
            self.assertEqual(snapshot["command"], "seedling-vision:migrate-predictions")
            self.assertEqual(snapshot["command_args"]["dataset_version"], "zks_v0_1")
            self.assertEqual(snapshot["metadata"]["scenes"], 1)
            self.assertEqual(registry["runs"][0]["command"], "seedling-vision:migrate-predictions")
            self.assertEqual({item["role"] for item in registry["runs"][0]["artifacts"]}, {"input", "output"})
            self.assertIn(
                ("run_snapshot", "output", str(snapshot_path)),
                {
                    (item["artifact_type"], item["role"], item["path"])
                    for item in registry["runs"][0]["artifacts"]
                },
            )
            self.assertTrue((output.parent / "artifact_registry.csv").exists())


def _legacy_payload() -> dict[str, object]:
    return {
        "images": [
            {
                "image": "tray001.jpg",
                "path": "tray001.jpg",
                "width": 100,
                "height": 100,
                "detections": [
                    {
                        "class_id": 0,
                        "name": "container",
                        "confidence": 0.99,
                        "box": [0, 0, 100, 100],
                    },
                    {
                        "class_id": 1,
                        "name": "crop_seedling",
                        "confidence": 0.9,
                        "box": [10, 10, 30, 30],
                    },
                    {
                        "class_id": 1,
                        "name": "crop_seedling",
                        "confidence": 0.8,
                        "box": [50, 50, 60, 60],
                    },
                ],
            }
        ]
    }


if __name__ == "__main__":
    unittest.main()
