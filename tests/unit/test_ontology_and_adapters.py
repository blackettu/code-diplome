from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from seedling_data.ontology import load_ontology, validate_dataset_class_names
from seedling_core.schemas import InferenceContext
from seedling_vision.adapters.recorded_prediction import RecordedPredictionDetector


class OntologyAndAdapterTests(unittest.TestCase):
    def test_ontology_accepts_current_seedlings_alias(self) -> None:
        ontology = load_ontology("configs/ontology/ontology_v0_1.yaml")
        result = validate_dataset_class_names(ontology, ["container", "seedlings"])

        self.assertTrue(result["ok"], result)
        self.assertIn("unknown_plant", ontology.class_names())
        self.assertIn("remove_extra_crop", ontology.action_labels)

    def test_ontology_rejects_missing_required_v0_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ontology.yaml"
            path.write_text(
                "\n".join(
                    [
                        "version: ontology_v0_1",
                        "object_classes:",
                        "  0:",
                        "    name: container",
                        "  1:",
                        "    name: crop_seedling",
                        "  2:",
                        "    name: weed",
                        "cell_states: [empty, single_crop]",
                        "action_labels: [keep, remove_weed]",
                        "safety_labels: [safe_to_act]",
                        "attributes: [tiny]",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "object_classes.3 is required"):
                load_ontology(path)

    def test_ontology_rejects_duplicate_aliases_and_dataset_class_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ontology.yaml"
            path.write_text(
                "\n".join(
                    [
                        "version: ontology_v0_1",
                        "object_classes:",
                        "  0:",
                        "    name: container",
                        "    aliases: [tray, Tray]",
                        "  1:",
                        "    name: crop_seedling",
                        "  2:",
                        "    name: weed",
                        "  3:",
                        "    name: unknown_plant",
                        "cell_states: [empty, single_crop, multiple_crop, weed_only, crop_and_weed, unknown, ambiguous]",
                        "action_labels: [keep, remove_weed, remove_extra_crop, human_review_required, no_action, rescan]",
                        "safety_labels: [safe_to_act, unsafe_to_act, calibration_required]",
                        "attributes: [tiny, low_confidence]",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate value"):
                load_ontology(path)

        ontology = load_ontology("configs/ontology/ontology_v0_1.yaml")
        result = validate_dataset_class_names(ontology, ["container", "container"])

        self.assertFalse(result["ok"])
        self.assertIn("duplicates dataset class name", result["errors"][0])

    def test_recorded_prediction_detector_reads_existing_contract(self) -> None:
        payload = {
            "images": [
                {
                    "image": "tray001.jpg",
                    "path": "tray001.jpg",
                    "width": 100,
                    "height": 50,
                    "detections": [
                        {
                            "class_id": 1,
                            "name": "crop_seedling",
                            "confidence": 0.9,
                            "box": [1, 2, 11, 22],
                            "center": [6, 12],
                            "area": 200,
                        }
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            detector = RecordedPredictionDetector()
            detector.load(str(path))

            result = detector.predict("tray001.jpg", context=InferenceContext(ontology_version="ontology_v0_1"))

        self.assertEqual(result.image_size_px, [100, 50])
        self.assertEqual(result.detections[0].class_name, "crop_seedling")
        self.assertEqual(result.detections[0].center_px, [6.0, 12.0])
        self.assertEqual(result.model_metadata.ontology_version, "ontology_v0_1")
        self.assertIsNone(detector.metadata().ontology_version)


if __name__ == "__main__":
    unittest.main()
