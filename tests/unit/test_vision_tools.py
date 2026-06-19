from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from seedling_core.schemas import ActionTarget, DetectionObject, DetectionResultV1, InferenceContext
from seedling_vision import (
    annotate_result_uncertainty,
    batch_predict,
    container_matching_summary,
    filter_and_merge_containers,
    write_detection_overlay,
)
from seedling_vision.adapters.onnx import ONNXDetector, parse_onnx_detections
from seedling_vision.adapters.ultralytics_yolo import UltralyticsYOLODetector
from seedling_vision.adapters import MockDetector, RecordedPredictionDetector


ROOT = Path(__file__).resolve().parents[2]


class VisionToolsTests(unittest.TestCase):
    def test_uncertainty_flags_low_confidence_tiny_and_edge(self) -> None:
        result = DetectionResultV1(
            image_ref="tray001.jpg",
            image_size_px=[100, 100],
            detections=[
                DetectionObject(
                    object_id="obj_001",
                    class_name="crop_seedling",
                    class_id=1,
                    confidence=0.2,
                    bbox_xyxy_px=[0, 0, 2, 2],
                )
            ],
        )

        annotated = annotate_result_uncertainty(result)

        self.assertIn("low_confidence", annotated.detections[0].attributes)
        self.assertIn("tiny", annotated.detections[0].attributes)
        self.assertIn("touches_image_edge", annotated.detections[0].attributes)
        self.assertIn("area_fraction", annotated.detections[0].uncertainty)

    def test_uncertainty_computes_entropy_from_class_scores(self) -> None:
        result = DetectionResultV1(
            image_ref="tray001.jpg",
            image_size_px=[100, 100],
            detections=[
                DetectionObject(
                    object_id="obj_001",
                    class_name="unknown_plant",
                    class_id=2,
                    confidence=0.7,
                    bbox_xyxy_px=[10, 10, 30, 30],
                    uncertainty={"class_score_0": 0.34, "class_score_1": 0.33, "class_score_2": 0.33},
                )
            ],
        )

        annotated = annotate_result_uncertainty(result)

        uncertainty = annotated.detections[0].uncertainty
        self.assertIn("high_class_entropy", annotated.detections[0].attributes)
        self.assertGreater(uncertainty["class_entropy_normalized"], 0.99)
        self.assertAlmostEqual(uncertainty["class_probability_margin"], 0.01, places=3)

    def test_batch_predict_with_recorded_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            predictions = Path(tmp) / "predictions.json"
            predictions.write_text(json.dumps(_predictions_payload()), encoding="utf-8")
            detector = RecordedPredictionDetector()
            detector.load(str(predictions))

            results = batch_predict(detector, ["tray001.jpg"])

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].image_ref, "tray001.jpg")

    def test_batch_predict_uses_context_factory_and_progress_indexes(self) -> None:
        detector = MockDetector()
        images = [Path("tray_a.png"), Path("tray_b.png")]
        progress_rows: list[tuple[int, str]] = []

        def context_factory(image: str | Path) -> InferenceContext:
            stem = Path(image).stem
            return InferenceContext(image_id=f"context_{stem}", image_size_px=[10, 20])

        results = batch_predict(
            detector,
            images,
            context_factory=context_factory,
            progress=lambda index, image: progress_rows.append((index, Path(image).name)),
        )

        self.assertEqual(progress_rows, [(1, "tray_a.png"), (2, "tray_b.png")])
        self.assertEqual([result.image_ref for result in results], ["context_tray_a", "context_tray_b"])
        self.assertEqual([result.image_size_px for result in results], [[10, 20], [10, 20]])

    def test_vision_batch_and_overlay_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            predictions = Path(tmp) / "predictions.json"
            image_path = Path(tmp) / "tray001.jpg"
            batch_out = Path(tmp) / "batch.json"
            progress_log = Path(tmp) / "batch_progress.jsonl"
            overlay_out = Path(tmp) / "overlay.jpg"
            predictions.write_text(json.dumps(_predictions_payload()), encoding="utf-8")
            Image.new("RGB", (100, 100), "white").save(image_path)

            batch = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_vision",
                    "batch-recorded",
                    "--predictions",
                    str(predictions),
                    "--images",
                    "tray001.jpg",
                    "--out",
                    str(batch_out),
                    "--uncertainty",
                    "--progress-log",
                    str(progress_log),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            overlay = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_vision",
                    "overlay",
                    "--predictions",
                    str(predictions),
                    "--image",
                    str(image_path),
                    "--out",
                    str(overlay_out),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            batch_payload = json.loads(batch.stdout)
            overlay_payload = json.loads(overlay.stdout)
            registry = json.loads((Path(tmp) / "artifact_registry.json").read_text(encoding="utf-8"))
            batch_snapshot_path = Path(tmp) / "batch.run_snapshot.json"
            overlay_snapshot_path = Path(tmp) / "overlay.run_snapshot.json"
            batch_snapshot = json.loads(batch_snapshot_path.read_text(encoding="utf-8"))
            overlay_snapshot = json.loads(overlay_snapshot_path.read_text(encoding="utf-8"))

            self.assertTrue(batch_payload["ok"])
            self.assertTrue(batch_payload["artifact_registry"])
            self.assertEqual(batch_payload["run_snapshot"], str(batch_snapshot_path))
            self.assertTrue(overlay_payload["ok"])
            self.assertTrue(overlay_payload["artifact_registry"])
            self.assertEqual(overlay_payload["run_snapshot"], str(overlay_snapshot_path))
            self.assertTrue(batch_out.exists())
            self.assertTrue(progress_log.exists())
            self.assertTrue(overlay_out.exists())
            self.assertEqual(batch_snapshot["command"], "seedling-vision:batch-recorded")
            self.assertTrue(batch_snapshot["command_args"]["uncertainty"])
            self.assertEqual(batch_snapshot["metadata"]["images"], 1)
            self.assertEqual(overlay_snapshot["command"], "seedling-vision:overlay")
            self.assertEqual(overlay_snapshot["command_args"]["image"], str(image_path))
            self.assertTrue((Path(tmp) / "artifact_registry.csv").exists())
            self.assertEqual(
                {run["command"] for run in registry["runs"]},
                {"seedling-vision:batch-recorded", "seedling-vision:overlay"},
            )
            batch_run = next(run for run in registry["runs"] if run["command"] == "seedling-vision:batch-recorded")
            overlay_run = next(run for run in registry["runs"] if run["command"] == "seedling-vision:overlay")
            self.assertIn(str(progress_log), {item["path"] for item in batch_run["artifacts"] if item["role"] == "output"})
            self.assertIn(str(batch_snapshot_path), {item["path"] for item in batch_run["artifacts"] if item["role"] == "output"})
            self.assertIn(
                ("run_snapshot", "output", str(overlay_snapshot_path)),
                {
                    (item["artifact_type"], item["role"], item["path"])
                    for item in overlay_run["artifacts"]
                },
            )
            self.assertIn(str(image_path), {item["path"] for item in overlay_run["artifacts"] if item["role"] == "input"})
            progress_rows = [
                json.loads(line)
                for line in progress_log.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(progress_rows[0]["event"], "predict_start")
            self.assertEqual(progress_rows[0]["total"], 1)
            self.assertEqual(progress_rows[-1]["event"], "batch_complete")

    def test_vision_overlay_cli_draws_grid_targets_and_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            predictions = Path(tmp) / "predictions.json"
            errors = Path(tmp) / "overlay_errors.json"
            image_path = Path(tmp) / "tray001.png"
            overlay_out = Path(tmp) / "overlay.png"
            predictions.write_text(json.dumps(_overlay_predictions_payload()), encoding="utf-8")
            errors.write_text(
                json.dumps(
                    {
                        "errors": [
                            {
                                "image": "tray001.png",
                                "category": "target_false_negative",
                                "point_px": [75, 75],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            Image.new("RGB", (100, 100), "white").save(image_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_vision",
                    "overlay",
                    "--predictions",
                    str(predictions),
                    "--image",
                    str(image_path),
                    "--out",
                    str(overlay_out),
                    "--include-scene",
                    "--grid-rows",
                    "2",
                    "--grid-cols",
                    "2",
                    "--errors",
                    str(errors),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(completed.stdout)
            registry = json.loads((Path(tmp) / "artifact_registry.json").read_text(encoding="utf-8"))
            snapshot_path = Path(tmp) / "overlay.run_snapshot.json"
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            with Image.open(overlay_out) as rendered:
                self.assertEqual(payload["cells"], 4)
                self.assertEqual(payload["targets"], 1)
                self.assertEqual(payload["errors"], 1)
                self.assertTrue(payload["artifact_registry"])
                self.assertEqual(payload["run_snapshot"], str(snapshot_path))
                self.assertEqual(snapshot["metadata"]["errors"], 1)
                self.assertEqual(registry["runs"][0]["metadata"]["errors"], 1)
                self.assertIn(str(errors), {item["path"] for item in registry["runs"][0]["artifacts"] if item["role"] == "input"})
                self.assertIn(str(snapshot_path), {item["path"] for item in registry["runs"][0]["artifacts"] if item["role"] == "output"})
                self.assertEqual(rendered.getpixel((50, 75)), (80, 150, 240))
                self.assertEqual(rendered.getpixel((35, 35)), (240, 80, 40))
                self.assertEqual(rendered.getpixel((75, 75)), (220, 30, 50))

    def test_overlay_resolves_error_taxonomy_target_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "tray001.png"
            overlay_out = Path(tmp) / "overlay.png"
            Image.new("RGB", (64, 64), "white").save(image_path)
            result = DetectionResultV1(image_ref="tray001.png", image_size_px=[64, 64], detections=[])
            target = ActionTarget(
                target_id="pred_target_001",
                cell_id="r00_c00",
                object_id="obj_001",
                target_type="remove_extra_crop",
                action_point_px=[20, 20],
            )

            write_detection_overlay(
                image_path,
                result,
                overlay_out,
                targets=[target],
                errors=[{"category": "target_coordinate_error", "pred_target_id": "pred_target_001"}],
            )

            with Image.open(overlay_out) as rendered:
                self.assertEqual(rendered.getpixel((20, 20)), (220, 30, 50))

    def test_ultralytics_adapter_maps_mock_result_to_detection_objects(self) -> None:
        class FakeBox:
            xyxy = np.asarray([[1.0, 2.0, 11.0, 22.0]], dtype=np.float32)
            cls = np.asarray([1], dtype=np.float32)
            conf = np.asarray([0.88], dtype=np.float32)

        class FakeResult:
            names = {0: "container", 1: "crop_seedling"}
            boxes = [FakeBox()]

        class FakeYOLO:
            names = {0: "container", 1: "crop_seedling"}
            last_predict_args: dict[str, object] = {}

            def __init__(self, model_uri: str) -> None:
                self.model_uri = model_uri

            def predict(self, image_array: np.ndarray, **predict_args: object) -> list[FakeResult]:
                self.__class__.last_predict_args = predict_args
                self.image_shape = image_array.shape
                return [FakeResult()]

        fake_module = types.SimpleNamespace(YOLO=FakeYOLO)
        detector = UltralyticsYOLODetector()

        with patch.dict(sys.modules, {"ultralytics": fake_module}):
            detector.load("runs/model.pt", device="cpu")
            result = detector.predict(
                np.zeros((24, 32, 3), dtype=np.uint8),
                context=InferenceContext(image_id="tray001.png", ontology_version="ontology_v0_1"),
            )

        self.assertEqual(result.image_ref, "tray001.png")
        self.assertEqual(result.image_size_px, [32, 24])
        self.assertEqual(result.model_metadata.model_id, "model")
        self.assertEqual(result.model_metadata.backend, "ultralytics_yolo")
        self.assertEqual(result.model_metadata.ontology_version, "ontology_v0_1")
        self.assertEqual(FakeYOLO.last_predict_args["device"], "cpu")
        detection = result.detections[0]
        self.assertEqual(detection.object_id, "obj_000000")
        self.assertEqual(detection.class_name, "crop_seedling")
        self.assertEqual(detection.class_id, 1)
        self.assertAlmostEqual(detection.confidence, 0.88, places=6)
        self.assertEqual(detection.bbox_xyxy_px, [1.0, 2.0, 11.0, 22.0])

    def test_onnx_parser_maps_xyxy_rows_to_detection_objects(self) -> None:
        rows = np.asarray([[[1, 2, 11, 22, 0.9, 1], [5, 5, 8, 8, 0.1, 1]]], dtype=np.float32)

        detections = parse_onnx_detections(
            [rows],
            image_size_px=[100, 50],
            class_names=["container", "crop_seedling"],
            confidence_threshold=0.25,
        )
        adapter = ONNXDetector(config={"class_names": ["container", "crop_seedling"]})

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_name, "crop_seedling")
        self.assertEqual(detections[0].bbox_xyxy_px, [1.0, 2.0, 11.0, 22.0])
        self.assertEqual(adapter.metadata().backend, "onnxruntime")

    def test_onnx_parser_supports_channel_first_class_scores(self) -> None:
        # Two boxes, channel-first: cx, cy, w, h, class0, class1, class2.
        rows = np.asarray(
            [
                [
                    [10.0, 30.0],
                    [10.0, 20.0],
                    [6.0, 8.0],
                    [4.0, 10.0],
                    [0.1, 0.2],
                    [0.2, 0.8],
                    [0.9, 0.1],
                ]
            ],
            dtype=np.float32,
        )

        detections = parse_onnx_detections(
            rows,
            image_size_px=[100, 50],
            class_names=["container", "crop_seedling", "weed"],
            confidence_threshold=0.25,
            output_format="cxcywh_scores",
        )

        self.assertEqual(len(detections), 2)
        self.assertEqual(detections[0].class_name, "weed")
        self.assertEqual(detections[0].bbox_xyxy_px, [7.0, 8.0, 13.0, 12.0])
        self.assertAlmostEqual(detections[0].uncertainty["class_score_2"], 0.9)
        self.assertEqual(detections[1].class_name, "crop_seedling")
        self.assertEqual(detections[1].confidence, 0.800000011920929)

    def test_onnx_parser_supports_objectness_class_scores(self) -> None:
        rows = np.asarray(
            [[[1, 2, 11, 22, 0.5, 0.1, 0.8], [5, 5, 8, 8, 0.4, 0.2, 0.3]]],
            dtype=np.float32,
        )

        detections = parse_onnx_detections(
            rows,
            image_size_px=[100, 50],
            class_names=["container", "crop_seedling"],
            confidence_threshold=0.35,
            output_format="xyxy_objectness_scores",
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_name, "crop_seedling")
        self.assertAlmostEqual(detections[0].confidence, 0.4)
        self.assertAlmostEqual(detections[0].uncertainty["objectness"], 0.5)
        self.assertAlmostEqual(detections[0].uncertainty["class_score_1"], 0.8)

    def test_container_postprocess_merges_small_nearby_boxes(self) -> None:
        detections = [
            DetectionObject("container_big", "container", 0, 0.8, [0, 0, 100, 100]),
            DetectionObject("container_small", "container", 0, 0.7, [90, 90, 110, 110]),
            DetectionObject("seedling", "crop_seedling", 1, 0.9, [10, 10, 20, 20]),
        ]

        containers = filter_and_merge_containers(detections, min_area_px2=5000, merge_distance_px=100)
        summary = container_matching_summary([detections[0]], containers, min_iou=0.5)

        self.assertEqual(len(containers), 1)
        self.assertEqual(containers[0].object_id, "container_big+container_small")
        self.assertIn("merged_container", containers[0].attributes)
        self.assertEqual(summary["matched_containers"], 1)

    def test_postprocess_containers_cli_writes_postprocessed_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            predictions = Path(tmp) / "predictions.json"
            output = Path(tmp) / "postprocessed.json"
            payload = _predictions_payload()
            payload["images"][0]["detections"].insert(
                0,
                {
                    "class_id": 0,
                    "name": "container",
                    "confidence": 0.8,
                    "box": [0, 0, 100, 100],
                    "center": [50, 50],
                    "area": 10000,
                },
            )
            predictions.write_text(json.dumps(payload), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_vision",
                    "postprocess-containers",
                    "--predictions",
                    str(predictions),
                    "--out",
                    str(output),
                    "--min-area",
                    "100",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            saved = json.loads(output.read_text(encoding="utf-8"))
            cli_payload = json.loads(completed.stdout)
            registry = json.loads((Path(tmp) / "artifact_registry.json").read_text(encoding="utf-8"))
            snapshot_path = Path(tmp) / "postprocessed.run_snapshot.json"
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

            self.assertTrue(cli_payload["ok"])
            self.assertTrue(cli_payload["artifact_registry"])
            self.assertEqual(cli_payload["run_snapshot"], str(snapshot_path))
            self.assertEqual(len(saved["images"][0]["containers"]), 1)
            self.assertEqual(snapshot["command"], "seedling-vision:postprocess-containers")
            self.assertEqual(snapshot["metadata"]["min_area"], 100.0)
            self.assertEqual(registry["runs"][0]["command"], "seedling-vision:postprocess-containers")
            self.assertEqual(registry["runs"][0]["metadata"]["min_area"], 100.0)
            self.assertIn(str(snapshot_path), {item["path"] for item in registry["runs"][0]["artifacts"] if item["role"] == "output"})


def _predictions_payload() -> dict[str, object]:
    return {
        "images": [
            {
                "image": "tray001.jpg",
                "path": "tray001.jpg",
                "width": 100,
                "height": 100,
                "detections": [
                    {
                        "class_id": 1,
                        "name": "crop_seedling",
                        "confidence": 0.9,
                        "box": [10, 10, 30, 30],
                        "center": [20, 20],
                        "area": 400,
                    }
                ],
            }
        ]
    }


def _overlay_predictions_payload() -> dict[str, object]:
    return {
        "images": [
            {
                "image": "tray001.png",
                "path": "tray001.png",
                "width": 100,
                "height": 100,
                "detections": [
                    {
                        "object_id": "tray",
                        "class_id": 0,
                        "name": "container",
                        "confidence": 0.95,
                        "box": [0, 0, 100, 100],
                        "center": [50, 50],
                        "area": 10000,
                    },
                    {
                        "object_id": "obj_keep",
                        "class_id": 1,
                        "name": "crop_seedling",
                        "confidence": 0.9,
                        "box": [10, 10, 30, 30],
                        "center": [20, 20],
                        "area": 400,
                    },
                    {
                        "object_id": "obj_remove",
                        "class_id": 1,
                        "name": "crop_seedling",
                        "confidence": 0.85,
                        "box": [30, 30, 40, 40],
                        "center": [35, 35],
                        "area": 100,
                    },
                ],
            }
        ]
    }


if __name__ == "__main__":
    unittest.main()
