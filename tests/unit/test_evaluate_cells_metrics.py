from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from seedling_calibration.schemas import CalibrationArtifact, ErrorSummary
from seedling_core.schemas import ActionTarget, CellState, RobotState, SceneState, TrayState
from seedling_experiments.config import validate_experiment_config
from seedling_experiments.evaluate import evaluate_cells_from_config
from seedling_reports.diagnostics import bootstrap_cell_metrics


class EvaluateCellsMetricsTests(unittest.TestCase):
    def test_evaluate_cells_reports_calibrated_target_error_mm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            images = dataset / "images"
            labels = dataset / "labels"
            output = root / "evaluation"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            Image.new("RGB", (100, 100), "white").save(images / "tray.png")
            (labels / "tray.txt").write_text(
                "\n".join(
                    [
                        "0 0.50000000 0.50000000 1.00000000 1.00000000",
                        "1 0.20000000 0.20000000 0.04000000 0.04000000",
                        "1 0.24000000 0.20000000 0.02000000 0.02000000",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            predictions_path = root / "predictions.json"
            predictions_path.write_text(
                json.dumps(
                    {
                        "images": [
                            {
                                "image": "tray.png",
                                "detections": [
                                    {"class_id": 0, "box": [0, 0, 100, 100], "confidence": 1.0},
                                    {"class_id": 1, "box": [18, 18, 22, 22], "confidence": 1.0},
                                    {"class_id": 1, "box": [27, 19, 29, 21], "confidence": 1.0},
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            calibration_path = root / "calibration.json"
            CalibrationArtifact(
                calibration_id="calib_test",
                created_at="2026-01-01T00:00:00+00:00",
                camera_id="cam01",
                tray_type="1x1",
                image_to_tray_homography=[[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]],
                tray_to_robot_transform=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                tool_offset_mm=[0, 0, 0],
                px_per_mm_x=2.0,
                px_per_mm_y=2.0,
                error_summary_mm=ErrorSummary(p95=0.1),
            ).to_json(calibration_path)
            config = {
                "evaluation": {
                    "dataset": str(dataset),
                    "predictions": str(predictions_path),
                    "output_dir": str(output),
                    "grid_rows": 1,
                    "grid_cols": 1,
                    "use_ground_truth_containers": True,
                    "target_match_distance_px": 10.0,
                    "target_match_distance_mm": 3.0,
                    "calibration": str(calibration_path),
                }
            }

            validate_experiment_config(config, "evaluate-cells", base_dir=root, check_paths=True)
            metrics = evaluate_cells_from_config(config, command_args={"config": str(root / "config.json")})
            written = json.loads((output / "cell_metrics.json").read_text(encoding="utf-8"))
            registry = json.loads((output / "artifact_registry.json").read_text(encoding="utf-8"))
            artifacts = {(artifact["artifact_type"], artifact["role"], Path(artifact["path"]).name) for artifact in registry["runs"][0]["artifacts"]}

            self.assertEqual(metrics["removal_targets"]["tp"], 1)
            self.assertAlmostEqual(metrics["removal_targets"]["mean_coordinate_error_px"], 4.0)
            self.assertAlmostEqual(metrics["removal_targets"]["mean_coordinate_error_mm"], 2.0)
            self.assertEqual(metrics["removal_targets"]["matched_distances_mm"], [2.0])
            self.assertEqual(metrics["removal_targets"]["calibration_id"], "calib_test")
            self.assertAlmostEqual(written["images"][0]["target_mean_coordinate_error_mm"], 2.0)
            self.assertIn(("directory", "input", "dataset"), artifacts)
            self.assertIn(("predictions", "input", "predictions.json"), artifacts)
            self.assertIn(("calibration", "input", "calibration.json"), artifacts)
            self.assertIn(("cell_metrics", "output", "cell_metrics.json"), artifacts)
            self.assertIn(("cell_confusion_matrix", "output", "cell_confusion_matrix.csv"), artifacts)

    def test_evaluate_cells_reports_legacy_cost_sensitive_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            images = dataset / "images"
            labels = dataset / "labels"
            output = root / "evaluation"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            Image.new("RGB", (100, 100), "white").save(images / "tray.png")
            (labels / "tray.txt").write_text(
                "\n".join(
                    [
                        "0 0.50000000 0.50000000 1.00000000 1.00000000",
                        "1 0.20000000 0.20000000 0.04000000 0.04000000",
                        "1 0.24000000 0.20000000 0.02000000 0.02000000",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            predictions_path = root / "predictions.json"
            predictions_path.write_text(
                json.dumps(
                    {
                        "images": [
                            {
                                "image": "tray.png",
                                "detections": [
                                    {"class_id": 0, "box": [0, 0, 100, 100], "confidence": 1.0},
                                    {"class_id": 1, "box": [18, 18, 22, 22], "confidence": 1.0},
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "evaluation": {
                    "dataset": str(dataset),
                    "predictions": str(predictions_path),
                    "output_dir": str(output),
                    "grid_rows": 1,
                    "grid_cols": 1,
                    "use_ground_truth_containers": True,
                }
            }

            validate_experiment_config(config, "evaluate-cells", base_dir=root, check_paths=True)
            metrics = evaluate_cells_from_config(config, command_args={"config": str(root / "config.json")})
            written = json.loads((output / "cell_metrics.json").read_text(encoding="utf-8"))

            self.assertEqual(metrics["removal_targets"]["fn"], 1)
            self.assertEqual(metrics["cost_sensitive"]["critical_error_counts"]["target_false_negative"], 1)
            self.assertEqual(metrics["cost_sensitive"]["critical_error_counts"]["missed_multiple_crop"], 1)
            self.assertEqual(metrics["cost_sensitive"]["critical_error_total"], 2)
            self.assertEqual(metrics["cost_sensitive"]["cost_breakdown"]["target_false_negative"], 12.0)
            self.assertEqual(metrics["cost_sensitive"]["cost_breakdown"]["missed_multiple_crop"], 8.0)
            self.assertEqual(metrics["cost_sensitive"]["total_cost"], 20.0)
            self.assertEqual(written["images"][0]["critical_error_total"], 2)

    def test_evaluate_cells_preserves_manifest_group_metadata_for_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            images = dataset / "images"
            labels = dataset / "labels"
            manifests = dataset / "manifests"
            output = root / "evaluation"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            manifests.mkdir(parents=True)
            for name in ["tray_a_001.png", "tray_a_002.png"]:
                Image.new("RGB", (100, 100), "white").save(images / name)
                (labels / f"{Path(name).stem}.txt").write_text(
                    "\n".join(
                        [
                            "0 0.50000000 0.50000000 1.00000000 1.00000000",
                            "1 0.20000000 0.20000000 0.04000000 0.04000000",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
            predictions_path = root / "predictions.json"
            predictions_path.write_text(
                json.dumps(
                    {
                        "images": [
                            {
                                "image": name,
                                "detections": [
                                    {"class_id": 0, "box": [0, 0, 100, 100], "confidence": 1.0},
                                    {"class_id": 1, "box": [18, 18, 22, 22], "confidence": 1.0},
                                ],
                            }
                            for name in ["tray_a_001.png", "tray_a_002.png"]
                        ]
                    }
                ),
                encoding="utf-8",
            )
            manifest_path = manifests / "image_manifest.csv"
            manifest_path.write_text(
                "\n".join(
                    [
                        "image_id,file_path,sha256,session_id,group_id,tray_id,split",
                        f"tray_a_001,{images / 'tray_a_001.png'},sha1,session_01,group_tray_a,tray_a,test",
                        f"tray_a_002,{images / 'tray_a_002.png'},sha2,session_01,group_tray_a,tray_a,test",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config = {
                "evaluation": {
                    "dataset": str(dataset),
                    "predictions": str(predictions_path),
                    "image_manifest": str(manifest_path),
                    "output_dir": str(output),
                    "grid_rows": 1,
                    "grid_cols": 1,
                    "use_ground_truth_containers": True,
                }
            }

            validate_experiment_config(config, "evaluate-cells", base_dir=root, check_paths=True)
            metrics = evaluate_cells_from_config(config, command_args={"config": str(root / "config.json")})

            self.assertEqual([row["group_id"] for row in metrics["images"]], ["group_tray_a", "group_tray_a"])
            self.assertEqual(metrics["images"][0]["tray_id"], "tray_a")
            self.assertEqual(metrics["images"][1]["session_id"], "session_01")
            self.assertEqual(bootstrap_cell_metrics(metrics["images"], iterations=10, seed=1)["groups"], 2)
            self.assertEqual(bootstrap_cell_metrics(metrics["images"], iterations=10, seed=1, group_key="group_id")["groups"], 1)

    def test_evaluate_cells_can_use_scene_state_schema_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "evaluation"
            gt_scene_path = root / "gt_scene.json"
            pred_scene_path = root / "pred_scene.json"
            gt_scene = _scene_state(
                scene_id="gt_scene",
                cell_state="multiple_crop",
                target_id="gt_target",
                action_point_px=[10, 10],
            )
            pred_scene = _scene_state(
                scene_id="pred_scene",
                cell_state="multiple_crop",
                target_id="pred_target",
                action_point_px=[14, 10],
            )
            gt_scene_path.write_text(json.dumps(gt_scene.to_dict()), encoding="utf-8")
            pred_scene_path.write_text(json.dumps(pred_scene.to_dict()), encoding="utf-8")
            config = {
                "evaluation": {
                    "gt_scene": str(gt_scene_path),
                    "pred_scene": str(pred_scene_path),
                    "output_dir": str(output),
                    "target_match_distance_px": 10.0,
                }
            }

            validate_experiment_config(config, "evaluate-cells", base_dir=root, check_paths=True)
            metrics = evaluate_cells_from_config(config, command_args={"config": str(root / "config.json")})
            written = json.loads((output / "cell_metrics.json").read_text(encoding="utf-8"))
            registry = json.loads((output / "artifact_registry.json").read_text(encoding="utf-8"))
            artifacts = {(artifact["artifact_type"], artifact["role"], Path(artifact["path"]).name) for artifact in registry["runs"][0]["artifacts"]}

            self.assertEqual(metrics["schema_mode"], "scene_state")
            self.assertEqual(metrics["cell_accuracy"], 1.0)
            self.assertEqual(metrics["cell_confusion_matrix"][2][2], 1)
            self.assertEqual(metrics["removal_targets"]["tp"], 1)
            self.assertAlmostEqual(metrics["removal_targets"]["mean_coordinate_error_px"], 4.0)
            self.assertEqual(metrics["removal_targets"]["matched_distances_px"], [4.0])
            self.assertEqual(written["scene_state_metrics"]["target_metrics"]["tp"], 1)
            self.assertIn(("gt_scene", "input", "gt_scene.json"), artifacts)
            self.assertIn(("pred_scene", "input", "pred_scene.json"), artifacts)
            self.assertIn(("cell_metrics", "output", "cell_metrics.json"), artifacts)

def _scene_state(
    scene_id: str,
    cell_state: str,
    target_id: str,
    action_point_px: list[float],
) -> SceneState:
    return SceneState(
        scene_id=scene_id,
        image_ref=f"{scene_id}.png",
        dataset_version="zks_v0_1",
        ontology_version="ontology_v0_1",
        image_size_px=[40, 40],
        tray=TrayState(tray_id="tray001", grid_rows=1, grid_cols=1, bbox_xyxy_px=[0, 0, 40, 40]),
        robot=RobotState(position_mm=[0, 0, 0], homed=True, mode="simulation"),
        cells=[
            CellState(
                cell_id="r00_c00",
                row=0,
                col=0,
                polygon_px=[[0, 0], [40, 0], [40, 40], [0, 40]],
                state=cell_state,
            )
        ],
        targets=[
            ActionTarget(
                target_id=target_id,
                cell_id="r00_c00",
                object_id="obj_remove",
                target_type="remove_extra_crop",
                action_point_px=action_point_px,
            )
        ],
    )


if __name__ == "__main__":
    unittest.main()
