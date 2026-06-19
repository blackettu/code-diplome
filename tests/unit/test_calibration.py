from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
from seedling_calibration import (
    CalibrationArtifact,
    CalibrationConfig,
    CalibrationValidator,
    CameraIntrinsicsArtifact,
    ErrorSummary,
    IntrinsicsObservation,
    attach_calibration_to_scene,
    attach_calibration_to_target,
    compute_error_budget,
    estimate_calibration_artifact,
    estimate_camera_intrinsics,
    image_px_to_tray_mm,
    load_intrinsics_observations,
    tray_mm_to_robot_frame_mm,
    validate_camera_intrinsics,
)
from seedling_core.schemas import ActionTarget, SceneState, TrayState


ROOT = Path(__file__).resolve().parents[2]


class CalibrationTests(unittest.TestCase):
    def _artifact(self, valid_until: str = "2099-01-01T00:00:00+00:00") -> CalibrationArtifact:
        return CalibrationArtifact(
            calibration_id="calib_001",
            created_at="2026-06-17T10:00:00+00:00",
            camera_id="cam01",
            tray_type="11x11",
            image_to_tray_homography=[
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 1.0],
            ],
            tray_to_robot_transform=[
                [1.0, 0.0, 10.0],
                [0.0, 1.0, 20.0],
                [0.0, 0.0, 1.0],
            ],
            tool_offset_mm=[1.0, 2.0, 3.0],
            px_per_mm_x=2.0,
            px_per_mm_y=2.0,
            valid_until=valid_until,
            error_summary_mm=ErrorSummary(p50=0.5, p95=1.5, p99=2.5, rms=0.9),
        )

    def test_image_and_robot_transforms(self) -> None:
        artifact = self._artifact()

        tray_point = image_px_to_tray_mm([20.0, 40.0], artifact)
        robot_point = tray_mm_to_robot_frame_mm(tray_point, artifact)

        self.assertEqual(tray_point, [10.0, 20.0])
        self.assertEqual(robot_point, [21.0, 42.0, 3.0])

    def test_attach_calibration_to_target(self) -> None:
        artifact = self._artifact()
        target = ActionTarget(
            target_id="target_001",
            cell_id="r00_c00",
            object_id="obj_001",
            target_type="remove_extra_crop",
            action_point_px=[20.0, 40.0],
        )

        converted = attach_calibration_to_target(target, artifact)

        self.assertEqual(converted.action_point_mm, [10.0, 20.0])
        self.assertEqual(converted.robot_point_mm, [21.0, 42.0, 3.0])
        self.assertEqual(converted.uncertainty_radius_mm, 1.5)

    def test_attach_calibration_to_scene_updates_targets_and_tray_id(self) -> None:
        artifact = self._artifact()
        scene = SceneState(
            scene_id="scene_001",
            image_ref="tray001.jpg",
            dataset_version="zks_v0_1",
            ontology_version="ontology_v0_1",
            image_size_px=[100, 100],
            tray=TrayState("tray001", 1, 1, bbox_xyxy_px=[0, 0, 100, 100]),
            targets=[
                ActionTarget(
                    target_id="target_001",
                    cell_id="r00_c00",
                    object_id="obj_001",
                    target_type="remove_extra_crop",
                    action_point_px=[20.0, 40.0],
                )
            ],
        )

        converted = attach_calibration_to_scene(scene, artifact)

        self.assertIsNone(scene.tray.calibration_id)
        self.assertIsNone(scene.targets[0].action_point_mm)
        self.assertEqual(converted.tray.calibration_id, "calib_001")
        self.assertEqual(converted.targets[0].action_point_mm, [10.0, 20.0])
        self.assertEqual(converted.targets[0].robot_point_mm, [21.0, 42.0, 3.0])

    def test_validator_rejects_expired_or_high_error_calibration(self) -> None:
        artifact = self._artifact(valid_until="2026-01-01T00:00:00+00:00")
        validator = CalibrationValidator(
            expected_tray_type="11x11",
            max_error_p95_mm=1.0,
            now=datetime(2026, 6, 17, tzinfo=timezone.utc),
        )

        result = validator.validate(artifact)

        self.assertFalse(result.ok)
        self.assertIn("calibration expired at 2026-01-01T00:00:00+00:00", result.errors)
        self.assertIn("error_summary_mm.p95=1.5 exceeds limit 1.0", result.errors)

    def test_validator_rejects_wrong_tray_type_and_rms_error(self) -> None:
        artifact = self._artifact()
        validator = CalibrationValidator(
            expected_tray_type="wrong_tray",
            max_error_p95_mm=2.0,
            max_error_rms_mm=0.5,
            now=datetime(2026, 6, 17, tzinfo=timezone.utc),
        )

        result = validator.validate(artifact)

        self.assertFalse(result.ok)
        self.assertIn("tray_type mismatch: expected 'wrong_tray', got '11x11'", result.errors)
        self.assertIn("error_summary_mm.rms=0.9 exceeds limit 0.5", result.errors)

    def test_estimate_calibration_artifact_from_four_points(self) -> None:
        config = _calibration_config()

        artifact = estimate_calibration_artifact(
            config,
            calibration_id="calib_estimated",
            created_at="2026-06-17T10:00:00+00:00",
        )

        self.assertEqual(artifact.calibration_id, "calib_estimated")
        self.assertAlmostEqual(artifact.error_summary_mm.max or 0.0, 0.0, places=6)
        converted = image_px_to_tray_mm([20.0, 40.0], artifact)
        self.assertAlmostEqual(converted[0], 10.0, places=6)
        self.assertAlmostEqual(converted[1], 20.0, places=6)

    def test_calibration_estimate_and_error_map_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "calibration_config.json"
            calibration_path = Path(tmp) / "calibration.json"
            error_map_path = Path(tmp) / "error_map.json"
            error_map_html_path = Path(tmp) / "error_map.html"
            config_path.write_text(json.dumps(_calibration_config().to_dict()), encoding="utf-8")

            estimate = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_calibration",
                    "estimate",
                    "--config",
                    str(config_path),
                    "--out",
                    str(calibration_path),
                    "--calibration-id",
                    "calib_cli",
                    "--created-at",
                    "2026-06-17T10:00:00+00:00",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            error_map = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_calibration",
                    "error-map",
                    "--config",
                    str(config_path),
                    "--calibration",
                    str(calibration_path),
                    "--out",
                    str(error_map_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            error_map_html = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_calibration",
                    "error-map",
                    "--config",
                    str(config_path),
                    "--calibration",
                    str(calibration_path),
                    "--out",
                    str(error_map_html_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            estimate_payload = json.loads(estimate.stdout)
            error_payload = json.loads(error_map.stdout)
            error_html_payload = json.loads(error_map_html.stdout)

            self.assertTrue(estimate_payload["ok"])
            self.assertTrue(error_payload["ok"])
            self.assertTrue(error_html_payload["ok"])
            self.assertTrue(estimate_payload["artifact_registry"])
            self.assertTrue(error_payload["artifact_registry"])
            calibration_snapshot_path = Path(tmp) / "calibration.run_snapshot.json"
            error_map_snapshot_path = Path(tmp) / "error_map.run_snapshot.json"
            calibration_snapshot = json.loads(calibration_snapshot_path.read_text(encoding="utf-8"))
            error_map_snapshot = json.loads(error_map_snapshot_path.read_text(encoding="utf-8"))
            self.assertEqual(estimate_payload["run_snapshot"], str(calibration_snapshot_path))
            self.assertEqual(error_payload["run_snapshot"], str(error_map_snapshot_path))
            self.assertEqual(error_html_payload["run_snapshot"], str(error_map_snapshot_path))
            self.assertEqual(calibration_snapshot["command"], "seedling-calibration:estimate")
            self.assertEqual(calibration_snapshot["metadata"]["calibration_id"], "calib_cli")
            self.assertEqual(error_map_snapshot["command"], "seedling-calibration:error-map")
            self.assertEqual(error_map_snapshot["metadata"]["format"], "html")
            self.assertTrue(calibration_path.exists())
            self.assertTrue(error_map_path.exists())
            self.assertTrue(error_map_html_path.exists())
            registry = json.loads((Path(tmp) / "artifact_registry.json").read_text(encoding="utf-8"))
            self.assertEqual(
                {run["command"] for run in registry["runs"]},
                {"seedling-calibration:estimate", "seedling-calibration:error-map"},
            )
            self.assertIn("input", {artifact["role"] for run in registry["runs"] for artifact in run["artifacts"]})
            self.assertIn("output", {artifact["role"] for run in registry["runs"] for artifact in run["artifacts"]})
            self.assertIn(
                ("run_snapshot", "output", str(calibration_snapshot_path)),
                {
                    (artifact["artifact_type"], artifact["role"], artifact["path"])
                    for run in registry["runs"]
                    for artifact in run["artifacts"]
                },
            )
            self.assertIn(
                ("run_snapshot", "output", str(error_map_snapshot_path)),
                {
                    (artifact["artifact_type"], artifact["role"], artifact["path"])
                    for run in registry["runs"]
                    for artifact in run["artifacts"]
                },
            )
            self.assertEqual(len(error_payload["rows"]), 4)
            html = error_map_html_path.read_text(encoding="utf-8")
            self.assertIn("Calibration Error Map", html)
            self.assertIn("Calibration residual heatmap", html)
            self.assertIn("<table>", html)

    def test_calibration_validate_cli_writes_report_and_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calibration_path = Path(tmp) / "calibration.json"
            validation_path = Path(tmp) / "calibration_validation.json"
            self._artifact().to_json(calibration_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_calibration",
                    "validate",
                    "--calibration",
                    str(calibration_path),
                    "--tray-type",
                    "11x11",
                    "--max-p95-mm",
                    "2.0",
                    "--out",
                    str(validation_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(completed.stdout)
            saved = json.loads(validation_path.read_text(encoding="utf-8"))
            snapshot_path = Path(tmp) / "calibration_validation.run_snapshot.json"
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            registry = json.loads((Path(tmp) / "artifact_registry.json").read_text(encoding="utf-8"))

            self.assertTrue(payload["ok"])
            self.assertTrue(saved["ok"])
            self.assertEqual(payload["run_snapshot"], str(snapshot_path))
            self.assertEqual(snapshot["command"], "seedling-calibration:validate")
            self.assertEqual(snapshot["metadata"]["tray_type"], "11x11")
            self.assertEqual(registry["runs"][0]["command"], "seedling-calibration:validate")
            self.assertIn(str(snapshot_path), {artifact["path"] for artifact in registry["runs"][0]["artifacts"]})

    def test_error_budget_uses_calibration_p95_and_reports_missing_components(self) -> None:
        report = compute_error_budget(
            {
                "e_detection": 1.0,
                "e_grid": 0.5,
                "e_calibration": None,
                "e_mechanics": 0.8,
            },
            calibration=self._artifact(),
            max_total_mm=3.0,
        )

        self.assertFalse(report.ok)
        self.assertFalse(report.complete)
        self.assertEqual(report.components_mm["e_calibration"], 1.5)
        self.assertIn("e_focus", report.missing_components)
        self.assertAlmostEqual(report.total_error_mm or 0.0, (1.0**2 + 0.5**2 + 1.5**2 + 0.8**2) ** 0.5)
        self.assertEqual(report.metadata["e_calibration_source"], "calibration.error_summary_mm.p95")
        self.assertEqual(report.to_dict()["schema_version"], "error_budget_v0_1")

    def test_error_budget_cli_writes_report_and_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            components_path = Path(tmp) / "error_budget_components.json"
            calibration_path = Path(tmp) / "calibration.json"
            report_path = Path(tmp) / "error_budget.json"
            self._artifact().to_json(calibration_path)
            components_path.write_text(
                json.dumps(
                    {
                        "components_mm": {
                            "e_detection": 1.0,
                            "e_grid": 0.5,
                            "e_calibration": None,
                            "e_mechanics": 0.8,
                            "e_focus": 0.4,
                            "e_latency": 0.2,
                            "e_biological_target": 1.2,
                        }
                    }
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_calibration",
                    "error-budget",
                    "--components",
                    str(components_path),
                    "--calibration",
                    str(calibration_path),
                    "--out",
                    str(report_path),
                    "--max-total-mm",
                    "3.0",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(completed.stdout)
            saved = json.loads(report_path.read_text(encoding="utf-8"))
            registry = json.loads((Path(tmp) / "artifact_registry.json").read_text(encoding="utf-8"))

            self.assertTrue(payload["ok"], payload)
            self.assertTrue(saved["complete"])
            self.assertEqual(saved["components_mm"]["e_calibration"], 1.5)
            self.assertLess(saved["total_error_mm"], 3.0)
            self.assertTrue(payload["artifact_registry"])
            snapshot_path = Path(tmp) / "error_budget.run_snapshot.json"
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_snapshot"], str(snapshot_path))
            self.assertEqual(snapshot["command"], "seedling-calibration:error-budget")
            self.assertEqual(snapshot["metadata"]["max_total_mm"], 3.0)
            self.assertEqual(registry["runs"][0]["command"], "seedling-calibration:error-budget")
            self.assertEqual([artifact["role"] for artifact in registry["runs"][0]["artifacts"]], ["input", "input", "output", "output"])
            self.assertIn(str(snapshot_path), {artifact["path"] for artifact in registry["runs"][0]["artifacts"]})

    def test_camera_intrinsics_artifact_validates(self) -> None:
        observation = IntrinsicsObservation(
            observation_id="obs_001",
            target_type="chessboard",
            image_points_px=[[0, 0], [100, 0], [100, 100], [0, 100]],
            object_points_mm=[[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]],
            image_size_px=[640, 480],
        )
        artifact = CameraIntrinsicsArtifact(
            camera_id="cam01",
            image_size_px=[640, 480],
            camera_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
            distortion_coefficients=[0, 0, 0, 0, 0],
            target_type=observation.target_type,
            reprojection_error_px=0.4,
            observations=[observation.observation_id],
        )

        result = validate_camera_intrinsics(artifact, max_reprojection_error_px=1.0)

        self.assertTrue(result["ok"], result)
        self.assertEqual(artifact.to_dict()["schema_version"], "camera_intrinsics_v0_1")

    def test_estimate_camera_intrinsics_uses_loaded_observations_and_cv2_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            observations_path = Path(tmp) / "intrinsics_observations.json"
            observations_path.write_text(
                json.dumps(
                    {
                        "observations": [
                            {
                                "observation_id": "obs_001",
                                "target_type": "aruco",
                                "image_points_px": [[10, 20], [110, 20], [110, 120], [10, 120]],
                                "object_points_mm": [[0, 0, 0], [20, 0, 0], [20, 20, 0], [0, 20, 0]],
                                "image_size_px": [640, 480],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            observations = load_intrinsics_observations(observations_path)

            class FakeCV2(types.ModuleType):
                def calibrateCamera(self, object_points, image_points, image_size, camera_matrix, distortion):
                    self.object_points_shape = object_points[0].shape
                    self.image_points_shape = image_points[0].shape
                    self.image_size = image_size
                    return (
                        0.42,
                        np.asarray([[500.0, 0.0, 320.0], [0.0, 510.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32),
                        np.asarray([[0.1, 0.01, 0.0, 0.0, 0.001]], dtype=np.float32),
                        None,
                        None,
                    )

            fake_cv2 = FakeCV2("cv2")
            with patch.dict(sys.modules, {"cv2": fake_cv2}):
                artifact = estimate_camera_intrinsics(observations, camera_id="cam01")

        self.assertEqual(fake_cv2.object_points_shape, (4, 3))
        self.assertEqual(fake_cv2.image_points_shape, (4, 2))
        self.assertEqual(fake_cv2.image_size, (640, 480))
        self.assertEqual(artifact.camera_id, "cam01")
        self.assertEqual(artifact.target_type, "aruco")
        self.assertEqual(artifact.observations, ["obs_001"])
        self.assertEqual(artifact.camera_matrix[0], [500.0, 0.0, 320.0])
        self.assertAlmostEqual(artifact.distortion_coefficients[0], 0.1, places=6)
        self.assertEqual(artifact.metadata["estimator"], "opencv_calibrateCamera")

    def test_intrinsics_observations_example_loads(self) -> None:
        observations = load_intrinsics_observations(ROOT / "configs/calibration/intrinsics_observations.example.json")

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].target_type, "chessboard")
        self.assertGreaterEqual(len(observations[0].image_points_px), 4)
        self.assertEqual(observations[0].image_size_px, [640, 480])

    def test_validate_intrinsics_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / "intrinsics.json"
            report_path = Path(tmp) / "intrinsics_report.json"
            CameraIntrinsicsArtifact(
                camera_id="cam01",
                image_size_px=[640, 480],
                camera_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
                distortion_coefficients=[0, 0, 0, 0, 0],
                target_type="fiducials",
                reprojection_error_px=0.4,
            ).to_json(artifact_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_calibration",
                    "validate-intrinsics",
                    "--intrinsics",
                    str(artifact_path),
                    "--max-reprojection-error-px",
                    "1.0",
                    "--out",
                    str(report_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(completed.stdout)

            self.assertTrue(payload["ok"])
            self.assertTrue(payload["artifact_registry"])
            self.assertTrue(report_path.exists())
            snapshot_path = Path(tmp) / "intrinsics_report.run_snapshot.json"
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_snapshot"], str(snapshot_path))
            self.assertEqual(snapshot["command"], "seedling-calibration:validate-intrinsics")
            self.assertEqual(snapshot["metadata"]["max_reprojection_error_px"], 1.0)
            registry = json.loads((Path(tmp) / "artifact_registry.json").read_text(encoding="utf-8"))
            self.assertEqual(registry["runs"][0]["command"], "seedling-calibration:validate-intrinsics")
            self.assertIn(str(snapshot_path), {artifact["path"] for artifact in registry["runs"][0]["artifacts"]})

def _calibration_config() -> CalibrationConfig:
    return CalibrationConfig(
        camera_id="cam01",
        tray_type="11x11",
        tray_size_mm=[50.0, 50.0],
        grid_rows=11,
        grid_cols=11,
        target_points_px=[[0, 0], [100, 0], [100, 100], [0, 100]],
        target_points_tray_mm=[[0, 0], [50, 0], [50, 50], [0, 50]],
        robot_reference_points_mm=[[10, 20, 0], [60, 20, 0], [60, 70, 0], [10, 70, 0]],
        valid_hours=8,
    )


if __name__ == "__main__":
    unittest.main()
