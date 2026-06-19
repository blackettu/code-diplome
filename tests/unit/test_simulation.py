from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from seedling_core.schemas import ActionTarget, CellState, DetectionObject, RobotState, SceneState, TrayState
from seedling_sim import (
    ActuatorErrorModel,
    DetectionNoiseModel,
    DomainRandomizationConfig,
    LogicalTraySimulator,
    PlantResponseModel,
    SceneGeneratorConfig,
    SimPlant,
    SimScene,
    SimSceneDetectionNoiseModel,
    SimSceneGenerator,
    SimTarget,
    domain_randomization_preset,
    render_scene_png,
    run_policy_on_scene,
    sim_scene_from_scene_state,
)


ROOT = Path(__file__).resolve().parents[2]


class SimulationTests(unittest.TestCase):
    def test_sim_scene_round_trip(self) -> None:
        scene = SimScene(
            scene_id="sim_001",
            grid_rows=11,
            grid_cols=11,
            cell_size_mm=[33.0, 33.0],
            plants=[SimPlant("plant_001", 0, 0, "crop_seedling", [10.0, 10.0])],
            targets=[SimTarget("target_001", "plant_001", "remove_extra_crop", [10.0, 10.0])],
        )

        restored = SimScene.from_dict(scene.to_dict())

        self.assertEqual(restored.to_dict(), scene.to_dict())

    def test_logical_tray_step_removes_target_on_hit(self) -> None:
        scene = SimScene(
            scene_id="sim_001",
            grid_rows=1,
            grid_cols=1,
            cell_size_mm=[33.0, 33.0],
            plants=[SimPlant("plant_001", 0, 0, "crop_seedling", [10.0, 10.0])],
            targets=[SimTarget("target_001", "plant_001", "remove_extra_crop", [10.0, 10.0])],
        )
        simulator = LogicalTraySimulator(
            actuator_model=ActuatorErrorModel(xy_sigma_mm=0.0, drift_sigma_mm=0.0, seed=42),
            seed=42,
        )
        state = simulator.reset(scene)

        outcome = simulator.step_target(state.targets[0])

        self.assertTrue(outcome.success)
        self.assertFalse(outcome.crop_damage)
        self.assertTrue(state.plants[0].removed)
        self.assertGreater(outcome.reward, 0.0)
        self.assertIsNone(outcome.info["plant_response"])

    def test_logical_tray_can_use_plant_response_model(self) -> None:
        scene = SimScene(
            scene_id="sim_response",
            grid_rows=1,
            grid_cols=1,
            cell_size_mm=[33.0, 33.0],
            plants=[SimPlant("plant_001", 0, 0, "crop_seedling", [10.0, 10.0])],
            targets=[SimTarget("target_001", "plant_001", "remove_extra_crop", [10.0, 10.0])],
        )
        simulator = LogicalTraySimulator(
            actuator_model=ActuatorErrorModel(xy_sigma_mm=0.0, drift_sigma_mm=0.0, seed=42),
            plant_response_model=PlantResponseModel(base_success_prob=0.0, seed=42),
            seed=42,
        )
        state = simulator.reset(scene)

        outcome = simulator.step_target(state.targets[0])

        self.assertFalse(outcome.success)
        self.assertFalse(state.plants[0].removed)
        self.assertEqual(outcome.info["plant_response"]["reason"], "biological_nonresponse")

    def test_plant_response_model_requires_review_for_unknown_plants(self) -> None:
        scene = SimScene(
            scene_id="sim_unknown_response",
            grid_rows=1,
            grid_cols=1,
            cell_size_mm=[33.0, 33.0],
            plants=[SimPlant("plant_001", 0, 0, "unknown_plant", [10.0, 10.0])],
            targets=[SimTarget("target_001", "plant_001", "remove_weed", [10.0, 10.0])],
        )
        simulator = LogicalTraySimulator(
            actuator_model=ActuatorErrorModel(xy_sigma_mm=0.0, drift_sigma_mm=0.0, seed=42),
            plant_response_model=PlantResponseModel(seed=42),
            seed=42,
        )
        state = simulator.reset(scene)

        outcome = simulator.step_target(state.targets[0])

        self.assertFalse(outcome.success)
        self.assertEqual(outcome.info["plant_response"]["reason"], "unknown_requires_review")
        self.assertTrue(outcome.info["plant_response"]["review_required"])

    def test_actuator_zone_error_is_reported_in_outcome_info(self) -> None:
        scene = SimScene(
            scene_id="sim_zone",
            grid_rows=1,
            grid_cols=1,
            cell_size_mm=[33.0, 33.0],
            plants=[SimPlant("plant_001", 0, 0, "crop_seedling", [10.0, 10.0])],
            targets=[SimTarget("target_001", "plant_001", "remove_extra_crop", [10.0, 10.0])],
        )
        simulator = LogicalTraySimulator(
            actuator_model=ActuatorErrorModel(
                xy_sigma_mm=0.0,
                drift_sigma_mm=0.0,
                zone_error_prob=1.0,
                zone_error_mm=4.0,
                seed=3,
            ),
            success_radius_mm=1.0,
            seed=3,
        )
        state = simulator.reset(scene)

        outcome = simulator.step_target(state.targets[0])

        self.assertFalse(outcome.success)
        self.assertTrue(outcome.info["zone_error_active"])
        self.assertAlmostEqual(
            (outcome.info["zone_error_mm"][0] ** 2 + outcome.info["zone_error_mm"][1] ** 2) ** 0.5,
            4.0,
        )

    def test_actuator_zone_error_validates_probability(self) -> None:
        with self.assertRaisesRegex(ValueError, "zone_error_prob"):
            ActuatorErrorModel(zone_error_prob=1.5)

    def test_detection_noise_can_drop_all_detections(self) -> None:
        detections = [
            DetectionObject(
                object_id="obj_001",
                class_name="crop_seedling",
                class_id=1,
                confidence=0.9,
                bbox_xyxy_px=[0, 0, 10, 10],
            )
        ]

        noisy = DetectionNoiseModel(missed_detection_prob=1.0, seed=42).apply(detections)

        self.assertEqual(noisy, [])

    def test_detection_noise_can_add_false_positive_detection(self) -> None:
        detections = [
            DetectionObject(
                object_id="obj_001",
                class_name="crop_seedling",
                class_id=1,
                confidence=0.9,
                bbox_xyxy_px=[0, 0, 20, 20],
            )
        ]

        noisy = DetectionNoiseModel(
            bbox_center_sigma_mm=0.0,
            classification_error_prob=0.0,
            missed_detection_prob=0.0,
            false_positive_prob=1.0,
            seed=7,
        ).apply(detections)
        false_positive = [item for item in noisy if "false_positive_noise" in item.attributes]

        self.assertEqual(len(noisy), 2)
        self.assertEqual(len(false_positive), 1)
        self.assertEqual(false_positive[0].class_name, "unknown_plant")
        self.assertEqual(false_positive[0].source_model, "detection_noise")
        self.assertGreater(false_positive[0].bbox_xyxy_px[2], false_positive[0].bbox_xyxy_px[0])
        self.assertGreater(false_positive[0].bbox_xyxy_px[3], false_positive[0].bbox_xyxy_px[1])

    def test_sim_scene_detection_noise_can_miss_and_add_false_positive(self) -> None:
        scene = SimScene(
            scene_id="sim_noise",
            grid_rows=1,
            grid_cols=1,
            cell_size_mm=[33.0, 33.0],
            plants=[SimPlant("plant_001", 0, 0, "crop_seedling", [10.0, 10.0])],
            targets=[SimTarget("target_001", "plant_001", "remove_extra_crop", [10.0, 10.0])],
        )

        noisy = SimSceneDetectionNoiseModel(
            missed_detection_prob=1.0,
            false_positive_prob=1.0,
            seed=11,
        ).apply(scene)

        self.assertEqual(len(noisy.plants), 1)
        self.assertEqual(len(noisy.targets), 1)
        self.assertEqual(noisy.plants[0].class_name, "unknown_plant")
        self.assertIn("false_positive_noise", noisy.plants[0].attributes)
        self.assertEqual(noisy.targets[0].target_type, "human_review_required")
        self.assertEqual(noisy.metadata["detection_noise"]["missed_detection_prob"], 1.0)

    def test_scene_generator_can_emit_empty_single_multiple_weed_and_unknown_cells(self) -> None:
        def generated(**overrides: object) -> SimScene:
            payload = {
                "grid_rows": 1,
                "grid_cols": 1,
                "p_empty": 0.0,
                "p_single_crop": 0.0,
                "p_multiple_crop": 0.0,
                "p_weed_present": 0.0,
                "p_unknown_present": 0.0,
                "seed": 1,
            }
            payload.update(overrides)
            config = SceneGeneratorConfig(**payload)
            return SimSceneGenerator(config).generate(f"scene_{next(iter(overrides))}")

        empty = generated(p_empty=1.0)
        single = generated(p_single_crop=1.0)
        multiple = generated(p_multiple_crop=1.0, max_extra_crops=1)
        weed = generated(p_weed_present=1.0)
        unknown = generated(p_unknown_present=1.0)

        self.assertEqual(empty.plants, [])
        self.assertEqual(len(single.plants), 1)
        self.assertEqual(single.targets, [])
        self.assertGreaterEqual(len(multiple.plants), 2)
        self.assertEqual(multiple.targets[0].target_type, "remove_extra_crop")
        self.assertEqual(weed.targets[0].target_type, "remove_weed")
        self.assertEqual(unknown.plants[0].class_name, "unknown_plant")
        self.assertEqual(unknown.targets[0].target_type, "human_review_required")

    def test_image_backed_scene_from_scene_state(self) -> None:
        scene = _scene_state()

        sim_scene = sim_scene_from_scene_state(scene, cell_size_mm=[50.0, 50.0])

        self.assertEqual(sim_scene.grid_rows, 1)
        self.assertEqual(len(sim_scene.plants), 1)
        self.assertEqual(len(sim_scene.targets), 1)
        self.assertEqual(sim_scene.metadata["image_ref"], "tray001.jpg")

    def test_render_scene_png_writes_synthetic_image(self) -> None:
        scene = SimScene(
            scene_id="sim_png",
            grid_rows=1,
            grid_cols=1,
            cell_size_mm=[33.0, 33.0],
            plants=[SimPlant("plant_001", 0, 0, "crop_seedling", [10.0, 10.0])],
            targets=[SimTarget("target_001", "plant_001", "remove_extra_crop", [10.0, 10.0])],
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "scene.png"

            render_scene_png(scene, output, width_px=120, randomization=domain_randomization_preset("none"), seed=1)

            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    def test_render_scene_png_applies_calibration_drift(self) -> None:
        scene = SimScene(
            scene_id="sim_png_drift",
            grid_rows=1,
            grid_cols=1,
            cell_size_mm=[30.0, 30.0],
            plants=[SimPlant("plant_001", 0, 0, "crop_seedling", [10.0, 10.0])],
            targets=[],
        )
        with tempfile.TemporaryDirectory() as tmp:
            no_drift = Path(tmp) / "no_drift.png"
            drifted = Path(tmp) / "drifted.png"

            render_scene_png(scene, no_drift, width_px=120, randomization=domain_randomization_preset("none"), seed=1)
            render_scene_png(
                scene,
                drifted,
                width_px=120,
                randomization=DomainRandomizationConfig(calibration_drift_mm=(5.0, 0.0)),
                seed=1,
            )

            with Image.open(no_drift) as base, Image.open(drifted) as shifted:
                self.assertEqual(base.getpixel((40, 40)), (40, 145, 78))
                self.assertNotEqual(shifted.getpixel((40, 40)), (40, 145, 78))
                self.assertEqual(shifted.getpixel((60, 40)), (40, 145, 78))

    def test_run_policy_on_scene_writes_replay_summary(self) -> None:
        scene = _sim_scene_with_targets()

        summary, replay, final_scene = run_policy_on_scene(scene, "route_planning", seed=1)

        self.assertTrue(summary["ok"])
        self.assertEqual(summary["policy"], "route_planning")
        self.assertEqual(summary["steps"], 2)
        self.assertEqual(len(replay.steps), 4)
        self.assertTrue(all(target.processed for target in final_scene.targets))

    def test_sim_cli_image_backed_and_synthetic_render(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scene_state_path = Path(tmp) / "scene_state.json"
            sim_scene_path = Path(tmp) / "sim_scene.json"
            image_path = Path(tmp) / "synthetic.png"
            scene_state_path.write_text(json.dumps(_scene_state().to_dict()), encoding="utf-8")

            image_backed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_sim",
                    "image-backed-scene",
                    "--scene-state",
                    str(scene_state_path),
                    "--out",
                    str(sim_scene_path),
                    "--cell-size-mm",
                    "50,50",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            synthetic = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_sim",
                    "render-synthetic",
                    "--scene",
                    str(sim_scene_path),
                    "--out",
                    str(image_path),
                    "--preset",
                    "greenhouse_default",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            presets = subprocess.run(
                [sys.executable, "-B", "-m", "seedling_sim", "randomization-presets"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            image_payload = json.loads(image_backed.stdout)
            synthetic_payload = json.loads(synthetic.stdout)
            registry = json.loads((Path(tmp) / "artifact_registry.json").read_text(encoding="utf-8"))
            image_snapshot_path = Path(tmp) / "sim_scene.run_snapshot.json"
            synthetic_snapshot_path = Path(tmp) / "synthetic.run_snapshot.json"
            image_snapshot = json.loads(image_snapshot_path.read_text(encoding="utf-8"))
            synthetic_snapshot = json.loads(synthetic_snapshot_path.read_text(encoding="utf-8"))

            self.assertTrue(image_payload["ok"])
            self.assertTrue(image_payload["artifact_registry"])
            self.assertEqual(image_payload["run_snapshot"], str(image_snapshot_path))
            self.assertEqual(image_snapshot["command"], "seedling-sim:image-backed-scene")
            self.assertEqual(image_snapshot["metadata"]["cell_size_mm"], [50.0, 50.0])
            self.assertTrue(synthetic_payload["ok"])
            self.assertTrue(synthetic_payload["artifact_registry"])
            self.assertEqual(synthetic_payload["run_snapshot"], str(synthetic_snapshot_path))
            self.assertEqual(synthetic_snapshot["command"], "seedling-sim:render-synthetic")
            self.assertEqual(synthetic_snapshot["metadata"]["preset"], "greenhouse_default")
            self.assertIn("greenhouse_default", json.loads(presets.stdout)["presets"])
            self.assertTrue(image_path.exists())
            self.assertTrue((Path(tmp) / "artifact_registry.csv").exists())
            self.assertEqual(
                {run["command"] for run in registry["runs"]},
                {"seedling-sim:image-backed-scene", "seedling-sim:render-synthetic"},
            )
            self.assertIn(
                str(image_snapshot_path),
                {artifact["path"] for run in registry["runs"] for artifact in run["artifacts"]},
            )
            self.assertIn(
                str(synthetic_snapshot_path),
                {artifact["path"] for run in registry["runs"] for artifact in run["artifacts"]},
            )

    def test_sim_cli_policy_replay_and_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scene_path = Path(tmp) / "scene.json"
            replay_path = Path(tmp) / "replay.json"
            compare_html = Path(tmp) / "compare.html"
            play_html = Path(tmp) / "play.html"
            _sim_scene_with_targets().to_json(scene_path)

            run_policy = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_sim",
                    "run-policy",
                    "--scene",
                    str(scene_path),
                    "--policy",
                    "route_planning",
                    "--out",
                    str(replay_path),
                    "--seed",
                    "1",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            compare = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_sim",
                    "compare-policies",
                    "--scenes",
                    str(scene_path),
                    "--policies",
                    "raster_scan",
                    "route_planning",
                    "--out",
                    str(compare_html),
                    "--seed",
                    "1",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            play = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_sim",
                    "play-policy",
                    "--scene",
                    str(scene_path),
                    "--policy",
                    "route_planning",
                    "--render",
                    "html",
                    "--out",
                    str(play_html),
                    "--seed",
                    "1",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            run_payload = json.loads(run_policy.stdout)
            compare_payload = json.loads(compare.stdout)
            play_payload = json.loads(play.stdout)
            registry = json.loads((Path(tmp) / "artifact_registry.json").read_text(encoding="utf-8"))
            replay_snapshot_path = Path(tmp) / "replay.run_snapshot.json"
            compare_snapshot_path = Path(tmp) / "compare.run_snapshot.json"
            play_snapshot_path = Path(tmp) / "play.run_snapshot.json"

            self.assertTrue(run_payload["ok"])
            self.assertTrue(run_payload["artifact_registry"])
            self.assertEqual(run_payload["run_snapshot"], str(replay_snapshot_path))
            self.assertEqual(json.loads(replay_snapshot_path.read_text(encoding="utf-8"))["metadata"]["policy"], "route_planning")
            self.assertEqual(compare_payload["rows"], 2)
            self.assertTrue(compare_payload["artifact_registry"])
            self.assertEqual(compare_payload["run_snapshot"], str(compare_snapshot_path))
            self.assertEqual(json.loads(compare_snapshot_path.read_text(encoding="utf-8"))["metadata"]["policies"], ["raster_scan", "route_planning"])
            self.assertTrue(play_payload["ok"])
            self.assertTrue(play_payload["artifact_registry"])
            self.assertEqual(play_payload["run_snapshot"], str(play_snapshot_path))
            self.assertEqual(json.loads(play_snapshot_path.read_text(encoding="utf-8"))["metadata"]["render"], "html")
            self.assertTrue(replay_path.exists())
            self.assertTrue(compare_html.exists())
            self.assertTrue(compare_html.with_suffix(".json").exists())
            self.assertTrue(play_html.exists())
            self.assertEqual(
                {run["command"] for run in registry["runs"]},
                {"seedling-sim:run-policy", "seedling-sim:compare-policies", "seedling-sim:play-policy"},
            )
            compare_run = next(run for run in registry["runs"] if run["command"] == "seedling-sim:compare-policies")
            self.assertEqual([artifact["role"] for artifact in compare_run["artifacts"]], ["input", "output", "output", "output"])
            self.assertIn(str(compare_snapshot_path), {artifact["path"] for artifact in compare_run["artifacts"]})

def _scene_state() -> SceneState:
    return SceneState(
        scene_id="scene_001",
        image_ref="tray001.jpg",
        dataset_version="zks_v0_1",
        ontology_version="ontology_v0_1",
        image_size_px=[100, 100],
        tray=TrayState("tray001", 1, 1, bbox_xyxy_px=[0, 0, 100, 100]),
        robot=RobotState(position_mm=[0, 0, 0], homed=True, mode="simulation"),
        detections=[
            DetectionObject(
                object_id="obj_001",
                class_name="crop_seedling",
                class_id=1,
                confidence=0.9,
                bbox_xyxy_px=[10, 10, 30, 30],
            )
        ],
        cells=[
            CellState(
                cell_id="r00_c00",
                row=0,
                col=0,
                polygon_px=[[0, 0], [100, 0], [100, 100], [0, 100]],
                state="single_crop",
                object_ids=["obj_001"],
            )
        ],
        targets=[
            ActionTarget(
                target_id="target_001",
                cell_id="r00_c00",
                object_id="obj_001",
                target_type="remove_extra_crop",
                action_point_px=[20, 20],
                action_point_mm=[10, 10],
                uncertainty_radius_mm=1.0,
            )
        ],
    )


def _sim_scene_with_targets() -> SimScene:
    return SimScene(
        scene_id="sim_policy",
        grid_rows=1,
        grid_cols=2,
        cell_size_mm=[33.0, 33.0],
        plants=[
            SimPlant("plant_a", 0, 0, "crop_seedling", [5.0, 5.0]),
            SimPlant("plant_b", 0, 1, "weed", [20.0, 5.0]),
        ],
        targets=[
            SimTarget("target_b", "plant_b", "remove_weed", [20.0, 5.0], min_distance_to_keep_mm=6.0),
            SimTarget("target_a", "plant_a", "remove_extra_crop", [5.0, 5.0], min_distance_to_keep_mm=6.0),
        ],
    )


if __name__ == "__main__":
    unittest.main()
