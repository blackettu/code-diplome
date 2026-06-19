from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from seedling_cells.evaluation import evaluate_scene_states
from seedling_core.schemas import ActionTarget, CellState, DetectionObject, RobotState, SceneState, TrayState


ROOT = Path(__file__).resolve().parents[2]


class SceneStateEvaluationTests(unittest.TestCase):
    def test_scene_state_evaluator_handles_richer_states_targets_and_costs(self) -> None:
        gt = _scene(
            scene_id="gt_scene",
            cells=[
                CellState("r00_c00", 0, 0, _poly(0), "multiple_crop", object_ids=["obj_remove", "obj_keep"], keep_object_id="obj_keep"),
                CellState("r00_c01", 0, 1, _poly(1), "crop_and_weed", object_ids=["weed_001"]),
                CellState("r00_c02", 0, 2, _poly(2), "unknown", object_ids=["unknown_001"]),
            ],
            targets=[
                ActionTarget(
                    "target_remove",
                    "r00_c00",
                    "obj_remove",
                    "remove_extra_crop",
                    [10, 10],
                    action_point_mm=[1, 1],
                ),
                ActionTarget(
                    "target_weed",
                    "r00_c01",
                    "weed_001",
                    "remove_weed",
                    [50, 10],
                    action_point_mm=[5, 1],
                ),
            ],
        )
        pred = _scene(
            scene_id="pred_scene",
            cells=[
                CellState("r00_c00", 0, 0, _poly(0), "multiple_crop", object_ids=["obj_remove", "obj_keep"], keep_object_id="obj_keep"),
                CellState("r00_c01", 0, 1, _poly(1), "single_crop", object_ids=["weed_001"]),
                CellState("r00_c02", 0, 2, _poly(2), "single_crop", object_ids=["unknown_001"]),
            ],
            targets=[
                ActionTarget(
                    "pred_remove",
                    "r00_c00",
                    "obj_remove",
                    "remove_extra_crop",
                    [12, 13],
                    action_point_mm=[1.2, 1.3],
                ),
                ActionTarget(
                    "pred_false",
                    "r00_c02",
                    "unknown_001",
                    "remove_extra_crop",
                    [90, 10],
                    action_point_mm=[9, 1],
                ),
            ],
        )

        metrics = evaluate_scene_states(gt, pred, target_match_distance_px=5, target_match_distance_mm=1.0)

        self.assertEqual(metrics["cell_metrics"]["total_cells"], 3)
        self.assertEqual(metrics["cell_metrics"]["richer_state_counts"]["crop_and_weed"], 1)
        self.assertEqual(metrics["target_metrics"]["tp"], 1)
        self.assertEqual(metrics["target_metrics"]["fp"], 1)
        self.assertEqual(metrics["target_metrics"]["fn"], 1)
        self.assertAlmostEqual(metrics["target_metrics"]["mean_error_px"], (2**2 + 3**2) ** 0.5)
        self.assertAlmostEqual(metrics["target_metrics"]["mean_error_mm"], (0.2**2 + 0.3**2) ** 0.5)
        self.assertEqual(metrics["target_metrics"]["expert_keep_remove"]["tp"], 1)
        self.assertGreater(metrics["cost_sensitive"]["total_cost"], 0.0)
        self.assertEqual(metrics["cost_sensitive"]["cost_breakdown"]["target_false_negative"], 12.0)
        self.assertEqual(metrics["cost_sensitive"]["cost_breakdown"]["missed_crop_and_weed"], 10.0)
        self.assertEqual(metrics["cost_sensitive"]["critical_error_counts"]["expert_false_removal"], 1)
        self.assertEqual(metrics["cost_sensitive"]["critical_error_counts"]["expert_missed_removal"], 1)
        self.assertEqual(metrics["cost_sensitive"]["critical_error_total"], 6)

    def test_scene_state_evaluator_reports_edge_and_corner_cell_metrics(self) -> None:
        gt_cells = _grid_cells(3, 3, "empty")
        pred_cells = _grid_cells(3, 3, "empty")
        pred_cells[0].state = "single_crop"  # corner error
        pred_cells[1].state = "single_crop"  # edge non-corner error
        pred_cells[4].state = "single_crop"  # center error
        gt = _scene("gt_scene", cells=gt_cells, targets=[])
        pred = _scene("pred_scene", cells=pred_cells, targets=[])

        metrics = evaluate_scene_states(gt, pred)

        self.assertAlmostEqual(metrics["cell_metrics"]["accuracy"], 6 / 9)
        self.assertEqual(metrics["cell_metrics"]["edge_cell_metrics"]["total_cells"], 8)
        self.assertEqual(metrics["cell_metrics"]["edge_cell_metrics"]["correct_cells"], 6)
        self.assertAlmostEqual(metrics["cell_metrics"]["edge_cell_metrics"]["accuracy"], 0.75)
        self.assertEqual(metrics["cell_metrics"]["corner_cell_metrics"]["total_cells"], 4)
        self.assertEqual(metrics["cell_metrics"]["corner_cell_metrics"]["correct_cells"], 3)
        self.assertEqual(metrics["cell_metrics"]["corner_cell_metrics"]["mismatches"][0]["cell_id"], "r00_c00")

    def test_evaluate_scenes_cli_writes_metrics(self) -> None:
        gt = _scene("gt_scene")
        pred = _scene("pred_scene")
        with tempfile.TemporaryDirectory() as tmp:
            gt_path = Path(tmp) / "gt.json"
            pred_path = Path(tmp) / "pred.json"
            out = Path(tmp) / "metrics.json"
            gt_path.write_text(json.dumps(gt.to_dict()), encoding="utf-8")
            pred_path.write_text(json.dumps(pred.to_dict()), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_experiments",
                    "evaluate-scenes",
                    "--gt",
                    str(gt_path),
                    "--pred",
                    str(pred_path),
                    "--out",
                    str(out),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(completed.stdout)
            registry = json.loads((Path(tmp) / "artifact_registry.json").read_text(encoding="utf-8"))

            self.assertEqual(payload["cell_metrics"]["accuracy"], 1.0)
            self.assertTrue(payload["artifact_registry"])
            self.assertTrue(out.exists())
            self.assertEqual(registry["runs"][0]["command"], "seedling-experiments:evaluate-scenes")


def _scene(
    scene_id: str,
    cells: list[CellState] | None = None,
    targets: list[ActionTarget] | None = None,
) -> SceneState:
    return SceneState(
        scene_id=scene_id,
        image_ref="tray001.jpg",
        dataset_version="zks_v0_1",
        ontology_version="ontology_v0_1",
        image_size_px=[120, 40],
        tray=TrayState("tray001", 1, 3, bbox_xyxy_px=[0, 0, 120, 40]),
        robot=RobotState(position_mm=[0, 0, 0], homed=True, mode="simulation"),
        detections=[
            DetectionObject("obj_remove", "crop_seedling", 1, 0.9, [8, 8, 12, 12]),
            DetectionObject("obj_keep", "crop_seedling", 1, 0.95, [15, 8, 22, 16]),
        ],
        cells=cells
        or [
            CellState("r00_c00", 0, 0, _poly(0), "multiple_crop", object_ids=["obj_remove", "obj_keep"], keep_object_id="obj_keep")
        ],
        targets=targets
        or [
            ActionTarget(
                "target_remove",
                "r00_c00",
                "obj_remove",
                "remove_extra_crop",
                [10, 10],
                action_point_mm=[1, 1],
            )
        ],
    )


def _poly(col: int) -> list[list[float]]:
    x1 = col * 40
    x2 = x1 + 40
    return [[x1, 0], [x2, 0], [x2, 40], [x1, 40]]


def _grid_cells(rows: int, cols: int, state: str) -> list[CellState]:
    cells = []
    for row in range(rows):
        for col in range(cols):
            x1 = col * 40
            y1 = row * 40
            cells.append(
                CellState(
                    f"r{row:02d}_c{col:02d}",
                    row,
                    col,
                    [[x1, y1], [x1 + 40, y1], [x1 + 40, y1 + 40], [x1, y1 + 40]],
                    state,
                )
            )
    return cells


if __name__ == "__main__":
    unittest.main()
