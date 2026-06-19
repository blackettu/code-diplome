from __future__ import annotations

import unittest

from seedling_cells.cell_state import CellStateBuilder
from seedling_cells.target_generation import LargestBBoxTargetStrategy, generate_largest_bbox_targets
from seedling_core.schemas import CellState, DetectionObject, TrayState


class TargetGenerationTests(unittest.TestCase):
    def test_largest_bbox_strategy_generates_extra_crop_targets(self) -> None:
        detections = [
            DetectionObject(
                object_id="obj_keep",
                class_name="crop_seedling",
                class_id=1,
                confidence=0.9,
                bbox_xyxy_px=[0, 0, 20, 20],
            ),
            DetectionObject(
                object_id="obj_remove",
                class_name="crop_seedling",
                class_id=1,
                confidence=0.8,
                bbox_xyxy_px=[25, 0, 35, 10],
                keypoints_px={"stem_base": [30, 10]},
            ),
        ]
        tray = TrayState(tray_id="tray001", grid_rows=1, grid_cols=1, bbox_xyxy_px=[0, 0, 100, 100])
        cells = CellStateBuilder().build(tray, detections)

        targets = generate_largest_bbox_targets(cells, detections)

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].object_id, "obj_remove")
        self.assertEqual(targets[0].target_type, "remove_extra_crop")
        self.assertEqual(targets[0].point_type, "stem_base")
        self.assertEqual(targets[0].action_point_px, [30.0, 10.0])
        self.assertEqual(cells[0].keep_object_id, "obj_keep")

    def test_largest_bbox_strategy_is_configurable_and_respects_keep_object(self) -> None:
        detections = {
            "crop_keep": DetectionObject(
                object_id="crop_keep",
                class_name="pine_seedling",
                class_id=10,
                confidence=0.95,
                bbox_xyxy_px=[0, 0, 10, 10],
            ),
            "crop_remove": DetectionObject(
                object_id="crop_remove",
                class_name="pine_seedling",
                class_id=10,
                confidence=0.8,
                bbox_xyxy_px=[20, 0, 60, 40],
            ),
            "not_crop": DetectionObject(
                object_id="not_crop",
                class_name="unknown_plant",
                class_id=3,
                confidence=0.7,
                bbox_xyxy_px=[70, 0, 90, 20],
            ),
        }
        cell = CellState(
            cell_id="r00_c00",
            row=0,
            col=0,
            polygon_px=[[0, 0], [100, 0], [100, 100], [0, 100]],
            state="multiple_crop",
            object_ids=["crop_keep", "crop_remove", "not_crop"],
            keep_object_id="crop_keep",
        )
        strategy = LargestBBoxTargetStrategy(
            decision_source="custom_largest_bbox",
            crop_class_names=("pine_seedling",),
        )

        targets = strategy.generate([cell], detections)

        self.assertEqual([target.object_id for target in targets], ["crop_remove"])
        self.assertEqual(targets[0].decision_source, "custom_largest_bbox")
        self.assertEqual(targets[0].target_type, "remove_extra_crop")
        self.assertEqual(targets[0].action_point_px, [40.0, 20.0])

    def test_unknown_cell_requires_review_and_no_auto_target(self) -> None:
        detections = [
            DetectionObject(
                object_id="obj_unknown",
                class_name="unknown_plant",
                class_id=3,
                confidence=0.7,
                bbox_xyxy_px=[0, 0, 20, 20],
            )
        ]
        tray = TrayState(tray_id="tray001", grid_rows=1, grid_cols=1, bbox_xyxy_px=[0, 0, 100, 100])
        cells = CellStateBuilder().build(tray, detections)

        targets = generate_largest_bbox_targets(cells, detections)

        self.assertEqual(cells[0].state, "unknown")
        self.assertTrue(cells[0].human_review_required)
        self.assertEqual(targets, [])

    def test_weed_only_cell_generates_remove_weed_target(self) -> None:
        detections = [
            DetectionObject(
                object_id="weed_001",
                class_name="weed",
                class_id=2,
                confidence=0.85,
                bbox_xyxy_px=[20, 20, 30, 40],
                keypoints_px={"stem_base": [25, 40]},
            )
        ]
        tray = TrayState(tray_id="tray001", grid_rows=1, grid_cols=1, bbox_xyxy_px=[0, 0, 100, 100])
        cells = CellStateBuilder().build(tray, detections)

        targets = generate_largest_bbox_targets(cells, detections)

        self.assertEqual(cells[0].state, "weed_only")
        self.assertFalse(cells[0].human_review_required)
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].object_id, "weed_001")
        self.assertEqual(targets[0].target_type, "remove_weed")
        self.assertEqual(targets[0].point_type, "stem_base")
        self.assertFalse(targets[0].human_review_required)

    def test_uncertain_weed_target_requires_review(self) -> None:
        detections = [
            DetectionObject(
                object_id="weed_low_conf",
                class_name="weed",
                class_id=2,
                confidence=0.4,
                bbox_xyxy_px=[20, 20, 30, 40],
                attributes=["low_confidence"],
            )
        ]
        tray = TrayState(tray_id="tray001", grid_rows=1, grid_cols=1, bbox_xyxy_px=[0, 0, 100, 100])
        cells = CellStateBuilder().build(tray, detections)

        targets = generate_largest_bbox_targets(cells, detections)

        self.assertEqual(cells[0].state, "weed_only")
        self.assertTrue(cells[0].human_review_required)
        self.assertIn("low_confidence", cells[0].risk_flags)
        self.assertEqual(len(targets), 1)
        self.assertTrue(targets[0].human_review_required)

    def test_high_entropy_single_crop_marks_cell_for_review(self) -> None:
        detections = [
            DetectionObject(
                object_id="crop_uncertain",
                class_name="crop_seedling",
                class_id=1,
                confidence=0.7,
                bbox_xyxy_px=[20, 20, 30, 40],
                attributes=["high_class_entropy"],
            )
        ]
        tray = TrayState(tray_id="tray001", grid_rows=1, grid_cols=1, bbox_xyxy_px=[0, 0, 100, 100])

        cells = CellStateBuilder().build(tray, detections)

        self.assertEqual(cells[0].state, "single_crop")
        self.assertTrue(cells[0].human_review_required)
        self.assertIn("high_class_entropy", cells[0].risk_flags)

    def test_cell_state_builder_maps_detections_to_full_tray_grid(self) -> None:
        detections = [
            DetectionObject(
                object_id="crop_top_left",
                class_name="crop_seedling",
                class_id=1,
                confidence=0.9,
                bbox_xyxy_px=[5, 5, 15, 15],
            ),
            DetectionObject(
                object_id="weed_bottom_right",
                class_name="weed",
                class_id=2,
                confidence=0.8,
                bbox_xyxy_px=[70, 70, 80, 80],
            ),
            DetectionObject(
                object_id="outside_tray",
                class_name="crop_seedling",
                class_id=1,
                confidence=0.9,
                bbox_xyxy_px=[120, 120, 130, 130],
            ),
        ]
        tray = TrayState(tray_id="tray001", grid_rows=2, grid_cols=2, bbox_xyxy_px=[0, 0, 100, 100])

        cells = CellStateBuilder().build(tray, detections)
        by_id = {cell.cell_id: cell for cell in cells}

        self.assertEqual(len(cells), 4)
        self.assertEqual(by_id["r00_c00"].state, "single_crop")
        self.assertEqual(by_id["r00_c00"].object_ids, ["crop_top_left"])
        self.assertEqual(by_id["r01_c01"].state, "weed_only")
        self.assertEqual(by_id["r01_c01"].object_ids, ["weed_bottom_right"])
        self.assertEqual(by_id["r00_c01"].state, "empty")
        self.assertEqual(by_id["r01_c00"].state, "empty")
        self.assertNotIn("outside_tray", [object_id for cell in cells for object_id in cell.object_ids])
        self.assertTrue(all(cell.polygon_px for cell in cells))

    def test_crop_and_weed_cell_generates_reviewed_remove_weed_target(self) -> None:
        detections = [
            DetectionObject(
                object_id="crop_001",
                class_name="crop_seedling",
                class_id=1,
                confidence=0.9,
                bbox_xyxy_px=[0, 0, 20, 20],
            ),
            DetectionObject(
                object_id="weed_001",
                class_name="weed",
                class_id=2,
                confidence=0.8,
                bbox_xyxy_px=[25, 0, 35, 10],
            ),
        ]
        tray = TrayState(tray_id="tray001", grid_rows=1, grid_cols=1, bbox_xyxy_px=[0, 0, 100, 100])
        cells = CellStateBuilder().build(tray, detections)

        targets = generate_largest_bbox_targets(cells, detections)

        self.assertEqual(cells[0].state, "crop_and_weed")
        self.assertTrue(cells[0].human_review_required)
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].object_id, "weed_001")
        self.assertEqual(targets[0].target_type, "remove_weed")
        self.assertTrue(targets[0].human_review_required)

    def test_cell_builder_with_tray_corners_ignores_points_outside_polygon(self) -> None:
        detections = [
            DetectionObject(
                object_id="inside_crop",
                class_name="crop_seedling",
                class_id=1,
                confidence=0.9,
                bbox_xyxy_px=[45, 45, 55, 55],
            ),
            DetectionObject(
                object_id="outside_bbox_only_crop",
                class_name="crop_seedling",
                class_id=1,
                confidence=0.9,
                bbox_xyxy_px=[4, 9, 6, 11],
            ),
        ]
        tray = TrayState(
            tray_id="tray001",
            grid_rows=1,
            grid_cols=1,
            corners_px=[[20, 0], [100, 0], [80, 100], [0, 100]],
        )

        cells = CellStateBuilder().build(tray, detections)

        self.assertEqual(cells[0].object_ids, ["inside_crop"])
        self.assertEqual(cells[0].state, "single_crop")


if __name__ == "__main__":
    unittest.main()
