from __future__ import annotations

import unittest

from seedling_core.schemas import (
    ActionCommand,
    DetectionObject,
    SceneState,
    TrayState,
)


class SchemaRoundTripTests(unittest.TestCase):
    def test_detection_object_computes_center_and_area(self) -> None:
        detection = DetectionObject(
            object_id="obj_001",
            class_name="crop_seedling",
            class_id=1,
            confidence=0.9,
            bbox_xyxy_px=[10, 20, 30, 60],
        )

        self.assertEqual(detection.center_px, [20.0, 40.0])
        self.assertEqual(detection.area_px2, 800.0)
        self.assertEqual(DetectionObject.from_dict(detection.to_dict()).to_dict(), detection.to_dict())

    def test_scene_state_round_trip(self) -> None:
        scene = SceneState(
            scene_id="scene_001",
            image_ref="tray001.jpg",
            dataset_version="zks_v0_1",
            ontology_version="ontology_v0_1",
            image_size_px=[640, 480],
            tray=TrayState(
                tray_id="tray001",
                grid_rows=11,
                grid_cols=11,
                bbox_xyxy_px=[0, 0, 550, 550],
            ),
            detections=[
                DetectionObject(
                    object_id="obj_001",
                    class_name="crop_seedling",
                    class_id=1,
                    confidence=0.91,
                    bbox_xyxy_px=[10, 20, 30, 60],
                )
            ],
        )

        restored = SceneState.from_dict(scene.to_dict())

        self.assertEqual(restored.to_dict(), scene.to_dict())

    def test_action_command_round_trip(self) -> None:
        command = ActionCommand(
            command_id="cmd_001",
            target_id="target_001",
            action_type="move_and_mark",
            robot_point_mm=[1, 2, 3],
            created_by_policy="test_policy",
        )

        self.assertEqual(ActionCommand.from_dict(command.to_dict()).to_dict(), command.to_dict())


if __name__ == "__main__":
    unittest.main()
