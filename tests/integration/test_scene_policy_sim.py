from __future__ import annotations

import unittest

from seedling_core.schemas import ActionTarget, CellState, RobotState, SafetyState, SceneState, TrayState
from seedling_decision.policies import RasterScanPolicy
from seedling_robot import SimulatorRobotAdapter
from seedling_sim import ActuatorErrorModel, LogicalTraySimulator, SimPlant, SimScene, SimTarget


class ScenePolicySimulationIntegrationTests(unittest.TestCase):
    def test_rule_based_policy_completes_fixture_scene_through_safety_gate(self) -> None:
        scene = SceneState(
            scene_id="scene_001",
            image_ref="image.jpg",
            dataset_version="zks",
            ontology_version="ontology",
            image_size_px=[100, 100],
            tray=TrayState("tray", 1, 1, bbox_xyxy_px=[0, 0, 100, 100]),
            robot=RobotState(position_mm=[0, 0, 0], homed=True, mode="simulation"),
            cells=[
                CellState(
                    cell_id="r00_c00",
                    row=0,
                    col=0,
                    polygon_px=[[0, 0], [100, 0], [100, 100], [0, 100]],
                    state="multiple_crop",
                )
            ],
            targets=[
                ActionTarget(
                    target_id="target_001",
                    cell_id="r00_c00",
                    object_id="plant_001",
                    target_type="remove_extra_crop",
                    action_point_px=[10, 10],
                    action_point_mm=[10, 10],
                    robot_point_mm=[10, 10, 0],
                    uncertainty_radius_mm=1.0,
                    min_distance_to_keep_mm=6.0,
                )
            ],
            safety=SafetyState(calibration_valid=True, interlock_ok=True),
        )
        sim_scene = SimScene(
            scene_id="sim_001",
            grid_rows=1,
            grid_cols=1,
            cell_size_mm=[33.0, 33.0],
            plants=[SimPlant("plant_001", 0, 0, "crop_seedling", [10.0, 10.0])],
            targets=[SimTarget("target_001", "plant_001", "remove_extra_crop", [10.0, 10.0])],
        )
        adapter = SimulatorRobotAdapter(
            simulator=LogicalTraySimulator(
                actuator_model=ActuatorErrorModel(xy_sigma_mm=0.0, drift_sigma_mm=0.0)
            ),
            sim_scene=sim_scene,
        )
        adapter.connect()
        adapter.home()
        command = RasterScanPolicy().next_action(scene)
        assert command is not None

        result = adapter.execute_command(command, scene)

        self.assertTrue(result.ok)
        self.assertTrue(result.safety_decision.allowed)
        self.assertTrue(adapter.sim_scene.plants[0].removed)


if __name__ == "__main__":
    unittest.main()
