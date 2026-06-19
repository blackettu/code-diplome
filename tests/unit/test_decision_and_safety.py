from __future__ import annotations

import math
import unittest
from pathlib import Path

from seedling_core.config import ConfigError
from seedling_core.schemas import (
    ActionCommand,
    ActionTarget,
    CellState,
    RobotState,
    SafetyState,
    SceneState,
    TrayState,
)
from seedling_decision.action_masks import ActionMaskBuilder
from seedling_decision.policies import (
    HumanReviewPolicy,
    NearestNeighborPolicy,
    NoOpPolicy,
    RasterScanPolicy,
    RiskAwareRulePolicy,
    RoutePlanningPolicy,
)
from seedling_decision.safety_gate import (
    ALLOW_SIMULATION,
    BLOCK_CALIBRATION_REQUIRED,
    BLOCK_INTERLOCK,
    BLOCK_REVIEW_REQUIRED,
    BLOCK_SYSTEM_ERROR,
    BLOCK_UNSAFE_TARGET,
    RobotTelemetry,
    SafetyGate,
    SafetyLimits,
    load_safety_limits,
    safety_limits_from_mapping,
)


ROOT = Path(__file__).resolve().parents[2]


class DecisionAndSafetyTests(unittest.TestCase):
    def _scene(self, calibration_valid: bool = True, homed: bool = True) -> SceneState:
        cells = [
            CellState(
                cell_id="r00_c00",
                row=0,
                col=0,
                polygon_px=[[0, 0], [10, 0], [10, 10], [0, 10]],
                state="multiple_crop",
            ),
            CellState(
                cell_id="r00_c01",
                row=0,
                col=1,
                polygon_px=[[10, 0], [20, 0], [20, 10], [10, 10]],
                state="unknown",
                human_review_required=True,
                risk_flags=["unknown_plant_present"],
            ),
        ]
        targets = [
            ActionTarget(
                target_id="target_safe",
                cell_id="r00_c00",
                object_id="obj_safe",
                target_type="remove_extra_crop",
                action_point_px=[5, 5],
                action_point_mm=[5, 5],
                robot_point_mm=[5, 5, 0],
                uncertainty_radius_mm=1.0,
                min_distance_to_keep_mm=6.0,
            ),
            ActionTarget(
                target_id="target_unknown",
                cell_id="r00_c01",
                object_id="obj_unknown",
                target_type="remove_extra_crop",
                action_point_px=[15, 5],
                action_point_mm=[15, 5],
                robot_point_mm=[15, 5, 0],
                human_review_required=True,
                uncertainty_radius_mm=1.0,
            ),
        ]
        return SceneState(
            scene_id="scene_001",
            image_ref="tray001.jpg",
            dataset_version="zks_v0_1",
            ontology_version="ontology_v0_1",
            image_size_px=[100, 100],
            tray=TrayState(tray_id="tray001", grid_rows=1, grid_cols=2, bbox_xyxy_px=[0, 0, 20, 10]),
            robot=RobotState(position_mm=[0, 0, 0], homed=homed, mode="simulation"),
            cells=cells,
            targets=targets,
            safety=SafetyState(calibration_valid=calibration_valid, interlock_ok=False, software_safe_mode=True),
        )

    def test_noop_policy_sends_all_targets_to_review(self) -> None:
        scene = self._scene()

        plan = NoOpPolicy().propose_plan(scene)

        self.assertEqual(plan.commands, [])
        self.assertEqual(plan.review_target_ids, ["target_safe", "target_unknown"])
        self.assertEqual(plan.metadata["review_reasons_by_target"]["target_safe"], ["noop_policy_requires_review"])

    def test_human_review_policy_surfaces_review_and_blocked_targets(self) -> None:
        scene = self._scene()
        scene.targets[0].forbidden_zone_ids = ["keep_seedling_zone"]

        plan = HumanReviewPolicy().propose_plan(scene)

        self.assertEqual(plan.commands, [])
        self.assertEqual(plan.review_target_ids, ["target_safe", "target_unknown"])
        reasons = plan.metadata["review_reasons_by_target"]
        self.assertIn("forbidden_zone_overlap", reasons["target_safe"])
        self.assertIn("human_review_required", reasons["target_unknown"])
        self.assertIn("unknown_plant_present", reasons["target_unknown"])

    def test_raster_policy_commands_only_valid_targets(self) -> None:
        scene = self._scene()

        plan = RasterScanPolicy().propose_plan(scene)

        self.assertEqual([command.target_id for command in plan.commands], ["target_safe"])
        self.assertIn("target_unknown", plan.review_target_ids)
        self.assertIn("human_review_required", plan.metadata["review_reasons_by_target"]["target_unknown"])

    def test_nearest_neighbor_policy_orders_safe_targets_by_robot_distance(self) -> None:
        scene = self._scene()
        scene.targets = [
            ActionTarget(
                target_id="target_far",
                cell_id="r00_c00",
                object_id="obj_far",
                target_type="remove_extra_crop",
                action_point_px=[18, 8],
                action_point_mm=[18, 8],
                robot_point_mm=[18, 8, 0],
                uncertainty_radius_mm=1.0,
                min_distance_to_keep_mm=6.0,
            ),
            ActionTarget(
                target_id="target_near",
                cell_id="r00_c00",
                object_id="obj_near",
                target_type="remove_extra_crop",
                action_point_px=[3, 4],
                action_point_mm=[3, 4],
                robot_point_mm=[3, 4, 0],
                uncertainty_radius_mm=1.0,
                min_distance_to_keep_mm=6.0,
            ),
        ]

        plan = NearestNeighborPolicy().propose_plan(scene)

        self.assertEqual([command.target_id for command in plan.commands], ["target_near", "target_far"])

    def test_route_planning_policy_uses_cumulative_greedy_route(self) -> None:
        scene = self._scene()
        scene.targets = [
            ActionTarget(
                target_id="target_a",
                cell_id="r00_c00",
                object_id="obj_a",
                target_type="remove_extra_crop",
                action_point_px=[5, 0],
                action_point_mm=[5, 0],
                robot_point_mm=[5, 0, 0],
                uncertainty_radius_mm=1.0,
                min_distance_to_keep_mm=6.0,
            ),
            ActionTarget(
                target_id="target_b",
                cell_id="r00_c00",
                object_id="obj_b",
                target_type="remove_extra_crop",
                action_point_px=[6, 0],
                action_point_mm=[6, 0],
                robot_point_mm=[6, 0, 0],
                uncertainty_radius_mm=1.0,
                min_distance_to_keep_mm=6.0,
            ),
            ActionTarget(
                target_id="target_c",
                cell_id="r00_c00",
                object_id="obj_c",
                target_type="remove_extra_crop",
                action_point_px=[4, 4],
                action_point_mm=[4, 4],
                robot_point_mm=[4, 4, 0],
                uncertainty_radius_mm=1.0,
                min_distance_to_keep_mm=6.0,
            ),
        ]

        plan = RoutePlanningPolicy().propose_plan(scene)

        self.assertEqual([command.target_id for command in plan.commands], ["target_a", "target_b", "target_c"])
        self.assertEqual(plan.commands[0].metadata["route_planning_method"], "greedy_nearest_neighbor")
        self.assertAlmostEqual(plan.commands[0].metadata["leg_distance_mm"], 5.0)
        self.assertAlmostEqual(plan.commands[1].metadata["leg_distance_mm"], 1.0)
        self.assertAlmostEqual(plan.commands[-1].metadata["cumulative_route_distance_mm"], 6.0 + math.sqrt(20.0))

    def test_risk_aware_policy_requires_homed_robot(self) -> None:
        scene = self._scene(homed=False)

        plan = RiskAwareRulePolicy().propose_plan(scene)

        self.assertEqual(plan.commands, [])
        self.assertIn("target_safe", plan.blocked_target_ids)
        self.assertIn("robot_not_homed", plan.metadata["blocked_reasons_by_target"]["target_safe"])

    def test_risk_aware_policy_blocks_forbidden_zone_and_outside_tray(self) -> None:
        scene = self._scene()
        scene.targets[0].forbidden_zone_ids = ["keep_seedling_zone"]
        scene.targets[0].action_point_px = [500.0, 500.0]

        plan = RiskAwareRulePolicy().propose_plan(scene)

        self.assertEqual(plan.commands, [])
        self.assertIn("target_safe", plan.blocked_target_ids)
        reasons = plan.metadata["blocked_reasons_by_target"]["target_safe"]
        self.assertIn("forbidden_zone_overlap", reasons)
        self.assertIn("target_outside_tray", reasons)

    def test_action_mask_reports_reasons(self) -> None:
        scene = self._scene(calibration_valid=False)
        scene.targets[0].forbidden_zone_ids = ["keep_seedling_zone"]
        scene.targets[0].action_point_px = [500, 500]

        mask = ActionMaskBuilder().build(scene)

        reasons = mask.reasons_by_target()
        self.assertIn("calibration_required", reasons["target_safe"])
        self.assertIn("forbidden_zone_overlap", reasons["target_safe"])
        self.assertIn("target_outside_tray", reasons["target_safe"])
        self.assertIn("human_review_required", reasons["target_unknown"])

    def test_action_mask_blocks_processed_and_unsafe_targets(self) -> None:
        scene = self._scene()
        scene.targets[0].risk_score = 0.8
        scene.targets[0].uncertainty_radius_mm = 3.0
        scene.targets[0].min_distance_to_keep_mm = 1.0

        mask = ActionMaskBuilder().build(scene, processed_target_ids={"target_safe"})

        self.assertNotIn("target_safe", mask.valid_target_ids())
        reasons = mask.reasons_by_target()["target_safe"]
        self.assertIn("already_processed", reasons)
        self.assertIn("risk_score_too_high", reasons)
        self.assertIn("target_uncertainty_too_high", reasons)
        self.assertIn("too_close_to_keep_seedling", reasons)

    def test_safety_gate_allows_safe_simulation_command(self) -> None:
        scene = self._scene()
        command = RasterScanPolicy().next_action(scene)
        assert command is not None

        decision = SafetyGate().validate(command, scene)

        self.assertEqual(decision.result, ALLOW_SIMULATION)
        self.assertTrue(decision.allowed)

    def test_safety_gate_blocks_missing_target_before_execution(self) -> None:
        scene = self._scene()
        command = ActionCommand(
            command_id="cmd_missing_target",
            target_id="target_missing",
            action_type="mark",
            robot_point_mm=[5.0, 5.0, 0.0],
            tool_profile="pointer_only",
            requires_operator_confirmation=False,
        )

        decision = SafetyGate().validate(command, scene)

        self.assertEqual(decision.result, BLOCK_SYSTEM_ERROR)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.target_id, "target_missing")
        self.assertEqual(decision.reasons, ["target_not_found"])

    def test_safety_gate_blocks_bad_calibration_and_close_target(self) -> None:
        scene = self._scene(calibration_valid=False)
        command = RasterScanPolicy(
            action_mask_builder=ActionMaskBuilder(require_calibration_valid=False)
        ).next_action(scene)
        assert command is not None

        calibration_decision = SafetyGate().validate(command, scene)
        self.assertEqual(calibration_decision.result, BLOCK_CALIBRATION_REQUIRED)

        scene.safety.calibration_valid = True
        scene.targets[0].min_distance_to_keep_mm = 1.0
        unsafe_decision = SafetyGate(limits=SafetyLimits(require_robot_homed=False)).validate(command, scene)
        self.assertEqual(unsafe_decision.result, BLOCK_UNSAFE_TARGET)
        self.assertIn("too_close_to_keep_seedling", unsafe_decision.reasons)

        scene.targets[0].min_distance_to_keep_mm = 6.0
        scene.targets[0].forbidden_zone_ids = ["keep_seedling_zone"]
        forbidden_decision = SafetyGate(limits=SafetyLimits(require_robot_homed=False)).validate(command, scene)
        self.assertEqual(forbidden_decision.result, BLOCK_UNSAFE_TARGET)
        self.assertIn("forbidden_zone_overlap", forbidden_decision.reasons)

    def test_safety_limits_load_example_config(self) -> None:
        limits = load_safety_limits(ROOT / "configs/safety/safety_limits.example.yaml")

        self.assertEqual(limits.mode, "dry_run_pointer")
        self.assertFalse(limits.allow_real_action)
        self.assertTrue(limits.require_operator_confirmation)
        self.assertTrue(limits.log_all_blocked_actions)

    def test_safety_limits_reject_invalid_config_values(self) -> None:
        invalid_configs = [
            {"mode": "supervised_real_action"},
            {"allow_real_action": "false"},
            {"max_target_uncertainty_mm": -1.0},
            {"unknown_limit": True},
        ]

        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises(ConfigError):
                    safety_limits_from_mapping(config)

    def test_safety_gate_blocks_invalid_robot_telemetry_mode(self) -> None:
        scene = self._scene()
        command = RasterScanPolicy().next_action(scene)
        assert command is not None

        decision = SafetyGate(
            limits=SafetyLimits(require_operator_confirmation=False, require_robot_homed=False)
        ).validate(
            command,
            scene,
            telemetry=RobotTelemetry(
                position_mm=[0.0, 0.0, 0.0],
                homed=True,
                mode="supervised_real_action",
                interlock_ok=True,
            ),
        )

        self.assertEqual(decision.result, BLOCK_SYSTEM_ERROR)
        self.assertFalse(decision.allowed)
        self.assertIn("invalid_robot_mode", decision.reasons)

    def test_safety_gate_blocks_robot_telemetry_faults(self) -> None:
        scene = self._scene()
        command = RasterScanPolicy().next_action(scene)
        assert command is not None

        gate = SafetyGate(
            limits=SafetyLimits(
                mode="dry_run_pointer",
                require_operator_confirmation=False,
                require_robot_homed=False,
            )
        )
        base_telemetry = {
            "position_mm": [0.0, 0.0, 0.0],
            "homed": True,
            "mode": "dry_run_pointer",
            "interlock_ok": True,
        }

        def telemetry(**overrides: object) -> RobotTelemetry:
            return RobotTelemetry(**{**base_telemetry, **overrides})

        limit_decision = gate.validate(
            command,
            scene,
            telemetry(limit_switch_ok=False),
        )
        self.assertEqual(limit_decision.result, BLOCK_INTERLOCK)
        self.assertIn("limit_switch_not_ok", limit_decision.reasons)

        estop_decision = gate.validate(
            command,
            scene,
            telemetry(emergency_stop_active=True),
        )
        self.assertEqual(estop_decision.result, BLOCK_INTERLOCK)
        self.assertIn("emergency_stop_active", estop_decision.reasons)

        interlock_decision = gate.validate(
            command,
            scene,
            telemetry(interlock_ok=False),
        )
        self.assertEqual(interlock_decision.result, BLOCK_INTERLOCK)
        self.assertIn("interlock_false", interlock_decision.reasons)

        zone_error_decision = gate.validate(
            command,
            scene,
            telemetry(error_map_p95_mm=3.0),
        )
        self.assertEqual(zone_error_decision.result, BLOCK_CALIBRATION_REQUIRED)
        self.assertIn("zone_error_too_high", zone_error_decision.reasons)

    def test_safety_gate_prioritizes_real_action_block_before_review(self) -> None:
        scene = self._scene()
        scene.safety.enclosure_closed = True
        command = ActionCommand(
            command_id="cmd_real_review_001",
            target_id="target_safe",
            action_type="laser_fire",
            robot_point_mm=[5.0, 5.0, 0.0],
            tool_profile="laser",
            requires_operator_confirmation=True,
        )

        blocked = SafetyGate(
            limits=SafetyLimits(require_calibration_valid=False, require_robot_homed=False)
        ).validate(command, scene)

        self.assertEqual(blocked.result, BLOCK_UNSAFE_TARGET)
        self.assertIn("real_action_not_allowed", blocked.reasons)
        self.assertNotIn("operator_confirmation_required", blocked.reasons)

        review = SafetyGate(
            limits=SafetyLimits(
                allow_real_action=True,
                require_calibration_valid=False,
                require_robot_homed=False,
            )
        ).validate(command, scene)

        self.assertEqual(review.result, BLOCK_REVIEW_REQUIRED)
        self.assertEqual(review.reasons, ["operator_confirmation_required"])

    def test_safety_gate_blocks_real_action_when_software_safe_mode_disabled(self) -> None:
        scene = self._scene()
        scene.safety.software_safe_mode = False
        command = ActionCommand(
            command_id="cmd_real_001",
            target_id="target_safe",
            action_type="laser_fire",
            robot_point_mm=[5.0, 5.0, 0.0],
            tool_profile="laser",
            requires_operator_confirmation=False,
        )

        decision = SafetyGate(
            limits=SafetyLimits(
                allow_real_action=True,
                require_operator_confirmation=False,
                require_robot_homed=False,
            )
        ).validate(command, scene)

        self.assertEqual(decision.result, BLOCK_UNSAFE_TARGET)
        self.assertIn("software_safe_mode_disabled", decision.reasons)
        self.assertIn("enclosure_not_closed", decision.reasons)

    def test_safety_gate_blocks_real_action_when_enclosure_is_open(self) -> None:
        scene = self._scene()
        scene.safety.software_safe_mode = True
        scene.safety.enclosure_closed = False
        command = ActionCommand(
            command_id="cmd_real_002",
            target_id="target_safe",
            action_type="laser_fire",
            robot_point_mm=[5.0, 5.0, 0.0],
            tool_profile="laser",
            requires_operator_confirmation=False,
        )

        decision = SafetyGate(
            limits=SafetyLimits(
                allow_real_action=True,
                require_operator_confirmation=False,
                require_robot_homed=False,
            )
        ).validate(command, scene)

        self.assertEqual(decision.result, BLOCK_UNSAFE_TARGET)
        self.assertEqual(decision.reasons, ["enclosure_not_closed"])


if __name__ == "__main__":
    unittest.main()
