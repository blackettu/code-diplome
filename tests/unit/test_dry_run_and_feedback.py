from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from seedling_core.schemas import ActionCommand, ActionTarget, CellState, RobotState, SafetyState, SceneState, TrayState
from seedling_decision.safety_gate import SafetyGate, SafetyLimits
from seedling_decision.policies import RasterScanPolicy
from seedling_robot import DryRunSerialAdapter, MotionCommand, MotionResult, SpeedProfile, replay_motion_commands
from seedling_ui.feedback import AnnotationFeedback, append_feedback, read_feedback


class DryRunAndFeedbackTests(unittest.TestCase):
    def _scene(self) -> SceneState:
        return SceneState(
            scene_id="scene_001",
            image_ref="image.jpg",
            dataset_version="zks",
            ontology_version="ontology",
            image_size_px=[100, 100],
            tray=TrayState("tray", 1, 1, bbox_xyxy_px=[0, 0, 100, 100]),
            robot=RobotState(position_mm=[0, 0, 0], homed=True, mode="dry_run_pointer"),
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
                    object_id="obj_001",
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

    def test_dry_run_adapter_serializes_command_after_safety_gate(self) -> None:
        scene = self._scene()
        command = RasterScanPolicy().next_action(scene)
        assert command is not None
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "commands.json"
            adapter = DryRunSerialAdapter(command_log_path=log_path)
            adapter.connect()
            adapter.home()

            result = adapter.execute_command(
                command,
                scene,
                SafetyGate(limits=SafetyLimits(mode="dry_run_pointer", require_operator_confirmation=False)),
            )

            self.assertTrue(result.ok)
            self.assertTrue(result.safety_decision.allowed)
            self.assertEqual(result.command.safety_gate_result, "ALLOW_DRY_RUN")
            self.assertTrue(log_path.exists())
            self.assertGreaterEqual(len(adapter.serialized_commands()), 3)
            payload = json.loads(log_path.read_text(encoding="utf-8"))
            motion = next(item for item in payload["commands"] if item["kind"] == "motion_command")
            self.assertEqual(motion["command"]["safety_token"], "ALLOW_DRY_RUN")
            self.assertEqual(motion["command"]["metadata"]["safety_gate_result"], "ALLOW_DRY_RUN")

    def test_dry_run_adapter_blocks_when_interlock_false(self) -> None:
        scene = self._scene()
        command = RasterScanPolicy().next_action(scene)
        assert command is not None
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "commands.json"
            adapter = DryRunSerialAdapter(command_log_path=log_path, interlock_ok=False)
            adapter.connect()
            adapter.home()

            result = adapter.execute_command(command, scene)
            payload = json.loads(log_path.read_text(encoding="utf-8"))

        self.assertFalse(result.ok)
        self.assertIn("interlock_false", result.safety_decision.reasons)
        self.assertEqual(result.command.safety_gate_result, "BLOCK_INTERLOCK")
        self.assertEqual(payload["commands"][-1]["kind"], "blocked_motion_command")
        self.assertEqual(payload["commands"][-1]["result"]["safety_decision"]["result"], "BLOCK_INTERLOCK")

    def test_dry_run_adapter_requires_homing_and_interlocks_for_direct_moves(self) -> None:
        adapter = DryRunSerialAdapter()
        adapter.connect()

        with self.assertRaisesRegex(RuntimeError, "must be homed"):
            adapter.move_to([1, 2, 0], SpeedProfile())

        adapter.home()
        adapter.interlock_ok = False
        with self.assertRaisesRegex(RuntimeError, "interlock"):
            adapter.move_to([1, 2, 0], SpeedProfile())

        adapter.interlock_ok = True
        adapter.limit_switch_ok = False
        with self.assertRaisesRegex(RuntimeError, "limit switch"):
            adapter.move_to([1, 2, 0], SpeedProfile())

    def test_dry_run_adapter_telemetry_reports_homing_limit_and_estop(self) -> None:
        adapter = DryRunSerialAdapter(limit_switch_ok=False)
        adapter.connect()

        before_home = adapter.telemetry()
        self.assertFalse(before_home.homed)
        self.assertFalse(before_home.limit_switch_ok)
        self.assertFalse(before_home.emergency_stop_active)

        adapter.limit_switch_ok = True
        adapter.home()
        adapter.emergency_stop()
        telemetry = adapter.telemetry()

        self.assertTrue(telemetry.homed)
        self.assertTrue(telemetry.limit_switch_ok)
        self.assertTrue(telemetry.emergency_stop_active)
        self.assertEqual(telemetry.mode, "dry_run_pointer")

    def test_dry_run_adapter_blocks_unsupported_tool_profile_before_motion(self) -> None:
        scene = self._scene()
        command = ActionCommand(
            command_id="cmd_laser_profile",
            target_id="target_001",
            action_type="mark",
            robot_point_mm=[10.0, 10.0, 0.0],
            tool_profile="laser",
            requires_operator_confirmation=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "commands.json"
            adapter = DryRunSerialAdapter(command_log_path=log_path)
            adapter.connect()
            adapter.home()

            result = adapter.execute_command(
                command,
                scene,
                SafetyGate(limits=SafetyLimits(mode="dry_run_pointer", require_operator_confirmation=False)),
            )
            commands = json.loads(log_path.read_text(encoding="utf-8"))["commands"]

        self.assertFalse(result.ok)
        self.assertEqual(result.message, "dry-run adapter only allows pointer_only")
        self.assertEqual(commands[-1]["kind"], "blocked_tool_profile")
        self.assertNotIn("move_to", [item["kind"] for item in commands])

    def test_motion_replay_uses_robot_adapter(self) -> None:
        adapter = DryRunSerialAdapter()
        commands = [
            MotionCommand("cmd_001", [1, 2, 0], SpeedProfile()),
            MotionCommand("cmd_002", [3, 4, 0], SpeedProfile()),
        ]

        results = replay_motion_commands(adapter, commands)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[-1].end_mm, [3.0, 4.0, 0.0])

    def test_motion_replay_stops_after_failed_move_by_default(self) -> None:
        class FailingDryRunAdapter(DryRunSerialAdapter):
            def move_to(self, point_mm: list[float], speed: SpeedProfile | None = None) -> MotionResult:
                result = super().move_to(point_mm, speed)
                return MotionResult(False, result.start_mm, result.end_mm, result.speed, "simulated failure")

        adapter = FailingDryRunAdapter()
        commands = [
            MotionCommand("cmd_001", [1, 2, 0], SpeedProfile()),
            MotionCommand("cmd_002", [3, 4, 0], SpeedProfile()),
        ]

        results = replay_motion_commands(adapter, commands)

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].ok)

    def test_feedback_jsonl_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "feedback.jsonl"
            append_feedback(
                path,
                AnnotationFeedback(
                    image_id="tray001.jpg",
                    target_id="target_001",
                    object_id="obj_001",
                    cell_id="r00_c00",
                    error_type="wrong_target",
                    comment="operator marked wrong target",
                    operator_id="operator_a",
                ),
            )

            rows = read_feedback(path)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].error_type, "wrong_target")


if __name__ == "__main__":
    unittest.main()
