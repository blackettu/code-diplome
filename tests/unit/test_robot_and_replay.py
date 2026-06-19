from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from seedling_core.schemas import ActionCommand, ActionPlan, ActionTarget, CellState, RobotState, SafetyState, SceneState, TrayState
from seedling_decision.action_masks import ActionMaskBuilder
from seedling_decision.policies import RasterScanPolicy
from seedling_robot import (
    HardwareInLoopReview,
    MotionCommand,
    RealGantrySerialAdapter,
    SimulatorRobotAdapter,
    SpeedProfile,
    run_dry_run_control_points,
    run_dry_run_plan,
    run_hil_pointer_plan,
    validate_hil_review,
)
from seedling_robot.cli import main as robot_cli_main
from seedling_sim import ActuatorErrorModel, LogicalTraySimulator, ReplayLog, ReplayLogger, SimPlant, SimScene, SimTarget


ROOT = Path(__file__).resolve().parents[2]


class RobotAndReplayTests(unittest.TestCase):
    def _scene_state(self, calibration_valid: bool = True) -> SceneState:
        return SceneState(
            scene_id="scene_001",
            image_ref="sim_scene",
            dataset_version="sim_v0",
            ontology_version="ontology_v0_1",
            image_size_px=[100, 100],
            tray=TrayState(tray_id="tray001", grid_rows=1, grid_cols=1, bbox_xyxy_px=[0, 0, 100, 100]),
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
            safety=SafetyState(calibration_valid=calibration_valid, interlock_ok=True),
        )

    def _sim_scene(self) -> SimScene:
        return SimScene(
            scene_id="sim_001",
            grid_rows=1,
            grid_cols=1,
            cell_size_mm=[33.0, 33.0],
            plants=[SimPlant("plant_001", 0, 0, "crop_seedling", [10.0, 10.0])],
            targets=[SimTarget("target_001", "plant_001", "remove_extra_crop", [10.0, 10.0])],
        )

    def test_simulator_robot_executes_command_through_safety_gate(self) -> None:
        scene = self._scene_state()
        command = RasterScanPolicy().next_action(scene)
        assert command is not None
        logger = ReplayLogger(replay_id="replay_001", scene_id="sim_001")
        adapter = SimulatorRobotAdapter(
            simulator=LogicalTraySimulator(
                actuator_model=ActuatorErrorModel(xy_sigma_mm=0.0, drift_sigma_mm=0.0, seed=42)
            ),
            sim_scene=self._sim_scene(),
            replay_logger=logger,
        )
        adapter.connect()
        adapter.home()

        result = adapter.execute_command(command, scene)

        self.assertTrue(result.ok)
        self.assertTrue(result.safety_decision.allowed)
        self.assertEqual(result.command.safety_gate_result, "ALLOW_SIMULATION")
        self.assertTrue(adapter.sim_scene.plants[0].removed)
        self.assertEqual(len(logger.log.steps), 1)
        self.assertEqual(logger.log.steps[0].event_type, "robot_execution")
        self.assertEqual(logger.log.steps[0].payload["command"]["safety_gate_result"], "ALLOW_SIMULATION")

    def test_simulator_robot_logs_blocked_command(self) -> None:
        scene = self._scene_state(calibration_valid=False)
        command = RasterScanPolicy(
            action_mask_builder=ActionMaskBuilder(require_calibration_valid=False)
        ).next_action(scene)
        assert command is not None
        logger = ReplayLogger(replay_id="replay_001", scene_id="sim_001")
        adapter = SimulatorRobotAdapter(sim_scene=self._sim_scene(), replay_logger=logger)
        adapter.connect()
        adapter.home()

        result = adapter.execute_command(command, scene)

        self.assertFalse(result.ok)
        self.assertFalse(result.safety_decision.allowed)
        self.assertEqual(result.command.safety_gate_result, "BLOCK_CALIBRATION_REQUIRED")
        self.assertEqual(len(logger.log.steps), 1)
        self.assertFalse(logger.log.steps[0].payload["ok"])
        self.assertEqual(logger.log.steps[0].payload["command"]["safety_gate_result"], "BLOCK_CALIBRATION_REQUIRED")

    def test_replay_log_json_round_trip(self) -> None:
        logger = ReplayLogger(replay_id="replay_001", scene_id="sim_001")
        logger.append("custom", {"value": 1})

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.json"
            logger.to_json(path)
            restored = ReplayLog.from_json(path)

        self.assertEqual(restored.to_dict(), logger.log.to_dict())

    def test_motion_replay_cli_writes_simulation_report_and_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            commands_path = root / "commands.json"
            report_path = root / "motion_replay.json"
            move_to = {
                "kind": "move_to",
                "point_mm": [1.0, 2.0, 0.0],
                "speed": SpeedProfile(xy_mm_s=25.0, z_mm_s=5.0).to_dict(),
                "mode": "dry_run_pointer",
            }
            second = MotionCommand(command_id="cmd_002", point_mm=[3.0, 4.0, 0.0])
            commands_path.write_text(
                json.dumps({"commands": [{"kind": "home"}, move_to, {"command": second.to_dict()}]}),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_robot",
                    "motion-replay",
                    "--commands",
                    str(commands_path),
                    "--out",
                    str(report_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(completed.stdout)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            registry = json.loads((root / "artifact_registry.json").read_text(encoding="utf-8"))

            self.assertTrue(payload["ok"])
            self.assertEqual(report["mode"], "simulation")
            self.assertEqual(report["commands"], 2)
            self.assertEqual(report["executed"], 2)
            self.assertEqual(report["results"][0]["result"]["end_mm"], [1.0, 2.0, 0.0])
            self.assertTrue(payload["artifact_registry"])
            snapshot_path = root / "motion_replay.run_snapshot.json"
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_snapshot"], str(snapshot_path))
            self.assertEqual(snapshot["command"], "seedling-robot:motion-replay")
            self.assertEqual(snapshot["metadata"]["commands"], 2)
            self.assertEqual(registry["runs"][0]["command"], "seedling-robot:motion-replay")
            self.assertEqual([item["role"] for item in registry["runs"][0]["artifacts"]], ["input", "output", "output"])
            self.assertIn(str(snapshot_path), {item["path"] for item in registry["runs"][0]["artifacts"]})

    def test_dry_run_control_points_write_report_and_command_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            points = Path(tmp) / "points.json"
            report = Path(tmp) / "report.json"
            command_log = Path(tmp) / "commands.json"
            points.write_text(json.dumps({"points": [{"point_id": "p1", "expected_mm": [1, 2, 0]}]}), encoding="utf-8")

            payload = run_dry_run_control_points(points, report, command_log_path=command_log)

            self.assertTrue(payload["ok"])
            self.assertEqual(payload["points"][0]["error_mm"], 0.0)
            self.assertEqual(payload["error_summary_mm"]["p50"], 0.0)
            self.assertEqual(payload["error_summary_mm"]["p95"], 0.0)
            self.assertEqual(payload["error_summary_mm"]["p99"], 0.0)
            self.assertTrue(report.exists())
            self.assertTrue(command_log.exists())

    def test_dry_run_control_points_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            points = Path(tmp) / "points.json"
            report = Path(tmp) / "report.json"
            command_log = Path(tmp) / "commands.json"
            points.write_text(json.dumps({"points": [{"point_id": "p1", "expected_mm": [1, 2, 0]}]}), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_robot",
                    "dry-run-test",
                    "--points",
                    str(points),
                    "--out",
                    str(report),
                    "--command-log",
                    str(command_log),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertTrue(json.loads(completed.stdout)["ok"])
            self.assertTrue(report.exists())
            payload = json.loads(completed.stdout)
            self.assertTrue(payload["artifact_registry"])
            snapshot_path = Path(tmp) / "report.run_snapshot.json"
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_snapshot"], str(snapshot_path))
            self.assertEqual(snapshot["command"], "seedling-robot:dry-run-test")
            self.assertEqual(snapshot["metadata"]["tolerance_mm"], 1.0)
            registry = json.loads((Path(tmp) / "artifact_registry.json").read_text(encoding="utf-8"))
            self.assertEqual(registry["runs"][0]["command"], "seedling-robot:dry-run-test")
            self.assertEqual(
                [artifact["role"] for artifact in registry["runs"][0]["artifacts"]],
                ["input", "output", "output", "output"],
            )
            self.assertIn(str(snapshot_path), {artifact["path"] for artifact in registry["runs"][0]["artifacts"]})

    def test_dry_run_plan_runner_writes_report_command_log_and_replay(self) -> None:
        scene = self._scene_state()
        command = ActionCommand(
            command_id="cmd_001",
            target_id="target_001",
            action_type="move_and_mark",
            robot_point_mm=[10.0, 10.0, 0.0],
            tool_profile="pointer_only",
            requires_operator_confirmation=False,
        )
        plan = ActionPlan(plan_id="plan_001", policy_id="test_policy", commands=[command])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_path = root / "scene.json"
            plan_path = root / "plan.json"
            report_path = root / "dry_run_plan_report.json"
            command_log = root / "dry_run_plan_commands.json"
            replay_path = root / "dry_run_plan_replay.json"
            scene_path.write_text(json.dumps(scene.to_dict()), encoding="utf-8")
            plan_path.write_text(json.dumps(plan.to_dict()), encoding="utf-8")

            report = run_dry_run_plan(
                scene_path,
                plan_path,
                report_path,
                command_log_path=command_log,
                replay_path=replay_path,
                interlock_ok=True,
            )

            commands = json.loads(command_log.read_text(encoding="utf-8"))["commands"]
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
            report_exists = report_path.exists()

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["mode"], "dry_run_pointer")
        self.assertEqual(report["executed"], 1)
        self.assertFalse(report["aborted"])
        self.assertEqual(report["skipped"], 0)
        self.assertTrue(report_exists)
        motion_payload = next(item for item in commands if item["kind"] == "motion_command")
        self.assertEqual(motion_payload["command"]["mode"], "dry_run_pointer")
        self.assertEqual(motion_payload["command"]["safety_token"], "ALLOW_DRY_RUN")
        self.assertEqual(replay["steps"][0]["event_type"], "robot_execution")

    def test_dry_run_plan_runner_stops_after_failed_command(self) -> None:
        scene = self._scene_state()
        blocked_command = ActionCommand(
            command_id="cmd_001",
            target_id="target_001",
            action_type="move_and_mark",
            robot_point_mm=[10.0, 10.0, 0.0],
            tool_profile="laser",
            requires_operator_confirmation=False,
        )
        skipped_command = ActionCommand(
            command_id="cmd_002",
            target_id="target_001",
            action_type="move_and_mark",
            robot_point_mm=[20.0, 20.0, 0.0],
            tool_profile="pointer_only",
            requires_operator_confirmation=False,
        )
        plan = ActionPlan(plan_id="plan_001", policy_id="test_policy", commands=[blocked_command, skipped_command])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_path = root / "scene.json"
            plan_path = root / "plan.json"
            report_path = root / "dry_run_plan_report.json"
            command_log = root / "dry_run_plan_commands.json"
            replay_path = root / "dry_run_plan_replay.json"
            scene_path.write_text(json.dumps(scene.to_dict()), encoding="utf-8")
            plan_path.write_text(json.dumps(plan.to_dict()), encoding="utf-8")

            report = run_dry_run_plan(
                scene_path,
                plan_path,
                report_path,
                command_log_path=command_log,
                replay_path=replay_path,
                interlock_ok=True,
            )
            commands = json.loads(command_log.read_text(encoding="utf-8"))["commands"]
            replay = json.loads(replay_path.read_text(encoding="utf-8"))

        self.assertFalse(report["ok"])
        self.assertTrue(report["aborted"])
        self.assertEqual(report["abort_reason"], "command_failed")
        self.assertEqual(report["commands"], 2)
        self.assertEqual(report["executed"], 0)
        self.assertEqual(report["blocked"], 2)
        self.assertEqual(report["skipped"], 1)
        self.assertEqual(len(report["results"]), 1)
        self.assertEqual(report["results"][0]["command"]["command_id"], "cmd_001")
        self.assertEqual(commands[-2]["kind"], "blocked_tool_profile")
        self.assertEqual(commands[-1]["kind"], "emergency_stop")
        self.assertEqual(len(replay["steps"]), 1)

    def test_dry_run_plan_cli_writes_artifact_registry(self) -> None:
        scene = self._scene_state()
        command = ActionCommand(
            command_id="cmd_001",
            target_id="target_001",
            action_type="move_and_mark",
            robot_point_mm=[10.0, 10.0, 0.0],
            tool_profile="pointer_only",
            requires_operator_confirmation=False,
        )
        plan = ActionPlan(plan_id="plan_001", policy_id="test_policy", commands=[command])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_path = root / "scene.json"
            plan_path = root / "plan.json"
            report_path = root / "dry_run_plan_report.json"
            command_log = root / "dry_run_plan_commands.json"
            replay_path = root / "dry_run_plan_replay.json"
            scene_path.write_text(json.dumps(scene.to_dict()), encoding="utf-8")
            plan_path.write_text(json.dumps(plan.to_dict()), encoding="utf-8")
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                robot_cli_main(
                    [
                        "dry-run-plan",
                        "--scene",
                        str(scene_path),
                        "--plan",
                        str(plan_path),
                        "--out",
                        str(report_path),
                        "--command-log",
                        str(command_log),
                        "--replay",
                        str(replay_path),
                        "--interlock-ok",
                    ]
                )

            payload = json.loads(stdout.getvalue())
            registry = json.loads((root / "artifact_registry.json").read_text(encoding="utf-8"))
            snapshot_path = root / "dry_run_plan_report.run_snapshot.json"
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

        self.assertTrue(payload["ok"])
        self.assertTrue(payload["artifact_registry"])
        self.assertEqual(payload["run_snapshot"], str(snapshot_path))
        self.assertEqual(snapshot["command"], "seedling-robot:dry-run-plan")
        self.assertTrue(snapshot["metadata"]["interlock_ok"])
        self.assertEqual(payload["mode"], "dry_run_pointer")
        self.assertEqual(registry["runs"][0]["command"], "seedling-robot:dry-run-plan")
        self.assertEqual(
            [artifact["role"] for artifact in registry["runs"][0]["artifacts"]],
            ["input", "input", "output", "output", "output", "output"],
        )
        self.assertIn(str(snapshot_path), {artifact["path"] for artifact in registry["runs"][0]["artifacts"]})

    def test_hil_review_validation_blocks_real_actuation(self) -> None:
        review = _hil_review(allow_real_actuation=True)

        result = validate_hil_review(review)

        self.assertFalse(result["ok"])
        self.assertIn("real_actuation_not_allowed_in_current_project", result["errors"])

    def test_hil_review_rejects_unknown_schema_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            review_path = Path(tmp) / "hil_review.json"
            payload = _hil_review().to_dict()
            payload["schema_version"] = "hardware_in_loop_review_v9"
            review_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unsupported HIL review schema_version"):
                HardwareInLoopReview.from_json(review_path)

    def test_hil_review_validation_requires_requested_mode_and_tool_profile(self) -> None:
        review = _hil_review(allowed_modes=["dry_run_pointer"], allowed_tool_profiles=[])

        result = validate_hil_review(
            review,
            required_mode="hardware_in_loop_pointer",
            required_tool_profile="pointer_only",
        )

        self.assertFalse(result["ok"])
        self.assertIn("mode_not_authorized:hardware_in_loop_pointer", result["errors"])
        self.assertIn("tool_profile_not_authorized:pointer_only", result["errors"])

    def test_hil_review_validation_blocks_positioning_error_above_limit(self) -> None:
        review = _hil_review(positioning_error_p95_mm=3.1)

        result = validate_hil_review(review)

        self.assertFalse(result["ok"])
        self.assertEqual(result["positioning_error_p95_mm"], 3.1)
        self.assertEqual(result["max_error_p95_mm"], 2.0)
        self.assertIn("positioning_error_p95_mm_exceeds_limit", result["errors"])

    def test_hil_review_validation_warns_when_positioning_error_missing(self) -> None:
        review = _hil_review(positioning_error_p95_mm=None)

        result = validate_hil_review(review)

        self.assertTrue(result["ok"])
        self.assertIn("positioning_error_p95_mm_missing", result["warnings"])

    def test_hil_review_validation_requires_positioning_error_when_requested(self) -> None:
        review = _hil_review(positioning_error_p95_mm=None)

        result = validate_hil_review(review, require_positioning_error=True)

        self.assertFalse(result["ok"])
        self.assertIn("positioning_error_p95_mm_required", result["errors"])

    def test_real_gantry_adapter_refuses_without_hardware_flag(self) -> None:
        adapter = RealGantrySerialAdapter(port="COM_TEST", review=_hil_review())

        with self.assertRaisesRegex(RuntimeError, "allow_hardware=True"):
            adapter.connect()

    def test_real_gantry_adapter_refuses_review_without_hil_mode(self) -> None:
        adapter = RealGantrySerialAdapter(
            port="COM_TEST",
            review=_hil_review(allowed_modes=["dry_run_pointer"]),
            allow_hardware=True,
        )

        with self.assertRaisesRegex(RuntimeError, "mode_not_authorized"):
            adapter.connect()

    def test_real_gantry_adapter_requires_positioning_error_evidence(self) -> None:
        adapter = RealGantrySerialAdapter(
            port="COM_TEST",
            review=_hil_review(positioning_error_p95_mm=None),
            allow_hardware=True,
        )

        with self.assertRaisesRegex(RuntimeError, "positioning_error_p95_mm_required"):
            adapter.connect()

    def test_real_gantry_hil_pointer_executes_action_command_through_safety_gate(self) -> None:
        scene = self._scene_state()
        command = ActionCommand(
            command_id="cmd_001",
            target_id="target_001",
            action_type="mark",
            robot_point_mm=[10.0, 10.0, 0.0],
            tool_profile="pointer_only",
            requires_operator_confirmation=False,
        )
        logger = ReplayLogger(replay_id="hil_replay", scene_id=scene.scene_id)
        with tempfile.TemporaryDirectory() as tmp:
            command_log = Path(tmp) / "hil_commands.json"
            fake_serial, restore = _install_fake_serial()
            try:
                adapter = RealGantrySerialAdapter(
                    port="COM_TEST",
                    review=_hil_review(),
                    allow_hardware=True,
                    command_log_path=command_log,
                    replay_logger=logger,
                    interlock_ok=True,
                )
                adapter.connect()
                adapter.home()

                result = adapter.execute_command(command, scene)
            finally:
                restore()

            self.assertTrue(result.ok, result.to_dict())
            self.assertEqual(result.safety_decision.result, "ALLOW_HARDWARE_IN_LOOP_POINTER")
            self.assertEqual(result.command.safety_gate_result, "ALLOW_HARDWARE_IN_LOOP_POINTER")
            self.assertEqual(fake_serial.instances[0].written, ["HOME\n", "MOVE X10.000 Y10.000 Z0.000 F50.000\n", "POINT\n"])
            self.assertEqual(len(logger.log.steps), 1)
            self.assertTrue(command_log.exists())
            log_payload = json.loads(command_log.read_text(encoding="utf-8"))["commands"]
            self.assertEqual(log_payload[-1]["kind"], "mark_or_act")
            motion_payload = next(item for item in log_payload if item["kind"] == "motion_command")
            self.assertEqual(
                motion_payload["command"]["metadata"]["safety_gate_result"],
                "ALLOW_HARDWARE_IN_LOOP_POINTER",
            )

    def test_real_gantry_hil_pointer_logs_blocked_safety_decision(self) -> None:
        scene = self._scene_state()
        command = ActionCommand(
            command_id="cmd_001",
            target_id="target_001",
            action_type="mark",
            robot_point_mm=[10.0, 10.0, 0.0],
            tool_profile="pointer_only",
            requires_operator_confirmation=False,
        )
        logger = ReplayLogger(replay_id="hil_replay", scene_id=scene.scene_id)
        with tempfile.TemporaryDirectory() as tmp:
            command_log = Path(tmp) / "hil_commands.json"
            fake_serial, restore = _install_fake_serial()
            try:
                adapter = RealGantrySerialAdapter(
                    port="COM_TEST",
                    review=_hil_review(),
                    allow_hardware=True,
                    command_log_path=command_log,
                    replay_logger=logger,
                    interlock_ok=False,
                )
                adapter.connect()
                adapter.home()

                result = adapter.execute_command(command, scene)
            finally:
                restore()

            self.assertFalse(result.ok)
            self.assertEqual(result.safety_decision.result, "BLOCK_INTERLOCK")
            self.assertEqual(result.command.safety_gate_result, "BLOCK_INTERLOCK")
            self.assertIn("interlock_false", result.safety_decision.reasons)
            self.assertEqual(fake_serial.instances[0].written, ["HOME\n"])
            self.assertFalse(logger.log.steps[0].payload["ok"])
            self.assertEqual(json.loads(command_log.read_text(encoding="utf-8"))["commands"][-1]["kind"], "blocked_motion_command")

    def test_real_gantry_hil_pointer_blocks_unsupported_tool_profile_before_motion(self) -> None:
        scene = self._scene_state()
        command = ActionCommand(
            command_id="cmd_001",
            target_id="target_001",
            action_type="mark",
            robot_point_mm=[10.0, 10.0, 0.0],
            tool_profile="laser",
            requires_operator_confirmation=False,
        )
        logger = ReplayLogger(replay_id="hil_replay", scene_id=scene.scene_id)
        with tempfile.TemporaryDirectory() as tmp:
            command_log = Path(tmp) / "hil_commands.json"
            fake_serial, restore = _install_fake_serial()
            try:
                adapter = RealGantrySerialAdapter(
                    port="COM_TEST",
                    review=_hil_review(),
                    allow_hardware=True,
                    command_log_path=command_log,
                    replay_logger=logger,
                    interlock_ok=True,
                )
                adapter.connect()
                adapter.home()

                result = adapter.execute_command(command, scene)
            finally:
                restore()

            commands = json.loads(command_log.read_text(encoding="utf-8"))["commands"]

        self.assertFalse(result.ok)
        self.assertEqual(result.message, "HIL adapter only allows pointer_only")
        self.assertEqual(fake_serial.instances[0].written, ["HOME\n"])
        self.assertEqual(commands[-1]["kind"], "blocked_tool_profile")
        self.assertNotIn("motion_command", [item["kind"] for item in commands])
        self.assertFalse(logger.log.steps[0].payload["ok"])

    def test_hil_pointer_runner_writes_report_command_log_and_replay(self) -> None:
        scene = self._scene_state()
        command = ActionCommand(
            command_id="cmd_001",
            target_id="target_001",
            action_type="move_and_mark",
            robot_point_mm=[10.0, 10.0, 0.0],
            tool_profile="pointer_only",
            requires_operator_confirmation=False,
        )
        plan = ActionPlan(plan_id="plan_001", policy_id="test_policy", commands=[command])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_path = root / "scene.json"
            plan_path = root / "plan.json"
            review_path = root / "review.json"
            report_path = root / "hil_report.json"
            command_log = root / "hil_commands.json"
            replay_path = root / "hil_replay.json"
            scene_path.write_text(json.dumps(scene.to_dict()), encoding="utf-8")
            plan_path.write_text(json.dumps(plan.to_dict()), encoding="utf-8")
            _hil_review().to_json(review_path)
            fake_serial, restore = _install_fake_serial()
            try:
                report = run_hil_pointer_plan(
                    scene_path,
                    plan_path,
                    review_path,
                    "COM_TEST",
                    report_path,
                    command_log_path=command_log,
                    replay_path=replay_path,
                    allow_hardware=True,
                    interlock_ok=True,
                )
            finally:
                restore()

            self.assertTrue(report["ok"], report)
            self.assertEqual(report["mode"], "hardware_in_loop_pointer")
            self.assertEqual(report["executed"], 1)
            self.assertFalse(report["aborted"])
            self.assertEqual(report["skipped"], 0)
            self.assertEqual(fake_serial.instances[0].written, ["HOME\n", "MOVE X10.000 Y10.000 Z0.000 F50.000\n", "POINT\n"])
            self.assertTrue(report_path.exists())
            self.assertTrue(command_log.exists())
            self.assertTrue(replay_path.exists())
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
            self.assertEqual(replay["steps"][0]["event_type"], "robot_execution")

    def test_hil_pointer_runner_stops_after_failed_command(self) -> None:
        scene = self._scene_state()
        blocked_command = ActionCommand(
            command_id="cmd_001",
            target_id="target_001",
            action_type="move_and_mark",
            robot_point_mm=[10.0, 10.0, 0.0],
            tool_profile="laser",
            requires_operator_confirmation=False,
        )
        skipped_command = ActionCommand(
            command_id="cmd_002",
            target_id="target_001",
            action_type="move_and_mark",
            robot_point_mm=[20.0, 20.0, 0.0],
            tool_profile="pointer_only",
            requires_operator_confirmation=False,
        )
        plan = ActionPlan(plan_id="plan_001", policy_id="test_policy", commands=[blocked_command, skipped_command])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_path = root / "scene.json"
            plan_path = root / "plan.json"
            review_path = root / "review.json"
            report_path = root / "hil_report.json"
            command_log = root / "hil_commands.json"
            replay_path = root / "hil_replay.json"
            scene_path.write_text(json.dumps(scene.to_dict()), encoding="utf-8")
            plan_path.write_text(json.dumps(plan.to_dict()), encoding="utf-8")
            _hil_review().to_json(review_path)
            fake_serial, restore = _install_fake_serial()
            try:
                report = run_hil_pointer_plan(
                    scene_path,
                    plan_path,
                    review_path,
                    "COM_TEST",
                    report_path,
                    command_log_path=command_log,
                    replay_path=replay_path,
                    allow_hardware=True,
                    interlock_ok=True,
                )
            finally:
                restore()

            commands = json.loads(command_log.read_text(encoding="utf-8"))["commands"]
            replay = json.loads(replay_path.read_text(encoding="utf-8"))

        self.assertFalse(report["ok"])
        self.assertTrue(report["aborted"])
        self.assertEqual(report["abort_reason"], "command_failed")
        self.assertEqual(report["commands"], 2)
        self.assertEqual(report["executed"], 0)
        self.assertEqual(report["blocked"], 2)
        self.assertEqual(report["skipped"], 1)
        self.assertEqual(len(report["results"]), 1)
        self.assertEqual(report["results"][0]["command"]["command_id"], "cmd_001")
        self.assertEqual(fake_serial.instances[0].written, ["HOME\n", "ESTOP\n"])
        self.assertEqual(commands[-2]["kind"], "blocked_tool_profile")
        self.assertEqual(commands[-1]["kind"], "emergency_stop")
        self.assertEqual(len(replay["steps"]), 1)

    def test_hil_pointer_runner_writes_invalid_review_report_without_connecting(self) -> None:
        scene = self._scene_state()
        command = ActionCommand(
            command_id="cmd_001",
            target_id="target_001",
            action_type="move_and_mark",
            robot_point_mm=[10.0, 10.0, 0.0],
            tool_profile="pointer_only",
            requires_operator_confirmation=False,
        )
        plan = ActionPlan(plan_id="plan_001", policy_id="test_policy", commands=[command])

        def fail_adapter(**_: object) -> object:
            raise AssertionError("adapter must not be constructed for invalid HIL review")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_path = root / "scene.json"
            plan_path = root / "plan.json"
            review_path = root / "review.json"
            report_path = root / "hil_report.json"
            command_log = root / "hil_commands.json"
            replay_path = root / "hil_replay.json"
            scene_path.write_text(json.dumps(scene.to_dict()), encoding="utf-8")
            plan_path.write_text(json.dumps(plan.to_dict()), encoding="utf-8")
            _hil_review(allowed_tool_profiles=[]).to_json(review_path)

            report = run_hil_pointer_plan(
                scene_path,
                plan_path,
                review_path,
                "COM_TEST",
                report_path,
                command_log_path=command_log,
                replay_path=replay_path,
                allow_hardware=True,
                adapter_factory=fail_adapter,
            )

            saved = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertFalse(report["ok"])
            self.assertTrue(report["aborted"])
            self.assertEqual(report["abort_reason"], "invalid_hil_review")
            self.assertEqual(report["blocked"], 1)
            self.assertEqual(report["results"], [])
            self.assertIn("tool_profile_not_authorized:pointer_only", report["review"]["errors"])
            self.assertEqual(saved["review"]["errors"], report["review"]["errors"])
            self.assertFalse(command_log.exists())
            self.assertFalse(replay_path.exists())

    def test_hil_pointer_runner_aborts_when_review_positioning_error_exceeds_limit(self) -> None:
        scene = self._scene_state()
        command = ActionCommand(
            command_id="cmd_001",
            target_id="target_001",
            action_type="move_and_mark",
            robot_point_mm=[10.0, 10.0, 0.0],
            tool_profile="pointer_only",
            requires_operator_confirmation=False,
        )
        plan = ActionPlan(plan_id="plan_001", policy_id="test_policy", commands=[command])

        def fail_adapter(**_: object) -> object:
            raise AssertionError("adapter must not be constructed for invalid HIL review")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_path = root / "scene.json"
            plan_path = root / "plan.json"
            review_path = root / "review.json"
            report_path = root / "hil_report.json"
            scene_path.write_text(json.dumps(scene.to_dict()), encoding="utf-8")
            plan_path.write_text(json.dumps(plan.to_dict()), encoding="utf-8")
            _hil_review(positioning_error_p95_mm=3.1).to_json(review_path)

            report = run_hil_pointer_plan(
                scene_path,
                plan_path,
                review_path,
                "COM_TEST",
                report_path,
                allow_hardware=True,
                adapter_factory=fail_adapter,
            )

            self.assertFalse(report["ok"])
            self.assertTrue(report["aborted"])
            self.assertEqual(report["abort_reason"], "invalid_hil_review")
            self.assertIn("positioning_error_p95_mm_exceeds_limit", report["review"]["errors"])

    def test_hil_pointer_runner_requires_positioning_error_evidence(self) -> None:
        scene = self._scene_state()
        command = ActionCommand(
            command_id="cmd_001",
            target_id="target_001",
            action_type="move_and_mark",
            robot_point_mm=[10.0, 10.0, 0.0],
            tool_profile="pointer_only",
            requires_operator_confirmation=False,
        )
        plan = ActionPlan(plan_id="plan_001", policy_id="test_policy", commands=[command])

        def fail_adapter(**_: object) -> object:
            raise AssertionError("adapter must not be constructed for invalid HIL review")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_path = root / "scene.json"
            plan_path = root / "plan.json"
            review_path = root / "review.json"
            report_path = root / "hil_report.json"
            scene_path.write_text(json.dumps(scene.to_dict()), encoding="utf-8")
            plan_path.write_text(json.dumps(plan.to_dict()), encoding="utf-8")
            _hil_review(positioning_error_p95_mm=None).to_json(review_path)

            report = run_hil_pointer_plan(
                scene_path,
                plan_path,
                review_path,
                "COM_TEST",
                report_path,
                allow_hardware=True,
                adapter_factory=fail_adapter,
            )

            self.assertFalse(report["ok"])
            self.assertEqual(report["abort_reason"], "invalid_hil_review")
            self.assertIn("positioning_error_p95_mm_required", report["review"]["errors"])

    def test_hil_pointer_cli_writes_artifact_registry(self) -> None:
        scene = self._scene_state()
        command = ActionCommand(
            command_id="cmd_001",
            target_id="target_001",
            action_type="move_and_mark",
            robot_point_mm=[10.0, 10.0, 0.0],
            tool_profile="pointer_only",
            requires_operator_confirmation=False,
        )
        plan = ActionPlan(plan_id="plan_001", policy_id="test_policy", commands=[command])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_path = root / "scene.json"
            plan_path = root / "plan.json"
            review_path = root / "review.json"
            report_path = root / "hil_report.json"
            command_log = root / "hil_commands.json"
            replay_path = root / "hil_replay.json"
            scene_path.write_text(json.dumps(scene.to_dict()), encoding="utf-8")
            plan_path.write_text(json.dumps(plan.to_dict()), encoding="utf-8")
            _hil_review().to_json(review_path)
            fake_serial, restore = _install_fake_serial()
            stdout = io.StringIO()
            try:
                with redirect_stdout(stdout):
                    robot_cli_main(
                        [
                            "hil-pointer-run",
                            "--scene",
                            str(scene_path),
                            "--plan",
                            str(plan_path),
                            "--review",
                            str(review_path),
                            "--port",
                            "COM_TEST",
                            "--out",
                            str(report_path),
                            "--command-log",
                            str(command_log),
                            "--replay",
                            str(replay_path),
                            "--allow-hardware",
                            "--interlock-ok",
                        ]
                    )
            finally:
                restore()

            payload = json.loads(stdout.getvalue())
            registry = json.loads((root / "artifact_registry.json").read_text(encoding="utf-8"))

            self.assertTrue(payload["ok"])
            self.assertTrue(payload["artifact_registry"])
            snapshot_path = root / "hil_report.run_snapshot.json"
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_snapshot"], str(snapshot_path))
            self.assertEqual(snapshot["command"], "seedling-robot:hil-pointer-run")
            self.assertTrue(snapshot["metadata"]["allow_hardware"])
            self.assertEqual(fake_serial.instances[0].written, ["HOME\n", "MOVE X10.000 Y10.000 Z0.000 F50.000\n", "POINT\n"])
            self.assertEqual(registry["runs"][0]["command"], "seedling-robot:hil-pointer-run")
            self.assertEqual(
                [artifact["role"] for artifact in registry["runs"][0]["artifacts"]],
                ["input", "input", "input", "output", "output", "output", "output"],
            )
            self.assertIn(str(snapshot_path), {artifact["path"] for artifact in registry["runs"][0]["artifacts"]})

    def test_validate_hil_review_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            review_path = Path(tmp) / "hil_review.json"
            out = Path(tmp) / "hil_validation.json"
            _hil_review().to_json(review_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_robot",
                    "validate-hil-review",
                    "--review",
                    str(review_path),
                    "--out",
                    str(out),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(completed.stdout)
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["emergency_stop_tested"])
            self.assertTrue(payload["artifact_registry"])
            snapshot_path = Path(tmp) / "hil_validation.run_snapshot.json"
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_snapshot"], str(snapshot_path))
            self.assertEqual(snapshot["command"], "seedling-robot:validate-hil-review")
            self.assertTrue(out.exists())
            registry = json.loads((Path(tmp) / "artifact_registry.json").read_text(encoding="utf-8"))
            self.assertEqual(registry["runs"][0]["command"], "seedling-robot:validate-hil-review")
            self.assertIn(str(snapshot_path), {artifact["path"] for artifact in registry["runs"][0]["artifacts"]})

    def test_validate_hil_review_cli_hil_pointer_requires_positioning_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            review_path = Path(tmp) / "hil_review.json"
            _hil_review(positioning_error_p95_mm=None).to_json(review_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_robot",
                    "validate-hil-review",
                    "--review",
                    str(review_path),
                    "--hil-pointer",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(completed.stdout)
            self.assertFalse(payload["ok"])
            self.assertIn("positioning_error_p95_mm_required", payload["errors"])

def _hil_review(
    allow_real_actuation: bool = False,
    allowed_modes: list[str] | None = None,
    allowed_tool_profiles: list[str] | None = None,
    positioning_error_p95_mm: float | None = 1.0,
) -> HardwareInLoopReview:
    return HardwareInLoopReview(
        review_id="hil_review_001",
        reviewer="safety_reviewer",
        approved_at="2026-06-17T00:00:00+00:00",
        expires_at="2099-01-01T00:00:00+00:00",
        gantry_id="gantry_dry_lab",
        allowed_modes=allowed_modes if allowed_modes is not None else ["hardware_in_loop_pointer"],
        allowed_tool_profiles=allowed_tool_profiles if allowed_tool_profiles is not None else ["pointer_only"],
        calibration_id="calib_001",
        positioning_error_p95_mm=positioning_error_p95_mm,
        emergency_stop_tested=True,
        allow_real_actuation=allow_real_actuation,
    )


class _FakeSerialPort:
    def __init__(self, port: str, baudrate: int, timeout: float) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.written: list[str] = []

    def write(self, payload: bytes) -> None:
        self.written.append(payload.decode("ascii"))


class _FakeSerialModule:
    def __init__(self) -> None:
        self.instances: list[_FakeSerialPort] = []

    def Serial(self, port: str, baudrate: int, timeout: float) -> _FakeSerialPort:
        serial = _FakeSerialPort(port, baudrate, timeout)
        self.instances.append(serial)
        return serial


def _install_fake_serial() -> tuple[_FakeSerialModule, object]:
    fake = _FakeSerialModule()
    previous = sys.modules.get("serial")
    sys.modules["serial"] = fake  # type: ignore[assignment]

    def restore() -> None:
        if previous is None:
            sys.modules.pop("serial", None)
        else:
            sys.modules["serial"] = previous

    return fake, restore


if __name__ == "__main__":
    unittest.main()
