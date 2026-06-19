from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from seedling_core.schemas import ActionCommand, SceneState
from seedling_decision.safety_gate import RobotTelemetry, SafetyGate
from seedling_robot.adapters.base import RobotAdapter
from seedling_robot.protocol import (
    MotionCommand,
    MotionResult,
    RobotExecutionResult,
    SpeedProfile,
    ToolProfile,
    ToolResult,
    attach_safety_decision,
)
from seedling_sim.replay import ReplayLogger


@dataclass
class DryRunSerialAdapter(RobotAdapter):
    """Safe serial-protocol stub.

    The adapter does not open a serial port and does not move hardware. It
    serializes the command that would be sent to a dry-run controller and keeps
    telemetry/replay records for operator review.
    """

    port: str = "DRY_RUN"
    command_log_path: str | Path | None = None
    replay_logger: ReplayLogger | None = None
    connected: bool = False
    stopped: bool = False
    interlock_ok: bool = True
    limit_switch_ok: bool = True
    _position_mm: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    _homed: bool = False
    _serialized_commands: list[dict[str, object]] = field(default_factory=list)

    def connect(self) -> None:
        self.connected = True
        self.stopped = False

    def home(self) -> None:
        self._require_connected()
        self._position_mm = [0.0, 0.0, 0.0]
        self._homed = True
        self._record({"kind": "home", "port": self.port})

    def move_to(self, point_mm: list[float], speed: SpeedProfile | None = None) -> MotionResult:
        self._require_ready()
        if len(point_mm) != 3:
            raise ValueError("point_mm must contain [x, y, z]")
        start = list(self._position_mm)
        self._position_mm = [float(value) for value in point_mm]
        self._record(
            {
                "kind": "move_to",
                "point_mm": list(self._position_mm),
                "speed": (speed or SpeedProfile()).to_dict(),
                "mode": "dry_run_pointer",
            }
        )
        return MotionResult(
            ok=True,
            start_mm=start,
            end_mm=list(self._position_mm),
            speed=speed or SpeedProfile(),
            message="dry-run command serialized; no hardware movement performed",
        )

    def mark_or_act(self, profile: ToolProfile | None = None) -> ToolResult:
        self._require_ready()
        profile = profile or ToolProfile()
        if profile.profile_id != "pointer_only":
            return ToolResult(ok=False, profile=profile, message="dry-run adapter only allows pointer_only")
        self._record({"kind": "mark_or_act", "profile": profile.to_dict(), "mode": "dry_run_pointer"})
        return ToolResult(ok=True, profile=profile, message="dry-run pointer mark serialized")

    def emergency_stop(self) -> None:
        self.stopped = True
        self._record({"kind": "emergency_stop"})

    def telemetry(self) -> RobotTelemetry:
        return RobotTelemetry(
            position_mm=list(self._position_mm),
            homed=self._homed,
            mode="dry_run_pointer",
            interlock_ok=self.interlock_ok,
            operator_armed=False,
            limit_switch_ok=self.limit_switch_ok,
            emergency_stop_active=self.stopped,
        )

    def execute_command(
        self,
        command: ActionCommand,
        scene: SceneState,
        safety_gate: SafetyGate | None = None,
    ) -> RobotExecutionResult:
        self._require_connected()
        decision = (safety_gate or SafetyGate()).validate(command, scene, self.telemetry())
        attach_safety_decision(command, decision)
        if not decision.allowed:
            result = RobotExecutionResult(False, command, decision, message="blocked by SafetyGate")
            self._record({"kind": "blocked_motion_command", "result": result.to_dict()})
            self._append_execution(result)
            return result
        tool_profile = ToolProfile(profile_id=command.tool_profile)
        if tool_profile.profile_id != "pointer_only":
            tool_result = ToolResult(False, tool_profile, "dry-run adapter only allows pointer_only")
            result = RobotExecutionResult(
                ok=False,
                command=command,
                safety_decision=decision,
                outcome={"tool_result": tool_result.to_dict()},
                message=tool_result.message,
            )
            self._record({"kind": "blocked_tool_profile", "result": result.to_dict()})
            self._append_execution(result)
            return result
        motion = MotionCommand.from_action_command(command, mode="dry_run_pointer", safety_token=decision.result)
        self._record({"kind": "motion_command", "command": motion.to_dict(), "safety_decision": decision.to_dict()})
        self.move_to(motion.point_mm, motion.speed)
        tool_result = self.mark_or_act(tool_profile)
        result = RobotExecutionResult(
            ok=tool_result.ok,
            command=command,
            safety_decision=decision,
            outcome={"motion_command": motion.to_dict(), "tool_result": tool_result.to_dict()},
            message="dry-run command serialized",
        )
        self._append_execution(result)
        return result

    def serialized_commands(self) -> list[dict[str, object]]:
        return list(self._serialized_commands)

    def _record(self, payload: dict[str, object]) -> None:
        self._serialized_commands.append(payload)
        if self.command_log_path:
            path = Path(self.command_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"commands": self._serialized_commands}, ensure_ascii=False, indent=2), encoding="utf-8")

    def _append_execution(self, result: RobotExecutionResult) -> None:
        if self.replay_logger is not None:
            self.replay_logger.append_execution(result)

    def _require_connected(self) -> None:
        if not self.connected:
            raise RuntimeError("DryRunSerialAdapter is not connected")

    def _require_ready(self) -> None:
        self._require_connected()
        if self.stopped:
            raise RuntimeError("DryRunSerialAdapter is stopped")
        if not self._homed:
            raise RuntimeError("DryRunSerialAdapter must be homed before movement")
        if not self.interlock_ok:
            raise RuntimeError("dry-run interlock is not OK")
        if not self.limit_switch_ok:
            raise RuntimeError("dry-run limit switch state is not OK")
