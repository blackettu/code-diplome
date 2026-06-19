from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from seedling_core.schemas import ActionCommand, SceneState
from seedling_decision.safety_gate import RobotTelemetry, SafetyGate, SafetyLimits
from seedling_robot.adapters.base import RobotAdapter
from seedling_robot.hil_safety import HardwareInLoopReview, validate_hil_review
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
class RealGantrySerialAdapter(RobotAdapter):
    """Guarded HIL pointer adapter.

    This adapter is intentionally limited to `pointer_only`. It requires an
    explicit safety-review artifact and `allow_hardware=True` before it can open
    a serial connection.
    """

    port: str
    review: HardwareInLoopReview
    baudrate: int = 115200
    timeout_s: float = 1.0
    allow_hardware: bool = False
    command_log_path: str | Path | None = None
    replay_logger: ReplayLogger | None = None
    interlock_ok: bool = False
    limit_switch_ok: bool = True
    _serial: Any | None = field(default=None, init=False)
    _position_mm: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    _homed: bool = False
    _stopped: bool = False
    _commands: list[dict[str, Any]] = field(default_factory=list)

    def connect(self) -> None:
        validation = validate_hil_review(
            self.review,
            required_mode="hardware_in_loop_pointer",
            required_tool_profile="pointer_only",
            require_positioning_error=True,
        )
        if not validation["ok"]:
            raise RuntimeError(f"HIL review is not valid: {validation['errors']}")
        if not self.allow_hardware:
            raise RuntimeError("allow_hardware=True is required to open a real serial connection")
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("pyserial is required for RealGantrySerialAdapter") from exc
        self._serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout_s)
        self._record({"kind": "connect", "port": self.port, "baudrate": self.baudrate})

    def home(self) -> None:
        self._require_connected()
        self._write_line("HOME")
        self._homed = True
        self._position_mm = [0.0, 0.0, 0.0]
        self._record({"kind": "home"})

    def move_to(self, point_mm: list[float], speed: SpeedProfile | None = None) -> MotionResult:
        self._require_ready()
        if len(point_mm) != 3:
            raise ValueError("point_mm must contain [x, y, z]")
        speed = speed or SpeedProfile()
        start = list(self._position_mm)
        target = [float(value) for value in point_mm]
        self._write_line(
            f"MOVE X{target[0]:.3f} Y{target[1]:.3f} Z{target[2]:.3f} F{speed.xy_mm_s:.3f}"
        )
        self._position_mm = target
        self._record({"kind": "move_to", "point_mm": target, "speed": speed.to_dict()})
        return MotionResult(ok=True, start_mm=start, end_mm=target, speed=speed, message="serial move command sent")

    def mark_or_act(self, profile: ToolProfile | None = None) -> ToolResult:
        self._require_ready()
        profile = profile or ToolProfile()
        if profile.profile_id != "pointer_only":
            return ToolResult(ok=False, profile=profile, message="HIL adapter only allows pointer_only")
        self._write_line("POINT")
        self._record({"kind": "mark_or_act", "profile": profile.to_dict()})
        return ToolResult(ok=True, profile=profile, message="pointer command sent")

    def emergency_stop(self) -> None:
        self._stopped = True
        if self._serial is not None:
            self._write_line("ESTOP")
        self._record({"kind": "emergency_stop"})

    def telemetry(self) -> RobotTelemetry:
        return RobotTelemetry(
            position_mm=list(self._position_mm),
            homed=self._homed,
            mode="hardware_in_loop_pointer",
            interlock_ok=self.interlock_ok,
            operator_armed=False,
            limit_switch_ok=self.limit_switch_ok,
            emergency_stop_active=self._stopped,
        )

    def execute_command(
        self,
        command: ActionCommand,
        scene: SceneState,
        safety_gate: SafetyGate | None = None,
    ) -> RobotExecutionResult:
        self._require_connected()
        gate = safety_gate or SafetyGate(
            SafetyLimits(
                mode="hardware_in_loop_pointer",
                require_operator_confirmation=True,
                allow_real_action=False,
            )
        )
        decision = gate.validate(command, scene, self.telemetry())
        attach_safety_decision(command, decision)
        if not decision.allowed:
            result = RobotExecutionResult(False, command, decision, message="blocked by SafetyGate")
            self._record({"kind": "blocked_motion_command", "result": result.to_dict()})
            self._append_execution(result)
            return result
        tool_profile = ToolProfile(profile_id=command.tool_profile)
        if tool_profile.profile_id != "pointer_only":
            tool_result = ToolResult(False, tool_profile, "HIL adapter only allows pointer_only")
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

        motion = MotionCommand.from_action_command(
            command,
            mode="hardware_in_loop_pointer",
            safety_token=decision.result,
        )
        self._record({"kind": "motion_command", "command": motion.to_dict(), "safety_decision": decision.to_dict()})
        self.move_to(motion.point_mm, motion.speed)
        tool_result = self.mark_or_act(tool_profile)
        result = RobotExecutionResult(
            ok=tool_result.ok,
            command=command,
            safety_decision=decision,
            outcome={"motion_command": motion.to_dict(), "tool_result": tool_result.to_dict()},
            message="hardware-in-loop pointer command sent" if tool_result.ok else tool_result.message,
        )
        self._append_execution(result)
        return result

    def serialized_commands(self) -> list[dict[str, Any]]:
        return list(self._commands)

    def _write_line(self, line: str) -> None:
        self._require_connected()
        assert self._serial is not None
        self._serial.write((line + "\n").encode("ascii"))

    def _record(self, payload: dict[str, Any]) -> None:
        self._commands.append(payload)
        if self.command_log_path:
            output = Path(self.command_log_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({"commands": self._commands}, ensure_ascii=False, indent=2), encoding="utf-8")

    def _append_execution(self, result: RobotExecutionResult) -> None:
        if self.replay_logger is not None:
            self.replay_logger.append_execution(result)

    def _require_connected(self) -> None:
        if self._serial is None:
            raise RuntimeError("RealGantrySerialAdapter is not connected")

    def _require_ready(self) -> None:
        self._require_connected()
        if self._stopped:
            raise RuntimeError("RealGantrySerialAdapter is stopped")
        if not self._homed:
            raise RuntimeError("RealGantrySerialAdapter must be homed before movement")
        if not self.interlock_ok:
            raise RuntimeError("hardware interlock is not OK")
        if not self.limit_switch_ok:
            raise RuntimeError("limit switch state is not OK")
