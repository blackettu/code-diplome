from __future__ import annotations

from dataclasses import dataclass, field

from seedling_core.schemas import ActionCommand, SceneState
from seedling_decision.safety_gate import RobotTelemetry, SafetyGate
from seedling_robot.adapters.base import RobotAdapter
from seedling_robot.protocol import (
    MotionResult,
    RobotExecutionResult,
    SpeedProfile,
    ToolProfile,
    ToolResult,
    attach_safety_decision,
)
from seedling_sim.logical_tray import LogicalTraySimulator
from seedling_sim.replay import ReplayLogger
from seedling_sim.schemas import SimScene, SimStepOutcome


@dataclass
class SimulatorRobotAdapter(RobotAdapter):
    simulator: LogicalTraySimulator | None = None
    sim_scene: SimScene | None = None
    replay_logger: ReplayLogger | None = None
    connected: bool = False
    stopped: bool = False
    _position_mm: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    _homed: bool = False

    def connect(self) -> None:
        self.connected = True
        self.stopped = False
        if self.simulator is not None and self.sim_scene is not None:
            self.sim_scene = self.simulator.reset(self.sim_scene)

    def home(self) -> None:
        self._require_connected()
        self._position_mm = [0.0, 0.0, 0.0]
        self._homed = True
        if self.sim_scene is not None:
            self.sim_scene.robot_position_mm = list(self._position_mm)

    def move_to(self, point_mm: list[float], speed: SpeedProfile | None = None) -> MotionResult:
        self._require_ready()
        if len(point_mm) != 3:
            raise ValueError("point_mm must contain [x, y, z]")
        start = list(self._position_mm)
        self._position_mm = [float(point_mm[0]), float(point_mm[1]), float(point_mm[2])]
        if self.sim_scene is not None:
            self.sim_scene.robot_position_mm = list(self._position_mm)
        return MotionResult(
            ok=True,
            start_mm=start,
            end_mm=list(self._position_mm),
            speed=speed or SpeedProfile(),
            message="simulated move completed",
        )

    def mark_or_act(self, profile: ToolProfile | None = None) -> ToolResult:
        self._require_ready()
        profile = profile or ToolProfile()
        if profile.profile_id != "pointer_only":
            return ToolResult(ok=False, profile=profile, message="simulator allows pointer_only by default")
        return ToolResult(ok=True, profile=profile, message="simulated mark completed")

    def emergency_stop(self) -> None:
        self.stopped = True

    def telemetry(self) -> RobotTelemetry:
        return RobotTelemetry(
            position_mm=list(self._position_mm),
            homed=self._homed,
            mode="simulation",
            interlock_ok=True,
            operator_armed=False,
        )

    def execute_command(
        self,
        command: ActionCommand,
        scene: SceneState,
        safety_gate: SafetyGate | None = None,
    ) -> RobotExecutionResult:
        self._require_ready()
        gate = safety_gate or SafetyGate()
        decision = gate.validate(command, scene, self.telemetry())
        attach_safety_decision(command, decision)
        if not decision.allowed:
            result = RobotExecutionResult(
                ok=False,
                command=command,
                safety_decision=decision,
                message="blocked by SafetyGate",
            )
            if self.replay_logger is not None:
                self.replay_logger.append_execution(result)
            return result

        outcome: SimStepOutcome | None = None
        if self.simulator is not None and self.sim_scene is not None:
            target = self.sim_scene.target_by_id(command.target_id)
            if target is not None:
                outcome = self.simulator.step_target(target)
                self._position_mm = [outcome.actual_point_mm[0], outcome.actual_point_mm[1], 0.0]
        if outcome is None:
            if command.robot_point_mm is None:
                raise ValueError("command.robot_point_mm is required when no SimTarget is available")
            self.move_to(command.robot_point_mm)
            self.mark_or_act(ToolProfile(profile_id=command.tool_profile))

        result = RobotExecutionResult(
            ok=True,
            command=command,
            safety_decision=decision,
            outcome=outcome.to_dict() if outcome else None,
            message="simulated command executed",
        )
        if self.replay_logger is not None:
            self.replay_logger.append_execution(result)
        return result

    def _require_connected(self) -> None:
        if not self.connected:
            raise RuntimeError("SimulatorRobotAdapter is not connected")

    def _require_ready(self) -> None:
        self._require_connected()
        if self.stopped:
            raise RuntimeError("SimulatorRobotAdapter is stopped")
