from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from seedling_core.schemas import ActionCommand
from seedling_decision.safety_gate import SafetyDecision


@dataclass(frozen=True)
class SpeedProfile:
    xy_mm_s: float = 50.0
    z_mm_s: float = 10.0

    def __post_init__(self) -> None:
        if self.xy_mm_s <= 0.0 or self.z_mm_s <= 0.0:
            raise ValueError("speed values must be positive")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SpeedProfile":
        if not data:
            return cls()
        return cls(
            xy_mm_s=float(data.get("xy_mm_s", 50.0)),
            z_mm_s=float(data.get("z_mm_s", 10.0)),
        )


@dataclass(frozen=True)
class ToolProfile:
    profile_id: str = "pointer_only"
    dwell_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dwell_ms < 0:
            raise ValueError("dwell_ms must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MotionResult:
    ok: bool
    start_mm: list[float]
    end_mm: list[float]
    speed: SpeedProfile
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "start_mm": self.start_mm,
            "end_mm": self.end_mm,
            "speed": self.speed.to_dict(),
            "message": self.message,
        }


@dataclass(frozen=True)
class MotionCommand:
    command_id: str
    point_mm: list[float]
    speed: SpeedProfile = field(default_factory=SpeedProfile)
    mode: str = "dry_run_pointer"
    safety_token: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.point_mm) != 3:
            raise ValueError("point_mm must contain [x, y, z]")
        object.__setattr__(self, "point_mm", [float(value) for value in self.point_mm])

    @classmethod
    def from_action_command(
        cls,
        command: ActionCommand,
        speed: SpeedProfile | None = None,
        mode: str = "dry_run_pointer",
        safety_token: str | None = None,
    ) -> "MotionCommand":
        if command.robot_point_mm is None:
            raise ValueError("ActionCommand.robot_point_mm is required to build MotionCommand")
        return cls(
            command_id=command.command_id,
            point_mm=command.robot_point_mm,
            speed=speed or SpeedProfile(),
            mode=mode,
            safety_token=safety_token,
            metadata={
                "target_id": command.target_id,
                "action_type": command.action_type,
                "tool_profile": command.tool_profile,
                "safety_gate_result": command.safety_gate_result,
            },
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MotionCommand":
        return cls(
            command_id=str(data["command_id"]),
            point_mm=list(data["point_mm"]),
            speed=SpeedProfile.from_dict(data.get("speed") if isinstance(data.get("speed"), dict) else None),
            mode=str(data.get("mode", "dry_run_pointer")),
            safety_token=str(data["safety_token"]) if data.get("safety_token") is not None else None,
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "point_mm": self.point_mm,
            "speed": self.speed.to_dict(),
            "mode": self.mode,
            "safety_token": self.safety_token,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    profile: ToolProfile
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "profile": self.profile.to_dict(), "message": self.message}


@dataclass(frozen=True)
class RobotExecutionResult:
    ok: bool
    command: ActionCommand
    safety_decision: SafetyDecision
    outcome: dict[str, Any] | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        command_payload = self.command.to_dict()
        command_payload["safety_gate_result"] = self.safety_decision.result
        return {
            "ok": self.ok,
            "command": command_payload,
            "safety_decision": self.safety_decision.to_dict(),
            "outcome": self.outcome,
            "message": self.message,
        }


def attach_safety_decision(command: ActionCommand, decision: SafetyDecision) -> ActionCommand:
    command.safety_gate_result = decision.result
    return command
