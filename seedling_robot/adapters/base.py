from __future__ import annotations

from abc import ABC, abstractmethod

from seedling_decision.safety_gate import RobotTelemetry
from seedling_robot.protocol import MotionResult, SpeedProfile, ToolProfile, ToolResult


class RobotAdapter(ABC):
    @abstractmethod
    def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def home(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def move_to(self, point_mm: list[float], speed: SpeedProfile | None = None) -> MotionResult:
        raise NotImplementedError

    @abstractmethod
    def mark_or_act(self, profile: ToolProfile | None = None) -> ToolResult:
        raise NotImplementedError

    @abstractmethod
    def emergency_stop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def telemetry(self) -> RobotTelemetry:
        raise NotImplementedError
