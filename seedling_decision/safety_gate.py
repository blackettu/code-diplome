from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from seedling_core.config import ConfigError, load_config_file
from seedling_core.schemas import ActionCommand, ActionTarget, SceneState


SAFETY_MODES = {
    "simulation",
    "dry_run_pointer",
    "hardware_in_loop_pointer",
    "supervised_mark",
}
_BOOLEAN_LIMIT_FIELDS = {
    "allow_real_action",
    "require_operator_confirmation",
    "require_calibration_valid",
    "require_robot_homed",
    "block_unknown_plant",
    "block_ambiguous_cell",
    "block_foreign_object",
    "block_if_interlock_false",
    "log_all_blocked_actions",
}
_NUMERIC_LIMIT_FIELDS = {
    "min_distance_to_keep_mm",
    "max_target_uncertainty_mm",
    "max_zone_error_p95_mm",
}

ALLOW_SIMULATION = "ALLOW_SIMULATION"
ALLOW_DRY_RUN = "ALLOW_DRY_RUN"
ALLOW_HARDWARE_IN_LOOP_POINTER = "ALLOW_HARDWARE_IN_LOOP_POINTER"
ALLOW_SUPERVISED_MARK = "ALLOW_SUPERVISED_MARK"
BLOCK_REVIEW_REQUIRED = "BLOCK_REVIEW_REQUIRED"
BLOCK_CALIBRATION_REQUIRED = "BLOCK_CALIBRATION_REQUIRED"
BLOCK_INTERLOCK = "BLOCK_INTERLOCK"
BLOCK_UNSAFE_TARGET = "BLOCK_UNSAFE_TARGET"
BLOCK_SYSTEM_ERROR = "BLOCK_SYSTEM_ERROR"


@dataclass(frozen=True)
class RobotTelemetry:
    position_mm: list[float]
    homed: bool
    mode: str = "simulation"
    interlock_ok: bool = False
    operator_armed: bool = False
    error_map_p95_mm: float | None = None
    limit_switch_ok: bool = True
    emergency_stop_active: bool = False

    @classmethod
    def from_scene(cls, scene: SceneState) -> "RobotTelemetry":
        return cls(
            position_mm=scene.robot.position_mm,
            homed=scene.robot.homed,
            mode=scene.robot.mode,
            interlock_ok=scene.safety.interlock_ok,
            operator_armed=False,
            limit_switch_ok=True,
            emergency_stop_active=False,
        )


@dataclass(frozen=True)
class SafetyLimits:
    mode: str = "dry_run_pointer"
    allow_real_action: bool = False
    require_operator_confirmation: bool = True
    require_calibration_valid: bool = True
    require_robot_homed: bool = True
    min_distance_to_keep_mm: float = 5.0
    max_target_uncertainty_mm: float = 2.0
    max_zone_error_p95_mm: float = 2.0
    block_unknown_plant: bool = True
    block_ambiguous_cell: bool = True
    block_foreign_object: bool = True
    block_if_interlock_false: bool = True
    log_all_blocked_actions: bool = True

    def __post_init__(self) -> None:
        _validate_safety_mode(self.mode, "mode")
        for field_name in _BOOLEAN_LIMIT_FIELDS:
            value = getattr(self, field_name)
            if type(value) is not bool:
                raise ValueError(f"{field_name} must be bool")
        for field_name in _NUMERIC_LIMIT_FIELDS:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be numeric")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")


def load_safety_limits(path: str | Path) -> SafetyLimits:
    """Load and validate safety limits from a JSON/YAML mapping."""
    return safety_limits_from_mapping(load_config_file(path), source=str(path))


def safety_limits_from_mapping(
    data: Mapping[str, Any],
    source: str = "safety limits",
) -> SafetyLimits:
    allowed_fields = {"mode"} | _BOOLEAN_LIMIT_FIELDS | _NUMERIC_LIMIT_FIELDS
    unknown_fields = sorted(set(data) - allowed_fields)
    if unknown_fields:
        raise ConfigError(f"{source}: unknown safety limit field(s): {', '.join(unknown_fields)}")

    values: dict[str, Any] = {}
    errors: list[str] = []
    for key, value in data.items():
        if key == "mode":
            if not isinstance(value, str):
                errors.append(f"`mode` must be str, got {type(value).__name__}")
            else:
                values[key] = value
            continue
        if key in _BOOLEAN_LIMIT_FIELDS:
            if type(value) is not bool:
                errors.append(f"`{key}` must be bool, got {type(value).__name__}")
            else:
                values[key] = value
            continue
        if key in _NUMERIC_LIMIT_FIELDS:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"`{key}` must be numeric, got {type(value).__name__}")
            else:
                values[key] = float(value)
    if errors:
        raise ConfigError(f"{source}: {'; '.join(errors)}")

    try:
        return SafetyLimits(**values)
    except ValueError as exc:
        raise ConfigError(f"{source}: {exc}") from exc


@dataclass(frozen=True)
class SafetyDecision:
    result: str
    allowed: bool
    reasons: list[str] = field(default_factory=list)
    command_id: str | None = None
    target_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "result": self.result,
            "allowed": self.allowed,
            "reasons": self.reasons,
            "command_id": self.command_id,
            "target_id": self.target_id,
        }


@dataclass(frozen=True)
class SafetyGate:
    limits: SafetyLimits = field(default_factory=SafetyLimits)

    def validate(
        self,
        command: ActionCommand,
        scene: SceneState,
        telemetry: RobotTelemetry | None = None,
    ) -> SafetyDecision:
        telemetry = telemetry or RobotTelemetry.from_scene(scene)
        target = _target_by_id(scene, command.target_id)
        if target is None:
            return _decision(BLOCK_SYSTEM_ERROR, command, None, ["target_not_found"])
        if telemetry.mode not in SAFETY_MODES:
            return _decision(BLOCK_SYSTEM_ERROR, command, target, ["invalid_robot_mode"])

        calibration_errors = self._calibration_errors(scene, telemetry)
        if calibration_errors:
            return _decision(BLOCK_CALIBRATION_REQUIRED, command, target, calibration_errors)

        interlock_errors = self._interlock_errors(telemetry)
        if interlock_errors:
            return _decision(BLOCK_INTERLOCK, command, target, interlock_errors)

        target_errors = self._target_errors(target, scene)
        if target_errors:
            return _decision(BLOCK_UNSAFE_TARGET, command, target, target_errors)

        if _is_real_action(command.action_type):
            real_action_errors = []
            if not scene.safety.software_safe_mode:
                real_action_errors.append("software_safe_mode_disabled")
            if not scene.safety.enclosure_closed:
                real_action_errors.append("enclosure_not_closed")
            if not self.limits.allow_real_action:
                real_action_errors.append("real_action_not_allowed")
            if real_action_errors:
                return _decision(BLOCK_UNSAFE_TARGET, command, target, real_action_errors)

        if command.requires_operator_confirmation and self.limits.require_operator_confirmation:
            return _decision(BLOCK_REVIEW_REQUIRED, command, target, ["operator_confirmation_required"])

        if telemetry.mode == "simulation" or self.limits.mode == "simulation":
            return _decision(ALLOW_SIMULATION, command, target, [], allowed=True)
        if telemetry.mode == "hardware_in_loop_pointer" or self.limits.mode == "hardware_in_loop_pointer":
            return _decision(ALLOW_HARDWARE_IN_LOOP_POINTER, command, target, [], allowed=True)
        if telemetry.mode == "dry_run_pointer" or self.limits.mode == "dry_run_pointer":
            return _decision(ALLOW_DRY_RUN, command, target, [], allowed=True)
        return _decision(ALLOW_SUPERVISED_MARK, command, target, [], allowed=True)

    def _calibration_errors(self, scene: SceneState, telemetry: RobotTelemetry) -> list[str]:
        errors = []
        if self.limits.require_calibration_valid and not scene.safety.calibration_valid:
            errors.append("calibration_required")
        if self.limits.require_robot_homed and not telemetry.homed:
            errors.append("robot_not_homed")
        if telemetry.error_map_p95_mm is not None and telemetry.error_map_p95_mm > self.limits.max_zone_error_p95_mm:
            errors.append("zone_error_too_high")
        return errors

    def _interlock_errors(self, telemetry: RobotTelemetry) -> list[str]:
        if self.limits.block_if_interlock_false and not telemetry.interlock_ok and telemetry.mode != "simulation":
            return ["interlock_false"]
        if telemetry.emergency_stop_active:
            return ["emergency_stop_active"]
        if not telemetry.limit_switch_ok:
            return ["limit_switch_not_ok"]
        return []

    def _target_errors(self, target: ActionTarget, scene: SceneState) -> list[str]:
        errors = []
        cell = next((item for item in scene.cells if item.cell_id == target.cell_id), None)
        if self.limits.block_unknown_plant and cell is not None:
            if cell.state == "unknown" or "unknown_plant_present" in cell.risk_flags:
                errors.append("unknown_plant_present")
        if self.limits.block_ambiguous_cell and cell is not None:
            if cell.state in {"ambiguous", "image_quality_insufficient", "invalid_geometry"}:
                errors.append("ambiguous_or_invalid_cell")
        if self.limits.block_foreign_object and cell is not None:
            if cell.state == "foreign_object_present" or "foreign_object_present" in cell.risk_flags:
                errors.append("foreign_object_present")
        if target.uncertainty_radius_mm is not None and target.uncertainty_radius_mm > self.limits.max_target_uncertainty_mm:
            errors.append("target_uncertainty_too_high")
        if target.min_distance_to_keep_mm is not None and target.min_distance_to_keep_mm < self.limits.min_distance_to_keep_mm:
            errors.append("too_close_to_keep_seedling")
        if target.forbidden_zone_ids:
            errors.append("forbidden_zone_overlap")
        if _outside_tray(target, scene):
            errors.append("target_outside_tray")
        return errors


def _target_by_id(scene: SceneState, target_id: str) -> ActionTarget | None:
    return next((target for target in scene.targets if target.target_id == target_id), None)


def _decision(
    result: str,
    command: ActionCommand,
    target: ActionTarget | None,
    reasons: list[str],
    allowed: bool = False,
) -> SafetyDecision:
    return SafetyDecision(
        result=result,
        allowed=allowed,
        reasons=reasons,
        command_id=command.command_id,
        target_id=target.target_id if target is not None else command.target_id,
    )


def _is_real_action(action_type: str) -> bool:
    return action_type in {"remove_weed", "remove_extra_crop", "laser_fire", "real_act", "cut", "burn"}


def _validate_safety_mode(value: str, label: str) -> None:
    if value not in SAFETY_MODES:
        modes = ", ".join(sorted(SAFETY_MODES))
        raise ValueError(f"{label} must be one of: {modes}")


def _outside_tray(target: ActionTarget, scene: SceneState) -> bool:
    bbox = scene.tray.bbox_xyxy_px
    if bbox is None and scene.tray.corners_px is not None:
        xs = [point[0] for point in scene.tray.corners_px]
        ys = [point[1] for point in scene.tray.corners_px]
        bbox = [min(xs), min(ys), max(xs), max(ys)]
    if bbox is None:
        return False
    x, y = target.action_point_px
    return not (bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3])
