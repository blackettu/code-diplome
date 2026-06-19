from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CALIBRATION_SCHEMA_VERSION = "calibration_artifact_v0_1"


@dataclass
class ErrorSummary:
    p50: float | None = None
    p95: float | None = None
    p99: float | None = None
    rms: float | None = None
    max: float | None = None

    def __post_init__(self) -> None:
        for name in ["p50", "p95", "p99", "rms", "max"]:
            value = getattr(self, name)
            if value is not None:
                value = float(value)
                if value < 0.0:
                    raise ValueError(f"error_summary_mm.{name} must be non-negative")
                setattr(self, name, value)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ErrorSummary":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CalibrationConfig:
    camera_id: str
    tray_type: str
    tray_size_mm: list[float]
    grid_rows: int
    grid_cols: int
    target_points_px: list[list[float]] = field(default_factory=list)
    target_points_tray_mm: list[list[float]] = field(default_factory=list)
    robot_reference_points_mm: list[list[float]] = field(default_factory=list)
    valid_hours: float = 8.0

    def __post_init__(self) -> None:
        self.camera_id = str(self.camera_id)
        self.tray_type = str(self.tray_type)
        self.tray_size_mm = _float_list("tray_size_mm", self.tray_size_mm, 2)
        self.grid_rows = int(self.grid_rows)
        self.grid_cols = int(self.grid_cols)
        if self.grid_rows <= 0 or self.grid_cols <= 0:
            raise ValueError("grid_rows and grid_cols must be positive")
        self.target_points_px = [_float_list("target_points_px[]", point, 2) for point in self.target_points_px]
        self.target_points_tray_mm = [
            _float_list("target_points_tray_mm[]", point, 2) for point in self.target_points_tray_mm
        ]
        self.robot_reference_points_mm = [
            _float_list("robot_reference_points_mm[]", point, 3) for point in self.robot_reference_points_mm
        ]
        if len(self.target_points_px) != len(self.target_points_tray_mm):
            raise ValueError("target_points_px and target_points_tray_mm must have the same length")
        self.valid_hours = float(self.valid_hours)
        if self.valid_hours <= 0.0:
            raise ValueError("valid_hours must be positive")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationConfig":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CalibrationArtifact:
    calibration_id: str
    created_at: str
    camera_id: str
    tray_type: str
    image_to_tray_homography: list[list[float]]
    tray_to_robot_transform: list[list[float]]
    tool_offset_mm: list[float]
    px_per_mm_x: float
    px_per_mm_y: float
    valid_until: str | None = None
    error_summary_mm: ErrorSummary = field(default_factory=ErrorSummary)
    schema_version: str = CALIBRATION_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.calibration_id = str(self.calibration_id)
        self.created_at = _iso_datetime(self.created_at, "created_at")
        self.camera_id = str(self.camera_id)
        self.tray_type = str(self.tray_type)
        self.image_to_tray_homography = _matrix("image_to_tray_homography", self.image_to_tray_homography, 3, 3)
        self.tray_to_robot_transform = _matrix("tray_to_robot_transform", self.tray_to_robot_transform, 3, 3)
        self.tool_offset_mm = _float_list("tool_offset_mm", self.tool_offset_mm, 3)
        self.px_per_mm_x = float(self.px_per_mm_x)
        self.px_per_mm_y = float(self.px_per_mm_y)
        if self.px_per_mm_x <= 0.0 or self.px_per_mm_y <= 0.0:
            raise ValueError("px_per_mm_x and px_per_mm_y must be positive")
        if self.valid_until is not None:
            self.valid_until = _iso_datetime(self.valid_until, "valid_until")
        if isinstance(self.error_summary_mm, dict):
            self.error_summary_mm = ErrorSummary.from_dict(self.error_summary_mm)
        self.metadata = dict(self.metadata)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationArtifact":
        payload = dict(data)
        payload.pop("schema_version", None)
        return cls(**payload)

    @classmethod
    def from_json(cls, path: str | Path) -> "CalibrationArtifact":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Calibration artifact must be a JSON object: {path}")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass
class CalibrationValidationResult:
    ok: bool
    calibration_id: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationValidationResult":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _float_list(name: str, value: Any, expected_length: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != expected_length:
        raise ValueError(f"{name} must contain {expected_length} numeric values")
    return [float(item) for item in value]


def _matrix(name: str, value: Any, rows: int, cols: int) -> list[list[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != rows:
        raise ValueError(f"{name} must be a {rows}x{cols} matrix")
    return [_float_list(f"{name}[]", row, cols) for row in value]


def _iso_datetime(value: str, name: str) -> str:
    text = str(value)
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO datetime") from exc
    return text
