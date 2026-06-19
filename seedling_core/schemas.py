from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = "seedling_scene_v0_1"


@dataclass
class DetectionObject:
    object_id: str
    class_name: str
    class_id: int
    confidence: float
    bbox_xyxy_px: list[float]
    center_px: list[float] | None = None
    area_px2: float | None = None
    mask_ref: str | None = None
    keypoints_px: dict[str, list[float]] = field(default_factory=dict)
    source_model: str | None = None
    uncertainty: dict[str, float] = field(default_factory=dict)
    attributes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.bbox_xyxy_px = _float_list("bbox_xyxy_px", self.bbox_xyxy_px, 4)
        if self.bbox_xyxy_px[2] < self.bbox_xyxy_px[0] or self.bbox_xyxy_px[3] < self.bbox_xyxy_px[1]:
            raise ValueError("bbox_xyxy_px must be ordered as [x1, y1, x2, y2]")
        self.confidence = float(self.confidence)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.center_px is None:
            x1, y1, x2, y2 = self.bbox_xyxy_px
            self.center_px = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
        else:
            self.center_px = _float_list("center_px", self.center_px, 2)
        if self.area_px2 is None:
            x1, y1, x2, y2 = self.bbox_xyxy_px
            self.area_px2 = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        else:
            self.area_px2 = float(self.area_px2)
        self.class_id = int(self.class_id)
        self.class_name = str(self.class_name)
        self.object_id = str(self.object_id)
        self.keypoints_px = {
            str(name): _float_list(f"keypoints_px.{name}", value, 2)
            for name, value in self.keypoints_px.items()
        }
        self.uncertainty = {str(key): float(value) for key, value in self.uncertainty.items()}
        self.attributes = [str(value) for value in self.attributes]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DetectionObject":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CellState:
    cell_id: str
    row: int
    col: int
    polygon_px: list[list[float]]
    state: str
    polygon_mm: list[list[float]] | None = None
    state_confidence: float = 1.0
    object_ids: list[str] = field(default_factory=list)
    keep_object_id: str | None = None
    removal_candidate_ids: list[str] = field(default_factory=list)
    human_review_required: bool = False
    risk_flags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.row = int(self.row)
        self.col = int(self.col)
        if self.row < 0 or self.col < 0:
            raise ValueError("row and col must be non-negative")
        self.cell_id = self.cell_id or cell_id(self.row, self.col)
        self.polygon_px = [_float_list("polygon_px[]", point, 2) for point in self.polygon_px]
        if len(self.polygon_px) < 3:
            raise ValueError("polygon_px must contain at least three points")
        if self.polygon_mm is not None:
            self.polygon_mm = [_float_list("polygon_mm[]", point, 2) for point in self.polygon_mm]
        self.state = str(self.state)
        self.state_confidence = float(self.state_confidence)
        if not 0.0 <= self.state_confidence <= 1.0:
            raise ValueError("state_confidence must be in [0, 1]")
        self.object_ids = [str(value) for value in self.object_ids]
        self.removal_candidate_ids = [str(value) for value in self.removal_candidate_ids]
        self.risk_flags = [str(value) for value in self.risk_flags]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CellState":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActionTarget:
    target_id: str
    cell_id: str
    object_id: str
    target_type: str
    action_point_px: list[float]
    action_point_mm: list[float] | None = None
    robot_point_mm: list[float] | None = None
    point_type: str = "bbox_center"
    uncertainty_radius_mm: float | None = None
    min_distance_to_keep_mm: float | None = None
    forbidden_zone_ids: list[str] = field(default_factory=list)
    decision_source: str = "unknown"
    human_review_required: bool = False
    risk_score: float = 0.0

    def __post_init__(self) -> None:
        self.target_id = str(self.target_id)
        self.cell_id = str(self.cell_id)
        self.object_id = str(self.object_id)
        self.target_type = str(self.target_type)
        self.action_point_px = _float_list("action_point_px", self.action_point_px, 2)
        if self.action_point_mm is not None:
            self.action_point_mm = _float_list("action_point_mm", self.action_point_mm, 2)
        if self.robot_point_mm is not None:
            self.robot_point_mm = _float_list("robot_point_mm", self.robot_point_mm, 3)
        if self.uncertainty_radius_mm is not None:
            self.uncertainty_radius_mm = float(self.uncertainty_radius_mm)
        if self.min_distance_to_keep_mm is not None:
            self.min_distance_to_keep_mm = float(self.min_distance_to_keep_mm)
        self.forbidden_zone_ids = [str(value) for value in self.forbidden_zone_ids]
        self.risk_score = float(self.risk_score)
        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError("risk_score must be in [0, 1]")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionTarget":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrayState:
    tray_id: str
    grid_rows: int
    grid_cols: int
    bbox_xyxy_px: list[float] | None = None
    corners_px: list[list[float]] | None = None
    calibration_id: str | None = None

    def __post_init__(self) -> None:
        self.tray_id = str(self.tray_id)
        self.grid_rows = int(self.grid_rows)
        self.grid_cols = int(self.grid_cols)
        if self.grid_rows <= 0 or self.grid_cols <= 0:
            raise ValueError("grid_rows and grid_cols must be positive")
        if self.bbox_xyxy_px is not None:
            self.bbox_xyxy_px = _float_list("bbox_xyxy_px", self.bbox_xyxy_px, 4)
        if self.corners_px is not None:
            self.corners_px = [_float_list("corners_px[]", point, 2) for point in self.corners_px]
            if len(self.corners_px) != 4:
                raise ValueError("corners_px must contain four points")
        if self.bbox_xyxy_px is None and self.corners_px is None:
            raise ValueError("TrayState requires bbox_xyxy_px or corners_px")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrayState":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RobotState:
    position_mm: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    homed: bool = False
    mode: str = "simulation"

    def __post_init__(self) -> None:
        self.position_mm = _float_list("position_mm", self.position_mm, 3)
        self.homed = bool(self.homed)
        self.mode = str(self.mode)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RobotState":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SafetyState:
    calibration_valid: bool = False
    enclosure_closed: bool = False
    software_safe_mode: bool = True
    interlock_ok: bool = False
    flags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.calibration_valid = bool(self.calibration_valid)
        self.enclosure_closed = bool(self.enclosure_closed)
        self.software_safe_mode = bool(self.software_safe_mode)
        self.interlock_ok = bool(self.interlock_ok)
        self.flags = [str(value) for value in self.flags]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SafetyState":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SceneState:
    scene_id: str
    image_ref: str
    dataset_version: str
    ontology_version: str
    image_size_px: list[int]
    tray: TrayState
    robot: RobotState = field(default_factory=RobotState)
    detections: list[DetectionObject] = field(default_factory=list)
    cells: list[CellState] = field(default_factory=list)
    targets: list[ActionTarget] = field(default_factory=list)
    safety: SafetyState = field(default_factory=SafetyState)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.scene_id = str(self.scene_id)
        self.image_ref = str(self.image_ref)
        self.dataset_version = str(self.dataset_version)
        self.ontology_version = str(self.ontology_version)
        self.image_size_px = [int(value) for value in _float_list("image_size_px", self.image_size_px, 2)]
        if isinstance(self.tray, dict):
            self.tray = TrayState.from_dict(self.tray)
        if isinstance(self.robot, dict):
            self.robot = RobotState.from_dict(self.robot)
        if isinstance(self.safety, dict):
            self.safety = SafetyState.from_dict(self.safety)
        self.detections = [
            item if isinstance(item, DetectionObject) else DetectionObject.from_dict(item)
            for item in self.detections
        ]
        self.cells = [
            item if isinstance(item, CellState) else CellState.from_dict(item)
            for item in self.cells
        ]
        self.targets = [
            item if isinstance(item, ActionTarget) else ActionTarget.from_dict(item)
            for item in self.targets
        ]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SceneState":
        payload = dict(data)
        payload.pop("schema_version", None)
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActionCommand:
    command_id: str
    target_id: str
    action_type: str
    robot_point_mm: list[float] | None = None
    tool_profile: str = "pointer_only"
    requires_operator_confirmation: bool = True
    safety_gate_result: str = "not_evaluated"
    created_by_policy: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.command_id = str(self.command_id)
        self.target_id = str(self.target_id)
        self.action_type = str(self.action_type)
        if self.robot_point_mm is not None:
            self.robot_point_mm = _float_list("robot_point_mm", self.robot_point_mm, 3)
        self.requires_operator_confirmation = bool(self.requires_operator_confirmation)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionCommand":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActionPlan:
    plan_id: str
    policy_id: str
    commands: list[ActionCommand] = field(default_factory=list)
    review_target_ids: list[str] = field(default_factory=list)
    blocked_target_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.commands = [
            command if isinstance(command, ActionCommand) else ActionCommand.from_dict(command)
            for command in self.commands
        ]
        self.review_target_ids = [str(value) for value in self.review_target_ids]
        self.blocked_target_ids = [str(value) for value in self.blocked_target_ids]
        self.metadata = {str(key): value for key, value in self.metadata.items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionPlan":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelMetadata:
    model_id: str
    backend: str
    model_uri: str | None = None
    ontology_version: str | None = None
    class_names: list[str] = field(default_factory=list)
    input_schema: str = "image_rgb"
    output_schema: str = "DetectionResultV1"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InferenceContext:
    image_id: str | None = None
    image_size_px: list[int] | None = None
    ontology_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.image_size_px is not None:
            self.image_size_px = [int(value) for value in _float_list("image_size_px", self.image_size_px, 2)]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InferenceContext":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DetectionResultV1:
    image_ref: str
    image_size_px: list[int]
    detections: list[DetectionObject]
    model_metadata: ModelMetadata | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.image_ref = str(self.image_ref)
        self.image_size_px = [int(value) for value in _float_list("image_size_px", self.image_size_px, 2)]
        self.detections = [
            item if isinstance(item, DetectionObject) else DetectionObject.from_dict(item)
            for item in self.detections
        ]
        if isinstance(self.model_metadata, dict):
            self.model_metadata = ModelMetadata.from_dict(self.model_metadata)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DetectionResultV1":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def cell_id(row: int, col: int) -> str:
    return f"r{int(row):02d}_c{int(col):02d}"


def _float_list(name: str, value: Any, expected_length: int) -> list[float]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list/tuple of length {expected_length}")
    if len(value) != expected_length:
        raise ValueError(f"{name} must contain {expected_length} values")
    return [float(item) for item in value]
