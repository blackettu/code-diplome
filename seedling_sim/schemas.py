from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SIM_SCENE_SCHEMA_VERSION = "sim_scene_v0_1"


@dataclass
class SimPlant:
    object_id: str
    row: int
    col: int
    class_name: str
    position_mm: list[float]
    confidence: float = 1.0
    radius_mm: float = 2.0
    removed: bool = False
    attributes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.object_id = str(self.object_id)
        self.row = int(self.row)
        self.col = int(self.col)
        if self.row < 0 or self.col < 0:
            raise ValueError("row and col must be non-negative")
        self.class_name = str(self.class_name)
        self.position_mm = _float_list("position_mm", self.position_mm, 2)
        self.confidence = float(self.confidence)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        self.radius_mm = float(self.radius_mm)
        if self.radius_mm <= 0.0:
            raise ValueError("radius_mm must be positive")
        self.attributes = [str(value) for value in self.attributes]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimPlant":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SimTarget:
    target_id: str
    object_id: str
    target_type: str
    point_mm: list[float]
    uncertainty_radius_mm: float = 1.0
    min_distance_to_keep_mm: float | None = None
    processed: bool = False

    def __post_init__(self) -> None:
        self.target_id = str(self.target_id)
        self.object_id = str(self.object_id)
        self.target_type = str(self.target_type)
        self.point_mm = _float_list("point_mm", self.point_mm, 2)
        self.uncertainty_radius_mm = float(self.uncertainty_radius_mm)
        if self.uncertainty_radius_mm < 0.0:
            raise ValueError("uncertainty_radius_mm must be non-negative")
        if self.min_distance_to_keep_mm is not None:
            self.min_distance_to_keep_mm = float(self.min_distance_to_keep_mm)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimTarget":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SimScene:
    scene_id: str
    grid_rows: int
    grid_cols: int
    cell_size_mm: list[float]
    plants: list[SimPlant] = field(default_factory=list)
    targets: list[SimTarget] = field(default_factory=list)
    robot_position_mm: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    step_index: int = 0
    schema_version: str = SIM_SCENE_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.scene_id = str(self.scene_id)
        self.grid_rows = int(self.grid_rows)
        self.grid_cols = int(self.grid_cols)
        if self.grid_rows <= 0 or self.grid_cols <= 0:
            raise ValueError("grid_rows and grid_cols must be positive")
        self.cell_size_mm = _float_list("cell_size_mm", self.cell_size_mm, 2)
        self.robot_position_mm = _float_list("robot_position_mm", self.robot_position_mm, 3)
        self.plants = [item if isinstance(item, SimPlant) else SimPlant.from_dict(item) for item in self.plants]
        self.targets = [item if isinstance(item, SimTarget) else SimTarget.from_dict(item) for item in self.targets]
        self.step_index = int(self.step_index)
        self._validate_bounds()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimScene":
        payload = dict(data)
        payload.pop("schema_version", None)
        return cls(**payload)

    @classmethod
    def from_json(cls, path: str | Path) -> "SimScene":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"SimScene must be a JSON object: {path}")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def plant_by_id(self, object_id: str) -> SimPlant | None:
        return next((plant for plant in self.plants if plant.object_id == object_id), None)

    def target_by_id(self, target_id: str) -> SimTarget | None:
        return next((target for target in self.targets if target.target_id == target_id), None)

    def _validate_bounds(self) -> None:
        for plant in self.plants:
            if plant.row >= self.grid_rows or plant.col >= self.grid_cols:
                raise ValueError(f"plant {plant.object_id} is outside grid")
        plant_ids = {plant.object_id for plant in self.plants}
        for target in self.targets:
            if target.object_id not in plant_ids:
                raise ValueError(f"target {target.target_id} references unknown plant {target.object_id}")


@dataclass
class SimStepOutcome:
    step_index: int
    target_id: str
    object_id: str
    commanded_point_mm: list[float]
    actual_point_mm: list[float]
    distance_error_mm: float
    success: bool
    crop_damage: bool
    reward: float
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.commanded_point_mm = _float_list("commanded_point_mm", self.commanded_point_mm, 2)
        self.actual_point_mm = _float_list("actual_point_mm", self.actual_point_mm, 2)
        self.distance_error_mm = float(self.distance_error_mm)
        self.reward = float(self.reward)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _float_list(name: str, value: Any, expected_length: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != expected_length:
        raise ValueError(f"{name} must contain {expected_length} numeric values")
    return [float(item) for item in value]
