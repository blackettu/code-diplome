from __future__ import annotations

import random
from dataclasses import dataclass

from seedling_sim.schemas import SimPlant, SimScene, SimTarget


@dataclass(frozen=True)
class SceneGeneratorConfig:
    grid_rows: int = 11
    grid_cols: int = 11
    cell_size_mm: tuple[float, float] = (33.0, 33.0)
    p_empty: float = 0.65
    p_single_crop: float = 0.25
    p_multiple_crop: float = 0.07
    p_weed_present: float = 0.03
    p_unknown_present: float = 0.0
    max_extra_crops: int = 2
    seed: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "SceneGeneratorConfig":
        cell_size = data.get("cell_size_mm", (33.0, 33.0))
        if isinstance(cell_size, list):
            cell_size = tuple(float(value) for value in cell_size)
        return cls(
            grid_rows=int(data.get("grid_rows", 11)),
            grid_cols=int(data.get("grid_cols", 11)),
            cell_size_mm=cell_size,  # type: ignore[arg-type]
            p_empty=float(data.get("p_empty", 0.65)),
            p_single_crop=float(data.get("p_single_crop", 0.25)),
            p_multiple_crop=float(data.get("p_multiple_crop", 0.07)),
            p_weed_present=float(data.get("p_weed_present", 0.03)),
            p_unknown_present=float(data.get("p_unknown_present", 0.0)),
            max_extra_crops=int(data.get("max_extra_crops", 2)),
            seed=int(data["seed"]) if data.get("seed") is not None else None,
        )

    def validate(self) -> None:
        if self.grid_rows <= 0 or self.grid_cols <= 0:
            raise ValueError("grid_rows and grid_cols must be positive")
        for name in ["p_empty", "p_single_crop", "p_multiple_crop", "p_weed_present", "p_unknown_present"]:
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        total = self.p_empty + self.p_single_crop + self.p_multiple_crop + self.p_weed_present + self.p_unknown_present
        if total <= 0.0:
            raise ValueError("scene probabilities must have positive sum")


class SimSceneGenerator:
    def __init__(self, config: SceneGeneratorConfig | None = None) -> None:
        self.config = config or SceneGeneratorConfig()
        self.config.validate()
        self._rng = random.Random(self.config.seed)

    def generate(self, scene_id: str) -> SimScene:
        plants: list[SimPlant] = []
        targets: list[SimTarget] = []
        object_index = 0
        target_index = 0
        for row in range(self.config.grid_rows):
            for col in range(self.config.grid_cols):
                cell_kind = self._sample_cell_kind()
                if cell_kind == "empty":
                    continue
                cell_plants = self._plants_for_cell(cell_kind, row, col, object_index)
                object_index += len(cell_plants)
                plants.extend(cell_plants)
                target_candidates = self._targets_for_cell(cell_kind, cell_plants, target_index)
                target_index += len(target_candidates)
                targets.extend(target_candidates)
        return SimScene(
            scene_id=scene_id,
            grid_rows=self.config.grid_rows,
            grid_cols=self.config.grid_cols,
            cell_size_mm=list(self.config.cell_size_mm),
            plants=plants,
            targets=targets,
            metadata={"generator": self.config.__dict__},
        )

    def generate_many(self, count: int, prefix: str = "sim_scene") -> list[SimScene]:
        return [self.generate(f"{prefix}_{index:06d}") for index in range(count)]

    def _sample_cell_kind(self) -> str:
        weighted = [
            ("empty", self.config.p_empty),
            ("single_crop", self.config.p_single_crop),
            ("multiple_crop", self.config.p_multiple_crop),
            ("weed", self.config.p_weed_present),
            ("unknown", self.config.p_unknown_present),
        ]
        total = sum(weight for _, weight in weighted)
        draw = self._rng.random() * total
        cumulative = 0.0
        for name, weight in weighted:
            cumulative += weight
            if draw <= cumulative:
                return name
        return weighted[-1][0]

    def _plants_for_cell(self, cell_kind: str, row: int, col: int, start_index: int) -> list[SimPlant]:
        count = 1
        class_name = "crop_seedling"
        if cell_kind == "multiple_crop":
            count = 1 + self._rng.randint(1, max(1, self.config.max_extra_crops))
        elif cell_kind == "weed":
            class_name = "weed"
        elif cell_kind == "unknown":
            class_name = "unknown_plant"
        plants = []
        for offset in range(count):
            position = self._random_position(row, col)
            plants.append(
                SimPlant(
                    object_id=f"plant_{start_index + offset:06d}",
                    row=row,
                    col=col,
                    class_name=class_name,
                    position_mm=position,
                    confidence=self._rng.uniform(0.75, 0.99) if class_name != "unknown_plant" else self._rng.uniform(0.35, 0.75),
                    radius_mm=self._rng.uniform(1.2, 3.0),
                )
            )
        return plants

    def _targets_for_cell(self, cell_kind: str, plants: list[SimPlant], start_index: int) -> list[SimTarget]:
        if cell_kind == "multiple_crop":
            keep = max(plants, key=lambda plant: plant.radius_mm)
            target_plants = [plant for plant in plants if plant.object_id != keep.object_id]
            return [
                SimTarget(
                    target_id=f"target_{start_index + index:06d}",
                    object_id=plant.object_id,
                    target_type="remove_extra_crop",
                    point_mm=list(plant.position_mm),
                    uncertainty_radius_mm=1.0,
                    min_distance_to_keep_mm=_distance(plant.position_mm, keep.position_mm),
                )
                for index, plant in enumerate(target_plants)
            ]
        if cell_kind == "weed":
            return [
                SimTarget(
                    target_id=f"target_{start_index:06d}",
                    object_id=plants[0].object_id,
                    target_type="remove_weed",
                    point_mm=list(plants[0].position_mm),
                    uncertainty_radius_mm=1.0,
                )
            ]
        if cell_kind == "unknown":
            return [
                SimTarget(
                    target_id=f"target_{start_index:06d}",
                    object_id=plants[0].object_id,
                    target_type="human_review_required",
                    point_mm=list(plants[0].position_mm),
                    uncertainty_radius_mm=2.5,
                )
            ]
        return []

    def _random_position(self, row: int, col: int) -> list[float]:
        cell_w, cell_h = self.config.cell_size_mm
        return [
            col * cell_w + self._rng.uniform(cell_w * 0.2, cell_w * 0.8),
            row * cell_h + self._rng.uniform(cell_h * 0.2, cell_h * 0.8),
        ]


def _distance(a: list[float], b: list[float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
