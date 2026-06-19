from __future__ import annotations

import random
from dataclasses import dataclass

from seedling_sim.schemas import SimPlant


@dataclass(frozen=True)
class PlantResponse:
    success: bool
    damage: bool
    review_required: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "success": self.success,
            "damage": self.damage,
            "review_required": self.review_required,
            "reason": self.reason,
        }


@dataclass
class PlantResponseModel:
    """Probabilistic biological response stub.

    This deliberately avoids laser power/dose modeling. It only gives the
    simulator a configurable success/damage surface for policy research.
    """

    success_radius_mm: float = 3.0
    damage_radius_mm: float = 2.0
    base_success_prob: float = 0.95
    unknown_requires_review: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        self.success_radius_mm = float(self.success_radius_mm)
        self.damage_radius_mm = float(self.damage_radius_mm)
        self.base_success_prob = float(self.base_success_prob)
        if not 0.0 <= self.base_success_prob <= 1.0:
            raise ValueError("base_success_prob must be in [0, 1]")
        self._rng = random.Random(self.seed)

    def evaluate(self, plant: SimPlant, distance_error_mm: float, crop_damage: bool) -> PlantResponse:
        if plant.class_name in {"unknown_plant", "unknown"} and self.unknown_requires_review:
            return PlantResponse(False, False, True, "unknown_requires_review")
        if crop_damage:
            return PlantResponse(False, True, False, "near_keep_crop")
        if distance_error_mm > self.success_radius_mm:
            return PlantResponse(False, False, False, "missed_target")
        margin = max(0.0, self.success_radius_mm - distance_error_mm) / max(self.success_radius_mm, 1e-9)
        probability = min(1.0, self.base_success_prob * (0.5 + 0.5 * margin))
        success = self._rng.random() <= probability
        return PlantResponse(success, False, False, "hit_target" if success else "biological_nonresponse")
