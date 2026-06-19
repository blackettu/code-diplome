from __future__ import annotations

import math
from dataclasses import dataclass, field

from seedling_sim.actuator_model import ActuatorErrorModel
from seedling_sim.plant_response_model import PlantResponseModel
from seedling_sim.schemas import SimScene, SimStepOutcome, SimTarget


@dataclass
class LogicalTraySimulator:
    actuator_model: ActuatorErrorModel | None = None
    plant_response_model: PlantResponseModel | None = None
    success_radius_mm: float = 3.0
    crop_damage_radius_mm: float = 2.0
    correct_remove_reward: float = 10.0
    crop_damage_penalty: float = -50.0
    miss_penalty: float = -5.0
    move_mm_penalty: float = 0.01
    seed: int | None = None
    scene: SimScene | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.actuator_model is None:
            self.actuator_model = ActuatorErrorModel(seed=self.seed)
        self.success_radius_mm = float(self.success_radius_mm)
        self.crop_damage_radius_mm = float(self.crop_damage_radius_mm)

    def reset(self, scene: SimScene) -> SimScene:
        self.scene = SimScene.from_dict(scene.to_dict())
        self.scene.step_index = 0
        return self.scene

    def step_target(self, target: SimTarget) -> SimStepOutcome:
        if self.scene is None:
            raise RuntimeError("Simulator must be reset with a scene before step_target()")
        active_target = self.scene.target_by_id(target.target_id)
        if active_target is None:
            raise ValueError(f"Unknown target: {target.target_id}")
        plant = self.scene.plant_by_id(active_target.object_id)
        if plant is None:
            raise ValueError(f"Target {target.target_id} references unknown plant {target.object_id}")

        sample = self.actuator_model.sample(active_target.point_mm)
        distance_error = _distance(sample.actual_point_mm, plant.position_mm)
        success = distance_error <= self.success_radius_mm
        crop_damage = self._crop_damage(sample.actual_point_mm, plant.object_id)
        plant_response = None
        if self.plant_response_model is not None:
            plant_response = self.plant_response_model.evaluate(plant, distance_error, crop_damage)
            success = plant_response.success
            crop_damage = plant_response.damage
        movement = _distance(self.scene.robot_position_mm[:2], sample.commanded_point_mm)
        reward = -movement * self.move_mm_penalty
        if success:
            plant.removed = True
            reward += self.correct_remove_reward
        else:
            reward += self.miss_penalty
        if crop_damage:
            reward += self.crop_damage_penalty

        active_target.processed = True
        self.scene.step_index += 1
        self.scene.robot_position_mm = [sample.actual_point_mm[0], sample.actual_point_mm[1], 0.0]
        return SimStepOutcome(
            step_index=self.scene.step_index,
            target_id=active_target.target_id,
            object_id=active_target.object_id,
            commanded_point_mm=sample.commanded_point_mm,
            actual_point_mm=sample.actual_point_mm,
            distance_error_mm=distance_error,
            success=success,
            crop_damage=crop_damage,
            reward=reward,
            info={
                "latency_ms": sample.latency_ms,
                "drift_mm": sample.drift_mm,
                "zone_error_mm": sample.zone_error_mm,
                "zone_error_active": sample.zone_error_active,
                "movement_mm": movement,
                "plant_response": plant_response.to_dict() if plant_response is not None else None,
            },
        )

    def _crop_damage(self, actual_point_mm: list[float], target_object_id: str) -> bool:
        assert self.scene is not None
        for plant in self.scene.plants:
            if plant.object_id == target_object_id or plant.removed:
                continue
            if plant.class_name in {"crop_seedling", "seedlings", "seedling"}:
                if _distance(actual_point_mm, plant.position_mm) <= self.crop_damage_radius_mm:
                    return True
        return False


def _distance(a: list[float], b: list[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
