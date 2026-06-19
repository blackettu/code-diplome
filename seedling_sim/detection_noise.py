from __future__ import annotations

import random
from dataclasses import dataclass

from seedling_core.schemas import DetectionObject
from seedling_sim.schemas import SimPlant, SimScene, SimTarget


@dataclass
class DetectionNoiseModel:
    bbox_center_sigma_mm: float = 1.0
    classification_error_prob: float = 0.03
    missed_detection_prob: float = 0.05
    false_positive_prob: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        self.bbox_center_sigma_mm = float(self.bbox_center_sigma_mm)
        self.classification_error_prob = float(self.classification_error_prob)
        self.missed_detection_prob = float(self.missed_detection_prob)
        self.false_positive_prob = float(self.false_positive_prob)
        for name in ["classification_error_prob", "missed_detection_prob", "false_positive_prob"]:
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        self._rng = random.Random(self.seed)

    def apply(self, detections: list[DetectionObject]) -> list[DetectionObject]:
        noisy: list[DetectionObject] = []
        for detection in detections:
            if self._rng.random() < self.missed_detection_prob:
                continue
            center = detection.center_px or [0.0, 0.0]
            dx = self._rng.gauss(0.0, self.bbox_center_sigma_mm)
            dy = self._rng.gauss(0.0, self.bbox_center_sigma_mm)
            bbox = list(detection.bbox_xyxy_px)
            bbox[0] += dx
            bbox[2] += dx
            bbox[1] += dy
            bbox[3] += dy
            class_name = detection.class_name
            class_id = detection.class_id
            attributes = list(detection.attributes)
            if self._rng.random() < self.classification_error_prob:
                class_name = "unknown_plant"
                class_id = 3
                attributes.append("classification_noise")
            noisy.append(
                DetectionObject(
                    object_id=detection.object_id,
                    class_name=class_name,
                    class_id=class_id,
                    confidence=detection.confidence,
                    bbox_xyxy_px=bbox,
                    center_px=[center[0] + dx, center[1] + dy],
                    source_model=detection.source_model,
                    uncertainty={**detection.uncertainty, "noise_sigma_mm": self.bbox_center_sigma_mm},
                    attributes=attributes,
                )
            )
        if self._rng.random() < self.false_positive_prob:
            noisy.append(self._false_positive(detections, len(noisy)))
        return noisy

    def _false_positive(self, detections: list[DetectionObject], index: int) -> DetectionObject:
        x_min, y_min, x_max, y_max = _detection_extent(detections)
        span_x = max(1.0, x_max - x_min)
        span_y = max(1.0, y_max - y_min)
        width = max(1.0, span_x * self._rng.uniform(0.03, 0.12))
        height = max(1.0, span_y * self._rng.uniform(0.03, 0.12))
        center_x = self._rng.uniform(x_min, x_max)
        center_y = self._rng.uniform(y_min, y_max)
        bbox = [
            max(x_min, center_x - width / 2.0),
            max(y_min, center_y - height / 2.0),
            min(x_max, center_x + width / 2.0),
            min(y_max, center_y + height / 2.0),
        ]
        if bbox[2] <= bbox[0]:
            bbox[2] = bbox[0] + 1.0
        if bbox[3] <= bbox[1]:
            bbox[3] = bbox[1] + 1.0
        return DetectionObject(
            object_id=f"noise_fp_{index:06d}",
            class_name="unknown_plant",
            class_id=3,
            confidence=self._rng.uniform(0.1, 0.45),
            bbox_xyxy_px=bbox,
            source_model="detection_noise",
            uncertainty={"false_positive_prob": self.false_positive_prob},
            attributes=["false_positive_noise"],
        )


def _detection_extent(detections: list[DetectionObject]) -> tuple[float, float, float, float]:
    if not detections:
        return 0.0, 0.0, 1.0, 1.0
    x_min = min(detection.bbox_xyxy_px[0] for detection in detections)
    y_min = min(detection.bbox_xyxy_px[1] for detection in detections)
    x_max = max(detection.bbox_xyxy_px[2] for detection in detections)
    y_max = max(detection.bbox_xyxy_px[3] for detection in detections)
    return x_min, y_min, x_max, y_max


@dataclass
class SimSceneDetectionNoiseModel:
    bbox_center_sigma_mm: float = 0.0
    classification_error_prob: float = 0.0
    missed_detection_prob: float = 0.0
    false_positive_prob: float = 0.0
    seed: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, object] | None, seed: int | None = None) -> "SimSceneDetectionNoiseModel":
        payload = data or {}
        return cls(
            bbox_center_sigma_mm=float(payload.get("bbox_center_sigma_mm", 0.0)),
            classification_error_prob=float(payload.get("classification_error_prob", 0.0)),
            missed_detection_prob=float(payload.get("missed_detection_prob", 0.0)),
            false_positive_prob=float(payload.get("false_positive_prob", 0.0)),
            seed=seed,
        )

    def __post_init__(self) -> None:
        self.bbox_center_sigma_mm = float(self.bbox_center_sigma_mm)
        self.classification_error_prob = float(self.classification_error_prob)
        self.missed_detection_prob = float(self.missed_detection_prob)
        self.false_positive_prob = float(self.false_positive_prob)
        for name in ["classification_error_prob", "missed_detection_prob", "false_positive_prob"]:
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.bbox_center_sigma_mm < 0.0:
            raise ValueError("bbox_center_sigma_mm must be non-negative")
        self._rng = random.Random(self.seed)

    def enabled(self) -> bool:
        return (
            self.bbox_center_sigma_mm > 0.0
            or self.classification_error_prob > 0.0
            or self.missed_detection_prob > 0.0
            or self.false_positive_prob > 0.0
        )

    def apply(self, scene: SimScene) -> SimScene:
        if not self.enabled():
            return scene
        plants: list[SimPlant] = []
        target_offsets: dict[str, list[float]] = {}
        noisy_classes: dict[str, str] = {}
        for plant in scene.plants:
            if self._rng.random() < self.missed_detection_prob:
                continue
            dx = self._rng.gauss(0.0, self.bbox_center_sigma_mm)
            dy = self._rng.gauss(0.0, self.bbox_center_sigma_mm)
            class_name = plant.class_name
            attributes = list(plant.attributes)
            confidence = plant.confidence
            if self._rng.random() < self.classification_error_prob:
                class_name = "unknown_plant"
                confidence = min(confidence, 0.5)
                attributes.append("classification_noise")
            noisy_classes[plant.object_id] = class_name
            target_offsets[plant.object_id] = [dx, dy]
            plants.append(
                SimPlant(
                    object_id=plant.object_id,
                    row=plant.row,
                    col=plant.col,
                    class_name=class_name,
                    position_mm=[plant.position_mm[0] + dx, plant.position_mm[1] + dy],
                    confidence=confidence,
                    radius_mm=plant.radius_mm,
                    removed=plant.removed,
                    attributes=attributes,
                )
            )
        targets = [
            self._noisy_target(target, target_offsets[target.object_id], noisy_classes[target.object_id])
            for target in scene.targets
            if target.object_id in target_offsets
        ]
        if self._rng.random() < self.false_positive_prob:
            plant, target = self._false_positive(scene, len(plants))
            plants.append(plant)
            targets.append(target)
        return SimScene(
            scene_id=scene.scene_id,
            grid_rows=scene.grid_rows,
            grid_cols=scene.grid_cols,
            cell_size_mm=list(scene.cell_size_mm),
            plants=plants,
            targets=targets,
            robot_position_mm=list(scene.robot_position_mm),
            step_index=scene.step_index,
            metadata={
                **scene.metadata,
                "detection_noise": {
                    "bbox_center_sigma_mm": self.bbox_center_sigma_mm,
                    "classification_error_prob": self.classification_error_prob,
                    "missed_detection_prob": self.missed_detection_prob,
                    "false_positive_prob": self.false_positive_prob,
                },
            },
        )

    def _noisy_target(self, target: SimTarget, offset: list[float], class_name: str) -> SimTarget:
        target_type = target.target_type
        uncertainty = target.uncertainty_radius_mm
        if class_name == "unknown_plant":
            target_type = "human_review_required"
            uncertainty = max(uncertainty, 2.5)
        return SimTarget(
            target_id=target.target_id,
            object_id=target.object_id,
            target_type=target_type,
            point_mm=[target.point_mm[0] + offset[0], target.point_mm[1] + offset[1]],
            uncertainty_radius_mm=uncertainty,
            min_distance_to_keep_mm=target.min_distance_to_keep_mm,
            processed=target.processed,
        )

    def _false_positive(self, scene: SimScene, index: int) -> tuple[SimPlant, SimTarget]:
        row = self._rng.randrange(scene.grid_rows)
        col = self._rng.randrange(scene.grid_cols)
        cell_w, cell_h = scene.cell_size_mm
        point = [
            col * cell_w + self._rng.uniform(cell_w * 0.2, cell_w * 0.8),
            row * cell_h + self._rng.uniform(cell_h * 0.2, cell_h * 0.8),
        ]
        object_id = f"noise_fp_{index:06d}"
        plant = SimPlant(
            object_id=object_id,
            row=row,
            col=col,
            class_name="unknown_plant",
            position_mm=point,
            confidence=self._rng.uniform(0.1, 0.45),
            attributes=["false_positive_noise"],
        )
        target = SimTarget(
            target_id=f"noise_target_{index:06d}",
            object_id=object_id,
            target_type="human_review_required",
            point_mm=point,
            uncertainty_radius_mm=2.5,
        )
        return plant, target
