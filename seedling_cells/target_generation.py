from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from seedling_core.schemas import ActionTarget, CellState, DetectionObject


@dataclass(frozen=True)
class LargestBBoxTargetStrategy:
    decision_source: str = "largest_bbox_v0"
    crop_class_names: tuple[str, ...] = ("crop_seedling", "seedlings", "seedling")
    weed_class_names: tuple[str, ...] = ("weed",)

    def generate(
        self,
        cells: Iterable[CellState],
        detections_by_id: Mapping[str, DetectionObject],
    ) -> list[ActionTarget]:
        targets: list[ActionTarget] = []
        for cell in cells:
            if cell.state in {"multiple_crop", "crop_and_weed"}:
                candidate_ids = cell.removal_candidate_ids or self._infer_removal_candidates(cell, detections_by_id)
                targets.extend(
                    self._targets_for_objects(cell, detections_by_id, candidate_ids, target_type="remove_extra_crop")
                )
            if cell.state in {"weed_only", "crop_and_weed"}:
                weed_ids = [
                    object_id
                    for object_id in cell.object_ids
                    if object_id in detections_by_id and detections_by_id[object_id].class_name in self.weed_class_names
                ]
                targets.extend(self._targets_for_objects(cell, detections_by_id, weed_ids, target_type="remove_weed"))
        return targets

    def _targets_for_objects(
        self,
        cell: CellState,
        detections_by_id: Mapping[str, DetectionObject],
        object_ids: Iterable[str],
        target_type: str,
    ) -> list[ActionTarget]:
        targets: list[ActionTarget] = []
        for object_id in object_ids:
            detection = detections_by_id.get(object_id)
            if detection is None:
                continue
            point_type, action_point_px = _action_point_for_detection(detection)
            targets.append(
                ActionTarget(
                    target_id=f"target_{cell.cell_id}_{object_id}",
                    cell_id=cell.cell_id,
                    object_id=object_id,
                    target_type=target_type,
                    action_point_px=action_point_px,
                    point_type=point_type,
                    decision_source=self.decision_source,
                    human_review_required=cell.human_review_required,
                    risk_score=_risk_score(cell, detection),
                )
            )
        return targets

    def _infer_removal_candidates(
        self,
        cell: CellState,
        detections_by_id: Mapping[str, DetectionObject],
    ) -> list[str]:
        crops = [
            detections_by_id[object_id]
            for object_id in cell.object_ids
            if object_id in detections_by_id and detections_by_id[object_id].class_name in self.crop_class_names
        ]
        if len(crops) <= 1:
            return []
        keep_id = cell.keep_object_id
        if keep_id is None:
            keep_id = max(crops, key=lambda item: item.area_px2 or 0.0).object_id
        return [item.object_id for item in crops if item.object_id != keep_id]


def generate_largest_bbox_targets(
    cells: Iterable[CellState],
    detections: Iterable[DetectionObject],
    decision_source: str = "largest_bbox_v0",
) -> list[ActionTarget]:
    detections_by_id = {item.object_id: item for item in detections}
    return LargestBBoxTargetStrategy(decision_source=decision_source).generate(cells, detections_by_id)


def _action_point_for_detection(detection: DetectionObject) -> tuple[str, list[float]]:
    stem_base = detection.keypoints_px.get("stem_base")
    if stem_base:
        return "stem_base", stem_base
    if detection.center_px is None:
        raise ValueError(f"Detection {detection.object_id} has no center_px")
    return "bbox_center", detection.center_px


def _risk_score(cell: CellState, detection: DetectionObject) -> float:
    score = 0.0
    if cell.human_review_required:
        score += 0.5
    if cell.risk_flags:
        score += 0.2
    if detection.confidence < 0.5:
        score += 0.2
    return min(score, 1.0)
