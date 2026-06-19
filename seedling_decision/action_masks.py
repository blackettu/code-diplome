from __future__ import annotations

from dataclasses import dataclass, field

from seedling_core.schemas import ActionTarget, CellState, SceneState


@dataclass(frozen=True)
class TargetMask:
    target_id: str
    valid: bool
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ActionMask:
    target_masks: list[TargetMask]

    def valid_target_ids(self) -> list[str]:
        return [item.target_id for item in self.target_masks if item.valid]

    def reasons_by_target(self) -> dict[str, list[str]]:
        return {item.target_id: item.reasons for item in self.target_masks}


@dataclass(frozen=True)
class ActionMaskBuilder:
    max_target_uncertainty_mm: float = 2.0
    min_distance_to_keep_mm: float = 5.0
    max_risk_score: float = 0.5
    require_calibration_valid: bool = True
    require_robot_homed: bool = False
    block_unknown_plant: bool = True
    block_ambiguous_cell: bool = True
    block_foreign_object: bool = True

    def build(self, scene: SceneState, processed_target_ids: set[str] | None = None) -> ActionMask:
        processed = processed_target_ids or set()
        cells = {cell.cell_id: cell for cell in scene.cells}
        target_masks = [
            self.evaluate_target(target, scene, cells.get(target.cell_id), processed)
            for target in scene.targets
        ]
        return ActionMask(target_masks=target_masks)

    def evaluate_target(
        self,
        target: ActionTarget,
        scene: SceneState,
        cell: CellState | None,
        processed_target_ids: set[str] | None = None,
    ) -> TargetMask:
        processed = processed_target_ids or set()
        reasons: list[str] = []
        if target.target_id in processed:
            reasons.append("already_processed")
        if target.human_review_required:
            reasons.append("human_review_required")
        if target.risk_score > self.max_risk_score:
            reasons.append("risk_score_too_high")
        if target.uncertainty_radius_mm is not None and target.uncertainty_radius_mm > self.max_target_uncertainty_mm:
            reasons.append("target_uncertainty_too_high")
        if target.min_distance_to_keep_mm is not None and target.min_distance_to_keep_mm < self.min_distance_to_keep_mm:
            reasons.append("too_close_to_keep_seedling")
        if target.forbidden_zone_ids:
            reasons.append("forbidden_zone_overlap")
        if _outside_tray(target, scene):
            reasons.append("target_outside_tray")
        if self.require_calibration_valid and not scene.safety.calibration_valid:
            reasons.append("calibration_required")
        if self.require_robot_homed and not scene.robot.homed:
            reasons.append("robot_not_homed")
        if cell is not None:
            if self.block_unknown_plant and (cell.state == "unknown" or "unknown_plant_present" in cell.risk_flags):
                reasons.append("unknown_plant_present")
            if self.block_ambiguous_cell and cell.state in {"ambiguous", "image_quality_insufficient", "invalid_geometry"}:
                reasons.append("ambiguous_or_invalid_cell")
            if self.block_foreign_object and (
                cell.state == "foreign_object_present" or "foreign_object_present" in cell.risk_flags
            ):
                reasons.append("foreign_object_present")
        return TargetMask(target_id=target.target_id, valid=not reasons, reasons=reasons)


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
