from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from seedling_core.schemas import CellState, DetectionObject, TrayState, cell_id

from .grid import cell_index, cell_index_for_grid_cells, generate_grid_cells, generate_grid_polygons


@dataclass(frozen=True)
class CellStateBuilder:
    crop_class_names: tuple[str, ...] = ("crop_seedling", "seedlings", "seedling")
    weed_class_names: tuple[str, ...] = ("weed",)
    unknown_class_names: tuple[str, ...] = ("unknown_plant", "unknown")
    review_attribute_names: tuple[str, ...] = (
        "low_confidence",
        "high_class_entropy",
        "tiny",
        "large_bbox",
        "touches_image_edge",
    )

    def build(self, tray: TrayState, detections: Iterable[DetectionObject]) -> list[CellState]:
        if tray.corners_px is not None:
            grid_cells = generate_grid_polygons(tray.corners_px, tray.grid_rows, tray.grid_cols)
            assignment_mode = "polygon"
        elif tray.bbox_xyxy_px is not None:
            grid_cells = generate_grid_cells(tray.bbox_xyxy_px, tray.grid_rows, tray.grid_cols)
            assignment_mode = "bbox"
        else:
            raise ValueError("TrayState requires bbox_xyxy_px or corners_px")

        objects_by_cell: dict[tuple[int, int], list[DetectionObject]] = {
            (cell.row, cell.col): [] for cell in grid_cells
        }
        for detection in detections:
            point = tuple(detection.center_px or [0.0, 0.0])
            if assignment_mode == "polygon":
                idx = cell_index_for_grid_cells(grid_cells, point)
            else:
                assert tray.bbox_xyxy_px is not None
                idx = cell_index(tray.bbox_xyxy_px, point, tray.grid_rows, tray.grid_cols)
            if idx is not None:
                objects_by_cell[idx].append(detection)

        states = []
        for grid_cell in grid_cells:
            objects = objects_by_cell[(grid_cell.row, grid_cell.col)]
            state, review, risk_flags = self._state_for_objects(objects)
            keep_object_id, removal_candidate_ids = self._largest_bbox_keep_and_remove(objects)
            states.append(
                CellState(
                    cell_id=cell_id(grid_cell.row, grid_cell.col),
                    row=grid_cell.row,
                    col=grid_cell.col,
                    polygon_px=grid_cell.polygon_px,
                    state=state,
                    state_confidence=_min_confidence(objects),
                    object_ids=[item.object_id for item in objects],
                    keep_object_id=keep_object_id,
                    removal_candidate_ids=removal_candidate_ids,
                    human_review_required=review,
                    risk_flags=risk_flags,
                )
            )
        return states

    def _state_for_objects(self, objects: list[DetectionObject]) -> tuple[str, bool, list[str]]:
        if not objects:
            return "empty", False, []
        crops = [item for item in objects if item.class_name in self.crop_class_names]
        weeds = [item for item in objects if item.class_name in self.weed_class_names]
        unknowns = [item for item in objects if item.class_name in self.unknown_class_names]
        risk_flags: list[str] = []
        if unknowns:
            risk_flags.append("unknown_plant_present")
        risk_flags.extend(_review_attributes(objects, self.review_attribute_names))
        review_required = bool(unknowns or risk_flags)
        if unknowns and not crops and not weeds:
            return "unknown", True, risk_flags
        if crops and weeds:
            return "crop_and_weed", True, risk_flags
        if weeds and not crops:
            return "weed_only", review_required, risk_flags
        if len(crops) == 1 and not unknowns:
            return "single_crop", review_required, risk_flags
        if len(crops) > 1 and not unknowns:
            return "multiple_crop", review_required, risk_flags
        return "ambiguous", True, risk_flags or ["ambiguous_cell"]

    def _largest_bbox_keep_and_remove(self, objects: list[DetectionObject]) -> tuple[str | None, list[str]]:
        crops = [item for item in objects if item.class_name in self.crop_class_names]
        if len(crops) <= 1:
            return (crops[0].object_id if crops else None), []
        ordered = sorted(crops, key=lambda item: item.area_px2 or 0.0, reverse=True)
        return ordered[0].object_id, [item.object_id for item in ordered[1:]]


def _min_confidence(objects: list[DetectionObject]) -> float:
    if not objects:
        return 1.0
    return min(item.confidence for item in objects)


def _review_attributes(objects: list[DetectionObject], review_attribute_names: tuple[str, ...]) -> list[str]:
    review_attribute_set = set(review_attribute_names)
    flags = {
        attribute
        for item in objects
        for attribute in item.attributes
        if attribute in review_attribute_set
    }
    return sorted(flags)
