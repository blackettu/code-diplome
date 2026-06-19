from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from seedling_core.schemas import ActionTarget, DetectionObject, SceneState

from .schemas import SimPlant, SimScene, SimTarget


def sim_scene_from_scene_state(
    scene: SceneState,
    cell_size_mm: list[float] | None = None,
) -> SimScene:
    cell_size = cell_size_mm or _cell_size_from_scene(scene)
    object_cells = _object_cells(scene)
    plants = [
        _plant_from_detection(detection, scene, object_cells, cell_size)
        for detection in scene.detections
        if detection.class_name != "container" and detection.class_id != 0
    ]
    plant_ids = {plant.object_id for plant in plants}
    targets = [
        _target_from_action_target(target, plants_by_id={plant.object_id: plant for plant in plants})
        for target in scene.targets
        if target.object_id in plant_ids
    ]
    return SimScene(
        scene_id=f"image_backed_{scene.scene_id}",
        grid_rows=scene.tray.grid_rows,
        grid_cols=scene.tray.grid_cols,
        cell_size_mm=cell_size,
        plants=plants,
        targets=targets,
        robot_position_mm=scene.robot.position_mm,
        metadata={
            "source_scene_id": scene.scene_id,
            "image_ref": scene.image_ref,
            "dataset_version": scene.dataset_version,
            "ontology_version": scene.ontology_version,
            "builder": "image_backed_scene_v0",
        },
    )


def sim_scene_from_scene_state_file(
    scene_path: str | Path,
    output_path: str | Path | None = None,
    cell_size_mm: list[float] | None = None,
) -> SimScene:
    scene = _load_scene_state(scene_path)
    sim_scene = sim_scene_from_scene_state(scene, cell_size_mm=cell_size_mm)
    if output_path:
        sim_scene.to_json(output_path)
    return sim_scene


def _plant_from_detection(
    detection: DetectionObject,
    scene: SceneState,
    object_cells: dict[str, tuple[int, int]],
    cell_size_mm: list[float],
) -> SimPlant:
    row, col = object_cells.get(detection.object_id, _cell_for_point(scene, detection.center_px or [0, 0]))
    position = _point_mm(scene, detection.center_px or [0, 0], cell_size_mm)
    return SimPlant(
        object_id=detection.object_id,
        row=row,
        col=col,
        class_name=detection.class_name,
        position_mm=position,
        confidence=detection.confidence,
        radius_mm=max(1.0, min(cell_size_mm) * 0.07),
        attributes=list(detection.attributes),
    )


def _target_from_action_target(
    target: ActionTarget,
    plants_by_id: dict[str, SimPlant],
) -> SimTarget:
    plant = plants_by_id[target.object_id]
    point = target.action_point_mm or plant.position_mm
    return SimTarget(
        target_id=target.target_id,
        object_id=target.object_id,
        target_type=target.target_type,
        point_mm=[float(point[0]), float(point[1])],
        uncertainty_radius_mm=target.uncertainty_radius_mm or 1.0,
        min_distance_to_keep_mm=target.min_distance_to_keep_mm,
    )


def _object_cells(scene: SceneState) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for cell in scene.cells:
        for object_id in cell.object_ids:
            result[object_id] = (cell.row, cell.col)
    return result


def _point_mm(scene: SceneState, point_px: list[float], cell_size_mm: list[float]) -> list[float]:
    tray_bbox = scene.tray.bbox_xyxy_px or _bbox_from_corners(scene.tray.corners_px)
    x1, y1, x2, y2 = tray_bbox
    tray_w_mm = scene.tray.grid_cols * cell_size_mm[0]
    tray_h_mm = scene.tray.grid_rows * cell_size_mm[1]
    px_w = max(1e-9, x2 - x1)
    px_h = max(1e-9, y2 - y1)
    return [
        (float(point_px[0]) - x1) / px_w * tray_w_mm,
        (float(point_px[1]) - y1) / px_h * tray_h_mm,
    ]


def _cell_for_point(scene: SceneState, point_px: list[float]) -> tuple[int, int]:
    tray_bbox = scene.tray.bbox_xyxy_px or _bbox_from_corners(scene.tray.corners_px)
    x1, y1, x2, y2 = tray_bbox
    col = min(max(int((point_px[0] - x1) / max(1e-9, x2 - x1) * scene.tray.grid_cols), 0), scene.tray.grid_cols - 1)
    row = min(max(int((point_px[1] - y1) / max(1e-9, y2 - y1) * scene.tray.grid_rows), 0), scene.tray.grid_rows - 1)
    return row, col


def _cell_size_from_scene(scene: SceneState) -> list[float]:
    tray_size = scene.tray.grid_cols * 33.0, scene.tray.grid_rows * 33.0
    metadata: dict[str, Any] = scene.to_dict().get("metadata", {}) if hasattr(scene, "metadata") else {}
    raw = metadata.get("cell_size_mm") if isinstance(metadata, dict) else None
    if isinstance(raw, list | tuple) and len(raw) == 2:
        return [float(raw[0]), float(raw[1])]
    return [tray_size[0] / scene.tray.grid_cols, tray_size[1] / scene.tray.grid_rows]


def _bbox_from_corners(corners: list[list[float]] | None) -> list[float]:
    if not corners:
        raise ValueError("Scene tray requires bbox_xyxy_px or corners_px")
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return [min(xs), min(ys), max(xs), max(ys)]


def _load_scene_state(path: str | Path) -> SceneState:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(data, dict) and "scenes" in data:
        scenes = data.get("scenes")
        if not isinstance(scenes, list) or not scenes:
            raise ValueError(f"SceneState bundle contains no scenes: {path}")
        data = scenes[0]
    if not isinstance(data, dict):
        raise ValueError(f"SceneState file must contain a JSON object: {path}")
    return SceneState.from_dict(data)
