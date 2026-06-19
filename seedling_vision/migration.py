from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from seedling_cells import CellStateBuilder, generate_largest_bbox_targets
from seedling_core.schemas import DetectionObject, DetectionResultV1, ModelMetadata, SceneState, TrayState


def legacy_predictions_to_scenes(
    predictions_path: str | Path,
    dataset_version: str,
    ontology_version: str,
    grid_rows: int = 11,
    grid_cols: int = 11,
    output_path: str | Path | None = None,
) -> list[SceneState]:
    records = load_legacy_prediction_records(predictions_path)
    scenes = [
        legacy_prediction_to_scene(
            record,
            dataset_version=dataset_version,
            ontology_version=ontology_version,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
        )
        for record in records
    ]
    if output_path:
        write_scene_states(output_path, scenes)
    return scenes


def load_legacy_prediction_records(predictions_path: str | Path) -> list[dict[str, Any]]:
    path = Path(predictions_path)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    images = data.get("images")
    if not isinstance(images, list):
        raise ValueError("Legacy predictions must contain an `images` list")
    return [item for item in images if isinstance(item, dict)]


def legacy_prediction_to_detection_result(record: dict[str, Any]) -> DetectionResultV1:
    image_name = str(record.get("image") or record.get("path") or "unknown_image")
    width = int(record.get("width") or 1)
    height = int(record.get("height") or 1)
    metadata = ModelMetadata(
        model_id=str(record.get("model_id") or "legacy_predictions"),
        backend="legacy_predictions_json",
        output_schema="DetectionResultV1",
    )
    detections = [
        _legacy_detection_to_object(item, index, metadata.model_id)
        for index, item in enumerate(record.get("detections", []) or [])
        if isinstance(item, dict)
    ]
    return DetectionResultV1(
        image_ref=str(record.get("path") or image_name),
        image_size_px=[width, height],
        detections=detections,
        model_metadata=metadata,
        extra={
            "containers": record.get("containers", []),
            "seedlings": record.get("seedlings", []),
            "container_analysis": record.get("container_analysis", []),
        },
    )


def legacy_prediction_to_scene(
    record: dict[str, Any],
    dataset_version: str,
    ontology_version: str,
    grid_rows: int = 11,
    grid_cols: int = 11,
) -> SceneState:
    detection_result = legacy_prediction_to_detection_result(record)
    tray_bbox = _tray_bbox(record, detection_result)
    image_name = Path(detection_result.image_ref).name
    tray = TrayState(
        tray_id=Path(image_name).stem,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        bbox_xyxy_px=tray_bbox,
    )
    plant_detections = [item for item in detection_result.detections if item.class_name != "container" and item.class_id != 0]
    cells = CellStateBuilder().build(tray, plant_detections)
    targets = generate_largest_bbox_targets(cells, plant_detections)
    return SceneState(
        scene_id=f"legacy_{Path(image_name).stem}",
        image_ref=detection_result.image_ref,
        dataset_version=dataset_version,
        ontology_version=ontology_version,
        image_size_px=detection_result.image_size_px,
        tray=tray,
        detections=detection_result.detections,
        cells=cells,
        targets=targets,
    )


def write_scene_states(path: str | Path, scenes: list[SceneState]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".jsonl":
        with output.open("w", encoding="utf-8") as handle:
            for scene in scenes:
                handle.write(json.dumps(scene.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        return
    output.write_text(
        json.dumps({"scenes": [scene.to_dict() for scene in scenes]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _legacy_detection_to_object(item: dict[str, Any], index: int, source_model: str) -> DetectionObject:
    class_id = int(item.get("class_id", 0))
    class_name = str(item.get("name") or _class_name(class_id))
    return DetectionObject(
        object_id=str(item.get("object_id") or f"obj_{index:06d}"),
        class_name=class_name,
        class_id=class_id,
        confidence=float(item.get("confidence", 1.0)),
        bbox_xyxy_px=_box(item),
        center_px=[float(value) for value in item["center"]] if item.get("center") else None,
        area_px2=float(item["area"]) if item.get("area") is not None else None,
        source_model=source_model,
    )


def _tray_bbox(record: dict[str, Any], detection_result: DetectionResultV1) -> list[float]:
    for container in record.get("containers", []) or []:
        if isinstance(container, dict):
            return _box(container)
    for detection in detection_result.detections:
        if detection.class_id == 0 or detection.class_name == "container":
            return detection.bbox_xyxy_px
    width, height = detection_result.image_size_px
    return [0.0, 0.0, float(width), float(height)]


def _box(item: dict[str, Any]) -> list[float]:
    raw = item.get("box") or item.get("bbox_xyxy_px") or item.get("bbox")
    if not isinstance(raw, list | tuple) or len(raw) != 4:
        raise ValueError(f"Legacy detection has no [x1, y1, x2, y2] box: {item}")
    return [float(value) for value in raw]


def _class_name(class_id: int) -> str:
    if class_id == 0:
        return "container"
    if class_id == 1:
        return "crop_seedling"
    if class_id == 2:
        return "weed"
    if class_id == 3:
        return "unknown_plant"
    return str(class_id)
