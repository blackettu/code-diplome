from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import save_run_snapshot, write_json
from .grid import Detection, assign_to_cells, bbox_iou, choose_removal_targets, count_matrix
from .image_io import register_heif_if_available
from .yolo import list_images

register_heif_if_available()


def predict_from_config(config: dict[str, Any]) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is required for prediction. Install requirements.txt.") from exc

    prediction = config.get("prediction", {})
    model_path = prediction["model"]
    images_dir = Path(prediction["images"])
    output_dir = Path(prediction.get("output_dir", "runs/predict"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_run_snapshot(output_dir, config, "predict")

    model = YOLO(model_path)
    images = list_images(images_dir)
    results: list[dict[str, Any]] = []
    for image_path in images:
        image = _load_rgb(image_path)
        predict_args: dict[str, Any] = {
            "conf": float(prediction.get("conf", 0.25)),
            "iou": float(prediction.get("iou", 0.7)),
            "imgsz": int(prediction.get("imgsz", 640)),
            "verbose": False,
        }
        if prediction.get("device") is not None:
            predict_args["device"] = prediction["device"]
        pred = model.predict(image, **predict_args)[0]
        detections = _ultralytics_to_detections(pred)
        image_result = analyze_detections(
            image_path=image_path,
            image_shape=image.shape,
            detections=detections,
            grid_rows=int(prediction.get("grid_rows", 11)),
            grid_cols=int(prediction.get("grid_cols", 11)),
            container_class=int(prediction.get("container_class", 0)),
            seedling_class=int(prediction.get("seedling_class", 1)),
            min_container_area=float(prediction.get("min_container_area", 10000)),
            merge_distance=float(prediction.get("merge_distance", 50)),
        )
        results.append(image_result)

    output = {"images_dir": str(images_dir.resolve()), "images": results}
    write_json(output_dir / "predictions.json", output)
    return output


def analyze_detections(
    image_path: Path,
    image_shape: tuple[int, ...],
    detections: list[Detection],
    grid_rows: int,
    grid_cols: int,
    container_class: int,
    seedling_class: int,
    min_container_area: float,
    merge_distance: float,
) -> dict[str, Any]:
    containers = [item for item in detections if item.class_id == container_class]
    seedlings = [item for item in detections if item.class_id == seedling_class]
    containers = filter_and_merge_containers(containers, min_container_area, merge_distance)

    analyzed_containers: list[dict[str, Any]] = []
    for index, container in enumerate(containers, 1):
        cells = assign_to_cells(container.box, seedlings, grid_rows, grid_cols)
        analyzed_containers.append(
            {
                "index": index,
                "box": container.box,
                "confidence": container.confidence,
                "matrix": count_matrix(cells),
                "removal_targets": choose_removal_targets(cells),
            }
        )

    height, width = image_shape[:2]
    return {
        "image": image_path.name,
        "path": str(image_path.resolve()),
        "width": int(width),
        "height": int(height),
        "detections": [_detection_to_dict(item) for item in detections],
        "containers": [_detection_to_dict(item) for item in containers],
        "seedlings": [_detection_to_dict(item) for item in seedlings],
        "container_analysis": analyzed_containers,
    }


def filter_and_merge_containers(
    containers: list[Detection],
    min_area: float,
    merge_distance: float,
) -> list[Detection]:
    filtered: list[Detection] = []
    small: list[Detection] = []
    for container in containers:
        if container.area < min_area:
            small.append(container)
        else:
            filtered.append(container)

    for small_container in small:
        nearest_index: int | None = None
        nearest_distance = float("inf")
        sx, sy = small_container.center
        for index, filtered_container in enumerate(filtered):
            fx, fy = filtered_container.center
            distance = float(np.hypot(sx - fx, sy - fy))
            if distance < nearest_distance and distance < merge_distance:
                nearest_index = index
                nearest_distance = distance
        if nearest_index is None:
            filtered.append(small_container)
            continue
        target = filtered[nearest_index]
        merged_box = _merge_boxes(target.box, small_container.box)
        filtered[nearest_index] = Detection(
            box=merged_box,
            class_id=target.class_id,
            confidence=max(target.confidence, small_container.confidence),
            name=target.name,
        )
    return filtered


def _ultralytics_to_detections(result: Any) -> list[Detection]:
    detections: list[Detection] = []
    names = getattr(result, "names", {})
    for box in result.boxes:
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0]]
        class_id = int(box.cls[0])
        detections.append(
            Detection(
                box=[x1, y1, x2, y2],
                class_id=class_id,
                confidence=float(box.conf[0]),
                name=names.get(class_id) if isinstance(names, dict) else None,
            )
        )
    return detections


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def _merge_boxes(a: list[float], b: list[float]) -> list[float]:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def _detection_to_dict(item: Detection) -> dict[str, Any]:
    return {
        "class_id": item.class_id,
        "name": item.name,
        "confidence": item.confidence,
        "box": item.box,
        "center": list(item.center),
        "area": item.area,
    }


def match_containers(
    gt_containers: list[Detection],
    predicted_containers: list[Detection],
    min_iou: float,
) -> list[tuple[Detection, Detection | None, float]]:
    matches: list[tuple[Detection, Detection | None, float]] = []
    used: set[int] = set()
    for gt in gt_containers:
        best_index = None
        best_iou = 0.0
        for index, predicted in enumerate(predicted_containers):
            if index in used:
                continue
            iou = bbox_iou(gt.box, predicted.box)
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_index is not None and best_iou >= min_iou:
            used.add(best_index)
            matches.append((gt, predicted_containers[best_index], best_iou))
        else:
            matches.append((gt, None, best_iou))
    return matches
