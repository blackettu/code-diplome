from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .config import save_run_snapshot, write_json
from .grid import Detection
from .image_io import register_heif_if_available
from .predict import analyze_detections
from .yolo import label_path_for, list_images, read_labels

register_heif_if_available()


def green_components_baseline_from_config(config: dict[str, Any]) -> dict[str, Any]:
    baseline = config.get("baseline", {})
    images_dir = Path(baseline["images"])
    output_dir = Path(baseline.get("output_dir", "runs/baseline_green"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_run_snapshot(output_dir, config, "baseline-green")

    labels_dir = Path(baseline["labels"]) if baseline.get("labels") else None
    results = []
    for image_path in list_images(images_dir):
        image = _load_rgb(image_path)
        detections = _detect_green_components(image, baseline)
        if bool(baseline.get("use_known_containers", True)) and labels_dir:
            detections.extend(
                _known_containers(
                    label_path_for(image_path, labels_dir),
                    image.shape[1],
                    image.shape[0],
                    int(baseline.get("container_class", 0)),
                )
            )
        results.append(
            analyze_detections(
                image_path=image_path,
                image_shape=image.shape,
                detections=detections,
                grid_rows=int(baseline.get("grid_rows", 11)),
                grid_cols=int(baseline.get("grid_cols", 11)),
                container_class=int(baseline.get("container_class", 0)),
                seedling_class=int(baseline.get("seedling_class", 1)),
                min_container_area=float(baseline.get("min_container_area", 10000)),
                merge_distance=float(baseline.get("merge_distance", 50)),
            )
        )

    output = {"images_dir": str(images_dir.resolve()), "images": results}
    write_json(output_dir / "predictions.json", output)
    return output


def _detect_green_components(image_rgb: np.ndarray, config: dict[str, Any]) -> list[Detection]:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lower = np.array(
        [
            int(config.get("h_min", 25)),
            int(config.get("s_min", 35)),
            int(config.get("v_min", 30)),
        ],
        dtype=np.uint8,
    )
    upper = np.array(
        [
            int(config.get("h_max", 95)),
            int(config.get("s_max", 255)),
            int(config.get("v_max", 255)),
        ],
        dtype=np.uint8,
    )
    mask = cv2.inRange(hsv, lower, upper)
    kernel_size = int(config.get("morph_kernel", 3))
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = float(config.get("min_seedling_area", 8))
    max_area = float(config.get("max_seedling_area", 100000))
    seedling_class = int(config.get("seedling_class", 1))
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        detections.append(
            Detection(
                box=[float(x), float(y), float(x + width), float(y + height)],
                class_id=seedling_class,
                confidence=1.0,
                name="green_component",
            )
        )
    return detections


def _known_containers(
    label_path: Path,
    width: int,
    height: int,
    container_class: int,
) -> list[Detection]:
    detections = []
    for label in read_labels(label_path):
        if label.class_id != container_class:
            continue
        detections.append(
            Detection(
                box=label.to_xyxy(width, height),
                class_id=container_class,
                confidence=1.0,
                name="known_container",
            )
        )
    return detections


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))
