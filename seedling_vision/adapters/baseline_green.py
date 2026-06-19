from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from seedling_core.schemas import DetectionObject, DetectionResultV1, InferenceContext, ModelMetadata

from .base import DetectorAdapter, ImageInput, metadata_for_context


class BaselineGreenDetector(DetectorAdapter):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._metadata = ModelMetadata(
            model_id="baseline_green_v0",
            backend="hsv_connected_components",
            class_names=["crop_seedling"],
            output_schema="DetectionResultV1",
        )

    def load(self, model_uri: str, device: str | None = None) -> None:
        self._metadata.model_uri = model_uri

    def predict(self, image: ImageInput, context: InferenceContext | None = None) -> DetectionResultV1:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for BaselineGreenDetector") from exc
        context = context or InferenceContext()
        image_rgb = _load_rgb(image)
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        lower = np.array(
            [
                int(self.config.get("h_min", 25)),
                int(self.config.get("s_min", 35)),
                int(self.config.get("v_min", 30)),
            ],
            dtype=np.uint8,
        )
        upper = np.array(
            [
                int(self.config.get("h_max", 95)),
                int(self.config.get("s_max", 255)),
                int(self.config.get("v_max", 255)),
            ],
            dtype=np.uint8,
        )
        mask = cv2.inRange(hsv, lower, upper)
        kernel_size = int(self.config.get("morph_kernel", 3))
        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = float(self.config.get("min_seedling_area", 8))
        max_area = float(self.config.get("max_seedling_area", 100000))
        detections: list[DetectionObject] = []
        for index, contour in enumerate(contours):
            area = float(cv2.contourArea(contour))
            if area < min_area or area > max_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            detections.append(
                DetectionObject(
                    object_id=f"green_{index:06d}",
                    class_name="crop_seedling",
                    class_id=int(self.config.get("seedling_class", 1)),
                    confidence=1.0,
                    bbox_xyxy_px=[float(x), float(y), float(x + width), float(y + height)],
                    source_model=self._metadata.model_id,
                    attributes=["baseline_green"],
                )
            )
        height, width = image_rgb.shape[:2]
        return DetectionResultV1(
            image_ref=str(context.image_id or _image_name(image)),
            image_size_px=[width, height],
            detections=detections,
            model_metadata=metadata_for_context(self.metadata(), context),
        )

    def metadata(self) -> ModelMetadata:
        return self._metadata


def _load_rgb(image: ImageInput) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    with Image.open(image) as opened:
        return np.array(opened.convert("RGB"))


def _image_name(image: ImageInput) -> str:
    if isinstance(image, (str, Path)):
        return Path(image).name
    return "in_memory_image"
