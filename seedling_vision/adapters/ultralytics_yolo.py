from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from seedling_core.schemas import DetectionObject, DetectionResultV1, InferenceContext, ModelMetadata

from .base import DetectorAdapter, ImageInput, metadata_for_context


class UltralyticsYOLODetector(DetectorAdapter):
    def __init__(self) -> None:
        self._model: Any | None = None
        self._model_uri: str | None = None
        self._device: str | None = None
        self._names: dict[int, str] = {}

    def load(self, model_uri: str, device: str | None = None) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is required for UltralyticsYOLODetector") from exc
        self._model = YOLO(model_uri)
        self._model_uri = model_uri
        self._device = device
        names = getattr(self._model, "names", {})
        self._names = {int(key): str(value) for key, value in names.items()} if isinstance(names, dict) else {}

    def predict(self, image: ImageInput, context: InferenceContext | None = None) -> DetectionResultV1:
        if self._model is None:
            raise RuntimeError("Detector must be loaded before predict()")
        context = context or InferenceContext()
        image_array = _load_rgb(image)
        predict_args: dict[str, Any] = {"verbose": False}
        if self._device is not None:
            predict_args["device"] = self._device
        result = self._model.predict(image_array, **predict_args)[0]
        names = getattr(result, "names", self._names)
        detections = []
        for index, box in enumerate(result.boxes):
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0]]
            class_id = int(box.cls[0])
            class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
            detections.append(
                DetectionObject(
                    object_id=f"obj_{index:06d}",
                    class_name=class_name,
                    class_id=class_id,
                    confidence=float(box.conf[0]),
                    bbox_xyxy_px=[x1, y1, x2, y2],
                    source_model=self.metadata().model_id,
                )
            )
        height, width = image_array.shape[:2]
        return DetectionResultV1(
            image_ref=str(context.image_id or _image_name(image)),
            image_size_px=[width, height],
            detections=detections,
            model_metadata=metadata_for_context(self.metadata(), context),
        )

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id=Path(self._model_uri or "ultralytics_yolo").stem,
            backend="ultralytics_yolo",
            model_uri=self._model_uri,
            class_names=[self._names[key] for key in sorted(self._names)],
        )


def _load_rgb(image: ImageInput) -> np.ndarray:
    if isinstance(image, np.ndarray):
        if image.ndim == 3:
            return image
        raise ValueError("numpy image input must be HxWxC")
    with Image.open(image) as opened:
        return np.array(opened.convert("RGB"))


def _image_name(image: ImageInput) -> str:
    if isinstance(image, (str, Path)):
        return Path(image).name
    return "in_memory_image"
