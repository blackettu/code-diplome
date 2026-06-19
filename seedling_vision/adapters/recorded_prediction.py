from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from seedling_core.schemas import DetectionObject, DetectionResultV1, InferenceContext, ModelMetadata

from .base import DetectorAdapter, ImageInput, metadata_for_context


class RecordedPredictionDetector(DetectorAdapter):
    """Detector adapter backed by the existing seedling_experiments predictions.json format."""

    def __init__(self) -> None:
        self._predictions: dict[str, dict[str, Any]] = {}
        self._metadata = ModelMetadata(model_id="recorded_predictions", backend="recorded_predictions")

    def load(self, model_uri: str, device: str | None = None) -> None:
        path = Path(model_uri)
        raw = json.loads(path.read_text(encoding="utf-8"))
        images = raw.get("images", [])
        if not isinstance(images, list):
            raise ValueError("Recorded predictions must contain an `images` list")
        self._predictions = {str(item["image"]): item for item in images if isinstance(item, dict) and "image" in item}
        self._metadata.model_uri = str(path)

    def predict(self, image: ImageInput, context: InferenceContext | None = None) -> DetectionResultV1:
        context = context or InferenceContext()
        image_key = str(context.image_id or _image_name(image))
        prediction = self._predictions.get(image_key)
        if prediction is None:
            raise KeyError(f"No recorded prediction for image {image_key!r}")
        width = int(prediction.get("width") or (context.image_size_px or [0, 0])[0])
        height = int(prediction.get("height") or (context.image_size_px or [0, 0])[1])
        detections = [
            _record_to_detection(item, index, self._metadata.model_id)
            for index, item in enumerate(prediction.get("detections", []))
        ]
        return DetectionResultV1(
            image_ref=str(prediction.get("path") or prediction.get("image") or image_key),
            image_size_px=[width, height],
            detections=detections,
            model_metadata=metadata_for_context(self.metadata(), context),
            extra={
                "containers": prediction.get("containers", []),
                "seedlings": prediction.get("seedlings", []),
                "container_analysis": prediction.get("container_analysis", []),
            },
        )

    def metadata(self) -> ModelMetadata:
        return self._metadata


def _record_to_detection(item: dict[str, Any], index: int, source_model: str) -> DetectionObject:
    class_id = int(item["class_id"])
    class_name = str(item.get("name") or _class_name(class_id))
    return DetectionObject(
        object_id=str(item.get("object_id") or f"obj_{index:06d}"),
        class_name=class_name,
        class_id=class_id,
        confidence=float(item.get("confidence", 1.0)),
        bbox_xyxy_px=[float(value) for value in item["box"]],
        center_px=[float(value) for value in item["center"]] if item.get("center") else None,
        area_px2=float(item["area"]) if item.get("area") is not None else None,
        source_model=source_model,
    )


def _image_name(image: ImageInput) -> str:
    if isinstance(image, (str, Path)):
        return Path(image).name
    return "in_memory_image"


def _class_name(class_id: int) -> str:
    if class_id == 0:
        return "container"
    if class_id == 1:
        return "crop_seedling"
    return str(class_id)
