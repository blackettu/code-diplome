from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from pathlib import Path
from typing import Any

from seedling_core.schemas import DetectionObject, DetectionResultV1, InferenceContext, ModelMetadata

ImageInput = str | Path | Any


class DetectorAdapter(ABC):
    @abstractmethod
    def load(self, model_uri: str, device: str | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, image: ImageInput, context: InferenceContext | None = None) -> DetectionResultV1:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> ModelMetadata:
        raise NotImplementedError


def metadata_for_context(metadata: ModelMetadata, context: InferenceContext | None = None) -> ModelMetadata:
    if context is None or context.ontology_version is None:
        return metadata
    return replace(metadata, ontology_version=context.ontology_version)


class MockDetector(DetectorAdapter):
    def __init__(self, detections: list[DetectionObject] | None = None, image_size_px: list[int] | None = None):
        self._detections = detections or []
        self._image_size_px = image_size_px or [1, 1]
        self._metadata = ModelMetadata(model_id="mock_detector", backend="mock")

    def load(self, model_uri: str, device: str | None = None) -> None:
        self._metadata.model_uri = model_uri

    def predict(self, image: ImageInput, context: InferenceContext | None = None) -> DetectionResultV1:
        context = context or InferenceContext()
        image_ref = str(context.image_id or _image_name(image))
        image_size = context.image_size_px or self._image_size_px
        return DetectionResultV1(
            image_ref=image_ref,
            image_size_px=image_size,
            detections=list(self._detections),
            model_metadata=metadata_for_context(self.metadata(), context),
        )

    def metadata(self) -> ModelMetadata:
        return self._metadata


def _image_name(image: ImageInput) -> str:
    if isinstance(image, (str, Path)):
        return Path(image).name
    return "in_memory_image"
