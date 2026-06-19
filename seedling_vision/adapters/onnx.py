from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from seedling_core.schemas import DetectionObject, DetectionResultV1, InferenceContext, ModelMetadata

from .base import DetectorAdapter, ImageInput, metadata_for_context


class ONNXDetector(DetectorAdapter):
    """Generic ONNX detector adapter with a small, explicit output contract."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._session: Any | None = None
        self._input_name: str | None = None
        self._metadata = ModelMetadata(
            model_id=str(self.config.get("model_id", "onnx_detector_v0")),
            backend="onnxruntime",
            class_names=[str(value) for value in self.config.get("class_names", [])],
            output_schema="DetectionResultV1",
        )

    def load(self, model_uri: str, device: str | None = None) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for ONNXDetector") from exc

        providers = _providers(device)
        self._session = ort.InferenceSession(str(model_uri), providers=providers)
        inputs = self._session.get_inputs()
        if not inputs:
            raise ValueError("ONNX model has no inputs")
        self._input_name = inputs[0].name
        self._metadata.model_uri = str(model_uri)

    def predict(self, image: ImageInput, context: InferenceContext | None = None) -> DetectionResultV1:
        if self._session is None or self._input_name is None:
            raise RuntimeError("ONNXDetector.load() must be called before predict()")
        context = context or InferenceContext()
        image_rgb = _load_rgb(image)
        input_tensor = _preprocess(image_rgb, self.config)
        outputs = self._session.run(None, {self._input_name: input_tensor})
        height, width = image_rgb.shape[:2]
        detections = parse_onnx_detections(
            outputs,
            image_size_px=[width, height],
            class_names=self._metadata.class_names,
            confidence_threshold=float(self.config.get("confidence_threshold", 0.25)),
            output_format=str(self.config.get("output_format", "xyxy_conf_class")),
            source_model=self._metadata.model_id,
        )
        return DetectionResultV1(
            image_ref=str(context.image_id or _image_name(image)),
            image_size_px=[width, height],
            detections=detections,
            model_metadata=metadata_for_context(self.metadata(), context),
        )

    def metadata(self) -> ModelMetadata:
        return self._metadata


def parse_onnx_detections(
    outputs: list[Any] | tuple[Any, ...] | np.ndarray,
    image_size_px: list[int],
    class_names: list[str] | None = None,
    confidence_threshold: float = 0.25,
    output_format: str = "xyxy_conf_class",
    source_model: str = "onnx_detector_v0",
) -> list[DetectionObject]:
    rows = _rows(outputs, output_format)
    detections: list[DetectionObject] = []
    for index, row in enumerate(rows):
        if row.shape[0] < _min_columns(output_format):
            continue
        confidence, class_id = _confidence_and_class(row, output_format)
        if confidence < confidence_threshold:
            continue
        bbox = _bbox_from_row(row, output_format)
        detections.append(
            DetectionObject(
                object_id=f"onnx_{index:06d}",
                class_name=_class_name(class_id, class_names or []),
                class_id=class_id,
                confidence=confidence,
                bbox_xyxy_px=_clip_bbox(bbox, image_size_px),
                source_model=source_model,
                uncertainty=_uncertainty_from_row(row, output_format),
            )
        )
    return detections


def _preprocess(image_rgb: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    input_size = config.get("input_size")
    image = Image.fromarray(image_rgb)
    if input_size:
        image = image.resize((int(input_size[0]), int(input_size[1])))
    array = np.asarray(image, dtype=np.float32)
    if bool(config.get("normalize", True)):
        array = array / 255.0
    if str(config.get("layout", "nchw")).lower() == "nchw":
        array = np.transpose(array, (2, 0, 1))
    return np.expand_dims(array, axis=0)


def _rows(outputs: list[Any] | tuple[Any, ...] | np.ndarray, output_format: str) -> np.ndarray:
    output = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    array = np.asarray(output, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 2:
        min_columns = _min_columns(output_format)
        if array.shape[1] < min_columns <= array.shape[0]:
            array = array.T
        elif min_columns <= array.shape[0] <= 256 and array.shape[1] > array.shape[0]:
            array = array.T
    if array.ndim != 2:
        raise ValueError(f"Unsupported ONNX output shape: {array.shape}")
    return array


def _min_columns(output_format: str) -> int:
    if output_format in {"xyxy_conf_class", "cxcywh_conf_class"}:
        return 6
    if output_format in {"xyxy_scores", "cxcywh_scores"}:
        return 5
    if output_format in {"xyxy_objectness_scores", "cxcywh_objectness_scores"}:
        return 6
    if output_format == "auto":
        return 5
    raise ValueError(f"Unsupported ONNX output_format: {output_format}")


def _confidence_and_class(row: np.ndarray, output_format: str) -> tuple[float, int]:
    if output_format in {"xyxy_conf_class", "cxcywh_conf_class"}:
        return float(row[4]), int(round(float(row[5])))
    if output_format in {"xyxy_scores", "cxcywh_scores"}:
        scores = row[4:]
        class_id = int(np.argmax(scores))
        return float(scores[class_id]), class_id
    if output_format in {"xyxy_objectness_scores", "cxcywh_objectness_scores"}:
        objectness = float(row[4])
        scores = row[5:]
        class_id = int(np.argmax(scores))
        return objectness * float(scores[class_id]), class_id
    if output_format == "auto":
        if row.shape[0] == 6:
            return float(row[4]), int(round(float(row[5])))
        scores = row[4:]
        class_id = int(np.argmax(scores))
        return float(scores[class_id]), class_id
    raise ValueError(f"Unsupported ONNX output_format: {output_format}")


def _uncertainty_from_row(row: np.ndarray, output_format: str) -> dict[str, float]:
    if output_format in {"xyxy_scores", "cxcywh_scores"}:
        return _class_score_uncertainty(row[4:])
    if output_format in {"xyxy_objectness_scores", "cxcywh_objectness_scores"}:
        uncertainty = {"objectness": float(row[4])}
        uncertainty.update(_class_score_uncertainty(row[5:]))
        return uncertainty
    if output_format == "auto" and row.shape[0] > 6:
        return _class_score_uncertainty(row[4:])
    return {}


def _class_score_uncertainty(scores: np.ndarray) -> dict[str, float]:
    return {f"class_score_{index}": float(score) for index, score in enumerate(scores)}


def _bbox_from_row(row: np.ndarray, output_format: str) -> list[float]:
    if output_format in {"xyxy_conf_class", "xyxy_scores", "xyxy_objectness_scores"}:
        return [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
    if output_format in {"cxcywh_conf_class", "cxcywh_scores", "cxcywh_objectness_scores"}:
        cx, cy, width, height = [float(value) for value in row[:4]]
        return [cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0]
    if output_format == "auto":
        return [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
    raise ValueError(f"Unsupported ONNX output_format: {output_format}")


def _clip_bbox(bbox: list[float], image_size_px: list[int]) -> list[float]:
    width, height = image_size_px
    x1 = min(max(float(bbox[0]), 0.0), float(width))
    y1 = min(max(float(bbox[1]), 0.0), float(height))
    x2 = min(max(float(bbox[2]), 0.0), float(width))
    y2 = min(max(float(bbox[3]), 0.0), float(height))
    return [x1, y1, max(x1, x2), max(y1, y2)]


def _load_rgb(image: ImageInput) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    with Image.open(image) as opened:
        return np.asarray(opened.convert("RGB"))


def _image_name(image: ImageInput) -> str:
    if isinstance(image, (str, Path)):
        return Path(image).name
    return "in_memory_image"


def _class_name(class_id: int, class_names: list[str]) -> str:
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    if class_id == 0:
        return "container"
    if class_id == 1:
        return "crop_seedling"
    return str(class_id)


def _providers(device: str | None) -> list[str] | None:
    if device and device.lower().startswith("cuda"):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device and device.lower() == "cpu":
        return ["CPUExecutionProvider"]
    return None
