"""Detector adapter implementations."""

from .baseline_green import BaselineGreenDetector
from .base import DetectorAdapter, MockDetector
from .onnx import ONNXDetector
from .recorded_prediction import RecordedPredictionDetector
from .ultralytics_yolo import UltralyticsYOLODetector

__all__ = [
    "DetectorAdapter",
    "BaselineGreenDetector",
    "MockDetector",
    "ONNXDetector",
    "RecordedPredictionDetector",
    "UltralyticsYOLODetector",
]
