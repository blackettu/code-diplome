from __future__ import annotations

from dataclasses import dataclass, replace
import math

from seedling_core.schemas import DetectionObject, DetectionResultV1


@dataclass(frozen=True)
class UncertaintyConfig:
    low_confidence_threshold: float = 0.5
    tiny_area_px2: float = 20.0
    max_area_fraction: float = 0.5
    edge_margin_px: float = 2.0
    high_entropy_threshold: float | None = 0.8
    class_score_prefix: str = "class_score_"


def annotate_result_uncertainty(
    result: DetectionResultV1,
    config: UncertaintyConfig | None = None,
) -> DetectionResultV1:
    config = config or UncertaintyConfig()
    detections = [
        annotate_detection_uncertainty(detection, result.image_size_px, config)
        for detection in result.detections
    ]
    return replace(result, detections=detections)


def annotate_detection_uncertainty(
    detection: DetectionObject,
    image_size_px: list[int],
    config: UncertaintyConfig | None = None,
) -> DetectionObject:
    config = config or UncertaintyConfig()
    width, height = image_size_px
    image_area = max(1.0, float(width * height))
    area_fraction = float(detection.area_px2 or 0.0) / image_area
    flags = list(detection.attributes)
    uncertainty = dict(detection.uncertainty)

    if detection.confidence < config.low_confidence_threshold:
        flags.append("low_confidence")
    if (detection.area_px2 or 0.0) < config.tiny_area_px2:
        flags.append("tiny")
    if area_fraction > config.max_area_fraction:
        flags.append("large_bbox")
    if _touches_edge(detection.bbox_xyxy_px, image_size_px, config.edge_margin_px):
        flags.append("touches_image_edge")

    uncertainty.update(
        {
            "confidence_margin": max(0.0, detection.confidence - config.low_confidence_threshold),
            "area_fraction": area_fraction,
        }
    )
    class_scores = _class_scores(uncertainty, config.class_score_prefix)
    entropy = _normalized_entropy(class_scores)
    if entropy is not None:
        uncertainty["class_entropy_normalized"] = entropy
        uncertainty["class_probability_margin"] = _probability_margin(class_scores)
        if config.high_entropy_threshold is not None and entropy >= config.high_entropy_threshold:
            flags.append("high_class_entropy")
    return replace(detection, attributes=sorted(set(flags)), uncertainty=uncertainty)


def _touches_edge(bbox: list[float], image_size_px: list[int], margin: float) -> bool:
    x1, y1, x2, y2 = bbox
    width, height = image_size_px
    return x1 <= margin or y1 <= margin or x2 >= width - margin or y2 >= height - margin


def _class_scores(uncertainty: dict[str, float], prefix: str) -> list[float]:
    values = [
        (key, float(value))
        for key, value in uncertainty.items()
        if key.startswith(prefix) and math.isfinite(float(value)) and float(value) >= 0.0
    ]
    values.sort(key=lambda item: _score_key(item[0], prefix))
    return [value for _, value in values]


def _score_key(key: str, prefix: str) -> tuple[int, str]:
    suffix = key[len(prefix) :]
    if suffix.isdigit():
        return int(suffix), suffix
    return 10_000, suffix


def _normalized_entropy(scores: list[float]) -> float | None:
    if len(scores) < 2:
        return None
    total = sum(scores)
    if total <= 0.0:
        return None
    probabilities = [score / total for score in scores if score > 0.0]
    if not probabilities:
        return None
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    return entropy / math.log(len(scores))


def _probability_margin(scores: list[float]) -> float:
    total = sum(scores)
    if total <= 0.0:
        return 0.0
    probabilities = sorted((score / total for score in scores), reverse=True)
    if len(probabilities) < 2:
        return probabilities[0]
    return probabilities[0] - probabilities[1]
