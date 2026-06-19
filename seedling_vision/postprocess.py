from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from seedling_core.schemas import DetectionObject


def filter_and_merge_containers(
    detections: list[DetectionObject],
    container_class_names: tuple[str, ...] = ("container",),
    container_class_ids: tuple[int, ...] = (0,),
    min_area_px2: float = 10000.0,
    merge_distance_px: float = 50.0,
) -> list[DetectionObject]:
    containers = [
        item
        for item in detections
        if item.class_name in container_class_names or item.class_id in container_class_ids
    ]
    large: list[DetectionObject] = []
    small: list[DetectionObject] = []
    for container in containers:
        if (container.area_px2 or 0.0) < min_area_px2:
            small.append(container)
        else:
            large.append(container)

    merged = list(large)
    for small_container in small:
        nearest_index = _nearest_container_index(small_container, merged, merge_distance_px)
        if nearest_index is None:
            merged.append(small_container)
            continue
        target = merged[nearest_index]
        merged[nearest_index] = merge_container_detections(target, small_container)
    return merged


def merge_container_detections(left: DetectionObject, right: DetectionObject) -> DetectionObject:
    bbox = [
        min(left.bbox_xyxy_px[0], right.bbox_xyxy_px[0]),
        min(left.bbox_xyxy_px[1], right.bbox_xyxy_px[1]),
        max(left.bbox_xyxy_px[2], right.bbox_xyxy_px[2]),
        max(left.bbox_xyxy_px[3], right.bbox_xyxy_px[3]),
    ]
    attributes = sorted(set([*left.attributes, *right.attributes, "merged_container"]))
    metadata = dict(left.uncertainty)
    metadata.update({f"merged_{key}": value for key, value in right.uncertainty.items()})
    return replace(
        left,
        object_id=f"{left.object_id}+{right.object_id}",
        confidence=max(left.confidence, right.confidence),
        bbox_xyxy_px=bbox,
        center_px=None,
        area_px2=None,
        uncertainty=metadata,
        attributes=attributes,
    )


def match_containers_by_iou(
    gt_containers: list[DetectionObject],
    predicted_containers: list[DetectionObject],
    min_iou: float = 0.5,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    used: set[int] = set()
    for gt in gt_containers:
        best_index: int | None = None
        best_iou = 0.0
        for index, predicted in enumerate(predicted_containers):
            if index in used:
                continue
            iou = bbox_iou(gt.bbox_xyxy_px, predicted.bbox_xyxy_px)
            if iou > best_iou:
                best_index = index
                best_iou = iou
        if best_index is not None and best_iou >= min_iou:
            used.add(best_index)
            matches.append(
                {
                    "gt_object_id": gt.object_id,
                    "pred_object_id": predicted_containers[best_index].object_id,
                    "iou": best_iou,
                    "matched": True,
                }
            )
        else:
            matches.append({"gt_object_id": gt.object_id, "pred_object_id": None, "iou": best_iou, "matched": False})
    return matches


def container_matching_summary(
    gt_containers: list[DetectionObject],
    predicted_containers: list[DetectionObject],
    min_iou: float = 0.5,
) -> dict[str, Any]:
    matches = match_containers_by_iou(gt_containers, predicted_containers, min_iou=min_iou)
    best_ious = [float(item["iou"]) for item in matches]
    return {
        "min_iou": min_iou,
        "total_gt_containers": len(gt_containers),
        "pred_containers": len(predicted_containers),
        "matched_containers": sum(1 for item in matches if item["matched"]),
        "container_recall": (
            sum(1 for item in matches if item["matched"]) / len(gt_containers)
            if gt_containers
            else None
        ),
        "best_iou_summary": _summary(best_ious),
        "best_iou_counts": {
            "gte_0_25": sum(1 for value in best_ious if value >= 0.25),
            "gte_0_50": sum(1 for value in best_ious if value >= 0.50),
            f"gte_{min_iou:.2f}": sum(1 for value in best_ious if value >= min_iou),
        },
        "unmatched_samples": [item for item in matches if not item["matched"]][:10],
        "matches": matches,
    }


def bbox_iou(left: list[float], right: list[float]) -> float:
    ax1, ay1, ax2, ay2 = left
    bx1, by1, bx2, by2 = right
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0.0 else 0.0


def _nearest_container_index(
    small_container: DetectionObject,
    large_containers: list[DetectionObject],
    merge_distance_px: float,
) -> int | None:
    nearest_index: int | None = None
    nearest_distance = float("inf")
    for index, candidate in enumerate(large_containers):
        distance = _distance(small_container.center_px or [0, 0], candidate.center_px or [0, 0])
        if distance < nearest_distance and distance < merge_distance_px:
            nearest_index = index
            nearest_distance = distance
    return nearest_index


def _distance(left: list[float], right: list[float]) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    ordered = sorted(values)
    return {"count": len(values), "min": ordered[0], "mean": sum(values) / len(values), "max": ordered[-1]}
