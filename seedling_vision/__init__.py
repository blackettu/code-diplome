"""Vision model adapter layer."""

from .batch import batch_predict
from .migration import (
    legacy_prediction_to_detection_result,
    legacy_prediction_to_scene,
    legacy_predictions_to_scenes,
    load_legacy_prediction_records,
    write_scene_states,
)
from .overlays import write_detection_overlay
from .postprocess import (
    bbox_iou,
    container_matching_summary,
    filter_and_merge_containers,
    match_containers_by_iou,
    merge_container_detections,
)
from .uncertainty import UncertaintyConfig, annotate_detection_uncertainty, annotate_result_uncertainty

__all__ = [
    "UncertaintyConfig",
    "annotate_detection_uncertainty",
    "annotate_result_uncertainty",
    "batch_predict",
    "bbox_iou",
    "container_matching_summary",
    "filter_and_merge_containers",
    "legacy_prediction_to_detection_result",
    "legacy_prediction_to_scene",
    "legacy_predictions_to_scenes",
    "load_legacy_prediction_records",
    "match_containers_by_iou",
    "merge_container_detections",
    "write_detection_overlay",
    "write_scene_states",
]
