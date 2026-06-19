from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from seedling_cells.evaluation import cost_sensitive_metrics_from_legacy

from .config import register_run_artifacts, save_run_snapshot, write_json
from .dataset import is_suspected_augmented_name
from .grid import (
    Detection,
    assign_to_cells,
    center_distance,
    choose_removal_targets,
    count_matrix,
)
from .predict import match_containers
from .image_io import register_heif_if_available
from .yolo import label_path_for, list_images, read_labels

register_heif_if_available()


def evaluate_cells_from_config(
    config: dict[str, Any],
    command: str = "evaluate-cells",
    command_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evaluation = config.get("evaluation", {})
    output_dir = Path(evaluation.get("output_dir", "runs/evaluation"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_run_snapshot(output_dir, config, command, command_args=command_args)

    if evaluation.get("gt_scene") or evaluation.get("pred_scene"):
        metrics = _evaluate_scene_state_cells(evaluation, output_dir)
        register_run_artifacts(
            output_dir,
            command,
            config=config,
            command_args=command_args,
            input_paths=_evaluation_input_paths(evaluation),
            output_paths=[
                output_dir / "cell_metrics.json",
                output_dir / "cell_confusion_matrix.csv",
            ],
        )
        return metrics

    predictions = _load_predictions(evaluation["predictions"])
    images_dir, labels_dir = _dataset_dirs(evaluation["dataset"], evaluation.get("split"))
    rows = int(evaluation.get("grid_rows", 11))
    cols = int(evaluation.get("grid_cols", 11))
    container_class = int(evaluation.get("container_class", 0))
    seedling_class = int(evaluation.get("seedling_class", 1))
    container_iou = float(evaluation.get("container_iou", 0.5))
    target_distance = float(evaluation.get("target_match_distance_px", 25.0))
    use_gt_containers = bool(evaluation.get("use_ground_truth_containers", False))
    container_source = str(evaluation.get("container_prediction_source", "detections"))
    strict_predictions = bool(evaluation.get("strict_predictions", True))
    augmented_markers = evaluation.get("augmented_name_markers")
    calibration_artifact = _load_calibration_artifact(
        evaluation.get("calibration") or evaluation.get("calibration_artifact")
    )
    target_distance_mm = _optional_float(evaluation.get("target_match_distance_mm"))
    if target_distance_mm is not None and calibration_artifact is None:
        raise ValueError("evaluation.target_match_distance_mm requires evaluation.calibration")
    image_manifest_metadata = _load_image_manifest_metadata(
        evaluation.get("image_manifest") or evaluation.get("manifest"),
        dataset_root=evaluation["dataset"],
    )

    confusion = [[0 for _ in range(3)] for _ in range(3)]
    container_accuracies: list[float] = []
    multi_counts = {"tp": 0, "fp": 0, "fn": 0}
    target_counts = {"tp": 0, "fp": 0, "fn": 0}
    target_distances: list[float] = []
    target_distances_mm: list[float] = []
    best_container_ious: list[float] = []
    unmatched_samples: list[dict[str, Any]] = []
    matched_containers = 0
    total_gt_containers = 0
    image_summaries: list[dict[str, Any]] = []

    image_paths = list_images(images_dir)
    coverage = _prediction_coverage(image_paths, predictions)
    if strict_predictions and coverage["missing_predictions"]:
        preview = ", ".join(coverage["missing_predictions"][:10])
        raise ValueError(
            "predictions.json is missing images from evaluation dataset: "
            f"{preview}. Set evaluation.strict_predictions: false to override."
        )
    suspected_eval_images = [
        image.name
        for image in image_paths
        if is_suspected_augmented_name(image.name, augmented_markers)
    ]
    if (
        suspected_eval_images
        and bool(evaluation.get("reject_augmented_eval_images", True))
        and evaluation.get("split") in {"val", "test"}
    ):
        preview = ", ".join(suspected_eval_images[:10])
        suffix = "" if len(suspected_eval_images) <= 10 else f", ... ({len(suspected_eval_images)} total)"
        raise ValueError(
            "Evaluation split appears to contain augmented images. "
            "Use a clean val/test split or set evaluation.reject_augmented_eval_images: false "
            f"to override. Suspected files: {preview}{suffix}"
        )

    for image_path in image_paths:
        with Image.open(image_path) as image:
            width, height = image.size
        gt_detections = _labels_to_detections(
            label_path_for(image_path, labels_dir),
            width,
            height,
        )
        pred_detections = _prediction_detections(predictions.get(image_path.name, {}))

        gt_containers = [item for item in gt_detections if item.class_id == container_class]
        gt_seedlings = [item for item in gt_detections if item.class_id == seedling_class]
        pred_containers = _prediction_container_detections(
            predictions.get(image_path.name, {}),
            source=container_source,
            container_class=container_class,
        )
        pred_seedlings = [item for item in pred_detections if item.class_id == seedling_class]

        if use_gt_containers:
            matches = [(gt, gt, 1.0) for gt in gt_containers]
        else:
            matches = match_containers(gt_containers, pred_containers, container_iou)
        best_container_ious.extend(iou for _, _, iou in matches)

        image_total_cells = 0
        image_correct_cells = 0
        image_multi_counts = {"tp": 0, "fp": 0, "fn": 0}
        image_target_counts = {"tp": 0, "fp": 0, "fn": 0}
        image_target_distances: list[float] = []
        image_target_distances_mm: list[float] = []
        image_confusion = [[0 for _ in range(3)] for _ in range(3)]
        total_gt_containers += len(gt_containers)

        for gt_container, pred_container, iou in matches:
            if pred_container is not None:
                matched_containers += 1
            gt_cells = assign_to_cells(gt_container.box, gt_seedlings, rows, cols)
            pred_reference_box = pred_container.box if pred_container is not None else gt_container.box
            pred_cells = assign_to_cells(pred_reference_box, pred_seedlings, rows, cols)
            gt_matrix = count_matrix(gt_cells)
            pred_matrix = count_matrix(pred_cells)
            correct, total = _update_confusion(confusion, gt_matrix, pred_matrix)
            _update_confusion(image_confusion, gt_matrix, pred_matrix)
            image_correct_cells += correct
            image_total_cells += total
            _update_multi_counts(multi_counts, gt_matrix, pred_matrix)
            _update_multi_counts(image_multi_counts, gt_matrix, pred_matrix)

            gt_targets = choose_removal_targets(gt_cells)
            pred_targets = choose_removal_targets(pred_cells)
            match_count, distances, distances_mm = _match_targets(
                gt_targets,
                pred_targets,
                target_distance,
                target_distance_mm=target_distance_mm,
                calibration_artifact=calibration_artifact,
            )
            target_counts["tp"] += match_count
            target_counts["fn"] += max(0, len(gt_targets) - match_count)
            target_counts["fp"] += max(0, len(pred_targets) - match_count)
            image_target_counts["tp"] += match_count
            image_target_counts["fn"] += max(0, len(gt_targets) - match_count)
            image_target_counts["fp"] += max(0, len(pred_targets) - match_count)
            target_distances.extend(distances)
            target_distances_mm.extend(distances_mm)
            image_target_distances.extend(distances)
            image_target_distances_mm.extend(distances_mm)

        image_cost_sensitive = cost_sensitive_metrics_from_legacy(image_confusion, image_target_counts)
        image_summaries.append(
            {
                "image": image_path.name,
                **image_manifest_metadata.get(image_path.name, {}),
                "gt_containers": len(gt_containers),
                "pred_containers": len(pred_containers),
                "matched_containers": sum(1 for _, pred, _ in matches if pred is not None),
                "best_container_iou": max((iou for _, _, iou in matches), default=None),
                "cell_accuracy": image_correct_cells / image_total_cells if image_total_cells else None,
                "cell_correct": image_correct_cells,
                "cell_total": image_total_cells,
                "multi_tp": image_multi_counts["tp"],
                "multi_fp": image_multi_counts["fp"],
                "multi_fn": image_multi_counts["fn"],
                "target_tp": image_target_counts["tp"],
                "target_fp": image_target_counts["fp"],
                "target_fn": image_target_counts["fn"],
                "target_mean_coordinate_error_px": (
                    sum(image_target_distances) / len(image_target_distances)
                    if image_target_distances
                    else None
                ),
                "target_mean_coordinate_error_mm": (
                    sum(image_target_distances_mm) / len(image_target_distances_mm)
                    if image_target_distances_mm
                    else None
                ),
                "cost_total": image_cost_sensitive["total_cost"],
                "normalized_cost_per_cell": image_cost_sensitive["normalized_cost_per_cell"],
                "critical_error_total": image_cost_sensitive["critical_error_total"],
                "critical_error_rate_per_cell": image_cost_sensitive["critical_error_rate_per_cell"],
                "critical_error_counts": image_cost_sensitive["critical_error_counts"],
                "cost_sensitive": image_cost_sensitive,
            }
        )
        if len(unmatched_samples) < 25:
            for index, (gt_container, pred_container, iou) in enumerate(matches):
                if pred_container is None:
                    unmatched_samples.append(
                        {
                            "image": image_path.name,
                            "gt_container_index": index,
                            "best_iou": iou,
                            "gt_box": gt_container.box,
                        }
                    )
                    if len(unmatched_samples) >= 25:
                        break
        if image_total_cells:
            container_accuracies.append(image_correct_cells / image_total_cells)

    metrics = {
        "cell_accuracy": _accuracy(confusion),
        "cell_macro": _macro_metrics(confusion),
        "cell_confusion_matrix": confusion,
        "multi_seedling_cell": _prf(multi_counts),
        "removal_targets": {
            **_prf(target_counts),
            "mean_coordinate_error_px": (
                sum(target_distances) / len(target_distances) if target_distances else None
            ),
            "mean_coordinate_error_mm": (
                sum(target_distances_mm) / len(target_distances_mm) if target_distances_mm else None
            ),
            "matched_distances_px": target_distances,
            "matched_distances_mm": target_distances_mm,
            "calibration_id": getattr(calibration_artifact, "calibration_id", None),
        },
        "cost_sensitive": cost_sensitive_metrics_from_legacy(confusion, target_counts),
        "container_recall": matched_containers / total_gt_containers if total_gt_containers else None,
        "container_matching": {
            "prediction_source": container_source,
            "iou_threshold": container_iou,
            "total_gt_containers": total_gt_containers,
            "matched_containers": matched_containers,
            "best_iou_summary": _summary(best_container_ious),
            "best_iou_counts": _iou_counts(best_container_ious, container_iou),
            "unmatched_samples": unmatched_samples,
        },
        "prediction_coverage": coverage,
        "suspected_augmented_eval_images": suspected_eval_images,
        "cell_accuracy_bootstrap_ci": _bootstrap_ci(container_accuracies),
        "images": image_summaries,
    }
    metrics_path = output_dir / "cell_metrics.json"
    confusion_path = output_dir / "cell_confusion_matrix.csv"
    write_json(metrics_path, metrics)
    _write_confusion_csv(confusion_path, confusion)
    register_run_artifacts(
        output_dir,
        command,
        config=config,
        command_args=command_args,
        input_paths=_evaluation_input_paths(evaluation),
        output_paths=[metrics_path, confusion_path],
    )
    return metrics


def _load_image_manifest_metadata(
    manifest_path: Any,
    *,
    dataset_root: str | Path,
) -> dict[str, dict[str, str]]:
    path = Path(manifest_path) if manifest_path else _find_image_manifest(dataset_root)
    if path is None:
        return {}
    fields = {
        "image_id",
        "session_id",
        "group_id",
        "tray_id",
        "site",
        "greenhouse",
        "capture_date",
        "target_species",
        "days_after_sowing",
        "grid_rows",
        "grid_cols",
        "camera_id",
        "split",
        "calibration_id",
        "lighting",
        "watering_state",
        "operator_id",
    }
    metadata: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            summary = {field: str(row[field]) for field in fields if row.get(field)}
            if not summary:
                continue
            for key in _manifest_image_keys(row):
                metadata[key] = summary
    return metadata


def _evaluation_input_paths(evaluation: dict[str, Any]) -> list[Any]:
    paths: list[Any] = []
    if evaluation.get("gt_scene") or evaluation.get("pred_scene"):
        paths.extend([evaluation.get("gt_scene"), evaluation.get("pred_scene")])
    else:
        paths.extend([evaluation.get("dataset"), evaluation.get("predictions")])
    paths.extend(
        [
            evaluation.get("calibration"),
            evaluation.get("calibration_artifact"),
            evaluation.get("image_manifest"),
            evaluation.get("manifest"),
        ]
    )
    return [path for path in paths if path]


def _find_image_manifest(dataset_root: str | Path) -> Path | None:
    root = Path(dataset_root)
    for relative in ("manifests/image_manifest.csv", "image_manifest.csv"):
        candidate = root / relative
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _manifest_image_keys(row: dict[str, str]) -> set[str]:
    keys = set()
    for column in ("image", "image_id", "file_path"):
        value = row.get(column)
        if not value:
            continue
        path = Path(value)
        keys.add(value)
        keys.add(path.name)
        keys.add(path.stem)
    return keys


def _evaluate_scene_state_cells(evaluation: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    from seedling_cells.evaluation import evaluate_scene_states, load_scene_state

    if not evaluation.get("gt_scene") or not evaluation.get("pred_scene"):
        raise ValueError("SceneState evaluation requires evaluation.gt_scene and evaluation.pred_scene")
    target_distance = float(evaluation.get("target_match_distance_px", 25.0))
    target_distance_mm = _optional_float(evaluation.get("target_match_distance_mm"))
    gt_scene = load_scene_state(evaluation["gt_scene"])
    pred_scene = load_scene_state(evaluation["pred_scene"])
    scene_metrics = evaluate_scene_states(
        gt_scene,
        pred_scene,
        target_match_distance_px=target_distance,
        target_match_distance_mm=target_distance_mm,
    )
    metrics = _scene_state_metrics_as_cell_metrics(scene_metrics)
    write_json(output_dir / "cell_metrics.json", metrics)
    _write_confusion_csv(output_dir / "cell_confusion_matrix.csv", metrics["cell_confusion_matrix"])
    return metrics


def _scene_state_metrics_as_cell_metrics(scene_metrics: dict[str, Any]) -> dict[str, Any]:
    cell_metrics = scene_metrics["cell_metrics"]
    target_metrics = scene_metrics["target_metrics"]
    cost_sensitive = scene_metrics.get("cost_sensitive", {})
    if not isinstance(cost_sensitive, dict):
        cost_sensitive = {}
    legacy_confusion = _legacy_confusion_from_scene_metrics(cell_metrics["confusion"])
    matched_distances_px = [
        float(match["distance_px"])
        for match in target_metrics.get("matches", [])
        if match.get("pred_target_id") is not None and match.get("distance_px") is not None
    ]
    matched_distances_mm = [
        float(match["distance_mm"])
        for match in target_metrics.get("matches", [])
        if match.get("pred_target_id") is not None and match.get("distance_mm") is not None
    ]
    target_counts = {
        "tp": int(target_metrics.get("tp", 0)),
        "fp": int(target_metrics.get("fp", 0)),
        "fn": int(target_metrics.get("fn", 0)),
    }
    image_summary = {
        "image": scene_metrics.get("pred_scene_id") or scene_metrics.get("gt_scene_id"),
        "cell_accuracy": cell_metrics.get("accuracy"),
        "cell_correct": cell_metrics.get("correct_cells"),
        "cell_total": cell_metrics.get("total_cells"),
        "target_tp": target_counts["tp"],
        "target_fp": target_counts["fp"],
        "target_fn": target_counts["fn"],
        "target_mean_coordinate_error_px": target_metrics.get("mean_error_px"),
        "target_mean_coordinate_error_mm": target_metrics.get("mean_error_mm"),
        "cost_total": cost_sensitive.get("total_cost"),
        "normalized_cost_per_cell": cost_sensitive.get("normalized_cost_per_cell"),
        "critical_error_total": cost_sensitive.get("critical_error_total"),
        "critical_error_rate_per_cell": cost_sensitive.get("critical_error_rate_per_cell"),
        "critical_error_counts": cost_sensitive.get("critical_error_counts", {}),
    }
    return {
        "schema_mode": "scene_state",
        "cell_accuracy": cell_metrics.get("accuracy"),
        "cell_macro": _macro_metrics(legacy_confusion),
        "cell_confusion_matrix": legacy_confusion,
        "multi_seedling_cell": _scene_multi_counts(legacy_confusion),
        "removal_targets": {
            **_prf(target_counts),
            "mean_coordinate_error_px": target_metrics.get("mean_error_px"),
            "mean_coordinate_error_mm": target_metrics.get("mean_error_mm"),
            "matched_distances_px": matched_distances_px,
            "matched_distances_mm": matched_distances_mm,
            "calibration_id": None,
            "expert_keep_remove": target_metrics.get("expert_keep_remove", {}),
        },
        "cost_sensitive": cost_sensitive,
        "container_recall": None,
        "container_matching": {
            "prediction_source": "scene_state",
            "iou_threshold": None,
            "total_gt_containers": None,
            "matched_containers": None,
            "best_iou_summary": _summary([]),
            "best_iou_counts": _iou_counts([], 0.0),
            "unmatched_samples": [],
        },
        "prediction_coverage": {
            "dataset_images": 1,
            "prediction_images": 1,
            "missing_predictions": cell_metrics.get("missing_predictions", []),
            "extra_predictions": cell_metrics.get("unexpected_predictions", []),
        },
        "suspected_augmented_eval_images": [],
        "cell_accuracy_bootstrap_ci": {
            "mean": cell_metrics.get("accuracy"),
            "low_95": None,
            "high_95": None,
            "n": 1,
        },
        "images": [image_summary],
        "scene_state_metrics": scene_metrics,
    }


def _load_predictions(path: str | Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {item["image"]: item for item in raw.get("images", [])}


def _load_calibration_artifact(path: Any) -> Any:
    if path is None or path == "":
        return None
    from seedling_calibration.schemas import CalibrationArtifact

    return CalibrationArtifact.from_json(path)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _dataset_dirs(dataset: str | Path, split: str | None) -> tuple[Path, Path]:
    root = Path(dataset)
    if split:
        images = root / split / "images"
        labels = root / split / "labels"
    else:
        images = root / "images"
        labels = root / "labels"
    if not images.exists():
        raise FileNotFoundError(f"Images directory not found: {images}")
    return images, labels


def _labels_to_detections(label_path: Path, width: int, height: int) -> list[Detection]:
    detections: list[Detection] = []
    for label in read_labels(label_path):
        detections.append(
            Detection(
                box=label.to_xyxy(width, height),
                class_id=label.class_id,
                confidence=1.0,
            )
        )
    return detections


def _prediction_detections(prediction: dict[str, Any]) -> list[Detection]:
    detections = []
    for item in prediction.get("detections", []):
        detections.append(
            Detection(
                box=[float(value) for value in item["box"]],
                class_id=int(item["class_id"]),
                confidence=float(item.get("confidence", 1.0)),
                name=item.get("name"),
            )
        )
    return detections


def _prediction_container_detections(
    prediction: dict[str, Any],
    source: str,
    container_class: int,
) -> list[Detection]:
    if source == "detections":
        return [
            item
            for item in _prediction_detections(prediction)
            if item.class_id == container_class
        ]
    if source == "containers":
        return [
            Detection(
                box=[float(value) for value in item["box"]],
                class_id=int(item.get("class_id", container_class)),
                confidence=float(item.get("confidence", 1.0)),
                name=item.get("name"),
            )
            for item in prediction.get("containers", [])
        ]
    raise ValueError(
        "evaluation.container_prediction_source must be either 'detections' or 'containers'"
    )


def _category(count: int) -> int:
    return 2 if count > 1 else count


def _legacy_confusion_from_scene_metrics(confusion: dict[str, dict[str, int]]) -> list[list[int]]:
    legacy = [[0 for _ in range(3)] for _ in range(3)]
    for gt_state, pred_counts in confusion.items():
        gt_category = _legacy_cell_category(gt_state)
        for pred_state, count in pred_counts.items():
            pred_category = _legacy_cell_category(pred_state)
            legacy[gt_category][pred_category] += int(count)
    return legacy


def _legacy_cell_category(state: str) -> int:
    if state == "multiple_crop":
        return 2
    if state in {"single_crop", "crop_and_weed"}:
        return 1
    return 0


def _scene_multi_counts(confusion: list[list[int]]) -> dict[str, Any]:
    counts = {
        "tp": confusion[2][2],
        "fp": confusion[0][2] + confusion[1][2],
        "fn": confusion[2][0] + confusion[2][1],
    }
    return _prf(counts)


def _update_confusion(
    confusion: list[list[int]],
    gt_matrix: list[list[int]],
    pred_matrix: list[list[int]],
) -> tuple[int, int]:
    correct = 0
    total = 0
    for row_index, gt_row in enumerate(gt_matrix):
        for col_index, gt_count in enumerate(gt_row):
            gt_cat = _category(gt_count)
            pred_cat = _category(pred_matrix[row_index][col_index])
            confusion[gt_cat][pred_cat] += 1
            correct += int(gt_cat == pred_cat)
            total += 1
    return correct, total


def _update_multi_counts(
    counts: dict[str, int],
    gt_matrix: list[list[int]],
    pred_matrix: list[list[int]],
) -> None:
    for row_index, gt_row in enumerate(gt_matrix):
        for col_index, gt_count in enumerate(gt_row):
            gt_multi = gt_count > 1
            pred_multi = pred_matrix[row_index][col_index] > 1
            if gt_multi and pred_multi:
                counts["tp"] += 1
            elif not gt_multi and pred_multi:
                counts["fp"] += 1
            elif gt_multi and not pred_multi:
                counts["fn"] += 1


def _match_targets(
    gt_targets: list[dict[str, Any]],
    pred_targets: list[dict[str, Any]],
    max_distance: float,
    target_distance_mm: float | None = None,
    calibration_artifact: Any = None,
) -> tuple[int, list[float], list[float]]:
    used_pred: set[int] = set()
    distances: list[float] = []
    distances_mm: list[float] = []
    for gt in gt_targets:
        best_index = None
        best_distance = float("inf")
        best_distance_mm = None
        gt_center = [float(value) for value in gt["remove_center"]]
        for index, pred in enumerate(pred_targets):
            if index in used_pred:
                continue
            pred_center = [float(value) for value in pred["remove_center"]]
            distance = center_distance(gt_center, pred_center)
            if distance > max_distance:
                continue
            distance_mm = _calibrated_center_distance_mm(gt_center, pred_center, calibration_artifact)
            if target_distance_mm is not None and (
                distance_mm is None or distance_mm > target_distance_mm
            ):
                continue
            if distance < best_distance:
                best_distance = distance
                best_index = index
                best_distance_mm = distance_mm
        if best_index is not None:
            used_pred.add(best_index)
            distances.append(best_distance)
            if best_distance_mm is not None:
                distances_mm.append(best_distance_mm)
    return len(distances), distances, distances_mm


def _calibrated_center_distance_mm(
    gt_center_px: list[float],
    pred_center_px: list[float],
    calibration_artifact: Any,
) -> float | None:
    if calibration_artifact is None:
        return None
    from seedling_calibration.transforms import image_px_to_tray_mm

    gt_mm = image_px_to_tray_mm(gt_center_px, calibration_artifact)
    pred_mm = image_px_to_tray_mm(pred_center_px, calibration_artifact)
    return center_distance(gt_mm, pred_mm)


def _accuracy(confusion: list[list[int]]) -> float | None:
    total = sum(sum(row) for row in confusion)
    if not total:
        return None
    correct = sum(confusion[index][index] for index in range(len(confusion)))
    return correct / total


def _macro_metrics(confusion: list[list[int]]) -> dict[str, Any]:
    per_class = {}
    precisions = []
    recalls = []
    f1s = []
    for class_id in range(3):
        tp = confusion[class_id][class_id]
        fp = sum(confusion[row][class_id] for row in range(3) if row != class_id)
        fn = sum(confusion[class_id][col] for col in range(3) if col != class_id)
        scores = _prf({"tp": tp, "fp": fp, "fn": fn})
        per_class[str(class_id)] = scores
        if scores["precision"] is not None:
            precisions.append(scores["precision"])
        if scores["recall"] is not None:
            recalls.append(scores["recall"])
        if scores["f1"] is not None:
            f1s.append(scores["f1"])
    return {
        "labels": {"0": "empty", "1": "single", "2": "multiple"},
        "per_class": per_class,
        "macro_precision": _mean(precisions),
        "macro_recall": _mean(recalls),
        "macro_f1": _mean(f1s),
    }


def _prf(counts: dict[str, int]) -> dict[str, Any]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall)
        else None
    )
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _bootstrap_ci(values: list[float], iterations: int = 1000, seed: int = 42) -> dict[str, Any]:
    if not values:
        return {"mean": None, "low_95": None, "high_95": None, "n": 0}
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        sample = [rng.choice(values) for _ in values]
        samples.append(sum(sample) / len(sample))
    samples.sort()
    low = samples[int(0.025 * len(samples))]
    high = samples[int(0.975 * len(samples))]
    return {"mean": sum(values) / len(values), "low_95": low, "high_95": high, "n": len(values)}


def _write_confusion_csv(path: Path, confusion: list[list[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gt/pred", "empty", "single", "multiple"])
        for label, row in zip(["empty", "single", "multiple"], confusion):
            writer.writerow([label, *row])


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _prediction_coverage(image_paths: list[Path], predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    dataset_images = [image.name for image in image_paths]
    dataset_set = set(dataset_images)
    prediction_set = set(predictions)
    missing = sorted(dataset_set - prediction_set)
    extra = sorted(prediction_set - dataset_set)
    return {
        "dataset_images": len(dataset_images),
        "prediction_images": len(prediction_set),
        "missing_predictions": missing,
        "extra_predictions": extra,
    }


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "mean": sum(values) / len(values),
        "max": ordered[-1],
    }


def _iou_counts(values: list[float], threshold: float) -> dict[str, int]:
    return {
        "gt_0": sum(1 for value in values if value > 0.0),
        "ge_0_25": sum(1 for value in values if value >= 0.25),
        "ge_0_5": sum(1 for value in values if value >= 0.5),
        "ge_threshold": sum(1 for value in values if value >= threshold),
    }
