from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from .config import save_run_snapshot, write_json
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


def evaluate_cells_from_config(config: dict[str, Any]) -> dict[str, Any]:
    evaluation = config.get("evaluation", {})
    output_dir = Path(evaluation.get("output_dir", "runs/evaluation"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_run_snapshot(output_dir, config, "evaluate-cells")

    predictions = _load_predictions(evaluation["predictions"])
    images_dir, labels_dir = _dataset_dirs(evaluation["dataset"], evaluation.get("split"))
    rows = int(evaluation.get("grid_rows", 11))
    cols = int(evaluation.get("grid_cols", 11))
    container_class = int(evaluation.get("container_class", 0))
    seedling_class = int(evaluation.get("seedling_class", 1))
    container_iou = float(evaluation.get("container_iou", 0.5))
    target_distance = float(evaluation.get("target_match_distance_px", 25.0))
    use_gt_containers = bool(evaluation.get("use_ground_truth_containers", False))

    confusion = [[0 for _ in range(3)] for _ in range(3)]
    container_accuracies: list[float] = []
    multi_counts = {"tp": 0, "fp": 0, "fn": 0}
    target_counts = {"tp": 0, "fp": 0, "fn": 0}
    target_distances: list[float] = []
    matched_containers = 0
    total_gt_containers = 0
    image_summaries: list[dict[str, Any]] = []

    for image_path in list_images(images_dir):
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
        pred_containers = [item for item in pred_detections if item.class_id == container_class]
        pred_seedlings = [item for item in pred_detections if item.class_id == seedling_class]

        if use_gt_containers:
            matches = [(gt, gt, 1.0) for gt in gt_containers]
        else:
            matches = match_containers(gt_containers, pred_containers, container_iou)

        image_total_cells = 0
        image_correct_cells = 0
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
            image_correct_cells += correct
            image_total_cells += total
            _update_multi_counts(multi_counts, gt_matrix, pred_matrix)

            gt_targets = choose_removal_targets(gt_cells)
            pred_targets = choose_removal_targets(pred_cells)
            match_count, distances = _match_targets(gt_targets, pred_targets, target_distance)
            target_counts["tp"] += match_count
            target_counts["fn"] += max(0, len(gt_targets) - match_count)
            target_counts["fp"] += max(0, len(pred_targets) - match_count)
            target_distances.extend(distances)

        image_summaries.append(
            {
                "image": image_path.name,
                "gt_containers": len(gt_containers),
                "matched_containers": sum(1 for _, pred, _ in matches if pred is not None),
                "cell_accuracy": image_correct_cells / image_total_cells if image_total_cells else None,
            }
        )
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
            "matched_distances_px": target_distances,
        },
        "container_recall": matched_containers / total_gt_containers if total_gt_containers else None,
        "cell_accuracy_bootstrap_ci": _bootstrap_ci(container_accuracies),
        "images": image_summaries,
    }
    write_json(output_dir / "cell_metrics.json", metrics)
    _write_confusion_csv(output_dir / "cell_confusion_matrix.csv", confusion)
    return metrics


def _load_predictions(path: str | Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {item["image"]: item for item in raw.get("images", [])}


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


def _category(count: int) -> int:
    return 2 if count > 1 else count


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
) -> tuple[int, list[float]]:
    used_pred: set[int] = set()
    distances: list[float] = []
    for gt in gt_targets:
        best_index = None
        best_distance = float("inf")
        gt_center = [float(value) for value in gt["remove_center"]]
        for index, pred in enumerate(pred_targets):
            if index in used_pred:
                continue
            pred_center = [float(value) for value in pred["remove_center"]]
            distance = center_distance(gt_center, pred_center)
            if distance < best_distance:
                best_distance = distance
                best_index = index
        if best_index is not None and best_distance <= max_distance:
            used_pred.add(best_index)
            distances.append(best_distance)
    return len(distances), distances


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
