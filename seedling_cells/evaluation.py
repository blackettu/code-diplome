from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from seedling_core.schemas import ActionTarget, CellState, SceneState


DEFAULT_STATE_ORDER = [
    "empty",
    "single_crop",
    "multiple_crop",
    "weed_only",
    "crop_and_weed",
    "unknown",
    "ambiguous",
    "image_quality_insufficient",
]

LEGACY_CELL_STATE_ORDER = ["empty", "single_crop", "multiple_crop"]


def evaluate_scene_states(
    gt_scene: SceneState,
    pred_scene: SceneState,
    target_match_distance_px: float = 25.0,
    target_match_distance_mm: float | None = None,
    state_order: list[str] | None = None,
) -> dict[str, Any]:
    states = state_order or DEFAULT_STATE_ORDER
    cell_metrics = evaluate_cell_states(gt_scene.cells, pred_scene.cells, states)
    target_metrics = evaluate_action_targets(
        gt_scene.targets,
        pred_scene.targets,
        max_distance_px=target_match_distance_px,
        max_distance_mm=target_match_distance_mm,
    )
    cost_metrics = cost_sensitive_metrics(cell_metrics, target_metrics)
    return {
        "gt_scene_id": gt_scene.scene_id,
        "pred_scene_id": pred_scene.scene_id,
        "cell_metrics": cell_metrics,
        "target_metrics": target_metrics,
        "cost_sensitive": cost_metrics,
    }


def evaluate_scene_state_files(
    gt_path: str | Path,
    pred_path: str | Path,
    output_path: str | Path | None = None,
    target_match_distance_px: float = 25.0,
    target_match_distance_mm: float | None = None,
) -> dict[str, Any]:
    gt_scene = load_scene_state(gt_path)
    pred_scene = load_scene_state(pred_path)
    metrics = evaluate_scene_states(
        gt_scene,
        pred_scene,
        target_match_distance_px=target_match_distance_px,
        target_match_distance_mm=target_match_distance_mm,
    )
    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def evaluate_cell_states(gt_cells: list[CellState], pred_cells: list[CellState], state_order: list[str]) -> dict[str, Any]:
    pred_by_id = {cell.cell_id: cell for cell in pred_cells}
    confusion: dict[str, dict[str, int]] = {state: {pred: 0 for pred in state_order} for state in state_order}
    missing_predictions: list[str] = []
    unexpected_predictions = sorted(set(pred_by_id) - {cell.cell_id for cell in gt_cells})
    cell_records: list[dict[str, Any]] = []
    total = 0
    correct = 0
    for gt in gt_cells:
        pred = pred_by_id.get(gt.cell_id)
        gt_state = _known_state(gt.state, state_order)
        pred_state = _known_state(pred.state if pred else "missing_prediction", state_order)
        if pred is None:
            missing_predictions.append(gt.cell_id)
        confusion.setdefault(gt_state, {state: 0 for state in [*state_order, pred_state]})
        for row in confusion.values():
            row.setdefault(pred_state, 0)
        confusion[gt_state][pred_state] += 1
        total += 1
        is_correct = gt_state == pred_state
        correct += int(is_correct)
        cell_records.append(
            {
                "cell_id": gt.cell_id,
                "row": gt.row,
                "col": gt.col,
                "gt_state": gt_state,
                "pred_state": pred_state,
                "correct": is_correct,
            }
        )
    return {
        "accuracy": correct / total if total else None,
        "total_cells": total,
        "correct_cells": correct,
        "confusion": confusion,
        "macro": _macro_from_confusion(confusion),
        "edge_cell_metrics": _region_cell_metrics(cell_records, region="edge"),
        "corner_cell_metrics": _region_cell_metrics(cell_records, region="corner"),
        "missing_predictions": missing_predictions,
        "unexpected_predictions": unexpected_predictions,
        "richer_state_counts": dict(sorted(Counter(cell.state for cell in gt_cells).items())),
    }


def evaluate_action_targets(
    gt_targets: list[ActionTarget],
    pred_targets: list[ActionTarget],
    max_distance_px: float = 25.0,
    max_distance_mm: float | None = None,
) -> dict[str, Any]:
    matches = _match_targets(gt_targets, pred_targets, max_distance_px=max_distance_px, max_distance_mm=max_distance_mm)
    matched_pred_ids = {match["pred_target_id"] for match in matches if match["pred_target_id"] is not None}
    false_positives = [target.target_id for target in pred_targets if target.target_id not in matched_pred_ids]
    false_negatives = [match["gt_target_id"] for match in matches if match["pred_target_id"] is None]
    tp = len(matches) - len(false_negatives)
    fp = len(false_positives)
    fn = len(false_negatives)
    px_errors = [match["distance_px"] for match in matches if match.get("distance_px") is not None and match["pred_target_id"] is not None]
    mm_errors = [match["distance_mm"] for match in matches if match.get("distance_mm") is not None and match["pred_target_id"] is not None]
    keep_remove = _expert_keep_remove_metrics(gt_targets, pred_targets)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if (tp + fp) else None,
        "recall": tp / (tp + fn) if (tp + fn) else None,
        "matches": matches,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "mean_error_px": sum(px_errors) / len(px_errors) if px_errors else None,
        "mean_error_mm": sum(mm_errors) / len(mm_errors) if mm_errors else None,
        "expert_keep_remove": keep_remove,
    }


def cost_sensitive_metrics(cell_metrics: dict[str, Any], target_metrics: dict[str, Any]) -> dict[str, Any]:
    confusion = cell_metrics.get("confusion", {})
    costs = {
        "missed_multiple_crop": 8.0,
        "false_multiple_crop": 3.0,
        "missed_weed": 10.0,
        "unknown_not_reviewed": 6.0,
        "target_false_negative": 12.0,
        "target_false_positive": 5.0,
    }
    expert_keep_remove = target_metrics.get("expert_keep_remove", {})
    critical_error_counts = {
        "target_false_negative": int(target_metrics.get("fn", 0)),
        "target_false_positive": int(target_metrics.get("fp", 0)),
        "missed_multiple_crop": _row_errors(confusion, "multiple_crop"),
        "false_multiple_crop": _col_errors(confusion, "multiple_crop"),
        "missed_weed_only": _row_errors(confusion, "weed_only"),
        "missed_crop_and_weed": _row_errors(confusion, "crop_and_weed"),
        "unknown_not_reviewed": _row_errors(confusion, "unknown"),
        "expert_false_removal": int(expert_keep_remove.get("fp", 0) or 0),
        "expert_missed_removal": int(expert_keep_remove.get("fn", 0) or 0),
    }
    cost_breakdown = {
        "target_false_negative": costs["target_false_negative"] * critical_error_counts["target_false_negative"],
        "target_false_positive": costs["target_false_positive"] * critical_error_counts["target_false_positive"],
        "missed_multiple_crop": costs["missed_multiple_crop"] * critical_error_counts["missed_multiple_crop"],
        "false_multiple_crop": costs["false_multiple_crop"] * critical_error_counts["false_multiple_crop"],
        "missed_weed_only": costs["missed_weed"] * critical_error_counts["missed_weed_only"],
        "missed_crop_and_weed": costs["missed_weed"] * critical_error_counts["missed_crop_and_weed"],
        "unknown_not_reviewed": costs["unknown_not_reviewed"] * critical_error_counts["unknown_not_reviewed"],
    }
    total_cost = sum(cost_breakdown.values())
    total_cells = max(int(cell_metrics.get("total_cells", 0) or 0), 1)
    critical_error_total = sum(critical_error_counts.values())
    return {
        "total_cost": total_cost,
        "costs": costs,
        "cost_breakdown": cost_breakdown,
        "critical_error_counts": critical_error_counts,
        "critical_error_total": critical_error_total,
        "critical_error_rate_per_cell": critical_error_total / total_cells,
        "normalized_cost_per_cell": total_cost / total_cells,
    }


def cost_sensitive_metrics_from_legacy(
    confusion: list[list[int]],
    target_metrics: dict[str, Any],
    state_order: list[str] | None = None,
) -> dict[str, Any]:
    states = state_order or LEGACY_CELL_STATE_ORDER
    state_confusion = _legacy_matrix_to_state_confusion(confusion, states)
    total_cells = sum(sum(int(count or 0) for count in row) for row in confusion)
    return cost_sensitive_metrics(
        {"confusion": state_confusion, "total_cells": total_cells},
        target_metrics,
    )


def load_scene_state(path: str | Path) -> SceneState:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(data, dict) and "scenes" in data:
        scenes = data.get("scenes")
        if not isinstance(scenes, list) or not scenes:
            raise ValueError(f"SceneState bundle contains no scenes: {path}")
        data = scenes[0]
    if not isinstance(data, dict):
        raise ValueError(f"SceneState file must contain a JSON object: {path}")
    return SceneState.from_dict(data)


def _match_targets(
    gt_targets: list[ActionTarget],
    pred_targets: list[ActionTarget],
    max_distance_px: float,
    max_distance_mm: float | None,
) -> list[dict[str, Any]]:
    used_pred: set[int] = set()
    matches: list[dict[str, Any]] = []
    for gt in gt_targets:
        best_index = None
        best_px = float("inf")
        best_mm = None
        for index, pred in enumerate(pred_targets):
            if index in used_pred:
                continue
            if pred.target_type != gt.target_type:
                continue
            distance_px = _distance(gt.action_point_px, pred.action_point_px)
            distance_mm = _distance(gt.action_point_mm, pred.action_point_mm) if gt.action_point_mm and pred.action_point_mm else None
            passes_px = distance_px <= max_distance_px
            passes_mm = max_distance_mm is None or (distance_mm is not None and distance_mm <= max_distance_mm)
            if passes_px and passes_mm and distance_px < best_px:
                best_index = index
                best_px = distance_px
                best_mm = distance_mm
        if best_index is None:
            matches.append(
                {
                    "gt_target_id": gt.target_id,
                    "pred_target_id": None,
                    "distance_px": None,
                    "distance_mm": None,
                }
            )
            continue
        used_pred.add(best_index)
        matches.append(
            {
                "gt_target_id": gt.target_id,
                "pred_target_id": pred_targets[best_index].target_id,
                "distance_px": best_px,
                "distance_mm": best_mm,
            }
        )
    return matches


def _expert_keep_remove_metrics(gt_targets: list[ActionTarget], pred_targets: list[ActionTarget]) -> dict[str, Any]:
    gt_remove = {target.object_id for target in gt_targets if target.target_type in {"remove_extra_crop", "remove_weed"}}
    pred_remove = {target.object_id for target in pred_targets if target.target_type in {"remove_extra_crop", "remove_weed"}}
    tp = len(gt_remove & pred_remove)
    fp = len(pred_remove - gt_remove)
    fn = len(gt_remove - pred_remove)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if (tp + fp) else None,
        "recall": tp / (tp + fn) if (tp + fn) else None,
        "gt_remove_object_ids": sorted(gt_remove),
        "pred_remove_object_ids": sorted(pred_remove),
    }


def _macro_from_confusion(confusion: dict[str, dict[str, int]]) -> dict[str, Any]:
    per_state = {}
    f1s = []
    states = sorted(confusion)
    for state in states:
        tp = confusion.get(state, {}).get(state, 0)
        fp = sum(confusion.get(other, {}).get(state, 0) for other in states if other != state)
        fn = sum(count for pred_state, count in confusion.get(state, {}).items() if pred_state != state)
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and (precision + recall) else None
        per_state[state] = {"precision": precision, "recall": recall, "f1": f1}
        if f1 is not None:
            f1s.append(f1)
    return {"per_state": per_state, "macro_f1": sum(f1s) / len(f1s) if f1s else None}


def _region_cell_metrics(cell_records: list[dict[str, Any]], region: str) -> dict[str, Any]:
    if not cell_records:
        return {
            "accuracy": None,
            "total_cells": 0,
            "correct_cells": 0,
            "mismatches": [],
            "gt_state_counts": {},
        }
    max_row = max(int(record["row"]) for record in cell_records)
    max_col = max(int(record["col"]) for record in cell_records)
    selected = [record for record in cell_records if _in_region(record, region, max_row, max_col)]
    total = len(selected)
    correct = sum(1 for record in selected if record["correct"])
    mismatches = [
        {
            "cell_id": record["cell_id"],
            "row": record["row"],
            "col": record["col"],
            "gt_state": record["gt_state"],
            "pred_state": record["pred_state"],
        }
        for record in selected
        if not record["correct"]
    ]
    return {
        "accuracy": correct / total if total else None,
        "total_cells": total,
        "correct_cells": correct,
        "mismatches": mismatches,
        "gt_state_counts": dict(sorted(Counter(record["gt_state"] for record in selected).items())),
    }


def _in_region(record: dict[str, Any], region: str, max_row: int, max_col: int) -> bool:
    row = int(record["row"])
    col = int(record["col"])
    on_row_edge = row == 0 or row == max_row
    on_col_edge = col == 0 or col == max_col
    if region == "edge":
        return on_row_edge or on_col_edge
    if region == "corner":
        return on_row_edge and on_col_edge
    raise ValueError(f"Unsupported cell metric region: {region}")


def _known_state(state: str, state_order: list[str]) -> str:
    return state if state in state_order else "other"


def _row_errors(confusion: dict[str, dict[str, int]], state: str) -> int:
    return sum(count for pred_state, count in confusion.get(state, {}).items() if pred_state != state)


def _col_errors(confusion: dict[str, dict[str, int]], state: str) -> int:
    return sum(row.get(state, 0) for gt_state, row in confusion.items() if gt_state != state)


def _legacy_matrix_to_state_confusion(
    confusion: list[list[int]],
    state_order: list[str],
) -> dict[str, dict[str, int]]:
    state_confusion: dict[str, dict[str, int]] = {}
    known_states = list(state_order)
    for row_index, row in enumerate(confusion):
        gt_state = known_states[row_index] if row_index < len(known_states) else f"class_{row_index}"
        state_confusion.setdefault(gt_state, {state: 0 for state in known_states})
        for col_index, raw_count in enumerate(row):
            pred_state = known_states[col_index] if col_index < len(known_states) else f"class_{col_index}"
            for counts in state_confusion.values():
                counts.setdefault(pred_state, 0)
            state_confusion[gt_state][pred_state] = state_confusion[gt_state].get(pred_state, 0) + int(raw_count or 0)
    for state in known_states:
        state_confusion.setdefault(state, {known_state: 0 for known_state in known_states})
    return state_confusion


def _distance(left: list[float] | None, right: list[float] | None) -> float:
    if left is None or right is None:
        return float("inf")
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))
