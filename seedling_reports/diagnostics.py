from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable


ERROR_CATEGORY_GROUPS = {
    "missing_prediction": "image",
    "extra_prediction": "image",
    "cell_missing_prediction": "image",
    "cell_extra_prediction": "image",
    "augmented_eval_image": "image",
    "container_unmatched": "geometry",
    "container_low_iou": "geometry",
    "target_coordinate_error": "geometry",
    "cell_state_mismatch": "biology",
    "multi_cell_false_positive": "biology",
    "multi_cell_false_negative": "biology",
    "target_false_positive": "decision",
    "target_false_negative": "decision",
}


def build_cell_metrics_diagnostics(
    metrics_path: str | Path,
    output_dir: str | Path | None = None,
    iterations: int = 1000,
    seed: int = 42,
    group_key: str = "image",
) -> dict[str, Any]:
    metrics = _load_metrics(metrics_path)
    bootstrap = bootstrap_cell_metrics(metrics.get("images", []), iterations=iterations, seed=seed, group_key=group_key)
    taxonomy = error_taxonomy(metrics)
    payload = {"metrics": str(metrics_path), "bootstrap": bootstrap, "error_taxonomy": taxonomy}
    if output_dir:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "bootstrap_ci.json").write_text(json.dumps(bootstrap, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "error_taxonomy.json").write_text(json.dumps(taxonomy, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def bootstrap_cell_metrics(
    image_rows: list[dict[str, Any]],
    iterations: int = 1000,
    seed: int = 42,
    group_key: str = "image",
) -> dict[str, Any]:
    groups = _groups(image_rows, group_key)
    sampled_rows = list(groups.values())
    return {
        "group_key": group_key,
        "groups": len(sampled_rows),
        "cell_accuracy": _bootstrap(sampled_rows, _cell_accuracy, iterations, seed),
        "multi_precision": _bootstrap(sampled_rows, lambda rows: _prf(rows, "multi", "precision"), iterations, seed),
        "multi_recall": _bootstrap(sampled_rows, lambda rows: _prf(rows, "multi", "recall"), iterations, seed),
        "target_precision": _bootstrap(sampled_rows, lambda rows: _prf(rows, "target", "precision"), iterations, seed),
        "target_recall": _bootstrap(sampled_rows, lambda rows: _prf(rows, "target", "recall"), iterations, seed),
        "container_recall": _bootstrap(sampled_rows, _container_recall, iterations, seed),
        "critical_error_rate_per_cell": _bootstrap(sampled_rows, _critical_error_rate_per_cell, iterations, seed),
        "normalized_cost_per_cell": _bootstrap(sampled_rows, _normalized_cost_per_cell, iterations, seed),
    }


def error_taxonomy(metrics: dict[str, Any]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    samples: dict[str, list[dict[str, Any]]] = {}
    scene_state_metrics = metrics.get("scene_state_metrics")
    has_scene_state_metrics = isinstance(scene_state_metrics, dict)

    if not has_scene_state_metrics:
        coverage = metrics.get("prediction_coverage", {})
        for image in coverage.get("missing_predictions", []) or []:
            _add(counts, group_counts, samples, "missing_prediction", {"image": image})
        for image in coverage.get("extra_predictions", []) or []:
            _add(counts, group_counts, samples, "extra_prediction", {"image": image})
    for image in metrics.get("suspected_augmented_eval_images", []) or []:
        _add(counts, group_counts, samples, "augmented_eval_image", {"image": image})

    container_matching = metrics.get("container_matching", {})
    for sample in container_matching.get("unmatched_samples", []) or []:
        _add(counts, group_counts, samples, "container_unmatched", dict(sample))

    threshold_value = container_matching.get("iou_threshold", 0.5)
    threshold = 0.5 if threshold_value is None else float(threshold_value)
    for row in metrics.get("images", []) or []:
        image = row.get("image")
        best_iou = row.get("best_container_iou")
        if best_iou is not None and float(best_iou) < threshold:
            _add(counts, group_counts, samples, "container_low_iou", {"image": image, "best_container_iou": best_iou})
        if not has_scene_state_metrics:
            if row.get("cell_accuracy") is not None and float(row["cell_accuracy"]) < 1.0:
                _add(counts, group_counts, samples, "cell_state_mismatch", {"image": image, "cell_accuracy": row["cell_accuracy"]})
            if int(row.get("multi_fp", 0) or 0) > 0:
                _add(counts, group_counts, samples, "multi_cell_false_positive", {"image": image, "multi_fp": row.get("multi_fp")})
            if int(row.get("multi_fn", 0) or 0) > 0:
                _add(counts, group_counts, samples, "multi_cell_false_negative", {"image": image, "multi_fn": row.get("multi_fn")})
            if int(row.get("target_fp", 0) or 0) > 0:
                _add(counts, group_counts, samples, "target_false_positive", {"image": image, "target_fp": row.get("target_fp")})
            if int(row.get("target_fn", 0) or 0) > 0:
                _add(counts, group_counts, samples, "target_false_negative", {"image": image, "target_fn": row.get("target_fn")})

    if has_scene_state_metrics:
        _add_scene_state_taxonomy(counts, group_counts, samples, scene_state_metrics)

    cost_sensitive = _cost_sensitive_summary(metrics)
    return {
        "counts": dict(sorted(counts.items())),
        "group_counts": dict(sorted(group_counts.items())),
        "category_groups": {
            category: ERROR_CATEGORY_GROUPS.get(category, "unknown")
            for category in sorted(counts)
        },
        "samples": {key: value[:10] for key, value in sorted(samples.items())},
        "cost_sensitive": cost_sensitive,
    }


def _bootstrap(
    grouped_rows: list[list[dict[str, Any]]],
    metric_fn: Callable[[list[dict[str, Any]]], float | None],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if not grouped_rows:
        return {"mean": None, "low_95": None, "high_95": None, "groups": 0}
    observed = metric_fn([row for group in grouped_rows for row in group])
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        sample_groups = [rng.choice(grouped_rows) for _ in grouped_rows]
        sample_rows = [row for group in sample_groups for row in group]
        value = metric_fn(sample_rows)
        if value is not None:
            samples.append(value)
    if not samples:
        return {"mean": observed, "low_95": None, "high_95": None, "groups": len(grouped_rows)}
    samples.sort()
    low = samples[int(0.025 * (len(samples) - 1))]
    high = samples[int(0.975 * (len(samples) - 1))]
    return {"mean": observed, "low_95": low, "high_95": high, "groups": len(grouped_rows)}


def _groups(rows: list[dict[str, Any]], group_key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(group_key) or row.get("image") or len(groups))
        groups.setdefault(key, []).append(row)
    return groups


def _cell_accuracy(rows: list[dict[str, Any]]) -> float | None:
    correct = sum(int(row.get("cell_correct", 0) or 0) for row in rows)
    total = sum(int(row.get("cell_total", 0) or 0) for row in rows)
    if total:
        return correct / total
    values = [float(row["cell_accuracy"]) for row in rows if row.get("cell_accuracy") is not None]
    return sum(values) / len(values) if values else None


def _container_recall(rows: list[dict[str, Any]]) -> float | None:
    matched = sum(int(row.get("matched_containers", 0) or 0) for row in rows)
    total = sum(int(row.get("gt_containers", 0) or 0) for row in rows)
    return matched / total if total else None


def _critical_error_rate_per_cell(rows: list[dict[str, Any]]) -> float | None:
    errors = sum(int(row.get("critical_error_total", 0) or 0) for row in rows)
    total = sum(int(row.get("cell_total", 0) or 0) for row in rows)
    if total:
        return errors / total
    values = [
        float(row["critical_error_rate_per_cell"])
        for row in rows
        if row.get("critical_error_rate_per_cell") is not None
    ]
    return sum(values) / len(values) if values else None


def _normalized_cost_per_cell(rows: list[dict[str, Any]]) -> float | None:
    cost = sum(float(row.get("cost_total", 0.0) or 0.0) for row in rows)
    total = sum(int(row.get("cell_total", 0) or 0) for row in rows)
    if total:
        return cost / total
    values = [
        float(row["normalized_cost_per_cell"])
        for row in rows
        if row.get("normalized_cost_per_cell") is not None
    ]
    return sum(values) / len(values) if values else None


def _prf(rows: list[dict[str, Any]], prefix: str, metric: str) -> float | None:
    tp = sum(int(row.get(f"{prefix}_tp", 0) or 0) for row in rows)
    fp = sum(int(row.get(f"{prefix}_fp", 0) or 0) for row in rows)
    fn = sum(int(row.get(f"{prefix}_fn", 0) or 0) for row in rows)
    if metric == "precision":
        return tp / (tp + fp) if (tp + fp) else None
    if metric == "recall":
        return tp / (tp + fn) if (tp + fn) else None
    raise ValueError(f"Unsupported PRF metric: {metric}")


def _add(
    counts: Counter[str],
    group_counts: Counter[str],
    samples: dict[str, list[dict[str, Any]]],
    category: str,
    sample: dict[str, Any],
    amount: int = 1,
) -> None:
    if amount <= 0:
        return
    group = ERROR_CATEGORY_GROUPS.get(category, "unknown")
    counts[category] += amount
    group_counts[group] += amount
    samples.setdefault(category, []).append({"group": group, **sample})


def _add_scene_state_taxonomy(
    counts: Counter[str],
    group_counts: Counter[str],
    samples: dict[str, list[dict[str, Any]]],
    scene_metrics: dict[str, Any],
) -> None:
    scene_sample = {
        "gt_scene_id": scene_metrics.get("gt_scene_id"),
        "pred_scene_id": scene_metrics.get("pred_scene_id"),
    }
    cell_metrics = scene_metrics.get("cell_metrics", {})
    if isinstance(cell_metrics, dict):
        for cell_id in cell_metrics.get("missing_predictions", []) or []:
            _add(counts, group_counts, samples, "cell_missing_prediction", {**scene_sample, "cell_id": cell_id})
        for cell_id in cell_metrics.get("unexpected_predictions", []) or []:
            _add(counts, group_counts, samples, "cell_extra_prediction", {**scene_sample, "cell_id": cell_id})
        confusion = cell_metrics.get("confusion", {})
        if isinstance(confusion, dict):
            _add_cell_confusion_taxonomy(counts, group_counts, samples, confusion, scene_sample)

    target_metrics = scene_metrics.get("target_metrics", {})
    if isinstance(target_metrics, dict):
        for target_id in target_metrics.get("false_positives", []) or []:
            _add(counts, group_counts, samples, "target_false_positive", {**scene_sample, "target_id": target_id})
        for target_id in target_metrics.get("false_negatives", []) or []:
            _add(counts, group_counts, samples, "target_false_negative", {**scene_sample, "target_id": target_id})
        for match in target_metrics.get("matches", []) or []:
            if not isinstance(match, dict) or match.get("pred_target_id") is None:
                continue
            distance_px = _optional_float(match.get("distance_px"))
            distance_mm = _optional_float(match.get("distance_mm"))
            if (distance_px is not None and distance_px > 0.0) or (distance_mm is not None and distance_mm > 0.0):
                _add(
                    counts,
                    group_counts,
                    samples,
                    "target_coordinate_error",
                    {
                        **scene_sample,
                        "gt_target_id": match.get("gt_target_id"),
                        "pred_target_id": match.get("pred_target_id"),
                        "distance_px": distance_px,
                        "distance_mm": distance_mm,
                    },
                )


def _add_cell_confusion_taxonomy(
    counts: Counter[str],
    group_counts: Counter[str],
    samples: dict[str, list[dict[str, Any]]],
    confusion: dict[str, Any],
    scene_sample: dict[str, Any],
) -> None:
    for gt_state, pred_counts in confusion.items():
        if not isinstance(pred_counts, dict):
            continue
        for pred_state, raw_count in pred_counts.items():
            if pred_state == gt_state:
                continue
            count = int(raw_count or 0)
            _add(
                counts,
                group_counts,
                samples,
                "cell_state_mismatch",
                {**scene_sample, "gt_state": gt_state, "pred_state": pred_state, "count": count},
                amount=count,
            )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _cost_sensitive_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    cost_sensitive = metrics.get("cost_sensitive")
    if not isinstance(cost_sensitive, dict):
        scene_metrics = metrics.get("scene_state_metrics", {})
        cost_sensitive = scene_metrics.get("cost_sensitive", {}) if isinstance(scene_metrics, dict) else {}
    if isinstance(cost_sensitive, dict) and cost_sensitive:
        return {
            "total_cost": cost_sensitive.get("total_cost"),
            "normalized_cost_per_cell": cost_sensitive.get("normalized_cost_per_cell"),
            "critical_error_total": cost_sensitive.get("critical_error_total"),
            "critical_error_rate_per_cell": cost_sensitive.get("critical_error_rate_per_cell"),
            "critical_error_counts": cost_sensitive.get("critical_error_counts", {}),
            "cost_breakdown": cost_sensitive.get("cost_breakdown", {}),
        }

    counts: Counter[str] = Counter()
    cost_total = 0.0
    cell_total = 0
    for row in metrics.get("images", []) or []:
        cell_total += int(row.get("cell_total", 0) or 0)
        cost_total += float(row.get("cost_total", 0.0) or 0.0)
        row_counts = row.get("critical_error_counts", {})
        if isinstance(row_counts, dict):
            for key, value in row_counts.items():
                counts[str(key)] += int(value or 0)
    critical_total = sum(counts.values())
    if not counts and cost_total == 0.0:
        return {}
    return {
        "total_cost": cost_total,
        "normalized_cost_per_cell": cost_total / cell_total if cell_total else None,
        "critical_error_total": critical_total,
        "critical_error_rate_per_cell": critical_total / cell_total if cell_total else None,
        "critical_error_counts": dict(sorted(counts.items())),
        "cost_breakdown": {},
    }


def _load_metrics(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"Cell metrics must be a JSON object: {path}")
    return data
