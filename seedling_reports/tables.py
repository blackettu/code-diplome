from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from seedling_data.post_action import read_post_action_observations, summarize_post_action_observations
from seedling_core.registry import ModelRegistry

from .registry import ArtifactRecord


def collect_artifacts(root: str | Path) -> list[Path]:
    base = Path(root)
    if base.is_file():
        return [base]
    names = {
        "dataset_audit.json",
        "raw_dataset_audit.json",
        "test_metrics.json",
        "val_metrics.json",
        "cell_metrics.json",
        "scene_metrics.json",
        "predictions.json",
        "prepare_summary.json",
        "split_summary.json",
        "rl_eval_metrics.json",
        "unified_rl_baselines.json",
        "offline_replay_eval.json",
        "critical_events.json",
        "rl_metrics.jsonl",
        "rl_metrics.csv",
        "dry_run_control_points.json",
        "dry_run_commands.json",
        "hil_pointer_report.json",
        "hil_pointer_commands.json",
        "hil_pointer_replay.json",
        "post_action_summary.json",
        "post_action_observations.jsonl",
        "error_budget.json",
        "feedback.jsonl",
        "annotation_tasks.jsonl",
        "model_registry_v0_1.yaml",
        "model_registry_v0_1.json",
    }
    return sorted(
        path
        for path in base.rglob("*")
        if path.is_file()
        and (
            path.name in names
            or path.parent.name == "replays"
            or _is_run_snapshot_file(path)
            or path.name.endswith("_commands.json")
            or path.name.endswith("_replay.json")
            or _looks_like_hardware_artifact(path)
        )
    )


def write_artifact_registry_csv(root: str | Path, output_path: str | Path) -> list[ArtifactRecord]:
    artifacts = [
        ArtifactRecord.from_path(path, artifact_type=_artifact_type(path), role="discovered")
        for path in collect_artifacts(root)
    ]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["artifact_type", "role", "path", "sha256", "bytes", "created_at"],
        )
        writer.writeheader()
        for artifact in artifacts:
            writer.writerow(
                {
                    "artifact_type": artifact.artifact_type,
                    "role": artifact.role,
                    "path": artifact.path,
                    "sha256": artifact.sha256,
                    "bytes": artifact.bytes,
                    "created_at": artifact.created_at,
                }
            )
    return artifacts


def write_dataset_summary_csv(root: str | Path, output_path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in collect_artifacts(root):
        if path.name not in {"dataset_audit.json", "raw_dataset_audit.json"}:
            continue
        data = _load_json(path)
        audit_stage = "raw" if path.name == "raw_dataset_audit.json" else "prepared"
        for split, split_data in data.get("splits", {}).items():
            rows.append(
                {
                    "artifact": str(path),
                    "audit_stage": audit_stage,
                    "dataset_root": data.get("dataset_root"),
                    "split": split,
                    "images": split_data.get("images"),
                    "labels": split_data.get("labels"),
                    "missing_labels": len(split_data.get("missing_labels", [])),
                    "empty_labels": len(split_data.get("empty_labels", [])),
                    "orphan_labels": len(split_data.get("orphan_labels", [])),
                    "suspected_augmented_count": split_data.get("suspected_augmented_count"),
                    "class_counts": json.dumps(split_data.get("class_counts", {}), ensure_ascii=False, sort_keys=True),
                }
            )
    _write_csv(output_path, rows, [
        "artifact",
        "audit_stage",
        "dataset_root",
        "split",
        "images",
        "labels",
        "missing_labels",
        "empty_labels",
        "orphan_labels",
        "suspected_augmented_count",
        "class_counts",
    ])
    return rows


def write_model_registry_csv(registry_path: str | Path | None, output_path: str | Path) -> list[dict[str, Any]]:
    if registry_path is None or not Path(registry_path).exists():
        rows: list[dict[str, Any]] = []
    else:
        registry = ModelRegistry.from_file(registry_path)
        validation = registry.validate()
        rows = [
            {
                "registry": str(registry_path),
                "registry_ok": validation["ok"],
                "model_id": model.model_id,
                "model_type": model.model_type,
                "framework": model.framework,
                "artifact_uri": model.artifact_uri,
                "config_uri": model.config_uri,
                "metrics_uri": model.metrics_uri,
                "dataset_version": model.dataset_version,
                "ontology_version": model.ontology_version,
                "input_schema": model.input_schema,
                "output_schema": model.output_schema,
                "status": model.status,
                "safety_level": model.safety_level,
                "calibration_requirements": json.dumps(
                    model.calibration_requirements,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "tags": ",".join(model.tags),
            }
            for model in registry.models
        ]
    _write_csv(output_path, rows, [
        "registry",
        "registry_ok",
        "model_id",
        "model_type",
        "framework",
        "artifact_uri",
        "config_uri",
        "metrics_uri",
        "dataset_version",
        "ontology_version",
        "input_schema",
        "output_schema",
        "status",
        "safety_level",
        "calibration_requirements",
        "tags",
    ])
    return rows


def write_object_level_results_csv(root: str | Path, output_path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in collect_artifacts(root):
        if not path.name.endswith("_metrics.json"):
            continue
        data = _load_json(path)
        if "map50" not in data and "map50_95" not in data:
            continue
        validation = data.get("validation", {})
        rows.append(
            {
                "artifact": str(path),
                "split": validation.get("split"),
                "model": validation.get("model"),
                "data": validation.get("data"),
                "map50": data.get("map50"),
                "map50_95": data.get("map50_95"),
                "precision_mean": data.get("precision_mean"),
                "recall_mean": data.get("recall_mean"),
            }
        )
    _write_csv(output_path, rows, [
        "artifact",
        "split",
        "model",
        "data",
        "map50",
        "map50_95",
        "precision_mean",
        "recall_mean",
    ])
    return rows


def write_task_level_results_csv(root: str | Path, output_path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in collect_artifacts(root):
        if path.name not in {"cell_metrics.json", "scene_metrics.json"}:
            continue
        data = _load_json(path)
        if path.name == "scene_metrics.json":
            scene_state_metrics = data
            cell_metrics = data.get("cell_metrics", {}) if isinstance(data.get("cell_metrics"), dict) else {}
            target_metrics = data.get("target_metrics", {}) if isinstance(data.get("target_metrics"), dict) else {}
            expert_keep_remove = _expert_keep_remove_metrics(target_metrics)
            cost_sensitive = data.get("cost_sensitive", {}) if isinstance(data.get("cost_sensitive"), dict) else {}
            row = {
                "artifact": str(path),
                "schema_mode": "scene_state",
                "gt_scene_id": data.get("gt_scene_id"),
                "pred_scene_id": data.get("pred_scene_id"),
                "cell_accuracy": cell_metrics.get("accuracy"),
                "cell_macro_f1": cell_metrics.get("macro", {}).get("macro_f1") if isinstance(cell_metrics.get("macro"), dict) else None,
                "edge_cell_accuracy": _nested_metric(cell_metrics, "edge_cell_metrics", "accuracy"),
                "corner_cell_accuracy": _nested_metric(cell_metrics, "corner_cell_metrics", "accuracy"),
                "multi_precision": None,
                "multi_recall": None,
                "target_precision": target_metrics.get("precision"),
                "target_recall": target_metrics.get("recall"),
                "mean_coordinate_error_px": target_metrics.get("mean_error_px"),
                "mean_coordinate_error_mm": target_metrics.get("mean_error_mm"),
                "container_recall": None,
                "expert_remove_precision": expert_keep_remove.get("precision"),
                "expert_remove_recall": expert_keep_remove.get("recall"),
                "expert_false_removal": expert_keep_remove.get("fp"),
                "expert_missed_removal": expert_keep_remove.get("fn"),
                "cost_total": cost_sensitive.get("total_cost"),
                "critical_error_total": cost_sensitive.get("critical_error_total"),
                "critical_error_rate_per_cell": cost_sensitive.get("critical_error_rate_per_cell"),
                "critical_error_counts": _json_cell(cost_sensitive.get("critical_error_counts", {})),
            }
            rows.append(row)
            continue
        multi = data.get("multi_seedling_cell", {})
        targets = data.get("removal_targets", {})
        scene_state_metrics = data.get("scene_state_metrics", {})
        if not isinstance(scene_state_metrics, dict):
            scene_state_metrics = {}
        cost_sensitive = scene_state_metrics.get("cost_sensitive", {})
        if not isinstance(cost_sensitive, dict):
            cost_sensitive = {}
        if not cost_sensitive:
            cost_sensitive = data.get("cost_sensitive", {})
            if not isinstance(cost_sensitive, dict):
                cost_sensitive = {}
        scene_target_metrics = scene_state_metrics.get("target_metrics", {})
        if not isinstance(scene_target_metrics, dict):
            scene_target_metrics = {}
        scene_cell_metrics = scene_state_metrics.get("cell_metrics", {})
        if not isinstance(scene_cell_metrics, dict):
            scene_cell_metrics = {}
        expert_keep_remove = _expert_keep_remove_metrics(scene_target_metrics)
        rows.append(
            {
                "artifact": str(path),
                "schema_mode": data.get("schema_mode") or "legacy_yolo",
                "gt_scene_id": scene_state_metrics.get("gt_scene_id"),
                "pred_scene_id": scene_state_metrics.get("pred_scene_id"),
                "cell_accuracy": data.get("cell_accuracy"),
                "cell_macro_f1": data.get("cell_macro", {}).get("macro_f1"),
                "edge_cell_accuracy": _nested_metric(scene_cell_metrics, "edge_cell_metrics", "accuracy"),
                "corner_cell_accuracy": _nested_metric(scene_cell_metrics, "corner_cell_metrics", "accuracy"),
                "multi_precision": multi.get("precision"),
                "multi_recall": multi.get("recall"),
                "target_precision": targets.get("precision"),
                "target_recall": targets.get("recall"),
                "mean_coordinate_error_px": targets.get("mean_coordinate_error_px"),
                "mean_coordinate_error_mm": targets.get("mean_coordinate_error_mm"),
                "container_recall": data.get("container_recall"),
                "expert_remove_precision": expert_keep_remove.get("precision"),
                "expert_remove_recall": expert_keep_remove.get("recall"),
                "expert_false_removal": expert_keep_remove.get("fp"),
                "expert_missed_removal": expert_keep_remove.get("fn"),
                "cost_total": cost_sensitive.get("total_cost"),
                "critical_error_total": cost_sensitive.get("critical_error_total"),
                "critical_error_rate_per_cell": cost_sensitive.get("critical_error_rate_per_cell"),
                "critical_error_counts": _json_cell(cost_sensitive.get("critical_error_counts", {})),
            }
        )
    _write_csv(output_path, rows, [
        "artifact",
        "schema_mode",
        "gt_scene_id",
        "pred_scene_id",
        "cell_accuracy",
        "cell_macro_f1",
        "edge_cell_accuracy",
        "corner_cell_accuracy",
        "multi_precision",
        "multi_recall",
        "target_precision",
        "target_recall",
        "mean_coordinate_error_px",
        "mean_coordinate_error_mm",
        "container_recall",
        "expert_remove_precision",
        "expert_remove_recall",
        "expert_false_removal",
        "expert_missed_removal",
        "cost_total",
        "critical_error_total",
        "critical_error_rate_per_cell",
        "critical_error_counts",
    ])
    return rows


def write_rl_results_csv(root: str | Path, output_path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in collect_artifacts(root):
        if path.name not in {"rl_eval_metrics.json", "unified_rl_baselines.json", "offline_replay_eval.json"}:
            continue
        data = _load_json(path)
        parent_mode = data.get("mode")
        parent_episodes = data.get("episodes")
        if isinstance(data.get("results"), list):
            for item in data["results"]:
                if not isinstance(item, dict):
                    continue
                rows.append(
                    {
                        "artifact": str(path),
                        "mode": parent_mode,
                        "policy": item.get("policy"),
                        "episodes": item.get("episodes", parent_episodes),
                        "reward_mean": item.get("reward_mean"),
                        "critical_error_rate": item.get("critical_error_rate"),
                        "successful_target_rate": item.get("successful_target_rate"),
                        "review_rate": item.get("review_rate"),
                        "actions_per_tray": item.get("actions_per_tray"),
                        "total_distance_mm": item.get("total_distance_mm"),
                        "mean_distance_error_mm": item.get("mean_distance_error_mm"),
                        "critical_events_count": item.get("critical_events_count"),
                        "allowed": item.get("allowed"),
                        "blocked": item.get("blocked"),
                        "block_rate": item.get("block_rate"),
                        "crop_damage": item.get("crop_damage"),
                    }
                )
            continue
        if "reward_mean" in data or "critical_error_rate" in data or "successful_target_rate" in data:
            rows.append(
                {
                    "artifact": str(path),
                    "mode": parent_mode,
                    "policy": data.get("policy") or data.get("checkpoint") or parent_mode or path.stem,
                    "episodes": parent_episodes,
                    "reward_mean": data.get("reward_mean"),
                    "critical_error_rate": data.get("critical_error_rate"),
                    "successful_target_rate": data.get("successful_target_rate"),
                    "review_rate": data.get("review_rate"),
                    "actions_per_tray": data.get("actions_per_tray") or data.get("action_events"),
                    "total_distance_mm": data.get("total_distance_mm"),
                    "mean_distance_error_mm": data.get("mean_distance_error_mm"),
                    "critical_events_count": data.get("critical_events_count"),
                    "allowed": data.get("allowed"),
                    "blocked": data.get("blocked"),
                    "block_rate": data.get("block_rate"),
                    "crop_damage": data.get("crop_damage"),
                }
            )
    _write_csv(output_path, rows, [
        "artifact",
        "mode",
        "policy",
        "episodes",
        "reward_mean",
        "critical_error_rate",
        "successful_target_rate",
        "review_rate",
        "actions_per_tray",
        "total_distance_mm",
        "mean_distance_error_mm",
        "critical_events_count",
        "allowed",
        "blocked",
        "block_rate",
        "crop_damage",
    ])
    return rows


def write_hardware_dry_run_results_csv(root: str | Path, output_path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in collect_artifacts(root):
        if path.suffix.lower() != ".json":
            continue
        data = _load_json(path)
        if _is_dry_run_control_point_report(data):
            rows.append(_dry_run_results_row(path, data))
        elif _is_dry_run_plan_report(data):
            rows.append(_dry_run_plan_results_row(path, data))
        elif _is_hil_pointer_report(data):
            rows.append(_hil_pointer_results_row(path, data))
    _write_csv(output_path, rows, [
        "artifact",
        "mode",
        "ok",
        "points",
        "commands",
        "executed",
        "blocked",
        "homing_success_rate",
        "move_success_rate",
        "positioning_error_p50_mm",
        "positioning_error_p95_mm",
        "positioning_error_p99_mm",
        "positioning_error_mean_mm",
        "positioning_error_max_mm",
        "repeatability_mm",
        "lost_steps_count",
        "limit_switch_events",
        "calibration_drift_mm",
        "telemetry_dropouts",
        "review_ok",
        "safety_blocks",
        "command_log",
        "replay",
    ])
    return rows


def write_safety_results_csv(root: str | Path, output_path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in collect_artifacts(root):
        if path.suffix.lower() != ".json":
            continue
        data = _load_json(path)
        if path.name == "offline_replay_eval.json" and "block_reason_counts" in data:
            rows.append(_offline_replay_safety_row(path, data))
        elif path.name == "critical_events.json" and isinstance(data.get("critical_events"), list):
            rows.append(_critical_events_safety_row(path, data))
        elif _is_dry_run_plan_report(data):
            rows.append(_dry_run_plan_safety_row(path, data))
        elif _is_hil_pointer_report(data):
            rows.append(_hil_report_safety_row(path, data))
        elif _is_replay_log(data):
            rows.append(_replay_log_safety_row(path, data))
    _write_csv(output_path, rows, [
        "artifact",
        "source",
        "events",
        "unsafe_action_attempted_count",
        "unsafe_action_blocked_count",
        "unsafe_action_escape_count",
        "foreign_object_detection_rate",
        "interlock_failure_count",
        "e_stop_test_passed",
        "operator_override_count",
        "calibration_expired_blocks",
        "real_action_guard_blocks",
        "unsupported_tool_profile_blocks",
        "aborted_runs",
        "skipped_commands",
        "review_required_count",
        "forbidden_zone_violation_count",
        "near_miss_count",
        "block_reason_counts",
    ])
    return rows


def write_annotation_feedback_results_csv(root: str | Path, output_path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in collect_artifacts(root):
        if path.name == "feedback.jsonl":
            rows.extend(_feedback_rows(path))
        elif path.name == "annotation_tasks.jsonl":
            rows.extend(_annotation_task_rows(path))
    _write_csv(output_path, rows, [
        "artifact",
        "source",
        "task_id",
        "image_id",
        "target_id",
        "object_id",
        "cell_id",
        "reason",
        "priority",
        "status",
        "operator_id",
        "created_at",
        "source_feedback_created_at",
        "comment",
        "proposed_correction",
    ])
    return rows


def write_post_action_results_csv(root: str | Path, output_path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in collect_artifacts(root):
        if path.name == "post_action_summary.json":
            data = _load_json(path)
            rows.append(_post_action_summary_row(path, data))
        elif path.name == "post_action_observations.jsonl":
            summary = summarize_post_action_observations(read_post_action_observations(path))
            rows.append(_post_action_summary_row(path, summary))
    _write_csv(output_path, rows, [
        "artifact",
        "source",
        "rows",
        "commands",
        "complete_commands",
        "incomplete_commands",
        "coverage_rate",
        "delayed_success_rate",
        "crop_damage_rate",
        "regrowth_rate",
        "uncertain_rate",
        "missing_required_observations",
        "latest_outcome_counts",
    ])
    return rows


def write_error_budget_results_csv(root: str | Path, output_path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in collect_artifacts(root):
        if path.name != "error_budget.json":
            continue
        data = _load_json(path)
        rows.append(_error_budget_row(path, data))
    _write_csv(output_path, rows, [
        "artifact",
        "schema_version",
        "ok",
        "complete",
        "total_error_mm",
        "max_total_mm",
        "missing_components",
        "e_detection",
        "e_grid",
        "e_calibration",
        "e_mechanics",
        "e_focus",
        "e_latency",
        "e_biological_target",
        "calibration_id",
        "component_source",
    ])
    return rows


def write_run_snapshot_results_csv(root: str | Path, output_path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in collect_artifacts(root):
        if not _is_run_snapshot_file(path):
            continue
        data = _load_json(path)
        rows.append(_run_snapshot_row(path, data))
    _write_csv(output_path, rows, [
        "artifact",
        "command",
        "config_hash",
        "has_config",
        "has_command_args",
        "has_environment",
        "python",
        "platform",
        "package_versions",
    ])
    return rows


def _write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        data = json.loads(line)
        if not isinstance(data, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        rows.append(data)
    return rows


def _feedback_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in _load_jsonl(path):
        rows.append(
            {
                "artifact": str(path),
                "source": "feedback",
                "task_id": None,
                "image_id": row.get("image_id"),
                "target_id": row.get("target_id"),
                "object_id": row.get("object_id"),
                "cell_id": row.get("cell_id"),
                "reason": row.get("error_type"),
                "priority": row.get("priority") or _feedback_priority(row.get("error_type")),
                "status": None,
                "operator_id": row.get("operator_id"),
                "created_at": row.get("created_at"),
                "source_feedback_created_at": None,
                "comment": row.get("comment"),
                "proposed_correction": _json_cell(row.get("proposed_correction", {})),
            }
        )
    return rows


def _annotation_task_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in _load_jsonl(path):
        rows.append(
            {
                "artifact": str(path),
                "source": "annotation_task",
                "task_id": row.get("task_id"),
                "image_id": row.get("image_id"),
                "target_id": row.get("target_id"),
                "object_id": row.get("object_id"),
                "cell_id": row.get("cell_id"),
                "reason": row.get("reason"),
                "priority": row.get("priority") or "normal",
                "status": row.get("status"),
                "operator_id": row.get("operator_id"),
                "created_at": None,
                "source_feedback_created_at": row.get("source_feedback_created_at"),
                "comment": row.get("comment"),
                "proposed_correction": _json_cell(row.get("proposed_correction", {})),
            }
        )
    return rows


def _feedback_priority(error_type: Any) -> str:
    if error_type in {"crop_damage", "unsafe_action"}:
        return "urgent"
    if error_type in {"wrong_target", "false_positive_target", "missed_target"}:
        return "high"
    return "normal"


def _expert_keep_remove_metrics(target_metrics: dict[str, Any]) -> dict[str, Any]:
    value = target_metrics.get("expert_keep_remove")
    return value if isinstance(value, dict) else {}


def _nested_metric(data: dict[str, Any], section: str, metric: str) -> Any:
    value = data.get(section)
    if not isinstance(value, dict):
        return None
    return value.get(metric)


def _json_cell(value: Any) -> str:
    if value is None or value == "":
        value = {}
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _artifact_type(path: Path) -> str:
    if _is_run_snapshot_file(path):
        return "run_snapshot"
    return path.name.removesuffix(path.suffix)


def _is_run_snapshot_file(path: Path) -> bool:
    return path.name == "run_snapshot.json" or path.name.endswith(".run_snapshot.json")


def _looks_like_hardware_artifact(path: Path) -> bool:
    if path.suffix.lower() != ".json":
        return False
    stem = path.stem.lower()
    if "dry_run" not in stem and "hil_pointer" not in stem:
        return False
    try:
        data = _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return _is_dry_run_control_point_report(data) or _is_dry_run_plan_report(data) or _is_hil_pointer_report(data)


def _is_dry_run_control_point_report(data: dict[str, Any]) -> bool:
    return isinstance(data.get("points"), list) and isinstance(data.get("error_summary_mm"), dict)


def _is_dry_run_plan_report(data: dict[str, Any]) -> bool:
    return data.get("mode") == "dry_run_pointer" and "commands" in data and "executed" in data and isinstance(data.get("results"), list)


def _is_hil_pointer_report(data: dict[str, Any]) -> bool:
    return data.get("mode") == "hardware_in_loop_pointer" and "commands" in data and "executed" in data


def _is_replay_log(data: dict[str, Any]) -> bool:
    return isinstance(data.get("steps"), list) and "replay_id" in data and "scene_id" in data


def _dry_run_results_row(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    points = [item for item in data.get("points", []) if isinstance(item, dict)]
    errors = [float(item["error_mm"]) for item in points if item.get("error_mm") is not None]
    ok_points = sum(1 for item in points if item.get("ok") is True)
    summary = data.get("error_summary_mm", {})
    return {
        "artifact": str(path),
        "mode": "dry_run_pointer",
        "ok": data.get("ok"),
        "points": len(points),
        "commands": len(points),
        "executed": ok_points,
        "blocked": sum(1 for item in points if item.get("ok") is False),
        "homing_success_rate": 1.0,
        "move_success_rate": _safe_ratio(ok_points, len(points)),
        "positioning_error_p50_mm": _percentile(errors, 50),
        "positioning_error_p95_mm": _percentile(errors, 95),
        "positioning_error_p99_mm": _percentile(errors, 99),
        "positioning_error_mean_mm": summary.get("mean") if isinstance(summary, dict) else None,
        "positioning_error_max_mm": summary.get("max") if isinstance(summary, dict) else None,
        "repeatability_mm": _repeatability_mm(points),
        "lost_steps_count": data.get("lost_steps_count"),
        "limit_switch_events": data.get("limit_switch_events"),
        "calibration_drift_mm": data.get("calibration_drift_mm"),
        "telemetry_dropouts": data.get("telemetry_dropouts"),
        "review_ok": None,
        "safety_blocks": 0,
        "command_log": data.get("command_log"),
        "replay": None,
    }


def _dry_run_plan_results_row(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    commands = int(data.get("commands") or 0)
    executed = int(data.get("executed") or 0)
    blocked = int(data.get("blocked") or 0)
    results = [item for item in data.get("results", []) if isinstance(item, dict)]
    return {
        "artifact": str(path),
        "mode": data.get("mode"),
        "ok": data.get("ok"),
        "points": None,
        "commands": commands,
        "executed": executed,
        "blocked": blocked,
        "homing_success_rate": 1.0,
        "move_success_rate": _safe_ratio(executed, commands),
        "positioning_error_p50_mm": None,
        "positioning_error_p95_mm": None,
        "positioning_error_p99_mm": None,
        "positioning_error_mean_mm": None,
        "positioning_error_max_mm": None,
        "repeatability_mm": None,
        "lost_steps_count": data.get("lost_steps_count"),
        "limit_switch_events": data.get("limit_switch_events"),
        "calibration_drift_mm": data.get("calibration_drift_mm"),
        "telemetry_dropouts": data.get("telemetry_dropouts"),
        "review_ok": None,
        "safety_blocks": sum(1 for result in results if not result.get("ok")),
        "command_log": data.get("command_log"),
        "replay": data.get("replay"),
    }


def _hil_pointer_results_row(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    commands = int(data.get("commands") or 0)
    executed = int(data.get("executed") or 0)
    blocked = int(data.get("blocked") or 0)
    review = data.get("review", {})
    results = [item for item in data.get("results", []) if isinstance(item, dict)]
    return {
        "artifact": str(path),
        "mode": data.get("mode"),
        "ok": data.get("ok"),
        "points": None,
        "commands": commands,
        "executed": executed,
        "blocked": blocked,
        "homing_success_rate": 1.0,
        "move_success_rate": _safe_ratio(executed, commands),
        "positioning_error_p50_mm": None,
        "positioning_error_p95_mm": None,
        "positioning_error_p99_mm": None,
        "positioning_error_mean_mm": None,
        "positioning_error_max_mm": None,
        "repeatability_mm": None,
        "lost_steps_count": data.get("lost_steps_count"),
        "limit_switch_events": data.get("limit_switch_events"),
        "calibration_drift_mm": data.get("calibration_drift_mm"),
        "telemetry_dropouts": data.get("telemetry_dropouts"),
        "review_ok": review.get("ok") if isinstance(review, dict) else None,
        "safety_blocks": sum(1 for result in results if not result.get("ok")),
        "command_log": data.get("command_log"),
        "replay": data.get("replay"),
    }


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _repeatability_mm(points: list[dict[str, Any]]) -> float | None:
    by_point: dict[str, list[list[float]]] = {}
    for point in points:
        actual = point.get("actual_mm")
        if not isinstance(actual, list) or len(actual) != 3:
            continue
        by_point.setdefault(str(point.get("point_id")), []).append([float(value) for value in actual])
    spreads = [_max_pairwise_distance(group) for group in by_point.values() if len(group) > 1]
    return max(spreads) if spreads else None


def _max_pairwise_distance(points: list[list[float]]) -> float:
    spread = 0.0
    for index, left in enumerate(points):
        for right in points[index + 1:]:
            spread = max(spread, math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right))))
    return spread


def _post_action_summary_row(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    missing = data.get("missing_required_observations", [])
    latest_counts = data.get("latest_outcome_counts", {})
    return {
        "artifact": str(path),
        "source": data.get("source") or path.name,
        "rows": data.get("rows"),
        "commands": data.get("commands"),
        "complete_commands": data.get("complete_commands"),
        "incomplete_commands": data.get("incomplete_commands"),
        "coverage_rate": data.get("coverage_rate"),
        "delayed_success_rate": data.get("delayed_success_rate"),
        "crop_damage_rate": data.get("crop_damage_rate"),
        "regrowth_rate": data.get("regrowth_rate"),
        "uncertain_rate": data.get("uncertain_rate"),
        "missing_required_observations": len(missing) if isinstance(missing, list) else missing,
        "latest_outcome_counts": json.dumps(latest_counts, ensure_ascii=False, sort_keys=True)
        if isinstance(latest_counts, dict)
        else latest_counts,
    }


def _error_budget_row(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    components = data.get("components_mm") if isinstance(data.get("components_mm"), dict) else {}
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    missing = data.get("missing_components", [])
    return {
        "artifact": str(path),
        "schema_version": data.get("schema_version"),
        "ok": data.get("ok"),
        "complete": data.get("complete"),
        "total_error_mm": data.get("total_error_mm"),
        "max_total_mm": data.get("max_total_mm"),
        "missing_components": json.dumps(missing, ensure_ascii=False, sort_keys=True) if isinstance(missing, list) else missing,
        "e_detection": components.get("e_detection"),
        "e_grid": components.get("e_grid"),
        "e_calibration": components.get("e_calibration"),
        "e_mechanics": components.get("e_mechanics"),
        "e_focus": components.get("e_focus"),
        "e_latency": components.get("e_latency"),
        "e_biological_target": components.get("e_biological_target"),
        "calibration_id": metadata.get("calibration_id"),
        "component_source": metadata.get("e_calibration_source"),
    }


def _run_snapshot_row(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    environment = data.get("environment") if isinstance(data.get("environment"), dict) else {}
    packages = environment.get("packages") if isinstance(environment.get("packages"), dict) else {}
    command_args = data.get("command_args")
    return {
        "artifact": str(path),
        "command": data.get("command"),
        "config_hash": data.get("config_hash"),
        "has_config": isinstance(data.get("config"), dict),
        "has_command_args": isinstance(command_args, dict) and bool(command_args),
        "has_environment": bool(environment),
        "python": environment.get("python"),
        "platform": environment.get("platform"),
        "package_versions": json.dumps(packages, ensure_ascii=False, sort_keys=True),
    }


def _offline_replay_safety_row(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    reason_counts = Counter({str(key): int(value) for key, value in data.get("block_reason_counts", {}).items()})
    events = int(data.get("action_events") or data.get("robot_execution_events") or 0)
    blocked = int(data.get("blocked") or 0)
    return _safety_row(
        path=path,
        source="offline_replay_eval",
        events=events,
        attempted=blocked,
        blocked=blocked,
        escapes=int(data.get("crop_damage") or 0),
        reason_counts=reason_counts,
        operator_override_count=data.get("operator_override_count"),
        near_miss_count=data.get("near_miss_count"),
        review_required_count=int(data.get("reviewed_targets") or 0) + _review_required_count(reason_counts),
    )


def _critical_events_safety_row(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    events = [item for item in data.get("critical_events", []) if isinstance(item, dict)]
    reason_counts: Counter[str] = Counter()
    escapes = 0
    for event in events:
        info = event.get("info") if isinstance(event.get("info"), dict) else {}
        event_name = str(event.get("event") or info.get("event") or "critical_event")
        reason_counts[event_name] += 1
        if isinstance(info.get("reasons"), list):
            reason_counts.update(str(reason) for reason in info["reasons"])
        if _event_has_crop_damage(event):
            escapes += 1
    blocked = _blocked_count(reason_counts)
    return _safety_row(
        path=path,
        source="critical_events",
        events=len(events),
        attempted=len(events),
        blocked=blocked,
        escapes=escapes,
        reason_counts=reason_counts,
    )


def _execution_results_safety_row(path: Path, source: str, raw_results: Any) -> dict[str, Any]:
    results = [item for item in raw_results if isinstance(item, dict)] if isinstance(raw_results, list) else []
    reason_counts: Counter[str] = Counter()
    blocked = 0
    for result in results:
        decision = result.get("safety_decision") if isinstance(result.get("safety_decision"), dict) else {}
        allowed = decision.get("allowed")
        if allowed is False or result.get("ok") is False:
            blocked += 1
            reason_counts.update(_reasons_from_decision_payload(decision))
            if allowed is not False:
                reason_counts.update(_adapter_block_reasons(result))
    return _safety_row(
        path=path,
        source=source,
        events=len(results),
        attempted=len(results),
        blocked=blocked,
        escapes=0,
        reason_counts=reason_counts,
    )


def _hil_report_safety_row(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    row = _execution_results_safety_row(path, "hardware_in_loop_pointer", data.get("results", []))
    review = data.get("review") if isinstance(data.get("review"), dict) else {}
    row["e_stop_test_passed"] = review.get("emergency_stop_tested")
    row["aborted_runs"] = 1 if data.get("aborted") is True else 0
    row["skipped_commands"] = int(data.get("skipped") or 0)
    return row


def _dry_run_plan_safety_row(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    row = _execution_results_safety_row(path, "dry_run_pointer", data.get("results", []))
    row["aborted_runs"] = 1 if data.get("aborted") is True else 0
    row["skipped_commands"] = int(data.get("skipped") or 0)
    return row


def _replay_log_safety_row(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    steps = [item for item in data.get("steps", []) if isinstance(item, dict)]
    reason_counts: Counter[str] = Counter()
    events = 0
    attempted = 0
    blocked = 0
    escapes = 0
    review_required = 0
    for step in steps:
        payload = step.get("payload") if isinstance(step.get("payload"), dict) else {}
        event_type = str(step.get("event_type") or "")
        info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
        event = str(payload.get("event") or info.get("event") or event_type)
        if event_type == "robot_execution":
            events += 1
            attempted += 1
            decision = payload.get("safety_decision") if isinstance(payload.get("safety_decision"), dict) else {}
            if decision.get("allowed") is False or payload.get("ok") is False:
                blocked += 1
                reason_counts.update(_reasons_from_decision_payload(decision))
        elif event_type == "rl_step":
            events += 1
            if event in {"unsafe_target", "invalid_target_index", "invalid_action"}:
                attempted += 1
                reason_counts[event] += 1
            if event == "review":
                review_required += int(info.get("reviewed_targets", payload.get("reviewed_targets", 0)) or 0)
            if _event_has_crop_damage(payload):
                escapes += 1
    return _safety_row(
        path=path,
        source="replay_log",
        events=events,
        attempted=attempted,
        blocked=blocked,
        escapes=escapes,
        reason_counts=reason_counts,
        review_required_count=review_required + _review_required_count(reason_counts),
    )


def _safety_row(
    *,
    path: Path,
    source: str,
    events: int,
    attempted: int,
    blocked: int,
    escapes: int,
    reason_counts: Counter[str],
    e_stop_test_passed: bool | None = None,
    operator_override_count: Any = None,
    near_miss_count: Any = None,
    review_required_count: int | None = None,
    aborted_runs: Any = 0,
    skipped_commands: Any = 0,
) -> dict[str, Any]:
    return {
        "artifact": str(path),
        "source": source,
        "events": events,
        "unsafe_action_attempted_count": attempted,
        "unsafe_action_blocked_count": blocked,
        "unsafe_action_escape_count": escapes,
        "foreign_object_detection_rate": _safe_ratio(_count_any(reason_counts, {"foreign_object_present"}), events),
        "interlock_failure_count": _count_any(reason_counts, {"BLOCK_INTERLOCK", "interlock_false", "limit_switch_not_ok"}),
        "e_stop_test_passed": e_stop_test_passed,
        "operator_override_count": operator_override_count,
        "calibration_expired_blocks": _count_any(
            reason_counts,
            {"BLOCK_CALIBRATION", "BLOCK_CALIBRATION_REQUIRED", "calibration_required", "robot_not_homed", "zone_error_too_high"},
        ),
        "real_action_guard_blocks": _count_any(
            reason_counts,
            {"real_action_not_allowed", "software_safe_mode_disabled", "enclosure_not_closed"},
        ),
        "unsupported_tool_profile_blocks": _count_any(reason_counts, {"unsupported_tool_profile"}),
        "aborted_runs": aborted_runs,
        "skipped_commands": skipped_commands,
        "review_required_count": review_required_count if review_required_count is not None else _review_required_count(reason_counts),
        "forbidden_zone_violation_count": _count_any(
            reason_counts,
            {"forbidden_zone_overlap", "target_outside_tray"},
        ),
        "near_miss_count": near_miss_count,
        "block_reason_counts": json.dumps(dict(sorted(reason_counts.items())), ensure_ascii=False, sort_keys=True),
    }


def _reasons_from_decision_payload(decision: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    raw_reasons = decision.get("reasons")
    if isinstance(raw_reasons, list):
        reasons.extend(str(reason) for reason in raw_reasons)
    elif raw_reasons:
        reasons.append(str(raw_reasons))
    if decision.get("result"):
        reasons.append(str(decision["result"]))
    return sorted(dict.fromkeys(reasons))


def _adapter_block_reasons(result: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    outcome = result.get("outcome") if isinstance(result.get("outcome"), dict) else {}
    tool_result = outcome.get("tool_result") if isinstance(outcome.get("tool_result"), dict) else {}
    profile = tool_result.get("profile") if isinstance(tool_result.get("profile"), dict) else {}
    message = str(result.get("message") or tool_result.get("message") or "")
    if profile.get("profile_id") and profile.get("profile_id") != "pointer_only":
        reasons.append("unsupported_tool_profile")
    elif "only allows pointer_only" in message:
        reasons.append("unsupported_tool_profile")
    return reasons


def _event_has_crop_damage(event: dict[str, Any]) -> bool:
    info = event.get("info") if isinstance(event.get("info"), dict) else event
    outcome = info.get("outcome") if isinstance(info.get("outcome"), dict) else {}
    return bool(outcome.get("crop_damage"))


def _blocked_count(reason_counts: Counter[str]) -> int:
    return sum(count for reason, count in reason_counts.items() if reason.startswith("BLOCK_"))


def _review_required_count(reason_counts: Counter[str]) -> int:
    return _count_any(reason_counts, {"BLOCK_REVIEW", "BLOCK_REVIEW_REQUIRED", "operator_confirmation_required"})


def _count_any(reason_counts: Counter[str], names: set[str]) -> int:
    return sum(reason_counts.get(name, 0) for name in names)
