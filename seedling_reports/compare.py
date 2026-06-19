from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .html import write_experiment_html_report


def compare_runs(
    run_dirs: list[str | Path],
    output_dir: str | Path,
    thresholds: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    thresholds = {key: value for key, value in (thresholds or {}).items() if value is not None}
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        root = Path(run_dir)
        row: dict[str, Any] = {"run": str(root)}
        row.update(_first_csv_metrics(root / "object_level_results.csv", ["map50", "map50_95", "precision_mean", "recall_mean"]))
        row.update(
            _first_csv_metrics(
                root / "task_level_results.csv",
                [
                    "cell_accuracy",
                    "cell_macro_f1",
                    "edge_cell_accuracy",
                    "corner_cell_accuracy",
                    "multi_precision",
                    "multi_recall",
                    "target_precision",
                    "target_recall",
                    "container_recall",
                ],
            )
        )
        row.update(_run_snapshot_metrics(root))
        row.update(_rl_metrics(root))
        row.update(_rl_stability_metrics(root))
        row.update(_hardware_metrics(root))
        row.update(_safety_metrics(root))
        row.update(_post_action_metrics(root))
        row.update(_error_budget_metrics(root))
        row.update(_threshold_status(row, thresholds))
        rows.append(row)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "compare_runs.csv", rows)
    write_experiment_html_report("Run Comparison", output / "compare_runs.html", [("Run Comparison", rows)])
    summary = {
        "ok": True,
        "runs": len(rows),
        "thresholds": thresholds,
        "passed": sum(1 for row in rows if row.get("threshold_status") == "pass"),
        "failed": sum(1 for row in rows if row.get("threshold_status") == "fail"),
    }
    (output / "compare_runs_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return rows


def _first_csv_metrics(path: Path, columns: list[str]) -> dict[str, Any]:
    if not path.exists():
        return {column: None for column in columns}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        first = next(reader, None)
    if first is None:
        return {column: None for column in columns}
    return {column: _coerce(first.get(column)) for column in columns}


def _rl_metrics(root: Path) -> dict[str, Any]:
    columns = [
        "rl_reward_mean",
        "rl_critical_error_rate",
        "rl_successful_target_rate",
        "rl_critical_events_count",
        "rl_allowed_actions",
        "rl_blocked_actions",
        "rl_block_rate",
        "rl_crop_damage_count",
    ]
    table_path = root / "rl_results.csv"
    if table_path.exists():
        data = _first_csv_metrics(
            table_path,
            [
                "reward_mean",
                "critical_error_rate",
                "successful_target_rate",
                "critical_events_count",
                "allowed",
                "blocked",
                "block_rate",
                "crop_damage",
            ],
        )
        return {
            "rl_reward_mean": data.get("reward_mean"),
            "rl_critical_error_rate": data.get("critical_error_rate"),
            "rl_successful_target_rate": data.get("successful_target_rate"),
            "rl_critical_events_count": data.get("critical_events_count"),
            "rl_allowed_actions": data.get("allowed"),
            "rl_blocked_actions": data.get("blocked"),
            "rl_block_rate": data.get("block_rate"),
            "rl_crop_damage_count": data.get("crop_damage"),
        }
    for name in ("rl_eval_metrics.json", "unified_rl_baselines.json", "offline_replay_eval.json"):
        path = root / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            continue
        metrics = _rl_payload_metrics(data)
        if metrics is not None:
            return metrics
    return {column: None for column in columns}


def _rl_payload_metrics(data: dict[str, Any]) -> dict[str, Any] | None:
    item = data
    if isinstance(data.get("results"), list):
        item = next((row for row in data["results"] if isinstance(row, dict)), {})
    metric_keys = {
        "reward_mean",
        "critical_error_rate",
        "successful_target_rate",
        "critical_events_count",
        "allowed",
        "blocked",
        "block_rate",
        "crop_damage",
    }
    if not any(key in item for key in metric_keys):
        return None
    return {
        "rl_reward_mean": _coerce(item.get("reward_mean")),
        "rl_critical_error_rate": _coerce(item.get("critical_error_rate")),
        "rl_successful_target_rate": _coerce(item.get("successful_target_rate")),
        "rl_critical_events_count": _coerce(item.get("critical_events_count")),
        "rl_allowed_actions": _coerce(item.get("allowed")),
        "rl_blocked_actions": _coerce(item.get("blocked")),
        "rl_block_rate": _coerce(item.get("block_rate")),
        "rl_crop_damage_count": _coerce(item.get("crop_damage")),
    }


def _run_snapshot_metrics(root: Path) -> dict[str, Any]:
    columns = {
        "run_snapshot_count": None,
        "run_snapshot_with_config_hash_count": None,
        "run_snapshot_missing_config_hash_count": None,
        "run_snapshot_with_command_args_count": None,
        "run_snapshot_missing_command_args_count": None,
    }
    rows = _csv_rows(root / "run_snapshot_results.csv")
    if not rows:
        rows = [
            {
                "config_hash": data.get("config_hash"),
                "has_command_args": isinstance(data.get("command_args"), dict) and bool(data.get("command_args")),
            }
            for data in _run_snapshot_payloads(root)
        ]
    if not rows:
        return columns
    with_hash = sum(1 for row in rows if row.get("config_hash"))
    with_command_args = sum(1 for row in rows if _truthy(row.get("has_command_args")))
    return {
        "run_snapshot_count": float(len(rows)),
        "run_snapshot_with_config_hash_count": float(with_hash),
        "run_snapshot_missing_config_hash_count": float(len(rows) - with_hash),
        "run_snapshot_with_command_args_count": float(with_command_args),
        "run_snapshot_missing_command_args_count": float(len(rows) - with_command_args),
    }


def _rl_stability_metrics(root: Path) -> dict[str, Any]:
    columns = {
        "rl_sweep_group_count": None,
        "rl_seed_count": None,
        "rl_sweep_group_id": None,
        "rl_reward_mean_stability_mean": None,
        "rl_reward_mean_stability_std": None,
        "rl_critical_error_rate_stability_max": None,
        "rl_successful_target_rate_stability_mean": None,
        "rl_stability_metrics_path": None,
    }
    path = _find_stability_summary(root)
    if path is None:
        return columns
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    groups = data.get("groups", []) if isinstance(data, dict) else []
    if not groups:
        return {**columns, "rl_stability_metrics_path": str(path)}
    best = max(groups, key=lambda group: _metric_stat(group, "reward_mean", "mean") or float("-inf"))
    return {
        "rl_sweep_group_count": len(groups),
        "rl_seed_count": best.get("seed_count"),
        "rl_sweep_group_id": best.get("group_id"),
        "rl_reward_mean_stability_mean": _metric_stat(best, "reward_mean", "mean"),
        "rl_reward_mean_stability_std": _metric_stat(best, "reward_mean", "std"),
        "rl_critical_error_rate_stability_max": _metric_stat(best, "critical_error_rate", "max"),
        "rl_successful_target_rate_stability_mean": _metric_stat(best, "successful_target_rate", "mean"),
        "rl_stability_metrics_path": str(path),
    }


def _find_stability_summary(root: Path) -> Path | None:
    candidates = [
        root / "sweep_stability_summary.json",
        root / "stability_report" / "sweep_stability_summary.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _hardware_metrics(root: Path) -> dict[str, Any]:
    columns = {
        "hardware_modes": None,
        "hardware_move_success_rate_min": None,
        "hardware_positioning_error_p95_mm_max": None,
        "hardware_safety_blocks": None,
    }
    rows = _csv_rows(root / "hardware_dry_run_results.csv")
    if not rows:
        rows = _direct_hardware_rows(root)
    if not rows:
        return columns
    move_rates = [_coerce(row.get("move_success_rate")) for row in rows]
    p95_errors = [_coerce(row.get("positioning_error_p95_mm")) for row in rows]
    safety_blocks = [_coerce(row.get("safety_blocks")) for row in rows]
    modes = sorted({str(row.get("mode")) for row in rows if row.get("mode")})
    return {
        "hardware_modes": ",".join(modes) if modes else None,
        "hardware_move_success_rate_min": _min_present(move_rates),
        "hardware_positioning_error_p95_mm_max": _max_present(p95_errors),
        "hardware_safety_blocks": _sum_present(safety_blocks),
    }


def _safety_metrics(root: Path) -> dict[str, Any]:
    columns = {
        "safety_unsafe_action_attempted_count": None,
        "safety_unsafe_action_blocked_count": None,
        "safety_unsafe_action_escape_count": None,
        "safety_interlock_failure_count": None,
        "safety_calibration_expired_blocks": None,
        "safety_real_action_guard_blocks": None,
        "safety_unsupported_tool_profile_blocks": None,
        "safety_aborted_runs": None,
        "safety_skipped_commands": None,
        "safety_review_required_count": None,
        "safety_forbidden_zone_violation_count": None,
    }
    rows = _csv_rows(root / "safety_results.csv")
    if not rows:
        rows = _direct_safety_rows(root)
    if not rows:
        return columns
    return {
        "safety_unsafe_action_attempted_count": _sum_column(rows, "unsafe_action_attempted_count"),
        "safety_unsafe_action_blocked_count": _sum_column(rows, "unsafe_action_blocked_count"),
        "safety_unsafe_action_escape_count": _sum_column(rows, "unsafe_action_escape_count"),
        "safety_interlock_failure_count": _sum_column(rows, "interlock_failure_count"),
        "safety_calibration_expired_blocks": _sum_column(rows, "calibration_expired_blocks"),
        "safety_real_action_guard_blocks": _sum_column(rows, "real_action_guard_blocks"),
        "safety_unsupported_tool_profile_blocks": _sum_column(rows, "unsupported_tool_profile_blocks"),
        "safety_aborted_runs": _sum_column(rows, "aborted_runs"),
        "safety_skipped_commands": _sum_column(rows, "skipped_commands"),
        "safety_review_required_count": _sum_column(rows, "review_required_count"),
        "safety_forbidden_zone_violation_count": _sum_column(rows, "forbidden_zone_violation_count"),
    }


def _post_action_metrics(root: Path) -> dict[str, Any]:
    columns = {
        "post_action_coverage_rate": None,
        "post_action_delayed_success_rate": None,
        "post_action_crop_damage_rate": None,
        "post_action_regrowth_rate": None,
        "post_action_incomplete_commands": None,
    }
    data = _first_csv_metrics(
        root / "post_action_results.csv",
        [
            "coverage_rate",
            "delayed_success_rate",
            "crop_damage_rate",
            "regrowth_rate",
            "incomplete_commands",
        ],
    )
    if not data:
        return columns
    return {
        "post_action_coverage_rate": data.get("coverage_rate"),
        "post_action_delayed_success_rate": data.get("delayed_success_rate"),
        "post_action_crop_damage_rate": data.get("crop_damage_rate"),
        "post_action_regrowth_rate": data.get("regrowth_rate"),
        "post_action_incomplete_commands": data.get("incomplete_commands"),
    }


def _error_budget_metrics(root: Path) -> dict[str, Any]:
    columns = {
        "error_budget_total_mm_max": None,
        "error_budget_complete_count": None,
        "error_budget_incomplete_count": None,
        "error_budget_ok_count": None,
    }
    rows = _csv_rows(root / "error_budget_results.csv")
    if not rows:
        rows = [
            {
                "complete": data.get("complete"),
                "ok": data.get("ok"),
                "total_error_mm": data.get("total_error_mm"),
            }
            for data in _json_artifacts(root, "error_budget.json")
        ]
    if not rows:
        return columns
    totals = [_coerce(row.get("total_error_mm")) for row in rows]
    complete_count = sum(1 for row in rows if _truthy(row.get("complete")))
    ok_count = sum(1 for row in rows if _truthy(row.get("ok")))
    return {
        "error_budget_total_mm_max": _max_present(totals),
        "error_budget_complete_count": float(complete_count),
        "error_budget_incomplete_count": float(len(rows) - complete_count),
        "error_budget_ok_count": float(ok_count),
    }


def _metric_stat(group: dict[str, Any], metric: str, stat: str) -> Any:
    metrics = group.get("metrics", {})
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(metric, {})
    if not isinstance(value, dict):
        return None
    return _coerce(value.get(stat))


def _threshold_status(row: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    if not thresholds:
        return {}
    rules = {
        "min_cell_accuracy": ("cell_accuracy", ">="),
        "min_edge_cell_accuracy": ("edge_cell_accuracy", ">="),
        "min_corner_cell_accuracy": ("corner_cell_accuracy", ">="),
        "min_target_recall": ("target_recall", ">="),
        "min_rl_successful_target_rate": ("rl_successful_target_rate", ">="),
        "max_rl_critical_error_rate": ("rl_critical_error_rate", "<="),
        "max_rl_critical_events_count": ("rl_critical_events_count", "<="),
        "max_rl_blocked_actions": ("rl_blocked_actions", "<="),
        "max_rl_block_rate": ("rl_block_rate", "<="),
        "max_rl_crop_damage_count": ("rl_crop_damage_count", "<="),
        "max_rl_reward_std": ("rl_reward_mean_stability_std", "<="),
        "min_rl_seed_count": ("rl_seed_count", ">="),
        "min_hardware_move_success_rate": ("hardware_move_success_rate_min", ">="),
        "max_hardware_positioning_error_p95_mm": ("hardware_positioning_error_p95_mm_max", "<="),
        "max_safety_unsafe_action_escape_count": ("safety_unsafe_action_escape_count", "<="),
        "max_safety_real_action_guard_blocks": ("safety_real_action_guard_blocks", "<="),
        "max_safety_unsupported_tool_profile_blocks": ("safety_unsupported_tool_profile_blocks", "<="),
        "max_safety_aborted_runs": ("safety_aborted_runs", "<="),
        "max_safety_skipped_commands": ("safety_skipped_commands", "<="),
        "max_safety_forbidden_zone_violation_count": ("safety_forbidden_zone_violation_count", "<="),
        "min_post_action_coverage_rate": ("post_action_coverage_rate", ">="),
        "min_post_action_delayed_success_rate": ("post_action_delayed_success_rate", ">="),
        "max_post_action_crop_damage_rate": ("post_action_crop_damage_rate", "<="),
        "max_post_action_regrowth_rate": ("post_action_regrowth_rate", "<="),
        "max_error_budget_total_mm": ("error_budget_total_mm_max", "<="),
        "require_error_budget_complete": ("error_budget_incomplete_count", "<="),
        "require_run_snapshots": ("run_snapshot_count", ">="),
        "max_run_snapshot_missing_config_hash": ("run_snapshot_missing_config_hash_count", "<="),
        "max_run_snapshot_missing_command_args": ("run_snapshot_missing_command_args_count", "<="),
    }
    failures: list[str] = []
    for threshold_name, expected in thresholds.items():
        if threshold_name not in rules:
            continue
        column, operator = rules[threshold_name]
        actual = _coerce(row.get(column))
        if actual is None:
            failures.append(f"{column}:missing")
            continue
        if operator == ">=" and float(actual) < float(expected):
            failures.append(f"{column}:{actual}<{expected}")
        elif operator == "<=" and float(actual) > float(expected):
            failures.append(f"{column}:{actual}>{expected}")
    return {
        "threshold_status": "fail" if failures else "pass",
        "threshold_failures": ";".join(failures),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _json_artifacts(root: Path, name: str) -> list[dict[str, Any]]:
    if root.is_file():
        paths = [root] if root.name == name else []
    else:
        paths = sorted(root.rglob(name)) if root.exists() else []
    payloads: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            payloads.append(data)
    return payloads


def _run_snapshot_payloads(root: Path) -> list[dict[str, Any]]:
    if root.is_file():
        paths = [root] if _is_run_snapshot_file(root) else []
    else:
        paths = sorted(path for path in root.rglob("*.json") if _is_run_snapshot_file(path)) if root.exists() else []
    payloads: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            payloads.append(data)
    return payloads


def _is_run_snapshot_file(path: Path) -> bool:
    return path.name == "run_snapshot.json" or path.name.endswith(".run_snapshot.json")


def _direct_hardware_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for data in _direct_hardware_payloads(root):
        if _is_dry_run_control_point_report(data):
            rows.append(_direct_control_point_hardware_row(data))
        elif _is_pointer_plan_report(data):
            rows.append(_direct_pointer_plan_hardware_row(data))
    return rows


def _direct_safety_rows(root: Path) -> list[dict[str, Any]]:
    return [
        _direct_pointer_plan_safety_row(data)
        for data in _direct_hardware_payloads(root)
        if _is_pointer_plan_report(data)
    ]


def _direct_hardware_payloads(root: Path) -> list[dict[str, Any]]:
    if root.is_file():
        paths = [root]
    else:
        names = {"dry_run_control_points.json", "dry_run_plan_report.json", "hil_pointer_report.json"}
        paths = sorted(
            path
            for path in root.rglob("*.json")
            if path.name in names or "dry_run" in path.stem.lower() or "hil_pointer" in path.stem.lower()
        ) if root.exists() else []
    payloads: list[dict[str, Any]] = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and (_is_dry_run_control_point_report(data) or _is_pointer_plan_report(data)):
            payloads.append(data)
    return payloads


def _is_dry_run_control_point_report(data: dict[str, Any]) -> bool:
    return isinstance(data.get("points"), list) and isinstance(data.get("error_summary_mm"), dict)


def _is_pointer_plan_report(data: dict[str, Any]) -> bool:
    return data.get("mode") in {"dry_run_pointer", "hardware_in_loop_pointer"} and "commands" in data and "executed" in data and isinstance(data.get("results"), list)


def _direct_control_point_hardware_row(data: dict[str, Any]) -> dict[str, Any]:
    points = [item for item in data.get("points", []) if isinstance(item, dict)]
    ok_points = sum(1 for item in points if item.get("ok") is True)
    errors = [_coerce(item.get("error_mm")) for item in points]
    return {
        "mode": "dry_run_pointer",
        "move_success_rate": _safe_ratio(ok_points, len(points)),
        "positioning_error_p95_mm": _percentile([float(value) for value in errors if isinstance(value, (int, float))], 95),
        "safety_blocks": 0,
    }


def _direct_pointer_plan_hardware_row(data: dict[str, Any]) -> dict[str, Any]:
    commands = int(data.get("commands") or 0)
    executed = int(data.get("executed") or 0)
    results = [item for item in data.get("results", []) if isinstance(item, dict)]
    return {
        "mode": data.get("mode"),
        "move_success_rate": _safe_ratio(executed, commands),
        "positioning_error_p95_mm": None,
        "safety_blocks": sum(1 for result in results if not result.get("ok")),
    }


def _direct_pointer_plan_safety_row(data: dict[str, Any]) -> dict[str, Any]:
    results = [item for item in data.get("results", []) if isinstance(item, dict)]
    reason_counts: Counter[str] = Counter()
    blocked = 0
    for result in results:
        decision = result.get("safety_decision") if isinstance(result.get("safety_decision"), dict) else {}
        allowed = decision.get("allowed")
        if allowed is False or result.get("ok") is False:
            blocked += 1
            reason_counts.update(_reasons_from_decision(decision))
            if allowed is not False:
                reason_counts.update(_adapter_block_reasons(result))
    return {
        "unsafe_action_attempted_count": len(results),
        "unsafe_action_blocked_count": blocked,
        "unsafe_action_escape_count": 0,
        "interlock_failure_count": _count_any(reason_counts, {"BLOCK_INTERLOCK", "interlock_false", "limit_switch_not_ok"}),
        "calibration_expired_blocks": _count_any(
            reason_counts,
            {"BLOCK_CALIBRATION", "BLOCK_CALIBRATION_REQUIRED", "calibration_required", "robot_not_homed", "zone_error_too_high"},
        ),
        "real_action_guard_blocks": _count_any(
            reason_counts,
            {"real_action_not_allowed", "software_safe_mode_disabled", "enclosure_not_closed"},
        ),
        "unsupported_tool_profile_blocks": _count_any(reason_counts, {"unsupported_tool_profile"}),
        "aborted_runs": 1 if data.get("aborted") is True else 0,
        "skipped_commands": int(data.get("skipped") or 0),
        "review_required_count": _count_any(reason_counts, {"BLOCK_REVIEW", "BLOCK_REVIEW_REQUIRED", "operator_confirmation_required"}),
        "forbidden_zone_violation_count": _count_any(reason_counts, {"forbidden_zone_overlap", "target_outside_tray"}),
    }


def _reasons_from_decision(decision: dict[str, Any]) -> list[str]:
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
    outcome = result.get("outcome") if isinstance(result.get("outcome"), dict) else {}
    tool_result = outcome.get("tool_result") if isinstance(outcome.get("tool_result"), dict) else {}
    profile = tool_result.get("profile") if isinstance(tool_result.get("profile"), dict) else {}
    message = str(result.get("message") or tool_result.get("message") or "")
    if profile.get("profile_id") and profile.get("profile_id") != "pointer_only":
        return ["unsupported_tool_profile"]
    if "only allows pointer_only" in message:
        return ["unsupported_tool_profile"]
    return []


def _count_any(reason_counts: Counter[str], names: set[str]) -> int:
    return sum(reason_counts.get(name, 0) for name in names)


def _sum_column(rows: list[dict[str, str]], column: str) -> float | None:
    return _sum_present(_coerce(row.get(column)) for row in rows)


def _sum_present(values: Any) -> float | None:
    present = [float(value) for value in values if value not in {None, ""}]
    return sum(present) if present else None


def _min_present(values: Any) -> float | None:
    present = [float(value) for value in values if value not in {None, ""}]
    return min(present) if present else None


def _max_present(values: Any) -> float | None:
    present = [float(value) for value in values if value not in {None, ""}]
    return max(present) if present else None


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
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _coerce(value: Any) -> Any:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
