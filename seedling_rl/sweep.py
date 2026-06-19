from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any

from seedling_core.config import load_config_file
from seedling_reports.registry import ArtifactRecord, write_run_registry_records


STABILITY_METRICS = (
    "reward_mean",
    "critical_error_rate",
    "successful_target_rate",
    "review_rate",
    "actions_per_tray",
)


def build_sweep_plan(config_path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    data = load_config_file(config_path)
    base = data.get("base", {}) if isinstance(data.get("base"), dict) else {}
    grid = data.get("grid", {}) if isinstance(data.get("grid"), dict) else {}
    seeds = data.get("seeds", [base.get("seed", 42)])
    runs = []
    keys = sorted(grid)
    values = [grid[key] if isinstance(grid[key], list) else [grid[key]] for key in keys]
    for index, combination in enumerate(itertools.product(*values), 1):
        params = dict(zip(keys, combination))
        for seed in seeds:
            run = dict(base)
            run.update(params)
            run["seed"] = int(seed)
            run["run_id"] = f"sweep_{index:03d}_seed_{int(seed):03d}"
            runs.append(run)
    payload = {
        "ok": True,
        "schema_version": "rl_sweep_plan_v0_1",
        "config_path": str(config_path),
        "runs": runs,
        "count": len(runs),
    }
    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_sweep_stability_report(
    plan_path: str | Path,
    metrics_paths: list[str | Path],
) -> dict[str, Any]:
    plan = _load_json(Path(plan_path))
    planned_runs = plan.get("runs", []) if isinstance(plan, dict) else []
    runs_by_id = {
        str(run["run_id"]): run
        for run in planned_runs
        if isinstance(run, dict) and run.get("run_id") is not None
    }
    grouped: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for metrics_path in metrics_paths:
        path = Path(metrics_path)
        data = _load_json(path)
        if not isinstance(data, dict):
            warnings.append(f"{path}: metrics file must contain a JSON object")
            continue
        metric_source = _metric_source(data)
        run_id = str(data.get("run_id") or metric_source.get("run_id") or path.parent.name)
        planned_run = runs_by_id.get(run_id, {})
        seed = _first_present(
            data.get("seed"),
            metric_source.get("seed"),
            _dict_get(data, "config", "seed"),
            planned_run.get("seed") if isinstance(planned_run, dict) else None,
        )
        group_config = _group_config(planned_run if planned_run else data.get("config", {}))
        if not group_config:
            group_config = _group_config(metric_source)
        policy = metric_source.get("policy") or data.get("policy")
        mode = data.get("mode") or metric_source.get("mode")
        if policy:
            group_config["policy"] = policy
        if mode:
            group_config["mode"] = mode
        group_id = _group_id(group_config)
        record = {
            "run_id": run_id,
            "seed": _maybe_int(seed),
            "metrics_path": str(path),
            "metrics": {
                metric: _maybe_float(metric_source.get(metric))
                for metric in STABILITY_METRICS
                if _maybe_float(metric_source.get(metric)) is not None
            },
        }
        grouped.setdefault(
            group_id,
            {
                "group_id": group_id,
                "config": group_config,
                "runs": [],
            },
        )["runs"].append(record)
    groups = [_summarize_group(group) for group in grouped.values()]
    groups.sort(key=lambda item: item["group_id"])
    return {
        "ok": True,
        "schema_version": "rl_sweep_stability_v0_1",
        "plan_path": str(plan_path),
        "metric_files": [str(Path(path)) for path in metrics_paths],
        "group_count": len(groups),
        "run_count": sum(len(group["runs"]) for group in groups),
        "groups": groups,
        "warnings": warnings,
    }


def write_sweep_stability_report(
    plan_path: str | Path,
    metrics_paths: list[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = build_sweep_stability_report(plan_path, metrics_paths)
    summary_path = output / "sweep_stability_summary.json"
    table_path = output / "sweep_stability_summary.csv"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_stability_csv(table_path, payload["groups"])
    artifacts = [
        ArtifactRecord.from_path(plan_path, "rl_sweep_plan", role="input", command="rl-sweep-report"),
        *[
            ArtifactRecord.from_path(path, "rl_eval_metrics", role="input", command="rl-sweep-report")
            for path in metrics_paths
        ],
        ArtifactRecord.from_path(summary_path, "rl_sweep_stability_summary", role="output", command="rl-sweep-report"),
        ArtifactRecord.from_path(table_path, "rl_sweep_stability_table", role="output", command="rl-sweep-report"),
    ]
    write_run_registry_records(
        output,
        "rl-sweep-report",
        artifacts,
        run_id=f"{output.name or 'sweep'}_rl_sweep_report",
        metadata={"plan_path": str(plan_path), "metrics": [str(Path(path)) for path in metrics_paths]},
    )
    return {**payload, "out": str(output), "summary_path": str(summary_path), "csv_path": str(table_path)}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _metric_source(data: dict[str, Any]) -> dict[str, Any]:
    results = data.get("results")
    if isinstance(results, list) and results and isinstance(results[0], dict):
        source = dict(results[0])
        if data.get("mode") is not None:
            source.setdefault("mode", data.get("mode"))
        return source
    return data


def _group_config(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    ignored = {"run_id", "seed", "run_dir", "tensorboard_log", "checkpoint", "replay_dir", "replay_paths"}
    config: dict[str, Any] = {}
    for key, value in data.items():
        if key in ignored or key.endswith("_path"):
            continue
        if key == "outputs" and isinstance(value, dict):
            filtered = {sub_key: sub_value for sub_key, sub_value in value.items() if sub_key not in {"run_dir", "tensorboard_dir"}}
            if filtered:
                config[key] = filtered
            continue
        if key in STABILITY_METRICS or key in {"results", "critical_events", "details"}:
            continue
        config[key] = value
    return config


def _group_id(config: dict[str, Any]) -> str:
    if not config:
        return "default"
    raw = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    chars = [char.lower() if char.isalnum() else "_" for char in raw]
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug[:96] or "default"


def _summarize_group(group: dict[str, Any]) -> dict[str, Any]:
    runs = group["runs"]
    metrics: dict[str, dict[str, float | int | None]] = {}
    for metric in STABILITY_METRICS:
        values = [run["metrics"][metric] for run in runs if metric in run["metrics"]]
        metrics[metric] = _summary_stats(values)
    seeds = sorted({run["seed"] for run in runs if run.get("seed") is not None})
    return {
        "group_id": group["group_id"],
        "config": group["config"],
        "run_count": len(runs),
        "seeds": seeds,
        "seed_count": len(seeds),
        "metrics": metrics,
        "runs": runs,
    }


def _summary_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None, "range": None}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "count": len(values),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
    }


def _write_stability_csv(path: Path, groups: list[dict[str, Any]]) -> None:
    fieldnames = ["group_id", "run_count", "seed_count", "seeds", "config"]
    for metric in STABILITY_METRICS:
        for stat in ("count", "mean", "std", "min", "max", "range"):
            fieldnames.append(f"{metric}_{stat}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group in groups:
            row: dict[str, Any] = {
                "group_id": group["group_id"],
                "run_count": group["run_count"],
                "seed_count": group["seed_count"],
                "seeds": ",".join(str(seed) for seed in group["seeds"]),
                "config": json.dumps(group["config"], ensure_ascii=False, sort_keys=True),
            }
            for metric in STABILITY_METRICS:
                for stat, value in group["metrics"][metric].items():
                    row[f"{metric}_{stat}"] = value
            writer.writerow(row)


def _dict_get(data: dict[str, Any], key: str, nested_key: str) -> Any:
    value = data.get(key)
    if isinstance(value, dict):
        return value.get(nested_key)
    return None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _maybe_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
