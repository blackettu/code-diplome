from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from seedling_core.run_snapshot import save_command_snapshot

from .architecture_audit import build_architecture_backlog_audit
from .compare import compare_runs
from .diagnostics import build_cell_metrics_diagnostics
from .html import write_experiment_html_report
from .readiness import build_readiness_report, build_software_readiness_smoke
from .scenario import compare_policy_scenario_file, write_policy_plan_file
from .tables import (
    write_annotation_feedback_results_csv,
    write_artifact_registry_csv,
    write_dataset_summary_csv,
    write_error_budget_results_csv,
    write_hardware_dry_run_results_csv,
    write_model_registry_csv,
    write_object_level_results_csv,
    write_post_action_results_csv,
    write_rl_results_csv,
    write_run_snapshot_results_csv,
    write_safety_results_csv,
    write_task_level_results_csv,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m seedling_reports")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build experiment CSV tables and an HTML report.")
    build.add_argument("--root", required=True, help="Directory with run artifacts.")
    build.add_argument("--out", required=True, help="Output directory.")
    build.add_argument("--title", default="Seedling Experiment Report")
    build.add_argument("--model-registry", default=None, help="Optional typed model registry YAML/JSON.")

    compare = subparsers.add_parser("compare-runs", help="Compare multiple report/run directories.")
    compare.add_argument("--runs", nargs="+", required=True)
    compare.add_argument("--out", required=True)
    compare.add_argument("--min-cell-accuracy", type=float, default=None)
    compare.add_argument("--min-edge-cell-accuracy", type=float, default=None)
    compare.add_argument("--min-corner-cell-accuracy", type=float, default=None)
    compare.add_argument("--min-target-recall", type=float, default=None)
    compare.add_argument("--min-rl-successful-target-rate", type=float, default=None)
    compare.add_argument("--max-rl-critical-error-rate", type=float, default=None)
    compare.add_argument("--max-rl-critical-events-count", type=float, default=None)
    compare.add_argument("--max-rl-blocked-actions", type=float, default=None)
    compare.add_argument("--max-rl-block-rate", type=float, default=None)
    compare.add_argument("--max-rl-crop-damage-count", type=float, default=None)
    compare.add_argument("--max-rl-reward-std", type=float, default=None)
    compare.add_argument("--min-rl-seed-count", type=float, default=None)
    compare.add_argument("--min-hardware-move-success-rate", type=float, default=None)
    compare.add_argument("--max-hardware-positioning-error-p95-mm", type=float, default=None)
    compare.add_argument("--max-safety-unsafe-action-escape-count", type=float, default=None)
    compare.add_argument("--max-safety-real-action-guard-blocks", type=float, default=None)
    compare.add_argument("--max-safety-unsupported-tool-profile-blocks", type=float, default=None)
    compare.add_argument("--max-safety-aborted-runs", type=float, default=None)
    compare.add_argument("--max-safety-skipped-commands", type=float, default=None)
    compare.add_argument("--max-safety-forbidden-zone-violation-count", type=float, default=None)
    compare.add_argument("--min-post-action-coverage-rate", type=float, default=None)
    compare.add_argument("--min-post-action-delayed-success-rate", type=float, default=None)
    compare.add_argument("--max-post-action-crop-damage-rate", type=float, default=None)
    compare.add_argument("--max-post-action-regrowth-rate", type=float, default=None)
    compare.add_argument("--max-error-budget-total-mm", type=float, default=None)
    compare.add_argument("--require-error-budget-complete", action="store_true")
    compare.add_argument("--require-run-snapshots", action="store_true")
    compare.add_argument("--max-run-snapshot-missing-config-hash", type=float, default=None)
    compare.add_argument("--max-run-snapshot-missing-command-args", type=float, default=None)

    scenario = subparsers.add_parser("scenario-compare", help="Compare policies on one SceneState file.")
    scenario.add_argument("--scene", required=True)
    scenario.add_argument("--policies", nargs="+", default=["raster_scan", "risk_aware_rule"])
    scenario.add_argument("--out", required=True)

    export_plan = subparsers.add_parser("export-plan", help="Export one policy ActionPlan for a SceneState.")
    export_plan.add_argument("--scene", required=True)
    export_plan.add_argument("--policy", required=True)
    export_plan.add_argument("--out", required=True)

    diagnose = subparsers.add_parser("diagnose-cell-metrics", help="Build bootstrap CI and error taxonomy reports.")
    diagnose.add_argument("--metrics", required=True)
    diagnose.add_argument("--out", required=True)
    diagnose.add_argument("--iterations", type=int, default=1000)
    diagnose.add_argument("--seed", type=int, default=42)
    diagnose.add_argument("--group-key", default="image")

    readiness = subparsers.add_parser("readiness-check", help="Build architecture DoD readiness evidence report.")
    readiness.add_argument("--root", required=True, help="Directory with generated artifacts to inspect.")
    readiness.add_argument("--out", required=True, help="Output directory for readiness_report JSON/CSV/HTML.")
    readiness.add_argument("--repo-root", default=None, help="Repository root for code/config/doc evidence.")

    architecture = subparsers.add_parser("architecture-audit", help="Build backlog ID evidence report from the architecture document.")
    architecture.add_argument("--doc", default="docs/seedlings_vilga_architecture_codex_rl_simulation.md")
    architecture.add_argument("--out", required=True, help="Output directory for architecture_backlog_audit JSON/CSV/HTML.")
    architecture.add_argument("--repo-root", default=None, help="Repository root for code/config/doc evidence.")

    software_smoke = subparsers.add_parser("software-readiness-smoke", help="Generate software-only readiness evidence artifacts.")
    software_smoke.add_argument("--out", required=True, help="Output directory for generated smoke artifacts.")
    software_smoke.add_argument("--repo-root", default=None, help="Repository root for code/config/doc evidence.")

    args = parser.parse_args(argv)
    if args.command == "build":
        output = Path(args.out)
        model_registry_path = args.model_registry or _default_model_registry_path()
        artifacts = write_artifact_registry_csv(args.root, output / "artifact_registry.csv")
        model_rows = write_model_registry_csv(model_registry_path, output / "model_registry.csv")
        dataset_rows = write_dataset_summary_csv(args.root, output / "dataset_summary.csv")
        object_rows = write_object_level_results_csv(args.root, output / "object_level_results.csv")
        task_rows = write_task_level_results_csv(args.root, output / "task_level_results.csv")
        snapshot_rows = write_run_snapshot_results_csv(args.root, output / "run_snapshot_results.csv")
        rl_rows = write_rl_results_csv(args.root, output / "rl_results.csv")
        hardware_rows = write_hardware_dry_run_results_csv(args.root, output / "hardware_dry_run_results.csv")
        safety_rows = write_safety_results_csv(args.root, output / "safety_results.csv")
        post_action_rows = write_post_action_results_csv(args.root, output / "post_action_results.csv")
        error_budget_rows = write_error_budget_results_csv(args.root, output / "error_budget_results.csv")
        feedback_rows = write_annotation_feedback_results_csv(args.root, output / "annotation_feedback_results.csv")
        write_experiment_html_report(
            args.title,
            output / "experiment_report.html",
            [
                ("Model Registry", model_rows),
                ("Dataset Summary", dataset_rows),
                ("Object-Level Results", object_rows),
                ("Task-Level Results", task_rows),
                ("Run Snapshot Results", snapshot_rows),
                ("RL Results", rl_rows),
                ("Hardware Dry-Run Results", hardware_rows),
                ("Safety Results", safety_rows),
                ("Post-Action Verification Results", post_action_rows),
                ("Error Budget Results", error_budget_rows),
                ("Annotation Feedback Results", feedback_rows),
                ("Artifacts", [artifact.to_dict() for artifact in artifacts]),
            ],
        )
        payload = {
            "ok": True,
            "out": str(output),
            "artifacts": len(artifacts),
            "model_rows": len(model_rows),
            "dataset_rows": len(dataset_rows),
            "object_rows": len(object_rows),
            "task_rows": len(task_rows),
            "run_snapshot_rows": len(snapshot_rows),
            "rl_rows": len(rl_rows),
            "hardware_rows": len(hardware_rows),
            "safety_rows": len(safety_rows),
            "post_action_rows": len(post_action_rows),
            "error_budget_rows": len(error_budget_rows),
            "annotation_feedback_rows": len(feedback_rows),
            "artifact_registry": str(output / "artifact_registry.json"),
        }
        (output / "report_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs = [
            output / "artifact_registry.csv",
            output / "model_registry.csv",
            output / "dataset_summary.csv",
            output / "object_level_results.csv",
            output / "task_level_results.csv",
            output / "run_snapshot_results.csv",
            output / "rl_results.csv",
            output / "hardware_dry_run_results.csv",
            output / "safety_results.csv",
            output / "post_action_results.csv",
            output / "error_budget_results.csv",
            output / "annotation_feedback_results.csv",
            output / "experiment_report.html",
            output / "report_summary.json",
        ]
        payload["run_snapshot"] = _write_reports_snapshot(
            output,
            "seedling-reports:build",
            args,
            inputs=[args.root, *([model_registry_path] if model_registry_path else [])],
            outputs=outputs,
            metadata={key: payload[key] for key in [
                "artifacts",
                "model_rows",
                "dataset_rows",
                "object_rows",
                "task_rows",
                "run_snapshot_rows",
                "rl_rows",
                "hardware_rows",
                "safety_rows",
                "post_action_rows",
                "error_budget_rows",
                "annotation_feedback_rows",
            ]},
        )
        if payload["run_snapshot"]:
            outputs.append(Path(payload["run_snapshot"]))
        payload["artifact_registry"] = _write_reports_registry(
            output,
            "seedling-reports:build",
            inputs=[args.root, *([model_registry_path] if model_registry_path else [])],
            outputs=outputs,
            metadata={key: payload[key] for key in [
                "artifacts",
                "model_rows",
                "dataset_rows",
                "object_rows",
                "task_rows",
                "run_snapshot_rows",
                "rl_rows",
                "hardware_rows",
                "safety_rows",
                "post_action_rows",
                "error_budget_rows",
                "annotation_feedback_rows",
            ]},
            csv=False,
        )
        (output / "report_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "compare-runs":
        thresholds = {
            "min_cell_accuracy": args.min_cell_accuracy,
            "min_edge_cell_accuracy": args.min_edge_cell_accuracy,
            "min_corner_cell_accuracy": args.min_corner_cell_accuracy,
            "min_target_recall": args.min_target_recall,
            "min_rl_successful_target_rate": args.min_rl_successful_target_rate,
            "max_rl_critical_error_rate": args.max_rl_critical_error_rate,
            "max_rl_critical_events_count": args.max_rl_critical_events_count,
            "max_rl_blocked_actions": args.max_rl_blocked_actions,
            "max_rl_block_rate": args.max_rl_block_rate,
            "max_rl_crop_damage_count": args.max_rl_crop_damage_count,
            "max_rl_reward_std": args.max_rl_reward_std,
            "min_rl_seed_count": args.min_rl_seed_count,
            "min_hardware_move_success_rate": args.min_hardware_move_success_rate,
            "max_hardware_positioning_error_p95_mm": args.max_hardware_positioning_error_p95_mm,
            "max_safety_unsafe_action_escape_count": args.max_safety_unsafe_action_escape_count,
            "max_safety_real_action_guard_blocks": args.max_safety_real_action_guard_blocks,
            "max_safety_unsupported_tool_profile_blocks": args.max_safety_unsupported_tool_profile_blocks,
            "max_safety_aborted_runs": args.max_safety_aborted_runs,
            "max_safety_skipped_commands": args.max_safety_skipped_commands,
            "max_safety_forbidden_zone_violation_count": args.max_safety_forbidden_zone_violation_count,
            "min_post_action_coverage_rate": args.min_post_action_coverage_rate,
            "min_post_action_delayed_success_rate": args.min_post_action_delayed_success_rate,
            "max_post_action_crop_damage_rate": args.max_post_action_crop_damage_rate,
            "max_post_action_regrowth_rate": args.max_post_action_regrowth_rate,
            "max_error_budget_total_mm": args.max_error_budget_total_mm,
            "require_error_budget_complete": 0.0 if args.require_error_budget_complete else None,
            "require_run_snapshots": 1.0 if args.require_run_snapshots else None,
            "max_run_snapshot_missing_config_hash": args.max_run_snapshot_missing_config_hash,
            "max_run_snapshot_missing_command_args": args.max_run_snapshot_missing_command_args,
        }
        rows = compare_runs(args.runs, args.out, thresholds=thresholds)
        compare_outputs = [
            Path(args.out) / "compare_runs.csv",
            Path(args.out) / "compare_runs.html",
            Path(args.out) / "compare_runs_summary.json",
        ]
        payload = {
            "ok": True,
            "out": args.out,
            "runs": len(rows),
            "failed": sum(1 for row in rows if row.get("threshold_status") == "fail"),
            "run_snapshot": _write_reports_snapshot(
                args.out,
                "seedling-reports:compare-runs",
                args,
                inputs=args.runs,
                outputs=compare_outputs,
                metadata={"runs": len(rows), "failed": sum(1 for row in rows if row.get("threshold_status") == "fail")},
            ),
            "artifact_registry": _write_reports_registry(
                args.out,
                "seedling-reports:compare-runs",
                inputs=args.runs,
                outputs=[*compare_outputs, *([Path(args.out) / "run_snapshot.json"] if (Path(args.out) / "run_snapshot.json").exists() else [])],
                metadata={"runs": len(rows), "failed": sum(1 for row in rows if row.get("threshold_status") == "fail")},
            ),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "scenario-compare":
        rows = compare_policy_scenario_file(args.scene, args.policies, args.out)
        output = Path(args.out)
        scenario_outputs = [output / "scenario_compare.json", output / "scenario_compare.html"]
        snapshot = _write_reports_snapshot(
            output,
            "seedling-reports:scenario-compare",
            args,
            inputs=[args.scene],
            outputs=scenario_outputs,
            metadata={"policies": args.policies},
        )
        payload = {
            "ok": True,
            "out": args.out,
            "policies": len(rows),
            "run_snapshot": snapshot,
            "artifact_registry": _write_reports_registry(
                output,
                "seedling-reports:scenario-compare",
                inputs=[args.scene],
                outputs=[*scenario_outputs, *([snapshot] if snapshot else [])],
                metadata={"policies": args.policies},
            ),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "export-plan":
        plan = write_policy_plan_file(args.scene, args.policy, args.out)
        snapshot = _write_reports_snapshot(
            Path(args.out).parent,
            "seedling-reports:export-plan",
            args,
            inputs=[args.scene],
            outputs=[args.out],
            metadata={"policy": args.policy, "commands": len(plan.commands)},
        )
        payload = {
            "ok": True,
            "out": args.out,
            "scene": args.scene,
            "policy": plan.policy_id,
            "commands": len(plan.commands),
            "review_targets": len(plan.review_target_ids),
            "blocked_targets": len(plan.blocked_target_ids),
            "run_snapshot": snapshot,
            "artifact_registry": _write_reports_registry(
                Path(args.out).parent,
                "seedling-reports:export-plan",
                inputs=[args.scene],
                outputs=[args.out, *([snapshot] if snapshot else [])],
                metadata={"policy": args.policy, "commands": len(plan.commands)},
            ),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "diagnose-cell-metrics":
        payload = build_cell_metrics_diagnostics(
            args.metrics,
            output_dir=args.out,
            iterations=args.iterations,
            seed=args.seed,
            group_key=args.group_key,
        )
        diagnose_outputs = [Path(args.out) / "bootstrap_ci.json", Path(args.out) / "error_taxonomy.json"]
        snapshot = _write_reports_snapshot(
            args.out,
            "seedling-reports:diagnose-cell-metrics",
            args,
            inputs=[args.metrics],
            outputs=diagnose_outputs,
            metadata={"iterations": args.iterations, "seed": args.seed, "group_key": args.group_key},
        )
        payload = {
            "ok": True,
            "out": args.out,
            **payload,
            "run_snapshot": snapshot,
            "artifact_registry": _write_reports_registry(
                args.out,
                "seedling-reports:diagnose-cell-metrics",
                inputs=[args.metrics],
                outputs=[*diagnose_outputs, *([snapshot] if snapshot else [])],
                metadata={"iterations": args.iterations, "seed": args.seed, "group_key": args.group_key},
            ),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "readiness-check":
        payload = build_readiness_report(args.root, args.out, repo_root=args.repo_root)
        readiness_outputs = [
            Path(args.out) / "readiness_report.json",
            Path(args.out) / "readiness_report.csv",
            Path(args.out) / "readiness_report.html",
        ]
        payload["run_snapshot"] = _write_reports_snapshot(
            args.out,
            "seedling-reports:readiness-check",
            args,
            inputs=[args.root, *([args.repo_root] if args.repo_root else [])],
            outputs=readiness_outputs,
            metadata={
                "counts": payload["counts"],
                "physical_validation_required": payload["physical_validation_required"],
            },
        )
        payload["artifact_registry"] = _write_reports_registry(
            args.out,
            "seedling-reports:readiness-check",
            inputs=[args.root, *([args.repo_root] if args.repo_root else [])],
            outputs=[*readiness_outputs, *([payload["run_snapshot"]] if payload["run_snapshot"] else [])],
            metadata={
                "counts": payload["counts"],
                "physical_validation_required": payload["physical_validation_required"],
            },
        )
        (Path(args.out) / "readiness_report.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "architecture-audit":
        payload = build_architecture_backlog_audit(args.doc, args.out, repo_root=args.repo_root)
        audit_outputs = [
            Path(args.out) / "architecture_backlog_audit.json",
            Path(args.out) / "architecture_backlog_audit.csv",
            Path(args.out) / "architecture_backlog_audit.html",
        ]
        payload["run_snapshot"] = _write_reports_snapshot(
            args.out,
            "seedling-reports:architecture-audit",
            args,
            inputs=[args.doc, *([args.repo_root] if args.repo_root else [])],
            outputs=audit_outputs,
            metadata={"counts": payload["counts"]},
        )
        payload["artifact_registry"] = _write_reports_registry(
            args.out,
            "seedling-reports:architecture-audit",
            inputs=[args.doc, *([args.repo_root] if args.repo_root else [])],
            outputs=[*audit_outputs, *([payload["run_snapshot"]] if payload["run_snapshot"] else [])],
            metadata={"counts": payload["counts"]},
        )
        (Path(args.out) / "architecture_backlog_audit.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    elif args.command == "software-readiness-smoke":
        payload = build_software_readiness_smoke(args.out, repo_root=args.repo_root)
        payload["run_snapshot"] = _write_reports_snapshot(
            args.out,
            "seedling-reports:software-readiness-smoke",
            args,
            inputs=[*([args.repo_root] if args.repo_root else [])],
            outputs=[Path(path) for path in payload["artifacts"]],
            metadata={
                "software_ready": payload["software_ready"],
                "physical_validation_required": payload["physical_validation_required"],
                "readiness_counts": payload["readiness_counts"],
            },
        )
        if payload["run_snapshot"]:
            payload["artifacts"].append(payload["run_snapshot"])
        payload["artifact_registry"] = _write_reports_registry(
            args.out,
            "seedling-reports:software-readiness-smoke",
            inputs=[*([args.repo_root] if args.repo_root else [])],
            outputs=[Path(path) for path in payload["artifacts"]],
            metadata={
                "software_ready": payload["software_ready"],
                "physical_validation_required": payload["physical_validation_required"],
                "readiness_counts": payload["readiness_counts"],
            },
        )
        (Path(args.out) / "software_readiness_smoke.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _default_model_registry_path() -> str | None:
    path = Path("configs/registry/model_registry_v0_1.yaml")
    return str(path) if path.exists() else None


def _write_reports_snapshot(
    run_dir: str | Path,
    command: str,
    args: argparse.Namespace,
    *,
    inputs: list[str | Path],
    outputs: list[str | Path],
    metadata: dict[str, Any] | None = None,
) -> str | None:
    try:
        return save_command_snapshot(
            run_dir,
            command,
            command_args=_namespace_payload(args),
            input_paths=inputs,
            output_paths=outputs,
            metadata=metadata,
        )
    except Exception:
        return None


def _write_reports_registry(
    run_dir: str | Path,
    command: str,
    *,
    inputs: list[str | Path],
    outputs: list[str | Path],
    metadata: dict[str, Any] | None = None,
    csv: bool = True,
) -> str | None:
    try:
        from seedling_reports.registry import ArtifactRecord, ExperimentRegistry, RunRecord, write_run_registry_records

        artifacts = [
            *(ArtifactRecord.from_path(path, artifact_type=_artifact_type(path), role="input", command=command) for path in inputs),
            *(ArtifactRecord.from_path(path, artifact_type=_artifact_type(path), role="output", command=command) for path in outputs),
        ]
        run_id = _registry_run_id(run_dir, command, outputs)
        if csv:
            write_run_registry_records(run_dir, command, artifacts, metadata=metadata, run_id=run_id)
        else:
            directory = Path(run_dir)
            registry = ExperimentRegistry(directory / "artifact_registry.json")
            registry.add_run(
                RunRecord(
                    run_id=run_id,
                    command=command,
                    run_dir=str(directory),
                    artifacts=artifacts,
                    metadata=metadata or {},
                )
            )
            registry.to_json()
        return str(Path(run_dir) / "artifact_registry.json")
    except Exception:
        return None


def _namespace_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {key: _jsonable(value) for key, value in vars(args).items()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _artifact_type(path: str | Path) -> str:
    candidate = Path(path)
    name = candidate.name
    if candidate.is_dir():
        return "directory"
    if name.endswith(".json"):
        return name.removesuffix(".json")
    if name.endswith(".jsonl"):
        return name.removesuffix(".jsonl")
    if name.endswith(".csv"):
        return name.removesuffix(".csv")
    if name.endswith(".md"):
        return "markdown_report"
    if name.endswith(".html"):
        return "html_report"
    if name.endswith(".yaml") or name.endswith(".yml"):
        return "config"
    return candidate.suffix.lstrip(".") or "artifact"


def _registry_run_id(run_dir: str | Path, command: str, outputs: list[str | Path]) -> str:
    directory = Path(run_dir).name or "run"
    output_stem = Path(outputs[0]).stem if outputs else "stdout"
    command_slug = "".join(char if char.isalnum() else "_" for char in command).strip("_")
    return f"{directory}_{command_slug}_{output_stem}"
