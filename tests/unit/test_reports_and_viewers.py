from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from seedling_core.schemas import ActionPlan, ActionTarget, CellState, RobotState, SafetyState, SceneState, TrayState
from seedling_reports.diagnostics import build_cell_metrics_diagnostics, error_taxonomy
from seedling_reports.architecture_audit import build_architecture_backlog_audit, parse_backlog_items
from seedling_reports.readiness import build_readiness_report, build_software_readiness_smoke
from seedling_reports.registry import ArtifactRecord, write_run_registry, write_run_registry_records
from seedling_reports.scenario import compare_policies_on_scene, write_policy_plan_file
from seedling_reports.tables import (
    write_annotation_feedback_results_csv,
    write_artifact_registry_csv,
    collect_artifacts,
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
from seedling_ui.offline_viewer import write_offline_viewer
from seedling_ui.report_export import write_report_export
from seedling_ui.replay_viewer import write_replay_viewer
from seedling_sim import ReplayLogger, SimPlant, SimScene, SimTarget


ROOT = Path(__file__).resolve().parents[2]


class ReportsAndViewersTests(unittest.TestCase):
    def test_report_tables_collect_known_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            run = root / "run1"
            run.mkdir(parents=True)
            (run / "dataset_audit.json").write_text(
                json.dumps(
                    {
                        "dataset_root": "dataset",
                        "splits": {
                            "test": {
                                "images": 2,
                                "labels": 3,
                                "missing_labels": ["a.jpg"],
                                "empty_labels": [],
                                "orphan_labels": ["orphan.txt"],
                                "suspected_augmented_count": 0,
                                "class_counts": {"container": 1, "seedlings": 2},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            (run / "raw_dataset_audit.json").write_text(
                json.dumps(
                    {
                        "dataset_root": "raw_dataset",
                        "splits": {
                            "all": {
                                "images": 3,
                                "labels": 4,
                                "missing_labels": ["missing.jpg"],
                                "empty_labels": ["empty.txt"],
                                "orphan_labels": [],
                                "suspected_augmented_count": 0,
                                "class_counts": {"container": 1, "seedlings": 3},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            (run / "test_metrics.json").write_text(
                json.dumps({"map50": 0.7, "map50_95": 0.5, "precision_mean": 0.8, "recall_mean": 0.6}),
                encoding="utf-8",
            )
            (run / "cell_metrics.json").write_text(
                json.dumps(
                    {
                        "cell_accuracy": 0.9,
                        "cell_macro": {"macro_f1": 0.8},
                        "multi_seedling_cell": {"precision": 0.7, "recall": 0.6},
                        "removal_targets": {
                            "precision": 0.75,
                            "recall": 0.65,
                            "mean_coordinate_error_px": 4.0,
                            "mean_coordinate_error_mm": 2.0,
                        },
                        "cost_sensitive": {
                            "total_cost": 20.0,
                            "critical_error_total": 2,
                            "critical_error_rate_per_cell": 0.25,
                            "critical_error_counts": {
                                "target_false_negative": 1,
                                "missed_multiple_crop": 1,
                            },
                        },
                        "container_recall": 0.5,
                    }
                ),
                encoding="utf-8",
            )
            (run / "unified_rl_baselines.json").write_text(
                json.dumps(
                    {
                        "mode": "evaluate-baselines",
                        "episodes": 2,
                        "results": [
                            {
                                "policy": "route_planning",
                                "episodes": 2,
                                "reward_mean": 12.5,
                                "critical_error_rate": 0.0,
                                "successful_target_rate": 1.0,
                                "review_rate": 0.1,
                                "actions_per_tray": 4.0,
                                "total_distance_mm": 44.0,
                                "mean_distance_error_mm": 0.5,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            out = Path(tmp) / "reports"

            artifacts = write_artifact_registry_csv(root, out / "artifact_registry.csv")
            dataset_rows = write_dataset_summary_csv(root, out / "dataset_summary.csv")
            object_rows = write_object_level_results_csv(root, out / "object_level_results.csv")
            task_rows = write_task_level_results_csv(root, out / "task_level_results.csv")
            rl_rows = write_rl_results_csv(root, out / "rl_results.csv")

            self.assertEqual(len(artifacts), 5)
            self.assertEqual(dataset_rows[0]["images"], 2)
            self.assertEqual(dataset_rows[0]["audit_stage"], "prepared")
            self.assertEqual(dataset_rows[1]["audit_stage"], "raw")
            self.assertEqual(dataset_rows[1]["split"], "all")
            self.assertEqual(object_rows[0]["map50"], 0.7)
            self.assertEqual(task_rows[0]["schema_mode"], "legacy_yolo")
            self.assertEqual(task_rows[0]["cell_accuracy"], 0.9)
            self.assertEqual(task_rows[0]["mean_coordinate_error_mm"], 2.0)
            self.assertEqual(task_rows[0]["cost_total"], 20.0)
            self.assertEqual(task_rows[0]["critical_error_total"], 2)
            self.assertIn("missed_multiple_crop", task_rows[0]["critical_error_counts"])
            self.assertEqual(rl_rows[0]["policy"], "route_planning")
            self.assertEqual(rl_rows[0]["reward_mean"], 12.5)
            self.assertEqual(rl_rows[0]["total_distance_mm"], 44.0)
            self.assertEqual(rl_rows[0]["mean_distance_error_mm"], 0.5)

    def test_task_level_results_preserve_scene_state_source_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            run = root / "scene_eval"
            run.mkdir(parents=True)
            (run / "cell_metrics.json").write_text(
                json.dumps(
                    {
                        "schema_mode": "scene_state",
                        "cell_accuracy": 1.0,
                        "cell_macro": {"macro_f1": 1.0},
                        "multi_seedling_cell": {"precision": 1.0, "recall": 1.0},
                        "removal_targets": {
                            "precision": 1.0,
                            "recall": 1.0,
                            "mean_coordinate_error_px": 4.0,
                        },
                        "scene_state_metrics": {
                            "gt_scene_id": "gt_scene",
                            "pred_scene_id": "pred_scene",
                            "cell_metrics": {
                                "edge_cell_metrics": {"accuracy": 0.75},
                                "corner_cell_metrics": {"accuracy": 0.5},
                            },
                            "target_metrics": {
                                "expert_keep_remove": {
                                    "precision": 0.5,
                                    "recall": 1.0,
                                    "fp": 1,
                                    "fn": 0,
                                }
                            },
                            "cost_sensitive": {
                                "total_cost": 7.0,
                                "critical_error_total": 2,
                                "critical_error_rate_per_cell": 0.5,
                                "critical_error_counts": {"expert_false_removal": 1},
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            rows = write_task_level_results_csv(root, Path(tmp) / "task_level_results.csv")

            self.assertEqual(rows[0]["schema_mode"], "scene_state")
            self.assertEqual(rows[0]["gt_scene_id"], "gt_scene")
            self.assertEqual(rows[0]["pred_scene_id"], "pred_scene")
            self.assertEqual(rows[0]["edge_cell_accuracy"], 0.75)
            self.assertEqual(rows[0]["corner_cell_accuracy"], 0.5)
            self.assertEqual(rows[0]["target_recall"], 1.0)
            self.assertEqual(rows[0]["expert_remove_precision"], 0.5)
            self.assertEqual(rows[0]["expert_remove_recall"], 1.0)
            self.assertEqual(rows[0]["expert_false_removal"], 1)
            self.assertEqual(rows[0]["expert_missed_removal"], 0)
            self.assertEqual(rows[0]["cost_total"], 7.0)
            self.assertEqual(rows[0]["critical_error_total"], 2)
            self.assertIn("expert_false_removal", rows[0]["critical_error_counts"])

    def test_task_level_results_include_direct_scene_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            run = root / "scene_eval"
            run.mkdir(parents=True)
            (run / "scene_metrics.json").write_text(
                json.dumps(
                    {
                        "gt_scene_id": "gt_scene",
                        "pred_scene_id": "pred_scene",
                        "cell_metrics": {
                            "accuracy": 0.8,
                            "macro": {"macro_f1": 0.75},
                            "edge_cell_metrics": {"accuracy": 0.7},
                            "corner_cell_metrics": {"accuracy": 0.6},
                        },
                        "target_metrics": {
                            "precision": 0.5,
                            "recall": 1.0,
                            "mean_error_px": 3.0,
                            "mean_error_mm": 0.3,
                            "expert_keep_remove": {
                                "precision": 0.75,
                                "recall": 0.6,
                                "fp": 2,
                                "fn": 3,
                            },
                        },
                        "cost_sensitive": {
                            "total_cost": 12.0,
                            "critical_error_total": 3,
                            "critical_error_rate_per_cell": 1.5,
                            "critical_error_counts": {"target_false_positive": 1},
                        },
                    }
                ),
                encoding="utf-8",
            )

            artifacts = collect_artifacts(root)
            rows = write_task_level_results_csv(root, Path(tmp) / "task_level_results.csv")

            self.assertIn(run / "scene_metrics.json", artifacts)
            self.assertEqual(rows[0]["schema_mode"], "scene_state")
            self.assertEqual(rows[0]["cell_macro_f1"], 0.75)
            self.assertEqual(rows[0]["edge_cell_accuracy"], 0.7)
            self.assertEqual(rows[0]["corner_cell_accuracy"], 0.6)
            self.assertEqual(rows[0]["mean_coordinate_error_mm"], 0.3)
            self.assertEqual(rows[0]["expert_remove_precision"], 0.75)
            self.assertEqual(rows[0]["expert_remove_recall"], 0.6)
            self.assertEqual(rows[0]["expert_false_removal"], 2)
            self.assertEqual(rows[0]["expert_missed_removal"], 3)
            self.assertEqual(rows[0]["critical_error_total"], 3)
            self.assertIn("target_false_positive", rows[0]["critical_error_counts"])

    def test_reports_build_cli_writes_rl_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            run = root / "run1"
            run.mkdir(parents=True)
            (run / "rl_eval_metrics.json").write_text(
                json.dumps(
                    {
                        "mode": "checkpoint",
                        "checkpoint": "model.zip",
                        "episodes": 3,
                        "reward_mean": 7.0,
                        "critical_error_rate": 0.2,
                        "successful_target_rate": 0.8,
                    }
                ),
                encoding="utf-8",
            )
            output = Path(tmp) / "report"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_reports",
                    "build",
                    "--root",
                    str(root),
                    "--out",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(completed.stdout)
            rl_results = (output / "rl_results.csv").read_text(encoding="utf-8")
            run_snapshot = json.loads((output / "run_snapshot.json").read_text(encoding="utf-8"))
            report_registry = json.loads((output / "artifact_registry.json").read_text(encoding="utf-8"))
            artifact_table = (output / "artifact_registry.csv").read_text(encoding="utf-8")
            self.assertEqual(payload["rl_rows"], 1)
            self.assertTrue(payload["artifact_registry"])
            self.assertEqual(payload["run_snapshot"], str(output / "run_snapshot.json"))
            self.assertEqual(run_snapshot["command"], "seedling-reports:build")
            self.assertEqual(run_snapshot["command_args"]["root"], str(root))
            self.assertEqual(report_registry["runs"][0]["command"], "seedling-reports:build")
            self.assertIn(
                ("run_snapshot", "output", str(output / "run_snapshot.json")),
                {
                    (artifact["artifact_type"], artifact["role"], artifact["path"])
                    for artifact in report_registry["runs"][0]["artifacts"]
                },
            )
            self.assertIn("reward_mean", rl_results)
            self.assertIn("7.0", rl_results)
            self.assertIn("rl_eval_metrics", artifact_table)

    def test_model_registry_table_reads_typed_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "model_registry.yaml"
            registry.write_text(
                """
version: test_registry
models:
  - model_id: detector_test
    model_type: detector
    framework: test
    artifact_uri: runs/model.pt
    input_schema: ImageInputV1
    output_schema: DetectionResultV1
    status: draft
    safety_level: offline_only
    dataset_version: zks_v0_1
    ontology_version: ontology_v0_1
""",
                encoding="utf-8",
            )

            rows = write_model_registry_csv(registry, Path(tmp) / "model_registry.csv")

            self.assertEqual(rows[0]["model_id"], "detector_test")
            self.assertEqual(rows[0]["model_type"], "detector")
            self.assertEqual(rows[0]["registry_ok"], True)

    def test_hardware_dry_run_results_include_control_points_and_hil(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            run = root / "run1"
            run.mkdir(parents=True)
            (run / "dry_run_control_points.json").write_text(
                json.dumps(
                    {
                        "ok": False,
                        "points": [
                            {
                                "point_id": "p1",
                                "expected_mm": [0, 0, 0],
                                "actual_mm": [0, 0, 0],
                                "error_mm": 0.0,
                                "ok": True,
                            },
                            {
                                "point_id": "p1",
                                "expected_mm": [0, 0, 0],
                                "actual_mm": [1, 0, 0],
                                "error_mm": 2.0,
                                "ok": False,
                            },
                        ],
                        "tolerance_mm": 1.0,
                        "error_summary_mm": {"count": 2, "mean": 1.0, "max": 2.0},
                        "command_log": "dry_run_commands.json",
                    }
                ),
                encoding="utf-8",
            )
            (run / "hil_pointer_report.json").write_text(
                json.dumps(
                    {
                        "ok": False,
                        "mode": "hardware_in_loop_pointer",
                        "scene_id": "scene_001",
                        "plan_id": "plan_001",
                        "review": {"ok": True, "errors": []},
                        "commands": 2,
                        "executed": 1,
                        "blocked": 1,
                        "command_log": "hil_pointer_commands.json",
                        "replay": "hil_pointer_replay.json",
                        "results": [{"ok": True}, {"ok": False}],
                    }
                ),
                encoding="utf-8",
            )
            (run / "dry_run_plan_report.json").write_text(
                json.dumps(
                    {
                        "ok": False,
                        "mode": "dry_run_pointer",
                        "scene_id": "scene_001",
                        "plan_id": "plan_001",
                        "commands": 2,
                        "executed": 1,
                        "blocked": 1,
                        "command_log": "dry_run_plan_commands.json",
                        "replay": "dry_run_plan_replay.json",
                        "results": [{"ok": True}, {"ok": False}],
                    }
                ),
                encoding="utf-8",
            )

            rows = write_hardware_dry_run_results_csv(root, Path(tmp) / "hardware.csv")

            dry_run_rows = [row for row in rows if row["mode"] == "dry_run_pointer"]
            dry_run = next(row for row in dry_run_rows if row["points"] == 2)
            dry_run_plan = next(row for row in dry_run_rows if row["points"] is None)
            hil = next(row for row in rows if row["mode"] == "hardware_in_loop_pointer")
            self.assertEqual(dry_run["points"], 2)
            self.assertEqual(dry_run["move_success_rate"], 0.5)
            self.assertEqual(dry_run["positioning_error_p50_mm"], 1.0)
            self.assertEqual(dry_run["repeatability_mm"], 1.0)
            self.assertEqual(dry_run_plan["commands"], 2)
            self.assertEqual(dry_run_plan["move_success_rate"], 0.5)
            self.assertEqual(dry_run_plan["safety_blocks"], 1)
            self.assertEqual(hil["commands"], 2)
            self.assertEqual(hil["review_ok"], True)
            self.assertEqual(hil["safety_blocks"], 1)

    def test_safety_results_include_offline_and_hil_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            run = root / "run1"
            run.mkdir(parents=True)
            (run / "offline_replay_eval.json").write_text(
                json.dumps(
                    {
                        "action_events": 4,
                        "blocked": 2,
                        "crop_damage": 1,
                        "reviewed_targets": 1,
                        "block_reason_counts": {
                            "BLOCK_INTERLOCK": 1,
                            "interlock_false": 1,
                            "calibration_required": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (run / "hil_pointer_report.json").write_text(
                json.dumps(
                    {
                        "ok": False,
                        "mode": "hardware_in_loop_pointer",
                        "commands": 1,
                        "executed": 0,
                        "blocked": 1,
                        "aborted": True,
                        "skipped": 1,
                        "review": {"ok": True, "emergency_stop_tested": True},
                        "results": [
                            {
                                "ok": False,
                                "safety_decision": {
                                    "allowed": False,
                                    "result": "BLOCK_UNSAFE_TARGET",
                                    "reasons": [
                                        "forbidden_zone_overlap",
                                        "software_safe_mode_disabled",
                                        "enclosure_not_closed",
                                    ],
                                },
                            },
                            {
                                "ok": False,
                                "safety_decision": {
                                    "allowed": True,
                                    "result": "ALLOW_HARDWARE_IN_LOOP_POINTER",
                                    "reasons": [],
                                },
                                "outcome": {
                                    "tool_result": {
                                        "ok": False,
                                        "profile": {"profile_id": "laser", "dwell_ms": 0, "metadata": {}},
                                        "message": "HIL adapter only allows pointer_only",
                                    }
                                },
                                "message": "HIL adapter only allows pointer_only",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (run / "dry_run_plan_report.json").write_text(
                json.dumps(
                    {
                        "ok": False,
                        "mode": "dry_run_pointer",
                        "commands": 2,
                        "executed": 0,
                        "blocked": 2,
                        "aborted": True,
                        "skipped": 1,
                        "results": [
                            {
                                "ok": False,
                                "safety_decision": {
                                    "allowed": False,
                                    "result": "BLOCK_REVIEW_REQUIRED",
                                    "reasons": ["operator_confirmation_required"],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            rows = write_safety_results_csv(root, Path(tmp) / "safety.csv")

            by_source = {row["source"]: row for row in rows}
            offline = by_source["offline_replay_eval"]
            dry_run = by_source["dry_run_pointer"]
            hil = by_source["hardware_in_loop_pointer"]
            self.assertEqual(offline["unsafe_action_blocked_count"], 2)
            self.assertEqual(offline["unsafe_action_escape_count"], 1)
            self.assertEqual(offline["interlock_failure_count"], 2)
            self.assertEqual(offline["calibration_expired_blocks"], 1)
            self.assertEqual(dry_run["unsafe_action_blocked_count"], 1)
            self.assertEqual(dry_run["review_required_count"], 2)
            self.assertEqual(dry_run["aborted_runs"], 1)
            self.assertEqual(dry_run["skipped_commands"], 1)
            self.assertEqual(hil["forbidden_zone_violation_count"], 1)
            self.assertEqual(hil["real_action_guard_blocks"], 2)
            self.assertEqual(hil["unsupported_tool_profile_blocks"], 1)
            self.assertEqual(hil["aborted_runs"], 1)
            self.assertEqual(hil["skipped_commands"], 1)
            self.assertEqual(hil["e_stop_test_passed"], True)
            self.assertIn("BLOCK_UNSAFE_TARGET", hil["block_reason_counts"])
            self.assertIn("unsupported_tool_profile", hil["block_reason_counts"])

    def test_post_action_results_include_raw_observations_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            run = root / "run1"
            run.mkdir(parents=True)
            (run / "post_action_observations.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(_post_action_row("cmd_001", 24, "target_survived")),
                        json.dumps(_post_action_row("cmd_001", 48, "target_survived")),
                        json.dumps(_post_action_row("cmd_001", 72, "target_removed")),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (run / "post_action_summary.json").write_text(
                json.dumps(
                    {
                        "source": "manual_summary",
                        "rows": 3,
                        "commands": 1,
                        "complete_commands": 1,
                        "incomplete_commands": 0,
                        "coverage_rate": 1.0,
                        "delayed_success_rate": 0.0,
                        "crop_damage_rate": 1.0,
                        "regrowth_rate": 0.0,
                        "uncertain_rate": 0.0,
                        "missing_required_observations": [],
                        "latest_outcome_counts": {"crop_damage": 1},
                    }
                ),
                encoding="utf-8",
            )

            rows = write_post_action_results_csv(root, Path(tmp) / "post_action.csv")

            self.assertEqual(len(rows), 2)
            by_source = {row["source"]: row for row in rows}
            self.assertEqual(by_source["post_action_observations.jsonl"]["delayed_success_rate"], 1.0)
            self.assertEqual(by_source["manual_summary"]["crop_damage_rate"], 1.0)

    def test_error_budget_results_include_components_and_completeness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            run = root / "run1"
            run.mkdir(parents=True)
            (run / "error_budget.json").write_text(
                json.dumps(
                    {
                        "schema_version": "error_budget_v0_1",
                        "ok": False,
                        "complete": False,
                        "total_error_mm": 2.4,
                        "max_total_mm": 3.0,
                        "missing_components": ["e_focus"],
                        "components_mm": {
                            "e_detection": 1.0,
                            "e_grid": 0.5,
                            "e_calibration": 1.5,
                            "e_mechanics": 0.8,
                            "e_focus": None,
                            "e_latency": 0.2,
                            "e_biological_target": 1.2,
                        },
                        "metadata": {
                            "calibration_id": "calib_001",
                            "e_calibration_source": "calibration.error_summary_mm.p95",
                        },
                    }
                ),
                encoding="utf-8",
            )

            rows = write_error_budget_results_csv(root, Path(tmp) / "error_budget.csv")

            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["schema_version"], "error_budget_v0_1")
            self.assertEqual(row["complete"], False)
            self.assertEqual(row["e_calibration"], 1.5)
            self.assertEqual(row["calibration_id"], "calib_001")
            self.assertIn("e_focus", row["missing_components"])

    def test_annotation_feedback_results_include_feedback_and_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "feedback.jsonl").write_text(
                json.dumps(
                    {
                        "image_id": "tray001.jpg",
                        "target_id": "target_001",
                        "error_type": "missed_target",
                        "comment": "needs relabel",
                        "priority": "urgent",
                        "proposed_correction": {"target_type": "remove_weed"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "annotation_tasks.jsonl").write_text(
                json.dumps(
                    {
                        "task_id": "task_001",
                        "image_id": "tray001.jpg",
                        "target_id": "target_001",
                        "reason": "missed_target",
                        "priority": "urgent",
                        "status": "open",
                        "proposed_correction": {"target_type": "remove_weed"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = write_annotation_feedback_results_csv(root, Path(tmp) / "feedback.csv")

            self.assertEqual(len(rows), 2)
            by_source = {row["source"]: row for row in rows}
            self.assertEqual(by_source["feedback"]["priority"], "urgent")
            self.assertEqual(by_source["annotation_task"]["status"], "open")
            self.assertIn("remove_weed", by_source["annotation_task"]["proposed_correction"])

    def test_reports_build_cli_writes_hardware_dry_run_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            run = root / "run1"
            run.mkdir(parents=True)
            (run / "hil_pointer_report.json").write_text(
                json.dumps(
                    {
                        "ok": True,
                        "mode": "hardware_in_loop_pointer",
                        "scene_id": "scene_001",
                        "plan_id": "plan_001",
                        "review": {"ok": True},
                        "commands": 1,
                        "executed": 1,
                        "blocked": 0,
                        "results": [{"ok": True}],
                    }
                ),
                encoding="utf-8",
            )
            (run / "dry_run_plan_report.json").write_text(
                json.dumps(
                    {
                        "ok": True,
                        "mode": "dry_run_pointer",
                        "scene_id": "scene_001",
                        "plan_id": "plan_001",
                        "commands": 1,
                        "executed": 1,
                        "blocked": 0,
                        "results": [{"ok": True}],
                    }
                ),
                encoding="utf-8",
            )
            (run / "post_action_summary.json").write_text(
                json.dumps(
                    {
                        "rows": 3,
                        "commands": 1,
                        "complete_commands": 1,
                        "incomplete_commands": 0,
                        "coverage_rate": 1.0,
                        "delayed_success_rate": 1.0,
                        "crop_damage_rate": 0.0,
                        "regrowth_rate": 0.0,
                        "uncertain_rate": 0.0,
                        "missing_required_observations": [],
                        "latest_outcome_counts": {"target_removed": 1},
                    }
                ),
                encoding="utf-8",
            )
            (run / "annotation_tasks.jsonl").write_text(
                json.dumps(
                    {
                        "task_id": "task_001",
                        "image_id": "tray001.jpg",
                        "reason": "bad_cell_state",
                        "priority": "high",
                        "status": "open",
                        "proposed_correction": {"cell_state": "weed_only"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (run / "error_budget.json").write_text(
                json.dumps(
                    {
                        "schema_version": "error_budget_v0_1",
                        "ok": True,
                        "complete": True,
                        "total_error_mm": 2.4,
                        "max_total_mm": 3.0,
                        "missing_components": [],
                        "components_mm": {
                            "e_detection": 1.0,
                            "e_grid": 0.5,
                            "e_calibration": 1.5,
                            "e_mechanics": 0.8,
                            "e_focus": 0.4,
                            "e_latency": 0.2,
                            "e_biological_target": 1.2,
                        },
                        "metadata": {"calibration_id": "calib_001"},
                    }
                ),
                encoding="utf-8",
            )
            (run / "run_snapshot.json").write_text(
                json.dumps(
                    {
                        "command": "prepare",
                        "command_args": {"config": "config.yaml"},
                        "config_hash": "abc123",
                        "config": {"dataset": {"raw_root": "raw"}},
                        "environment": {
                            "python": "3.11",
                            "platform": "test-platform",
                            "packages": {"numpy": "1.0"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            model_registry = Path(tmp) / "model_registry.yaml"
            model_registry.write_text(
                """
version: test_registry
models:
  - model_id: detector_test
    model_type: detector
    framework: test
    artifact_uri: runs/model.pt
    input_schema: ImageInputV1
    output_schema: DetectionResultV1
    status: draft
    safety_level: offline_only
""",
                encoding="utf-8",
            )
            output = Path(tmp) / "report"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_reports",
                    "build",
                    "--root",
                    str(root),
                    "--out",
                    str(output),
                    "--model-registry",
                    str(model_registry),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(completed.stdout)
            hardware_results = (output / "hardware_dry_run_results.csv").read_text(encoding="utf-8")
            safety_results = (output / "safety_results.csv").read_text(encoding="utf-8")
            post_action_results = (output / "post_action_results.csv").read_text(encoding="utf-8")
            error_budget_results = (output / "error_budget_results.csv").read_text(encoding="utf-8")
            run_snapshot_results = (output / "run_snapshot_results.csv").read_text(encoding="utf-8")
            feedback_results = (output / "annotation_feedback_results.csv").read_text(encoding="utf-8")
            model_registry_results = (output / "model_registry.csv").read_text(encoding="utf-8")
            html = (output / "experiment_report.html").read_text(encoding="utf-8")
            self.assertEqual(payload["model_rows"], 1)
            self.assertEqual(payload["hardware_rows"], 2)
            self.assertEqual(payload["safety_rows"], 2)
            self.assertEqual(payload["post_action_rows"], 1)
            self.assertEqual(payload["error_budget_rows"], 1)
            self.assertEqual(payload["run_snapshot_rows"], 1)
            self.assertEqual(payload["annotation_feedback_rows"], 1)
            self.assertIn("hardware_in_loop_pointer", hardware_results)
            self.assertIn("dry_run_pointer", hardware_results)
            self.assertIn("unsafe_action_blocked_count", safety_results)
            self.assertIn("delayed_success_rate", post_action_results)
            self.assertIn("total_error_mm", error_budget_results)
            self.assertIn("calib_001", error_budget_results)
            self.assertIn("abc123", run_snapshot_results)
            self.assertIn("prepare", run_snapshot_results)
            self.assertIn("weed_only", feedback_results)
            self.assertIn("detector_test", model_registry_results)
            self.assertIn("Model Registry", html)
            self.assertIn("Hardware Dry-Run Results", html)
            self.assertIn("Safety Results", html)
            self.assertIn("Post-Action Verification Results", html)
            self.assertIn("Error Budget Results", html)
            self.assertIn("Run Snapshot Results", html)
            self.assertIn("Annotation Feedback Results", html)

    def test_collect_artifacts_includes_rl_critical_events_and_replays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "eval"
            replay_dir = root / "replays"
            replay_dir.mkdir(parents=True)
            (root / "critical_events.json").write_text('{"critical_events":[]}', encoding="utf-8")
            (root / "offline_replay_eval.json").write_text('{"reward_mean":1.0,"critical_error_rate":0.0}', encoding="utf-8")
            (root / "run_snapshot.json").write_text('{"command":"prepare","config_hash":"abc"}', encoding="utf-8")
            (root / "raw_dataset_audit.json").write_text('{"splits":{}}', encoding="utf-8")
            (root / "rl_metrics.jsonl").write_text('{"event":"target_step"}\n', encoding="utf-8")
            (root / "rl_metrics.csv").write_text("event,reward\nstop,0\n", encoding="utf-8")
            (replay_dir / "baseline_episode_0000.json").write_text('{"steps":[]}', encoding="utf-8")

            artifacts = collect_artifacts(root)

            self.assertEqual(
                {path.name for path in artifacts},
                {
                    "critical_events.json",
                    "offline_replay_eval.json",
                    "run_snapshot.json",
                    "raw_dataset_audit.json",
                    "rl_metrics.jsonl",
                    "rl_metrics.csv",
                    "baseline_episode_0000.json",
                },
            )

    def test_rl_results_include_offline_replay_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "offline"
            root.mkdir()
            (root / "offline_replay_eval.json").write_text(
                json.dumps(
                    {
                        "reward_mean": 4.0,
                        "critical_error_rate": 0.25,
                        "successful_target_rate": 0.5,
                        "review_rate": 0.1,
                        "action_events": 8,
                        "total_distance_mm": 12.0,
                        "mean_distance_error_mm": 0.75,
                        "critical_events_count": 2,
                        "allowed": 3,
                        "blocked": 1,
                        "block_rate": 0.25,
                        "crop_damage": 1,
                    }
                ),
                encoding="utf-8",
            )
            out = Path(tmp) / "rl_results.csv"

            rows = write_rl_results_csv(root, out)

            self.assertEqual(rows[0]["policy"], "offline_replay_eval")
            self.assertEqual(rows[0]["actions_per_tray"], 8)
            self.assertEqual(rows[0]["total_distance_mm"], 12.0)
            self.assertEqual(rows[0]["critical_events_count"], 2)
            self.assertEqual(rows[0]["allowed"], 3)
            self.assertEqual(rows[0]["blocked"], 1)
            self.assertEqual(rows[0]["block_rate"], 0.25)
            self.assertEqual(rows[0]["crop_damage"], 1)

    def test_compare_runs_cli_gates_rl_safety_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "report_a"
            run.mkdir()
            (run / "rl_results.csv").write_text(
                "\n".join(
                    [
                        "policy,reward_mean,critical_error_rate,successful_target_rate,critical_events_count,allowed,blocked,block_rate,crop_damage",
                        "rl_policy,4.0,0.2,0.5,2,3,1,0.25,1",
                    ]
                ),
                encoding="utf-8",
            )
            (run / "task_level_results.csv").write_text(
                "\n".join(
                    [
                        "cell_accuracy,edge_cell_accuracy,corner_cell_accuracy,target_recall",
                        "0.9,0.7,0.4,0.8",
                    ]
                ),
                encoding="utf-8",
            )
            (run / "safety_results.csv").write_text(
                "\n".join(
                    [
                        "aborted_runs,skipped_commands",
                        "1,2",
                    ]
                ),
                encoding="utf-8",
            )
            (run / "run_snapshot_results.csv").write_text(
                "\n".join(
                    [
                        "command,config_hash,has_command_args",
                        "prepare,,False",
                    ]
                ),
                encoding="utf-8",
            )
            output = Path(tmp) / "compare"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_reports",
                    "compare-runs",
                    "--runs",
                    str(run),
                    "--out",
                    str(output),
                    "--max-rl-critical-events-count",
                    "1",
                    "--max-rl-blocked-actions",
                    "0",
                    "--max-rl-block-rate",
                    "0.1",
                    "--max-rl-crop-damage-count",
                    "0",
                    "--min-edge-cell-accuracy",
                    "0.8",
                    "--min-corner-cell-accuracy",
                    "0.6",
                    "--max-safety-aborted-runs",
                    "0",
                    "--max-safety-skipped-commands",
                    "0",
                    "--require-run-snapshots",
                    "--max-run-snapshot-missing-config-hash",
                    "0",
                    "--max-run-snapshot-missing-command-args",
                    "0",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            with (output / "compare_runs.csv").open("r", encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            summary = json.loads((output / "compare_runs_summary.json").read_text(encoding="utf-8"))

            self.assertEqual(row["rl_critical_events_count"], "2.0")
            self.assertEqual(row["rl_blocked_actions"], "1.0")
            self.assertEqual(row["rl_block_rate"], "0.25")
            self.assertEqual(row["rl_crop_damage_count"], "1.0")
            self.assertEqual(row["edge_cell_accuracy"], "0.7")
            self.assertEqual(row["corner_cell_accuracy"], "0.4")
            self.assertEqual(row["safety_aborted_runs"], "1.0")
            self.assertEqual(row["safety_skipped_commands"], "2.0")
            self.assertEqual(row["run_snapshot_count"], "1.0")
            self.assertEqual(row["run_snapshot_missing_config_hash_count"], "1.0")
            self.assertEqual(row["run_snapshot_missing_command_args_count"], "1.0")
            self.assertEqual(row["threshold_status"], "fail")
            self.assertIn("rl_block_rate:0.25>0.1", row["threshold_failures"])
            self.assertIn("edge_cell_accuracy:0.7<0.8", row["threshold_failures"])
            self.assertIn("corner_cell_accuracy:0.4<0.6", row["threshold_failures"])
            self.assertIn("safety_aborted_runs:1.0>0.0", row["threshold_failures"])
            self.assertIn("safety_skipped_commands:2.0>0.0", row["threshold_failures"])
            self.assertIn("run_snapshot_missing_config_hash_count:1.0>0.0", row["threshold_failures"])
            self.assertIn("run_snapshot_missing_command_args_count:1.0>0.0", row["threshold_failures"])
            self.assertEqual(summary["failed"], 1)

    def test_write_run_registry_creates_json_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run1"
            run_dir.mkdir()
            (run_dir / "config.yaml").write_text("x: 1\n", encoding="utf-8")
            (run_dir / "run_snapshot.json").write_text("{}", encoding="utf-8")

            run = write_run_registry(run_dir, "test", [run_dir / "config.yaml", run_dir / "run_snapshot.json"])

            self.assertEqual(run.command, "test")
            self.assertTrue((run_dir / "artifact_registry.json").exists())
            self.assertTrue((run_dir / "artifact_registry.csv").exists())

    def test_readiness_report_api_and_cli_write_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            out = Path(tmp) / "readiness"
            cli_out = Path(tmp) / "readiness_cli"
            root.mkdir()
            _write_readiness_calibration(root / "calibration.json")
            (root / "error_map.json").write_text(
                json.dumps({"calibration_id": "calib_001", "rows": [], "error_summary_mm": {"p95": 1.0}}),
                encoding="utf-8",
            )
            (root / "rl_results.csv").write_text(
                "policy,reward_mean\nrl_no_model,0.0\nroute_planning,1.0\n",
                encoding="utf-8",
            )
            (root / "hardware_dry_run_results.csv").write_text("mode,move_success_rate\ndry_run_pointer,1.0\n", encoding="utf-8")
            (root / "safety_results.csv").write_text("source,unsafe_action_blocked_count\ndry_run_pointer,0\n", encoding="utf-8")
            (root / "dry_run_plan_report.json").write_text(
                json.dumps({"mode": "dry_run_pointer", "commands": 1, "executed": 1, "results": [{"ok": True}]}),
                encoding="utf-8",
            )
            (root / "dry_run_plan_commands.json").write_text(json.dumps({"commands": []}), encoding="utf-8")

            report = build_readiness_report(root, out, repo_root=ROOT)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_reports",
                    "readiness-check",
                    "--root",
                    str(root),
                    "--out",
                    str(cli_out),
                    "--repo-root",
                    str(ROOT),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )

            payload = json.loads(completed.stdout)
            items = {item["item_id"]: item for item in report["items"]}
            cli_items = {item["item_id"]: item for item in payload["items"]}
            registry = json.loads((cli_out / "artifact_registry.json").read_text(encoding="utf-8"))

            self.assertTrue((out / "readiness_report.json").exists())
            self.assertTrue((out / "readiness_report.csv").exists())
            self.assertTrue((out / "readiness_report.html").exists())
            self.assertEqual(items["DOD-006"]["status"], "pass")
            self.assertEqual(items["DOD-007"]["status"], "pass")
            self.assertEqual(items["DOD-008"]["status"], "pass")
            self.assertEqual(items["DOD-011"]["status"], "pass")
            self.assertEqual(items["DOD-014"]["status"], "pass")
            self.assertEqual(items["EXT-001"]["status"], "external_required")
            self.assertEqual(cli_items["DOD-015"]["status"], "pass")
            self.assertTrue(payload["physical_validation_required"])
            self.assertTrue(payload["artifact_registry"])
            self.assertEqual(registry["runs"][0]["command"], "seedling-reports:readiness-check")

    def test_readiness_requires_rl_and_baseline_comparison_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            scenario_dir = root / "scenario_compare"
            scenario_dir.mkdir(parents=True)
            compare_path = scenario_dir / "scenario_compare.json"
            compare_path.write_text(
                json.dumps(
                    {
                        "scene_id": "scene_001",
                        "rows": [
                            {"policy_selector": "raster_scan", "policy": "raster_scan"},
                            {"policy_selector": "route_planning", "policy": "route_planning"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            baseline_only = build_readiness_report(root, repo_root=ROOT)
            baseline_item = {item["item_id"]: item for item in baseline_only["items"]}["DOD-011"]

            self.assertEqual(baseline_item["status"], "missing")
            self.assertIn("no RL policy row", baseline_item["notes"])

            compare_path.write_text(
                json.dumps(
                    {
                        "scene_id": "scene_001",
                        "rows": [
                            {"policy_selector": "raster_scan", "policy": "raster_scan"},
                            {"policy_selector": "rl_no_model", "policy": "rl_no_model"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with_rl = build_readiness_report(root, repo_root=ROOT)
            rl_item = {item["item_id"]: item for item in with_rl["items"]}["DOD-011"]

            self.assertEqual(rl_item["status"], "pass")
            self.assertIn(str(compare_path), rl_item["evidence"])

    def test_readiness_rejects_invalid_calibration_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            root.mkdir()
            _write_readiness_calibration(
                root / "calibration.json",
                valid_until="2020-01-01T00:00:00+00:00",
                p95=3.0,
            )
            (root / "error_map.json").write_text(
                json.dumps({"calibration_id": "calib_001", "rows": [], "error_summary_mm": {"p95": 3.0}}),
                encoding="utf-8",
            )

            report = build_readiness_report(root, repo_root=ROOT)
            items = {item["item_id"]: item for item in report["items"]}

            self.assertEqual(items["DOD-006"]["status"], "missing")
            self.assertIn("No valid calibration artifact found", items["DOD-006"]["notes"])
            self.assertIn("calibration expired", items["DOD-006"]["notes"])

    def test_readiness_rejects_invalid_typed_model_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            registry_dir = repo / "configs" / "registry"
            registry_dir.mkdir(parents=True)
            (registry_dir / "model_registry_v0_1.yaml").write_text(
                """
version: model_registry_v0_1
models:
  - model_id: unsafe_rl_policy
    model_type: rl_policy
    framework: sb3_contrib
    artifact_uri: runs/rl/model.zip
    input_schema: SeedlingTrayEnvObservationV1
    output_schema: ActionIndexV1
    status: draft
    safety_level: supervised
    metadata:
      direct_hardware_access: true
      requires_action_mask: false
""",
                encoding="utf-8",
            )

            report = build_readiness_report(Path(tmp) / "run", repo_root=repo)
            items = {item["item_id"]: item for item in report["items"]}

            self.assertEqual(items["DOD-007"]["status"], "missing")
            self.assertIn("rl_policy must remain offline_only/simulation_only/dry_run", items["DOD-007"]["notes"])
            self.assertIn("direct_hardware_access=false", items["DOD-007"]["notes"])

    def test_readiness_rejects_component_registry_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            registry_dir = repo / "configs" / "registry"
            registry_dir.mkdir(parents=True)
            (registry_dir / "policies_v0_1.yaml").write_text(
                """
version: policies_registry_v0_1
entries:
  - entry_id: policy_unsafe_metadata
    kind: policy
    name: unsafe_metadata
    backend: seedling_decision.policies.NoOpPolicy
    status: draft
    metadata:
      safety_level: dry_run
""",
                encoding="utf-8",
            )

            report = build_readiness_report(Path(tmp) / "run", repo_root=repo)
            items = {item["item_id"]: item for item in report["items"]}

            self.assertEqual(items["DOD-003"]["status"], "missing")
            self.assertIn("metadata.requires_safety_gate=true", items["DOD-003"]["notes"])

    def test_readiness_rejects_invalid_dataset_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            dataset_dir = repo / "configs" / "datasets"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "dataset_v0_1.yaml").write_text(
                """
dataset_version: zks_v0_1
root: data/zks_v0_1
metadata: {}
grid:
  rows: 0
  cols: 11
classes:
  0: tray
""",
                encoding="utf-8",
            )

            report = build_readiness_report(Path(tmp) / "run", repo_root=repo)
            items = {item["item_id"]: item for item in report["items"]}

            self.assertEqual(items["DOD-008"]["status"], "missing")
            self.assertIn("ontology_version is required", items["DOD-008"]["notes"])
            self.assertIn("metadata.image_manifest is required", items["DOD-008"]["notes"])
            self.assertIn("grid.rows must be a positive integer", items["DOD-008"]["notes"])
            self.assertIn("classes.0 must be container", items["DOD-008"]["notes"])

    def test_readiness_external_gates_reject_placeholder_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            root.mkdir()
            (root / "hil_pointer_report.json").write_text(
                json.dumps(
                    {
                        "ok": True,
                        "mode": "hardware_in_loop_pointer",
                        "review": {"ok": True},
                        "commands": 1,
                        "executed": 1,
                        "blocked": 0,
                        "results": [{"ok": True}],
                    }
                ),
                encoding="utf-8",
            )
            (root / "post_action_summary.json").write_text(
                json.dumps(
                    {
                        "rows": 3,
                        "commands": 1,
                        "complete_commands": 1,
                        "incomplete_commands": 0,
                        "coverage_rate": 1.0,
                        "delayed_success_rate": 1.0,
                        "crop_damage_rate": 0.0,
                        "regrowth_rate": 0.0,
                        "uncertain_rate": 0.0,
                        "missing_required_observations": [],
                    }
                ),
                encoding="utf-8",
            )

            report = build_readiness_report(root, repo_root=ROOT)
            items = {item["item_id"]: item for item in report["items"]}

            self.assertEqual(items["EXT-001"]["status"], "external_required")
            self.assertIn("review_id is required", items["EXT-001"]["notes"])
            self.assertEqual(items["EXT-002"]["status"], "external_required")
            self.assertIn("required_hours must be present", items["EXT-002"]["notes"])
            self.assertTrue(report["physical_validation_required"])

    def test_readiness_external_gates_pass_with_valid_real_world_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            root.mkdir()
            (root / "hil_pointer_commands.json").write_text(json.dumps({"commands": []}), encoding="utf-8")
            (root / "hil_pointer_replay.json").write_text(json.dumps({"events": []}), encoding="utf-8")
            (root / "hil_pointer_report.json").write_text(
                json.dumps(
                    {
                        "ok": True,
                        "mode": "hardware_in_loop_pointer",
                        "scene_id": "scene_001",
                        "plan_id": "plan_001",
                        "review": {
                            "ok": True,
                            "review_id": "hil-review-001",
                            "gantry_id": "vilga-gantry-dev",
                            "emergency_stop_tested": True,
                            "interlock_required": True,
                            "limit_switch_required": True,
                            "allowed_modes": ["hardware_in_loop_pointer"],
                            "allowed_tool_profiles": ["pointer_only"],
                            "allow_real_actuation": False,
                            "positioning_error_p95_mm": 1.0,
                            "errors": [],
                        },
                        "commands": 1,
                        "executed": 1,
                        "blocked": 0,
                        "skipped": 0,
                        "aborted": False,
                        "command_log": "hil_pointer_commands.json",
                        "replay": "hil_pointer_replay.json",
                        "results": [{"ok": True}],
                    }
                ),
                encoding="utf-8",
            )
            (root / "post_action_observations.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({**_post_action_row("cmd_001", 24, "target_survived"), "observer_id": "observer_a"}),
                        json.dumps({**_post_action_row("cmd_001", 48, "target_survived"), "observer_id": "observer_a"}),
                        json.dumps({**_post_action_row("cmd_001", 72, "target_removed"), "observer_id": "observer_a"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = build_readiness_report(root, repo_root=ROOT)
            items = {item["item_id"]: item for item in report["items"]}

            self.assertEqual(items["EXT-001"]["status"], "pass")
            self.assertEqual(items["EXT-002"]["status"], "pass")
            self.assertFalse(report["physical_validation_required"])

    def test_architecture_backlog_audit_api_and_cli_write_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            doc = ROOT / "docs" / "seedlings_vilga_architecture_codex_rl_simulation.md"
            out = Path(tmp) / "architecture_audit"
            cli_out = Path(tmp) / "architecture_audit_cli"

            parsed = parse_backlog_items(doc)
            report = build_architecture_backlog_audit(doc, out, repo_root=ROOT)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_reports",
                    "architecture-audit",
                    "--doc",
                    str(doc),
                    "--out",
                    str(cli_out),
                    "--repo-root",
                    str(ROOT),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )

            payload = json.loads(completed.stdout)
            items = {item["item_id"]: item for item in report["items"]}
            cli_items = {item["item_id"]: item for item in payload["items"]}
            registry = json.loads((cli_out / "artifact_registry.json").read_text(encoding="utf-8"))

            self.assertGreater(len(parsed), 100)
            self.assertEqual(items["A-001"]["status"], "pass")
            self.assertEqual(items["D-010"]["status"], "pass")
            self.assertEqual(items["I-009"]["status"], "external_required")
            self.assertEqual(cli_items["M-010"]["status"], "pass")
            self.assertTrue((out / "architecture_backlog_audit.json").exists())
            self.assertTrue((out / "architecture_backlog_audit.csv").exists())
            self.assertTrue((out / "architecture_backlog_audit.html").exists())
            self.assertEqual(payload["counts"]["external_required"], 1)
            self.assertNotIn("missing", payload["counts"])
            self.assertTrue(payload["artifact_registry"])
            self.assertEqual(registry["runs"][0]["command"], "seedling-reports:architecture-audit")

    def test_architecture_audit_checks_sensitive_evidence_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            doc = root / "docs" / "backlog.md"
            pyproject = root / "pyproject.toml"
            precommit = root / ".pre-commit-config.yaml"
            ci_workflow = root / ".github" / "workflows" / "ci.yml"
            requirements_base = root / "requirements" / "base.txt"
            requirements_vision = root / "requirements" / "vision.txt"
            requirements_rl = root / "requirements" / "rl.txt"
            requirements_robot = root / "requirements" / "robot.txt"
            requirements_dev = root / "requirements" / "dev.txt"
            ontology_config = root / "configs" / "ontology" / "ontology_v0_1.yaml"
            ontology_py = root / "seedling_data" / "ontology.py"
            manifest_py = root / "seedling_data" / "manifests.py"
            duplicates_py = root / "seedling_data" / "duplicates.py"
            annotations_py = root / "seedling_data" / "annotations.py"
            data_registry_py = root / "seedling_data" / "registry.py"
            changelog_py = root / "seedling_data" / "changelog.py"
            dataset_py = root / "seedling_experiments" / "dataset.py"
            dataset_card_doc = root / "docs" / "DATASET_CARD_TEMPLATE.md"
            annotation_guide_doc = root / "docs" / "ANNOTATION_GUIDE.md"
            dataset_changelog_doc = root / "docs" / "DATASET_CHANGELOG_TEMPLATE.md"
            docs_readme = root / "docs" / "README.md"
            docs_architecture = root / "docs" / "ARCHITECTURE.md"
            docs_rl_spec = root / "docs" / "RL_SPEC.md"
            docs_sim_spec = root / "docs" / "SIMULATION_SPEC.md"
            docs_model_plugin = root / "docs" / "MODEL_PLUGIN_API.md"
            docs_safety = root / "docs" / "SAFETY_CONCEPT.md"
            docs_failure_modes = root / "docs" / "FAILURE_MODES.md"
            docs_article_report = root / "docs" / "ARTICLE_REPORT_TEMPLATE.md"
            docs_operator_manual = root / "docs" / "OPERATOR_MANUAL_DRAFT.md"
            ontology_tests = root / "tests" / "unit" / "test_ontology_and_adapters.py"
            data_tests = root / "tests" / "unit" / "test_data_registry_annotations.py"
            dataset_split_tests = root / "tests" / "unit" / "test_dataset_split.py"
            cli_tests = root / "tests" / "unit" / "test_cli_smoke.py"
            schema_tests = root / "tests" / "unit" / "test_schemas.py"
            migration_tests = root / "tests" / "unit" / "test_vision_migration.py"
            reports_tests = root / "tests" / "unit" / "test_reports_and_viewers.py"
            vision_tests = root / "tests" / "unit" / "test_vision_tools.py"
            registry_selector_tests = root / "tests" / "unit" / "test_registry_selector_feedback.py"
            target_tests = root / "tests" / "unit" / "test_target_generation.py"
            calibration_tests = root / "tests" / "unit" / "test_calibration.py"
            decision_tests = root / "tests" / "unit" / "test_decision_and_safety.py"
            simulation_tests = root / "tests" / "unit" / "test_simulation.py"
            rl_env_tests = root / "tests" / "unit" / "test_rl_env.py"
            rl_training_tests = root / "tests" / "unit" / "test_rl_training_and_compare.py"
            dry_run_tests = root / "tests" / "unit" / "test_dry_run_and_feedback.py"
            robot_tests = root / "tests" / "unit" / "test_robot_and_replay.py"
            run_snapshot = root / "seedling_core" / "run_snapshot.py"
            core_config_py = root / "seedling_core" / "config.py"
            schemas_py = root / "seedling_core" / "schemas.py"
            core_registry_py = root / "seedling_core" / "registry.py"
            experiments_main_py = root / "seedling_experiments" / "__main__.py"
            experiment_cli_py = root / "seedling_experiments" / "cli.py"
            experiment_config_py = root / "seedling_experiments" / "config.py"
            registry_py = root / "seedling_reports" / "registry.py"
            reports_cli_py = root / "seedling_reports" / "cli.py"
            report_tables_py = root / "seedling_reports" / "tables.py"
            report_html_py = root / "seedling_reports" / "html.py"
            scenario = root / "seedling_reports" / "scenario.py"
            model_registry_config = root / "configs" / "registry" / "model_registry_v0_1.yaml"
            detector_base_py = root / "seedling_vision" / "adapters" / "base.py"
            ultralytics_py = root / "seedling_vision" / "adapters" / "ultralytics_yolo.py"
            baseline_green_py = root / "seedling_vision" / "adapters" / "baseline_green.py"
            recorded_prediction_py = root / "seedling_vision" / "adapters" / "recorded_prediction.py"
            onnx_py = root / "seedling_vision" / "adapters" / "onnx.py"
            uncertainty_py = root / "seedling_vision" / "uncertainty.py"
            postprocess_py = root / "seedling_vision" / "postprocess.py"
            overlays_py = root / "seedling_vision" / "overlays.py"
            batch_py = root / "seedling_vision" / "batch.py"
            vision_cli = root / "seedling_vision" / "cli.py"
            migration_py = root / "seedling_vision" / "migration.py"
            cell_state_py = root / "seedling_cells" / "cell_state.py"
            target_generation_py = root / "seedling_cells" / "target_generation.py"
            calibration_schemas_py = root / "seedling_calibration" / "schemas.py"
            calibration_transforms_py = root / "seedling_calibration" / "transforms.py"
            calibration_validator_py = root / "seedling_calibration" / "validator.py"
            calibration_estimation_py = root / "seedling_calibration" / "estimation.py"
            calibration_cli_py = root / "seedling_calibration" / "cli.py"
            calibration_intrinsics_py = root / "seedling_calibration" / "intrinsics.py"
            calibration_doc = root / "docs" / "CALIBRATION_PROTOCOL.md"
            grid_py = root / "seedling_cells" / "grid.py"
            sim_schemas_py = root / "seedling_sim" / "schemas.py"
            sim_logical_tray_py = root / "seedling_sim" / "logical_tray.py"
            sim_detection_noise_py = root / "seedling_sim" / "detection_noise.py"
            sim_actuator_model_py = root / "seedling_sim" / "actuator_model.py"
            sim_plant_response_py = root / "seedling_sim" / "plant_response_model.py"
            sim_scene_generator_py = root / "seedling_sim" / "scene_generator.py"
            sim_image_backed_py = root / "seedling_sim" / "image_backed.py"
            sim_replay_py = root / "seedling_sim" / "replay.py"
            sim_renderers_py = root / "seedling_sim" / "renderers.py"
            sim_domain_randomization_py = root / "seedling_sim" / "domain_randomization.py"
            offline_viewer_py = root / "seedling_ui" / "offline_viewer.py"
            replay_viewer_py = root / "seedling_ui" / "replay_viewer.py"
            ui_cli_py = root / "seedling_ui" / "cli.py"
            report_export_py = root / "seedling_ui" / "report_export.py"
            feedback_py = root / "seedling_ui" / "feedback.py"
            safety_gate_py = root / "seedling_decision" / "safety_gate.py"
            action_masks_py = root / "seedling_decision" / "action_masks.py"
            policies_base_py = root / "seedling_decision" / "policies" / "base.py"
            policies_builtin_py = root / "seedling_decision" / "policies" / "builtin.py"
            protocol_py = root / "seedling_robot" / "protocol.py"
            dry_run_adapter_py = root / "seedling_robot" / "adapters" / "dry_run_serial.py"
            evaluate_py = root / "seedling_experiments" / "evaluate.py"
            scene_evaluation_py = root / "seedling_cells" / "evaluation.py"
            diagnostics_py = root / "seedling_reports" / "diagnostics.py"
            grid_tests = root / "tests" / "unit" / "test_grid.py"
            evaluate_tests = root / "tests" / "unit" / "test_evaluate_cells_metrics.py"
            scene_eval_tests = root / "tests" / "unit" / "test_scene_state_evaluation.py"
            integration_tests = root / "tests" / "integration" / "test_scene_policy_sim.py"
            compare = root / "seedling_reports" / "compare.py"
            performance = root / "tests" / "performance" / "test_performance_smoke.py"
            ontology_config.parent.mkdir(parents=True)
            requirements_base.parent.mkdir(parents=True)
            ci_workflow.parent.mkdir(parents=True)
            model_registry_config.parent.mkdir(parents=True)
            ontology_py.parent.mkdir(parents=True)
            ontology_tests.parent.mkdir(parents=True)
            run_snapshot.parent.mkdir(parents=True)
            experiment_cli_py.parent.mkdir(parents=True)
            scenario.parent.mkdir(parents=True)
            detector_base_py.parent.mkdir(parents=True)
            batch_py.parent.mkdir(parents=True, exist_ok=True)
            cell_state_py.parent.mkdir(parents=True)
            calibration_schemas_py.parent.mkdir(parents=True)
            sim_schemas_py.parent.mkdir(parents=True)
            replay_viewer_py.parent.mkdir(parents=True)
            safety_gate_py.parent.mkdir(parents=True)
            action_masks_py.parent.mkdir(parents=True, exist_ok=True)
            policies_builtin_py.parent.mkdir(parents=True)
            protocol_py.parent.mkdir(parents=True)
            dry_run_adapter_py.parent.mkdir(parents=True)
            performance.parent.mkdir(parents=True)
            integration_tests.parent.mkdir(parents=True)
            doc.parent.mkdir(parents=True)
            doc.write_text(
                "\n".join(
                    [
                        "| ID | Priority | Title | Acceptance |",
                        "| --- | --- | --- | --- |",
                        "| A-001 | P0 | package skeleton | Python package entry points exist. |",
                        "| A-002 | P0 | requirements split | Base, vision, RL, robot and dev requirements are split. |",
                        "| A-003 | P0 | config loader | Configs validate sections, fields and paths. |",
                        "| A-004 | P0 | shared schemas | Shared dataclasses serialize and validate. |",
                        "| A-006 | P1 | artifact registry | Every run creates artifact_registry.json with input/output links. |",
                        "| A-007 | P1 | migration tools | Legacy predictions convert to SceneState. |",
                        "| A-008 | P1 | CI and pre-commit | Checks run formatting, lint and tests. |",
                        "| B-006 | P0 | image_manifest.csv generator | Every image has hash, size, group_id, session_id and tray_id. |",
                        "| B-008 | P0 | near-duplicate check | Hash and perceptual hash are checked between splits. |",
                        "| B-003 | P1 | dataset card template | Dataset card captures identity, source, split and safety limits. |",
                        "| B-004 | P0 | annotation guide | Annotation guide defines object, cell and action-point labeling. |",
                        "| B-005 | P0 | grouped split | Groups are split before augmentation. |",
                        "| B-007 | P0 | augmentation leakage check | Val/test have no augmented-source leakage. |",
                        "| B-009 | P0 | cell annotations | Cell annotation schema and audit exist. |",
                        "| B-010 | P0 | action point annotations | Expert target/action-point annotations exist. |",
                        "| B-011 | P0 | cell/action consistency | Action points are cross-checked against cell annotations. |",
                        "| B-012 | P1 | safety annotation flags | Unsafe/high-uncertainty targets require review. |",
                        "| B-013 | P1 | dataset summary report | Markdown/HTML summary includes split/classes/cells/actions. |",
                        "| B-014 | P1 | dataset changelog | Dataset versions have immutable changelog entries. |",
                        "| C-001 | P0 | DetectorAdapter interface | Detectors expose load, predict and metadata. |",
                        "| C-002 | P0 | Ultralytics YOLO adapter | Predict returns DetectionResultV1. |",
                        "| C-003 | P0 | baseline green adapter | HSV baseline returns detector output schema. |",
                        "| C-004 | P0 | recorded prediction adapter | Existing predictions.json can be replayed. |",
                        "| C-005 | P1 | ONNX adapter | ONNX models connect without Ultralytics runtime. |",
                        "| C-006 | P1 | uncertainty estimator | Confidence, bbox and entropy flags are exposed. |",
                        "| C-007 | P1 | postprocess containers | Evaluator can use explicit container outputs. |",
                        "| C-008 | P1 | model metadata | Registry stores model/dataset/ontology/schema metadata. |",
                        "| C-009 | P2 | batch inference | Batch image processing and progress logs are supported. |",
                        "| C-010 | P2 | visual overlay report | Bbox, grid, targets and errors are drawn. |",
                        "| D-001 | P0 | grid builder | Tray bbox/corners split into rows and columns. |",
                        "| D-002 | P0 | CellState schema | Cell states track classes, objects, review flags and risk. |",
                        "| D-003 | P0 | CellStateBuilder | Detections and tray geometry produce CellState rows. |",
                        "| D-004 | P0 | target generation module | Largest-bbox baseline is a strategy, not hardcoded. |",
                        "| D-005 | P0 | cell metrics | Cell accuracy, macro F1 and target metrics are computed. |",
                        "| D-006 | P1 | container matching diagnostics | Missing/low-IoU container cases are reported. |",
                        "| D-007 | P1 | cost-sensitive metrics | Critical biological/decision errors have separate costs. |",
                        "| D-008 | P1 | SceneState evaluator | Rich cell states and targets are evaluated. |",
                        "| D-009 | P1 | edge/corner metrics | Edge and corner cells are summarized separately. |",
                        "| D-010 | P1 | error taxonomy | Errors are grouped by image, geometry, biology and decision. |",
                        "| D-011 | P2 | bootstrap CI | Bootstrap intervals are generated for core metrics. |",
                        "| D-012 | P2 | error taxonomy report | Diagnostics writes taxonomy artifacts. |",
                        "| E-001 | P0 | Calibration schemas | JSON contains homography, transforms, px/mm and error summary. |",
                        "| E-002 | P0 | image px to tray mm transform | Known points map correctly. |",
                        "| E-003 | P0 | tray mm to robot frame transform | Tool offset is supported. |",
                        "| E-004 | P0 | calibration validator | Expiry, RMS/P95 and tray type are checked. |",
                        "| E-005 | P1 | error map generation | Heatmap report is generated. |",
                        "| E-006 | P1 | coordinate conversion in targets | ActionTarget receives px/mm/robot coordinates. |",
                        "| E-007 | P1 | calibration CLI | estimate/validate/error-map commands create artifacts. |",
                        "| E-008 | P2 | camera intrinsics calibration | chessboard/aruco/fiducials are supported. |",
                        "| F-001 | P0 | DecisionPolicy interface | Policies are SceneState to ActionPlan compatible. |",
                        "| F-002 | P0 | NoOpPolicy | Targets are sent to review. |",
                        "| F-003 | P0 | RasterScanPolicy | Safe targets are ordered by row and column. |",
                        "| F-004 | P0 | NearestNeighborPolicy | Targets are ordered by robot distance. |",
                        "| F-005 | P0 | RiskAwareRulePolicy | Risk rules filter unsafe targets. |",
                        "| F-006 | P0 | ActionMaskBuilder | Unsafe, invalid and processed actions are masked. |",
                        "| F-007 | P0 | SafetyGate | Action execution requires a safety decision. |",
                        "| F-008 | P0 | forbidden zones | Risk rules account for keep zones and tray bounds. |",
                        "| F-009 | P1 | HumanReviewPolicy | Runtime gets review targets with reasons. |",
                        "| F-010 | P1 | policy comparison metrics | Policies are compared on one scene. |",
                        "| F-011 | P2 | RoutePlanningPolicy | Route baseline minimizes movement greedily. |",
                        "| G-001 | P0 | SimScene schema | Scenes serialize to JSON and validate. |",
                        "| G-002 | P0 | logical tray simulator | Scene, plants, targets and robot state step outcomes. |",
                        "| G-003 | P0 | detection noise model | Misses, false positives and class noise are configurable. |",
                        "| G-004 | P0 | actuator error model | Mechanical error, latency, drift and zone error exist. |",
                        "| G-005 | P0 | plant response model stub | Probabilistic success/damage without real laser parameters. |",
                        "| G-006 | P0 | scene generator | empty/single/multiple/weed/unknown scenes can be generated. |",
                        "| G-007 | P1 | image-backed scenes | Scenes build from real SceneState evidence. |",
                        "| G-008 | P1 | replay logger | Each step stores obs/action/reward/info. |",
                        "| G-009 | P1 | static renderer | SVG/HTML/PNG render grid, plants, targets and path. |",
                        "| G-010 | P1 | policy replay viewer | Actions can be stepped through visually. |",
                        "| G-011 | P2 | synthetic image renderer | Simple 2D images are generated. |",
                        "| G-012 | P2 | domain randomization presets | Lighting/focus/noise/calibration drift presets exist. |",
                        "| I-004 | P0 | dry-run serial adapter | Commands run in safe mode without dangerous actuation. |",
                        "| I-005 | P0 | robot telemetry | SafetyGate sees robot_homed and limit status. |",
                        "| J-001 | P0 | offline viewer | Loads image, predictions, cell state and targets. |",
                        "| J-002 | P0 | simulation replay viewer | Play/pause/step/reset, timeline actions. |",
                        "| J-003 | P0 | review/block reasons | Operator sees why a target is blocked. |",
                        "| J-004 | P1 | model/policy selector | Detector and policy can be selected from registry. |",
                        "| J-006 | P1 | report export | HTML/Markdown report for tray or episode. |",
                        "| J-007 | P2 | annotation feedback mode | Operator can mark model error for relabeling. |",
                        "| K-001 | P0 | unified experiment CLI | prepare/train/val/predict/evaluate/sim/rl have one style. |",
                        "| K-002 | P0 | experiment registry | Each command registers inputs and outputs. |",
                        "| K-003 | P0 | dataset_summary.csv | Table is built from artifacts. |",
                        "| K-004 | P0 | object_level_results.csv | Source is test_metrics.json. |",
                        "| K-005 | P0 | task_level_results.csv | Source is cell_metrics.json. |",
                        "| K-006 | P1 | artifact_registry.csv | Artifact table supports paper and reproducibility. |",
                        "| K-007 | P1 | HTML report generator | One experiment report includes images and tables. |",
                        "| L-001 | P0 | Schema unit tests | Schema tests cover object geometry and round trips. |",
                        "| L-002 | P0 | Grid unit tests | Grid tests cover boundaries, assignment and polygons. |",
                        "| L-003 | P0 | Target generation tests | Target tests cover crop, weed and review cases. |",
                        "| L-004 | P0 | Safety and policy tests | Safety tests cover masks and robot telemetry. |",
                        "| L-005 | P0 | Calibration tests | Calibration tests cover transforms, estimation and validation. |",
                        "| L-006 | P0 | Policy integration test | Rule policy completes a scene through SafetyGate. |",
                        "| L-007 | P0 | Simulator integration test | Robot adapter executes simulator commands. |",
                        "| L-008 | P1 | RL and compare tests | RL train, policy adapter and seed stability are covered. |",
                        "| L-009 | P1 | Cell metrics tests | Metrics tests cover calibrated target errors and scene inputs. |",
                        "| L-010 | P1 | CLI smoke tests | CLI help, requirements and dry-run commands are covered. |",
                        "| M-001 | P0 | README docs | README links documents, quick start and reports. |",
                        "| M-002 | P0 | Architecture docs | Architecture covers skeleton, flow, safety and registries. |",
                        "| M-003 | P0 | RL spec docs | RL spec covers env, rewards, algorithms and sweeps. |",
                        "| M-004 | P0 | Simulation spec docs | Simulation spec covers simulator, noise and run-policy. |",
                        "| M-005 | P0 | Plugin API docs | Plugin docs cover detector, policy and safety contracts. |",
                        "| M-006 | P0 | Safety concept docs | Safety concept covers gates, blocks and HIL review. |",
                        "| M-007 | P0 | Calibration protocol docs | Calibration docs cover artifact, validator and registry. |",
                        "| M-008 | P1 | Failure modes docs | Failure docs cover detection and mitigation rows. |",
                        "| M-009 | P1 | Article report template | Report template covers sources, tables and reproducibility. |",
                        "| M-010 | P1 | Operator manual draft | Manual covers modes, dry-run, feedback and verification. |",
                        "| A-005 | P1 | Добавить run_snapshot как общий сервис | Все команды сохраняют версию Python, платформу, command args, config hash. |",
                        "| B-001 | P0 | Добавить configs/ontology/ontology_v0_1.yaml | Есть классы объектов, cell states, action labels, attributes. |",
                        "| B-002 | P0 | Реализовать seedling_data.ontology | Онтология валидируется; class names совпадают с датасетом. |",
                        "| J-005 | P1 | Добавить scenario comparison | На одной сцене сравниваются rule-based и RL. |",
                        "| K-008 | P2 | Добавить compare-runs | Сравнение моделей, thresholds, RL seeds. |",
                        "| L-011 | P2 | Performance tests | Batch inference and sim training не деградируют. |",
                    ]
                ),
                encoding="utf-8",
            )
            pyproject.write_text("[project]\nname='stub'\n", encoding="utf-8")
            precommit.write_text("repos: []\n", encoding="utf-8")
            ci_workflow.write_text("name: CI\n", encoding="utf-8")
            requirements_base.write_text("numpy\n", encoding="utf-8")
            requirements_vision.write_text("opencv-python\n", encoding="utf-8")
            requirements_rl.write_text("gymnasium\n", encoding="utf-8")
            requirements_robot.write_text("", encoding="utf-8")
            requirements_dev.write_text("pytest\n", encoding="utf-8")
            ontology_config.write_text("version: ontology_v0_1\n", encoding="utf-8")
            ontology_py.write_text("def load_ontology(): pass\n", encoding="utf-8")
            core_config_py.write_text("class ConfigError: pass\n", encoding="utf-8")
            schemas_py.write_text("class SceneState: pass\n", encoding="utf-8")
            experiments_main_py.write_text("from .cli import main\n", encoding="utf-8")
            migration_py.write_text("def legacy_predictions_to_scenes(): pass\n", encoding="utf-8")
            ontology_tests.write_text("def test_ontology_accepts_current_seedlings_alias(): pass\n", encoding="utf-8")
            schema_tests.write_text("def test_schema_smoke(): pass\n", encoding="utf-8")
            migration_tests.write_text("def test_migration_smoke(): pass\n", encoding="utf-8")
            manifest_py.write_text("def build_image_manifest(): pass\n", encoding="utf-8")
            duplicates_py.write_text("def find_cross_split_duplicates(): pass\n", encoding="utf-8")
            annotations_py.write_text("class CellAnnotation: pass\n", encoding="utf-8")
            data_registry_py.write_text("class DatasetRegistry: pass\n", encoding="utf-8")
            changelog_py.write_text("def audit_dataset_changelog(): pass\n", encoding="utf-8")
            dataset_py.write_text("def make_grouped_split(): pass\n", encoding="utf-8")
            dataset_card_doc.write_text("# Dataset Card\n", encoding="utf-8")
            annotation_guide_doc.write_text("# Annotation Guide\n", encoding="utf-8")
            dataset_changelog_doc.write_text("# Dataset Changelog\n", encoding="utf-8")
            docs_readme.write_text("# README\n", encoding="utf-8")
            docs_architecture.write_text("# Architecture\n", encoding="utf-8")
            docs_rl_spec.write_text("# RL\n", encoding="utf-8")
            docs_sim_spec.write_text("# Simulation\n", encoding="utf-8")
            docs_model_plugin.write_text("# Plugin API\n", encoding="utf-8")
            docs_safety.write_text("# Safety\n", encoding="utf-8")
            docs_failure_modes.write_text("# Failure Modes\n", encoding="utf-8")
            docs_article_report.write_text("# Article Report\n", encoding="utf-8")
            docs_operator_manual.write_text("# Operator Manual\n", encoding="utf-8")
            data_tests.write_text("def test_data_smoke(): pass\n", encoding="utf-8")
            dataset_split_tests.write_text("def test_split_smoke(): pass\n", encoding="utf-8")
            reports_tests.write_text("def test_reports_smoke(): pass\n", encoding="utf-8")
            core_registry_py.write_text("class ModelRegistry: pass\n", encoding="utf-8")
            experiment_cli_py.write_text("def main(): pass\n", encoding="utf-8")
            experiment_config_py.write_text("def load_config(): pass\n", encoding="utf-8")
            report_tables_py.write_text("def collect_artifacts(): pass\n", encoding="utf-8")
            report_html_py.write_text("def write_experiment_html_report(): pass\n", encoding="utf-8")
            reports_cli_py.write_text("def main(): pass\n", encoding="utf-8")
            model_registry_config.write_text("version: model_registry_v0_1\n", encoding="utf-8")
            detector_base_py.write_text("class DetectorAdapter: pass\n", encoding="utf-8")
            ultralytics_py.write_text("class UltralyticsYOLODetector: pass\n", encoding="utf-8")
            baseline_green_py.write_text("class BaselineGreenDetector: pass\n", encoding="utf-8")
            recorded_prediction_py.write_text("class RecordedPredictionDetector: pass\n", encoding="utf-8")
            onnx_py.write_text("class ONNXDetector: pass\n", encoding="utf-8")
            uncertainty_py.write_text("class UncertaintyConfig: pass\n", encoding="utf-8")
            postprocess_py.write_text("def filter_and_merge_containers(): pass\n", encoding="utf-8")
            overlays_py.write_text("def write_detection_overlay(): pass\n", encoding="utf-8")
            batch_py.write_text("def batch_predict(): pass\n", encoding="utf-8")
            vision_cli.write_text("def main(): pass\n", encoding="utf-8")
            vision_tests.write_text("def test_vision_smoke(): pass\n", encoding="utf-8")
            registry_selector_tests.write_text("def test_registry_smoke(): pass\n", encoding="utf-8")
            cell_state_py.write_text("class CellStateBuilder: pass\n", encoding="utf-8")
            grid_py.write_text("def generate_grid(): pass\n", encoding="utf-8")
            target_generation_py.write_text("def generate_targets(): pass\n", encoding="utf-8")
            target_tests.write_text("def test_target_smoke(): pass\n", encoding="utf-8")
            evaluate_py.write_text("def evaluate_cells_from_config(): pass\n", encoding="utf-8")
            scene_evaluation_py.write_text("def evaluate_scene_states(): pass\n", encoding="utf-8")
            diagnostics_py.write_text("def error_taxonomy(): pass\n", encoding="utf-8")
            grid_tests.write_text("def test_grid_smoke(): pass\n", encoding="utf-8")
            evaluate_tests.write_text("def test_evaluate_smoke(): pass\n", encoding="utf-8")
            scene_eval_tests.write_text("def test_scene_eval_smoke(): pass\n", encoding="utf-8")
            calibration_schemas_py.write_text("class CalibrationArtifact: pass\n", encoding="utf-8")
            calibration_transforms_py.write_text("def image_px_to_tray_mm(): pass\n", encoding="utf-8")
            calibration_validator_py.write_text("class CalibrationValidator: pass\n", encoding="utf-8")
            calibration_estimation_py.write_text("def write_error_map(): pass\n", encoding="utf-8")
            calibration_cli_py.write_text("def main(): pass\n", encoding="utf-8")
            calibration_intrinsics_py.write_text("class CameraIntrinsicsArtifact: pass\n", encoding="utf-8")
            calibration_doc.write_text("# Calibration\n", encoding="utf-8")
            calibration_tests.write_text("def test_calibration_smoke(): pass\n", encoding="utf-8")
            sim_schemas_py.write_text("class SimScene: pass\n", encoding="utf-8")
            sim_logical_tray_py.write_text("class LogicalTraySimulator: pass\n", encoding="utf-8")
            sim_detection_noise_py.write_text("class DetectionNoiseModel: pass\n", encoding="utf-8")
            sim_actuator_model_py.write_text("class ActuatorErrorModel: pass\n", encoding="utf-8")
            sim_plant_response_py.write_text("class PlantResponseModel: pass\n", encoding="utf-8")
            sim_scene_generator_py.write_text("class SimSceneGenerator: pass\n", encoding="utf-8")
            sim_image_backed_py.write_text("def sim_scene_from_scene_state(): pass\n", encoding="utf-8")
            sim_replay_py.write_text("class ReplayLogger: pass\n", encoding="utf-8")
            sim_renderers_py.write_text("def render_scene_svg(): pass\n", encoding="utf-8")
            sim_domain_randomization_py.write_text("class DomainRandomizationConfig: pass\n", encoding="utf-8")
            offline_viewer_py.write_text("def write_offline_viewer(): pass\n", encoding="utf-8")
            replay_viewer_py.write_text("def write_replay_viewer(): pass\n", encoding="utf-8")
            ui_cli_py.write_text("def main(): pass\n", encoding="utf-8")
            report_export_py.write_text("def write_report_export(): pass\n", encoding="utf-8")
            feedback_py.write_text("class AnnotationFeedback: pass\n", encoding="utf-8")
            simulation_tests.write_text("def test_sim_smoke(): pass\n", encoding="utf-8")
            rl_env_tests.write_text("def test_rl_env_smoke(): pass\n", encoding="utf-8")
            rl_training_tests.write_text("def test_rl_training_smoke(): pass\n", encoding="utf-8")
            integration_tests.write_text("def test_integration_smoke(): pass\n", encoding="utf-8")
            safety_gate_py.write_text("class SafetyGate: pass\n", encoding="utf-8")
            action_masks_py.write_text("class ActionMaskBuilder: pass\n", encoding="utf-8")
            policies_base_py.write_text("class DecisionPolicy: pass\n", encoding="utf-8")
            policies_builtin_py.write_text("class RiskAwareRulePolicy: pass\n", encoding="utf-8")
            protocol_py.write_text("class RobotExecutionResult: pass\n", encoding="utf-8")
            dry_run_adapter_py.write_text("class DryRunSerialAdapter: pass\n", encoding="utf-8")
            decision_tests.write_text("def test_safety_smoke(): pass\n", encoding="utf-8")
            dry_run_tests.write_text("def test_dry_run_smoke(): pass\n", encoding="utf-8")
            robot_tests.write_text("def test_robot_smoke(): pass\n", encoding="utf-8")
            run_snapshot.write_text("def save_command_snapshot(): pass\n", encoding="utf-8")
            registry_py.write_text("class ArtifactRecord: pass\n", encoding="utf-8")
            cli_tests.write_text("def test_cli_smoke(): pass\n", encoding="utf-8")
            scenario.write_text("def compare_policies_on_scene(): pass\n", encoding="utf-8")
            compare.write_text("def compare_runs(): pass\nmin_rl_seed_count = True\n", encoding="utf-8")
            performance.write_text("from seedling_vision import batch_predict\n", encoding="utf-8")

            missing_content = build_architecture_backlog_audit(doc, repo_root=root)
            missing_items = {item["item_id"]: item for item in missing_content["items"]}

            self.assertEqual(missing_items["A-001"]["status"], "missing")
            self.assertIn("seedling-experiments", missing_items["A-001"]["notes"])
            self.assertEqual(missing_items["A-002"]["status"], "missing")
            self.assertIn("PyYAML", missing_items["A-002"]["notes"])
            self.assertEqual(missing_items["A-003"]["status"], "missing")
            self.assertIn("FieldSpec", missing_items["A-003"]["notes"])
            self.assertEqual(missing_items["A-004"]["status"], "missing")
            self.assertIn("DetectionObject", missing_items["A-004"]["notes"])
            self.assertEqual(missing_items["A-005"]["status"], "missing")
            self.assertIn("command_args", missing_items["A-005"]["notes"])
            self.assertEqual(missing_items["A-006"]["status"], "missing")
            self.assertIn("artifact_registry.csv", missing_items["A-006"]["notes"])
            self.assertEqual(missing_items["A-007"]["status"], "missing")
            self.assertIn("legacy_prediction_to_scene", missing_items["A-007"]["notes"])
            self.assertEqual(missing_items["A-008"]["status"], "missing")
            self.assertIn("ruff-pre-commit", missing_items["A-008"]["notes"])
            self.assertEqual(missing_items["B-001"]["status"], "missing")
            self.assertIn("unknown_plant", missing_items["B-001"]["notes"])
            self.assertEqual(missing_items["B-002"]["status"], "missing")
            self.assertIn("REQUIRED_OBJECT_CLASSES", missing_items["B-002"]["notes"])
            self.assertEqual(missing_items["B-006"]["status"], "missing")
            self.assertIn("group_id", missing_items["B-006"]["notes"])
            self.assertEqual(missing_items["B-008"]["status"], "missing")
            self.assertIn("missing_split_rows", missing_items["B-008"]["notes"])
            self.assertEqual(missing_items["B-003"]["status"], "missing")
            self.assertIn("Ограничения безопасности и использования", missing_items["B-003"]["notes"])
            self.assertEqual(missing_items["B-004"]["status"], "missing")
            self.assertIn("Точки действия", missing_items["B-004"]["notes"])
            self.assertEqual(missing_items["B-005"]["status"], "missing")
            self.assertIn("metadata_group_column", missing_items["B-005"]["notes"])
            self.assertEqual(missing_items["B-007"]["status"], "missing")
            self.assertIn("cross_split_base_leakage", missing_items["B-007"]["notes"])
            self.assertEqual(missing_items["B-009"]["status"], "missing")
            self.assertIn("audit_cell_annotations", missing_items["B-009"]["notes"])
            self.assertEqual(missing_items["B-010"]["status"], "missing")
            self.assertIn("audit_action_points", missing_items["B-010"]["notes"])
            self.assertEqual(missing_items["B-011"]["status"], "missing")
            self.assertIn("crop_and_weed action point", missing_items["B-011"]["notes"])
            self.assertEqual(missing_items["B-012"]["status"], "missing")
            self.assertIn("unsafe_distance_rows", missing_items["B-012"]["notes"])
            self.assertEqual(missing_items["B-013"]["status"], "missing")
            self.assertIn("DatasetRegistryEntry", missing_items["B-013"]["notes"])
            self.assertEqual(missing_items["B-014"]["status"], "missing")
            self.assertIn("REQUIRED_FIELDS", missing_items["B-014"]["notes"])
            self.assertEqual(missing_items["C-001"]["status"], "missing")
            self.assertIn("DetectorAdapter(ABC)", missing_items["C-001"]["notes"])
            self.assertEqual(missing_items["C-002"]["status"], "missing")
            self.assertIn("from ultralytics import YOLO", missing_items["C-002"]["notes"])
            self.assertEqual(missing_items["C-003"]["status"], "missing")
            self.assertIn("cv2.inRange", missing_items["C-003"]["notes"])
            self.assertEqual(missing_items["C-004"]["status"], "missing")
            self.assertIn("_record_to_detection", missing_items["C-004"]["notes"])
            self.assertEqual(missing_items["C-005"]["status"], "missing")
            self.assertIn("parse_onnx_detections", missing_items["C-005"]["notes"])
            self.assertEqual(missing_items["C-006"]["status"], "missing")
            self.assertIn("high_entropy_threshold", missing_items["C-006"]["notes"])
            self.assertEqual(missing_items["C-007"]["status"], "missing")
            self.assertIn("container_matching_summary", missing_items["C-007"]["notes"])
            self.assertEqual(missing_items["C-008"]["status"], "missing")
            self.assertIn("calibration_requirements", missing_items["C-008"]["notes"])
            self.assertEqual(missing_items["C-009"]["status"], "missing")
            self.assertIn("progress", missing_items["C-009"]["notes"])
            self.assertEqual(missing_items["C-010"]["status"], "missing")
            self.assertIn("_draw_target_marker", missing_items["C-010"]["notes"])
            self.assertEqual(missing_items["D-001"]["status"], "missing")
            self.assertIn("GridCell", missing_items["D-001"]["notes"])
            self.assertEqual(missing_items["D-002"]["status"], "missing")
            self.assertIn("unknown_class_names", missing_items["D-002"]["notes"])
            self.assertEqual(missing_items["D-003"]["status"], "missing")
            self.assertIn("generate_grid_cells", missing_items["D-003"]["notes"])
            self.assertEqual(missing_items["D-004"]["status"], "missing")
            self.assertIn("LargestBBoxTargetStrategy", missing_items["D-004"]["notes"])
            self.assertEqual(missing_items["D-005"]["status"], "missing")
            self.assertIn("_macro_metrics", missing_items["D-005"]["notes"])
            self.assertEqual(missing_items["D-006"]["status"], "missing")
            self.assertIn("container_matching", missing_items["D-006"]["notes"])
            self.assertEqual(missing_items["D-007"]["status"], "missing")
            self.assertIn("critical_error_counts", missing_items["D-007"]["notes"])
            self.assertEqual(missing_items["D-008"]["status"], "missing")
            self.assertIn("DEFAULT_STATE_ORDER", missing_items["D-008"]["notes"])
            self.assertEqual(missing_items["D-009"]["status"], "missing")
            self.assertIn("edge_cell_metrics", missing_items["D-009"]["notes"])
            self.assertEqual(missing_items["D-010"]["status"], "missing")
            self.assertIn("ERROR_CATEGORY_GROUPS", missing_items["D-010"]["notes"])
            self.assertEqual(missing_items["D-011"]["status"], "missing")
            self.assertIn("bootstrap_ci.json", missing_items["D-011"]["notes"])
            self.assertEqual(missing_items["D-012"]["status"], "missing")
            self.assertIn("target_coordinate_error", missing_items["D-012"]["notes"])
            self.assertEqual(missing_items["E-001"]["status"], "missing")
            self.assertIn("CALIBRATION_SCHEMA_VERSION", missing_items["E-001"]["notes"])
            self.assertEqual(missing_items["E-002"]["status"], "missing")
            self.assertIn("apply_homography", missing_items["E-002"]["notes"])
            self.assertEqual(missing_items["E-003"]["status"], "missing")
            self.assertIn("tray_mm_to_robot_frame_mm", missing_items["E-003"]["notes"])
            self.assertEqual(missing_items["E-004"]["status"], "missing")
            self.assertIn("max_error_rms_mm", missing_items["E-004"]["notes"])
            self.assertEqual(missing_items["E-005"]["status"], "missing")
            self.assertIn("_error_map_html", missing_items["E-005"]["notes"])
            self.assertEqual(missing_items["E-006"]["status"], "missing")
            self.assertIn("attach_calibration_to_scene", missing_items["E-006"]["notes"])
            self.assertEqual(missing_items["E-007"]["status"], "missing")
            self.assertIn("seedling-calibration:validate", missing_items["E-007"]["notes"])
            self.assertEqual(missing_items["E-008"]["status"], "missing")
            self.assertIn("SUPPORTED_TARGET_TYPES", missing_items["E-008"]["notes"])
            self.assertEqual(missing_items["F-001"]["status"], "missing")
            self.assertIn("processed_target_ids", missing_items["F-001"]["notes"])
            self.assertEqual(missing_items["F-002"]["status"], "missing")
            self.assertIn("noop_policy_requires_review", missing_items["F-002"]["notes"])
            self.assertEqual(missing_items["F-003"]["status"], "missing")
            self.assertIn("_ordered_targets", missing_items["F-003"]["notes"])
            self.assertEqual(missing_items["F-004"]["status"], "missing")
            self.assertIn("NearestNeighborPolicy", missing_items["F-004"]["notes"])
            self.assertEqual(missing_items["F-005"]["status"], "missing")
            self.assertIn("max_risk_score", missing_items["F-005"]["notes"])
            self.assertEqual(missing_items["F-006"]["status"], "missing")
            self.assertIn("processed_target_ids", missing_items["F-006"]["notes"])
            self.assertEqual(missing_items["F-007"]["status"], "missing")
            self.assertIn("def validate", missing_items["F-007"]["notes"])
            self.assertEqual(missing_items["F-008"]["status"], "missing")
            self.assertIn("forbidden_zone_overlap", missing_items["F-008"]["notes"])
            self.assertEqual(missing_items["F-009"]["status"], "missing")
            self.assertIn("HumanReviewPolicy", missing_items["F-009"]["notes"])
            self.assertEqual(missing_items["F-010"]["status"], "missing")
            self.assertIn("route_distance_mm", missing_items["F-010"]["notes"])
            self.assertEqual(missing_items["F-011"]["status"], "missing")
            self.assertIn("RoutePlanningPolicy", missing_items["F-011"]["notes"])
            self.assertEqual(missing_items["G-001"]["status"], "missing")
            self.assertIn("SIM_SCENE_SCHEMA_VERSION", missing_items["G-001"]["notes"])
            self.assertEqual(missing_items["G-002"]["status"], "missing")
            self.assertIn("step_target", missing_items["G-002"]["notes"])
            self.assertEqual(missing_items["G-003"]["status"], "missing")
            self.assertIn("false_positive_noise", missing_items["G-003"]["notes"])
            self.assertEqual(missing_items["G-004"]["status"], "missing")
            self.assertIn("zone_error_active", missing_items["G-004"]["notes"])
            self.assertEqual(missing_items["G-005"]["status"], "missing")
            self.assertIn("biological_nonresponse", missing_items["G-005"]["notes"])
            self.assertEqual(missing_items["G-006"]["status"], "missing")
            self.assertIn("p_unknown_present", missing_items["G-006"]["notes"])
            self.assertEqual(missing_items["G-007"]["status"], "missing")
            self.assertIn("image_backed_scene_v0", missing_items["G-007"]["notes"])
            self.assertEqual(missing_items["G-008"]["status"], "missing")
            self.assertIn("ReplayStep", missing_items["G-008"]["notes"])
            self.assertEqual(missing_items["G-009"]["status"], "missing")
            self.assertIn("render_scene_html", missing_items["G-009"]["notes"])
            self.assertEqual(missing_items["G-010"]["status"], "missing")
            self.assertIn("data-action='play'", missing_items["G-010"]["notes"])
            self.assertEqual(missing_items["G-011"]["status"], "missing")
            self.assertIn("ImageEnhance.Brightness", missing_items["G-011"]["notes"])
            self.assertEqual(missing_items["G-012"]["status"], "missing")
            self.assertIn("DOMAIN_RANDOMIZATION_PRESETS", missing_items["G-012"]["notes"])
            self.assertEqual(missing_items["I-004"]["status"], "missing")
            self.assertIn('port: str = "DRY_RUN"', missing_items["I-004"]["notes"])
            self.assertEqual(missing_items["I-005"]["status"], "missing")
            self.assertIn("RobotTelemetry", missing_items["I-005"]["notes"])
            self.assertEqual(missing_items["J-001"]["status"], "missing")
            self.assertIn("render_offline_viewer_html", missing_items["J-001"]["notes"])
            self.assertEqual(missing_items["J-002"]["status"], "missing")
            self.assertIn('data-action="reset"', missing_items["J-002"]["notes"])
            self.assertEqual(missing_items["J-003"]["status"], "missing")
            self.assertIn("blocked_reasons", missing_items["J-003"]["notes"])
            self.assertEqual(missing_items["J-004"]["status"], "missing")
            self.assertIn("ComponentRegistry.from_file", missing_items["J-004"]["notes"])
            self.assertEqual(missing_items["J-006"]["status"], "missing")
            self.assertIn("build_report_payload", missing_items["J-006"]["notes"])
            self.assertEqual(missing_items["J-007"]["status"], "missing")
            self.assertIn("FeedbackAnnotationTask", missing_items["J-007"]["notes"])
            self.assertEqual(missing_items["K-001"]["status"], "missing")
            self.assertIn("run_sim_smoke", missing_items["K-001"]["notes"])
            self.assertEqual(missing_items["K-002"]["status"], "missing")
            self.assertIn("write_run_registry_records", missing_items["K-002"]["notes"])
            self.assertEqual(missing_items["K-003"]["status"], "missing")
            self.assertIn("dataset_audit.json", missing_items["K-003"]["notes"])
            self.assertEqual(missing_items["K-004"]["status"], "missing")
            self.assertIn("map50_95", missing_items["K-004"]["notes"])
            self.assertEqual(missing_items["K-005"]["status"], "missing")
            self.assertIn("scene_metrics.json", missing_items["K-005"]["notes"])
            self.assertEqual(missing_items["K-006"]["status"], "missing")
            self.assertIn("artifact_registry.csv", missing_items["K-006"]["notes"])
            self.assertEqual(missing_items["K-007"]["status"], "missing")
            self.assertIn("experiment_report.html", missing_items["K-007"]["notes"])
            self.assertEqual(missing_items["L-001"]["status"], "missing")
            self.assertIn("test_detection_object_computes_center_and_area", missing_items["L-001"]["notes"])
            self.assertEqual(missing_items["L-002"]["status"], "missing")
            self.assertIn("test_assign_to_cells_and_count_matrix", missing_items["L-002"]["notes"])
            self.assertEqual(missing_items["L-003"]["status"], "missing")
            self.assertIn("test_largest_bbox_strategy_generates_extra_crop_targets", missing_items["L-003"]["notes"])
            self.assertEqual(missing_items["L-004"]["status"], "missing")
            self.assertIn("test_action_mask_reports_reasons", missing_items["L-004"]["notes"])
            self.assertEqual(missing_items["L-005"]["status"], "missing")
            self.assertIn("test_image_and_robot_transforms", missing_items["L-005"]["notes"])
            self.assertEqual(missing_items["L-006"]["status"], "missing")
            self.assertIn(
                "test_rule_based_policy_completes_fixture_scene_through_safety_gate",
                missing_items["L-006"]["notes"],
            )
            self.assertEqual(missing_items["L-007"]["status"], "missing")
            self.assertIn("LogicalTraySimulator", missing_items["L-007"]["notes"])
            self.assertEqual(missing_items["L-008"]["status"], "missing")
            self.assertIn("test_rl_train_dry_run_writes_summary_and_registry", missing_items["L-008"]["notes"])
            self.assertEqual(missing_items["L-009"]["status"], "missing")
            self.assertIn("test_evaluate_cells_can_use_scene_state_schema_inputs", missing_items["L-009"]["notes"])
            self.assertEqual(missing_items["L-010"]["status"], "missing")
            self.assertIn("test_architecture_cli_modules_show_help", missing_items["L-010"]["notes"])
            self.assertEqual(missing_items["M-001"]["status"], "missing")
            self.assertIn("Карта документов", missing_items["M-001"]["notes"])
            self.assertEqual(missing_items["M-002"]["status"], "missing")
            self.assertIn("Архитектурный каркас", missing_items["M-002"]["notes"])
            self.assertEqual(missing_items["M-003"]["status"], "missing")
            self.assertIn("SeedlingTrayEnv", missing_items["M-003"]["notes"])
            self.assertEqual(missing_items["M-004"]["status"], "missing")
            self.assertIn("LogicalTraySimulator", missing_items["M-004"]["notes"])
            self.assertEqual(missing_items["M-005"]["status"], "missing")
            self.assertIn("DetectorAdapter", missing_items["M-005"]["notes"])
            self.assertEqual(missing_items["M-006"]["status"], "missing")
            self.assertIn("BLOCK_CALIBRATION_REQUIRED", missing_items["M-006"]["notes"])
            self.assertEqual(missing_items["M-007"]["status"], "missing")
            self.assertIn("CalibrationConfig", missing_items["M-007"]["notes"])
            self.assertEqual(missing_items["M-008"]["status"], "missing")
            self.assertIn("Калибровка отсутствует", missing_items["M-008"]["notes"])
            self.assertEqual(missing_items["M-009"]["status"], "missing")
            self.assertIn("Исходные артефакты", missing_items["M-009"]["notes"])
            self.assertEqual(missing_items["M-010"]["status"], "missing")
            self.assertIn("Процедура сухого прогона", missing_items["M-010"]["notes"])
            self.assertEqual(missing_items["J-005"]["status"], "missing")
            self.assertIn("rl_no_model", missing_items["J-005"]["notes"])
            self.assertEqual(missing_items["K-008"]["status"], "missing")
            self.assertIn("max_run_snapshot_missing_command_args", missing_items["K-008"]["notes"])
            self.assertEqual(missing_items["L-011"]["status"], "missing")
            self.assertIn("train_from_config", missing_items["L-011"]["notes"])

            pyproject.write_text(
                "\n".join(
                    [
                        "[project.scripts]",
                        "seedling-experiments = 'seedling_experiments.cli:main'",
                        "seedling-data = 'seedling_data.cli:main'",
                        "seedling-reports = 'seedling_reports.cli:main'",
                        "seedling-vision = 'seedling_vision.cli:main'",
                        "[tool.setuptools.packages.find]",
                        "include = ['seedling_*']",
                        "pre-commit",
                        "pytest",
                        "ruff",
                    ]
                ),
                encoding="utf-8",
            )
            experiments_main_py.write_text(
                "from .cli import main\n\nif __name__ == \"__main__\":\n    main()\n",
                encoding="utf-8",
            )
            requirements_base.write_text("numpy\nPillow\nPyYAML\n", encoding="utf-8")
            requirements_vision.write_text("opencv-python\nonnxruntime\npillow-heif\n", encoding="utf-8")
            requirements_rl.write_text("gymnasium\nstable-baselines3\nsb3-contrib\n", encoding="utf-8")
            requirements_robot.write_text("pyserial\n", encoding="utf-8")
            requirements_dev.write_text("pre-commit\npytest\nruff\n", encoding="utf-8")
            core_config_py.write_text(
                "\n".join(
                    [
                        "class ConfigError: pass",
                        "class FieldSpec: pass",
                        "def load_config_file(): pass",
                        "def require_sections(): pass",
                        "def validate_fields(): pass",
                        "def validate_existing_paths(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            schemas_py.write_text(
                "\n".join(
                    [
                        "SCHEMA_VERSION = 'seedling_scene_v0_1'",
                        "class DetectionObject: pass",
                        "class CellState: pass",
                        "class ActionTarget: pass",
                        "class TrayState: pass",
                        "class RobotState: pass",
                        "class SafetyState: pass",
                        "class SceneState: pass",
                        "class DetectionResultV1: pass",
                        "class ActionCommand: pass",
                        "class ActionPlan: pass",
                        "def to_dict(): pass",
                        "def from_dict(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            migration_py.write_text(
                "\n".join(
                    [
                        "def legacy_predictions_to_scenes(): pass",
                        "def legacy_prediction_to_detection_result(): pass",
                        "def legacy_prediction_to_scene(): pass",
                        "def write_scene_states(): pass",
                        "DetectionResultV1",
                        "CellStateBuilder",
                        "generate_largest_bbox_targets",
                    ]
                ),
                encoding="utf-8",
            )
            precommit.write_text("pre-commit-hooks\nruff-pre-commit\nruff-format\n", encoding="utf-8")
            ci_workflow.write_text(
                "actions/setup-python\npre-commit run --all-files\npython -B -m unittest discover -s tests\n",
                encoding="utf-8",
            )
            schema_tests.write_text(
                "\n".join(
                    [
                        "def test_detection_object_computes_center_and_area(): pass",
                        "def test_scene_state_round_trip(): pass",
                        "def test_action_command_round_trip(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            migration_tests.write_text(
                "\n".join(
                    [
                        "def test_legacy_predictions_convert_to_scene_state_with_targets(): pass",
                        "def test_migrate_predictions_cli_writes_scene_json(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            ontology_config.write_text(
                "\n".join(
                    [
                        "version: ontology_v0_1",
                        "object_classes:",
                        "  3: {name: unknown_plant}",
                        "action_labels: [remove_extra_crop, human_review_required]",
                        "attributes: [low_confidence]",
                    ]
                ),
                encoding="utf-8",
            )
            ontology_py.write_text(
                "\n".join(
                    [
                        "REQUIRED_OBJECT_CLASSES = {}",
                        "REQUIRED_CELL_STATES = set()",
                        "REQUIRED_ACTION_LABELS = set()",
                        "def validate_dataset_class_names(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            ontology_tests.write_text(
                "\n".join(
                    [
                        "def test_ontology_rejects_missing_required_v0_labels(): pass",
                        "def test_ontology_rejects_duplicate_aliases_and_dataset_class_names(): pass",
                        "def test_recorded_prediction_detector_reads_existing_contract(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            manifest_py.write_text(
                "\n".join(
                    [
                        "MANIFEST_FIELDS = ['sha256', 'group_id', 'session_id', 'tray_id']",
                        "def write_image_manifest(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            duplicates_py.write_text(
                "\n".join(
                    [
                        "exact_sha256_duplicates = []",
                        "near_perceptual_duplicates = []",
                        "missing_split_rows = []",
                        "missing_sha256_rows = []",
                        "max_hamming = 4",
                    ]
                ),
                encoding="utf-8",
            )
            dataset_card_doc.write_text(
                "\n".join(
                    [
                        "Версия набора данных",
                        "Версия онтологии",
                        "Версия руководства по разметке",
                        "Версия разбиения",
                        "Версия калибровки",
                        "Ограничения безопасности и использования",
                        "artifact_registry.json",
                    ]
                ),
                encoding="utf-8",
            )
            annotation_guide_doc.write_text(
                "\n".join(
                    [
                        "Метки объектов",
                        "Ограничивающие рамки",
                        "Разметка ячеек",
                        "cell_annotations.jsonl",
                        "Допустимые значения `state`",
                        "unknown",
                        "image_quality_insufficient",
                        "Точки действия",
                        "action_points.jsonl",
                        "human_review_required: true",
                        "min_distance_to_keep_px",
                        "uncertainty_radius_px",
                        "forbidden_zone_ids",
                        "near-duplicates",
                    ]
                ),
                encoding="utf-8",
            )
            dataset_py.write_text(
                "\n".join(
                    [
                        "DEFAULT_AUGMENTED_NAME_MARKERS = ()",
                        "def make_grouped_split(metadata_group_column=None):",
                        "    read_group_map",
                        "    split_manifest.csv",
                        "    data.yaml",
                        "    seed",
                        "    _assign_groups",
                        "def validate_split_integrity():",
                        "    is_suspected_augmented_name",
                        "    base_image_key",
                        "    eval_augmented_images",
                        "    cross_split_base_leakage",
                    ]
                ),
                encoding="utf-8",
            )
            annotations_py.write_text(
                "\n".join(
                    [
                        "class CellAnnotation: pass",
                        "class ActionPointAnnotation: pass",
                        "def read_cell_annotations(): pass",
                        "def write_cell_annotations(): pass",
                        "def audit_cell_annotations():",
                        "    state_counts",
                        "    duplicate cell annotation",
                        "def read_action_points(): pass",
                        "def write_action_points(): pass",
                        "def audit_action_points():",
                        "    min_safe_distance_px",
                        "    max_uncertainty_px",
                        "    forbidden_zone_ids",
                        "    human_review_required",
                        "    cell_id is not present in cell annotations",
                        "    crop_and_weed action point requires human_review_required=true",
                        "    forbidden_zone_ids require human_review_required=true",
                        "    target_type human_review_required must set human_review_required=true",
                        "    unsafe_distance_rows",
                        "    high_uncertainty_rows",
                        "    review_required_rows",
                    ]
                ),
                encoding="utf-8",
            )
            data_registry_py.write_text(
                "\n".join(
                    [
                        "class DatasetRegistryEntry: pass",
                        "class DatasetRegistry: pass",
                        "def add_dataset(): pass",
                        "def validate_dataset_registry(): pass",
                        "def write_dataset_summary(): pass",
                        "def collect_dataset_hashes(): pass",
                        "cell_annotations",
                        "action_points",
                    ]
                ),
                encoding="utf-8",
            )
            changelog_py.write_text(
                "\n".join(
                    [
                        "CHANGELOG_CANDIDATES = ()",
                        "REQUIRED_FIELDS = ()",
                        "REQUIRED_SECTIONS = ()",
                        "def audit_dataset_changelog():",
                        "    Source dataset root",
                        "    Validation",
                    ]
                ),
                encoding="utf-8",
            )
            dataset_changelog_doc.write_text(
                "\n".join(
                    [
                        "Шаблон журнала изменений набора данных",
                        "Статус: черновик / заморожен / архивирован",
                        "Корень исходного набора данных",
                        "### Проверка",
                    ]
                ),
                encoding="utf-8",
            )
            docs_readme.write_text(
                "\n".join(
                    [
                        "Карта документов",
                        "Быстрый старт",
                        "seedling_reports build",
                        "artifact_registry.json",
                    ]
                ),
                encoding="utf-8",
            )
            docs_architecture.write_text(
                "\n".join(
                    [
                        "Архитектурный каркас",
                        "Поток данных",
                        "SafetyGate",
                        "Реестры и селекторы",
                    ]
                ),
                encoding="utf-8",
            )
            docs_rl_spec.write_text(
                "\n".join(
                    [
                        "SeedlingTrayEnv",
                        "Награда",
                        "MaskablePPO",
                        "RecurrentPPO",
                        "offline-replay-eval",
                        "sweep",
                    ]
                ),
                encoding="utf-8",
            )
            docs_sim_spec.write_text(
                "\n".join(
                    [
                        "LogicalTraySimulator",
                        "DetectionNoiseModel",
                        "ActuatorErrorModel",
                        "PlantResponseModel",
                        "run-policy",
                    ]
                ),
                encoding="utf-8",
            )
            docs_model_plugin.write_text(
                "\n".join(
                    [
                        "DetectorAdapter",
                        "Реестр моделей и политик",
                        "Миграция старых предсказаний",
                        "Адаптеры политик и робота",
                        "SafetyGate",
                    ]
                ),
                encoding="utf-8",
            )
            docs_safety.write_text(
                "\n".join(
                    [
                        "SafetyGate",
                        "BLOCK_CALIBRATION_REQUIRED",
                        "BLOCK_INTERLOCK",
                        "dry-run-plan",
                        "HardwareInLoopReview",
                    ]
                ),
                encoding="utf-8",
            )
            docs_failure_modes.write_text(
                "\n".join(
                    [
                        "Режим отказа",
                        "Как обнаруживается",
                        "Мера снижения",
                        "Калибровка отсутствует",
                        "Межблокировка не подтверждена",
                        "опасный профиль инструмента",
                    ]
                ),
                encoding="utf-8",
            )
            docs_article_report.write_text(
                "\n".join(
                    [
                        "Исходные артефакты",
                        "Таблица 0. Реестр моделей",
                        "Состав набора данных",
                        "Метрики безопасности",
                        "Воспроизводимость",
                    ]
                ),
                encoding="utf-8",
            )
            docs_operator_manual.write_text(
                "\n".join(
                    [
                        "Режимы",
                        "Процедура сухого прогона",
                        "Процедура HIL с указателем",
                        "Обратная связь",
                        "Проверка после действия",
                    ]
                ),
                encoding="utf-8",
            )
            data_tests.write_text(
                "\n".join(
                    [
                        "def test_image_manifest_generator_fills_group_session_and_tray_ids(): pass",
                        "def test_image_manifest_cli_writes_artifact_registry(): pass",
                        "def test_duplicate_check_finds_cross_split_identical_image(): pass",
                        "def test_duplicate_check_rejects_manifest_without_split_hash_or_path(): pass",
                        "def test_cell_and_action_annotation_audits_accept_valid_rows(): pass",
                        "def test_action_annotation_audit_requires_review_for_crop_and_weed_cells(): pass",
                        "def test_action_annotation_audit_requires_review_for_unsafe_targets(): pass",
                        "def test_action_annotation_audit_accepts_review_flags_for_unsafe_targets(): pass",
                        "def test_action_annotation_audit_cli_supports_safety_thresholds(): pass",
                        "def test_dataset_registry_add_validate_and_summarize(): pass",
                        "def test_registry_cli_smoke(): pass",
                        "def test_dataset_changelog_audit_and_cli(): pass",
                        "def test_dataset_changelog_accepts_source_dataset_root_path(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            dataset_split_tests.write_text(
                "def test_grouped_split_uses_image_manifest_group_ids(): pass\n",
                encoding="utf-8",
            )
            detector_base_py.write_text(
                "\n".join(
                    [
                        "class DetectorAdapter(ABC):",
                        "    def load(self): pass",
                        "    def predict(self) -> DetectionResultV1: pass",
                        "    def metadata(self) -> ModelMetadata: pass",
                        "def metadata_for_context(): pass",
                        "class MockDetector: pass",
                    ]
                ),
                encoding="utf-8",
            )
            ultralytics_py.write_text(
                "\n".join(
                    [
                        "class UltralyticsYOLODetector:",
                        "    from ultralytics import YOLO",
                        "    def predict(self):",
                        "        DetectionObject(",
                        "        DetectionResultV1(",
                        "        model_metadata=metadata_for_context",
                        "        backend=\"ultralytics_yolo\"",
                        "def _load_rgb(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            baseline_green_py.write_text(
                "\n".join(
                    [
                        "class BaselineGreenDetector:",
                        "    cv2.cvtColor",
                        "    cv2.inRange",
                        "    cv2.findContours",
                        "    DetectionObject(",
                        "    attributes=[\"baseline_green\"]",
                        "    output_schema=\"DetectionResultV1\"",
                    ]
                ),
                encoding="utf-8",
            )
            recorded_prediction_py.write_text(
                "\n".join(
                    [
                        "class RecordedPredictionDetector:",
                        "    Recorded predictions must contain an `images` list",
                        "    extra={",
                        "    \"containers\": prediction.get",
                        "    metadata_for_context",
                        "def _record_to_detection(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            onnx_py.write_text(
                "\n".join(
                    [
                        "class ONNXDetector:",
                        "    import onnxruntime as ort",
                        "    output_format",
                        "    DetectionObject(",
                        "def parse_onnx_detections(): pass",
                        "def _preprocess(): pass",
                        "def _providers(): pass",
                        "def _uncertainty_from_row(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            uncertainty_py.write_text(
                "\n".join(
                    [
                        "class UncertaintyConfig:",
                        "    low_confidence_threshold = 0.5",
                        "    tiny_area_px2 = 20.0",
                        "    high_entropy_threshold = 0.8",
                        "def annotate_detection_uncertainty():",
                        "    class_entropy_normalized",
                        "    class_probability_margin",
                        "    touches_image_edge",
                    ]
                ),
                encoding="utf-8",
            )
            postprocess_py.write_text(
                "\n".join(
                    [
                        "def filter_and_merge_containers(): pass",
                        "def merge_container_detections(): pass",
                        "def match_containers_by_iou(): pass",
                        "def container_matching_summary():",
                        "    best_iou_counts",
                        "    unmatched_samples",
                        "def bbox_iou(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            core_registry_py.write_text(
                "\n".join(
                    [
                        "class ComponentRegistry:",
                        "    def select(self): pass",
                        "class ModelRegistryRecord:",
                        "    input_schema = 'ImageInputV1'",
                        "    output_schema = 'DetectionResultV1'",
                        "    calibration_requirements = {}",
                        "    safety_level = 'offline_only'",
                        "    model_type = 'detector'",
                        "class ModelRegistry:",
                        "    def select(self): pass",
                        "requires_action_mask",
                        "direct_hardware_access",
                    ]
                ),
                encoding="utf-8",
            )
            experiment_cli_py.write_text(
                "\n".join(
                    [
                        "_add_config_command(subparsers, \"train\"",
                        "_add_config_command(subparsers, \"val\"",
                        "_add_config_command(subparsers, \"predict\"",
                        "_add_config_command(subparsers, \"evaluate\"",
                        "_add_config_command(subparsers, \"prepare\"",
                        "subparsers.add_parser(\"sim\"",
                        "subparsers.add_parser(\"rl\"",
                        "def run_sim_smoke(): pass",
                        "def run_rl_command(): pass",
                        "def _write_experiment_registry():",
                        "    ArtifactRecord.from_path",
                        "    role=\"input\"",
                        "    role=\"output\"",
                        "    write_run_registry_records",
                    ]
                ),
                encoding="utf-8",
            )
            experiment_config_py.write_text(
                "\n".join(
                    [
                        "def save_run_snapshot(): pass",
                        "def validate_experiment_config(): pass",
                        "FieldSpec.parse",
                        "validate_existing_paths",
                        "def config_hash(): pass",
                        "def _existing_path_fields(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            model_registry_config.write_text(
                "\n".join(
                    [
                        "version: model_registry_v0_1",
                        "models:",
                        "  - model_id: baseline_green_detector_v0",
                        "    output_schema: DetectionResultV1",
                        "    dataset_version: zks_v0_1",
                        "    ontology_version: ontology_v0_1",
                        "    calibration_requirements: {required: false}",
                        "    safety_level: offline_only",
                    ]
                ),
                encoding="utf-8",
            )
            overlays_py.write_text(
                "\n".join(
                    [
                        "ERROR_COLOR = (220, 30, 50)",
                        "TARGET_COLOR = (240, 80, 40)",
                        "def write_detection_overlay():",
                        "    draw.rectangle",
                        "    draw.line",
                        "    _draw_target_marker",
                        "def _draw_target_marker(): pass",
                        "def _draw_issue(): pass",
                        "def _draw_issue_legend(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            batch_py.write_text(
                "def batch_predict(images, context_factory=None, progress=None):\n    return enumerate(images, 1)\n",
                encoding="utf-8",
            )
            vision_cli.write_text(
                "\n".join(
                    [
                        "command = 'batch-recorded'",
                        "option = 'progress-log'",
                        "predict_start = True",
                        "batch_complete = True",
                    ]
                ),
                encoding="utf-8",
            )
            vision_tests.write_text(
                "\n".join(
                    [
                        "def test_ultralytics_adapter_maps_mock_result_to_detection_objects(): pass",
                        "def test_onnx_parser_maps_xyxy_rows_to_detection_objects(): pass",
                        "def test_onnx_parser_supports_channel_first_class_scores(): pass",
                        "def test_onnx_parser_supports_objectness_class_scores(): pass",
                        "def test_uncertainty_flags_low_confidence_tiny_and_edge(): pass",
                        "def test_uncertainty_computes_entropy_from_class_scores(): pass",
                        "def test_container_postprocess_merges_small_nearby_boxes(): pass",
                        "def test_postprocess_containers_cli_writes_postprocessed_json(): pass",
                        "def test_batch_predict_uses_context_factory_and_progress_indexes(): pass",
                        "def test_vision_batch_and_overlay_cli(): pass",
                        "def test_vision_overlay_cli_draws_grid_targets_and_errors(): pass",
                        "def test_overlay_resolves_error_taxonomy_target_aliases(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            registry_selector_tests.write_text(
                "\n".join(
                    [
                        "def test_selector_cli_emits_selected_entry(): pass",
                        "def test_selector_cli_validates_component_registry(): pass",
                        "def test_baseline_green_detector_finds_green_region(): pass",
                        "def test_typed_model_registry_validates_required_metadata(): pass",
                        "def test_typed_model_registry_blocks_production_candidate_without_review(): pass",
                        "def test_typed_model_registry_blocks_rl_direct_hardware_access(): pass",
                        "def test_feedback_rows_convert_to_annotation_tasks(): pass",
                        "def test_feedback_priority_override_is_preserved_in_annotation_task(): pass",
                        "def test_feedback_cli_writes_annotation_task_jsonl(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            grid_py.write_text(
                "\n".join(
                    [
                        "class GridCell: pass",
                        "def generate_grid(): pass",
                        "def generate_grid_cells(): pass",
                        "def generate_grid_polygons(): pass",
                        "def cell_index(): pass",
                        "def cell_index_for_grid_cells(): pass",
                        "def point_in_polygon(): pass",
                        "def _validate_grid(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            cell_state_py.write_text(
                "\n".join(
                    [
                        "class CellStateBuilder:",
                        "    crop_class_names = ()",
                        "    weed_class_names = ()",
                        "    unknown_class_names = ()",
                        "    review_attribute_names = ()",
                        "    def build(self):",
                        "        generate_grid_cells",
                        "        generate_grid_polygons",
                        "        cell_index_for_grid_cells",
                        "        CellState(",
                        "        object_ids",
                        "        human_review_required",
                        "        risk_flags",
                    ]
                ),
                encoding="utf-8",
            )
            target_tests.write_text(
                "\n".join(
                    [
                        "def test_largest_bbox_strategy_generates_extra_crop_targets(): pass",
                        "def test_largest_bbox_strategy_is_configurable_and_respects_keep_object(): pass",
                        "def test_unknown_cell_requires_review_and_no_auto_target(): pass",
                        "def test_weed_only_cell_generates_remove_weed_target(): pass",
                        "def test_high_entropy_single_crop_marks_cell_for_review(): pass",
                        "def test_crop_and_weed_cell_generates_reviewed_remove_weed_target(): pass",
                        "def test_cell_state_builder_maps_detections_to_full_tray_grid(): pass",
                        "def test_cell_builder_with_tray_corners_ignores_points_outside_polygon(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            grid_tests.write_text(
                "\n".join(
                    [
                        "def test_cell_index_uses_last_cell_for_max_boundary(): pass",
                        "def test_assign_to_cells_and_count_matrix(): pass",
                        "def test_generate_grid_polygons_supports_tray_corners(): pass",
                        "def test_corner_grid_assignment_uses_cell_polygons_not_outer_bbox(): pass",
                        "def test_generate_grid_rejects_invalid_bbox(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            evaluate_tests.write_text(
                "\n".join(
                    [
                        "def test_evaluate_cells_reports_calibrated_target_error_mm(): pass",
                        "def test_evaluate_cells_reports_legacy_cost_sensitive_errors(): pass",
                        "def test_evaluate_cells_can_use_scene_state_schema_inputs(): pass",
                        "def test_evaluate_cells_preserves_manifest_group_metadata_for_bootstrap(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            scene_eval_tests.write_text(
                "\n".join(
                    [
                        "def test_scene_state_evaluator_handles_richer_states_targets_and_costs(): pass",
                        "def test_scene_state_evaluator_reports_edge_and_corner_cell_metrics(): pass",
                        "def test_evaluate_scenes_cli_writes_metrics(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            target_generation_py.write_text(
                "\n".join(
                    [
                        "class LargestBBoxTargetStrategy:",
                        "    decision_source = 'largest_bbox_v0'",
                        "    crop_class_names = ('crop_seedling',)",
                        "    weed_class_names = ('weed',)",
                        "    def generate(self):",
                        "        remove_extra_crop",
                        "        remove_weed",
                        "        ActionTarget(",
                        "    def _infer_removal_candidates(self):",
                        "        keep_object_id",
                    ]
                ),
                encoding="utf-8",
            )
            evaluate_py.write_text(
                "\n".join(
                    [
                        "def evaluate_cells_from_config():",
                        "    match_containers",
                        "    container_iou",
                        "    best_container_ious",
                        "    unmatched_samples",
                        "    container_matching",
                        "    prediction_source",
                        "    prediction_coverage",
                        "    cell_accuracy",
                        "    multi_seedling_cell",
                        "    removal_targets",
                        "    cell_confusion_matrix.csv",
                        "    cost_sensitive_metrics_from_legacy",
                        "    critical_error_counts",
                        "    critical_error_total",
                        "    critical_error_rate_per_cell",
                        "    normalized_cost_per_cell",
                        "    cost_sensitive",
                        "def _update_confusion(): pass",
                        "def _macro_metrics(): pass",
                        "def _prf(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            scene_evaluation_py.write_text(
                "\n".join(
                    [
                        "DEFAULT_STATE_ORDER = ['weed_only', 'crop_and_weed', 'unknown', 'ambiguous']",
                        "def evaluate_scene_states():",
                        "    target_match_distance_mm",
                        "def evaluate_cell_states(): pass",
                        "def evaluate_action_targets():",
                        "    expert_keep_remove",
                        "    mean_error_mm",
                        "def cost_sensitive_metrics():",
                        "    target_false_negative",
                        "    target_false_positive",
                        "    unknown_not_reviewed",
                        "    expert_false_removal",
                        "    expert_missed_removal",
                        "    cost_breakdown",
                        "def _region_cell_metrics():",
                        "    edge_cell_metrics",
                        "    corner_cell_metrics",
                    ]
                ),
                encoding="utf-8",
            )
            diagnostics_py.write_text(
                "\n".join(
                    [
                        "ERROR_CATEGORY_GROUPS = {'image': 'image', 'geometry': 'geometry', 'biology': 'biology', 'decision': 'decision'}",
                        "def build_cell_metrics_diagnostics():",
                        "    bootstrap_ci.json",
                        "    error_taxonomy.json",
                        "def bootstrap_cell_metrics():",
                        "    group_key",
                        "    critical_error_rate_per_cell",
                        "    normalized_cost_per_cell",
                        "def error_taxonomy():",
                        "    category_groups",
                        "    container_unmatched",
                        "    target_coordinate_error",
                        "    cell_state_mismatch",
                        "def _add_scene_state_taxonomy(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            calibration_schemas_py.write_text(
                "\n".join(
                    [
                        "CALIBRATION_SCHEMA_VERSION = 'calibration_artifact_v0_1'",
                        "class ErrorSummary: pass",
                        "class CalibrationConfig: pass",
                        "class CalibrationArtifact:",
                        "    image_to_tray_homography = []",
                        "    tray_to_robot_transform = []",
                        "    tool_offset_mm = []",
                        "    error_summary_mm = None",
                        "    def to_json(self): pass",
                        "class CalibrationValidationResult: pass",
                    ]
                ),
                encoding="utf-8",
            )
            calibration_transforms_py.write_text(
                "\n".join(
                    [
                        "def apply_homography(point_xy, matrix):",
                        "    raise ValueError('homography denominator is too close to zero')",
                        "def image_px_to_tray_mm(point_px, artifact):",
                        "    return artifact.image_to_tray_homography",
                        "def tray_mm_to_robot_frame_mm(point_mm, artifact, include_tool_offset=True):",
                        "    artifact.tray_to_robot_transform",
                        "    artifact.tool_offset_mm",
                        "def attach_calibration_to_target(target, artifact):",
                        "    point_mm = target.action_point_mm or image_px_to_tray_mm(target.action_point_px, artifact)",
                        "    robot_point = tray_mm_to_robot_frame_mm(point_mm, artifact)",
                        "    uncertainty = artifact.error_summary_mm.p95",
                        "def attach_calibration_to_scene(scene, artifact):",
                        "    calibration_id=artifact.calibration_id",
                        "    targets=attach_calibration_to_targets",
                    ]
                ),
                encoding="utf-8",
            )
            calibration_validator_py.write_text(
                "\n".join(
                    [
                        "class CalibrationValidator:",
                        "    expected_tray_type = None",
                        "    max_error_p95_mm = 2.0",
                        "    max_error_rms_mm = None",
                        "    valid_until = None",
                        "    def validate(self):",
                        "        calibration expired",
                        "        error_summary_mm.p95",
                        "        error_summary_mm.rms",
                        "        px_per_mm_x",
                    ]
                ),
                encoding="utf-8",
            )
            calibration_estimation_py.write_text(
                "\n".join(
                    [
                        "def write_error_map(): pass",
                        "def calibration_residuals():",
                        "    expected_tray_mm",
                        "    predicted_tray_mm",
                        "    error_summary_mm",
                        "def _error_map_html(): pass",
                        "def _error_map_grid(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            calibration_doc.write_text(
                "\n".join(
                    [
                        "Calibration Protocol",
                        "CalibrationConfig",
                        "CalibrationArtifact",
                        "CalibrationValidator",
                        "error_map",
                        "artifact_registry.json",
                        "error-map",
                        "сводку ошибок",
                        "карты ошибок",
                    ]
                ),
                encoding="utf-8",
            )
            calibration_cli_py.write_text(
                "\n".join(
                    [
                        "subparsers.add_parser(\"validate\"",
                        "subparsers.add_parser(\"estimate\"",
                        "subparsers.add_parser(\"error-map\"",
                        "def _write_calibration_snapshot(): pass",
                        "def _write_calibration_registry(): pass",
                        "seedling-calibration:validate",
                        "seedling-calibration:estimate",
                        "seedling-calibration:error-map",
                    ]
                ),
                encoding="utf-8",
            )
            calibration_intrinsics_py.write_text(
                "\n".join(
                    [
                        "INTRINSICS_SCHEMA_VERSION = 'camera_intrinsics_v0_1'",
                        "SUPPORTED_TARGET_TYPES = {\"chessboard\", \"aruco\", \"fiducials\"}",
                        "class IntrinsicsObservation: pass",
                        "class CameraIntrinsicsArtifact:",
                        "    reprojection_error_px = None",
                        "def estimate_camera_intrinsics():",
                        "    cv2.calibrateCamera",
                        "def validate_camera_intrinsics(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            calibration_tests.write_text(
                "\n".join(
                    [
                        "def test_image_and_robot_transforms(): pass",
                        "def test_estimate_calibration_artifact_from_four_points(): pass",
                        "def test_validator_rejects_expired_or_high_error_calibration(): pass",
                        "def test_validator_rejects_wrong_tray_type_and_rms_error(): pass",
                        "def test_calibration_estimate_and_error_map_cli(): pass",
                        "def test_attach_calibration_to_target(): pass",
                        "def test_attach_calibration_to_scene_updates_targets_and_tray_id(): pass",
                        "def test_calibration_validate_cli_writes_report_and_registry(): pass",
                        "def test_camera_intrinsics_artifact_validates(): pass",
                        "def test_estimate_camera_intrinsics_uses_loaded_observations_and_cv2_contract(): pass",
                        "def test_intrinsics_observations_example_loads(): pass",
                        "def test_validate_intrinsics_cli(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            sim_schemas_py.write_text(
                "\n".join(
                    [
                        "SIM_SCENE_SCHEMA_VERSION = 'sim_scene_v0_1'",
                        "class SimPlant: pass",
                        "class SimTarget: pass",
                        "class SimScene:",
                        "    def from_json(self): pass",
                        "    def to_json(self): pass",
                        "    def plant_by_id(self): pass",
                        "    def target_by_id(self): pass",
                        "    def _validate_bounds(self): pass",
                        "class SimStepOutcome: pass",
                    ]
                ),
                encoding="utf-8",
            )
            sim_logical_tray_py.write_text(
                "\n".join(
                    [
                        "class LogicalTraySimulator:",
                        "    ActuatorErrorModel",
                        "    PlantResponseModel",
                        "    move_mm_penalty = 0.01",
                        "    def reset(self): pass",
                        "    def step_target(self):",
                        "        crop_damage",
                        "        SimStepOutcome(",
                    ]
                ),
                encoding="utf-8",
            )
            sim_detection_noise_py.write_text(
                "\n".join(
                    [
                        "class DetectionNoiseModel:",
                        "    bbox_center_sigma_mm = 1.0",
                        "    classification_error_prob = 0.03",
                        "    missed_detection_prob = 0.05",
                        "    false_positive_prob = 0.0",
                        "    classification_noise",
                        "    false_positive_noise",
                        "class SimSceneDetectionNoiseModel: pass",
                    ]
                ),
                encoding="utf-8",
            )
            sim_actuator_model_py.write_text(
                "\n".join(
                    [
                        "class ActuatorSample:",
                        "    zone_error_active = False",
                        "class ActuatorErrorModel:",
                        "    xy_sigma_mm = 0.8",
                        "    drift_sigma_mm = 0.2",
                        "    latency_ms_mean = 0.0",
                        "    zone_error_prob = 0.0",
                        "    zone_error_mm = 0.0",
                    ]
                ),
                encoding="utf-8",
            )
            sim_plant_response_py.write_text(
                "\n".join(
                    [
                        "class PlantResponse: pass",
                        "class PlantResponseModel:",
                        "    unknown_requires_review = True",
                        "    avoids laser power/dose modeling",
                        "    near_keep_crop",
                        "    missed_target",
                        "    biological_nonresponse",
                    ]
                ),
                encoding="utf-8",
            )
            sim_scene_generator_py.write_text(
                "\n".join(
                    [
                        "class SceneGeneratorConfig:",
                        "    p_empty = 0.65",
                        "    p_single_crop = 0.25",
                        "    p_multiple_crop = 0.07",
                        "    p_weed_present = 0.03",
                        "    p_unknown_present = 0.0",
                        "class SimSceneGenerator:",
                        "    def generate_many(self): pass",
                        "    human_review_required",
                    ]
                ),
                encoding="utf-8",
            )
            sim_image_backed_py.write_text(
                "\n".join(
                    [
                        "def sim_scene_from_scene_state():",
                        "    image_backed_scene_v0",
                        "def sim_scene_from_scene_state_file(): pass",
                        "def _plant_from_detection(): pass",
                        "def _target_from_action_target(): pass",
                        "def _point_mm(): pass",
                        "def _load_scene_state(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            sim_replay_py.write_text(
                "\n".join(
                    [
                        "class ReplayStep: pass",
                        "class ReplayLog:",
                        "    def from_json(self): pass",
                        "class ReplayLogger:",
                        "    def append(self): pass",
                        "    def append_execution(self): pass",
                        "    def to_json(self): pass",
                    ]
                ),
                encoding="utf-8",
            )
            sim_renderers_py.write_text(
                "\n".join(
                    [
                        "from PIL import ImageDraw, ImageFilter, ImageEnhance",
                        "from seedling_sim.domain_randomization import DomainRandomizationConfig",
                        "Replay",
                        "def render_scene_svg(scene): pass",
                        "def render_scene_html(scene):",
                        "    render_scene_svg(scene)",
                        "def render_scene_png(scene):",
                        "    ImageDraw.Draw",
                        "    plant_jitter_px",
                        "    ImageEnhance.Brightness",
                        "    ImageFilter.GaussianBlur",
                        "    np.random.default_rng",
                    ]
                ),
                encoding="utf-8",
            )
            sim_domain_randomization_py.write_text(
                "\n".join(
                    [
                        "class DomainRandomizationConfig:",
                        "    calibration_drift_mm = (0.0, 0.0)",
                        "DOMAIN_RANDOMIZATION_PRESETS = {",
                        "    'greenhouse_default': None,",
                        "    'low_light_noisy': None,",
                        "    'wet_substrate': None,",
                        "}",
                        "def domain_randomization_preset(name): pass",
                    ]
                ),
                encoding="utf-8",
            )
            offline_viewer_py.write_text(
                "\n".join(
                    [
                        "def write_offline_viewer(): pass",
                        "def render_offline_viewer_html(): pass",
                        "def _is_scene_state(): pass",
                        "def _render_scene_state_viewer_html(): pass",
                        "def _scene_cells_table(): pass",
                        "def _scene_targets_table(): pass",
                        "def _scene_detections_table(): pass",
                        "def _scene_target_reasons():",
                        "    forbidden_zone_overlap",
                        "    human_review_required",
                        "    cell_review_required",
                        "def _target_reasons():",
                        "    review_reasons",
                        "    block_reasons",
                        "    blocked_reasons",
                        "    safety_decision",
                        "def _feedback_panel():",
                        "    data-feedback-panel",
                        "    data-feedback-error-type",
                        "    data-feedback-json",
                        "    wrong_target",
                        "    unsafe_action",
                    ]
                ),
                encoding="utf-8",
            )
            replay_viewer_py.write_text(
                "\n".join(
                    [
                        "def write_replay_viewer(): pass",
                        "def render_replay_viewer_html(): pass",
                        "def _timeline_controls():",
                        "    data-action=\"reset\"",
                        "    data-action='play'",
                        "    data-action='pause'",
                        "def _safety_summary_table(): pass",
                        "def _timeline_table(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            ui_cli_py.write_text(
                "\n".join(
                    [
                        "selector = subparsers.add_parser",
                        "model_registry = subparsers.add_parser",
                        "ComponentRegistry.from_file",
                        "ModelRegistry.from_file",
                        "registry.select",
                        "model_id=args.model_id",
                    ]
                ),
                encoding="utf-8",
            )
            report_export_py.write_text(
                "\n".join(
                    [
                        "def write_report_export(): pass",
                        "def build_report_payload(): pass",
                        "def _scene_summary(): pass",
                        "def _distributions(): pass",
                        "def _target_rows(): pass",
                        "def _replay_summary():",
                        "    reason_counts",
                        "    safety_blocks",
                        "def _render_markdown(): pass",
                        "def _render_html(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            feedback_py.write_text(
                "\n".join(
                    [
                        "VALID_FEEDBACK_PRIORITIES = set()",
                        "DEFAULT_PRIORITY_BY_ERROR_TYPE = {}",
                        "class AnnotationFeedback: pass",
                        "class FeedbackAnnotationTask: pass",
                        "def append_feedback(): pass",
                        "def read_feedback(): pass",
                        "def feedback_to_annotation_tasks(): pass",
                        "def write_annotation_tasks_from_feedback(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            simulation_tests.write_text(
                "\n".join(
                    [
                        "def test_sim_scene_round_trip(): pass",
                        "def test_logical_tray_step_removes_target_on_hit(): pass",
                        "def test_logical_tray_can_use_plant_response_model(): pass",
                        "def test_detection_noise_can_drop_all_detections(): pass",
                        "def test_detection_noise_can_add_false_positive_detection(): pass",
                        "def test_sim_scene_detection_noise_can_miss_and_add_false_positive(): pass",
                        "def test_actuator_zone_error_is_reported_in_outcome_info(): pass",
                        "def test_actuator_zone_error_validates_probability(): pass",
                        "def test_plant_response_model_requires_review_for_unknown_plants(): pass",
                        "def test_scene_generator_can_emit_empty_single_multiple_weed_and_unknown_cells(): pass",
                        "def test_image_backed_scene_from_scene_state(): pass",
                        "def test_sim_cli_image_backed_and_synthetic_render(): pass",
                        "def test_render_scene_png_writes_synthetic_image(): pass",
                        "def test_render_scene_png_applies_calibration_drift(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            rl_env_tests.write_text(
                "def test_static_renderer_outputs_svg_and_html(): pass\n",
                encoding="utf-8",
            )
            rl_training_tests.write_text(
                "\n".join(
                    [
                        "def test_rl_train_dry_run_writes_summary_and_registry(): pass",
                        "def test_rl_policy_adapter_reports_blocked_action_mask_reasons(): pass",
                        "def test_compare_runs_applies_thresholds_and_reads_rl_seed_stability(): pass",
                        "def test_sweep_stability_report_groups_runs_across_seeds(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            safety_gate_py.write_text(
                "\n".join(
                    [
                        "class RobotTelemetry:",
                        "    homed = True",
                        "    limit_switch_ok = True",
                        "    emergency_stop_active = False",
                        "class SafetyGate:",
                        "    def validate(self):",
                        "        SafetyDecision",
                        "        BLOCK_CALIBRATION_REQUIRED",
                        "        BLOCK_INTERLOCK",
                        "        BLOCK_UNSAFE_TARGET",
                        "        BLOCK_REVIEW_REQUIRED",
                        "        robot_not_homed",
                        "        limit_switch_not_ok",
                        "        forbidden_zone_overlap",
                        "        target_outside_tray",
                        "        BLOCK_UNSAFE_TARGET",
                        "        target.forbidden_zone_ids",
                        "        _decision(",
                        "def _outside_tray(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            action_masks_py.write_text(
                "\n".join(
                    [
                        "class TargetMask: pass",
                        "class ActionMask:",
                        "    def valid_target_ids(self): pass",
                        "    def reasons_by_target(self): pass",
                        "class ActionMaskBuilder:",
                        "    def build(self, processed_target_ids=None): pass",
                        "processed_target_ids",
                        "already_processed",
                        "risk_score_too_high",
                        "target_uncertainty_too_high",
                        "too_close_to_keep_seedling",
                        "calibration_required",
                        "robot_not_homed",
                        "unknown_plant_present",
                        "ambiguous_or_invalid_cell",
                        "foreign_object_present",
                        "target.forbidden_zone_ids",
                        "forbidden_zone_overlap",
                        "target_outside_tray",
                        "def _outside_tray(): pass",
                        "return TargetMask",
                    ]
                ),
                encoding="utf-8",
            )
            policies_base_py.write_text(
                "\n".join(
                    [
                        "class DecisionPolicy(ABC):",
                        "    policy_id: str",
                        "    processed_target_ids = set()",
                        "    def reset(self): pass",
                        "    @abstractmethod",
                        "    def propose_plan(self): pass",
                        "    def next_action(self): pass",
                        "    def _command_for_target(self):",
                        "        ActionCommand(",
                        "        requires_operator_confirmation=target.human_review_required",
                        "        safety_gate_result=\"not_evaluated\"",
                        "        coordinate_source",
                    ]
                ),
                encoding="utf-8",
            )
            policies_builtin_py.write_text(
                "\n".join(
                    [
                        "class NoOpPolicy:",
                        "    policy_id: str = \"noop\"",
                        "    review_ids = [target.target_id for target in scene.targets]",
                        "    commands=[]",
                        "    noop_policy_requires_review",
                        "class HumanReviewPolicy:",
                        "    policy_id: str = \"human_review\"",
                        "    review_reasons_by_target",
                        "    human_review_required",
                        "    reasons.get(target.target_id)",
                        "    commands=[]",
                        "class RasterScanPolicy:",
                        "    policy_id: str = \"raster_scan\"",
                        "    ActionMaskBuilder",
                        "    def _ordered_targets(self):",
                        "        cell_order = {cell.cell_id: (cell.row, cell.col)}",
                        "        _build_plan(self, scene, self._ordered_targets",
                        "class NearestNeighborPolicy:",
                        "    policy_id: str = \"nearest_neighbor\"",
                        "    _target_distance_from_robot",
                        "    scene.robot.position_mm",
                        "    _build_plan(self, scene, ordered",
                        "class RiskAwareRulePolicy:",
                        "    max_target_uncertainty_mm=2.0",
                        "    min_distance_to_keep_mm=5.0",
                        "    max_risk_score=0.25",
                        "    require_calibration_valid=True",
                        "    require_robot_homed=True",
                        "    target.risk_score",
                        "    _target_distance_from_robot",
                        "class RoutePlanningPolicy:",
                        "    policy_id: str = \"route_planning\"",
                        "    route_planning_method",
                        "    greedy_nearest_neighbor",
                        "    leg_distance_mm",
                        "    cumulative_route_distance_mm",
                        "def _build_plan(): pass",
                        "def _greedy_route():",
                        "    current = _target_point_mm",
                        "blocked_target_ids",
                        "blocked_reasons_by_target",
                        "ActionMaskBuilder",
                    ]
                ),
                encoding="utf-8",
            )
            protocol_py.write_text("class RobotExecutionResult: pass\n", encoding="utf-8")
            dry_run_adapter_py.write_text(
                "\n".join(
                    [
                        "class DryRunSerialAdapter:",
                        "    port: str = \"DRY_RUN\"",
                        "    command_log_path = None",
                        "    def serialized_commands(self): pass",
                        "    def execute_command(self): pass",
                        "SafetyGate",
                        "def telemetry(self):",
                        "    homed=self._homed",
                        "    interlock_ok=self.interlock_ok",
                        "    limit_switch_ok=self.limit_switch_ok",
                        "    emergency_stop_active=self.stopped",
                        "def _require_ready(self): pass",
                        "pointer_only",
                        "no hardware movement performed",
                        "dry-run command serialized",
                        "decision.allowed",
                        "attach_safety_decision",
                        "blocked by SafetyGate",
                    ]
                ),
                encoding="utf-8",
            )
            decision_tests.write_text(
                "\n".join(
                    [
                        "def test_noop_policy_sends_all_targets_to_review(): pass",
                        "def test_nearest_neighbor_policy_orders_safe_targets_by_robot_distance(): pass",
                        "def test_route_planning_policy_uses_cumulative_greedy_route(): pass",
                        "def test_safety_gate_allows_safe_simulation_command(): pass",
                        "def test_safety_gate_blocks_bad_calibration_and_close_target(): pass",
                        "def test_safety_gate_blocks_missing_target_before_execution(): pass",
                        "def test_risk_aware_policy_requires_homed_robot(): pass",
                        "def test_risk_aware_policy_blocks_forbidden_zone_and_outside_tray(): pass",
                        "def test_action_mask_reports_reasons(): pass",
                        "def test_action_mask_blocks_processed_and_unsafe_targets(): pass",
                        "def test_human_review_policy_surfaces_review_and_blocked_targets(): pass",
                        "def test_raster_policy_commands_only_valid_targets(): pass",
                        "def test_safety_gate_blocks_robot_telemetry_faults(): pass",
                        "def test_safety_gate_blocks_real_action_when_software_safe_mode_disabled(): pass",
                        "def test_safety_gate_blocks_real_action_when_enclosure_is_open(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            dry_run_tests.write_text(
                "\n".join(
                    [
                        "def test_dry_run_adapter_serializes_command_after_safety_gate(): pass",
                        "def test_dry_run_adapter_blocks_when_interlock_false(): pass",
                        "def test_dry_run_adapter_requires_homing_and_interlocks_for_direct_moves(): pass",
                        "def test_dry_run_adapter_telemetry_reports_homing_limit_and_estop(): pass",
                        "def test_dry_run_adapter_blocks_unsupported_tool_profile_before_motion(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            robot_tests.write_text(
                "\n".join(
                    [
                        "def test_replay_log_json_round_trip(): pass",
                        "def test_simulator_robot_executes_command_through_safety_gate(): pass",
                        "def test_dry_run_control_points_write_report_and_command_log(): pass",
                        "def test_dry_run_plan_runner_writes_report_command_log_and_replay(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            integration_tests.write_text(
                "\n".join(
                    [
                        "from seedling_core.schemas import SceneState",
                        "from seedling_decision.policies.builtin import RasterScanPolicy",
                        "from seedling_sim.logical_tray import LogicalTraySimulator",
                        "from seedling_sim.actuator_model import ActuatorErrorModel",
                        "from seedling_robot.adapters.simulator import SimulatorRobotAdapter",
                        "def test_rule_based_policy_completes_fixture_scene_through_safety_gate(): pass",
                        "def test_simulator_adapter_executes_command_and_reports_outcome():",
                        "    execute_command",
                        "    safety_decision.allowed",
                        "    removed",
                    ]
                ),
                encoding="utf-8",
            )
            run_snapshot.write_text(
                "\n".join(
                    [
                        "config_hash = True",
                        "command_args = True",
                        "def environment_snapshot(): pass",
                        "def register_run_artifacts(input_paths=None, output_paths=None): pass",
                    ]
                ),
                encoding="utf-8",
            )
            report_tables_py.write_text(
                "\n".join(
                    [
                        "def collect_artifacts(): pass",
                        "def write_artifact_registry_csv():",
                        "    ArtifactRecord.from_path",
                        "    run_snapshot",
                        "def write_dataset_summary_csv():",
                        "    dataset_audit.json",
                        "    raw_dataset_audit.json",
                        "    audit_stage",
                        "    class_counts",
                        "def write_object_level_results_csv():",
                        "    _metrics.json",
                        "    map50",
                        "    map50_95",
                        "    precision_mean",
                        "    recall_mean",
                        "def write_task_level_results_csv():",
                        "    cell_metrics.json",
                        "    scene_metrics.json",
                        "    schema_mode",
                        "    cell_accuracy",
                        "    target_recall",
                        "    mean_coordinate_error_mm",
                        "    expert_keep_remove",
                        "    cost_sensitive",
                        "def _artifact_type(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            report_html_py.write_text(
                "\n".join(
                    [
                        "def write_experiment_html_report():",
                        "    <!doctype html>",
                        "    <table><thead>",
                        "    No rows.",
                    ]
                ),
                encoding="utf-8",
            )
            reports_cli_py.write_text(
                "\n".join(
                    [
                        "write_experiment_html_report",
                        "experiment_report.html",
                        "Dataset Summary",
                        "Object-Level Results",
                        "Task-Level Results",
                        "Artifacts",
                    ]
                ),
                encoding="utf-8",
            )
            registry_py.write_text(
                "\n".join(
                    [
                        "class ArtifactRecord: pass",
                        "class RunRecord: pass",
                        "class ExperimentRegistry:",
                        "    def to_json(self): pass",
                        "    def to_csv(self): pass",
                        "role = 'output'",
                        "sha256 = True",
                        "artifact_registry_json_path = 'artifact_registry.json'",
                        "artifact_registry_csv_path = 'artifact_registry.csv'",
                        "def write_run_registry_records(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            cli_tests.write_text(
                "\n".join(
                    [
                        "def test_architecture_cli_modules_show_help(): pass",
                        "def test_core_register_run_artifacts_replaces_snapshot_record_with_outputs(): pass",
                        "def test_unified_experiment_sim_smoke_writes_report(): pass",
                        "def test_unified_experiment_rl_dry_run_writes_report(): pass",
                        "def test_unified_experiment_rl_baseline_eval_writes_registry(): pass",
                        "def test_prepare_cli_snapshot_records_command_args(): pass",
                        "def test_requirement_splits_match_pyproject_dependency_groups(): pass",
                        "def test_experiment_config_validation_checks_existing_input_paths(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            reports_tests.write_text(
                "\n".join(
                    [
                        "def test_write_run_registry_records_preserves_artifact_roles(): pass",
                        "def test_report_tables_collect_known_artifacts(): pass",
                        "def test_task_level_results_preserve_scene_state_source_ids(): pass",
                        "def test_task_level_results_include_direct_scene_metrics(): pass",
                        "def test_write_run_registry_creates_json_and_csv(): pass",
                        "def test_reports_build_cli_writes_rl_results(): pass",
                        "def test_reports_build_cli_writes_hardware_dry_run_results(): pass",
                        "def test_cell_metrics_diagnostics_build_bootstrap_and_taxonomy(): pass",
                        "def test_error_taxonomy_counts_target_and_cell_errors(): pass",
                        "def test_error_taxonomy_uses_scene_state_metrics_without_legacy_duplicates(): pass",
                        "def test_diagnose_cell_metrics_cli(): pass",
                        "def test_scenario_compare_reports_policy_plan_differences(): pass",
                        "def test_scenario_compare_cli_writes_json_and_html(): pass",
                        "def test_scenario_compare_supports_rl_checkpoint_selector(): pass",
                        "def test_offline_viewer_writes_html(): pass",
                        "def test_offline_viewer_writes_scene_state_cells_and_targets(): pass",
                        "def test_offline_viewer_selects_scene_state_from_bundle(): pass",
                        "def test_replay_viewer_writes_safety_reasons(): pass",
                        "def test_ui_viewer_cli_writes_artifact_registry(): pass",
                        "def test_report_export_writes_markdown_and_html(): pass",
                    ]
                ),
                encoding="utf-8",
            )
            scenario.write_text(
                "\n".join(
                    [
                        "def compare_policies_on_scene():",
                        "    policy_selector",
                        "    route_distance_mm",
                        "    review_targets",
                        "    blocked_targets",
                        "    review_reasons",
                        "    blocked_reasons",
                        "    rl_checkpoint",
                        "    scenario_compare.json",
                        "    scenario_compare.html",
                        "rl_no_model = True",
                        "rl_checkpoint = True",
                    ]
                ),
                encoding="utf-8",
            )
            compare.write_text(
                "\n".join(
                    [
                        "min_rl_seed_count = True",
                        "max_run_snapshot_missing_config_hash = True",
                        "max_run_snapshot_missing_command_args = True",
                    ]
                ),
                encoding="utf-8",
            )
            performance.write_text(
                "\n".join(
                    [
                        "from seedling_vision import batch_predict",
                        "from seedling_rl import train_from_config",
                        "def test_rl_training_dry_run_path_is_bounded(): pass",
                    ]
                ),
                encoding="utf-8",
            )

            passing_content = build_architecture_backlog_audit(doc, repo_root=root)
            passing_items = {item["item_id"]: item for item in passing_content["items"]}

            self.assertEqual(passing_items["A-001"]["status"], "pass")
            self.assertEqual(passing_items["A-002"]["status"], "pass")
            self.assertEqual(passing_items["A-003"]["status"], "pass")
            self.assertEqual(passing_items["A-004"]["status"], "pass")
            self.assertEqual(passing_items["A-005"]["status"], "pass")
            self.assertEqual(passing_items["A-006"]["status"], "pass")
            self.assertEqual(passing_items["A-007"]["status"], "pass")
            self.assertEqual(passing_items["A-008"]["status"], "pass")
            self.assertEqual(passing_items["B-001"]["status"], "pass")
            self.assertEqual(passing_items["B-002"]["status"], "pass")
            self.assertEqual(passing_items["B-006"]["status"], "pass")
            self.assertEqual(passing_items["B-008"]["status"], "pass")
            self.assertEqual(passing_items["B-003"]["status"], "pass")
            self.assertEqual(passing_items["B-004"]["status"], "pass")
            self.assertEqual(passing_items["B-005"]["status"], "pass")
            self.assertEqual(passing_items["B-007"]["status"], "pass")
            self.assertEqual(passing_items["B-009"]["status"], "pass")
            self.assertEqual(passing_items["B-010"]["status"], "pass")
            self.assertEqual(passing_items["B-011"]["status"], "pass")
            self.assertEqual(passing_items["B-012"]["status"], "pass")
            self.assertEqual(passing_items["B-013"]["status"], "pass")
            self.assertEqual(passing_items["B-014"]["status"], "pass")
            self.assertEqual(passing_items["C-001"]["status"], "pass")
            self.assertEqual(passing_items["C-002"]["status"], "pass")
            self.assertEqual(passing_items["C-003"]["status"], "pass")
            self.assertEqual(passing_items["C-004"]["status"], "pass")
            self.assertEqual(passing_items["C-005"]["status"], "pass")
            self.assertEqual(passing_items["C-006"]["status"], "pass")
            self.assertEqual(passing_items["C-007"]["status"], "pass")
            self.assertEqual(passing_items["C-008"]["status"], "pass")
            self.assertEqual(passing_items["C-009"]["status"], "pass")
            self.assertEqual(passing_items["C-010"]["status"], "pass")
            self.assertEqual(passing_items["D-001"]["status"], "pass")
            self.assertEqual(passing_items["D-002"]["status"], "pass")
            self.assertEqual(passing_items["D-003"]["status"], "pass")
            self.assertEqual(passing_items["D-004"]["status"], "pass")
            self.assertEqual(passing_items["D-005"]["status"], "pass")
            self.assertEqual(passing_items["D-006"]["status"], "pass")
            self.assertEqual(passing_items["D-007"]["status"], "pass")
            self.assertEqual(passing_items["D-008"]["status"], "pass")
            self.assertEqual(passing_items["D-009"]["status"], "pass")
            self.assertEqual(passing_items["D-010"]["status"], "pass")
            self.assertEqual(passing_items["D-011"]["status"], "pass")
            self.assertEqual(passing_items["D-012"]["status"], "pass")
            self.assertEqual(passing_items["E-001"]["status"], "pass")
            self.assertEqual(passing_items["E-002"]["status"], "pass")
            self.assertEqual(passing_items["E-003"]["status"], "pass")
            self.assertEqual(passing_items["E-004"]["status"], "pass")
            self.assertEqual(passing_items["E-005"]["status"], "pass")
            self.assertEqual(passing_items["E-006"]["status"], "pass")
            self.assertEqual(passing_items["E-007"]["status"], "pass")
            self.assertEqual(passing_items["E-008"]["status"], "pass")
            self.assertEqual(passing_items["F-001"]["status"], "pass")
            self.assertEqual(passing_items["F-002"]["status"], "pass")
            self.assertEqual(passing_items["F-003"]["status"], "pass")
            self.assertEqual(passing_items["F-004"]["status"], "pass")
            self.assertEqual(passing_items["F-005"]["status"], "pass")
            self.assertEqual(passing_items["F-006"]["status"], "pass")
            self.assertEqual(passing_items["F-007"]["status"], "pass")
            self.assertEqual(passing_items["F-008"]["status"], "pass")
            self.assertEqual(passing_items["F-009"]["status"], "pass")
            self.assertEqual(passing_items["F-010"]["status"], "pass")
            self.assertEqual(passing_items["F-011"]["status"], "pass")
            self.assertEqual(passing_items["G-001"]["status"], "pass")
            self.assertEqual(passing_items["G-002"]["status"], "pass")
            self.assertEqual(passing_items["G-003"]["status"], "pass")
            self.assertEqual(passing_items["G-004"]["status"], "pass")
            self.assertEqual(passing_items["G-005"]["status"], "pass")
            self.assertEqual(passing_items["G-006"]["status"], "pass")
            self.assertEqual(passing_items["G-007"]["status"], "pass")
            self.assertEqual(passing_items["G-008"]["status"], "pass")
            self.assertEqual(passing_items["G-009"]["status"], "pass")
            self.assertEqual(passing_items["G-010"]["status"], "pass")
            self.assertEqual(passing_items["G-011"]["status"], "pass")
            self.assertEqual(passing_items["G-012"]["status"], "pass")
            self.assertEqual(passing_items["I-004"]["status"], "pass")
            self.assertEqual(passing_items["I-005"]["status"], "pass")
            self.assertEqual(passing_items["J-001"]["status"], "pass")
            self.assertEqual(passing_items["J-002"]["status"], "pass")
            self.assertEqual(passing_items["J-003"]["status"], "pass")
            self.assertEqual(passing_items["J-004"]["status"], "pass")
            self.assertEqual(passing_items["J-005"]["status"], "pass")
            self.assertEqual(passing_items["J-006"]["status"], "pass")
            self.assertEqual(passing_items["J-007"]["status"], "pass")
            self.assertEqual(passing_items["K-001"]["status"], "pass")
            self.assertEqual(passing_items["K-002"]["status"], "pass")
            self.assertEqual(passing_items["K-003"]["status"], "pass")
            self.assertEqual(passing_items["K-004"]["status"], "pass")
            self.assertEqual(passing_items["K-005"]["status"], "pass")
            self.assertEqual(passing_items["K-006"]["status"], "pass")
            self.assertEqual(passing_items["K-007"]["status"], "pass")
            self.assertEqual(passing_items["K-008"]["status"], "pass")
            self.assertEqual(passing_items["L-001"]["status"], "pass")
            self.assertEqual(passing_items["L-002"]["status"], "pass")
            self.assertEqual(passing_items["L-003"]["status"], "pass")
            self.assertEqual(passing_items["L-004"]["status"], "pass")
            self.assertEqual(passing_items["L-005"]["status"], "pass")
            self.assertEqual(passing_items["L-006"]["status"], "pass")
            self.assertEqual(passing_items["L-007"]["status"], "pass")
            self.assertEqual(passing_items["L-008"]["status"], "pass")
            self.assertEqual(passing_items["L-009"]["status"], "pass")
            self.assertEqual(passing_items["L-010"]["status"], "pass")
            self.assertEqual(passing_items["L-011"]["status"], "pass")
            self.assertEqual(passing_items["M-001"]["status"], "pass")
            self.assertEqual(passing_items["M-002"]["status"], "pass")
            self.assertEqual(passing_items["M-003"]["status"], "pass")
            self.assertEqual(passing_items["M-004"]["status"], "pass")
            self.assertEqual(passing_items["M-005"]["status"], "pass")
            self.assertEqual(passing_items["M-006"]["status"], "pass")
            self.assertEqual(passing_items["M-007"]["status"], "pass")
            self.assertEqual(passing_items["M-008"]["status"], "pass")
            self.assertEqual(passing_items["M-009"]["status"], "pass")
            self.assertEqual(passing_items["M-010"]["status"], "pass")

    def test_software_readiness_smoke_generates_artifact_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            api_out = Path(tmp) / "software_smoke_api"
            cli_out = Path(tmp) / "software_smoke_cli"

            report = build_software_readiness_smoke(api_out, repo_root=ROOT)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_reports",
                    "software-readiness-smoke",
                    "--out",
                    str(cli_out),
                    "--repo-root",
                    str(ROOT),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )

            payload = json.loads(completed.stdout)
            readiness = json.loads((cli_out / "readiness" / "readiness_report.json").read_text(encoding="utf-8"))
            registry = json.loads((cli_out / "artifact_registry.json").read_text(encoding="utf-8"))
            api_summary = json.loads((api_out / "software_readiness_smoke.json").read_text(encoding="utf-8"))
            scenario_compare = json.loads((cli_out / "scenario_compare" / "scenario_compare.json").read_text(encoding="utf-8"))
            readiness_items = {item["item_id"]: item for item in readiness["items"]}

            self.assertTrue(report["software_ready"], report["readiness_counts"])
            self.assertTrue(payload["software_ready"], payload["readiness_counts"])
            self.assertTrue(payload["physical_validation_required"])
            self.assertEqual(payload["readiness_counts"].get("missing"), None)
            self.assertEqual(readiness["counts"]["external_required"], 2)
            self.assertIn("rl_no_model", {row["policy_selector"] for row in scenario_compare["rows"]})
            self.assertEqual(readiness_items["DOD-011"]["status"], "pass")
            self.assertEqual(payload["architecture_backlog_counts"]["external_required"], 1)
            self.assertIn(
                str(api_out / "architecture_backlog" / "architecture_backlog_audit.json"),
                api_summary["artifacts"],
            )
            self.assertIn(str(api_out / "software_readiness_smoke.json"), api_summary["artifacts"])
            self.assertTrue((cli_out / "calibration" / "calibration.json").exists())
            self.assertTrue((cli_out / "calibration" / "error_map.json").exists())
            self.assertTrue((cli_out / "architecture_backlog" / "architecture_backlog_audit.json").exists())
            self.assertTrue((cli_out / "architecture_backlog" / "architecture_backlog_audit.csv").exists())
            self.assertTrue((cli_out / "architecture_backlog" / "architecture_backlog_audit.html").exists())
            self.assertTrue((cli_out / "robot" / "dry_run_plan_report.json").exists())
            self.assertTrue((cli_out / "hardware_dry_run_results.csv").exists())
            self.assertTrue((cli_out / "safety_results.csv").exists())
            self.assertEqual(registry["runs"][0]["command"], "seedling-reports:software-readiness-smoke")

    def test_run_snapshot_results_include_reproducibility_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            run = root / "run1"
            run.mkdir(parents=True)
            (run / "run_snapshot.json").write_text(
                json.dumps(
                    {
                        "command": "prepare",
                        "command_args": {"config": "config.yaml"},
                        "config_hash": "abc123",
                        "config": {"dataset": {"raw_root": "raw"}},
                        "environment": {
                            "python": "3.11",
                            "platform": "test-platform",
                            "packages": {"numpy": "1.0"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            (run / "offline_viewer.run_snapshot.json").write_text(
                json.dumps(
                    {
                        "command": "seedling-ui:offline-viewer",
                        "command_args": {"out": "offline_viewer.html"},
                        "config_hash": "def456",
                        "config": {"outputs": ["offline_viewer.html"]},
                        "environment": {
                            "python": "3.12",
                            "platform": "viewer-platform",
                            "packages": {"PyYAML": "6.0"},
                        },
                    }
                ),
                encoding="utf-8",
            )

            rows = write_run_snapshot_results_csv(root, Path(tmp) / "snapshots.csv")
            by_command = {row["command"]: row for row in rows}

            self.assertEqual(len(rows), 2)
            self.assertEqual(by_command["prepare"]["config_hash"], "abc123")
            self.assertEqual(by_command["prepare"]["has_config"], True)
            self.assertEqual(by_command["prepare"]["has_command_args"], True)
            self.assertEqual(by_command["prepare"]["python"], "3.11")
            self.assertIn("numpy", by_command["prepare"]["package_versions"])
            self.assertEqual(by_command["seedling-ui:offline-viewer"]["config_hash"], "def456")
            self.assertEqual(by_command["seedling-ui:offline-viewer"]["platform"], "viewer-platform")

    def test_write_run_registry_records_preserves_artifact_roles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_roles"
            run_dir.mkdir()
            input_path = run_dir / "config.yaml"
            output_path = run_dir / "summary.json"
            input_path.write_text("x: 1\n", encoding="utf-8")
            output_path.write_text("{}", encoding="utf-8")

            write_run_registry_records(
                run_dir,
                "wrapper",
                [
                    ArtifactRecord.from_path(input_path, artifact_type="config", role="input", command="wrapper"),
                    ArtifactRecord.from_path(output_path, artifact_type="summary", role="output", command="wrapper"),
                ],
            )

            registry = json.loads((run_dir / "artifact_registry.json").read_text(encoding="utf-8"))
            roles = [artifact["role"] for artifact in registry["runs"][0]["artifacts"]]
            self.assertEqual(roles, ["input", "output"])

    def test_write_run_registry_records_keeps_multiple_commands_in_one_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "reports"
            run_dir.mkdir()
            first = run_dir / "sim.json"
            second = run_dir / "rl.json"
            first.write_text("{}", encoding="utf-8")
            second.write_text("{}", encoding="utf-8")

            write_run_registry_records(
                run_dir,
                "seedling-experiments:sim",
                [ArtifactRecord.from_path(first, artifact_type="sim", role="output")],
            )
            write_run_registry_records(
                run_dir,
                "seedling-experiments:rl:evaluate-baselines",
                [ArtifactRecord.from_path(second, artifact_type="rl", role="output")],
            )

            registry = json.loads((run_dir / "artifact_registry.json").read_text(encoding="utf-8"))
            self.assertEqual(len(registry["runs"]), 2)
            self.assertEqual(
                {run["command"] for run in registry["runs"]},
                {"seedling-experiments:sim", "seedling-experiments:rl:evaluate-baselines"},
            )

    def test_offline_viewer_writes_html(self) -> None:
        payload = {
            "images": [
                {
                    "image": "tray001.jpg",
                    "width": 100,
                    "height": 80,
                    "detections": [
                        {"class_id": 1, "name": "crop_seedling", "confidence": 0.9, "box": [10, 10, 20, 30]}
                    ],
                    "containers": [{"box": [0, 0, 100, 80]}],
                    "container_analysis": [
                        {
                            "index": 1,
                            "box": [0, 0, 100, 80],
                            "matrix": [[2]],
                            "removal_targets": [
                                {
                                    "row": 0,
                                    "col": 0,
                                    "remove_center": [15, 20],
                                    "human_review_required": True,
                                    "safety_decision": {
                                        "allowed": False,
                                        "result": "BLOCK_UNCERTAINTY",
                                        "reasons": ["target_uncertainty_too_high"],
                                    },
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            predictions = Path(tmp) / "predictions.json"
            output = Path(tmp) / "viewer.html"
            predictions.write_text(json.dumps(payload), encoding="utf-8")

            write_offline_viewer(predictions, output)

            html = output.read_text(encoding="utf-8")
            self.assertIn("tray001.jpg", html)
            self.assertIn("Removal Targets", html)
            self.assertIn("target_uncertainty_too_high", html)
            self.assertIn("human_review_required", html)
            self.assertIn("BLOCK_UNCERTAINTY", html)
            self.assertIn("Annotation Feedback", html)
            self.assertIn("data-feedback-panel", html)
            self.assertIn("wrong_target", html)
            self.assertIn("legacy_target", html)

    def test_offline_viewer_writes_scene_state_cells_and_targets(self) -> None:
        scene = _scene_state()
        scene.cells[0].risk_flags = ["operator_review_requested"]
        scene.targets[1].forbidden_zone_ids = ["keep_seedling_zone"]
        with tempfile.TemporaryDirectory() as tmp:
            scene_path = Path(tmp) / "scene.json"
            output = Path(tmp) / "viewer.html"
            scene_path.write_text(json.dumps(scene.to_dict()), encoding="utf-8")

            write_offline_viewer(scene_path, output)

            html = output.read_text(encoding="utf-8")
            self.assertIn("Scene Summary", html)
            self.assertIn("Cells", html)
            self.assertIn("Action Targets", html)
            self.assertIn("scene_001", html)
            self.assertIn("multiple_crop", html)
            self.assertIn("target_review", html)
            self.assertIn("human_review_required", html)
            self.assertIn("forbidden_zone_overlap", html)
            self.assertIn("operator_review_requested", html)
            self.assertIn("robot_point_mm", html)
            self.assertIn("Annotation Feedback", html)
            self.assertIn("data-feedback-panel", html)
            self.assertIn("target_review", html)
            self.assertIn("bad_cell_state", html)
            self.assertIn("offline_viewer", html)

    def test_offline_viewer_selects_scene_state_from_bundle(self) -> None:
        scene_a = _scene_state().to_dict()
        scene_b = _scene_state().to_dict()
        scene_b["scene_id"] = "scene_002"
        scene_b["image_ref"] = "tray002.jpg"
        scene_b["targets"][0]["target_id"] = "target_002"
        with tempfile.TemporaryDirectory() as tmp:
            scene_path = Path(tmp) / "scenes.json"
            output = Path(tmp) / "viewer.html"
            scene_path.write_text(json.dumps({"scenes": [scene_a, scene_b]}), encoding="utf-8")

            write_offline_viewer(scene_path, output, image_name="scene_002")

            html = output.read_text(encoding="utf-8")
            self.assertIn("scene_002", html)
            self.assertIn("tray002.jpg", html)
            self.assertIn("target_002", html)
            self.assertNotIn("scene_001", html)

    def test_replay_viewer_writes_safety_reasons(self) -> None:
        scene = SimScene(
            scene_id="scene001",
            grid_rows=1,
            grid_cols=1,
            cell_size_mm=[33.0, 33.0],
            plants=[SimPlant("plant_001", 0, 0, "crop_seedling", [10.0, 10.0])],
            targets=[SimTarget("target_001", "plant_001", "remove_extra_crop", [10.0, 10.0])],
        )
        logger = ReplayLogger(replay_id="replay001", scene_id="scene001")
        logger.append(
            "robot_execution",
            {
                "ok": False,
                "safety_decision": {
                    "allowed": False,
                    "result": "BLOCK_CALIBRATION_REQUIRED",
                    "reasons": ["calibration_required"],
                },
                "message": "blocked",
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            scene_path = Path(tmp) / "scene.json"
            replay_path = Path(tmp) / "replay.json"
            output = Path(tmp) / "replay.html"
            scene.to_json(scene_path)
            logger.to_json(replay_path)

            write_replay_viewer(scene_path, replay_path, output)

            html = output.read_text(encoding="utf-8")
            self.assertIn("Safety Summary", html)
            self.assertIn("blocked_steps", html)
            self.assertIn("reason_counts", html)
            self.assertIn("BLOCK_CALIBRATION_REQUIRED", html)
            self.assertIn("calibration_required", html)
            self.assertIn("calibration_required: 1", html)
            self.assertIn('data-replay-controls', html)
            self.assertIn('data-action="play"', html)
            self.assertIn('data-action="pause"', html)
            self.assertIn('data-action="step"', html)
            self.assertIn('data-action="reset"', html)
            self.assertIn('data-timeline-row="0"', html)

    def test_ui_viewer_cli_writes_artifact_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "predictions.json"
            offline_out = root / "offline_viewer.html"
            scene_path = root / "scene.json"
            replay_path = root / "replay.json"
            replay_out = root / "replay_viewer.html"
            predictions.write_text(json.dumps({"images": [{"image": "tray001.jpg", "detections": []}]}), encoding="utf-8")
            SimScene(scene_id="scene001", grid_rows=1, grid_cols=1, cell_size_mm=[33.0, 33.0]).to_json(scene_path)
            ReplayLogger(replay_id="replay001", scene_id="scene001").to_json(replay_path)

            offline = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_ui",
                    "offline-viewer",
                    "--predictions",
                    str(predictions),
                    "--out",
                    str(offline_out),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            replay = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_ui",
                    "replay-viewer",
                    "--scene",
                    str(scene_path),
                    "--replay",
                    str(replay_path),
                    "--out",
                    str(replay_out),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            offline_payload = json.loads(offline.stdout)
            replay_payload = json.loads(replay.stdout)
            registry = json.loads((root / "artifact_registry.json").read_text(encoding="utf-8"))
            offline_snapshot = json.loads((root / "offline_viewer.run_snapshot.json").read_text(encoding="utf-8"))
            replay_snapshot = json.loads((root / "replay_viewer.run_snapshot.json").read_text(encoding="utf-8"))

            self.assertTrue(offline_payload["artifact_registry"])
            self.assertTrue(replay_payload["artifact_registry"])
            self.assertEqual(offline_payload["run_snapshot"], str(root / "offline_viewer.run_snapshot.json"))
            self.assertEqual(replay_payload["run_snapshot"], str(root / "replay_viewer.run_snapshot.json"))
            self.assertEqual(offline_snapshot["command"], "seedling-ui:offline-viewer")
            self.assertEqual(replay_snapshot["command"], "seedling-ui:replay-viewer")
            self.assertTrue((root / "artifact_registry.csv").exists())
            self.assertEqual(
                {run["command"] for run in registry["runs"]},
                {"seedling-ui:offline-viewer", "seedling-ui:replay-viewer"},
            )
            replay_run = next(run for run in registry["runs"] if run["command"] == "seedling-ui:replay-viewer")
            self.assertEqual([artifact["role"] for artifact in replay_run["artifacts"]], ["input", "input", "output", "output"])
            self.assertIn(str(root / "replay_viewer.run_snapshot.json"), {artifact["path"] for artifact in replay_run["artifacts"]})

    def test_report_export_writes_markdown_and_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scene_path = Path(tmp) / "scene.json"
            replay_path = Path(tmp) / "replay.json"
            markdown_path = Path(tmp) / "tray_report.md"
            html_path = Path(tmp) / "tray_report.html"
            scene = _scene_state()
            scene.cells[0].risk_flags = ["operator_review_requested"]
            scene.targets[1].forbidden_zone_ids = ["keep_seedling_zone"]
            scene_path.write_text(json.dumps(scene.to_dict()), encoding="utf-8")
            logger = ReplayLogger(replay_id="replay001", scene_id="scene_001")
            logger.append("target_step", {"target_id": "target_001", "reward": 2.0, "ok": True})
            logger.append(
                "robot_execution",
                {
                    "ok": False,
                    "target_id": "target_review",
                    "safety_decision": {
                        "allowed": False,
                        "result": "BLOCK_REVIEW",
                        "reasons": ["operator_confirmation_required"],
                    },
                },
            )
            logger.to_json(replay_path)

            report = write_report_export(scene_path, markdown_path, replay_path=replay_path)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_ui",
                    "report-export",
                    "--scene",
                    str(scene_path),
                    "--replay",
                    str(replay_path),
                    "--out",
                    str(html_path),
                    "--title",
                    "Tray Report",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout)

            self.assertEqual(report["scene_id"], "scene_001")
            self.assertEqual(payload["targets"], 2)
            self.assertEqual(payload["replay_steps"], 2)
            self.assertTrue(payload["artifact_registry"])
            self.assertEqual(payload["run_snapshot"], str(Path(tmp) / "tray_report.run_snapshot.json"))
            run_snapshot = json.loads((Path(tmp) / "tray_report.run_snapshot.json").read_text(encoding="utf-8"))
            self.assertEqual(run_snapshot["command"], "seedling-ui:report-export")
            self.assertEqual(run_snapshot["metadata"]["scene_id"], "scene_001")
            registry = json.loads((Path(tmp) / "artifact_registry.json").read_text(encoding="utf-8"))
            self.assertEqual(registry["runs"][0]["command"], "seedling-ui:report-export")
            self.assertEqual(registry["runs"][0]["metadata"]["scene_id"], "scene_001")
            markdown = markdown_path.read_text(encoding="utf-8")
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("## Scene Summary", markdown)
            self.assertIn("target_001", markdown)
            self.assertIn("human_review_required", markdown)
            self.assertIn("forbidden_zone_overlap", markdown)
            self.assertIn("operator_review_requested", markdown)
            self.assertIn("Replay Summary", html)
            self.assertIn("Tray Report", html)
            self.assertIn("BLOCK_REVIEW", html)
            self.assertIn("operator_confirmation_required", html)
            self.assertIn("reason_counts", html)

    def test_scenario_compare_reports_policy_plan_differences(self) -> None:
        scene = _scene_state()

        rows = compare_policies_on_scene(scene, ["noop", "raster_scan", "route_planning", "human_review", "rl_no_model"])

        by_policy = {row["policy"]: row for row in rows}
        self.assertEqual(by_policy["noop"]["commands"], 0)
        self.assertEqual(by_policy["raster_scan"]["commands"], 1)
        self.assertEqual(by_policy["route_planning"]["commands"], 1)
        self.assertGreater(by_policy["route_planning"]["route_distance_mm"], 0.0)
        self.assertEqual(by_policy["human_review"]["review_targets"], 1)
        self.assertIn("human_review_required", by_policy["human_review"]["review_reasons"])
        self.assertEqual(by_policy["rl_no_model"]["commands"], 0)
        self.assertEqual(by_policy["rl_no_model"]["rl_action_kind"], "no_model")
        self.assertEqual(by_policy["rl_no_model"]["review_targets"], 1)

    def test_scenario_compare_cli_writes_json_and_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scene_path = Path(tmp) / "scene.json"
            output = Path(tmp) / "scenario_compare"
            scene_path.write_text(json.dumps(_scene_state().to_dict()), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_reports",
                    "scenario-compare",
                    "--scene",
                    str(scene_path),
                    "--policies",
                    "raster_scan",
                    "route_planning",
                    "rl_no_model",
                    "risk_aware_rule",
                    "--out",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(completed.stdout)
            registry = json.loads((output / "artifact_registry.json").read_text(encoding="utf-8"))
            run_snapshot = json.loads((output / "run_snapshot.json").read_text(encoding="utf-8"))

            self.assertTrue(payload["ok"])
            self.assertTrue(payload["artifact_registry"])
            self.assertEqual(payload["run_snapshot"], str(output / "run_snapshot.json"))
            self.assertEqual(run_snapshot["command"], "seedling-reports:scenario-compare")
            self.assertTrue((output / "scenario_compare.json").exists())
            self.assertTrue((output / "scenario_compare.html").exists())
            compare_payload = json.loads((output / "scenario_compare.json").read_text(encoding="utf-8"))
            self.assertIn("rl_no_model", {row["policy"] for row in compare_payload["rows"]})
            self.assertEqual(registry["runs"][0]["command"], "seedling-reports:scenario-compare")

    def test_export_policy_plan_writes_action_plan_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scene_path = Path(tmp) / "scene.json"
            output = Path(tmp) / "action_plan.json"
            cli_output = Path(tmp) / "action_plan_cli.json"
            scene_path.write_text(json.dumps(_scene_state().to_dict()), encoding="utf-8")

            plan = write_policy_plan_file(scene_path, "route_planning", output)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_reports",
                    "export-plan",
                    "--scene",
                    str(scene_path),
                    "--policy",
                    "route_planning",
                    "--out",
                    str(cli_output),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout)
            saved = json.loads(output.read_text(encoding="utf-8"))
            cli_saved = json.loads(cli_output.read_text(encoding="utf-8"))
            registry = json.loads((Path(tmp) / "artifact_registry.json").read_text(encoding="utf-8"))
            run_snapshot = json.loads((Path(tmp) / "run_snapshot.json").read_text(encoding="utf-8"))

            self.assertEqual(plan.policy_id, "route_planning")
            self.assertEqual(payload["commands"], 1)
            self.assertTrue(payload["artifact_registry"])
            self.assertEqual(payload["run_snapshot"], str(Path(tmp) / "run_snapshot.json"))
            self.assertEqual(run_snapshot["command"], "seedling-reports:export-plan")
            self.assertEqual(saved["plan"]["commands"][0]["target_id"], "target_001")
            self.assertEqual(cli_saved["plan"]["policy_id"], "route_planning")
            self.assertEqual(registry["runs"][0]["command"], "seedling-reports:export-plan")

    def test_export_rl_no_model_plan_writes_adapter_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scene_path = Path(tmp) / "scene.json"
            output = Path(tmp) / "rl_action_plan.json"
            scene_path.write_text(json.dumps(_scene_state().to_dict()), encoding="utf-8")

            plan = write_policy_plan_file(scene_path, "rl_no_model", output)
            saved = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(plan.policy_id, "rl_no_model")
            self.assertEqual(saved["plan"]["metadata"]["rl_action_kind"], "no_model")
            self.assertEqual(saved["commands"], 0)

    def test_scenario_compare_supports_rl_checkpoint_selector(self) -> None:
        class FakeRLAdapter:
            def propose_plan(self, scene: SceneState) -> ActionPlan:
                return ActionPlan(
                    plan_id="fake_rl_plan",
                    policy_id="rl_policy_adapter",
                    commands=[],
                    metadata={"rl_action_kind": "stop"},
                )

        scene = _scene_state()
        with patch("seedling_rl.RLPolicyAdapter.from_sb3", return_value=FakeRLAdapter()) as from_sb3:
            rows = compare_policies_on_scene(scene, ["rl_checkpoint:runs/rl/model.zip"])

        from_sb3.assert_called_once_with("runs/rl/model.zip", policy_id="rl_policy_adapter")
        self.assertEqual(rows[0]["policy_selector"], "rl_checkpoint:runs/rl/model.zip")
        self.assertEqual(rows[0]["policy"], "rl_policy_adapter")
        self.assertEqual(rows[0]["rl_action_kind"], "stop")

    def test_cell_metrics_diagnostics_build_bootstrap_and_taxonomy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "cell_metrics.json"
            output = Path(tmp) / "diagnostics"
            metrics_path.write_text(json.dumps(_cell_metrics_payload()), encoding="utf-8")

            diagnostics = build_cell_metrics_diagnostics(metrics_path, output, iterations=50, seed=1)

            self.assertIn("cell_accuracy", diagnostics["bootstrap"])
            self.assertEqual(diagnostics["bootstrap"]["groups"], 2)
            self.assertEqual(diagnostics["bootstrap"]["critical_error_rate_per_cell"]["mean"], 0.25)
            self.assertEqual(diagnostics["bootstrap"]["normalized_cost_per_cell"]["mean"], 2.5)
            self.assertIn("container_unmatched", diagnostics["error_taxonomy"]["counts"])
            self.assertEqual(diagnostics["error_taxonomy"]["group_counts"]["geometry"], 2)
            self.assertEqual(diagnostics["error_taxonomy"]["category_groups"]["target_false_negative"], "decision")
            self.assertEqual(diagnostics["error_taxonomy"]["cost_sensitive"]["critical_error_total"], 2)
            self.assertTrue((output / "bootstrap_ci.json").exists())
            self.assertTrue((output / "error_taxonomy.json").exists())

    def test_error_taxonomy_counts_target_and_cell_errors(self) -> None:
        taxonomy = error_taxonomy(_cell_metrics_payload())

        self.assertEqual(taxonomy["counts"]["missing_prediction"], 1)
        self.assertEqual(taxonomy["counts"]["cell_state_mismatch"], 1)
        self.assertEqual(taxonomy["counts"]["target_false_negative"], 1)
        self.assertEqual(taxonomy["group_counts"]["image"], 1)
        self.assertEqual(taxonomy["group_counts"]["geometry"], 2)
        self.assertEqual(taxonomy["group_counts"]["biology"], 3)
        self.assertEqual(taxonomy["group_counts"]["decision"], 1)
        self.assertEqual(taxonomy["samples"]["container_low_iou"][0]["group"], "geometry")
        self.assertEqual(taxonomy["cost_sensitive"]["total_cost"], 20.0)
        self.assertEqual(taxonomy["cost_sensitive"]["critical_error_counts"]["missed_multiple_crop"], 1)

    def test_error_taxonomy_uses_scene_state_metrics_without_legacy_duplicates(self) -> None:
        taxonomy = error_taxonomy(_scene_state_cell_metrics_payload())

        self.assertNotIn("missing_prediction", taxonomy["counts"])
        self.assertEqual(taxonomy["counts"]["cell_missing_prediction"], 1)
        self.assertEqual(taxonomy["counts"]["cell_extra_prediction"], 1)
        self.assertEqual(taxonomy["counts"]["cell_state_mismatch"], 2)
        self.assertEqual(taxonomy["counts"]["target_coordinate_error"], 1)
        self.assertEqual(taxonomy["counts"]["target_false_positive"], 2)
        self.assertEqual(taxonomy["counts"]["target_false_negative"], 1)
        self.assertEqual(taxonomy["group_counts"]["image"], 2)
        self.assertEqual(taxonomy["group_counts"]["biology"], 2)
        self.assertEqual(taxonomy["group_counts"]["geometry"], 1)
        self.assertEqual(taxonomy["group_counts"]["decision"], 3)
        self.assertEqual(taxonomy["samples"]["cell_state_mismatch"][0]["gt_scene_id"], "gt_scene")
        self.assertEqual(taxonomy["samples"]["target_false_positive"][0]["target_id"], "pred_false_001")
        self.assertEqual(taxonomy["category_groups"]["target_coordinate_error"], "geometry")

    def test_diagnose_cell_metrics_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "cell_metrics.json"
            output = Path(tmp) / "diagnostics"
            metrics_path.write_text(json.dumps(_cell_metrics_payload()), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_reports",
                    "diagnose-cell-metrics",
                    "--metrics",
                    str(metrics_path),
                    "--out",
                    str(output),
                    "--iterations",
                    "20",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(completed.stdout)
            registry = json.loads((output / "artifact_registry.json").read_text(encoding="utf-8"))

            self.assertTrue(payload["ok"])
            self.assertTrue(payload["artifact_registry"])
            self.assertTrue((output / "bootstrap_ci.json").exists())
            self.assertEqual(registry["runs"][0]["command"], "seedling-reports:diagnose-cell-metrics")

def _scene_state() -> SceneState:
    return SceneState(
        scene_id="scene_001",
        image_ref="tray001.jpg",
        dataset_version="zks_v0_1",
        ontology_version="ontology_v0_1",
        image_size_px=[100, 100],
        tray=TrayState(tray_id="tray001", grid_rows=1, grid_cols=1, bbox_xyxy_px=[0, 0, 100, 100]),
        robot=RobotState(position_mm=[0, 0, 0], homed=True, mode="simulation"),
        cells=[
            CellState(
                cell_id="r00_c00",
                row=0,
                col=0,
                polygon_px=[[0, 0], [100, 0], [100, 100], [0, 100]],
                state="multiple_crop",
            )
        ],
        targets=[
            ActionTarget(
                target_id="target_001",
                cell_id="r00_c00",
                object_id="obj_001",
                target_type="remove_extra_crop",
                action_point_px=[10, 10],
                action_point_mm=[10, 10],
                robot_point_mm=[10, 10, 0],
                uncertainty_radius_mm=1.0,
                min_distance_to_keep_mm=6.0,
            ),
            ActionTarget(
                target_id="target_review",
                cell_id="r00_c00",
                object_id="obj_002",
                target_type="remove_extra_crop",
                action_point_px=[20, 20],
                action_point_mm=[20, 20],
                robot_point_mm=[20, 20, 0],
                human_review_required=True,
                uncertainty_radius_mm=1.0,
                min_distance_to_keep_mm=6.0,
            )
        ],
        safety=SafetyState(calibration_valid=True, interlock_ok=True),
    )


def _cell_metrics_payload() -> dict[str, object]:
    return {
        "prediction_coverage": {
            "missing_predictions": ["missing.jpg"],
            "extra_predictions": [],
        },
        "suspected_augmented_eval_images": [],
        "container_matching": {
            "iou_threshold": 0.5,
            "unmatched_samples": [{"image": "tray002.jpg", "best_iou": 0.2}],
        },
        "images": [
            {
                "image": "tray001.jpg",
                "gt_containers": 1,
                "pred_containers": 1,
                "matched_containers": 1,
                "best_container_iou": 0.9,
                "cell_accuracy": 1.0,
                "cell_correct": 4,
                "cell_total": 4,
                "multi_tp": 1,
                "multi_fp": 0,
                "multi_fn": 0,
                "target_tp": 1,
                "target_fp": 0,
                "target_fn": 0,
                "cost_total": 0.0,
                "normalized_cost_per_cell": 0.0,
                "critical_error_total": 0,
                "critical_error_rate_per_cell": 0.0,
                "critical_error_counts": {},
            },
            {
                "image": "tray002.jpg",
                "gt_containers": 1,
                "pred_containers": 1,
                "matched_containers": 0,
                "best_container_iou": 0.2,
                "cell_accuracy": 0.5,
                "cell_correct": 2,
                "cell_total": 4,
                "multi_tp": 0,
                "multi_fp": 1,
                "multi_fn": 1,
                "target_tp": 0,
                "target_fp": 0,
                "target_fn": 1,
                "cost_total": 20.0,
                "normalized_cost_per_cell": 5.0,
                "critical_error_total": 2,
                "critical_error_rate_per_cell": 0.5,
                "critical_error_counts": {
                    "target_false_negative": 1,
                    "missed_multiple_crop": 1,
                },
            },
        ],
        "cost_sensitive": {
            "total_cost": 20.0,
            "normalized_cost_per_cell": 2.5,
            "critical_error_total": 2,
            "critical_error_rate_per_cell": 0.25,
            "critical_error_counts": {
                "target_false_negative": 1,
                "missed_multiple_crop": 1,
            },
            "cost_breakdown": {
                "target_false_negative": 12.0,
                "missed_multiple_crop": 8.0,
            },
        },
    }


def _scene_state_cell_metrics_payload() -> dict[str, object]:
    return {
        "schema_mode": "scene_state",
        "prediction_coverage": {
            "missing_predictions": ["r00_c03"],
            "extra_predictions": ["r00_c99"],
        },
        "suspected_augmented_eval_images": [],
        "container_matching": {
            "prediction_source": "scene_state",
            "iou_threshold": None,
            "unmatched_samples": [],
        },
        "images": [
            {
                "image": "pred_scene",
                "cell_accuracy": 0.5,
                "target_fp": 99,
                "target_fn": 99,
            }
        ],
        "scene_state_metrics": {
            "gt_scene_id": "gt_scene",
            "pred_scene_id": "pred_scene",
            "cell_metrics": {
                "accuracy": 0.5,
                "total_cells": 4,
                "correct_cells": 2,
                "missing_predictions": ["r00_c03"],
                "unexpected_predictions": ["r00_c99"],
                "confusion": {
                    "single_crop": {"single_crop": 1},
                    "multiple_crop": {"single_crop": 1},
                    "crop_and_weed": {"single_crop": 1},
                },
            },
            "target_metrics": {
                "tp": 1,
                "fp": 2,
                "fn": 1,
                "matches": [
                    {
                        "gt_target_id": "target_ok",
                        "pred_target_id": "pred_ok",
                        "distance_px": 3.0,
                        "distance_mm": 0.4,
                    },
                    {
                        "gt_target_id": "target_missed",
                        "pred_target_id": None,
                        "distance_px": None,
                        "distance_mm": None,
                    },
                ],
                "false_positives": ["pred_false_001", "pred_false_002"],
                "false_negatives": ["target_missed"],
            },
        },
    }


def _post_action_row(command_id: str, hours: float, outcome: str) -> dict[str, object]:
    return {
        "command_id": command_id,
        "target_id": "target_001",
        "scene_id": "scene_001",
        "tray_id": "tray_001",
        "observed_at": f"2026-06-18T{int(hours) % 24:02d}:00:00+00:00",
        "hours_after_action": hours,
        "outcome": outcome,
        "image_ref": f"after_{int(hours)}h.png",
    }


def _write_readiness_calibration(
    path: Path,
    *,
    valid_until: str = "2030-01-01T00:00:00+00:00",
    p95: float = 1.0,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "calibration_artifact_v0_1",
                "calibration_id": "calib_001",
                "created_at": "2026-06-18T00:00:00+00:00",
                "camera_id": "cam_001",
                "tray_type": "11x11",
                "image_to_tray_homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "tray_to_robot_transform": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "tool_offset_mm": [0, 0, 0],
                "px_per_mm_x": 2.0,
                "px_per_mm_y": 2.0,
                "valid_until": valid_until,
                "error_summary_mm": {"p50": 0.5, "p95": p95, "p99": p95, "rms": 0.75, "max": p95},
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
