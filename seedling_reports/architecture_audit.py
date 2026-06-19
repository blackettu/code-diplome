from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .html import write_experiment_html_report


BACKLOG_TABLE_RE = re.compile(
    r"^\|\s*(?P<item_id>[A-M]-\d{3})\s*\|\s*(?P<priority>P\d)\s*\|\s*(?P<title>.*?)\s*\|\s*(?P<acceptance>.*?)\s*\|\s*$"
)

EXTERNAL_BACKLOG_IDS = {"I-009"}

EVIDENCE_RULES: dict[str, list[str]] = {
    "A-001": ["pyproject.toml", "seedling_experiments/__main__.py"],
    "A-002": ["requirements/base.txt", "requirements/vision.txt", "requirements/rl.txt", "requirements/robot.txt", "requirements/dev.txt"],
    "A-003": ["seedling_core/config.py", "seedling_experiments/config.py"],
    "A-004": ["seedling_core/schemas.py", "tests/unit/test_schemas.py"],
    "A-005": ["seedling_core/run_snapshot.py"],
    "A-006": ["seedling_reports/registry.py", "seedling_core/run_snapshot.py", "tests/unit/test_reports_and_viewers.py", "tests/unit/test_cli_smoke.py"],
    "A-007": ["seedling_vision/migration.py", "tests/unit/test_vision_migration.py"],
    "A-008": [".pre-commit-config.yaml", ".github/workflows/ci.yml", "pyproject.toml"],
    "B-001": ["configs/ontology/ontology_v0_1.yaml"],
    "B-002": ["seedling_data/ontology.py", "tests/unit/test_ontology_and_adapters.py"],
    "B-003": ["docs/DATASET_CARD_TEMPLATE.md"],
    "B-004": ["docs/ANNOTATION_GUIDE.md"],
    "B-005": ["seedling_experiments/dataset.py", "tests/unit/test_dataset_split.py"],
    "B-006": ["seedling_data/manifests.py", "tests/unit/test_data_registry_annotations.py"],
    "B-007": ["seedling_experiments/dataset.py", "tests/unit/test_dataset_split.py"],
    "B-008": ["seedling_data/duplicates.py", "tests/unit/test_data_registry_annotations.py"],
    "B-009": ["seedling_data/annotations.py", "docs/ANNOTATION_GUIDE.md", "tests/unit/test_data_registry_annotations.py"],
    "B-010": ["seedling_data/annotations.py", "docs/ANNOTATION_GUIDE.md", "tests/unit/test_data_registry_annotations.py"],
    "B-011": ["seedling_data/annotations.py", "tests/unit/test_data_registry_annotations.py"],
    "B-012": ["seedling_data/annotations.py", "tests/unit/test_data_registry_annotations.py"],
    "B-013": ["seedling_data/registry.py", "tests/unit/test_data_registry_annotations.py"],
    "B-014": ["seedling_data/changelog.py", "docs/DATASET_CHANGELOG_TEMPLATE.md", "tests/unit/test_data_registry_annotations.py"],
    "C-001": ["seedling_vision/adapters/base.py"],
    "C-002": ["seedling_vision/adapters/ultralytics_yolo.py", "tests/unit/test_vision_tools.py"],
    "C-003": ["seedling_vision/adapters/baseline_green.py", "tests/unit/test_registry_selector_feedback.py"],
    "C-004": ["seedling_vision/adapters/recorded_prediction.py", "tests/unit/test_ontology_and_adapters.py"],
    "C-005": ["seedling_vision/adapters/onnx.py", "tests/unit/test_vision_tools.py"],
    "C-006": ["seedling_vision/uncertainty.py", "tests/unit/test_vision_tools.py"],
    "C-007": ["seedling_vision/postprocess.py", "tests/unit/test_vision_tools.py"],
    "C-008": ["seedling_core/registry.py", "configs/registry/model_registry_v0_1.yaml", "tests/unit/test_registry_selector_feedback.py"],
    "C-009": ["seedling_vision/batch.py", "seedling_vision/cli.py", "tests/unit/test_vision_tools.py"],
    "C-010": ["seedling_vision/overlays.py", "tests/unit/test_vision_tools.py"],
    "D-001": ["seedling_cells/grid.py", "tests/unit/test_grid.py"],
    "D-002": ["seedling_cells/cell_state.py", "tests/unit/test_grid.py"],
    "D-003": ["seedling_cells/cell_state.py", "tests/unit/test_target_generation.py"],
    "D-004": ["seedling_cells/target_generation.py", "tests/unit/test_target_generation.py"],
    "D-005": ["seedling_experiments/evaluate.py", "tests/unit/test_evaluate_cells_metrics.py"],
    "D-006": ["seedling_experiments/evaluate.py", "seedling_vision/postprocess.py", "tests/unit/test_evaluate_cells_metrics.py"],
    "D-007": ["seedling_experiments/evaluate.py", "tests/unit/test_evaluate_cells_metrics.py"],
    "D-008": ["seedling_cells/evaluation.py", "tests/unit/test_scene_state_evaluation.py"],
    "D-009": ["seedling_cells/evaluation.py", "tests/unit/test_scene_state_evaluation.py"],
    "D-010": ["seedling_cells/evaluation.py", "seedling_reports/diagnostics.py", "tests/unit/test_reports_and_viewers.py"],
    "D-011": ["seedling_reports/diagnostics.py", "tests/unit/test_evaluate_cells_metrics.py"],
    "D-012": ["seedling_reports/diagnostics.py", "tests/unit/test_reports_and_viewers.py"],
    "E-001": ["seedling_calibration/schemas.py"],
    "E-002": ["seedling_calibration/transforms.py", "tests/unit/test_calibration.py"],
    "E-003": ["seedling_calibration/transforms.py", "tests/unit/test_calibration.py"],
    "E-004": ["seedling_calibration/validator.py", "tests/unit/test_calibration.py"],
    "E-005": ["seedling_calibration/estimation.py", "docs/CALIBRATION_PROTOCOL.md", "tests/unit/test_calibration.py"],
    "E-006": ["seedling_calibration/transforms.py", "tests/unit/test_calibration.py"],
    "E-007": ["seedling_calibration/cli.py", "tests/unit/test_calibration.py"],
    "E-008": ["seedling_calibration/intrinsics.py", "tests/unit/test_calibration.py"],
    "F-001": ["seedling_decision/policies/base.py"],
    "F-002": ["seedling_decision/policies/builtin.py"],
    "F-003": ["seedling_decision/policies/builtin.py"],
    "F-004": ["seedling_decision/policies/builtin.py"],
    "F-005": ["seedling_decision/policies/builtin.py"],
    "F-006": ["seedling_decision/action_masks.py", "tests/unit/test_decision_and_safety.py"],
    "F-007": [
        "seedling_decision/safety_gate.py",
        "seedling_robot/adapters/dry_run_serial.py",
        "tests/unit/test_decision_and_safety.py",
        "tests/unit/test_dry_run_and_feedback.py",
    ],
    "F-008": [
        "seedling_decision/action_masks.py",
        "seedling_decision/policies/builtin.py",
        "seedling_decision/safety_gate.py",
        "tests/unit/test_decision_and_safety.py",
    ],
    "F-009": ["seedling_decision/policies/builtin.py"],
    "F-010": ["seedling_reports/scenario.py", "tests/unit/test_reports_and_viewers.py"],
    "F-011": ["seedling_decision/policies/builtin.py"],
    "G-001": ["seedling_sim/schemas.py", "tests/unit/test_simulation.py"],
    "G-002": ["seedling_sim/logical_tray.py", "tests/unit/test_simulation.py"],
    "G-003": ["seedling_sim/detection_noise.py", "tests/unit/test_simulation.py"],
    "G-004": ["seedling_sim/actuator_model.py", "tests/unit/test_simulation.py"],
    "G-005": ["seedling_sim/plant_response_model.py", "tests/unit/test_simulation.py"],
    "G-006": ["seedling_sim/scene_generator.py", "tests/unit/test_simulation.py"],
    "G-007": ["seedling_sim/image_backed.py", "tests/unit/test_simulation.py"],
    "G-008": ["seedling_sim/replay.py", "tests/unit/test_robot_and_replay.py"],
    "G-009": ["seedling_sim/renderers.py", "tests/unit/test_simulation.py", "tests/unit/test_rl_env.py"],
    "G-010": ["seedling_ui/replay_viewer.py", "tests/unit/test_reports_and_viewers.py"],
    "G-011": ["seedling_sim/renderers.py", "tests/unit/test_simulation.py"],
    "G-012": ["seedling_sim/domain_randomization.py", "tests/unit/test_simulation.py"],
    "H-001": ["seedling_rl/envs.py", "tests/unit/test_rl_training_and_compare.py"],
    "H-002": ["seedling_rl/envs.py", "tests/unit/test_rl_env.py"],
    "H-003": ["seedling_rl/reward.py", "tests/unit/test_rl_env.py"],
    "H-004": ["seedling_rl/baseline_eval.py", "tests/unit/test_rl_env.py"],
    "H-005": ["seedling_rl/training.py", "tests/unit/test_rl_training_and_compare.py"],
    "H-006": ["seedling_rl/training.py", "tests/unit/test_rl_training_and_compare.py"],
    "H-007": ["seedling_rl/policy_adapter.py", "tests/unit/test_rl_training_and_compare.py"],
    "H-008": ["seedling_rl/callbacks.py", "tests/unit/test_rl_training_and_compare.py"],
    "H-009": ["seedling_rl/vectorized_env.py", "tests/unit/test_rl_training_and_compare.py"],
    "H-010": ["seedling_rl/curriculum.py", "tests/unit/test_rl_training_and_compare.py"],
    "H-011": ["seedling_rl/policy_adapter.py", "configs/rl/recurrent_ppo_v0.yaml", "tests/unit/test_rl_training_and_compare.py"],
    "H-012": ["seedling_rl/offline_replay.py", "tests/unit/test_rl_training_and_compare.py"],
    "H-013": ["seedling_rl/sweep.py", "tests/unit/test_rl_training_and_compare.py"],
    "I-001": ["seedling_robot/adapters/base.py", "docs/ROBOT_PROTOCOL.md"],
    "I-002": ["seedling_robot/adapters/simulator.py", "tests/unit/test_robot_and_replay.py"],
    "I-003": ["seedling_robot/protocol.py", "tests/unit/test_robot_and_replay.py"],
    "I-004": [
        "seedling_robot/adapters/dry_run_serial.py",
        "tests/unit/test_dry_run_and_feedback.py",
        "tests/unit/test_robot_and_replay.py",
    ],
    "I-005": [
        "seedling_robot/protocol.py",
        "seedling_decision/safety_gate.py",
        "seedling_robot/adapters/dry_run_serial.py",
        "tests/unit/test_decision_and_safety.py",
        "tests/unit/test_dry_run_and_feedback.py",
    ],
    "I-006": ["docs/ROBOT_PROTOCOL.md"],
    "I-007": ["seedling_robot/motion_replay.py", "tests/unit/test_dry_run_and_feedback.py", "tests/unit/test_robot_and_replay.py"],
    "I-008": ["seedling_robot/dry_run_runner.py", "seedling_robot/cli.py", "tests/unit/test_robot_and_replay.py"],
    "I-009": ["seedling_robot/hil_runner.py", "seedling_robot/adapters/real_gantry_serial.py", "docs/ROBOT_PROTOCOL.md"],
    "J-001": ["seedling_ui/offline_viewer.py", "tests/unit/test_reports_and_viewers.py"],
    "J-002": ["seedling_ui/replay_viewer.py", "tests/unit/test_reports_and_viewers.py"],
    "J-003": ["seedling_ui/offline_viewer.py", "seedling_decision/policies/builtin.py", "tests/unit/test_reports_and_viewers.py"],
    "J-004": ["seedling_ui/cli.py", "seedling_core/registry.py", "tests/unit/test_registry_selector_feedback.py"],
    "J-005": ["seedling_reports/scenario.py"],
    "J-006": ["seedling_ui/report_export.py", "tests/unit/test_reports_and_viewers.py"],
    "J-007": ["seedling_ui/feedback.py", "seedling_ui/offline_viewer.py", "tests/unit/test_registry_selector_feedback.py"],
    "K-001": ["seedling_experiments/cli.py", "tests/unit/test_cli_smoke.py"],
    "K-002": [
        "seedling_experiments/cli.py",
        "seedling_experiments/config.py",
        "seedling_reports/registry.py",
        "tests/unit/test_cli_smoke.py",
        "tests/unit/test_reports_and_viewers.py",
    ],
    "K-003": ["seedling_reports/tables.py", "tests/unit/test_reports_and_viewers.py"],
    "K-004": ["seedling_reports/tables.py", "tests/unit/test_reports_and_viewers.py"],
    "K-005": ["seedling_reports/tables.py", "tests/unit/test_reports_and_viewers.py"],
    "K-006": ["seedling_reports/registry.py", "seedling_reports/tables.py", "tests/unit/test_reports_and_viewers.py"],
    "K-007": ["seedling_reports/html.py", "seedling_reports/cli.py", "tests/unit/test_reports_and_viewers.py"],
    "K-008": ["seedling_reports/compare.py"],
    "L-001": ["tests/unit/test_schemas.py"],
    "L-002": ["tests/unit/test_grid.py"],
    "L-003": ["tests/unit/test_target_generation.py"],
    "L-004": ["tests/unit/test_decision_and_safety.py"],
    "L-005": ["tests/unit/test_calibration.py"],
    "L-006": ["tests/integration/test_scene_policy_sim.py"],
    "L-007": ["tests/integration/test_scene_policy_sim.py"],
    "L-008": ["tests/unit/test_rl_training_and_compare.py"],
    "L-009": ["tests/unit/test_evaluate_cells_metrics.py"],
    "L-010": ["tests/unit/test_cli_smoke.py"],
    "L-011": ["tests/performance/test_performance_smoke.py"],
    "M-001": ["docs/README.md"],
    "M-002": ["docs/ARCHITECTURE.md"],
    "M-003": ["docs/RL_SPEC.md"],
    "M-004": ["docs/SIMULATION_SPEC.md"],
    "M-005": ["docs/MODEL_PLUGIN_API.md"],
    "M-006": ["docs/SAFETY_CONCEPT.md"],
    "M-007": ["docs/CALIBRATION_PROTOCOL.md"],
    "M-008": ["docs/FAILURE_MODES.md"],
    "M-009": ["docs/ARTICLE_REPORT_TEMPLATE.md"],
    "M-010": ["docs/OPERATOR_MANUAL_DRAFT.md"],
}

CONTENT_RULES: dict[str, dict[str, list[str]]] = {
    "A-001": {
        "pyproject.toml": [
            "[project.scripts]",
            "seedling-experiments",
            "seedling-data",
            "seedling-reports",
            "seedling-vision",
            "[tool.setuptools.packages.find]",
            "seedling_*",
        ],
        "seedling_experiments/__main__.py": [
            "from .cli import main",
            "if __name__ == \"__main__\"",
        ],
    },
    "A-002": {
        "requirements/base.txt": [
            "numpy",
            "Pillow",
            "PyYAML",
        ],
        "requirements/vision.txt": [
            "opencv-python",
            "onnxruntime",
            "pillow-heif",
        ],
        "requirements/rl.txt": [
            "gymnasium",
            "stable-baselines3",
            "sb3-contrib",
        ],
        "requirements/robot.txt": [
            "pyserial",
        ],
        "requirements/dev.txt": [
            "pre-commit",
            "pytest",
            "ruff",
        ],
        "tests/unit/test_cli_smoke.py": [
            "test_requirement_splits_match_pyproject_dependency_groups",
        ],
    },
    "A-003": {
        "seedling_core/config.py": [
            "class ConfigError",
            "class FieldSpec",
            "def load_config_file",
            "def require_sections",
            "def validate_fields",
            "def validate_existing_paths",
        ],
        "seedling_experiments/config.py": [
            "def validate_experiment_config",
            "FieldSpec.parse",
            "validate_existing_paths",
            "def _existing_path_fields",
        ],
        "tests/unit/test_cli_smoke.py": [
            "test_experiment_config_validation_checks_existing_input_paths",
        ],
    },
    "A-004": {
        "seedling_core/schemas.py": [
            "SCHEMA_VERSION",
            "class DetectionObject",
            "class CellState",
            "class ActionTarget",
            "class TrayState",
            "class RobotState",
            "class SafetyState",
            "class SceneState",
            "class DetectionResultV1",
            "class ActionCommand",
            "class ActionPlan",
            "def to_dict",
            "def from_dict",
        ],
        "tests/unit/test_schemas.py": [
            "test_detection_object_computes_center_and_area",
            "test_scene_state_round_trip",
            "test_action_command_round_trip",
        ],
    },
    "A-005": {
        "seedling_core/run_snapshot.py": [
            "config_hash",
            "command_args",
            "environment_snapshot",
        ],
    },
    "A-006": {
        "seedling_reports/registry.py": [
            "ArtifactRecord",
            "RunRecord",
            "role",
            "artifact_registry.csv",
            "write_run_registry_records",
        ],
        "seedling_core/run_snapshot.py": [
            "register_run_artifacts",
            "input_paths",
            "output_paths",
        ],
        "tests/unit/test_reports_and_viewers.py": [
            "test_write_run_registry_records_preserves_artifact_roles",
        ],
        "tests/unit/test_cli_smoke.py": [
            "test_core_register_run_artifacts_replaces_snapshot_record_with_outputs",
        ],
    },
    "A-007": {
        "seedling_vision/migration.py": [
            "def legacy_predictions_to_scenes",
            "def legacy_prediction_to_detection_result",
            "def legacy_prediction_to_scene",
            "def write_scene_states",
            "DetectionResultV1",
            "CellStateBuilder",
            "generate_largest_bbox_targets",
        ],
        "tests/unit/test_vision_migration.py": [
            "test_legacy_predictions_convert_to_scene_state_with_targets",
            "test_migrate_predictions_cli_writes_scene_json",
        ],
    },
    "A-008": {
        ".pre-commit-config.yaml": [
            "pre-commit-hooks",
            "ruff-pre-commit",
            "ruff-format",
        ],
        ".github/workflows/ci.yml": [
            "actions/setup-python",
            "pre-commit run --all-files",
            "python -B -m unittest discover -s tests",
        ],
        "pyproject.toml": [
            "pre-commit",
            "pytest",
            "ruff",
        ],
    },
    "B-001": {
        "configs/ontology/ontology_v0_1.yaml": [
            "unknown_plant",
            "remove_extra_crop",
            "human_review_required",
            "low_confidence",
        ],
    },
    "B-002": {
        "seedling_data/ontology.py": [
            "REQUIRED_OBJECT_CLASSES",
            "REQUIRED_CELL_STATES",
            "REQUIRED_ACTION_LABELS",
            "validate_dataset_class_names",
        ],
        "tests/unit/test_ontology_and_adapters.py": [
            "test_ontology_rejects_missing_required_v0_labels",
            "test_ontology_rejects_duplicate_aliases_and_dataset_class_names",
        ],
    },
    "B-003": {
        "docs/DATASET_CARD_TEMPLATE.md": [
            "Версия набора данных",
            "Версия онтологии",
            "Версия руководства по разметке",
            "Версия разбиения",
            "Версия калибровки",
            "Ограничения безопасности и использования",
            "artifact_registry.json",
        ],
    },
    "B-004": {
        "docs/ANNOTATION_GUIDE.md": [
            "Метки объектов",
            "Ограничивающие рамки",
            "Разметка ячеек",
            "Точки действия",
            "human_review_required: true",
            "forbidden_zone_ids",
            "near-duplicates",
        ],
    },
    "B-005": {
        "seedling_experiments/dataset.py": [
            "def make_grouped_split",
            "metadata_group_column",
            "read_group_map",
            "split_manifest.csv",
            "data.yaml",
            "seed",
            "_assign_groups",
        ],
        "tests/unit/test_dataset_split.py": [
            "test_grouped_split_uses_image_manifest_group_ids",
        ],
    },
    "B-006": {
        "seedling_data/manifests.py": [
            "MANIFEST_FIELDS",
            "sha256",
            "group_id",
            "session_id",
            "tray_id",
            "write_image_manifest",
        ],
        "tests/unit/test_data_registry_annotations.py": [
            "test_image_manifest_generator_fills_group_session_and_tray_ids",
            "test_image_manifest_cli_writes_artifact_registry",
        ],
    },
    "B-007": {
        "seedling_experiments/dataset.py": [
            "def validate_split_integrity",
            "is_suspected_augmented_name",
            "base_image_key",
            "eval_augmented_images",
            "cross_split_base_leakage",
            "DEFAULT_AUGMENTED_NAME_MARKERS",
        ],
        "tests/unit/test_dataset_split.py": [
            "test_grouped_split_uses_image_manifest_group_ids",
        ],
    },
    "B-008": {
        "seedling_data/duplicates.py": [
            "exact_sha256_duplicates",
            "near_perceptual_duplicates",
            "missing_split_rows",
            "missing_sha256_rows",
            "max_hamming",
        ],
        "tests/unit/test_data_registry_annotations.py": [
            "test_duplicate_check_finds_cross_split_identical_image",
            "test_duplicate_check_rejects_manifest_without_split_hash_or_path",
        ],
    },
    "B-009": {
        "seedling_data/annotations.py": [
            "class CellAnnotation",
            "def read_cell_annotations",
            "def write_cell_annotations",
            "def audit_cell_annotations",
            "state_counts",
            "duplicate cell annotation",
        ],
        "docs/ANNOTATION_GUIDE.md": [
            "cell_annotations.jsonl",
            "Допустимые значения `state`",
            "unknown",
            "image_quality_insufficient",
        ],
        "tests/unit/test_data_registry_annotations.py": [
            "test_cell_and_action_annotation_audits_accept_valid_rows",
            "test_action_annotation_audit_requires_review_for_crop_and_weed_cells",
        ],
    },
    "B-010": {
        "seedling_data/annotations.py": [
            "class ActionPointAnnotation",
            "def read_action_points",
            "def write_action_points",
            "def audit_action_points",
            "min_safe_distance_px",
            "max_uncertainty_px",
            "forbidden_zone_ids",
            "human_review_required",
        ],
        "docs/ANNOTATION_GUIDE.md": [
            "action_points.jsonl",
            "min_distance_to_keep_px",
            "uncertainty_radius_px",
            "forbidden_zone_ids",
        ],
        "tests/unit/test_data_registry_annotations.py": [
            "test_action_annotation_audit_requires_review_for_unsafe_targets",
            "test_action_annotation_audit_accepts_review_flags_for_unsafe_targets",
            "test_action_annotation_audit_cli_supports_safety_thresholds",
        ],
    },
    "B-011": {
        "seedling_data/annotations.py": [
            "cell_id is not present in cell annotations",
            "crop_and_weed action point requires human_review_required=true",
            "forbidden_zone_ids require human_review_required=true",
            "target_type human_review_required must set human_review_required=true",
        ],
        "tests/unit/test_data_registry_annotations.py": [
            "test_action_annotation_audit_requires_review_for_crop_and_weed_cells",
            "test_action_annotation_audit_requires_review_for_unsafe_targets",
        ],
    },
    "B-012": {
        "seedling_data/annotations.py": [
            "min_safe_distance_px",
            "max_uncertainty_px",
            "unsafe_distance_rows",
            "high_uncertainty_rows",
            "review_required_rows",
        ],
        "tests/unit/test_data_registry_annotations.py": [
            "test_action_annotation_audit_cli_supports_safety_thresholds",
        ],
    },
    "B-013": {
        "seedling_data/registry.py": [
            "class DatasetRegistryEntry",
            "class DatasetRegistry",
            "def add_dataset",
            "def validate_dataset_registry",
            "def write_dataset_summary",
            "collect_dataset_hashes",
            "cell_annotations",
            "action_points",
        ],
        "tests/unit/test_data_registry_annotations.py": [
            "test_dataset_registry_add_validate_and_summarize",
            "test_registry_cli_smoke",
        ],
    },
    "B-014": {
        "seedling_data/changelog.py": [
            "CHANGELOG_CANDIDATES",
            "REQUIRED_FIELDS",
            "REQUIRED_SECTIONS",
            "def audit_dataset_changelog",
            "Source dataset root",
            "Validation",
        ],
        "docs/DATASET_CHANGELOG_TEMPLATE.md": [
            "Шаблон журнала изменений набора данных",
            "Статус: черновик / заморожен / архивирован",
            "Корень исходного набора данных",
            "### Проверка",
        ],
        "tests/unit/test_data_registry_annotations.py": [
            "test_dataset_changelog_audit_and_cli",
            "test_dataset_changelog_accepts_source_dataset_root_path",
        ],
    },
    "C-001": {
        "seedling_vision/adapters/base.py": [
            "class DetectorAdapter(ABC)",
            "def load",
            "def predict",
            "def metadata",
            "DetectionResultV1",
            "ModelMetadata",
            "metadata_for_context",
            "class MockDetector",
        ],
    },
    "C-002": {
        "seedling_vision/adapters/ultralytics_yolo.py": [
            "class UltralyticsYOLODetector",
            "from ultralytics import YOLO",
            "DetectionObject(",
            "DetectionResultV1(",
            "model_metadata=metadata_for_context",
            "backend=\"ultralytics_yolo\"",
            "def _load_rgb",
        ],
        "tests/unit/test_vision_tools.py": [
            "test_ultralytics_adapter_maps_mock_result_to_detection_objects",
        ],
    },
    "C-003": {
        "seedling_vision/adapters/baseline_green.py": [
            "class BaselineGreenDetector",
            "cv2.cvtColor",
            "cv2.inRange",
            "cv2.findContours",
            "DetectionObject(",
            "attributes=[\"baseline_green\"]",
            "output_schema=\"DetectionResultV1\"",
        ],
        "tests/unit/test_registry_selector_feedback.py": [
            "test_baseline_green_detector_finds_green_region",
        ],
    },
    "C-004": {
        "seedling_vision/adapters/recorded_prediction.py": [
            "class RecordedPredictionDetector",
            "Recorded predictions must contain an `images` list",
            "_record_to_detection",
            "extra={",
            "\"containers\": prediction.get",
            "metadata_for_context",
        ],
        "tests/unit/test_ontology_and_adapters.py": [
            "test_recorded_prediction_detector_reads_existing_contract",
        ],
    },
    "C-005": {
        "seedling_vision/adapters/onnx.py": [
            "class ONNXDetector",
            "import onnxruntime as ort",
            "parse_onnx_detections",
            "def _preprocess",
            "def _providers",
            "output_format",
            "_uncertainty_from_row",
            "DetectionObject(",
        ],
        "tests/unit/test_vision_tools.py": [
            "test_onnx_parser_maps_xyxy_rows_to_detection_objects",
            "test_onnx_parser_supports_channel_first_class_scores",
            "test_onnx_parser_supports_objectness_class_scores",
        ],
    },
    "C-006": {
        "seedling_vision/uncertainty.py": [
            "class UncertaintyConfig",
            "low_confidence_threshold",
            "tiny_area_px2",
            "high_entropy_threshold",
            "annotate_detection_uncertainty",
            "class_entropy_normalized",
            "class_probability_margin",
            "touches_image_edge",
        ],
        "tests/unit/test_vision_tools.py": [
            "test_uncertainty_flags_low_confidence_tiny_and_edge",
            "test_uncertainty_computes_entropy_from_class_scores",
        ],
    },
    "C-007": {
        "seedling_vision/postprocess.py": [
            "filter_and_merge_containers",
            "merge_container_detections",
            "match_containers_by_iou",
            "container_matching_summary",
            "best_iou_counts",
            "unmatched_samples",
            "bbox_iou",
        ],
        "tests/unit/test_vision_tools.py": [
            "test_container_postprocess_merges_small_nearby_boxes",
            "test_postprocess_containers_cli_writes_postprocessed_json",
        ],
    },
    "C-008": {
        "seedling_core/registry.py": [
            "class ModelRegistryRecord",
            "input_schema",
            "output_schema",
            "calibration_requirements",
            "safety_level",
            "class ModelRegistry",
            "requires_action_mask",
            "direct_hardware_access",
        ],
        "configs/registry/model_registry_v0_1.yaml": [
            "model_registry_v0_1",
            "baseline_green_detector_v0",
            "DetectionResultV1",
            "dataset_version",
            "ontology_version",
            "calibration_requirements",
            "safety_level",
        ],
        "tests/unit/test_registry_selector_feedback.py": [
            "test_typed_model_registry_validates_required_metadata",
            "test_typed_model_registry_blocks_production_candidate_without_review",
            "test_typed_model_registry_blocks_rl_direct_hardware_access",
        ],
    },
    "C-009": {
        "seedling_vision/batch.py": [
            "context_factory",
            "progress",
            "enumerate(images, 1)",
        ],
        "seedling_vision/cli.py": [
            "batch-recorded",
            "progress-log",
            "predict_start",
            "batch_complete",
        ],
        "tests/unit/test_vision_tools.py": [
            "test_batch_predict_uses_context_factory_and_progress_indexes",
            "test_vision_batch_and_overlay_cli",
        ],
    },
    "C-010": {
        "seedling_vision/overlays.py": [
            "def write_detection_overlay",
            "draw.rectangle",
            "draw.line",
            "_draw_target_marker",
            "_draw_issue",
            "_draw_issue_legend",
            "ERROR_COLOR",
            "TARGET_COLOR",
        ],
        "tests/unit/test_vision_tools.py": [
            "test_vision_batch_and_overlay_cli",
            "test_vision_overlay_cli_draws_grid_targets_and_errors",
            "test_overlay_resolves_error_taxonomy_target_aliases",
        ],
    },
    "D-001": {
        "seedling_cells/grid.py": [
            "class GridCell",
            "def generate_grid",
            "def generate_grid_cells",
            "def generate_grid_polygons",
            "def cell_index",
            "def cell_index_for_grid_cells",
            "def point_in_polygon",
            "def _validate_grid",
        ],
        "tests/unit/test_grid.py": [
            "test_cell_index_uses_last_cell_for_max_boundary",
            "test_generate_grid_polygons_supports_tray_corners",
            "test_corner_grid_assignment_uses_cell_polygons_not_outer_bbox",
            "test_generate_grid_rejects_invalid_bbox",
        ],
    },
    "D-002": {
        "seedling_cells/cell_state.py": [
            "class CellStateBuilder",
            "crop_class_names",
            "weed_class_names",
            "unknown_class_names",
            "review_attribute_names",
            "generate_grid_polygons",
            "generate_grid_cells",
            "object_ids",
            "human_review_required",
            "risk_flags",
        ],
        "tests/unit/test_target_generation.py": [
            "test_unknown_cell_requires_review_and_no_auto_target",
            "test_high_entropy_single_crop_marks_cell_for_review",
            "test_cell_state_builder_maps_detections_to_full_tray_grid",
            "test_cell_builder_with_tray_corners_ignores_points_outside_polygon",
        ],
    },
    "D-003": {
        "seedling_cells/cell_state.py": [
            "class CellStateBuilder",
            "def build",
            "generate_grid_cells",
            "generate_grid_polygons",
            "cell_index_for_grid_cells",
            "CellState(",
            "object_ids",
            "human_review_required",
        ],
        "tests/unit/test_target_generation.py": [
            "test_cell_state_builder_maps_detections_to_full_tray_grid",
            "test_cell_builder_with_tray_corners_ignores_points_outside_polygon",
        ],
    },
    "D-004": {
        "seedling_cells/target_generation.py": [
            "class LargestBBoxTargetStrategy",
            "decision_source",
            "crop_class_names",
            "weed_class_names",
            "def generate",
            "_infer_removal_candidates",
            "keep_object_id",
            "remove_extra_crop",
            "remove_weed",
            "ActionTarget(",
        ],
        "tests/unit/test_target_generation.py": [
            "test_largest_bbox_strategy_generates_extra_crop_targets",
            "test_largest_bbox_strategy_is_configurable_and_respects_keep_object",
            "test_weed_only_cell_generates_remove_weed_target",
            "test_crop_and_weed_cell_generates_reviewed_remove_weed_target",
        ],
    },
    "D-005": {
        "seedling_experiments/evaluate.py": [
            "def evaluate_cells_from_config",
            "def _update_confusion",
            "def _macro_metrics",
            "def _prf",
            "cell_accuracy",
            "multi_seedling_cell",
            "removal_targets",
            "cell_confusion_matrix.csv",
        ],
        "tests/unit/test_evaluate_cells_metrics.py": [
            "test_evaluate_cells_reports_calibrated_target_error_mm",
            "test_evaluate_cells_reports_legacy_cost_sensitive_errors",
        ],
    },
    "D-006": {
        "seedling_experiments/evaluate.py": [
            "match_containers",
            "container_iou",
            "best_container_ious",
            "unmatched_samples",
            "container_matching",
            "prediction_source",
            "prediction_coverage",
        ],
        "seedling_vision/postprocess.py": [
            "def container_matching_summary",
            "best_iou_counts",
            "unmatched_samples",
        ],
        "tests/unit/test_evaluate_cells_metrics.py": [
            "test_evaluate_cells_reports_legacy_cost_sensitive_errors",
        ],
    },
    "D-007": {
        "seedling_experiments/evaluate.py": [
            "cost_sensitive_metrics_from_legacy",
            "critical_error_counts",
            "critical_error_total",
            "critical_error_rate_per_cell",
            "normalized_cost_per_cell",
            "cost_sensitive",
        ],
        "tests/unit/test_evaluate_cells_metrics.py": [
            "test_evaluate_cells_reports_legacy_cost_sensitive_errors",
            "test_evaluate_cells_preserves_manifest_group_metadata_for_bootstrap",
        ],
    },
    "D-008": {
        "seedling_cells/evaluation.py": [
            "def evaluate_scene_states",
            "def evaluate_cell_states",
            "def evaluate_action_targets",
            "DEFAULT_STATE_ORDER",
            "weed_only",
            "crop_and_weed",
            "unknown",
            "ambiguous",
        ],
        "tests/unit/test_scene_state_evaluation.py": [
            "test_scene_state_evaluator_handles_richer_states_targets_and_costs",
        ],
    },
    "D-009": {
        "seedling_cells/evaluation.py": [
            "def _region_cell_metrics",
            "edge_cell_metrics",
            "corner_cell_metrics",
            "target_match_distance_mm",
            "expert_keep_remove",
            "mean_error_mm",
        ],
        "tests/unit/test_scene_state_evaluation.py": [
            "test_scene_state_evaluator_reports_edge_and_corner_cell_metrics",
            "test_evaluate_scenes_cli_writes_metrics",
        ],
    },
    "D-010": {
        "seedling_cells/evaluation.py": [
            "def cost_sensitive_metrics",
            "target_false_negative",
            "target_false_positive",
            "unknown_not_reviewed",
            "expert_false_removal",
            "expert_missed_removal",
            "cost_breakdown",
        ],
        "seedling_reports/diagnostics.py": [
            "def error_taxonomy",
            "ERROR_CATEGORY_GROUPS",
            "image",
            "geometry",
            "biology",
            "decision",
        ],
        "tests/unit/test_reports_and_viewers.py": [
            "test_error_taxonomy_counts_target_and_cell_errors",
            "test_error_taxonomy_uses_scene_state_metrics_without_legacy_duplicates",
        ],
    },
    "D-011": {
        "seedling_reports/diagnostics.py": [
            "def build_cell_metrics_diagnostics",
            "def bootstrap_cell_metrics",
            "group_key",
            "critical_error_rate_per_cell",
            "normalized_cost_per_cell",
            "bootstrap_ci.json",
            "error_taxonomy.json",
        ],
        "tests/unit/test_evaluate_cells_metrics.py": [
            "test_evaluate_cells_preserves_manifest_group_metadata_for_bootstrap",
        ],
        "tests/unit/test_reports_and_viewers.py": [
            "test_cell_metrics_diagnostics_build_bootstrap_and_taxonomy",
            "test_diagnose_cell_metrics_cli",
        ],
    },
    "D-012": {
        "seedling_reports/diagnostics.py": [
            "def error_taxonomy",
            "category_groups",
            "container_unmatched",
            "target_coordinate_error",
            "cell_state_mismatch",
            "_add_scene_state_taxonomy",
        ],
        "tests/unit/test_reports_and_viewers.py": [
            "test_error_taxonomy_counts_target_and_cell_errors",
            "test_error_taxonomy_uses_scene_state_metrics_without_legacy_duplicates",
            "test_diagnose_cell_metrics_cli",
        ],
    },
    "E-001": {
        "seedling_calibration/schemas.py": [
            "CALIBRATION_SCHEMA_VERSION",
            "class ErrorSummary",
            "class CalibrationConfig",
            "class CalibrationArtifact",
            "class CalibrationValidationResult",
            "image_to_tray_homography",
            "tray_to_robot_transform",
            "tool_offset_mm",
            "error_summary_mm",
            "def to_json",
        ],
    },
    "E-002": {
        "seedling_calibration/transforms.py": [
            "def apply_homography",
            "homography denominator is too close to zero",
            "def image_px_to_tray_mm",
            "artifact.image_to_tray_homography",
        ],
        "tests/unit/test_calibration.py": [
            "test_image_and_robot_transforms",
            "test_estimate_calibration_artifact_from_four_points",
        ],
    },
    "E-003": {
        "seedling_calibration/transforms.py": [
            "def tray_mm_to_robot_frame_mm",
            "include_tool_offset",
            "artifact.tray_to_robot_transform",
            "artifact.tool_offset_mm",
        ],
        "tests/unit/test_calibration.py": [
            "test_image_and_robot_transforms",
        ],
    },
    "E-004": {
        "seedling_calibration/validator.py": [
            "class CalibrationValidator",
            "expected_tray_type",
            "max_error_p95_mm",
            "max_error_rms_mm",
            "valid_until",
            "calibration expired",
            "error_summary_mm.p95",
            "error_summary_mm.rms",
            "px_per_mm_x",
        ],
        "tests/unit/test_calibration.py": [
            "test_validator_rejects_expired_or_high_error_calibration",
            "test_validator_rejects_wrong_tray_type_and_rms_error",
        ],
    },
    "E-005": {
        "seedling_calibration/estimation.py": [
            "def write_error_map",
            "calibration_residuals",
            "error_summary_mm",
            "_error_map_html",
            "_error_map_grid",
            "expected_tray_mm",
            "predicted_tray_mm",
        ],
        "docs/CALIBRATION_PROTOCOL.md": [
            "error-map",
            "сводку ошибок",
            "карты ошибок",
        ],
        "tests/unit/test_calibration.py": [
            "test_calibration_estimate_and_error_map_cli",
        ],
    },
    "E-006": {
        "seedling_calibration/transforms.py": [
            "def attach_calibration_to_target",
            "action_point_mm or image_px_to_tray_mm",
            "robot_point = tray_mm_to_robot_frame_mm",
            "uncertainty = artifact.error_summary_mm.p95",
            "def attach_calibration_to_scene",
            "calibration_id=artifact.calibration_id",
            "targets=attach_calibration_to_targets",
        ],
        "tests/unit/test_calibration.py": [
            "test_attach_calibration_to_target",
            "test_attach_calibration_to_scene_updates_targets_and_tray_id",
        ],
    },
    "E-007": {
        "seedling_calibration/cli.py": [
            "subparsers.add_parser(\"validate\"",
            "subparsers.add_parser(\"estimate\"",
            "subparsers.add_parser(\"error-map\"",
            "_write_calibration_snapshot",
            "_write_calibration_registry",
            "seedling-calibration:validate",
            "seedling-calibration:estimate",
            "seedling-calibration:error-map",
        ],
        "tests/unit/test_calibration.py": [
            "test_calibration_estimate_and_error_map_cli",
            "test_calibration_validate_cli_writes_report_and_registry",
        ],
    },
    "E-008": {
        "seedling_calibration/intrinsics.py": [
            "INTRINSICS_SCHEMA_VERSION",
            "SUPPORTED_TARGET_TYPES = {\"chessboard\", \"aruco\", \"fiducials\"}",
            "class IntrinsicsObservation",
            "class CameraIntrinsicsArtifact",
            "def estimate_camera_intrinsics",
            "cv2.calibrateCamera",
            "def validate_camera_intrinsics",
            "reprojection_error_px",
        ],
        "tests/unit/test_calibration.py": [
            "test_camera_intrinsics_artifact_validates",
            "test_estimate_camera_intrinsics_uses_loaded_observations_and_cv2_contract",
            "test_intrinsics_observations_example_loads",
            "test_validate_intrinsics_cli",
        ],
    },
    "F-001": {
        "seedling_decision/policies/base.py": [
            "class DecisionPolicy(ABC)",
            "policy_id: str",
            "processed_target_ids",
            "def reset",
            "@abstractmethod",
            "def propose_plan",
            "def next_action",
            "def _command_for_target",
            "ActionCommand(",
            "requires_operator_confirmation=target.human_review_required",
            "safety_gate_result=\"not_evaluated\"",
            "coordinate_source",
        ],
    },
    "F-002": {
        "seedling_decision/policies/builtin.py": [
            "class NoOpPolicy",
            "policy_id: str = \"noop\"",
            "review_ids = [target.target_id for target in scene.targets]",
            "commands=[]",
            "noop_policy_requires_review",
        ],
        "tests/unit/test_decision_and_safety.py": [
            "test_noop_policy_sends_all_targets_to_review",
        ],
    },
    "F-003": {
        "seedling_decision/policies/builtin.py": [
            "class RasterScanPolicy",
            "policy_id: str = \"raster_scan\"",
            "ActionMaskBuilder",
            "def _ordered_targets",
            "cell_order = {cell.cell_id: (cell.row, cell.col)",
            "_build_plan(self, scene, self._ordered_targets",
        ],
        "tests/unit/test_decision_and_safety.py": [
            "test_raster_policy_commands_only_valid_targets",
        ],
    },
    "F-004": {
        "seedling_decision/policies/builtin.py": [
            "class NearestNeighborPolicy",
            "policy_id: str = \"nearest_neighbor\"",
            "_target_distance_from_robot",
            "scene.robot.position_mm",
            "_build_plan(self, scene, ordered",
        ],
        "tests/unit/test_decision_and_safety.py": [
            "test_nearest_neighbor_policy_orders_safe_targets_by_robot_distance",
        ],
    },
    "F-005": {
        "seedling_decision/policies/builtin.py": [
            "class RiskAwareRulePolicy",
            "max_target_uncertainty_mm=2.0",
            "min_distance_to_keep_mm=5.0",
            "max_risk_score=0.25",
            "require_calibration_valid=True",
            "require_robot_homed=True",
            "target.risk_score",
            "_target_distance_from_robot",
        ],
        "tests/unit/test_decision_and_safety.py": [
            "test_risk_aware_policy_requires_homed_robot",
            "test_risk_aware_policy_blocks_forbidden_zone_and_outside_tray",
        ],
    },
    "F-007": {
        "seedling_decision/safety_gate.py": [
            "class SafetyGate",
            "def validate",
            "SafetyDecision",
            "BLOCK_CALIBRATION_REQUIRED",
            "BLOCK_INTERLOCK",
            "BLOCK_UNSAFE_TARGET",
            "BLOCK_REVIEW_REQUIRED",
            "_decision(",
        ],
        "seedling_robot/adapters/dry_run_serial.py": [
            "SafetyGate",
            "decision.allowed",
            "attach_safety_decision",
            "blocked by SafetyGate",
        ],
        "tests/unit/test_decision_and_safety.py": [
            "test_safety_gate_allows_safe_simulation_command",
            "test_safety_gate_blocks_bad_calibration_and_close_target",
            "test_safety_gate_blocks_missing_target_before_execution",
            "test_safety_gate_blocks_robot_telemetry_faults",
            "test_safety_gate_blocks_real_action_when_software_safe_mode_disabled",
            "test_safety_gate_blocks_real_action_when_enclosure_is_open",
        ],
        "tests/unit/test_dry_run_and_feedback.py": [
            "test_dry_run_adapter_serializes_command_after_safety_gate",
            "test_dry_run_adapter_blocks_when_interlock_false",
            "test_dry_run_adapter_blocks_unsupported_tool_profile_before_motion",
        ],
    },
    "F-006": {
        "seedling_decision/action_masks.py": [
            "class TargetMask",
            "class ActionMask",
            "def valid_target_ids",
            "def reasons_by_target",
            "class ActionMaskBuilder",
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
        ],
        "tests/unit/test_decision_and_safety.py": [
            "test_action_mask_reports_reasons",
            "test_action_mask_blocks_processed_and_unsafe_targets",
            "test_raster_policy_commands_only_valid_targets",
            "test_risk_aware_policy_requires_homed_robot",
        ],
    },
    "F-008": {
        "seedling_decision/action_masks.py": [
            "forbidden_zone_overlap",
            "target_outside_tray",
            "def _outside_tray",
            "target.forbidden_zone_ids",
            "return TargetMask",
        ],
        "seedling_decision/policies/builtin.py": [
            "class RiskAwareRulePolicy",
            "_build_plan",
            "blocked_target_ids",
            "blocked_reasons_by_target",
            "ActionMaskBuilder",
        ],
        "seedling_decision/safety_gate.py": [
            "forbidden_zone_overlap",
            "target_outside_tray",
            "BLOCK_UNSAFE_TARGET",
            "target.forbidden_zone_ids",
            "def _outside_tray",
        ],
        "tests/unit/test_decision_and_safety.py": [
            "test_action_mask_reports_reasons",
            "test_human_review_policy_surfaces_review_and_blocked_targets",
            "test_risk_aware_policy_blocks_forbidden_zone_and_outside_tray",
            "test_safety_gate_blocks_bad_calibration_and_close_target",
        ],
    },
    "F-009": {
        "seedling_decision/policies/builtin.py": [
            "class HumanReviewPolicy",
            "policy_id: str = \"human_review\"",
            "review_reasons_by_target",
            "human_review_required",
            "reasons.get(target.target_id)",
            "commands=[]",
        ],
        "tests/unit/test_decision_and_safety.py": [
            "test_human_review_policy_surfaces_review_and_blocked_targets",
        ],
    },
    "F-010": {
        "seedling_reports/scenario.py": [
            "def compare_policies_on_scene",
            "policy_selector",
            "route_distance_mm",
            "review_targets",
            "blocked_targets",
            "review_reasons",
            "blocked_reasons",
            "rl_checkpoint",
            "scenario_compare.json",
            "scenario_compare.html",
        ],
        "tests/unit/test_reports_and_viewers.py": [
            "test_scenario_compare_reports_policy_plan_differences",
            "test_scenario_compare_cli_writes_json_and_html",
            "test_scenario_compare_supports_rl_checkpoint_selector",
        ],
    },
    "F-011": {
        "seedling_decision/policies/builtin.py": [
            "class RoutePlanningPolicy",
            "policy_id: str = \"route_planning\"",
            "def _greedy_route",
            "route_planning_method",
            "greedy_nearest_neighbor",
            "leg_distance_mm",
            "cumulative_route_distance_mm",
            "current = _target_point_mm",
        ],
        "tests/unit/test_decision_and_safety.py": [
            "test_route_planning_policy_uses_cumulative_greedy_route",
        ],
    },
    "G-001": {
        "seedling_sim/schemas.py": [
            "SIM_SCENE_SCHEMA_VERSION",
            "class SimPlant",
            "class SimTarget",
            "class SimScene",
            "class SimStepOutcome",
            "def from_json",
            "def to_json",
            "plant_by_id",
            "target_by_id",
            "_validate_bounds",
        ],
        "tests/unit/test_simulation.py": [
            "test_sim_scene_round_trip",
        ],
    },
    "G-002": {
        "seedling_sim/logical_tray.py": [
            "class LogicalTraySimulator",
            "def reset",
            "def step_target",
            "ActuatorErrorModel",
            "PlantResponseModel",
            "crop_damage",
            "move_mm_penalty",
            "SimStepOutcome(",
        ],
        "tests/unit/test_simulation.py": [
            "test_logical_tray_step_removes_target_on_hit",
            "test_logical_tray_can_use_plant_response_model",
        ],
    },
    "G-003": {
        "seedling_sim/detection_noise.py": [
            "class DetectionNoiseModel",
            "bbox_center_sigma_mm",
            "classification_error_prob",
            "missed_detection_prob",
            "false_positive_prob",
            "classification_noise",
            "false_positive_noise",
            "class SimSceneDetectionNoiseModel",
        ],
        "tests/unit/test_simulation.py": [
            "test_detection_noise_can_drop_all_detections",
            "test_detection_noise_can_add_false_positive_detection",
            "test_sim_scene_detection_noise_can_miss_and_add_false_positive",
        ],
    },
    "G-004": {
        "seedling_sim/actuator_model.py": [
            "class ActuatorSample",
            "class ActuatorErrorModel",
            "xy_sigma_mm",
            "drift_sigma_mm",
            "latency_ms_mean",
            "zone_error_prob",
            "zone_error_mm",
            "zone_error_active",
        ],
        "tests/unit/test_simulation.py": [
            "test_actuator_zone_error_is_reported_in_outcome_info",
            "test_actuator_zone_error_validates_probability",
        ],
    },
    "G-005": {
        "seedling_sim/plant_response_model.py": [
            "class PlantResponse",
            "class PlantResponseModel",
            "unknown_requires_review",
            "near_keep_crop",
            "missed_target",
            "biological_nonresponse",
            "avoids laser power/dose modeling",
        ],
        "tests/unit/test_simulation.py": [
            "test_logical_tray_can_use_plant_response_model",
            "test_plant_response_model_requires_review_for_unknown_plants",
        ],
    },
    "G-006": {
        "seedling_sim/scene_generator.py": [
            "class SceneGeneratorConfig",
            "class SimSceneGenerator",
            "p_empty",
            "p_single_crop",
            "p_multiple_crop",
            "p_weed_present",
            "p_unknown_present",
            "generate_many",
            "human_review_required",
        ],
        "tests/unit/test_simulation.py": [
            "test_scene_generator_can_emit_empty_single_multiple_weed_and_unknown_cells",
        ],
    },
    "G-007": {
        "seedling_sim/image_backed.py": [
            "def sim_scene_from_scene_state",
            "def sim_scene_from_scene_state_file",
            "image_backed_scene_v0",
            "_plant_from_detection",
            "_target_from_action_target",
            "_point_mm",
            "_load_scene_state",
        ],
        "tests/unit/test_simulation.py": [
            "test_image_backed_scene_from_scene_state",
            "test_sim_cli_image_backed_and_synthetic_render",
        ],
    },
    "G-008": {
        "seedling_sim/replay.py": [
            "class ReplayStep",
            "class ReplayLog",
            "class ReplayLogger",
            "def append",
            "def append_execution",
            "def to_json",
            "def from_json",
        ],
        "tests/unit/test_robot_and_replay.py": [
            "test_replay_log_json_round_trip",
            "test_simulator_robot_executes_command_through_safety_gate",
        ],
    },
    "G-009": {
        "seedling_sim/renderers.py": [
            "def render_scene_svg",
            "def render_scene_html",
            "render_scene_svg(scene)",
            "Replay",
            "def render_scene_png",
            "ImageDraw.Draw",
        ],
        "tests/unit/test_rl_env.py": [
            "test_static_renderer_outputs_svg_and_html",
        ],
        "tests/unit/test_simulation.py": [
            "test_render_scene_png_writes_synthetic_image",
        ],
    },
    "G-010": {
        "seedling_ui/replay_viewer.py": [
            "def write_replay_viewer",
            "def render_replay_viewer_html",
            "_timeline_controls",
            "data-action='play'",
            "data-action='pause'",
            "_safety_summary_table",
            "_timeline_table",
        ],
        "tests/unit/test_reports_and_viewers.py": [
            "test_replay_viewer_writes_safety_reasons",
        ],
    },
    "G-011": {
        "seedling_sim/renderers.py": [
            "def render_scene_png",
            "DomainRandomizationConfig",
            "plant_jitter_px",
            "ImageEnhance.Brightness",
            "ImageFilter.GaussianBlur",
            "np.random.default_rng",
        ],
        "tests/unit/test_simulation.py": [
            "test_render_scene_png_writes_synthetic_image",
            "test_sim_cli_image_backed_and_synthetic_render",
        ],
    },
    "G-012": {
        "seedling_sim/domain_randomization.py": [
            "class DomainRandomizationConfig",
            "DOMAIN_RANDOMIZATION_PRESETS",
            "greenhouse_default",
            "low_light_noisy",
            "wet_substrate",
            "calibration_drift_mm",
            "domain_randomization_preset",
        ],
        "tests/unit/test_simulation.py": [
            "test_render_scene_png_applies_calibration_drift",
            "test_sim_cli_image_backed_and_synthetic_render",
        ],
    },
    "H-001": {
        "seedling_rl/envs.py": [
            "class SeedlingTrayEnv",
            "def reset",
            "def step",
            "def render",
            "def action_masks",
            "observation_space",
            "action_space",
        ],
        "tests/unit/test_rl_env.py": [
            "test_seedling_tray_env_reset_step_and_action_mask",
            "test_seedling_tray_env_passes_gymnasium_checker",
        ],
    },
    "H-002": {
        "seedling_rl/envs.py": [
            "cell_tensor",
            "target_features",
            "target_mask",
            "action_mask",
            "review_action",
            "stop_action",
            "_safe_to_act",
        ],
        "tests/unit/test_rl_env.py": [
            "test_seedling_tray_env_reset_step_and_action_mask",
            "test_tray_env_config_applies_detection_noise_block",
        ],
    },
    "H-003": {
        "seedling_rl/reward.py": [
            "class RewardConfig",
            "correct_remove",
            "correct_review",
            "episode_complete",
            "move_mm_penalty",
            "repeated_action",
            "unsafe_action",
            "crop_damage",
            "stop_with_remaining_target",
        ],
        "seedling_rl/envs.py": [
            "config.reward.correct_remove",
            "config.reward.correct_review",
            "config.reward.unsafe_action",
            "config.reward.episode_complete",
        ],
    },
    "H-004": {
        "seedling_rl/baseline_eval.py": [
            "class BaselineEvaluationResult",
            "def evaluate_baseline_policy",
            "def evaluate_baseline_suite",
            "def select_baseline_action",
            "noop",
            "raster_scan",
            "nearest_neighbor",
            "route_planning",
        ],
        "tests/unit/test_rl_env.py": [
            "test_baseline_suite_returns_policy_metrics",
        ],
    },
    "H-005": {
        "seedling_rl/training.py": [
            "class RLTrainingConfig",
            "def train_from_config",
            "dry_run",
            "RLMetricsLogger",
            "seedling-rl:train",
            "_algorithm_class",
            "MaskablePPO",
            "RecurrentPPO",
        ],
        "tests/unit/test_rl_training_and_compare.py": [
            "test_rl_train_dry_run_writes_summary_and_registry",
            "test_rl_train_uses_vectorized_env_when_n_envs_gt_one",
        ],
    },
    "H-006": {
        "seedling_rl/training.py": [
            "def evaluate_checkpoint",
            "def evaluate_model",
            "def evaluate_baseline_with_replays",
            "critical_events",
            "replay_paths",
            "successful_target_rate",
            "mean_distance_error_mm",
            "seedling-rl:evaluate",
        ],
        "tests/unit/test_rl_training_and_compare.py": [
            "test_rl_evaluate_baseline_writes_metrics",
            "test_evaluate_model_preserves_recurrent_state",
        ],
    },
    "H-007": {
        "seedling_rl/policy_adapter.py": [
            "class RLPolicyAdapter",
            "DecisionPolicy",
            "ActionMaskBuilder",
            "def from_sb3",
            "def propose_plan",
            "rl_action_kind",
            "blocked_target",
            "target_command",
        ],
        "tests/unit/test_rl_training_and_compare.py": [
            "test_rl_policy_adapter_reports_blocked_action_mask_reasons",
            "test_rl_policy_adapter_preserves_recurrent_state_until_reset",
        ],
    },
    "H-008": {
        "seedling_rl/callbacks.py": [
            "RL_METRIC_COLUMNS",
            "class RLMetricsLogger",
            "def log_step",
            "critical_error",
            "review_rate",
            "total_distance_mm",
            "mean_distance_error_mm",
            "def make_sb3_metric_callback",
        ],
        "tests/unit/test_rl_training_and_compare.py": [
            "test_rl_train_dry_run_writes_summary_and_registry",
            "test_rl_evaluate_baseline_writes_metrics",
        ],
    },
    "H-009": {
        "seedling_rl/vectorized_env.py": [
            "class VectorizedEnvSpec",
            "def vectorized_env_spec",
            "def make_vectorized_env",
            "DummyVecEnv",
            "def _env_factory",
            "config.seed + index",
        ],
        "tests/unit/test_rl_training_and_compare.py": [
            "test_vectorized_env_spec_and_curriculum_plan",
            "test_rl_train_uses_vectorized_env_when_n_envs_gt_one",
        ],
    },
    "H-010": {
        "seedling_rl/curriculum.py": [
            "class CurriculumStage",
            "def default_curriculum",
            "easy_single_crop",
            "mixed_multiple",
            "uncertain_review",
            "def write_curriculum_plan",
        ],
        "tests/unit/test_rl_training_and_compare.py": [
            "test_vectorized_env_spec_and_curriculum_plan",
            "test_rl_extra_cli_commands",
        ],
    },
    "H-011": {
        "seedling_rl/policy_adapter.py": [
            "recurrent",
            "_recurrent_state",
            "_episode_start",
            "episode_start=np.asarray",
            "def _is_recurrent_model",
        ],
        "configs/rl/recurrent_ppo_v0.yaml": [
            "algorithm: recurrent_ppo",
            "recurrent: true",
        ],
        "tests/unit/test_rl_training_and_compare.py": [
            "test_recurrent_ppo_train_dry_run_records_recurrent_flag",
            "test_recurrent_ppo_example_config_loads",
            "test_training_config_parses_recurrent_boolean_strings",
            "test_rl_policy_adapter_loader_accepts_recurrent_ppo",
        ],
    },
    "H-012": {
        "seedling_rl/offline_replay.py": [
            "def evaluate_replay_logs",
            "CRITICAL_EVENTS",
            "robot_execution_events",
            "rl_step_events",
            "block_reason_counts",
            "critical_events_count",
            "mean_distance_error_mm",
        ],
        "tests/unit/test_rl_training_and_compare.py": [
            "test_offline_replay_evaluation_counts_blocked_events",
            "test_offline_replay_evaluation_counts_adapter_failures_as_blocked",
            "test_offline_replay_evaluation_summarizes_rl_step_success_and_distance",
        ],
    },
    "H-013": {
        "seedling_rl/sweep.py": [
            "def build_sweep_plan",
            "schema_version",
            "rl_sweep_plan_v0_1",
            "def build_sweep_stability_report",
            "STABILITY_METRICS",
            "seed_count",
            "def write_sweep_stability_report",
        ],
        "tests/unit/test_rl_training_and_compare.py": [
            "test_sweep_plan_expands_grid_and_seeds",
            "test_sweep_stability_report_groups_runs_across_seeds",
            "test_rl_extra_cli_commands",
        ],
    },
    "I-001": {
        "seedling_robot/adapters/base.py": [
            "class RobotAdapter(ABC)",
            "def connect",
            "def home",
            "def move_to",
            "def mark_or_act",
            "def emergency_stop",
            "def telemetry",
            "RobotTelemetry",
        ],
        "docs/ROBOT_PROTOCOL.md": [
            "RobotAdapter",
            "SafetyGate",
        ],
    },
    "I-002": {
        "seedling_robot/adapters/simulator.py": [
            "class SimulatorRobotAdapter",
            "LogicalTraySimulator",
            "def execute_command",
            "SafetyGate",
            "attach_safety_decision",
            "self.simulator.step_target",
            "replay_logger.append_execution",
            "mode=\"simulation\"",
        ],
        "tests/unit/test_robot_and_replay.py": [
            "test_simulator_robot_executes_command_through_safety_gate",
            "test_simulator_robot_logs_blocked_command",
        ],
    },
    "I-003": {
        "seedling_robot/protocol.py": [
            "class SpeedProfile",
            "class ToolProfile",
            "class MotionResult",
            "class MotionCommand",
            "safety_token",
            "from_action_command",
            "class RobotExecutionResult",
            "attach_safety_decision",
        ],
        "tests/unit/test_robot_and_replay.py": [
            "test_motion_replay_cli_writes_simulation_report_and_registry",
        ],
    },
    "I-005": {
        "seedling_decision/safety_gate.py": [
            "class RobotTelemetry",
            "homed",
            "limit_switch_ok",
            "emergency_stop_active",
            "robot_not_homed",
            "limit_switch_not_ok",
            "BLOCK_INTERLOCK",
        ],
        "seedling_robot/adapters/dry_run_serial.py": [
            "def telemetry",
            "homed=self._homed",
            "interlock_ok=self.interlock_ok",
            "limit_switch_ok=self.limit_switch_ok",
            "emergency_stop_active=self.stopped",
        ],
        "tests/unit/test_decision_and_safety.py": [
            "test_risk_aware_policy_requires_homed_robot",
            "test_safety_gate_blocks_robot_telemetry_faults",
        ],
        "tests/unit/test_dry_run_and_feedback.py": [
            "test_dry_run_adapter_requires_homing_and_interlocks_for_direct_moves",
            "test_dry_run_adapter_telemetry_reports_homing_limit_and_estop",
        ],
    },
    "I-004": {
        "seedling_robot/adapters/dry_run_serial.py": [
            "class DryRunSerialAdapter",
            "port: str = \"DRY_RUN\"",
            "command_log_path",
            "serialized_commands",
            "pointer_only",
            "no hardware movement performed",
            "dry-run command serialized",
            "def execute_command",
            "def telemetry",
            "def _require_ready",
        ],
        "tests/unit/test_dry_run_and_feedback.py": [
            "test_dry_run_adapter_serializes_command_after_safety_gate",
            "test_dry_run_adapter_blocks_unsupported_tool_profile_before_motion",
            "test_dry_run_adapter_requires_homing_and_interlocks_for_direct_moves",
        ],
        "tests/unit/test_robot_and_replay.py": [
            "test_dry_run_control_points_write_report_and_command_log",
            "test_dry_run_plan_runner_writes_report_command_log_and_replay",
        ],
    },
    "I-006": {
        "docs/ROBOT_PROTOCOL.md": [
            "MotionCommand",
            "SpeedProfile",
            "ToolProfile",
            "SafetyGate",
            "replay_motion_commands",
            "dry-run-plan",
            "hil-pointer-run",
        ],
    },
    "I-007": {
        "seedling_robot/motion_replay.py": [
            "def replay_motion_commands",
            "adapter.connect",
            "adapter.home",
            "adapter.move_to",
            "stop_on_failure",
            "def load_motion_commands",
            "def run_motion_replay",
        ],
        "tests/unit/test_dry_run_and_feedback.py": [
            "test_motion_replay_uses_robot_adapter",
            "test_motion_replay_stops_after_failed_move_by_default",
        ],
        "tests/unit/test_robot_and_replay.py": [
            "test_motion_replay_cli_writes_simulation_report_and_registry",
        ],
    },
    "I-008": {
        "seedling_robot/dry_run_runner.py": [
            "def run_dry_run_plan",
            "DryRunSerialAdapter",
            "SafetyGate",
            "operator_confirmed",
            "interlock_ok",
            "limit_switch_ok",
            "emergency_stop",
            "skipped",
        ],
        "seedling_robot/cli.py": [
            "dry-run-test",
            "dry-run-plan",
            "seedling-robot:dry-run-test",
            "seedling-robot:dry-run-plan",
            "_write_robot_snapshot",
            "_write_robot_registry",
        ],
        "tests/unit/test_robot_and_replay.py": [
            "test_dry_run_control_points_cli",
            "test_dry_run_plan_cli_writes_artifact_registry",
            "test_dry_run_plan_runner_stops_after_failed_command",
        ],
    },
    "J-001": {
        "seedling_ui/offline_viewer.py": [
            "def write_offline_viewer",
            "def render_offline_viewer_html",
            "def _is_scene_state",
            "def _render_scene_state_viewer_html",
            "def _scene_cells_table",
            "def _scene_targets_table",
            "def _scene_detections_table",
            "data-feedback-panel",
        ],
        "tests/unit/test_reports_and_viewers.py": [
            "test_offline_viewer_writes_html",
            "test_offline_viewer_writes_scene_state_cells_and_targets",
            "test_offline_viewer_selects_scene_state_from_bundle",
        ],
    },
    "J-002": {
        "seedling_ui/replay_viewer.py": [
            "def write_replay_viewer",
            "def render_replay_viewer_html",
            "def _timeline_controls",
            "data-action=\"reset\"",
            "data-action='play'",
            "def _safety_summary_table",
            "def _timeline_table",
        ],
        "tests/unit/test_reports_and_viewers.py": [
            "test_replay_viewer_writes_safety_reasons",
            "test_ui_viewer_cli_writes_artifact_registry",
        ],
    },
    "J-003": {
        "seedling_ui/offline_viewer.py": [
            "def _target_reasons",
            "review_reasons",
            "block_reasons",
            "blocked_reasons",
            "safety_decision",
            "forbidden_zone_overlap",
            "human_review_required",
            "cell_review_required",
        ],
        "seedling_decision/policies/builtin.py": [
            "review_reasons_by_target",
            "blocked_reasons_by_target",
            "blocked_target_ids",
            "human_review_required",
        ],
        "tests/unit/test_reports_and_viewers.py": [
            "test_offline_viewer_writes_html",
            "test_offline_viewer_writes_scene_state_cells_and_targets",
            "test_replay_viewer_writes_safety_reasons",
        ],
    },
    "J-004": {
        "seedling_ui/cli.py": [
            "selector = subparsers.add_parser",
            "model_registry = subparsers.add_parser",
            "ComponentRegistry.from_file",
            "ModelRegistry.from_file",
            "registry.select",
            "model_id=args.model_id",
        ],
        "seedling_core/registry.py": [
            "class ComponentRegistry",
            "class ModelRegistry",
            "def select",
            "model_type",
            "output_schema",
            "safety_level",
            "requires_action_mask",
            "direct_hardware_access",
        ],
        "tests/unit/test_registry_selector_feedback.py": [
            "test_selector_cli_emits_selected_entry",
            "test_selector_cli_validates_component_registry",
            "test_typed_model_registry_validates_required_metadata",
        ],
    },
    "J-005": {
        "seedling_reports/scenario.py": [
            "compare_policies_on_scene",
            "rl_no_model",
            "rl_checkpoint",
        ],
    },
    "J-006": {
        "seedling_ui/report_export.py": [
            "def write_report_export",
            "def build_report_payload",
            "def _scene_summary",
            "def _distributions",
            "def _target_rows",
            "def _replay_summary",
            "def _render_markdown",
            "def _render_html",
            "reason_counts",
            "safety_blocks",
        ],
        "tests/unit/test_reports_and_viewers.py": [
            "test_report_export_writes_markdown_and_html",
            "test_ui_viewer_cli_writes_artifact_registry",
        ],
    },
    "J-007": {
        "seedling_ui/feedback.py": [
            "class AnnotationFeedback",
            "class FeedbackAnnotationTask",
            "VALID_FEEDBACK_PRIORITIES",
            "DEFAULT_PRIORITY_BY_ERROR_TYPE",
            "def append_feedback",
            "def read_feedback",
            "def feedback_to_annotation_tasks",
            "def write_annotation_tasks_from_feedback",
        ],
        "seedling_ui/offline_viewer.py": [
            "def _feedback_panel",
            "data-feedback-error-type",
            "data-feedback-json",
            "wrong_target",
            "unsafe_action",
        ],
        "tests/unit/test_registry_selector_feedback.py": [
            "test_feedback_rows_convert_to_annotation_tasks",
            "test_feedback_priority_override_is_preserved_in_annotation_task",
            "test_feedback_cli_writes_annotation_task_jsonl",
        ],
    },
    "K-001": {
        "seedling_experiments/cli.py": [
            "_add_config_command(subparsers, \"train\"",
            "_add_config_command(subparsers, \"val\"",
            "_add_config_command(subparsers, \"predict\"",
            "_add_config_command(subparsers, \"evaluate\"",
            "_add_config_command(subparsers, \"prepare\"",
            "subparsers.add_parser(\"sim\"",
            "subparsers.add_parser(\"rl\"",
            "def run_sim_smoke",
            "def run_rl_command",
        ],
        "tests/unit/test_cli_smoke.py": [
            "test_unified_experiment_sim_smoke_writes_report",
            "test_unified_experiment_rl_dry_run_writes_report",
            "test_unified_experiment_rl_baseline_eval_writes_registry",
        ],
    },
    "K-002": {
        "seedling_experiments/cli.py": [
            "def _write_experiment_registry",
            "ArtifactRecord.from_path",
            "role=\"input\"",
            "role=\"output\"",
            "write_run_registry_records",
        ],
        "seedling_experiments/config.py": [
            "save_run_snapshot",
            "validate_experiment_config",
            "config_hash",
            "_existing_path_fields",
        ],
        "seedling_reports/registry.py": [
            "class ArtifactRecord",
            "class RunRecord",
            "class ExperimentRegistry",
            "def to_json",
            "def to_csv",
            "def write_run_registry_records",
        ],
        "tests/unit/test_cli_smoke.py": [
            "test_prepare_cli_snapshot_records_command_args",
        ],
        "tests/unit/test_reports_and_viewers.py": [
            "test_write_run_registry_records_preserves_artifact_roles",
        ],
    },
    "K-003": {
        "seedling_reports/tables.py": [
            "def write_dataset_summary_csv",
            "dataset_audit.json",
            "raw_dataset_audit.json",
            "audit_stage",
            "class_counts",
        ],
        "tests/unit/test_reports_and_viewers.py": [
            "test_report_tables_collect_known_artifacts",
        ],
    },
    "K-004": {
        "seedling_reports/tables.py": [
            "def write_object_level_results_csv",
            "_metrics.json",
            "map50",
            "map50_95",
            "precision_mean",
            "recall_mean",
        ],
        "tests/unit/test_reports_and_viewers.py": [
            "test_report_tables_collect_known_artifacts",
        ],
    },
    "K-005": {
        "seedling_reports/tables.py": [
            "def write_task_level_results_csv",
            "cell_metrics.json",
            "scene_metrics.json",
            "schema_mode",
            "cell_accuracy",
            "target_recall",
            "mean_coordinate_error_mm",
            "expert_keep_remove",
            "cost_sensitive",
        ],
        "tests/unit/test_reports_and_viewers.py": [
            "test_report_tables_collect_known_artifacts",
            "test_task_level_results_preserve_scene_state_source_ids",
            "test_task_level_results_include_direct_scene_metrics",
        ],
    },
    "K-006": {
        "seedling_reports/registry.py": [
            "class ArtifactRecord",
            "role",
            "sha256",
            "def write_run_registry_records",
            "artifact_registry.json",
            "artifact_registry.csv",
        ],
        "seedling_reports/tables.py": [
            "def write_artifact_registry_csv",
            "def collect_artifacts",
            "ArtifactRecord.from_path",
            "run_snapshot",
            "def _artifact_type",
        ],
        "tests/unit/test_reports_and_viewers.py": [
            "test_write_run_registry_creates_json_and_csv",
            "test_write_run_registry_records_preserves_artifact_roles",
        ],
    },
    "K-007": {
        "seedling_reports/html.py": [
            "def write_experiment_html_report",
            "<!doctype html>",
            "<table><thead>",
            "No rows.",
        ],
        "seedling_reports/cli.py": [
            "write_experiment_html_report",
            "experiment_report.html",
            "Dataset Summary",
            "Object-Level Results",
            "Task-Level Results",
            "Artifacts",
        ],
        "tests/unit/test_reports_and_viewers.py": [
            "test_reports_build_cli_writes_rl_results",
            "test_reports_build_cli_writes_hardware_dry_run_results",
        ],
    },
    "K-008": {
        "seedling_reports/compare.py": [
            "min_rl_seed_count",
            "max_run_snapshot_missing_config_hash",
            "max_run_snapshot_missing_command_args",
        ],
    },
    "L-001": {
        "tests/unit/test_schemas.py": [
            "test_detection_object_computes_center_and_area",
            "test_scene_state_round_trip",
            "test_action_command_round_trip",
        ],
    },
    "L-002": {
        "tests/unit/test_grid.py": [
            "test_cell_index_uses_last_cell_for_max_boundary",
            "test_assign_to_cells_and_count_matrix",
            "test_generate_grid_polygons_supports_tray_corners",
            "test_corner_grid_assignment_uses_cell_polygons_not_outer_bbox",
        ],
    },
    "L-003": {
        "tests/unit/test_target_generation.py": [
            "test_largest_bbox_strategy_generates_extra_crop_targets",
            "test_unknown_cell_requires_review_and_no_auto_target",
            "test_weed_only_cell_generates_remove_weed_target",
            "test_crop_and_weed_cell_generates_reviewed_remove_weed_target",
        ],
    },
    "L-004": {
        "tests/unit/test_decision_and_safety.py": [
            "test_action_mask_reports_reasons",
            "test_action_mask_blocks_processed_and_unsafe_targets",
            "test_safety_gate_blocks_bad_calibration_and_close_target",
            "test_safety_gate_blocks_robot_telemetry_faults",
        ],
    },
    "L-005": {
        "tests/unit/test_calibration.py": [
            "test_image_and_robot_transforms",
            "test_estimate_calibration_artifact_from_four_points",
            "test_validator_rejects_wrong_tray_type_and_rms_error",
            "test_calibration_validate_cli_writes_report_and_registry",
        ],
    },
    "L-006": {
        "tests/integration/test_scene_policy_sim.py": [
            "test_rule_based_policy_completes_fixture_scene_through_safety_gate",
            "SceneState",
            "RasterScanPolicy",
            "SimulatorRobotAdapter",
        ],
    },
    "L-007": {
        "tests/integration/test_scene_policy_sim.py": [
            "LogicalTraySimulator",
            "ActuatorErrorModel",
            "execute_command",
            "safety_decision.allowed",
            "removed",
        ],
    },
    "L-008": {
        "tests/unit/test_rl_training_and_compare.py": [
            "test_rl_train_dry_run_writes_summary_and_registry",
            "test_rl_policy_adapter_reports_blocked_action_mask_reasons",
            "test_compare_runs_applies_thresholds_and_reads_rl_seed_stability",
            "test_sweep_stability_report_groups_runs_across_seeds",
        ],
    },
    "L-009": {
        "tests/unit/test_evaluate_cells_metrics.py": [
            "test_evaluate_cells_reports_calibrated_target_error_mm",
            "test_evaluate_cells_reports_legacy_cost_sensitive_errors",
            "test_evaluate_cells_can_use_scene_state_schema_inputs",
        ],
    },
    "L-010": {
        "tests/unit/test_cli_smoke.py": [
            "test_architecture_cli_modules_show_help",
            "test_requirement_splits_match_pyproject_dependency_groups",
            "test_prepare_cli_snapshot_records_command_args",
            "test_unified_experiment_rl_dry_run_writes_report",
        ],
    },
    "L-011": {
        "tests/performance/test_performance_smoke.py": [
            "batch_predict",
            "train_from_config",
            "test_rl_training_dry_run_path_is_bounded",
        ],
    },
    "M-001": {
        "docs/README.md": [
            "Карта документов",
            "Быстрый старт",
            "seedling_reports build",
            "artifact_registry.json",
        ],
    },
    "M-002": {
        "docs/ARCHITECTURE.md": [
            "Архитектурный каркас",
            "Поток данных",
            "SafetyGate",
            "Реестры и селекторы",
        ],
    },
    "M-003": {
        "docs/RL_SPEC.md": [
            "SeedlingTrayEnv",
            "Награда",
            "MaskablePPO",
            "RecurrentPPO",
            "offline-replay-eval",
            "sweep",
        ],
    },
    "M-004": {
        "docs/SIMULATION_SPEC.md": [
            "LogicalTraySimulator",
            "DetectionNoiseModel",
            "ActuatorErrorModel",
            "PlantResponseModel",
            "run-policy",
        ],
    },
    "M-005": {
        "docs/MODEL_PLUGIN_API.md": [
            "DetectorAdapter",
            "Реестр моделей и политик",
            "Миграция старых предсказаний",
            "Адаптеры политик и робота",
            "SafetyGate",
        ],
    },
    "M-006": {
        "docs/SAFETY_CONCEPT.md": [
            "SafetyGate",
            "BLOCK_CALIBRATION_REQUIRED",
            "BLOCK_INTERLOCK",
            "dry-run-plan",
            "HardwareInLoopReview",
        ],
    },
    "M-007": {
        "docs/CALIBRATION_PROTOCOL.md": [
            "CalibrationConfig",
            "CalibrationArtifact",
            "CalibrationValidator",
            "error_map",
            "artifact_registry.json",
        ],
    },
    "M-008": {
        "docs/FAILURE_MODES.md": [
            "Режим отказа",
            "Как обнаруживается",
            "Мера снижения",
            "Калибровка отсутствует",
            "Межблокировка не подтверждена",
            "опасный профиль инструмента",
        ],
    },
    "M-009": {
        "docs/ARTICLE_REPORT_TEMPLATE.md": [
            "Исходные артефакты",
            "Таблица 0. Реестр моделей",
            "Состав набора данных",
            "Метрики безопасности",
            "Воспроизводимость",
        ],
    },
    "M-010": {
        "docs/OPERATOR_MANUAL_DRAFT.md": [
            "Режимы",
            "Процедура сухого прогона",
            "Процедура HIL с указателем",
            "Обратная связь",
            "Проверка после действия",
        ],
    },
}


@dataclass(frozen=True)
class BacklogItem:
    item_id: str
    priority: str
    title: str
    acceptance: str


def build_architecture_backlog_audit(
    architecture_doc: str | Path,
    output_dir: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    doc_path = Path(architecture_doc)
    repo = Path(repo_root) if repo_root is not None else doc_path.parent.parent
    items = parse_backlog_items(doc_path)
    rows = [_audit_row(item, repo) for item in items]
    counts = _counts(rows)
    payload = {
        "schema_version": "architecture_backlog_audit_v0_1",
        "ok": counts.get("missing", 0) == 0,
        "architecture_doc": str(doc_path),
        "repo_root": str(repo),
        "counts": counts,
        "items": rows,
    }
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "architecture_backlog_audit.json"
        csv_path = output / "architecture_backlog_audit.csv"
        html_path = output / "architecture_backlog_audit.html"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_csv(csv_path, rows)
        write_experiment_html_report("Architecture Backlog Audit", html_path, [("Backlog Items", _html_rows(rows))])
        payload["summary_path"] = str(json_path)
        payload["csv_path"] = str(csv_path)
        payload["html_path"] = str(html_path)
    return payload


def parse_backlog_items(path: str | Path) -> list[BacklogItem]:
    rows: list[BacklogItem] = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        match = BACKLOG_TABLE_RE.match(line)
        if not match:
            continue
        rows.append(BacklogItem(**match.groupdict()))
    return rows


def _audit_row(item: BacklogItem, repo: Path) -> dict[str, Any]:
    expected = EVIDENCE_RULES.get(item.item_id, [])
    existing = [path for path in expected if (repo / path).exists()]
    missing = [path for path in expected if not (repo / path).exists()]
    content_errors = _content_errors(item.item_id, repo)
    if item.item_id in EXTERNAL_BACKLOG_IDS:
        status = "external_required"
        notes = "Software scaffold exists if evidence is present; real hardware validation remains external."
    elif expected and not missing and not content_errors:
        status = "pass"
        notes = ""
    elif expected:
        status = "missing"
        notes = "Required evidence paths are missing." if missing else "Evidence content checks failed."
    else:
        status = "unmapped"
        notes = "No evidence rule is defined for this backlog item."
    if content_errors:
        detail = " ".join(content_errors)
        notes = f"{notes} {detail}".strip()
    return {
        "item_id": item.item_id,
        "priority": item.priority,
        "title": _strip_markup(item.title),
        "acceptance": _strip_markup(item.acceptance),
        "status": status,
        "evidence": existing,
        "missing_evidence": missing,
        "notes": notes,
    }


def _content_errors(item_id: str, repo: Path) -> list[str]:
    errors: list[str] = []
    for relative_path, required_markers in CONTENT_RULES.get(item_id, {}).items():
        path = repo / relative_path
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            errors.append(f"{relative_path}: content check could not read file: {exc}")
            continue
        missing = [marker for marker in required_markers if marker not in text]
        if missing:
            errors.append(f"{relative_path}: missing content markers {', '.join(missing)}.")
    return errors


def _strip_markup(value: str) -> str:
    return value.replace("`", "").strip()


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["item_id", "priority", "title", "acceptance", "status", "evidence", "missing_evidence", "notes"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "evidence": _json_cell(row["evidence"]), "missing_evidence": _json_cell(row["missing_evidence"])})


def _html_rows(rows: list[dict[str, Any]]) -> list[dict[str, object]]:
    return [
        {
            "item_id": row["item_id"],
            "priority": row["priority"],
            "title": row["title"],
            "status": row["status"],
            "evidence": ", ".join(row["evidence"]),
            "missing_evidence": ", ".join(row["missing_evidence"]),
            "notes": row["notes"],
        }
        for row in rows
    ]


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def rows_to_dict(rows: list[BacklogItem]) -> list[dict[str, Any]]:
    return [asdict(row) for row in rows]
