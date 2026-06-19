from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .html import write_experiment_html_report


@dataclass(frozen=True)
class ReadinessItem:
    item_id: str
    requirement: str
    status: str
    evidence: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence"] = list(self.evidence)
        return payload


def build_readiness_report(
    root: str | Path,
    output_dir: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    artifact_root = Path(root)
    repository = Path(repo_root) if repo_root is not None else Path.cwd()
    items = _readiness_items(repository, artifact_root)
    counts = _status_counts(items)
    payload = {
        "schema_version": "readiness_report_v0_1",
        "ok": counts.get("missing", 0) == 0,
        "software_ready": counts.get("missing", 0) == 0,
        "physical_validation_required": counts.get("external_required", 0) > 0,
        "root": str(artifact_root),
        "repo_root": str(repository),
        "counts": counts,
        "items": [item.to_dict() for item in items],
    }
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        summary_path = output / "readiness_report.json"
        csv_path = output / "readiness_report.csv"
        html_path = output / "readiness_report.html"
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_readiness_csv(csv_path, items)
        write_experiment_html_report(
            "Seedling Architecture Readiness",
            html_path,
            [("Readiness Items", [_row(item) for item in items])],
        )
        payload["summary_path"] = str(summary_path)
        payload["csv_path"] = str(csv_path)
        payload["html_path"] = str(html_path)
    return payload


def build_software_readiness_smoke(
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    from seedling_calibration import CalibrationConfig, estimate_calibration_artifact
    from seedling_calibration.estimation import write_error_map
    from seedling_core.schemas import ActionTarget, CellState, RobotState, SafetyState, SceneState, TrayState
    from seedling_reports.architecture_audit import build_architecture_backlog_audit
    from seedling_reports.scenario import compare_policy_scenario_file, write_policy_plan_file
    from seedling_reports.tables import write_hardware_dry_run_results_csv, write_safety_results_csv
    from seedling_robot.dry_run_runner import run_dry_run_plan

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(repo_root) if repo_root is not None else Path.cwd()

    calibration_dir = output / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    calibration_config_path = calibration_dir / "calibration_config.json"
    calibration_path = calibration_dir / "calibration.json"
    error_map_path = calibration_dir / "error_map.json"
    calibration_config = CalibrationConfig(
        camera_id="smoke_cam",
        tray_type="11x11",
        tray_size_mm=[50.0, 50.0],
        grid_rows=11,
        grid_cols=11,
        target_points_px=[[0, 0], [100, 0], [100, 100], [0, 100]],
        target_points_tray_mm=[[0, 0], [50, 0], [50, 50], [0, 50]],
        robot_reference_points_mm=[[10, 20, 0], [60, 20, 0], [60, 70, 0], [10, 70, 0]],
        valid_hours=24 * 3650,
    )
    calibration_config_path.write_text(json.dumps(calibration_config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    calibration = estimate_calibration_artifact(
        calibration_config,
        calibration_id="software_readiness_smoke",
        created_at="2026-06-18T00:00:00+00:00",
    )
    calibration.to_json(calibration_path)
    write_error_map(calibration_config_path, calibration_path, error_map_path)

    scene_path = output / "scene_state.json"
    plan_path = output / "action_plan.json"
    scene = _software_smoke_scene()
    scene_path.write_text(json.dumps(scene.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    write_policy_plan_file(scene_path, "route_planning", plan_path)

    scenario_dir = output / "scenario_compare"
    compare_policy_scenario_file(scene_path, ["raster_scan", "route_planning", "risk_aware_rule", "rl_no_model"], scenario_dir)

    robot_dir = output / "robot"
    robot_dir.mkdir(parents=True, exist_ok=True)
    dry_run_report = robot_dir / "dry_run_plan_report.json"
    command_log = robot_dir / "dry_run_plan_commands.json"
    replay_path = robot_dir / "dry_run_plan_replay.json"
    dry_run_payload = run_dry_run_plan(
        scene_path,
        plan_path,
        dry_run_report,
        command_log_path=command_log,
        replay_path=replay_path,
        interlock_ok=True,
    )

    hardware_csv = output / "hardware_dry_run_results.csv"
    safety_csv = output / "safety_results.csv"
    hardware_rows = write_hardware_dry_run_results_csv(output, hardware_csv)
    safety_rows = write_safety_results_csv(output, safety_csv)
    architecture_doc = repo / "docs" / "seedlings_vilga_architecture_codex_rl_simulation.md"
    architecture_audit = None
    architecture_artifacts: list[Path] = []
    if architecture_doc.exists():
        architecture_audit = build_architecture_backlog_audit(
            architecture_doc,
            output / "architecture_backlog",
            repo_root=repo,
        )
        architecture_artifacts = [
            output / "architecture_backlog" / "architecture_backlog_audit.json",
            output / "architecture_backlog" / "architecture_backlog_audit.csv",
            output / "architecture_backlog" / "architecture_backlog_audit.html",
        ]
    readiness = build_readiness_report(output, output / "readiness", repo_root=repo)

    artifacts = [
        calibration_config_path,
        calibration_path,
        error_map_path,
        scene_path,
        plan_path,
        scenario_dir / "scenario_compare.json",
        scenario_dir / "scenario_compare.html",
        dry_run_report,
        command_log,
        replay_path,
        hardware_csv,
        safety_csv,
        *architecture_artifacts,
        output / "readiness" / "readiness_report.json",
        output / "readiness" / "readiness_report.csv",
        output / "readiness" / "readiness_report.html",
    ]
    payload = {
        "ok": dry_run_payload["ok"] and readiness["software_ready"],
        "schema_version": "software_readiness_smoke_v0_1",
        "out": str(output),
        "dry_run_ok": dry_run_payload["ok"],
        "hardware_rows": len(hardware_rows),
        "safety_rows": len(safety_rows),
        "readiness_counts": readiness["counts"],
        "architecture_backlog_counts": architecture_audit["counts"] if architecture_audit else None,
        "architecture_backlog_audit": str(output / "architecture_backlog" / "architecture_backlog_audit.json")
        if architecture_audit
        else None,
        "software_ready": readiness["software_ready"],
        "physical_validation_required": readiness["physical_validation_required"],
        "artifacts": [str(path) for path in artifacts],
        "readiness_report": str(output / "readiness" / "readiness_report.json"),
    }
    summary_path = output / "software_readiness_smoke.json"
    payload["artifacts"].append(str(summary_path))
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _software_smoke_scene() -> Any:
    from seedling_core.schemas import ActionTarget, CellState, RobotState, SafetyState, SceneState, TrayState

    return SceneState(
        scene_id="software_readiness_scene",
        image_ref="software_readiness.png",
        dataset_version="software_readiness",
        ontology_version="ontology_v0_1",
        image_size_px=[100, 100],
        tray=TrayState("tray_smoke", 1, 1, bbox_xyxy_px=[0, 0, 100, 100], calibration_id="software_readiness_smoke"),
        robot=RobotState(position_mm=[0, 0, 0], homed=True, mode="dry_run_pointer"),
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
                target_id="target_smoke_001",
                cell_id="r00_c00",
                object_id="plant_extra_001",
                target_type="remove_extra_crop",
                action_point_px=[20, 20],
                action_point_mm=[10, 10],
                robot_point_mm=[20, 30, 0],
                uncertainty_radius_mm=1.0,
                min_distance_to_keep_mm=6.0,
                human_review_required=False,
            )
        ],
        safety=SafetyState(calibration_valid=True, interlock_ok=True, software_safe_mode=True),
    )


def _readiness_items(repo: Path, root: Path) -> list[ReadinessItem]:
    return [
        _repo_item(
            "DOD-001",
            "SceneState / CellState / ActionTarget / ActionCommand schemas exist.",
            repo,
            ["seedling_core/schemas.py"],
        ),
        _component_registry_item(
            "DOD-002",
            "Detector models connect through DetectorAdapter and registries.",
            repo,
            repo_paths=["seedling_vision/adapters/base.py", "configs/registry/models_v0_1.yaml"],
            registry_path="configs/registry/models_v0_1.yaml",
            required_kinds=["detector"],
        ),
        _component_registry_item(
            "DOD-003",
            "Policies connect through DecisionPolicy and registry.",
            repo,
            repo_paths=["seedling_decision/policies/base.py", "configs/registry/policies_v0_1.yaml"],
            registry_path="configs/registry/policies_v0_1.yaml",
            required_kinds=["policy"],
        ),
        _repo_item(
            "DOD-004",
            "Robot and simulator connect through RobotAdapter.",
            repo,
            [
                "seedling_robot/adapters/base.py",
                "seedling_robot/adapters/simulator.py",
                "seedling_robot/adapters/dry_run_serial.py",
            ],
        ),
        _repo_item(
            "DOD-005",
            "Action execution is gated by SafetyGate contracts.",
            repo,
            ["seedling_decision/safety_gate.py", "docs/SAFETY_CONCEPT.md"],
        ),
        _calibration_item(
            "DOD-006",
            "Calibration artifact and error map evidence exists.",
            repo,
            root,
            repo_paths=["seedling_calibration/schemas.py", "seedling_calibration/cli.py"],
            artifact_patterns=["**/calibration.json", "**/error_map.json", "**/*error_map*.json", "**/*error_map*.html"],
        ),
        _model_registry_item(
            "DOD-007",
            "Model registry exists.",
            repo,
            registry_path="configs/registry/model_registry_v0_1.yaml",
        ),
        _dataset_descriptor_item(
            "DOD-008",
            "Dataset registry exists.",
            repo,
            dataset_path="configs/datasets/dataset_v0_1.yaml",
        ),
        _repo_item(
            "DOD-009",
            "Simulation environment exists.",
            repo,
            ["seedling_rl/envs.py", "configs/simulation/tray_env_v0.yaml"],
        ),
        _repo_item(
            "DOD-010",
            "RL training block exists.",
            repo,
            ["seedling_rl/training.py", "configs/rl/maskable_ppo_v0.yaml"],
        ),
        _rl_baseline_comparison_item(
            "DOD-011",
            "RL has been compared with baseline policies in generated artifacts.",
            root,
            ["**/rl_results.csv", "**/unified_rl_baselines.json", "**/scenario_compare.json", "**/policy_comparison.json", "**/policy_comparison.html"],
        ),
        _repo_item(
            "DOD-012",
            "Replay viewer exists.",
            repo,
            ["seedling_ui/replay_viewer.py"],
        ),
        _repo_item(
            "DOD-013",
            "Offline analysis UI exists.",
            repo,
            ["seedling_ui/offline_viewer.py"],
        ),
        _artifact_or_repo_item(
            "DOD-014",
            "Dry-run mode without dangerous actuation exists.",
            repo,
            root,
            repo_paths=["seedling_robot/dry_run_runner.py", "seedling_robot/adapters/dry_run_serial.py"],
            artifact_patterns=["**/dry_run_plan_report.json", "**/dry_run_control_points.json"],
        ),
        _artifact_item(
            "DOD-015",
            "Actions are logged in command logs or replay artifacts.",
            root,
            ["**/*commands.json", "**/*replay.json", "**/replays/*.json", "**/training_log.jsonl"],
        ),
        _artifact_item(
            "DOD-016",
            "Metrics are generated from artifacts.",
            root,
            [
                "**/dataset_summary.csv",
                "**/object_level_results.csv",
                "**/task_level_results.csv",
                "**/rl_results.csv",
                "**/hardware_dry_run_results.csv",
                "**/safety_results.csv",
            ],
        ),
        _repo_item(
            "DOD-017",
            "Tests cover grid, calibration, safety, simulation and policies.",
            repo,
            [
                "tests/unit/test_grid.py",
                "tests/unit/test_calibration.py",
                "tests/unit/test_decision_and_safety.py",
                "tests/unit/test_simulation.py",
                "tests/integration/test_scene_policy_sim.py",
            ],
        ),
        _repo_item(
            "DOD-018",
            "Documentation describes limitations and blocks autonomous dangerous actuation.",
            repo,
            ["docs/SAFETY_CONCEPT.md", "docs/OPERATOR_MANUAL_DRAFT.md", "docs/ROBOT_PROTOCOL.md"],
        ),
        _hil_external_item(
            "EXT-001",
            "Physical gantry HIL validation is externally verified.",
            root,
            ["**/hil_pointer_report.json", "**/hil_pointer_commands.json", "**/hil_pointer_replay.json"],
            "Requires a real supervised stand run and operator safety review; software artifacts alone are not proof.",
        ),
        _post_action_external_item(
            "EXT-002",
            "Delayed biological post-action validation is externally verified.",
            root,
            ["**/post_action_results.csv", "**/post_action_summary.json", "**/post_action_observations.jsonl"],
            "Requires real 24h/48h/72h observations under a documented biological protocol.",
        ),
    ]


def _repo_item(item_id: str, requirement: str, repo: Path, paths: list[str]) -> ReadinessItem:
    evidence = _existing_repo_paths(repo, paths)
    return ReadinessItem(
        item_id=item_id,
        requirement=requirement,
        status="pass" if len(evidence) == len(paths) else "missing",
        evidence=evidence,
        notes="" if len(evidence) == len(paths) else "Required repository evidence is missing.",
    )


def _artifact_item(item_id: str, requirement: str, root: Path, patterns: list[str]) -> ReadinessItem:
    evidence = _find_artifacts(root, patterns)
    return ReadinessItem(
        item_id=item_id,
        requirement=requirement,
        status="pass" if evidence else "missing",
        evidence=evidence,
        notes="" if evidence else "No matching generated artifact found under root.",
    )


_BASELINE_POLICY_IDS = {
    "human_review",
    "nearest_neighbor",
    "nearest_neighbor_v0",
    "noop",
    "raster_scan",
    "risk_aware_rule",
    "route_planning",
    "route_planning_v0",
    "rule_based_v0",
}


def _rl_baseline_comparison_item(item_id: str, requirement: str, root: Path, patterns: list[str]) -> ReadinessItem:
    artifact_paths = _find_artifact_paths(root, patterns)
    evidence = [str(path) for path in artifact_paths]
    rl_evidence: set[str] = set()
    baseline_evidence: set[str] = set()
    validation_notes: list[str] = []
    for path in artifact_paths:
        try:
            policies = _comparison_policy_tokens(path)
        except (OSError, ValueError, json.JSONDecodeError, csv.Error) as exc:
            validation_notes.append(f"{path}: unreadable comparison artifact: {exc}")
            continue
        if not policies:
            validation_notes.append(f"{path}: no policy identifiers found")
            continue
        if any(_is_rl_policy_token(token) for token in policies):
            rl_evidence.add(str(path))
        if any(_is_baseline_policy_token(token) for token in policies):
            baseline_evidence.add(str(path))

    if rl_evidence and baseline_evidence:
        return ReadinessItem(
            item_id=item_id,
            requirement=requirement,
            status="pass",
            evidence=evidence,
            notes="Validated generated comparison evidence containing both RL and baseline policies.",
        )
    if not evidence:
        notes = "No matching generated comparison artifact found under root."
    elif not rl_evidence and not baseline_evidence:
        notes = "No generated comparison artifact contains recognizable RL or baseline policy identifiers."
    elif not rl_evidence:
        notes = "Generated comparison artifacts contain baseline policies but no RL policy row."
    else:
        notes = "Generated comparison artifacts contain RL policies but no baseline policy row."
    if validation_notes:
        notes = f"{notes} Latest validation issue: {validation_notes[0]}"
    return ReadinessItem(item_id=item_id, requirement=requirement, status="missing", evidence=evidence, notes=notes)


def _comparison_policy_tokens(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _comparison_policy_tokens_from_csv(path)
    if suffix in {".json", ".jsonl"}:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        return _comparison_policy_tokens_from_json(data)
    if suffix in {".html", ".htm"}:
        return _comparison_policy_tokens_from_text(path.read_text(encoding="utf-8", errors="ignore"))
    return []


def _comparison_policy_tokens_from_csv(path: Path) -> list[str]:
    tokens: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tokens.extend(_comparison_policy_tokens_from_row(row))
    return tokens


def _comparison_policy_tokens_from_json(data: Any) -> list[str]:
    tokens: list[str] = []
    if isinstance(data, dict):
        tokens.extend(_comparison_policy_tokens_from_row(data))
        for key in ("rows", "results", "policies"):
            value = data.get(key)
            if isinstance(value, list):
                for item in value:
                    tokens.extend(_comparison_policy_tokens_from_json(item))
            elif isinstance(value, dict):
                tokens.extend(_comparison_policy_tokens_from_json(value))
    elif isinstance(data, list):
        for item in data:
            tokens.extend(_comparison_policy_tokens_from_json(item))
    elif isinstance(data, str):
        tokens.append(data)
    return tokens


def _comparison_policy_tokens_from_row(row: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    mode = str(row.get("mode", "")).strip().lower()
    if mode == "checkpoint" or row.get("checkpoint"):
        tokens.append("rl_checkpoint")
    for key in ("policy_selector", "requested_policy", "policy", "baseline", "mode", "checkpoint"):
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            tokens.append(text)
    return tokens


def _comparison_policy_tokens_from_text(text: str) -> list[str]:
    normalized = text.lower()
    tokens = [policy for policy in _BASELINE_POLICY_IDS if policy in normalized]
    for marker in ("rl_no_model", "rl_policy_adapter", "rl_checkpoint", "checkpoint_episode", "rl_eval_metrics"):
        if marker in normalized:
            tokens.append(marker)
    return tokens


def _is_rl_policy_token(token: str) -> bool:
    normalized = token.strip().lower()
    if not normalized:
        return False
    return (
        normalized in {"checkpoint", "rl", "rl_checkpoint", "rl_no_model", "rl_policy_adapter"}
        or normalized.startswith("rl_")
        or normalized.startswith("rl:")
        or normalized.startswith("rl_checkpoint:")
        or "rlpolicyadapter" in normalized
    )


def _is_baseline_policy_token(token: str) -> bool:
    normalized = token.strip().lower()
    return normalized in _BASELINE_POLICY_IDS


def _artifact_or_repo_item(
    item_id: str,
    requirement: str,
    repo: Path,
    root: Path,
    *,
    repo_paths: list[str],
    artifact_patterns: list[str],
) -> ReadinessItem:
    repo_evidence = _existing_repo_paths(repo, repo_paths)
    artifact_evidence = _find_artifacts(root, artifact_patterns)
    evidence = [*repo_evidence, *artifact_evidence]
    status = "pass" if len(repo_evidence) == len(repo_paths) and artifact_evidence else "missing"
    notes = "" if status == "pass" else "Repository support exists only if listed; generated artifacts are still required for run readiness."
    return ReadinessItem(item_id, requirement, status, evidence, notes)


def _calibration_item(
    item_id: str,
    requirement: str,
    repo: Path,
    root: Path,
    *,
    repo_paths: list[str],
    artifact_patterns: list[str],
) -> ReadinessItem:
    repo_evidence = _existing_repo_paths(repo, repo_paths)
    artifact_paths = _find_artifact_paths(root, artifact_patterns)
    calibration_paths = [path for path in artifact_paths if path.name == "calibration.json"]
    error_map_paths = [
        path
        for path in artifact_paths
        if "error_map" in path.name and path.suffix.lower() in {".json", ".html", ".htm"}
    ]
    evidence = [*repo_evidence, *(str(path) for path in artifact_paths)]
    errors: list[str] = []
    if len(repo_evidence) != len(repo_paths):
        errors.append("Required calibration repository support is missing.")
    if not calibration_paths:
        errors.append("No calibration.json artifact found under root.")
    if not error_map_paths:
        errors.append("No error_map artifact found under root.")

    valid_calibrations: list[Path] = []
    validation_notes: list[str] = []
    for path in calibration_paths:
        result = _validate_readiness_calibration(path)
        if result:
            validation_notes.extend(result)
        else:
            valid_calibrations.append(path)
    if calibration_paths and not valid_calibrations:
        errors.append("No valid calibration artifact found.")
        if validation_notes:
            errors.append(validation_notes[0])

    return ReadinessItem(
        item_id=item_id,
        requirement=requirement,
        status="missing" if errors else "pass",
        evidence=evidence,
        notes=" ".join(errors),
    )


def _validate_readiness_calibration(path: Path) -> list[str]:
    try:
        from seedling_calibration import CalibrationArtifact, CalibrationValidator

        artifact = CalibrationArtifact.from_json(path)
        validation = CalibrationValidator(max_error_p95_mm=2.0).validate(artifact)
    except Exception as exc:
        return [f"{path}: calibration validation failed: {exc}"]
    errors = [f"{path}: {error}" for error in validation.errors]
    errors.extend(f"{path}: {warning}" for warning in validation.warnings)
    return errors


def _model_registry_item(
    item_id: str,
    requirement: str,
    repo: Path,
    *,
    registry_path: str,
) -> ReadinessItem:
    registry_file = repo / registry_path
    evidence = _existing_repo_paths(repo, [registry_path])
    errors: list[str] = []
    if not registry_file.exists():
        errors.append(f"Model registry file is missing: {registry_path}.")
    else:
        try:
            from seedling_core.registry import ModelRegistry

            registry = ModelRegistry.from_file(registry_file)
            validation = registry.validate()
            if not registry.models:
                errors.append("Model registry has no model records.")
            if not validation["ok"]:
                errors.extend(str(error) for error in validation["errors"])
        except Exception as exc:
            errors.append(f"Model registry validation failed: {exc}")
    return ReadinessItem(
        item_id=item_id,
        requirement=requirement,
        status="missing" if errors else "pass",
        evidence=evidence,
        notes=" ".join(errors),
    )


def _dataset_descriptor_item(
    item_id: str,
    requirement: str,
    repo: Path,
    *,
    dataset_path: str,
) -> ReadinessItem:
    descriptor = repo / dataset_path
    evidence = _existing_repo_paths(repo, [dataset_path])
    errors: list[str] = []
    if not descriptor.exists():
        errors.append(f"Dataset descriptor file is missing: {dataset_path}.")
    else:
        try:
            from seedling_core.config import load_config_file

            data = load_config_file(descriptor)
            errors.extend(_dataset_descriptor_errors(data))
        except Exception as exc:
            errors.append(f"Dataset descriptor validation failed: {exc}")
    return ReadinessItem(
        item_id=item_id,
        requirement=requirement,
        status="missing" if errors else "pass",
        evidence=evidence,
        notes=" ".join(errors),
    )


def _dataset_descriptor_errors(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field_name in ["dataset_version", "ontology_version", "root", "images_dir", "labels_dir"]:
        if not str(data.get(field_name, "")).strip():
            errors.append(f"{field_name} is required")
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("metadata must be a mapping")
    else:
        for field_name in ["image_manifest", "split_manifest"]:
            if not str(metadata.get(field_name, "")).strip():
                errors.append(f"metadata.{field_name} is required")
    grid = data.get("grid")
    if not isinstance(grid, dict):
        errors.append("grid must be a mapping")
    else:
        for field_name in ["rows", "cols"]:
            value = grid.get(field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                errors.append(f"grid.{field_name} must be a positive integer")
    classes = data.get("classes")
    if not isinstance(classes, dict):
        errors.append("classes must be a mapping")
    else:
        normalized = {str(key): str(value) for key, value in classes.items()}
        if normalized.get("0") != "container":
            errors.append("classes.0 must be container")
        if normalized.get("1") != "crop_seedling":
            errors.append("classes.1 must be crop_seedling")
    return errors


def _component_registry_item(
    item_id: str,
    requirement: str,
    repo: Path,
    *,
    repo_paths: list[str],
    registry_path: str,
    required_kinds: list[str],
) -> ReadinessItem:
    evidence = _existing_repo_paths(repo, repo_paths)
    registry_file = repo / registry_path
    errors: list[str] = []
    if len(evidence) != len(repo_paths):
        errors.append("Required repository evidence is missing.")
    if registry_file.exists():
        try:
            from seedling_core.registry import ComponentRegistry

            registry = ComponentRegistry.from_file(registry_file)
            validation = registry.validate()
            if not validation["ok"]:
                errors.extend(str(error) for error in validation["errors"])
            errors.extend(str(warning) for warning in validation.get("warnings", []))
            for kind in required_kinds:
                if not registry.filter(kind=kind):
                    errors.append(f"Registry has no entries for kind={kind}.")
        except Exception as exc:
            errors.append(f"Registry validation failed: {exc}")
    else:
        errors.append(f"Registry file is missing: {registry_path}.")
    return ReadinessItem(
        item_id=item_id,
        requirement=requirement,
        status="missing" if errors else "pass",
        evidence=evidence,
        notes=" ".join(errors),
    )


def _external_item(
    item_id: str,
    requirement: str,
    root: Path,
    patterns: list[str],
    notes: str,
) -> ReadinessItem:
    evidence = _find_artifacts(root, patterns)
    return ReadinessItem(
        item_id=item_id,
        requirement=requirement,
        status="external_required",
        evidence=evidence,
        notes=notes,
    )


def _hil_external_item(
    item_id: str,
    requirement: str,
    root: Path,
    patterns: list[str],
    notes: str,
) -> ReadinessItem:
    evidence_paths = _find_artifact_paths(root, patterns)
    report_paths = [path for path in evidence_paths if path.name == "hil_pointer_report.json"]
    errors_by_report = [_validate_hil_external_report(path) for path in report_paths]
    passing_reports = [result for result in errors_by_report if not result]
    if passing_reports:
        return ReadinessItem(
            item_id=item_id,
            requirement=requirement,
            status="pass",
            evidence=[str(path) for path in evidence_paths],
            notes="Validated real HIL pointer report with operator review, command log and replay evidence.",
        )
    detail = _external_detail(errors_by_report)
    return ReadinessItem(
        item_id=item_id,
        requirement=requirement,
        status="external_required",
        evidence=[str(path) for path in evidence_paths],
        notes=f"{notes} {detail}".strip(),
    )


def _post_action_external_item(
    item_id: str,
    requirement: str,
    root: Path,
    patterns: list[str],
    notes: str,
) -> ReadinessItem:
    evidence_paths = _find_artifact_paths(root, patterns)
    raw_paths = [path for path in evidence_paths if path.name == "post_action_observations.jsonl"]
    summary_paths = [path for path in evidence_paths if path.name == "post_action_summary.json"]
    errors_by_artifact = [
        *(_validate_post_action_observations(path) for path in raw_paths),
        *(_validate_post_action_summary(path) for path in summary_paths),
    ]
    passing_artifacts = [result for result in errors_by_artifact if not result]
    if passing_artifacts:
        return ReadinessItem(
            item_id=item_id,
            requirement=requirement,
            status="pass",
            evidence=[str(path) for path in evidence_paths],
            notes="Validated complete 24h/48h/72h post-action observations with observer evidence.",
        )
    detail = (
        "Raw post_action_observations.jsonl or post_action_summary.json evidence is required."
        if evidence_paths and not errors_by_artifact
        else _external_detail(errors_by_artifact)
    )
    return ReadinessItem(
        item_id=item_id,
        requirement=requirement,
        status="external_required",
        evidence=[str(path) for path in evidence_paths],
        notes=f"{notes} {detail}".strip(),
    )


def _validate_hil_external_report(path: Path) -> list[str]:
    try:
        data = _load_json_object(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"{path}: unreadable HIL report: {exc}"]

    errors: list[str] = []
    if data.get("mode") != "hardware_in_loop_pointer":
        errors.append("mode must be hardware_in_loop_pointer")
    if data.get("ok") is not True:
        errors.append("report ok must be true")
    commands = _as_int(data.get("commands"))
    executed = _as_int(data.get("executed"))
    blocked = _as_int(data.get("blocked"), default=0)
    skipped = _as_int(data.get("skipped"), default=0)
    if commands is None or commands < 1:
        errors.append("commands must be positive")
    if executed is None or commands is None or executed != commands:
        errors.append("executed must equal commands")
    if blocked not in (0, None):
        errors.append("blocked must be zero")
    if skipped not in (0, None):
        errors.append("skipped must be zero")
    if data.get("aborted") is True:
        errors.append("aborted must be false")

    review = data.get("review")
    if not isinstance(review, dict):
        errors.append("review object is required")
    else:
        if review.get("ok") is not True:
            errors.append("review ok must be true")
        if not review.get("review_id"):
            errors.append("review_id is required")
        if not review.get("gantry_id"):
            errors.append("gantry_id is required")
        if review.get("emergency_stop_tested") is not True:
            errors.append("emergency_stop_tested must be true")
        if review.get("interlock_required") is not True:
            errors.append("interlock_required must be true")
        if review.get("limit_switch_required") is not True:
            errors.append("limit_switch_required must be true")
        if review.get("positioning_error_p95_mm") is None:
            errors.append("positioning_error_p95_mm is required")
        if review.get("allow_real_actuation") is True:
            errors.append("allow_real_actuation must be false for pointer-only HIL evidence")
        review_errors = review.get("errors")
        if isinstance(review_errors, list) and review_errors:
            errors.append(f"review errors present: {','.join(str(item) for item in review_errors)}")

    if not _linked_artifact_exists(path, data.get("command_log"), "hil_pointer_commands.json"):
        errors.append("command_log artifact is missing")
    if not _linked_artifact_exists(path, data.get("replay"), "hil_pointer_replay.json"):
        errors.append("replay artifact is missing")
    return [f"{path}: {error}" for error in errors]


def _validate_post_action_observations(path: Path) -> list[str]:
    try:
        from seedling_data.post_action import audit_post_action_observations, read_post_action_observations

        audit = audit_post_action_observations(path)
        rows = read_post_action_observations(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"{path}: unreadable post-action observations: {exc}"]

    errors = _post_action_summary_errors(path, audit["summary"])
    if not audit["ok"]:
        errors.extend(str(error) for error in audit["errors"])
    missing_observers = [row.command_id for row in rows if not row.observer_id]
    missing_images = [row.command_id for row in rows if not row.image_ref]
    if missing_observers:
        errors.append(f"observer_id is required for all observations: {','.join(sorted(set(missing_observers)))}")
    if missing_images:
        errors.append(f"image_ref is required for all observations: {','.join(sorted(set(missing_images)))}")
    return [f"{path}: {error}" for error in errors]


def _validate_post_action_summary(path: Path) -> list[str]:
    try:
        data = _load_json_object(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"{path}: unreadable post-action summary: {exc}"]
    errors = _post_action_summary_errors(path, data)
    if not _has_post_action_observer_evidence(data):
        errors.append("observer_ids or biological_review approval is required")
    return [f"{path}: {error}" for error in errors]


def _post_action_summary_errors(path: Path, data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if data.get("ok") is False:
        errors.append("summary ok must not be false")
    commands = _as_int(data.get("commands"))
    rows = _as_int(data.get("rows"))
    if commands is None or commands < 1:
        errors.append("commands must be positive")
    if rows is None or rows < 1:
        errors.append("rows must be positive")
    if _as_float(data.get("coverage_rate")) < 1.0:
        errors.append("coverage_rate must be 1.0")
    if data.get("missing_required_observations"):
        errors.append("missing_required_observations must be empty")
    if _as_float(data.get("delayed_success_rate")) < 1.0:
        errors.append("delayed_success_rate must be 1.0")
    if _as_float(data.get("crop_damage_rate")) > 0.0:
        errors.append("crop_damage_rate must be 0.0")
    if _as_float(data.get("regrowth_rate")) > 0.0:
        errors.append("regrowth_rate must be 0.0")
    if _as_float(data.get("uncertain_rate")) > 0.0:
        errors.append("uncertain_rate must be 0.0")
    required_hours = data.get("required_hours")
    if not isinstance(required_hours, list):
        errors.append("required_hours must be present")
    else:
        try:
            parsed_hours = sorted(float(value) for value in required_hours)
        except (TypeError, ValueError):
            parsed_hours = []
        if parsed_hours != [24.0, 48.0, 72.0]:
            errors.append("required_hours must be 24,48,72")
    return errors


def _has_post_action_observer_evidence(data: dict[str, Any]) -> bool:
    observer_ids = data.get("observer_ids")
    if isinstance(observer_ids, list) and any(str(item).strip() for item in observer_ids):
        return True
    review = data.get("biological_review")
    return (
        isinstance(review, dict)
        and review.get("ok") is True
        and bool(str(review.get("reviewer", "")).strip())
        and bool(str(review.get("approved_at", "")).strip())
    )


def _external_detail(results: list[list[str]]) -> str:
    if not results:
        return "No candidate evidence artifact was found."
    first_errors = next((errors for errors in results if errors), [])
    if not first_errors:
        return ""
    return "Latest validation issue: " + first_errors[0]


def _load_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("expected JSON object")
    return data


def _as_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _linked_artifact_exists(report_path: Path, value: Any, default_name: str) -> bool:
    candidates: list[Path] = []
    if isinstance(value, str) and value.strip():
        linked = Path(value)
        candidates.append(linked)
        if not linked.is_absolute():
            candidates.append(report_path.parent / linked)
    candidates.append(report_path.parent / default_name)
    return any(candidate.exists() and candidate.is_file() for candidate in candidates)


def _existing_repo_paths(repo: Path, paths: list[str]) -> list[str]:
    return [str(repo / path) for path in paths if (repo / path).exists()]


def _find_artifact_paths(root: Path, patterns: list[str]) -> list[Path]:
    if not root.exists():
        return []
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(path for path in root.glob(pattern) if path.is_file())
    return sorted(dict.fromkeys(matches))


def _find_artifacts(root: Path, patterns: list[str]) -> list[str]:
    return [str(path) for path in _find_artifact_paths(root, patterns)]


def _status_counts(items: list[ReadinessItem]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item.status] = counts.get(item.status, 0) + 1
    return counts


def _row(item: ReadinessItem) -> dict[str, object]:
    return {
        "item_id": item.item_id,
        "status": item.status,
        "requirement": item.requirement,
        "evidence_count": len(item.evidence),
        "notes": item.notes,
    }


def _write_readiness_csv(path: Path, items: list[ReadinessItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "status", "requirement", "evidence", "notes"])
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "item_id": item.item_id,
                    "status": item.status,
                    "requirement": item.requirement,
                    "evidence": json.dumps(item.evidence, ensure_ascii=False),
                    "notes": item.notes,
                }
            )
