from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from seedling_core.run_snapshot import save_command_snapshot

from .error_budget import write_error_budget_report
from .estimation import estimate_calibration_artifact, load_calibration_config, write_error_map
from .intrinsics import (
    CameraIntrinsicsArtifact,
    estimate_camera_intrinsics,
    load_intrinsics_observations,
    validate_camera_intrinsics,
)
from .schemas import CalibrationArtifact
from .validator import CalibrationValidator


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m seedling_calibration")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate a calibration artifact.")
    validate.add_argument("--calibration", required=True)
    validate.add_argument("--tray-type", default=None)
    validate.add_argument("--max-p95-mm", type=float, default=2.0)
    validate.add_argument("--out", default=None)

    estimate = subparsers.add_parser("estimate", help="Estimate calibration artifact from control points.")
    estimate.add_argument("--config", required=True)
    estimate.add_argument("--out", required=True)
    estimate.add_argument("--calibration-id", default="calibration_v0")
    estimate.add_argument("--created-at", default=None)
    estimate.add_argument("--tool-offset-mm", default="0,0,0")

    error_map = subparsers.add_parser("error-map", help="Write calibration residual map for control points.")
    error_map.add_argument("--config", required=True)
    error_map.add_argument("--calibration", required=True)
    error_map.add_argument("--out", required=True)

    error_budget = subparsers.add_parser("error-budget", help="Write RSS error budget from measured component errors.")
    error_budget.add_argument("--components", required=True, help="JSON object or {components_mm:{...}} with error components in millimetres.")
    error_budget.add_argument("--out", required=True)
    error_budget.add_argument("--calibration", default=None, help="Optional calibration artifact used to fill e_calibration from p95.")
    error_budget.add_argument("--max-total-mm", type=float, default=None)

    estimate_intrinsics = subparsers.add_parser("estimate-intrinsics", help="Estimate camera intrinsics from chessboard/aruco/fiducial observations.")
    estimate_intrinsics.add_argument("--observations", required=True)
    estimate_intrinsics.add_argument("--camera-id", required=True)
    estimate_intrinsics.add_argument("--out", required=True)

    validate_intrinsics = subparsers.add_parser("validate-intrinsics", help="Validate camera intrinsics artifact.")
    validate_intrinsics.add_argument("--intrinsics", required=True)
    validate_intrinsics.add_argument("--max-reprojection-error-px", type=float, default=None)
    validate_intrinsics.add_argument("--out", default=None)

    args = parser.parse_args(argv)
    if args.command == "validate":
        artifact = CalibrationArtifact.from_json(args.calibration)
        result = CalibrationValidator(
            expected_tray_type=args.tray_type,
            max_error_p95_mm=args.max_p95_mm,
        ).validate(artifact)
        payload = result.to_dict()
        if args.out:
            output = Path(args.out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            inputs = [args.calibration]
            outputs = [output]
            metadata = {"tray_type": args.tray_type, "max_p95_mm": args.max_p95_mm}
            snapshot = _write_calibration_snapshot(
                output.parent,
                "seedling-calibration:validate",
                args,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
            )
            payload["run_snapshot"] = snapshot
            payload["artifact_registry"] = _write_calibration_registry(
                output.parent,
                "seedling-calibration:validate",
                inputs=inputs,
                outputs=[*outputs, *([snapshot] if snapshot else [])],
                metadata=metadata,
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "estimate":
        config = load_calibration_config(args.config)
        artifact = estimate_calibration_artifact(
            config,
            calibration_id=args.calibration_id,
            created_at=args.created_at,
            tool_offset_mm=_csv_floats(args.tool_offset_mm, 3),
        )
        artifact.to_json(args.out)
        output = Path(args.out)
        inputs = [args.config]
        outputs = [output]
        metadata = {"calibration_id": args.calibration_id}
        snapshot = _write_calibration_snapshot(
            output.parent,
            "seedling-calibration:estimate",
            args,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
        registry_path = _write_calibration_registry(
            output.parent,
            "seedling-calibration:estimate",
            inputs=inputs,
            outputs=[*outputs, *([snapshot] if snapshot else [])],
            metadata=metadata,
        )
        payload = {"ok": True, "out": args.out, "artifact": artifact.to_dict(), "run_snapshot": snapshot, "artifact_registry": registry_path}
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "error-map":
        payload = write_error_map(args.config, args.calibration, args.out)
        output = Path(args.out)
        inputs = [args.config, args.calibration]
        outputs = [output]
        metadata = {"format": output.suffix.lstrip(".").lower() or "json"}
        snapshot = _write_calibration_snapshot(
            output.parent,
            "seedling-calibration:error-map",
            args,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
        registry_path = _write_calibration_registry(
            output.parent,
            "seedling-calibration:error-map",
            inputs=inputs,
            outputs=[*outputs, *([snapshot] if snapshot else [])],
            metadata=metadata,
        )
        print(json.dumps({"ok": True, "out": args.out, "run_snapshot": snapshot, "artifact_registry": registry_path, **payload}, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "error-budget":
        report = write_error_budget_report(
            args.components,
            args.out,
            calibration_path=args.calibration,
            max_total_mm=args.max_total_mm,
        )
        output = Path(args.out)
        inputs = [args.components, *([args.calibration] if args.calibration else [])]
        outputs = [output]
        metadata = {"max_total_mm": args.max_total_mm}
        snapshot = _write_calibration_snapshot(
            output.parent,
            "seedling-calibration:error-budget",
            args,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
        registry_path = _write_calibration_registry(
            output.parent,
            "seedling-calibration:error-budget",
            inputs=inputs,
            outputs=[*outputs, *([snapshot] if snapshot else [])],
            metadata=metadata,
        )
        payload = {"out": args.out, "run_snapshot": snapshot, "artifact_registry": registry_path, **report.to_dict()}
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "estimate-intrinsics":
        observations = load_intrinsics_observations(args.observations)
        artifact = estimate_camera_intrinsics(observations, camera_id=args.camera_id)
        artifact.to_json(args.out)
        output = Path(args.out)
        inputs = [args.observations]
        outputs = [output]
        metadata = {"camera_id": args.camera_id}
        snapshot = _write_calibration_snapshot(
            output.parent,
            "seedling-calibration:estimate-intrinsics",
            args,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
        registry_path = _write_calibration_registry(
            output.parent,
            "seedling-calibration:estimate-intrinsics",
            inputs=inputs,
            outputs=[*outputs, *([snapshot] if snapshot else [])],
            metadata=metadata,
        )
        payload = {"ok": True, "out": args.out, "artifact": artifact.to_dict(), "run_snapshot": snapshot, "artifact_registry": registry_path}
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "validate-intrinsics":
        artifact = CameraIntrinsicsArtifact.from_json(args.intrinsics)
        payload = validate_camera_intrinsics(artifact, max_reprojection_error_px=args.max_reprojection_error_px)
        if args.out:
            output = Path(args.out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            inputs = [args.intrinsics]
            outputs = [output]
            metadata = {"max_reprojection_error_px": args.max_reprojection_error_px}
            snapshot = _write_calibration_snapshot(
                output.parent,
                "seedling-calibration:validate-intrinsics",
                args,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
            )
            payload["run_snapshot"] = snapshot
            payload["artifact_registry"] = _write_calibration_registry(
                output.parent,
                "seedling-calibration:validate-intrinsics",
                inputs=inputs,
                outputs=[*outputs, *([snapshot] if snapshot else [])],
                metadata=metadata,
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _csv_floats(value: str, expected_length: int) -> list[float]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != expected_length:
        raise ValueError(f"Expected {expected_length} comma-separated floats")
    return [float(part) for part in parts]


def _write_calibration_registry(
    run_dir: str | Path,
    command: str,
    *,
    inputs: list[str | Path],
    outputs: list[str | Path],
    metadata: dict[str, Any] | None = None,
) -> str | None:
    try:
        from seedling_reports.registry import ArtifactRecord, write_run_registry_records

        artifacts = [
            *(ArtifactRecord.from_path(path, artifact_type=_artifact_type(path), role="input", command=command) for path in inputs),
            *(ArtifactRecord.from_path(path, artifact_type=_artifact_type(path), role="output", command=command) for path in outputs),
        ]
        write_run_registry_records(
            run_dir,
            command,
            artifacts,
            metadata=metadata,
            run_id=_registry_run_id(run_dir, command, outputs),
        )
        return str(Path(run_dir) / "artifact_registry.json")
    except Exception:
        return None


def _write_calibration_snapshot(
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
            snapshot_name=_snapshot_name(command, outputs),
        )
    except Exception:
        return None


def _snapshot_name(command: str, outputs: list[str | Path]) -> str:
    stem = Path(outputs[0]).stem if outputs else "".join(char if char.isalnum() else "_" for char in command).strip("_")
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in stem).strip("_")
    return f"{safe or 'command'}.run_snapshot.json"


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
    name = Path(path).name
    if name == "run_snapshot.json" or name.endswith(".run_snapshot.json"):
        return "run_snapshot"
    if name.endswith(".json"):
        return name.removesuffix(".json")
    if name.endswith(".html"):
        return "html_report"
    if name.endswith(".yaml") or name.endswith(".yml"):
        return "config"
    return Path(path).suffix.lstrip(".") or "artifact"


def _registry_run_id(run_dir: str | Path, command: str, outputs: list[str | Path]) -> str:
    directory = Path(run_dir).name or "run"
    output_stem = Path(outputs[0]).stem if outputs else "stdout"
    command_slug = "".join(char if char.isalnum() else "_" for char in command).strip("_")
    return f"{directory}_{command_slug}_{output_stem}"
