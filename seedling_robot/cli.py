from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from seedling_core.run_snapshot import save_command_snapshot

from .dry_run_runner import run_dry_run_plan
from .dry_run_test import run_dry_run_control_points
from .hil_runner import run_hil_pointer_plan
from .hil_safety import HardwareInLoopReview, validate_hil_review
from .motion_replay import run_motion_replay


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m seedling_robot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dry_run = subparsers.add_parser("dry-run-test", help="Run safe dry-run pointer moves to control points.")
    dry_run.add_argument("--points", required=True)
    dry_run.add_argument("--out", required=True)
    dry_run.add_argument("--command-log", default=None)
    dry_run.add_argument("--tolerance-mm", type=float, default=1.0)

    dry_run_plan = subparsers.add_parser("dry-run-plan", help="Run a safe dry-run pointer ActionPlan.")
    dry_run_plan.add_argument("--scene", required=True)
    dry_run_plan.add_argument("--plan", required=True)
    dry_run_plan.add_argument("--out", required=True)
    dry_run_plan.add_argument("--command-log", default=None)
    dry_run_plan.add_argument("--replay", default=None)
    dry_run_plan.add_argument("--operator-confirmed", action="store_true")
    dry_run_plan.add_argument("--interlock-ok", action="store_true")
    dry_run_plan.add_argument("--limit-switch-fault", action="store_true")

    motion_replay = subparsers.add_parser("motion-replay", help="Replay MotionCommand JSON in the simulator.")
    motion_replay.add_argument("--commands", required=True)
    motion_replay.add_argument("--out", required=True)
    motion_replay.add_argument("--continue-on-failure", action="store_true")

    hil = subparsers.add_parser("validate-hil-review", help="Validate hardware-in-loop safety review artifact.")
    hil.add_argument("--review", required=True)
    hil.add_argument("--out", default=None)
    hil.add_argument(
        "--hil-pointer",
        action="store_true",
        help="Validate as a hardware_in_loop_pointer / pointer_only execution preflight.",
    )

    hil_run = subparsers.add_parser("hil-pointer-run", help="Run a reviewed HIL pointer ActionPlan.")
    hil_run.add_argument("--scene", required=True)
    hil_run.add_argument("--plan", required=True)
    hil_run.add_argument("--review", required=True)
    hil_run.add_argument("--port", required=True)
    hil_run.add_argument("--out", required=True)
    hil_run.add_argument("--command-log", default=None)
    hil_run.add_argument("--replay", default=None)
    hil_run.add_argument("--allow-hardware", action="store_true")
    hil_run.add_argument("--operator-confirmed", action="store_true")
    hil_run.add_argument("--baudrate", type=int, default=115200)
    hil_run.add_argument("--timeout-s", type=float, default=1.0)
    hil_run.add_argument("--interlock-ok", action="store_true")
    hil_run.add_argument("--limit-switch-fault", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "dry-run-test":
        payload = run_dry_run_control_points(
            args.points,
            args.out,
            command_log_path=args.command_log,
            tolerance_mm=args.tolerance_mm,
        )
        inputs = [args.points]
        outputs = [args.out, *([args.command_log] if args.command_log else [])]
        metadata = {"tolerance_mm": args.tolerance_mm}
        snapshot = _write_robot_snapshot(
            Path(args.out).parent,
            "seedling-robot:dry-run-test",
            args,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
        payload["run_snapshot"] = snapshot
        payload["artifact_registry"] = _write_robot_registry(
            Path(args.out).parent,
            "seedling-robot:dry-run-test",
            inputs=inputs,
            outputs=[*outputs, *([snapshot] if snapshot else [])],
            metadata=metadata,
        )
        print(json.dumps({"ok": payload["ok"], "out": args.out, "points": len(payload["points"]), "run_snapshot": snapshot, "artifact_registry": payload["artifact_registry"]}, indent=2))
    elif args.command == "dry-run-plan":
        payload = run_dry_run_plan(
            scene_path=args.scene,
            plan_path=args.plan,
            out_path=args.out,
            command_log_path=args.command_log,
            replay_path=args.replay,
            operator_confirmed=args.operator_confirmed,
            interlock_ok=args.interlock_ok,
            limit_switch_ok=not args.limit_switch_fault,
        )
        inputs = [args.scene, args.plan]
        outputs = [args.out, *([args.command_log] if args.command_log else []), *([args.replay] if args.replay else [])]
        metadata = {
            "operator_confirmed": args.operator_confirmed,
            "interlock_ok": args.interlock_ok,
            "limit_switch_ok": not args.limit_switch_fault,
        }
        snapshot = _write_robot_snapshot(
            Path(args.out).parent,
            "seedling-robot:dry-run-plan",
            args,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
        payload["run_snapshot"] = snapshot
        payload["artifact_registry"] = _write_robot_registry(
            Path(args.out).parent,
            "seedling-robot:dry-run-plan",
            inputs=inputs,
            outputs=[*outputs, *([snapshot] if snapshot else [])],
            metadata=metadata,
        )
        print(
            json.dumps(
                {key: payload[key] for key in ["ok", "mode", "scene_id", "plan_id", "commands", "executed", "blocked", "skipped", "aborted", "run_snapshot", "artifact_registry"]},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "motion-replay":
        payload = run_motion_replay(
            args.commands,
            args.out,
            stop_on_failure=not args.continue_on_failure,
        )
        inputs = [args.commands]
        outputs = [args.out]
        metadata = {
            "mode": payload["mode"],
            "commands": payload["commands"],
            "stop_on_failure": payload["stop_on_failure"],
        }
        snapshot = _write_robot_snapshot(
            Path(args.out).parent,
            "seedling-robot:motion-replay",
            args,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
        payload["run_snapshot"] = snapshot
        payload["artifact_registry"] = _write_robot_registry(
            Path(args.out).parent,
            "seedling-robot:motion-replay",
            inputs=inputs,
            outputs=[*outputs, *([snapshot] if snapshot else [])],
            metadata=metadata,
        )
        print(
            json.dumps(
                {
                    "ok": payload["ok"],
                    "out": args.out,
                    "commands": payload["commands"],
                    "executed": payload["executed"],
                    "run_snapshot": snapshot,
                    "artifact_registry": payload["artifact_registry"],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "validate-hil-review":
        review = HardwareInLoopReview.from_json(args.review)
        payload = validate_hil_review(
            review,
            required_mode="hardware_in_loop_pointer" if args.hil_pointer else None,
            required_tool_profile="pointer_only" if args.hil_pointer else None,
            require_positioning_error=args.hil_pointer,
        )
        if args.out:
            output = Path(args.out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            inputs = [args.review]
            outputs = [output]
            metadata = {"review_id": review.review_id}
            snapshot = _write_robot_snapshot(
                output.parent,
                "seedling-robot:validate-hil-review",
                args,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
            )
            payload["run_snapshot"] = snapshot
            payload["artifact_registry"] = _write_robot_registry(
                output.parent,
                "seedling-robot:validate-hil-review",
                inputs=inputs,
                outputs=[*outputs, *([snapshot] if snapshot else [])],
                metadata=metadata,
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "hil-pointer-run":
        payload = run_hil_pointer_plan(
            scene_path=args.scene,
            plan_path=args.plan,
            review_path=args.review,
            port=args.port,
            out_path=args.out,
            command_log_path=args.command_log,
            replay_path=args.replay,
            allow_hardware=args.allow_hardware,
            operator_confirmed=args.operator_confirmed,
            baudrate=args.baudrate,
            timeout_s=args.timeout_s,
            interlock_ok=args.interlock_ok,
            limit_switch_ok=not args.limit_switch_fault,
        )
        inputs = [args.scene, args.plan, args.review]
        outputs = [args.out, *([args.command_log] if args.command_log else []), *([args.replay] if args.replay else [])]
        metadata = {
            "port": args.port,
            "allow_hardware": args.allow_hardware,
            "interlock_ok": args.interlock_ok,
            "limit_switch_ok": not args.limit_switch_fault,
        }
        snapshot = _write_robot_snapshot(
            Path(args.out).parent,
            "seedling-robot:hil-pointer-run",
            args,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
        payload["run_snapshot"] = snapshot
        payload["artifact_registry"] = _write_robot_registry(
            Path(args.out).parent,
            "seedling-robot:hil-pointer-run",
            inputs=inputs,
            outputs=[*outputs, *([snapshot] if snapshot else [])],
            metadata=metadata,
        )
        print(json.dumps({key: payload[key] for key in ["ok", "mode", "scene_id", "plan_id", "commands", "executed", "blocked", "skipped", "aborted", "run_snapshot", "artifact_registry"]}, ensure_ascii=False, indent=2, sort_keys=True))


def _write_robot_registry(
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


def _write_robot_snapshot(
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
    if name.endswith(".jsonl"):
        return name.removesuffix(".jsonl")
    return Path(path).suffix.lstrip(".") or "artifact"


def _registry_run_id(run_dir: str | Path, command: str, outputs: list[str | Path]) -> str:
    directory = Path(run_dir).name or "run"
    output_stem = Path(outputs[0]).stem if outputs else "stdout"
    command_slug = "".join(char if char.isalnum() else "_" for char in command).strip("_")
    return f"{directory}_{command_slug}_{output_stem}"
