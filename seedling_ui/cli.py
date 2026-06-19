from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from seedling_core.run_snapshot import save_command_snapshot
from seedling_core.registry import ComponentRegistry, ModelRegistry
from .feedback import AnnotationFeedback, append_feedback, read_feedback
from .offline_viewer import write_offline_viewer
from .report_export import write_report_export
from .replay_viewer import write_replay_viewer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m seedling_ui")
    subparsers = parser.add_subparsers(dest="command", required=True)

    offline = subparsers.add_parser("offline-viewer", help="Build static HTML viewer for predictions.json or SceneState JSON.")
    offline.add_argument("--predictions", required=True)
    offline.add_argument("--out", required=True)
    offline.add_argument("--image-name", default=None)
    offline.add_argument("--image-path", default=None)

    replay = subparsers.add_parser("replay-viewer", help="Build static HTML viewer for SimScene and ReplayLog.")
    replay.add_argument("--scene", required=True)
    replay.add_argument("--replay", required=True)
    replay.add_argument("--out", required=True)

    report = subparsers.add_parser("report-export", help="Export tray/episode report as Markdown or HTML.")
    report.add_argument("--scene", required=True)
    report.add_argument("--out", required=True)
    report.add_argument("--replay", default=None)
    report.add_argument("--title", default=None)

    feedback = subparsers.add_parser("feedback", help="Append or summarize annotation feedback JSONL.")
    feedback.add_argument("--feedback", required=True, help="Path to feedback JSONL.")
    feedback.add_argument("--image-id", default=None)
    feedback.add_argument("--error-type", default=None)
    feedback.add_argument("--comment", default="")
    feedback.add_argument("--target-id", default=None)
    feedback.add_argument("--object-id", default=None)
    feedback.add_argument("--cell-id", default=None)
    feedback.add_argument("--operator-id", default=None)
    feedback.add_argument("--priority", default=None)
    feedback.add_argument("--proposed-correction-json", default=None)
    feedback.add_argument("--summary", action="store_true")
    feedback.add_argument("--annotation-tasks-out", default=None)

    selector = subparsers.add_parser("selector", help="Select model/policy entry from component registry.")
    selector.add_argument("--registry", required=True)
    selector.add_argument("--kind", default=None)
    selector.add_argument("--name", default=None)
    selector.add_argument("--entry-id", default=None)
    selector.add_argument("--status", default=None)
    selector.add_argument("--tag", action="append", default=[])
    selector.add_argument("--validate", action="store_true")

    model_registry = subparsers.add_parser("model-registry", help="Validate or select typed model registry records.")
    model_registry.add_argument("--registry", required=True)
    model_registry.add_argument("--validate", action="store_true")
    model_registry.add_argument("--model-id", default=None)
    model_registry.add_argument("--model-type", default=None)
    model_registry.add_argument("--status", default=None)
    model_registry.add_argument("--safety-level", default=None)
    model_registry.add_argument("--tag", action="append", default=[])

    args = parser.parse_args(argv)
    if args.command == "offline-viewer":
        write_offline_viewer(args.predictions, args.out, image_name=args.image_name, image_path=args.image_path)
        inputs = [args.predictions, *([args.image_path] if args.image_path else [])]
        outputs = [args.out]
        snapshot = _write_ui_snapshot(
            Path(args.out).parent,
            "seedling-ui:offline-viewer",
            args,
            inputs=inputs,
            outputs=outputs,
            metadata={"image_name": args.image_name},
        )
        payload = {
            "ok": True,
            "out": args.out,
            "run_snapshot": snapshot,
            "artifact_registry": _write_ui_registry(
                Path(args.out).parent,
                "seedling-ui:offline-viewer",
                inputs=inputs,
                outputs=[*outputs, *([snapshot] if snapshot else [])],
                metadata={"image_name": args.image_name},
            ),
        }
    elif args.command == "replay-viewer":
        write_replay_viewer(args.scene, args.replay, args.out)
        inputs = [args.scene, args.replay]
        outputs = [args.out]
        snapshot = _write_ui_snapshot(
            Path(args.out).parent,
            "seedling-ui:replay-viewer",
            args,
            inputs=inputs,
            outputs=outputs,
        )
        payload = {
            "ok": True,
            "out": args.out,
            "run_snapshot": snapshot,
            "artifact_registry": _write_ui_registry(
                Path(args.out).parent,
                "seedling-ui:replay-viewer",
                inputs=inputs,
                outputs=[*outputs, *([snapshot] if snapshot else [])],
            ),
        }
    elif args.command == "report-export":
        report = write_report_export(args.scene, args.out, replay_path=args.replay, title=args.title)
        inputs = [args.scene, *([args.replay] if args.replay else [])]
        outputs = [args.out]
        metadata = {
            "title": args.title,
            "scene_id": report["scene_id"],
            "scene_kind": report["scene_kind"],
            "targets": len(report["targets"]),
            "replay_steps": len(report["replay_events"]),
        }
        snapshot = _write_ui_snapshot(
            Path(args.out).parent,
            "seedling-ui:report-export",
            args,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
        payload = {
            "ok": True,
            "out": args.out,
            "scene_id": report["scene_id"],
            "scene_kind": report["scene_kind"],
            "targets": len(report["targets"]),
            "replay_steps": len(report["replay_events"]),
            "run_snapshot": snapshot,
            "artifact_registry": _write_ui_registry(
                Path(args.out).parent,
                "seedling-ui:report-export",
                inputs=inputs,
                outputs=[*outputs, *([snapshot] if snapshot else [])],
                metadata=metadata,
            ),
        }
    elif args.command == "feedback":
        if args.summary:
            rows = read_feedback(args.feedback)
            counts: dict[str, int] = {}
            for row in rows:
                counts[row.error_type] = counts.get(row.error_type, 0) + 1
            payload = {"ok": True, "feedback": args.feedback, "rows": len(rows), "error_type_counts": counts}
        else:
            if not args.image_id or not args.error_type:
                raise SystemExit("--image-id and --error-type are required unless --summary is used")
            row = AnnotationFeedback(
                image_id=args.image_id,
                error_type=args.error_type,
                comment=args.comment,
                target_id=args.target_id,
                object_id=args.object_id,
                cell_id=args.cell_id,
                operator_id=args.operator_id,
                priority=args.priority,
                proposed_correction=_json_object_arg(args.proposed_correction_json, "--proposed-correction-json"),
            )
            append_feedback(args.feedback, row)
            metadata = {"image_id": args.image_id, "error_type": args.error_type}
            snapshot = _write_ui_snapshot(
                Path(args.feedback).parent,
                "seedling-ui:feedback:add",
                args,
                inputs=[],
                outputs=[args.feedback],
                metadata=metadata,
            )
            payload = {
                "ok": True,
                "feedback": args.feedback,
                "row": row.to_dict(),
                "run_snapshot": snapshot,
                "artifact_registry": _write_ui_registry(
                    Path(args.feedback).parent,
                    "seedling-ui:feedback:add",
                    inputs=[],
                    outputs=[args.feedback, *([snapshot] if snapshot else [])],
                    metadata=metadata,
                ),
            }
        if args.annotation_tasks_out:
            from .feedback import write_annotation_tasks_from_feedback

            tasks = write_annotation_tasks_from_feedback(args.feedback, args.annotation_tasks_out)
            payload["annotation_tasks_out"] = args.annotation_tasks_out
            payload["annotation_tasks"] = len(tasks)
            snapshot = _write_ui_snapshot(
                Path(args.annotation_tasks_out).parent,
                "seedling-ui:feedback:annotation-tasks",
                args,
                inputs=[args.feedback],
                outputs=[args.annotation_tasks_out],
                metadata={"tasks": len(tasks), "summary": args.summary},
            )
            payload["run_snapshot"] = snapshot
            payload["artifact_registry"] = _write_ui_registry(
                Path(args.annotation_tasks_out).parent,
                "seedling-ui:feedback:annotation-tasks",
                inputs=[args.feedback],
                outputs=[args.annotation_tasks_out, *([snapshot] if snapshot else [])],
                metadata={"tasks": len(tasks), "summary": args.summary},
            )
    elif args.command == "selector":
        registry = ComponentRegistry.from_file(args.registry)
        validation = registry.validate()
        if args.validate:
            payload = {"registry": args.registry, **validation}
        else:
            selected = registry.select(
                kind=args.kind,
                name=args.name,
                entry_id=args.entry_id,
                status=args.status,
                tags=args.tag,
            )
            payload = {"ok": validation["ok"], "registry": args.registry, "entry": selected.to_dict(), "validation": validation}
    elif args.command == "model-registry":
        registry = ModelRegistry.from_file(args.registry)
        validation = registry.validate()
        if args.validate:
            payload = {"registry": args.registry, **validation}
        else:
            selected = registry.select(
                model_id=args.model_id,
                model_type=args.model_type,
                status=args.status,
                safety_level=args.safety_level,
                tags=args.tag,
            )
            payload = {
                "ok": validation["ok"],
                "registry": args.registry,
                "model": selected.to_dict(),
                "validation": validation,
            }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _write_ui_registry(
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


def _write_ui_snapshot(
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


def _json_object_arg(value: str | None, name: str) -> dict[str, Any]:
    if value in {None, ""}:
        return {}
    data = json.loads(value)
    if not isinstance(data, dict):
        raise SystemExit(f"{name} must be a JSON object")
    return data


def _artifact_type(path: str | Path) -> str:
    candidate = Path(path)
    name = candidate.name
    suffix = candidate.suffix.lower()
    if candidate.is_dir():
        return "directory"
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}:
        return "image"
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
