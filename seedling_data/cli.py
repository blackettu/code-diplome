from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from seedling_core.run_snapshot import save_command_snapshot

from .annotations import audit_action_points, audit_cell_annotations
from .changelog import audit_dataset_changelog
from .duplicates import find_cross_split_duplicates
from .manifests import build_image_manifest
from .ontology import load_ontology, validate_dataset_class_names
from .post_action import audit_post_action_observations, parse_required_hours, summarize_post_action_file
from .registry import DEFAULT_DATASET_REGISTRY_PATH, add_dataset, validate_dataset_registry, write_dataset_summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m seedling_data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("image-manifest", help="Generate image_manifest.csv.")
    manifest.add_argument("--dataset-root", required=True)
    manifest.add_argument("--images-dir", default=None)
    manifest.add_argument("--output", default=None)
    manifest.add_argument("--split", default=None)
    manifest.add_argument("--group-regex", default=None)
    manifest.add_argument("--session-id", default=None)
    manifest.add_argument("--tray-id", default=None)

    ontology = subparsers.add_parser("validate-ontology", help="Validate ontology and optional class names.")
    ontology.add_argument("--ontology", required=True)
    ontology.add_argument("--class-names", default=None, help="Comma-separated dataset class names.")

    cell_audit = subparsers.add_parser("audit-cell-annotations", help="Audit cell_annotations.jsonl.")
    cell_audit.add_argument("--path", required=True)
    cell_audit.add_argument("--ontology", default=None)
    cell_audit.add_argument("--grid-rows", type=int, default=None)
    cell_audit.add_argument("--grid-cols", type=int, default=None)

    action_audit = subparsers.add_parser("audit-action-points", help="Audit action_points.jsonl.")
    action_audit.add_argument("--path", required=True)
    action_audit.add_argument("--ontology", default=None)
    action_audit.add_argument("--cell-annotations", default=None)
    action_audit.add_argument("--min-safe-distance-px", type=float, default=None)
    action_audit.add_argument("--max-uncertainty-px", type=float, default=None)

    duplicates = subparsers.add_parser("near-duplicates", help="Find exact/perceptual duplicates across splits.")
    duplicates.add_argument("--manifest", required=True)
    duplicates.add_argument("--max-hamming", type=int, default=4)
    duplicates.add_argument("--out", default=None)

    changelog = subparsers.add_parser("audit-changelog", help="Audit dataset changelog for a version.")
    changelog.add_argument("--dataset-root", default=None)
    changelog.add_argument("--changelog", default=None)
    changelog.add_argument("--dataset-version", required=True)

    post_action = subparsers.add_parser("audit-post-action", help="Audit post-action verification observations.")
    post_action.add_argument("--path", required=True)
    post_action.add_argument("--required-hours", default="24,48,72")

    post_action_summary = subparsers.add_parser(
        "summarize-post-action",
        help="Audit and summarize post-action verification observations.",
    )
    post_action_summary.add_argument("--path", required=True)
    post_action_summary.add_argument("--out", required=True)
    post_action_summary.add_argument("--required-hours", default="24,48,72")

    registry = subparsers.add_parser("registry", help="Dataset registry commands.")
    registry_subparsers = registry.add_subparsers(dest="registry_command", required=True)
    registry_add = registry_subparsers.add_parser("add", help="Add or update a dataset registry entry.")
    registry_add.add_argument("--registry", default=str(DEFAULT_DATASET_REGISTRY_PATH))
    registry_add.add_argument("--dataset-root", required=True)
    registry_add.add_argument("--name", required=True)
    registry_add.add_argument("--ontology-version", default="ontology_v0_1")
    registry_add.add_argument("--annotation-guide-version", default="annotation_guide_v0_1")
    registry_add.add_argument("--split-version", default=None)
    registry_add.add_argument("--calibration-version", default=None)
    registry_add.add_argument("--changelog", default=None)
    registry_add.add_argument("--status", default="draft")

    registry_validate = registry_subparsers.add_parser("validate", help="Validate registered dataset hashes.")
    registry_validate.add_argument("--registry", default=str(DEFAULT_DATASET_REGISTRY_PATH))
    registry_validate.add_argument("--dataset", default=None)

    registry_summary = registry_subparsers.add_parser("summarize", help="Write Markdown dataset summary.")
    registry_summary.add_argument("--registry", default=str(DEFAULT_DATASET_REGISTRY_PATH))
    registry_summary.add_argument("--dataset", required=True)
    registry_summary.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    if args.command == "image-manifest":
        rows = build_image_manifest(
            dataset_root=args.dataset_root,
            images_dir=args.images_dir,
            output_path=args.output,
            split=args.split,
            group_regex=args.group_regex,
            defaults=_manifest_defaults(args),
        )
        payload = {"images": len(rows), "output": args.output}
        if args.output:
            inputs = [
                args.dataset_root,
                *([_manifest_images_input(args.dataset_root, args.images_dir)] if args.images_dir else []),
            ]
            outputs = [args.output]
            metadata = {
                "split": args.split,
                "group_regex": args.group_regex,
                "session_id": args.session_id,
                "tray_id": args.tray_id,
            }
            snapshot = _write_data_snapshot(
                Path(args.output).parent,
                "seedling-data:image-manifest",
                args,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
            )
            payload["run_snapshot"] = snapshot
            payload["artifact_registry"] = _write_data_registry(
                Path(args.output).parent,
                "seedling-data:image-manifest",
                inputs=inputs,
                outputs=[*outputs, *([snapshot] if snapshot else [])],
                metadata=metadata,
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.command == "validate-ontology":
        loaded = load_ontology(args.ontology)
        class_names = [item.strip() for item in args.class_names.split(",")] if args.class_names else []
        result = validate_dataset_class_names(loaded, class_names) if class_names else {"ok": True}
        print(json.dumps({"ontology": loaded.version, **result}, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "audit-cell-annotations":
        payload = audit_cell_annotations(
            args.path,
            ontology_path=args.ontology,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "audit-action-points":
        payload = audit_action_points(
            args.path,
            ontology_path=args.ontology,
            cell_annotations_path=args.cell_annotations,
            min_safe_distance_px=args.min_safe_distance_px,
            max_uncertainty_px=args.max_uncertainty_px,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "near-duplicates":
        payload = find_cross_split_duplicates(args.manifest, max_hamming=args.max_hamming)
        if args.out:
            output = Path(args.out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
            inputs = [args.manifest]
            outputs = [output]
            metadata = {"max_hamming": args.max_hamming}
            snapshot = _write_data_snapshot(
                output.parent,
                "seedling-data:near-duplicates",
                args,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
            )
            payload["run_snapshot"] = snapshot
            payload["artifact_registry"] = _write_data_registry(
                output.parent,
                "seedling-data:near-duplicates",
                inputs=inputs,
                outputs=[*outputs, *([snapshot] if snapshot else [])],
                metadata=metadata,
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "audit-changelog":
        payload = audit_dataset_changelog(
            args.changelog,
            dataset_root=args.dataset_root,
            dataset_version=args.dataset_version,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "audit-post-action":
        payload = audit_post_action_observations(
            args.path,
            required_hours=parse_required_hours(args.required_hours),
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "summarize-post-action":
        payload = summarize_post_action_file(
            args.path,
            output_path=args.out,
            required_hours=parse_required_hours(args.required_hours),
        )
        inputs = [args.path]
        outputs = [args.out]
        metadata = {"required_hours": parse_required_hours(args.required_hours)}
        snapshot = _write_data_snapshot(
            Path(args.out).parent,
            "seedling-data:summarize-post-action",
            args,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
        payload["run_snapshot"] = snapshot
        payload["artifact_registry"] = _write_data_registry(
            Path(args.out).parent,
            "seedling-data:summarize-post-action",
            inputs=inputs,
            outputs=[*outputs, *([snapshot] if snapshot else [])],
            metadata=metadata,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "registry":
        if args.registry_command == "add":
            entry = add_dataset(
                registry_path=args.registry,
                dataset_root=args.dataset_root,
                name=args.name,
                ontology_version=args.ontology_version,
                annotation_guide_version=args.annotation_guide_version,
                split_version=args.split_version,
                calibration_version=args.calibration_version,
                changelog_path=args.changelog,
                status=args.status,
            )
            payload = {"ok": True, "registry": args.registry, "entry": entry.to_dict()}
            inputs = [args.dataset_root, *([args.changelog] if args.changelog else [])]
            outputs = [args.registry]
            metadata = {"dataset": args.name, "status": args.status}
            snapshot = _write_data_snapshot(
                Path(args.registry).parent,
                "seedling-data:registry:add",
                args,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
            )
            payload["run_snapshot"] = snapshot
            payload["artifact_registry"] = _write_data_registry(
                Path(args.registry).parent,
                "seedling-data:registry:add",
                inputs=inputs,
                outputs=[*outputs, *([snapshot] if snapshot else [])],
                metadata=metadata,
            )
        elif args.registry_command == "validate":
            payload = validate_dataset_registry(args.registry, dataset=args.dataset)
        elif args.registry_command == "summarize":
            summary = write_dataset_summary(args.registry, dataset=args.dataset, output_path=args.out)
            payload = {"ok": True, "registry": args.registry, "out": args.out, "summary": summary}
            inputs = [args.registry]
            outputs = [args.out]
            metadata = {"dataset": args.dataset}
            snapshot = _write_data_snapshot(
                Path(args.out).parent,
                "seedling-data:registry:summarize",
                args,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
            )
            payload["run_snapshot"] = snapshot
            payload["artifact_registry"] = _write_data_registry(
                Path(args.out).parent,
                "seedling-data:registry:summarize",
                inputs=inputs,
                outputs=[*outputs, *([snapshot] if snapshot else [])],
                metadata=metadata,
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _write_data_registry(
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


def _write_data_snapshot(
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


def _manifest_images_input(dataset_root: str | Path, images_dir: str | Path) -> Path:
    images_path = Path(images_dir)
    return images_path if images_path.is_absolute() else Path(dataset_root) / images_path


def _manifest_defaults(args: Any) -> dict[str, str]:
    defaults = {}
    if args.session_id:
        defaults["session_id"] = args.session_id
    if args.tray_id:
        defaults["tray_id"] = args.tray_id
    return defaults


def _artifact_type(path: str | Path) -> str:
    candidate = Path(path)
    name = candidate.name
    if candidate.is_dir():
        return "directory"
    if name == "run_snapshot.json" or name.endswith(".run_snapshot.json"):
        return "run_snapshot"
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
