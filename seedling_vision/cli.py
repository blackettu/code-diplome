from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from seedling_core.run_snapshot import save_command_snapshot
from seedling_core.schemas import SceneState

from .adapters import RecordedPredictionDetector
from .batch import batch_predict
from .migration import (
    legacy_prediction_to_detection_result,
    legacy_prediction_to_scene,
    legacy_predictions_to_scenes,
    load_legacy_prediction_records,
)
from .overlays import write_detection_overlay
from .postprocess import filter_and_merge_containers
from .uncertainty import annotate_result_uncertainty


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m seedling_vision")
    subparsers = parser.add_subparsers(dest="command", required=True)

    migrate = subparsers.add_parser("migrate-predictions", help="Convert legacy predictions.json to SceneState JSON.")
    migrate.add_argument("--predictions", required=True)
    migrate.add_argument("--out", required=True)
    migrate.add_argument("--dataset-version", required=True)
    migrate.add_argument("--ontology-version", default="ontology_v0_1")
    migrate.add_argument("--grid-rows", type=int, default=11)
    migrate.add_argument("--grid-cols", type=int, default=11)

    batch = subparsers.add_parser("batch-recorded", help="Run recorded-prediction adapter over image keys/paths.")
    batch.add_argument("--predictions", required=True)
    batch.add_argument("--images", nargs="+", required=True)
    batch.add_argument("--out", required=True)
    batch.add_argument("--uncertainty", action="store_true")
    batch.add_argument("--progress-log", default=None, help="Optional JSONL progress log for batch inference.")

    overlay = subparsers.add_parser("overlay", help="Draw detection overlay from recorded predictions.")
    overlay.add_argument("--predictions", required=True)
    overlay.add_argument("--image", required=True)
    overlay.add_argument("--out", required=True)
    overlay.add_argument("--uncertainty", action="store_true")
    overlay.add_argument("--include-scene", action="store_true", help="Build and draw grid cells and targets from predictions.")
    overlay.add_argument("--scene", help="Optional SceneState JSON/JSONL with cells and targets to draw.")
    overlay.add_argument("--errors", help="Optional JSON with errors, critical events or error_taxonomy samples to draw.")
    overlay.add_argument("--dataset-version", default="zks_v0_1")
    overlay.add_argument("--ontology-version", default="ontology_v0_1")
    overlay.add_argument("--grid-rows", type=int, default=11)
    overlay.add_argument("--grid-cols", type=int, default=11)

    postprocess = subparsers.add_parser("postprocess-containers", help="Postprocess container detections.")
    postprocess.add_argument("--predictions", required=True)
    postprocess.add_argument("--out", required=True)
    postprocess.add_argument("--min-area", type=float, default=10000.0)
    postprocess.add_argument("--merge-distance", type=float, default=50.0)

    args = parser.parse_args(argv)
    if args.command == "migrate-predictions":
        scenes = legacy_predictions_to_scenes(
            predictions_path=args.predictions,
            dataset_version=args.dataset_version,
            ontology_version=args.ontology_version,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            output_path=args.out,
        )
        output = Path(args.out)
        inputs = [args.predictions]
        outputs = [output]
        metadata = {
            "dataset_version": args.dataset_version,
            "ontology_version": args.ontology_version,
            "grid_rows": args.grid_rows,
            "grid_cols": args.grid_cols,
            "scenes": len(scenes),
        }
        snapshot = _write_vision_snapshot(
            output.parent,
            "seedling-vision:migrate-predictions",
            args,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
        payload = {
            "ok": True,
            "predictions": args.predictions,
            "out": args.out,
            "scenes": len(scenes),
            "run_snapshot": snapshot,
            "artifact_registry": _write_vision_registry(
                output.parent,
                "seedling-vision:migrate-predictions",
                inputs=inputs,
                outputs=[*outputs, *([snapshot] if snapshot else [])],
                metadata=metadata,
            ),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "batch-recorded":
        detector = RecordedPredictionDetector()
        detector.load(args.predictions)
        progress_handle = _open_progress_log(args.progress_log)
        try:
            total_images = len(args.images)

            def progress(index: int, image: str | Path) -> None:
                if progress_handle:
                    progress_handle.write(
                        json.dumps(
                            {
                                "event": "predict_start",
                                "index": index,
                                "total": total_images,
                                "image": str(image),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    progress_handle.flush()

            results = batch_predict(detector, args.images, progress=progress if progress_handle else None)
            if progress_handle:
                progress_handle.write(
                    json.dumps(
                        {"event": "batch_complete", "images": len(results), "out": args.out},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
        finally:
            if progress_handle:
                progress_handle.close()
        if args.uncertainty:
            results = [annotate_result_uncertainty(result) for result in results]
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"images": [result.to_dict() for result in results]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        inputs = [args.predictions, *args.images]
        outputs = [output, *([args.progress_log] if args.progress_log else [])]
        metadata = {"images": len(results), "uncertainty": args.uncertainty}
        snapshot = _write_vision_snapshot(
            output.parent,
            "seedling-vision:batch-recorded",
            args,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
        registry_path = _write_vision_registry(
            output.parent,
            "seedling-vision:batch-recorded",
            inputs=inputs,
            outputs=[*outputs, *([snapshot] if snapshot else [])],
            metadata=metadata,
        )
        payload = {
            "ok": True,
            "predictions": args.predictions,
            "out": args.out,
            "images": len(results),
            "progress_log": args.progress_log,
            "run_snapshot": snapshot,
            "artifact_registry": registry_path,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "overlay":
        detector = RecordedPredictionDetector()
        detector.load(args.predictions)
        result = detector.predict(args.image)
        if args.uncertainty:
            result = annotate_result_uncertainty(result)
        scene = None
        if args.scene:
            scene = _load_scene(args.scene, args.image)
        elif args.include_scene:
            scene = legacy_prediction_to_scene(
                _load_legacy_record(args.predictions, args.image),
                dataset_version=args.dataset_version,
                ontology_version=args.ontology_version,
                grid_rows=args.grid_rows,
                grid_cols=args.grid_cols,
            )
        errors = _load_overlay_errors(args.errors, image=args.image) if args.errors else []
        cells = scene.cells if scene else None
        targets = scene.targets if scene else None
        write_detection_overlay(args.image, result, args.out, cells=cells, targets=targets, errors=errors)
        inputs = [
            args.predictions,
            args.image,
            *([args.scene] if args.scene else []),
            *([args.errors] if args.errors else []),
        ]
        outputs = [args.out]
        metadata = {
            "uncertainty": args.uncertainty,
            "include_scene": args.include_scene,
            "dataset_version": args.dataset_version,
            "ontology_version": args.ontology_version,
            "grid_rows": args.grid_rows,
            "grid_cols": args.grid_cols,
            "detections": len(result.detections),
            "cells": len(cells or []),
            "targets": len(targets or []),
            "errors": len(errors),
        }
        snapshot = _write_vision_snapshot(
            Path(args.out).parent,
            "seedling-vision:overlay",
            args,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
        payload = {
            "ok": True,
            "predictions": args.predictions,
            "out": args.out,
            "detections": len(result.detections),
            "cells": len(cells or []),
            "targets": len(targets or []),
            "errors": len(errors),
            "run_snapshot": snapshot,
            "artifact_registry": _write_vision_registry(
                Path(args.out).parent,
                "seedling-vision:overlay",
                inputs=inputs,
                outputs=[*outputs, *([snapshot] if snapshot else [])],
                metadata=metadata,
            ),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "postprocess-containers":
        records = load_legacy_prediction_records(args.predictions)
        images = []
        for record in records:
            result = legacy_prediction_to_detection_result(record)
            containers = filter_and_merge_containers(
                result.detections,
                min_area_px2=args.min_area,
                merge_distance_px=args.merge_distance,
            )
            images.append(
                {
                    "image": Path(result.image_ref).name,
                    "image_ref": result.image_ref,
                    "width": result.image_size_px[0],
                    "height": result.image_size_px[1],
                    "detections": [item.to_dict() for item in result.detections],
                    "containers": [item.to_dict() for item in containers],
                }
            )
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"images": images}, ensure_ascii=False, indent=2), encoding="utf-8")
        inputs = [args.predictions]
        outputs = [output]
        metadata = {"images": len(images), "min_area": args.min_area, "merge_distance": args.merge_distance}
        snapshot = _write_vision_snapshot(
            output.parent,
            "seedling-vision:postprocess-containers",
            args,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
        payload = {
            "ok": True,
            "predictions": args.predictions,
            "out": args.out,
            "images": len(images),
            "run_snapshot": snapshot,
            "artifact_registry": _write_vision_registry(
                output.parent,
                "seedling-vision:postprocess-containers",
                inputs=inputs,
                outputs=[*outputs, *([snapshot] if snapshot else [])],
                metadata=metadata,
            ),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _write_vision_registry(
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


def _write_vision_snapshot(
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
    candidate = Path(path)
    name = candidate.name
    suffix = candidate.suffix.lower()
    if candidate.is_dir():
        return "directory"
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}:
        return "image"
    if name == "run_snapshot.json" or name.endswith(".run_snapshot.json"):
        return "run_snapshot"
    if name.endswith(".json"):
        return name.removesuffix(".json")
    if name.endswith(".jsonl"):
        return name.removesuffix(".jsonl")
    if name.endswith(".csv"):
        return name.removesuffix(".csv")
    if name.endswith(".yaml") or name.endswith(".yml"):
        return "config"
    return candidate.suffix.lstrip(".") or "artifact"


def _registry_run_id(run_dir: str | Path, command: str, outputs: list[str | Path]) -> str:
    directory = Path(run_dir).name or "run"
    output_stem = Path(outputs[0]).stem if outputs else "stdout"
    command_slug = "".join(char if char.isalnum() else "_" for char in command).strip("_")
    return f"{directory}_{command_slug}_{output_stem}"


def _open_progress_log(progress_log: str | Path | None):
    if not progress_log:
        return None
    path = Path(progress_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def _load_legacy_record(predictions_path: str | Path, image: str | Path) -> dict[str, Any]:
    image_name = Path(image).name
    records = load_legacy_prediction_records(predictions_path)
    for record in records:
        names = [record.get("image"), Path(str(record.get("path"))).name if record.get("path") else None]
        if image_name in {str(name) for name in names if name}:
            return record
    raise KeyError(f"Image {image_name!r} not found in {predictions_path}")


def _load_scene(scene_path: str | Path, image: str | Path) -> SceneState:
    path = Path(scene_path)
    if path.suffix.lower() == ".jsonl":
        scenes = [
            SceneState.from_dict(json.loads(line))
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    else:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict) and isinstance(data.get("scenes"), list):
            scenes = [SceneState.from_dict(item) for item in data["scenes"] if isinstance(item, dict)]
        elif isinstance(data, dict):
            scenes = [SceneState.from_dict(data)]
        else:
            raise ValueError(f"Scene file must be SceneState JSON or scenes list: {scene_path}")
    if not scenes:
        raise ValueError(f"Scene file contains no scenes: {scene_path}")
    image_name = Path(image).name
    for scene in scenes:
        if Path(scene.image_ref).name == image_name or scene.scene_id == image_name or scene.scene_id == Path(image_name).stem:
            return scene
    return scenes[0]


def _load_overlay_errors(errors_path: str | Path, image: str | Path | None = None) -> list[dict[str, Any]]:
    data = json.loads(Path(errors_path).read_text(encoding="utf-8-sig"))
    errors = _extract_errors(data)
    if image is None:
        return errors
    image_name = Path(image).name
    filtered = [
        issue
        for issue in errors
        if not issue.get("image") or Path(str(issue.get("image"))).name == image_name
    ]
    return filtered


def _extract_errors(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        raise ValueError("Overlay errors must be a JSON object or list")
    for key in ("errors", "critical_events", "events", "issues"):
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    samples = data.get("samples")
    if isinstance(samples, dict):
        rows: list[dict[str, Any]] = []
        for category, items in samples.items():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        rows.append({"category": str(category), **item})
        return rows
    return [data]
