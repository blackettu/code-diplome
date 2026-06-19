from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .config import config_hash, load_config, save_run_snapshot, validate_experiment_config, write_json


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m seedling_experiments",
        description="Reproducible experiments for seedling detection and cell-level evaluation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_config_command(subparsers, "train", "Train YOLO and save a run snapshot.")
    _add_config_command(subparsers, "val", "Validate YOLO on val/test split.")
    _add_config_command(subparsers, "predict", "Run YOLO prediction and save predictions.json.")
    _add_config_command(subparsers, "evaluate", "Evaluate cell matrices and removal targets.")
    _add_config_command(subparsers, "evaluate-cells", "Evaluate cell matrices and removal targets.")
    _add_config_command(subparsers, "baseline-green", "Run HSV green connected-components baseline.")
    _add_config_command(subparsers, "prepare", "Run split, train-only augmentation, and audit from config.")

    sim_parser = subparsers.add_parser("sim", help="Run a unified simulator smoke step from TrayEnv YAML.")
    sim_parser.add_argument("--config", default="configs/simulation/tray_env_v0.yaml")
    sim_parser.add_argument("--out", default="runs/sim/unified_sim_smoke.json")
    sim_parser.add_argument("--seed", type=int, default=None)

    rl_parser = subparsers.add_parser("rl", help="Run RL training/evaluation through the unified experiment CLI.")
    rl_parser.add_argument("--config", default="configs/rl/maskable_ppo_v0.yaml")
    rl_parser.add_argument("--mode", choices=["train", "evaluate-baselines"], default="train")
    rl_parser.add_argument("--dry-run", action="store_true", help="Validate RL training config without importing SB3.")
    rl_parser.add_argument("--episodes", type=int, default=10)
    rl_parser.add_argument("--out", default=None)

    split_parser = subparsers.add_parser("split", help="Create grouped train/val/test split before augmentation.")
    split_parser.add_argument("--source", required=True, help="YOLO source root with images/labels.")
    split_parser.add_argument("--output", required=True, help="Prepared dataset output root.")
    split_parser.add_argument("--train", type=float, default=0.7)
    split_parser.add_argument("--val", type=float, default=0.2)
    split_parser.add_argument("--test", type=float, default=0.1)
    split_parser.add_argument("--seed", type=int, default=42)
    split_parser.add_argument("--group-regex", default=None)
    split_parser.add_argument("--metadata", default=None, help="Optional image_manifest.csv with group_id.")
    split_parser.add_argument("--metadata-group-column", default="group_id")
    split_parser.add_argument("--class-names", default="container,seedlings")

    audit_parser = subparsers.add_parser("audit", help="Audit YOLO dataset statistics.")
    audit_parser.add_argument("--dataset", required=True)
    audit_parser.add_argument("--output", default=None)
    audit_parser.add_argument("--class-names", default="container,seedlings")
    audit_parser.add_argument(
        "--augmented-name-markers",
        default=None,
        help="Comma-separated substrings that mark augmented image names.",
    )

    check_parser = subparsers.add_parser(
        "check-split",
        help="Check prepared split for augmented val/test files and cross-split base leakage.",
    )
    check_parser.add_argument("--dataset", required=True)
    check_parser.add_argument("--output", default=None)
    check_parser.add_argument(
        "--augmented-name-markers",
        default=None,
        help="Comma-separated substrings that mark augmented image names.",
    )

    scene_eval = subparsers.add_parser("evaluate-scenes", help="Evaluate predicted SceneState JSON against expert SceneState JSON.")
    scene_eval.add_argument("--gt", required=True, help="Ground-truth/expert SceneState JSON.")
    scene_eval.add_argument("--pred", required=True, help="Predicted SceneState JSON.")
    scene_eval.add_argument("--out", required=True)
    scene_eval.add_argument("--target-distance-px", type=float, default=25.0)
    scene_eval.add_argument("--target-distance-mm", type=float, default=None)

    args = parser.parse_args(argv)
    if args.command == "split":
        from .dataset import make_grouped_split

        result = make_grouped_split(
            source_root=args.source,
            output_root=args.output,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            seed=args.seed,
            group_regex=args.group_regex,
            metadata_path=args.metadata,
            metadata_group_column=args.metadata_group_column,
            class_names=_class_names(args.class_names),
        )
        result["artifact_registry"] = _write_experiment_registry(
            run_dir=args.output,
            command="seedling-experiments:split",
            inputs=[args.source, *([args.metadata] if args.metadata else [])],
            outputs=[
                Path(args.output) / "data.yaml",
                Path(args.output) / "split_manifest.csv",
                Path(args.output) / "split_summary.json",
            ],
            metadata={"train": args.train, "val": args.val, "test": args.test, "seed": args.seed},
        )
        _print_result(result)
    elif args.command == "audit":
        from .dataset import audit_yolo_dataset

        result = audit_yolo_dataset(
            dataset_root=args.dataset,
            output_path=args.output,
            class_names=_class_names(args.class_names),
            augmented_name_markers=_optional_list(args.augmented_name_markers),
        )
        if args.output:
            result["artifact_registry"] = _write_experiment_registry(
                run_dir=Path(args.output).parent,
                command="seedling-experiments:audit",
                inputs=[args.dataset],
                outputs=[args.output],
                metadata={"class_names": _class_names(args.class_names)},
            )
        _print_result(result)
    elif args.command == "check-split":
        from .dataset import validate_split_integrity

        result = validate_split_integrity(
            dataset_root=args.dataset,
            output_path=args.output,
            augmented_name_markers=_optional_list(args.augmented_name_markers),
        )
        if args.output:
            result["artifact_registry"] = _write_experiment_registry(
                run_dir=Path(args.output).parent,
                command="seedling-experiments:check-split",
                inputs=[args.dataset],
                outputs=[args.output],
                metadata={"augmented_name_markers": _optional_list(args.augmented_name_markers)},
            )
        _print_result(result)
    elif args.command == "evaluate-scenes":
        from seedling_cells.evaluation import evaluate_scene_state_files

        result = evaluate_scene_state_files(
            args.gt,
            args.pred,
            output_path=args.out,
            target_match_distance_px=args.target_distance_px,
            target_match_distance_mm=args.target_distance_mm,
        )
        result["artifact_registry"] = _write_experiment_registry(
            run_dir=Path(args.out).parent,
            command="seedling-experiments:evaluate-scenes",
            inputs=[args.gt, args.pred],
            outputs=[args.out],
            metadata={
                "target_distance_px": args.target_distance_px,
                "target_distance_mm": args.target_distance_mm,
            },
        )
        _print_result(result)
    elif args.command == "sim":
        result = run_sim_smoke(args.config, output_path=args.out, seed=args.seed)
        _print_result(result)
    elif args.command == "rl":
        result = run_rl_command(
            args.config,
            mode=args.mode,
            dry_run=args.dry_run,
            episodes=args.episodes,
            output_path=args.out,
        )
        _print_result(result)
    else:
        config = load_config(args.config)
        validate_experiment_config(config, args.command, check_paths=True)
        result = _run_config_command(args.command, config, args)
        _print_result(result)


def _add_config_command(subparsers: argparse._SubParsersAction, name: str, help_text: str) -> None:
    parser = subparsers.add_parser(name, help=help_text)
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config.")
    if name == "val":
        parser.add_argument("--split", default="val", choices=["train", "val", "test"])


def _run_config_command(command: str, config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    command_args = _namespace_to_dict(args)
    if command == "prepare":
        return prepare_from_config(config, command_args=command_args)
    if command == "train":
        from .train import train_yolo_from_config

        return train_yolo_from_config(config, command_args=command_args)
    if command == "val":
        from .train import validate_yolo_from_config

        return validate_yolo_from_config(config, split=args.split, command_args=command_args)
    if command == "predict":
        from .predict import predict_from_config

        return predict_from_config(config, command_args=command_args)
    if command in {"evaluate", "evaluate-cells"}:
        from .evaluate import evaluate_cells_from_config

        return evaluate_cells_from_config(config, command=command, command_args=command_args)
    if command == "baseline-green":
        from .baselines import green_components_baseline_from_config

        return green_components_baseline_from_config(config, command_args=command_args)
    raise ValueError(f"Unknown command: {command}")


def prepare_from_config(config: dict[str, Any], command_args: dict[str, Any] | None = None) -> dict[str, Any]:
    from .augmentation import augment_train_split
    from .dataset import (
        assert_no_suspected_augmented_source,
        audit_yolo_dataset,
        make_grouped_split,
        validate_split_integrity,
    )

    dataset = config.get("dataset", {})
    split_config = dataset.get("split", {})
    class_names = dataset.get("class_names", ["container", "seedlings"])
    output_root = dataset["prepared_root"]
    augmented_markers = dataset.get("augmented_name_markers")
    save_run_snapshot(output_root, config, "prepare", command_args=command_args)
    if not bool(dataset.get("allow_augmented_source", False)):
        assert_no_suspected_augmented_source(
            dataset_root=dataset["raw_root"],
            augmented_name_markers=augmented_markers,
        )

    raw_audit_summary = audit_yolo_dataset(
        dataset_root=dataset["raw_root"],
        output_path=Path(output_root) / "raw_dataset_audit.json",
        class_names=class_names,
        augmented_name_markers=augmented_markers,
    )

    split_summary = make_grouped_split(
        source_root=dataset["raw_root"],
        output_root=output_root,
        train_ratio=float(split_config.get("train", 0.7)),
        val_ratio=float(split_config.get("val", 0.2)),
        test_ratio=float(split_config.get("test", 0.1)),
        seed=int(split_config.get("seed", 42)),
        group_regex=split_config.get("group_regex"),
        metadata_path=split_config.get("metadata") or split_config.get("metadata_path"),
        metadata_group_column=split_config.get("metadata_group_column", "group_id"),
        class_names=class_names,
    )

    augmentation_summary = None
    if dataset.get("augment_train"):
        augmentation_summary = augment_train_split(
            split_root=output_root,
            augmentations=dataset["augment_train"],
            split_name="train",
        )
    else:
        augmentation_summary = _write_empty_augmentation_summary(output_root, split_name="train")

    split_integrity = validate_split_integrity(
        dataset_root=output_root,
        output_path=Path(output_root) / "split_integrity_report.json",
        augmented_name_markers=augmented_markers,
    )
    if not split_integrity["ok"] and not bool(dataset.get("allow_split_integrity_issues", False)):
        raise ValueError(
            "Prepared split integrity check failed. See "
            f"{Path(output_root) / 'split_integrity_report.json'} or set "
            "dataset.allow_split_integrity_issues: true to override."
        )

    audit_summary = audit_yolo_dataset(
        dataset_root=output_root,
        output_path=Path(output_root) / "dataset_audit.json",
        class_names=class_names,
        augmented_name_markers=augmented_markers,
    )
    prepare_summary_path = Path(output_root) / "prepare_summary.json"
    result = {
        "raw_audit": raw_audit_summary,
        "split": split_summary,
        "augmentation": augmentation_summary,
        "split_integrity": split_integrity,
        "audit": audit_summary,
        "artifact_registry": str(Path(output_root) / "artifact_registry.json"),
    }
    write_json(prepare_summary_path, result)
    _write_experiment_registry(
        run_dir=output_root,
        command="prepare",
        inputs=_prepare_inputs(config, command_args),
        outputs=_existing_paths(
            [
                Path(output_root) / "config.yaml",
                Path(output_root) / "run_snapshot.json",
                Path(output_root) / "raw_dataset_audit.json",
                Path(output_root) / "data.yaml",
                Path(output_root) / "split_summary.json",
                Path(output_root) / "split_manifest.csv",
                Path(output_root) / "train_augmentation_manifest.csv",
                Path(output_root) / "split_integrity_report.json",
                Path(output_root) / "dataset_audit.json",
                prepare_summary_path,
            ]
        ),
        metadata={"command_args": command_args or {}, "config_hash": config_hash(config)},
        run_id=_snapshot_run_id(output_root, "prepare"),
    )
    return result


def run_sim_smoke(
    config_path: str | Path,
    output_path: str | Path | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    from seedling_core.config import load_config_file
    from seedling_rl.envs import SeedlingTrayEnv, TrayEnvConfig

    config = TrayEnvConfig.from_dict(load_config_file(config_path))
    env = SeedlingTrayEnv(config)
    observation, info = env.reset(seed=seed if seed is not None else config.seed)
    first_valid = next(index for index, value in enumerate(observation["action_mask"]) if int(value) == 1)
    _, reward, terminated, truncated, step_info = env.step(first_valid)
    result = {
        "ok": True,
        "command": "sim",
        "config": str(config_path),
        "scene_id": info["scene_id"],
        "targets": int(info["targets"]),
        "first_action": int(first_valid),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "event": step_info.get("event"),
    }
    if output_path:
        write_json(output_path, result)
        result["artifact_registry"] = _write_wrapper_registry(
            run_dir=Path(output_path).parent,
            command="seedling-experiments:sim",
            config_path=config_path,
            output_paths=[output_path],
        )
    return result


def run_rl_command(
    config_path: str | Path,
    mode: str = "train",
    dry_run: bool = False,
    episodes: int = 10,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if mode == "train":
        from seedling_rl.training import train_from_config

        result = train_from_config(config_path, dry_run=dry_run)
        if output_path:
            write_json(output_path, result)
            result["artifact_registry"] = _write_wrapper_registry(
                run_dir=Path(output_path).parent,
                command=f"seedling-experiments:rl:{mode}",
                config_path=config_path,
                output_paths=[output_path],
                metadata={"dry_run": dry_run},
            )
        return result
    if mode == "evaluate-baselines":
        from seedling_core.config import load_config_file
        from seedling_rl.baseline_eval import evaluate_baseline_suite
        from seedling_rl.envs import TrayEnvConfig

        raw_config = load_config_file(config_path)
        env_config_path = raw_config.get("env_config", "configs/simulation/tray_env_v0.yaml")
        compare = raw_config.get("eval", {}).get("compare_baselines") if isinstance(raw_config.get("eval"), dict) else None
        policies = [str(policy) for policy in compare] if isinstance(compare, list) else None
        env_config = TrayEnvConfig.from_dict(load_config_file(env_config_path))
        result = {
            "ok": True,
            "command": "rl",
            "mode": mode,
            "episodes": int(episodes),
            "results": [
                item.to_dict()
                for item in evaluate_baseline_suite(env_config, episodes=episodes, policies=policies)
            ],
        }
        effective_output_path = Path(output_path) if output_path else Path("runs/rl/unified_baseline_eval.json")
        write_json(effective_output_path, result)
        result["artifact_registry"] = _write_wrapper_registry(
            run_dir=effective_output_path.parent,
            command=f"seedling-experiments:rl:{mode}",
            config_path=config_path,
            output_paths=[effective_output_path],
            metadata={"episodes": episodes, "policies": policies},
        )
        return result
    raise ValueError(f"Unknown RL mode: {mode}")


def _write_wrapper_registry(
    run_dir: str | Path,
    command: str,
    config_path: str | Path,
    output_paths: list[str | Path],
    metadata: dict[str, Any] | None = None,
) -> str | None:
    return _write_experiment_registry(
        run_dir=run_dir,
        command=command,
        inputs=[config_path],
        outputs=output_paths,
        metadata=metadata,
    )


def _write_experiment_registry(
    run_dir: str | Path,
    command: str,
    *,
    inputs: list[str | Path],
    outputs: list[str | Path],
    metadata: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> str | None:
    try:
        from seedling_reports.registry import ArtifactRecord, write_run_registry_records

        artifacts = [
            *[
                ArtifactRecord.from_path(path, artifact_type=_artifact_type(path), role="input", command=command)
                for path in inputs
            ],
            *[
                ArtifactRecord.from_path(path, artifact_type=_artifact_type(path), role="output", command=command)
                for path in outputs
            ],
        ]
        write_run_registry_records(
            run_dir,
            command,
            artifacts,
            metadata=metadata,
            run_id=run_id or _registry_run_id(run_dir, command, outputs),
        )
        return str(Path(run_dir) / "artifact_registry.json")
    except Exception:
        return None


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
    if name.endswith(".yaml") or name.endswith(".yml"):
        return "config"
    return candidate.suffix.lstrip(".") or "artifact"


def _registry_run_id(run_dir: str | Path, command: str, outputs: list[str | Path]) -> str:
    directory = Path(run_dir).name or "run"
    output_stem = Path(outputs[0]).stem if outputs else "stdout"
    command_slug = "".join(char if char.isalnum() else "_" for char in command).strip("_")
    return f"{directory}_{command_slug}_{output_stem}"


def _snapshot_run_id(run_dir: str | Path, command: str) -> str:
    directory = Path(run_dir).name or "run"
    command_slug = "".join(char if char.isalnum() else "_" for char in command).strip("_")
    return f"{directory}_{command_slug}_snapshot"


def _prepare_inputs(config: dict[str, Any], command_args: dict[str, Any] | None) -> list[str | Path]:
    dataset = config.get("dataset", {})
    split_config = dataset.get("split", {}) if isinstance(dataset.get("split"), dict) else {}
    raw_paths: list[str | Path | None] = [
        (command_args or {}).get("config"),
        dataset.get("raw_root"),
        split_config.get("metadata") or split_config.get("metadata_path"),
    ]
    return _unique_paths(path for path in raw_paths if path)


def _existing_paths(paths: list[str | Path]) -> list[str | Path]:
    return [path for path in _unique_paths(paths) if Path(path).exists()]


def _unique_paths(paths) -> list[str | Path]:
    seen: set[str] = set()
    result: list[str | Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def _write_empty_augmentation_summary(split_root: str | Path, split_name: str = "train") -> dict[str, Any]:
    import csv

    root = Path(split_root)
    manifest_path = root / f"{split_name}_augmentation_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_image", "augmented_image", "kind", "augmentation"],
        )
        writer.writeheader()
    try:
        from .yolo import list_images

        source_images = len(list_images(root / split_name / "images"))
    except Exception:
        source_images = 0
    return {
        "split_root": str(root.resolve()),
        "split": split_name,
        "source_images": source_images,
        "created_images": 0,
        "manifest": str(manifest_path.resolve()),
        "enabled": False,
    }


def _class_names(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _optional_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def _print_result(result: dict[str, Any]) -> None:
    import json

    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def _namespace_to_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {key: _jsonable(value) for key, value in sorted(vars(args).items())}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)
