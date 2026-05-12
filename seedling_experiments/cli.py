from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .config import load_config, write_json


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m seedling_experiments",
        description="Reproducible experiments for seedling detection and cell-level evaluation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_config_command(subparsers, "train", "Train YOLO and save a run snapshot.")
    _add_config_command(subparsers, "val", "Validate YOLO on val/test split.")
    _add_config_command(subparsers, "predict", "Run YOLO prediction and save predictions.json.")
    _add_config_command(subparsers, "evaluate-cells", "Evaluate cell matrices and removal targets.")
    _add_config_command(subparsers, "baseline-green", "Run HSV green connected-components baseline.")
    _add_config_command(subparsers, "prepare", "Run split, train-only augmentation, and audit from config.")

    split_parser = subparsers.add_parser("split", help="Create grouped train/val/test split before augmentation.")
    split_parser.add_argument("--source", required=True, help="YOLO source root with images/labels.")
    split_parser.add_argument("--output", required=True, help="Prepared dataset output root.")
    split_parser.add_argument("--train", type=float, default=0.7)
    split_parser.add_argument("--val", type=float, default=0.2)
    split_parser.add_argument("--test", type=float, default=0.1)
    split_parser.add_argument("--seed", type=int, default=42)
    split_parser.add_argument("--group-regex", default=None)
    split_parser.add_argument("--class-names", default="container,seedlings")

    audit_parser = subparsers.add_parser("audit", help="Audit YOLO dataset statistics.")
    audit_parser.add_argument("--dataset", required=True)
    audit_parser.add_argument("--output", default=None)
    audit_parser.add_argument("--class-names", default="container,seedlings")

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
            class_names=_class_names(args.class_names),
        )
        _print_result(result)
    elif args.command == "audit":
        from .dataset import audit_yolo_dataset

        result = audit_yolo_dataset(
            dataset_root=args.dataset,
            output_path=args.output,
            class_names=_class_names(args.class_names),
        )
        _print_result(result)
    else:
        config = load_config(args.config)
        result = _run_config_command(args.command, config, args)
        _print_result(result)


def _add_config_command(subparsers: argparse._SubParsersAction, name: str, help_text: str) -> None:
    parser = subparsers.add_parser(name, help=help_text)
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config.")
    if name == "val":
        parser.add_argument("--split", default="val", choices=["train", "val", "test"])


def _run_config_command(command: str, config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if command == "prepare":
        return prepare_from_config(config)
    if command == "train":
        from .train import train_yolo_from_config

        return train_yolo_from_config(config)
    if command == "val":
        from .train import validate_yolo_from_config

        return validate_yolo_from_config(config, split=args.split)
    if command == "predict":
        from .predict import predict_from_config

        return predict_from_config(config)
    if command == "evaluate-cells":
        from .evaluate import evaluate_cells_from_config

        return evaluate_cells_from_config(config)
    if command == "baseline-green":
        from .baselines import green_components_baseline_from_config

        return green_components_baseline_from_config(config)
    raise ValueError(f"Unknown command: {command}")


def prepare_from_config(config: dict[str, Any]) -> dict[str, Any]:
    from .augmentation import augment_train_split
    from .dataset import audit_yolo_dataset, make_grouped_split

    dataset = config.get("dataset", {})
    split_config = dataset.get("split", {})
    class_names = dataset.get("class_names", ["container", "seedlings"])
    output_root = dataset["prepared_root"]
    split_summary = make_grouped_split(
        source_root=dataset["raw_root"],
        output_root=output_root,
        train_ratio=float(split_config.get("train", 0.7)),
        val_ratio=float(split_config.get("val", 0.2)),
        test_ratio=float(split_config.get("test", 0.1)),
        seed=int(split_config.get("seed", 42)),
        group_regex=split_config.get("group_regex"),
        class_names=class_names,
    )

    augmentation_summary = None
    if dataset.get("augment_train"):
        augmentation_summary = augment_train_split(
            split_root=output_root,
            augmentations=dataset["augment_train"],
            split_name="train",
        )

    audit_summary = audit_yolo_dataset(
        dataset_root=output_root,
        output_path=Path(output_root) / "dataset_audit.json",
        class_names=class_names,
    )
    result = {
        "split": split_summary,
        "augmentation": augmentation_summary,
        "audit": audit_summary,
    }
    write_json(Path(output_root) / "prepare_summary.json", result)
    return result


def _class_names(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _print_result(result: dict[str, Any]) -> None:
    import json

    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
