from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from seedling_core.config import (
    ConfigError,
    FieldSpec,
    load_config_file,
    require_sections,
    validate_existing_paths,
    validate_fields,
)
from seedling_core.run_snapshot import (
    config_hash,
    environment_snapshot,
    package_version,
    register_run_artifacts,
    save_run_snapshot,
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML experiment config."""
    return load_config_file(path)


def validate_experiment_config(
    config: dict[str, Any],
    command: str,
    base_dir: str | Path | None = None,
    check_paths: bool = False,
) -> None:
    required_sections = {
        "prepare": ["dataset"],
        "train": ["training"],
        "val": [],
        "predict": ["prediction"],
        "evaluate": ["evaluation"],
        "evaluate-cells": ["evaluation"],
        "baseline-green": ["baseline"],
    }.get(command, [])
    require_sections(config, required_sections)
    specs: list[FieldSpec] = []
    if command == "prepare":
        specs.extend(
            [
                FieldSpec.parse("dataset.raw_root", str),
                FieldSpec.parse("dataset.prepared_root", str),
                FieldSpec.parse("dataset.split", dict, required=False),
            ]
        )
    elif command == "train":
        specs.append(FieldSpec.parse("training.data", str))
    elif command == "val":
        if not _has(config, "validation.data") and not _has(config, "training.data"):
            raise ConfigError("Validation requires `validation.data` or `training.data`")
        if not _has(config, "validation.model") and not _has(config, "training.model"):
            raise ConfigError("Validation requires `validation.model` or `training.model`")
    elif command == "predict":
        specs.extend(
            [
                FieldSpec.parse("prediction.model", str),
                FieldSpec.parse("prediction.images", str),
            ]
        )
    elif command in {"evaluate", "evaluate-cells"}:
        if _has(config, "evaluation.gt_scene") or _has(config, "evaluation.pred_scene"):
            specs.extend(
                [
                    FieldSpec.parse("evaluation.gt_scene", str),
                    FieldSpec.parse("evaluation.pred_scene", str),
                ]
            )
        else:
            specs.extend(
                [
                    FieldSpec.parse("evaluation.dataset", str),
                    FieldSpec.parse("evaluation.predictions", str),
                ]
            )
        specs.append(FieldSpec.parse("evaluation.calibration", str, required=False))
        specs.append(FieldSpec.parse("evaluation.image_manifest", str, required=False))
    elif command == "baseline-green":
        specs.extend(
            [
                FieldSpec.parse("baseline.images", str),
                FieldSpec.parse("baseline.labels", (str, type(None)), required=False),
            ]
        )
    validate_fields(config, specs)
    if check_paths:
        validate_existing_paths(config, _existing_path_fields(command, config), base_dir=base_dir)


def write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_yaml(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
    except ImportError:
        write_json(output.with_suffix(".json"), data)
        return
    output.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _has(config: dict[str, Any], dotted_path: str) -> bool:
    current: Any = config
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return current is not None


def _existing_path_fields(command: str, config: dict[str, Any]) -> list[str]:
    if command == "prepare":
        return ["dataset.raw_root"]
    if command == "train":
        return ["training.data"]
    if command == "val":
        fields = []
        if _has(config, "validation.data"):
            fields.append("validation.data")
        elif _has(config, "training.data"):
            fields.append("training.data")
        if _has(config, "validation.model"):
            fields.append("validation.model")
        elif _has(config, "training.model"):
            fields.append("training.model")
        return fields
    if command == "predict":
        return ["prediction.model", "prediction.images"]
    if command in {"evaluate", "evaluate-cells"}:
        if _has(config, "evaluation.gt_scene") or _has(config, "evaluation.pred_scene"):
            fields = ["evaluation.gt_scene", "evaluation.pred_scene"]
        else:
            fields = ["evaluation.dataset", "evaluation.predictions"]
        if _has(config, "evaluation.calibration"):
            fields.append("evaluation.calibration")
        if _has(config, "evaluation.image_manifest"):
            fields.append("evaluation.image_manifest")
        return fields
    if command == "baseline-green":
        return ["baseline.images", "baseline.labels"]
    return []
