from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import save_run_snapshot, write_json


def train_yolo_from_config(config: dict[str, Any]) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is required for training. Install requirements.txt.") from exc

    training = config.get("training", {})
    model_path = training.get("model", "yolo11n.pt")
    data_yaml = training["data"]
    project = Path(training.get("project", "runs"))
    name = training.get("name", "seedlings")
    output_dir = project / name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_run_snapshot(output_dir, config, "train")

    train_args = {
        "data": data_yaml,
        "epochs": int(training.get("epochs", 200)),
        "batch": int(training.get("batch", 16)),
        "imgsz": int(training.get("imgsz", 640)),
        "project": str(project),
        "name": name,
        "exist_ok": bool(training.get("exist_ok", True)),
        "seed": int(training.get("seed", 42)),
        "deterministic": bool(training.get("deterministic", True)),
        "save": bool(training.get("save", True)),
    }
    for optional_key in ["device", "optimizer", "lr0", "lrf", "weight_decay", "patience", "workers"]:
        if optional_key in training and training[optional_key] is not None:
            train_args[optional_key] = training[optional_key]

    model = YOLO(model_path)
    train_result = model.train(**train_args)

    metrics: dict[str, Any] = {"train_results_dir": str(getattr(train_result, "save_dir", output_dir))}
    validation = config.get("validation", {})
    if validation.get("run_after_train", True):
        metrics["val"] = _metrics_to_dict(model.val(data=data_yaml, split="val"))
    if validation.get("test_after_train", False):
        metrics["test"] = _metrics_to_dict(model.val(data=data_yaml, split="test"))
    write_json(output_dir / "metrics_summary.json", metrics)
    return metrics


def validate_yolo_from_config(config: dict[str, Any], split: str = "val") -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is required for validation. Install requirements.txt.") from exc

    validation = config.get("validation", {})
    model_path = validation.get("model") or config.get("training", {}).get("model")
    data_yaml = validation.get("data") or config.get("training", {}).get("data")
    if not model_path or not data_yaml:
        raise ValueError("Validation requires validation.model/training.model and validation.data/training.data")

    output_dir = Path(validation.get("output_dir", "runs/validation"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_run_snapshot(output_dir, config, f"val:{split}")

    model = YOLO(model_path)
    args = {
        "data": data_yaml,
        "split": split,
        "imgsz": int(validation.get("imgsz", config.get("training", {}).get("imgsz", 640))),
    }
    for optional_key in ["device", "batch", "conf", "iou"]:
        if optional_key in validation and validation[optional_key] is not None:
            args[optional_key] = validation[optional_key]
    metrics = _metrics_to_dict(model.val(**args))
    write_json(output_dir / f"{split}_metrics.json", metrics)
    return metrics


def _metrics_to_dict(metrics: Any) -> dict[str, Any]:
    box = getattr(metrics, "box", None)
    if box is None:
        return {"raw": str(metrics)}
    return {
        "map50": float(getattr(box, "map50", 0.0)),
        "map50_95": float(getattr(box, "map", 0.0)),
        "precision_per_class": _float_list(getattr(box, "p", [])),
        "recall_per_class": _float_list(getattr(box, "r", [])),
        "precision_mean": _mean(getattr(box, "p", [])),
        "recall_mean": _mean(getattr(box, "r", [])),
    }


def _float_list(values: Any) -> list[float]:
    try:
        return [float(value) for value in values]
    except TypeError:
        return []


def _mean(values: Any) -> float | None:
    values_list = _float_list(values)
    if not values_list:
        return None
    return sum(values_list) / len(values_list)
