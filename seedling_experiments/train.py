from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import register_run_artifacts, save_run_snapshot, write_json
from .yolo import label_path_for, list_images, read_labels


def train_yolo_from_config(config: dict[str, Any], command_args: dict[str, Any] | None = None) -> dict[str, Any]:
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
    save_run_snapshot(output_dir, config, "train", command_args=command_args)

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
    metrics_path = output_dir / "metrics_summary.json"
    write_json(metrics_path, metrics)
    result_dir = Path(metrics["train_results_dir"])
    extra_outputs = [metrics_path]
    if result_dir != output_dir:
        extra_outputs.append(result_dir)
    weights_dir = result_dir / "weights"
    if weights_dir.exists():
        extra_outputs.extend(sorted(weights_dir.glob("*.pt")))
    register_run_artifacts(
        output_dir,
        "train",
        config=config,
        command_args=command_args,
        input_paths=[model_path, data_yaml],
        output_paths=extra_outputs,
    )
    return metrics


def validate_yolo_from_config(
    config: dict[str, Any],
    split: str = "val",
    command_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
    save_run_snapshot(output_dir, config, f"val:{split}", command_args=command_args)

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
    metrics["validation"] = {
        "source": "ultralytics.val",
        "split": split,
        "model": str(Path(model_path)),
        "data": str(Path(data_yaml)),
        "imgsz": args["imgsz"],
        "conf": args.get("conf"),
        "iou": args.get("iou"),
    }
    metrics["dataset"] = _split_stats_from_data_yaml(data_yaml, split)
    metrics_path = output_dir / f"{split}_metrics.json"
    write_json(metrics_path, metrics)
    register_run_artifacts(
        output_dir,
        f"val:{split}",
        config=config,
        command_args=command_args,
        input_paths=[model_path, data_yaml],
        output_paths=[metrics_path],
    )
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


def _split_stats_from_data_yaml(data_yaml: str | Path, split: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return {"error": "PyYAML is required to inspect data.yaml"}

    data_path = Path(data_yaml)
    data = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return {"error": f"Invalid data.yaml: {data_path}"}

    class_names = _class_names_from_data_yaml(data)
    images_value = data.get(split)
    if not images_value:
        return {"error": f"Split {split!r} not found in {data_path}"}

    root = Path(data.get("path") or data_path.parent)
    images_dir = Path(images_value)
    if not images_dir.is_absolute():
        images_dir = root / images_dir
    labels_dir = images_dir.parent / "labels"

    images = list_images(images_dir) if images_dir.exists() else []
    class_counts: dict[str, int] = {name: 0 for name in class_names}
    missing_labels: list[str] = []
    total_labels = 0
    for image_path in images:
        label_path = label_path_for(image_path, labels_dir)
        if not label_path.exists():
            missing_labels.append(image_path.name)
            continue
        for label in read_labels(label_path):
            total_labels += 1
            if label.class_id < len(class_names):
                key = class_names[label.class_id]
            else:
                key = str(label.class_id)
            class_counts[key] = class_counts.get(key, 0) + 1

    return {
        "data_yaml": str(data_path.resolve()),
        "split": split,
        "images_dir": str(images_dir.resolve()),
        "labels_dir": str(labels_dir.resolve()),
        "images": len(images),
        "labels": total_labels,
        "class_names": class_names,
        "class_counts": class_counts,
        "missing_labels": missing_labels,
    }


def _class_names_from_data_yaml(data: dict[str, Any]) -> list[str]:
    names = data.get("names", [])
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=_class_name_sort_key)]
    if isinstance(names, list):
        return [str(name) for name in names]
    return []


def _class_name_sort_key(value: Any) -> tuple[int, int | str]:
    text = str(value)
    if text.isdigit():
        return (0, int(text))
    return (1, text)
