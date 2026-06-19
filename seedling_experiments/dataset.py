from __future__ import annotations

import csv
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import write_json
from .image_io import register_heif_if_available
from .yolo import label_path_for, list_images, read_labels, write_data_yaml
from seedling_data.manifests import read_group_map

register_heif_if_available()

DEFAULT_AUGMENTED_NAME_MARKERS = (
    "_filter",
    "_flip",
    "_scale",
    "_rotate",
    "_crop",
    "_brightness",
    "_contrast",
    "_color",
    "_sharpness",
    "_hflip",
    "_vflip",
)


def _image_size(path: Path) -> tuple[int, int] | None:
    from PIL import Image

    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None


def audit_yolo_dataset(
    dataset_root: str | Path,
    output_path: str | Path | None = None,
    class_names: list[str] | None = None,
    augmented_name_markers: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    root = Path(dataset_root)
    markers = _normalise_markers(augmented_name_markers)
    image_dirs = [
        candidate
        for candidate in [
            root / "images",
            root / "train" / "images",
            root / "val" / "images",
            root / "test" / "images",
        ]
        if candidate.exists()
    ]
    if not image_dirs:
        raise FileNotFoundError(f"No YOLO image directories found under {root}")

    split_stats: dict[str, Any] = {}
    total_classes: Counter[int] = Counter()
    all_widths: list[float] = []
    all_heights: list[float] = []

    for images_dir in image_dirs:
        split_name = images_dir.parent.name if images_dir.parent != root else "all"
        labels_dir = images_dir.parent / "labels"
        images = list_images(images_dir)
        class_counts: Counter[int] = Counter()
        missing_labels: list[str] = []
        empty_labels: list[str] = []
        invalid_labels: list[dict[str, str]] = []
        unknown_class_ids: list[dict[str, int | str]] = []
        bbox_out_of_range: list[dict[str, int | str | float]] = []
        corrupt_images: list[str] = []
        suspected_augmented_images: list[str] = []
        bbox_widths: list[float] = []
        bbox_heights: list[float] = []

        for image_path in images:
            if is_suspected_augmented_name(image_path.name, markers):
                suspected_augmented_images.append(image_path.name)
            size = _image_size(image_path)
            if size is None:
                corrupt_images.append(image_path.name)
            label_path = label_path_for(image_path, labels_dir)
            if not label_path.exists():
                missing_labels.append(image_path.name)
                continue
            try:
                labels = read_labels(label_path)
            except ValueError as exc:
                invalid_labels.append({"label": label_path.name, "error": str(exc)})
                continue
            if not labels:
                empty_labels.append(label_path.name)
            for label in labels:
                if class_names is not None and label.class_id >= len(class_names):
                    unknown_class_ids.append({"label": label_path.name, "class_id": label.class_id})
                if not _label_in_range(label):
                    bbox_out_of_range.append(
                        {
                            "label": label_path.name,
                            "class_id": label.class_id,
                            "x_center": label.x_center,
                            "y_center": label.y_center,
                            "width": label.width,
                            "height": label.height,
                        }
                    )
                class_counts[label.class_id] += 1
                total_classes[label.class_id] += 1
                bbox_widths.append(label.width)
                bbox_heights.append(label.height)
                all_widths.append(label.width)
                all_heights.append(label.height)

        orphan_labels = []
        if labels_dir.exists():
            image_stems = {image.stem for image in images}
            orphan_labels = [
                label.name
                for label in sorted(labels_dir.glob("*.txt"))
                if label.stem not in image_stems
            ]

        split_stats[split_name] = {
            "images": len(images),
            "labels": sum(class_counts.values()),
            "class_counts": _named_counts(class_counts, class_names),
            "missing_labels": missing_labels,
            "empty_labels": empty_labels,
            "invalid_labels": invalid_labels,
            "unknown_class_ids": unknown_class_ids,
            "bbox_out_of_range": bbox_out_of_range,
            "orphan_labels": orphan_labels,
            "corrupt_images": corrupt_images,
            "suspected_augmented_images": suspected_augmented_images,
            "suspected_augmented_count": len(suspected_augmented_images),
            "bbox_width_norm": _summary(bbox_widths),
            "bbox_height_norm": _summary(bbox_heights),
        }

    result = {
        "dataset_root": str(root.resolve()),
        "splits": split_stats,
        "total_class_counts": _named_counts(total_classes, class_names),
        "bbox_width_norm": _summary(all_widths),
        "bbox_height_norm": _summary(all_heights),
    }
    if output_path:
        write_json(output_path, result)
    return result


def make_grouped_split(
    source_root: str | Path,
    output_root: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
    group_regex: str | None = None,
    metadata_path: str | Path | None = None,
    metadata_group_column: str = "group_id",
    class_names: list[str] | None = None,
    copy_images: bool = True,
) -> dict[str, Any]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    source = Path(source_root)
    output = Path(output_root)
    images_dir = source / "images"
    labels_dir = source / "labels"
    images = list_images(images_dir)
    if not images:
        raise FileNotFoundError(f"No images found in {images_dir}")

    groups: dict[str, list[Path]] = defaultdict(list)
    pattern = re.compile(group_regex) if group_regex else None
    metadata_groups = read_group_map(metadata_path, metadata_group_column) if metadata_path else {}
    for image_path in images:
        group = (
            metadata_groups.get(image_path.name)
            or metadata_groups.get(image_path.stem)
            or metadata_groups.get(str(image_path.resolve()))
            or _group_for_image(image_path, pattern)
        )
        groups[group].append(image_path)

    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    split_names = _assign_groups(group_items, len(images), train_ratio, val_ratio)

    rows: list[dict[str, str]] = []
    for split_name, split_groups in split_names.items():
        out_images = output / split_name / "images"
        out_labels = output / split_name / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)
        for group_id, group_images in split_groups:
            for image_path in group_images:
                label_path = label_path_for(image_path, labels_dir)
                target_image = out_images / image_path.name
                target_label = out_labels / f"{image_path.stem}.txt"
                if copy_images:
                    shutil.copy2(image_path, target_image)
                else:
                    target_image.symlink_to(image_path.resolve())
                if label_path.exists():
                    shutil.copy2(label_path, target_label)
                else:
                    target_label.write_text("", encoding="utf-8")
                rows.append(
                    {
                        "image": image_path.name,
                        "group": group_id,
                        "split": split_name,
                        "source_image": str(image_path.resolve()),
                        "source_label": str(label_path.resolve()),
                    }
                )

    class_names = class_names or ["container", "seedlings"]
    write_data_yaml(output / "data.yaml", output, class_names)
    _write_manifest(output / "split_manifest.csv", rows)
    summary = {
        "source_root": str(source.resolve()),
        "output_root": str(output.resolve()),
        "seed": seed,
        "group_regex": group_regex,
        "metadata_path": str(Path(metadata_path).resolve()) if metadata_path else None,
        "metadata_group_column": metadata_group_column,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "groups": {split: len(items) for split, items in split_names.items()},
        "images": Counter(row["split"] for row in rows),
        "data_yaml": str((output / "data.yaml").resolve()),
    }
    write_json(output / "split_summary.json", summary)
    return summary


def validate_split_integrity(
    dataset_root: str | Path,
    output_path: str | Path | None = None,
    augmented_name_markers: list[str] | tuple[str, ...] | None = None,
    eval_splits: tuple[str, ...] = ("val", "test"),
) -> dict[str, Any]:
    root = Path(dataset_root)
    markers = _normalise_markers(augmented_name_markers)
    split_images = _images_by_split(root)
    suspected_by_split = {
        split: [image.name for image in images if is_suspected_augmented_name(image.name, markers)]
        for split, images in split_images.items()
    }

    base_to_splits: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for split, images in split_images.items():
        for image in images:
            base_to_splits[base_image_key(image.name, markers)][split].append(image.name)

    leakage = []
    for base_key, splits in sorted(base_to_splits.items()):
        if len(splits) <= 1:
            continue
        leakage.append(
            {
                "base_image": base_key,
                "splits": {
                    split: names
                    for split, names in sorted(splits.items())
                },
            }
        )

    eval_augmented = {
        split: suspected_by_split.get(split, [])
        for split in eval_splits
        if suspected_by_split.get(split)
    }
    result = {
        "dataset_root": str(root.resolve()),
        "markers": list(markers),
        "images": {split: len(images) for split, images in split_images.items()},
        "suspected_augmented_images": suspected_by_split,
        "eval_augmented_images": eval_augmented,
        "cross_split_base_leakage": leakage,
        "ok": not eval_augmented and not leakage,
    }
    if output_path:
        write_json(output_path, result)
    return result


def assert_no_suspected_augmented_source(
    dataset_root: str | Path,
    augmented_name_markers: list[str] | tuple[str, ...] | None = None,
) -> None:
    root = Path(dataset_root)
    markers = _normalise_markers(augmented_name_markers)
    images = list_images(root / "images")
    suspected = [image.name for image in images if is_suspected_augmented_name(image.name, markers)]
    if suspected:
        preview = ", ".join(suspected[:10])
        suffix = "" if len(suspected) <= 10 else f", ... ({len(suspected)} total)"
        raise ValueError(
            "Raw dataset appears to contain augmented images. "
            "Prepare expects clean source images only; set "
            "dataset.allow_augmented_source: true to override. "
            f"Suspected files: {preview}{suffix}"
        )


def is_suspected_augmented_name(
    image_name: str,
    augmented_name_markers: list[str] | tuple[str, ...] | None = None,
) -> bool:
    stem = Path(image_name).stem.lower()
    return any(marker in stem for marker in _normalise_markers(augmented_name_markers))


def base_image_key(
    image_name: str,
    augmented_name_markers: list[str] | tuple[str, ...] | None = None,
) -> str:
    stem = Path(image_name).stem.lower()
    first_marker = min(
        (stem.find(marker) for marker in _normalise_markers(augmented_name_markers) if marker in stem),
        default=-1,
    )
    return stem if first_marker < 0 else stem[:first_marker]


def _images_by_split(root: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    if (root / "images").exists():
        result["all"] = list_images(root / "images")
    for split in ["train", "val", "test"]:
        images_dir = root / split / "images"
        if images_dir.exists():
            result[split] = list_images(images_dir)
    if not result:
        raise FileNotFoundError(f"No YOLO image directories found under {root}")
    return result


def _normalise_markers(
    markers: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    raw_markers = markers or DEFAULT_AUGMENTED_NAME_MARKERS
    return tuple(marker.lower() for marker in raw_markers if marker)


def _assign_groups(
    group_items: list[tuple[str, list[Path]]],
    total_images: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, list[tuple[str, list[Path]]]]:
    targets = {
        "train": total_images * train_ratio,
        "val": total_images * val_ratio,
    }
    result: dict[str, list[tuple[str, list[Path]]]] = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}
    for group_id, images in group_items:
        if counts["train"] < targets["train"]:
            split = "train"
        elif counts["val"] < targets["val"]:
            split = "val"
        else:
            split = "test"
        result[split].append((group_id, images))
        counts[split] += len(images)
    return result


def _group_for_image(image_path: Path, pattern: re.Pattern[str] | None) -> str:
    if pattern is None:
        return image_path.stem
    match = pattern.search(image_path.stem)
    if not match:
        return image_path.stem
    if match.groups():
        return match.group(1)
    return match.group(0)


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image", "group", "split", "source_image", "source_label"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _label_in_range(label: Any) -> bool:
    return (
        label.class_id >= 0
        and 0.0 <= label.x_center <= 1.0
        and 0.0 <= label.y_center <= 1.0
        and 0.0 < label.width <= 1.0
        and 0.0 < label.height <= 1.0
        and 0.0 <= label.x_center - label.width / 2
        and label.x_center + label.width / 2 <= 1.0
        and 0.0 <= label.y_center - label.height / 2
        and label.y_center + label.height / 2 <= 1.0
    )


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "mean": sum(values) / len(values),
        "max": ordered[-1],
    }


def _named_counts(counts: Counter[int], class_names: list[str] | None) -> dict[str, int]:
    result: dict[str, int] = {}
    for class_id, count in sorted(counts.items()):
        if class_names and class_id < len(class_names):
            name = class_names[class_id]
        else:
            name = str(class_id)
        result[name] = count
    return result
