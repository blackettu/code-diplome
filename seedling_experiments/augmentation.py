from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance

from .image_io import register_heif_if_available
from .yolo import label_path_for, list_images, read_labels, write_labels

register_heif_if_available()


ENHANCERS = {
    "brightness": ImageEnhance.Brightness,
    "contrast": ImageEnhance.Contrast,
    "color": ImageEnhance.Color,
    "sharpness": ImageEnhance.Sharpness,
}


def augment_train_split(
    split_root: str | Path,
    augmentations: dict[str, Any],
    split_name: str = "train",
) -> dict[str, Any]:
    """Apply label-safe augmentations only to the selected split."""
    root = Path(split_root)
    images_dir = root / split_name / "images"
    labels_dir = root / split_name / "labels"
    if not images_dir.exists():
        raise FileNotFoundError(f"Split images directory not found: {images_dir}")

    rows: list[dict[str, str]] = []
    created = 0
    source_images = list_images(images_dir)

    for image_path in source_images:
        labels = read_labels(label_path_for(image_path, labels_dir))
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            for name, factors in (augmentations.get("photometric") or {}).items():
                if name not in ENHANCERS:
                    raise ValueError(f"Unsupported photometric augmentation: {name}")
                for factor in factors:
                    suffix = f"{name}_{_factor_suffix(float(factor))}"
                    output_image = images_dir / f"{image_path.stem}_{suffix}{image_path.suffix}"
                    output_label = labels_dir / f"{image_path.stem}_{suffix}.txt"
                    ENHANCERS[name](image).enhance(float(factor)).save(output_image)
                    write_labels(output_label, labels)
                    rows.append(_manifest_row(image_path, output_image, "photometric", suffix))
                    created += 1

            if augmentations.get("hflip", False):
                suffix = "hflip"
                output_image = images_dir / f"{image_path.stem}_{suffix}{image_path.suffix}"
                output_label = labels_dir / f"{image_path.stem}_{suffix}.txt"
                image.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(output_image)
                write_labels(output_label, [label.hflip() for label in labels])
                rows.append(_manifest_row(image_path, output_image, "geometry", suffix))
                created += 1

            if augmentations.get("vflip", False):
                suffix = "vflip"
                output_image = images_dir / f"{image_path.stem}_{suffix}{image_path.suffix}"
                output_label = labels_dir / f"{image_path.stem}_{suffix}.txt"
                image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(output_image)
                write_labels(output_label, [label.vflip() for label in labels])
                rows.append(_manifest_row(image_path, output_image, "geometry", suffix))
                created += 1

    manifest_path = root / f"{split_name}_augmentation_manifest.csv"
    _write_manifest(manifest_path, rows)
    return {
        "split_root": str(root.resolve()),
        "split": split_name,
        "source_images": len(source_images),
        "created_images": created,
        "manifest": str(manifest_path.resolve()),
    }


def _manifest_row(source: Path, output: Path, kind: str, augmentation: str) -> dict[str, str]:
    return {
        "source_image": source.name,
        "augmented_image": output.name,
        "kind": kind,
        "augmentation": augmentation,
    }


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_image", "augmented_image", "kind", "augmentation"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _factor_suffix(factor: float) -> str:
    return f"{factor:g}".replace("-", "m").replace(".", "p")
