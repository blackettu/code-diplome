from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".heic", ".HEIC"}


@dataclass(frozen=True)
class YoloLabel:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    @classmethod
    def from_line(cls, line: str) -> "YoloLabel":
        parts = line.strip().split()
        if len(parts) < 5:
            raise ValueError(f"Invalid YOLO label line: {line!r}")
        return cls(int(float(parts[0])), *(float(value) for value in parts[1:5]))

    def to_line(self) -> str:
        return (
            f"{self.class_id} {self.x_center:.8f} {self.y_center:.8f} "
            f"{self.width:.8f} {self.height:.8f}"
        )

    def to_xyxy(self, image_width: int, image_height: int) -> list[float]:
        cx = self.x_center * image_width
        cy = self.y_center * image_height
        w = self.width * image_width
        h = self.height * image_height
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    def hflip(self) -> "YoloLabel":
        return YoloLabel(
            self.class_id,
            1.0 - self.x_center,
            self.y_center,
            self.width,
            self.height,
        )

    def vflip(self) -> "YoloLabel":
        return YoloLabel(
            self.class_id,
            self.x_center,
            1.0 - self.y_center,
            self.width,
            self.height,
        )


def list_images(images_dir: str | Path) -> list[Path]:
    root = Path(images_dir)
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in {suffix.lower() for suffix in IMAGE_SUFFIXES}
    )


def read_labels(path: str | Path) -> list[YoloLabel]:
    label_path = Path(path)
    if not label_path.exists():
        return []
    labels: list[YoloLabel] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            labels.append(YoloLabel.from_line(stripped))
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_number}: {exc}") from exc
    return labels


def write_labels(path: str | Path, labels: Iterable[YoloLabel]) -> None:
    label_path = Path(path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(
        "\n".join(label.to_line() for label in labels) + "\n",
        encoding="utf-8",
    )


def label_path_for(image_path: Path, labels_dir: Path) -> Path:
    return labels_dir / f"{image_path.stem}.txt"


def write_data_yaml(
    path: str | Path,
    dataset_root: str | Path,
    class_names: list[str],
    include_test: bool = True,
) -> None:
    output = Path(path)
    dataset = Path(dataset_root).resolve()
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(class_names))
    lines = [
        f"path: {dataset.as_posix()}",
        "train: train/images",
        "val: val/images",
    ]
    if include_test:
        lines.append("test: test/images")
    lines.append("names:")
    lines.append(names)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
