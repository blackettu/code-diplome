from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import Any

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".heic"}
MANIFEST_FIELDS = [
    "image_id",
    "file_path",
    "sha256",
    "session_id",
    "group_id",
    "tray_id",
    "site",
    "greenhouse",
    "capture_date",
    "target_species",
    "days_after_sowing",
    "grid_rows",
    "grid_cols",
    "camera_id",
    "width",
    "height",
    "split",
    "calibration_id",
    "lighting",
    "watering_state",
    "operator_id",
]


def build_image_manifest(
    dataset_root: str | Path,
    output_path: str | Path | None = None,
    images_dir: str | Path | None = None,
    split: str | None = None,
    group_regex: str | None = None,
    defaults: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    root = Path(dataset_root)
    image_root = Path(images_dir) if images_dir else root / (split or "") / "images"
    if not image_root.is_absolute():
        image_root = root / image_root if images_dir else image_root
    if not image_root.exists():
        raise FileNotFoundError(f"Images directory not found: {image_root}")
    pattern = re.compile(group_regex) if group_regex else None
    defaults = defaults or {}
    rows = []
    for image_path in _list_images(image_root):
        width, height = _image_size(image_path)
        image_id = image_path.stem
        identity = _identity_metadata(root, image_path, pattern, defaults)
        row = {field: "" for field in MANIFEST_FIELDS}
        row.update({key: str(value) for key, value in defaults.items() if key in row})
        row.update(
            {
                "image_id": image_id,
                "file_path": str(image_path.resolve()),
                "sha256": _sha256(image_path),
                "session_id": identity["session_id"],
                "group_id": identity["group_id"],
                "tray_id": identity["tray_id"],
                "width": str(width),
                "height": str(height),
                "split": split or str(defaults.get("split", "")),
            }
        )
        rows.append(row)
    if output_path:
        write_image_manifest(output_path, rows)
    return rows


def write_image_manifest(path: str | Path, rows: list[dict[str, str]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_group_map(
    manifest_path: str | Path,
    group_column: str = "group_id",
) -> dict[str, str]:
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if group_column not in (reader.fieldnames or []):
            raise ValueError(f"Manifest {path} does not contain group column {group_column!r}")
        result: dict[str, str] = {}
        for row in reader:
            group_id = row.get(group_column) or row.get("image_id") or ""
            for key in _manifest_image_keys(row):
                result[key] = group_id
        return result


def _manifest_image_keys(row: dict[str, str]) -> set[str]:
    keys = set()
    for column in ["image", "image_id", "file_path"]:
        value = row.get(column)
        if not value:
            continue
        path = Path(value)
        keys.add(value)
        keys.add(path.name)
        keys.add(path.stem)
    return keys


def _list_images(images_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _group_id(image_path: Path, pattern: re.Pattern[str] | None) -> str:
    match = pattern.search(image_path.stem) if pattern else None
    named = match.groupdict() if match else {}
    if named.get("group_id"):
        return named["group_id"]
    if pattern is None:
        return image_path.stem
    if not match:
        return image_path.stem
    if match.groups():
        return match.group(1)
    return match.group(0)


def _identity_metadata(
    dataset_root: Path,
    image_path: Path,
    pattern: re.Pattern[str] | None,
    defaults: dict[str, Any],
) -> dict[str, str]:
    match = pattern.search(image_path.stem) if pattern else None
    named = match.groupdict() if match else {}
    group_id = str(defaults.get("group_id") or _group_id(image_path, pattern))
    session_id = str(defaults.get("session_id") or named.get("session_id") or _session_id(dataset_root, image_path))
    tray_id = str(defaults.get("tray_id") or named.get("tray_id") or group_id)
    return {"session_id": session_id, "group_id": group_id, "tray_id": tray_id}


def _session_id(dataset_root: Path, image_path: Path) -> str:
    try:
        relative = image_path.resolve().relative_to(dataset_root.resolve())
    except ValueError:
        relative = image_path
    parts = relative.parts
    if len(parts) >= 3 and parts[-2].lower() == "images":
        return parts[-3]
    if len(parts) >= 2 and parts[-2].lower() != "images":
        return parts[-2]
    return dataset_root.name or image_path.parent.name
