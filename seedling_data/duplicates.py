from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def find_cross_split_duplicates(
    manifest_path: str | Path,
    max_hamming: int = 4,
) -> dict[str, Any]:
    rows = _read_manifest(manifest_path)
    quality_errors = _manifest_quality_errors(rows)
    exact = _exact_hash_duplicates(rows)
    near, missing_files = _perceptual_duplicates(rows, max_hamming=max_hamming)
    return {
        "ok": not exact and not near and not missing_files and not any(quality_errors.values()),
        "manifest": str(manifest_path),
        "rows": len(rows),
        **quality_errors,
        "exact_sha256_duplicates": exact,
        "near_perceptual_duplicates": near,
        "missing_files": missing_files,
        "max_hamming": max_hamming,
    }


def _read_manifest(path: str | Path) -> list[dict[str, str]]:
    manifest = Path(path)
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _manifest_quality_errors(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    missing_split_rows = []
    missing_sha256_rows = []
    missing_file_path_rows = []
    for row in rows:
        reference = _row_reference(row)
        if not row.get("split"):
            missing_split_rows.append(reference)
        if not row.get("sha256"):
            missing_sha256_rows.append(reference)
        if not row.get("file_path"):
            missing_file_path_rows.append(reference)
    return {
        "missing_split_rows": missing_split_rows,
        "missing_sha256_rows": missing_sha256_rows,
        "missing_file_path_rows": missing_file_path_rows,
    }


def _exact_hash_duplicates(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_hash: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        digest = row.get("sha256", "")
        if not digest:
            continue
        by_hash.setdefault(digest, []).append(row)
    duplicates = []
    for digest, items in sorted(by_hash.items()):
        splits = {item.get("split", "") for item in items}
        if len(items) > 1 and len(splits) > 1:
            duplicates.append(
                {
                    "sha256": digest,
                    "splits": sorted(splits),
                    "images": [_image_id(item) for item in items],
                }
            )
    return duplicates


def _perceptual_duplicates(rows: list[dict[str, str]], max_hamming: int) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    hashes = []
    missing_files = []
    for row in rows:
        path_value = row.get("file_path", "")
        if not path_value:
            continue
        image_path = Path(path_value)
        if not image_path.exists():
            missing_files.append(
                {
                    "image": _image_id(row),
                    "file_path": path_value,
                    "split": row.get("split", ""),
                }
            )
            continue
        hashes.append((row, _average_hash(image_path)))
    duplicates = []
    for left_index, (left_row, left_hash) in enumerate(hashes):
        for right_row, right_hash in hashes[left_index + 1 :]:
            if left_row.get("split", "") == right_row.get("split", ""):
                continue
            distance = _hamming(left_hash, right_hash)
            if distance <= max_hamming:
                duplicates.append(
                    {
                        "left": _image_id(left_row),
                        "right": _image_id(right_row),
                        "left_split": left_row.get("split", ""),
                        "right_split": right_row.get("split", ""),
                        "hamming": distance,
                    }
                )
    return duplicates, missing_files


def _average_hash(path: Path, size: int = 8) -> int:
    with Image.open(path) as image:
        gray = image.convert("L").resize((size, size))
    pixels = np.asarray(gray, dtype=np.float32)
    average = float(pixels.mean())
    bits = pixels >= average
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bool(bit))
    return value


def _hamming(left: int, right: int) -> int:
    return int((left ^ right).bit_count())


def _image_id(row: dict[str, str]) -> str:
    return row.get("image_id") or Path(row.get("file_path", "")).name


def _row_reference(row: dict[str, str]) -> dict[str, str]:
    return {
        "image": _image_id(row),
        "file_path": row.get("file_path", ""),
        "split": row.get("split", ""),
    }
