from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .ontology import Ontology, load_ontology


@dataclass(frozen=True)
class CellAnnotation:
    image_id: str
    cell_id: str
    row: int
    col: int
    state: str
    object_ids: list[str] = field(default_factory=list)
    keep_object_id: str | None = None
    human_review_required: bool = False
    annotator_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_id", str(self.image_id))
        object.__setattr__(self, "cell_id", str(self.cell_id))
        object.__setattr__(self, "row", int(self.row))
        object.__setattr__(self, "col", int(self.col))
        object.__setattr__(self, "state", str(self.state))
        object.__setattr__(self, "object_ids", [str(value) for value in self.object_ids])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CellAnnotation":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionPointAnnotation:
    image_id: str
    cell_id: str
    object_id: str
    target_type: str
    action_point_px: list[float]
    action_point_mm: list[float] | None = None
    point_type: str = "bbox_center"
    uncertainty_radius_px: float | None = None
    min_distance_to_keep_px: float | None = None
    forbidden_zone_ids: list[str] = field(default_factory=list)
    human_review_required: bool = False
    decision_rule: str = "expert"
    annotator_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_id", str(self.image_id))
        object.__setattr__(self, "cell_id", str(self.cell_id))
        object.__setattr__(self, "object_id", str(self.object_id))
        object.__setattr__(self, "target_type", str(self.target_type))
        object.__setattr__(self, "action_point_px", _float_list("action_point_px", self.action_point_px, 2))
        if self.action_point_mm is not None:
            object.__setattr__(self, "action_point_mm", _float_list("action_point_mm", self.action_point_mm, 2))
        if self.uncertainty_radius_px is not None:
            object.__setattr__(self, "uncertainty_radius_px", float(self.uncertainty_radius_px))
        if self.min_distance_to_keep_px is not None:
            object.__setattr__(self, "min_distance_to_keep_px", float(self.min_distance_to_keep_px))
        object.__setattr__(self, "forbidden_zone_ids", [str(value) for value in self.forbidden_zone_ids])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionPointAnnotation":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_cell_annotations(path: str | Path) -> list[CellAnnotation]:
    return [CellAnnotation.from_dict(row) for row in _read_jsonl(path)]


def read_action_points(path: str | Path) -> list[ActionPointAnnotation]:
    return [ActionPointAnnotation.from_dict(row) for row in _read_jsonl(path)]


def write_cell_annotations(path: str | Path, rows: Iterable[CellAnnotation]) -> None:
    _write_jsonl(path, [row.to_dict() for row in rows])


def write_action_points(path: str | Path, rows: Iterable[ActionPointAnnotation]) -> None:
    _write_jsonl(path, [row.to_dict() for row in rows])


def audit_cell_annotations(
    path: str | Path,
    ontology_path: str | Path | None = None,
    grid_rows: int | None = None,
    grid_cols: int | None = None,
) -> dict[str, Any]:
    ontology = load_ontology(ontology_path) if ontology_path else None
    rows = read_cell_annotations(path)
    errors: list[str] = []
    warnings: list[str] = []
    seen: set[tuple[str, str]] = set()
    state_counts: Counter[str] = Counter()

    for index, row in enumerate(rows, 1):
        location = f"{path}:{index}"
        if row.row < 0 or row.col < 0:
            errors.append(f"{location}: row/col must be non-negative")
        if grid_rows is not None and row.row >= grid_rows:
            errors.append(f"{location}: row {row.row} is outside grid_rows={grid_rows}")
        if grid_cols is not None and row.col >= grid_cols:
            errors.append(f"{location}: col {row.col} is outside grid_cols={grid_cols}")
        if ontology and row.state not in ontology.cell_states:
            errors.append(f"{location}: unknown cell state {row.state!r}")
        if row.keep_object_id and row.object_ids and row.keep_object_id not in row.object_ids:
            warnings.append(f"{location}: keep_object_id is not present in object_ids")
        key = (row.image_id, row.cell_id)
        if key in seen:
            errors.append(f"{location}: duplicate cell annotation for {row.image_id}/{row.cell_id}")
        seen.add(key)
        state_counts[row.state] += 1

    return {
        "ok": not errors,
        "rows": len(rows),
        "errors": errors,
        "warnings": warnings,
        "state_counts": dict(sorted(state_counts.items())),
    }


def audit_action_points(
    path: str | Path,
    ontology_path: str | Path | None = None,
    cell_annotations_path: str | Path | None = None,
    min_safe_distance_px: float | None = None,
    max_uncertainty_px: float | None = None,
) -> dict[str, Any]:
    ontology = load_ontology(ontology_path) if ontology_path else None
    rows = read_action_points(path)
    known_cells = _known_cells(cell_annotations_path) if cell_annotations_path else {}
    min_safe_distance_px = float(min_safe_distance_px) if min_safe_distance_px is not None else None
    max_uncertainty_px = float(max_uncertainty_px) if max_uncertainty_px is not None else None
    errors: list[str] = []
    warnings: list[str] = []
    target_counts: Counter[str] = Counter()
    review_required_rows = 0
    unsafe_distance_rows = 0
    high_uncertainty_rows = 0
    forbidden_zone_rows = 0
    crop_and_weed_action_rows = 0

    for index, row in enumerate(rows, 1):
        location = f"{path}:{index}"
        cell = known_cells.get((row.image_id, row.cell_id))
        if ontology and row.target_type not in ontology.action_labels:
            errors.append(f"{location}: unknown target_type {row.target_type!r}")
        if row.action_point_px[0] < 0 or row.action_point_px[1] < 0:
            errors.append(f"{location}: action_point_px must be non-negative")
        if row.uncertainty_radius_px is not None and row.uncertainty_radius_px < 0:
            errors.append(f"{location}: uncertainty_radius_px must be non-negative")
        if row.min_distance_to_keep_px is not None and row.min_distance_to_keep_px < 0:
            errors.append(f"{location}: min_distance_to_keep_px must be non-negative")
        if known_cells and cell is None:
            warnings.append(f"{location}: cell_id is not present in cell annotations")
        if row.human_review_required:
            review_required_rows += 1
        if cell is not None and cell.state == "crop_and_weed" and _is_actionable_target(row.target_type):
            crop_and_weed_action_rows += 1
            if not row.human_review_required:
                errors.append(f"{location}: crop_and_weed action point requires human_review_required=true")
        if row.forbidden_zone_ids:
            forbidden_zone_rows += 1
            if not row.human_review_required:
                errors.append(f"{location}: forbidden_zone_ids require human_review_required=true")
        if row.target_type == "human_review_required" and not row.human_review_required:
            errors.append(f"{location}: target_type human_review_required must set human_review_required=true")
        if min_safe_distance_px is not None and _is_actionable_target(row.target_type):
            if row.min_distance_to_keep_px is None:
                warnings.append(f"{location}: min_distance_to_keep_px is missing for actionable target")
            elif row.min_distance_to_keep_px < min_safe_distance_px:
                unsafe_distance_rows += 1
                if not row.human_review_required:
                    errors.append(
                        f"{location}: min_distance_to_keep_px={row.min_distance_to_keep_px} "
                        f"is below min_safe_distance_px={min_safe_distance_px} and requires human_review_required=true"
                    )
        if max_uncertainty_px is not None and row.uncertainty_radius_px is not None:
            if row.uncertainty_radius_px > max_uncertainty_px:
                high_uncertainty_rows += 1
                if not row.human_review_required:
                    errors.append(
                        f"{location}: uncertainty_radius_px={row.uncertainty_radius_px} "
                        f"exceeds max_uncertainty_px={max_uncertainty_px} and requires human_review_required=true"
                    )
        target_counts[row.target_type] += 1

    return {
        "ok": not errors,
        "rows": len(rows),
        "errors": errors,
        "warnings": warnings,
        "target_counts": dict(sorted(target_counts.items())),
        "review_required_rows": review_required_rows,
        "unsafe_distance_rows": unsafe_distance_rows,
        "high_uncertainty_rows": high_uncertainty_rows,
        "forbidden_zone_rows": forbidden_zone_rows,
        "crop_and_weed_action_rows": crop_and_weed_action_rows,
    }


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        data = json.loads(line)
        if not isinstance(data, dict):
            raise ValueError(f"{source}:{line_number}: row must be a JSON object")
        rows.append(data)
    return rows


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _known_cells(path: str | Path) -> dict[tuple[str, str], CellAnnotation]:
    return {(row.image_id, row.cell_id): row for row in read_cell_annotations(path)}


def _is_actionable_target(target_type: str) -> bool:
    return target_type in {"remove_weed", "remove_extra_crop"}


def _float_list(name: str, value: Any, expected_length: int) -> list[float]:
    if not isinstance(value, list | tuple) or len(value) != expected_length:
        raise ValueError(f"{name} must contain {expected_length} numeric values")
    return [float(item) for item in value]
