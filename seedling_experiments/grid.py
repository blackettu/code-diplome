from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Detection:
    box: list[float]
    class_id: int = 1
    confidence: float = 1.0
    name: str | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def generate_grid(bbox: list[float], rows: int, cols: int) -> list[list[tuple[float, float, float, float]]]:
    x_min, y_min, x_max, y_max = bbox
    cell_w = (x_max - x_min) / cols
    cell_h = (y_max - y_min) / rows
    return [
        [
            (
                x_min + col * cell_w,
                y_min + row * cell_h,
                x_min + (col + 1) * cell_w,
                y_min + (row + 1) * cell_h,
            )
            for col in range(cols)
        ]
        for row in range(rows)
    ]


def cell_index(bbox: list[float], point: tuple[float, float], rows: int, cols: int) -> tuple[int, int] | None:
    x_min, y_min, x_max, y_max = bbox
    px, py = point
    if x_max <= x_min or y_max <= y_min:
        return None
    if not (x_min <= px <= x_max and y_min <= py <= y_max):
        return None
    col = min(int((px - x_min) / (x_max - x_min) * cols), cols - 1)
    row = min(int((py - y_min) / (y_max - y_min) * rows), rows - 1)
    return row, col


def assign_to_cells(
    container_box: list[float],
    seedlings: Iterable[Detection],
    rows: int,
    cols: int,
) -> list[list[list[Detection]]]:
    cells: list[list[list[Detection]]] = [[[] for _ in range(cols)] for _ in range(rows)]
    for seedling in seedlings:
        idx = cell_index(container_box, seedling.center, rows, cols)
        if idx is not None:
            cells[idx[0]][idx[1]].append(seedling)
    return cells


def count_matrix(cells: list[list[list[Detection]]]) -> list[list[int]]:
    return [[len(cell) for cell in row] for row in cells]


def choose_removal_targets(cells: list[list[list[Detection]]]) -> list[dict[str, object]]:
    """Choose all seedlings except the largest box in each multi-seedling cell."""
    targets: list[dict[str, object]] = []
    for row_index, row in enumerate(cells):
        for col_index, detections in enumerate(row):
            if len(detections) <= 1:
                continue
            sorted_detections = sorted(detections, key=lambda item: item.area, reverse=True)
            kept = sorted_detections[0]
            for removed in sorted_detections[1:]:
                targets.append(
                    {
                        "row": row_index,
                        "col": col_index,
                        "keep_box": kept.box,
                        "remove_box": removed.box,
                        "remove_center": list(removed.center),
                    }
                )
    return targets


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    union = (
        max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        + max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        - intersection
    )
    return intersection / union if union else 0.0


def center_distance(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
