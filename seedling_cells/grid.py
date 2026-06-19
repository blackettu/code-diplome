from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from seedling_core.schemas import cell_id


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


@dataclass(frozen=True)
class GridCell:
    row: int
    col: int
    bbox_xyxy_px: list[float]
    polygon_px: list[list[float]]

    @property
    def cell_id(self) -> str:
        return cell_id(self.row, self.col)


def generate_grid(bbox: list[float], rows: int, cols: int) -> list[list[tuple[float, float, float, float]]]:
    _validate_grid(rows, cols)
    x_min, y_min, x_max, y_max = _bbox(bbox)
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


def generate_grid_cells(bbox: list[float], rows: int, cols: int) -> list[GridCell]:
    cells: list[GridCell] = []
    for row_index, row in enumerate(generate_grid(bbox, rows, cols)):
        for col_index, cell_bbox in enumerate(row):
            x1, y1, x2, y2 = cell_bbox
            cells.append(
                GridCell(
                    row=row_index,
                    col=col_index,
                    bbox_xyxy_px=[x1, y1, x2, y2],
                    polygon_px=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                )
            )
    return cells


def generate_grid_polygons(
    corners_px: list[list[float]],
    rows: int,
    cols: int,
) -> list[GridCell]:
    """Generate cells from tray corners ordered TL, TR, BR, BL."""
    _validate_grid(rows, cols)
    if len(corners_px) != 4:
        raise ValueError("corners_px must contain four points ordered TL, TR, BR, BL")
    tl, tr, br, bl = [tuple(_point(point)) for point in corners_px]
    cells: list[GridCell] = []
    for row in range(rows):
        v0 = row / rows
        v1 = (row + 1) / rows
        for col in range(cols):
            u0 = col / cols
            u1 = (col + 1) / cols
            polygon = [
                _bilinear(tl, tr, br, bl, u0, v0),
                _bilinear(tl, tr, br, bl, u1, v0),
                _bilinear(tl, tr, br, bl, u1, v1),
                _bilinear(tl, tr, br, bl, u0, v1),
            ]
            xs = [point[0] for point in polygon]
            ys = [point[1] for point in polygon]
            cells.append(
                GridCell(
                    row=row,
                    col=col,
                    bbox_xyxy_px=[min(xs), min(ys), max(xs), max(ys)],
                    polygon_px=[list(point) for point in polygon],
                )
            )
    return cells


def cell_index_for_grid_cells(
    grid_cells: Iterable[GridCell],
    point: tuple[float, float],
) -> tuple[int, int] | None:
    """Return the first grid cell whose polygon contains point."""
    px, py = float(point[0]), float(point[1])
    for grid_cell in grid_cells:
        if point_in_polygon((px, py), grid_cell.polygon_px):
            return grid_cell.row, grid_cell.col
    return None


def point_in_polygon(point: tuple[float, float], polygon: list[list[float]]) -> bool:
    """Ray-casting point-in-polygon test with boundary treated as inside."""
    if len(polygon) < 3:
        raise ValueError("polygon must contain at least three points")
    x, y = float(point[0]), float(point[1])
    vertices = [_point(vertex) for vertex in polygon]
    inside = False
    previous = vertices[-1]
    for current in vertices:
        if _point_on_segment((x, y), previous, current):
            return True
        xi, yi = current
        xj, yj = previous
        intersects = (yi > y) != (yj > y)
        if intersects:
            x_intersection = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x <= x_intersection:
                inside = not inside
        previous = current
    return inside


def cell_index(bbox: list[float], point: tuple[float, float], rows: int, cols: int) -> tuple[int, int] | None:
    _validate_grid(rows, cols)
    x_min, y_min, x_max, y_max = _bbox(bbox)
    px, py = point
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
    ax1, ay1, ax2, ay2 = _bbox(a)
    bx1, by1, bx2, by2 = _bbox(b)
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
    if len(a) != 2 or len(b) != 2:
        raise ValueError("center_distance expects two [x, y] points")
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _validate_grid(rows: int, cols: int) -> None:
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")


def _bbox(bbox: list[float]) -> tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise ValueError("bbox must contain [x1, y1, x2, y2]")
    x1, y1, x2, y2 = [float(value) for value in bbox]
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must have positive width and height")
    return x1, y1, x2, y2


def _point(point: list[float]) -> tuple[float, float]:
    if len(point) != 2:
        raise ValueError("point must contain [x, y]")
    return float(point[0]), float(point[1])


def _bilinear(
    tl: tuple[float, float],
    tr: tuple[float, float],
    br: tuple[float, float],
    bl: tuple[float, float],
    u: float,
    v: float,
) -> tuple[float, float]:
    top = (tl[0] * (1.0 - u) + tr[0] * u, tl[1] * (1.0 - u) + tr[1] * u)
    bottom = (bl[0] * (1.0 - u) + br[0] * u, bl[1] * (1.0 - u) + br[1] * u)
    return top[0] * (1.0 - v) + bottom[0] * v, top[1] * (1.0 - v) + bottom[1] * v


def _point_on_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
    eps: float = 1e-9,
) -> bool:
    px, py = point
    sx, sy = start
    ex, ey = end
    cross = (px - sx) * (ey - sy) - (py - sy) * (ex - sx)
    if abs(cross) > eps:
        return False
    return min(sx, ex) - eps <= px <= max(sx, ex) + eps and min(sy, ey) - eps <= py <= max(sy, ey) + eps
