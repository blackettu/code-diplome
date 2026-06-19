from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from seedling_core.schemas import ActionTarget, CellState, DetectionResultV1


BBOX_COLORS = {
    "container": (80, 120, 220),
    "crop_seedling": (20, 150, 70),
    "seedlings": (20, 150, 70),
    "seedling": (20, 150, 70),
    "weed": (220, 120, 20),
}
GRID_COLOR = (80, 150, 240)
TARGET_COLOR = (240, 80, 40)
REVIEW_TARGET_COLOR = (220, 80, 160)
ERROR_COLOR = (220, 30, 50)
LABEL_BG = (255, 255, 255)


def write_detection_overlay(
    image_path: str | Path,
    result: DetectionResultV1,
    output_path: str | Path,
    cells: list[CellState] | None = None,
    targets: list[ActionTarget] | None = None,
    errors: list[dict[str, Any]] | None = None,
) -> None:
    with Image.open(image_path) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    cells_by_id = {cell.cell_id: cell for cell in cells or []}
    targets_by_id = {target.target_id: target for target in targets or []}
    detections_by_id = {detection.object_id: detection for detection in result.detections}

    for detection in result.detections:
        color = _class_color(detection.class_name)
        draw.rectangle(detection.bbox_xyxy_px, outline=color, width=2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        _draw_label(draw, (detection.bbox_xyxy_px[0] + 2, detection.bbox_xyxy_px[1] + 2), label, color)
    for cell in cells or []:
        polygon = [tuple(point) for point in cell.polygon_px]
        color = ERROR_COLOR if cell.risk_flags else GRID_COLOR
        draw.line([*polygon, polygon[0]], fill=color, width=1)
        if cell.object_ids or cell.human_review_required or cell.risk_flags:
            cx, cy = _polygon_center(cell.polygon_px)
            state = cell.state.replace("_", " ")
            _draw_label(draw, (cx + 2, cy + 2), f"{cell.cell_id} {state}", color)
    for target in targets or []:
        color = REVIEW_TARGET_COLOR if target.human_review_required or target.forbidden_zone_ids else TARGET_COLOR
        _draw_target_marker(draw, target.action_point_px, color)
        label = target.target_id
        if target.risk_score:
            label = f"{label} risk={target.risk_score:.2f}"
        _draw_label(draw, (target.action_point_px[0] + 6, target.action_point_px[1] + 6), label, color)
    for issue in errors or []:
        _draw_issue(draw, issue, cells_by_id, targets_by_id, detections_by_id)
    if errors:
        _draw_issue_legend(draw, errors)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _class_color(class_name: str) -> tuple[int, int, int]:
    return BBOX_COLORS.get(class_name, (180, 60, 160))


def _draw_label(
    draw: ImageDraw.ImageDraw,
    position: tuple[float, float],
    text: str,
    fill: tuple[int, int, int],
) -> None:
    x, y = position
    text = str(text)
    bbox = draw.textbbox((x, y), text)
    draw.rectangle([bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1], fill=LABEL_BG)
    draw.text((x, y), text, fill=fill)


def _draw_target_marker(
    draw: ImageDraw.ImageDraw,
    point: list[float],
    color: tuple[int, int, int],
    radius: int = 5,
) -> None:
    x, y = [float(value) for value in point]
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=color, width=2)
    draw.line([(x - radius, y), (x + radius, y)], fill=color, width=2)
    draw.line([(x, y - radius), (x, y + radius)], fill=color, width=2)


def _draw_issue(
    draw: ImageDraw.ImageDraw,
    issue: dict[str, Any],
    cells_by_id: dict[str, CellState],
    targets_by_id: dict[str, ActionTarget],
    detections_by_id: dict[str, Any],
) -> None:
    bbox = _issue_bbox(issue, detections_by_id)
    point = _issue_point(issue, cells_by_id, targets_by_id, detections_by_id)
    label = _issue_label(issue)
    if bbox is not None:
        draw.rectangle(bbox, outline=ERROR_COLOR, width=3)
        _draw_label(draw, (bbox[0] + 2, bbox[1] + 14), label, ERROR_COLOR)
    if point is not None:
        _draw_error_marker(draw, point)
        _draw_label(draw, (point[0] + 7, point[1] + 7), label, ERROR_COLOR)


def _issue_bbox(issue: dict[str, Any], detections_by_id: dict[str, Any]) -> list[float] | None:
    raw = issue.get("bbox_xyxy_px") or issue.get("box") or issue.get("bbox")
    if isinstance(raw, list | tuple) and len(raw) == 4:
        return [float(value) for value in raw]
    object_id = _first_issue_value(issue, ("object_id", "pred_object_id", "gt_object_id"))
    if object_id and str(object_id) in detections_by_id:
        return [float(value) for value in detections_by_id[str(object_id)].bbox_xyxy_px]
    return None


def _issue_point(
    issue: dict[str, Any],
    cells_by_id: dict[str, CellState],
    targets_by_id: dict[str, ActionTarget],
    detections_by_id: dict[str, Any],
) -> list[float] | None:
    for key in ("point_px", "action_point_px", "remove_center", "center", "center_px"):
        raw = issue.get(key)
        if isinstance(raw, list | tuple) and len(raw) == 2:
            return [float(value) for value in raw]
    target_id = _first_issue_value(issue, ("target_id", "pred_target_id", "gt_target_id", "command_target_id"))
    if target_id and str(target_id) in targets_by_id:
        return [float(value) for value in targets_by_id[str(target_id)].action_point_px]
    object_id = _first_issue_value(issue, ("object_id", "pred_object_id", "gt_object_id"))
    if object_id and str(object_id) in detections_by_id:
        center = detections_by_id[str(object_id)].center_px
        if center is not None:
            return [float(value) for value in center]
    cell_id = _first_issue_value(issue, ("cell_id", "pred_cell_id", "gt_cell_id"))
    if cell_id and str(cell_id) in cells_by_id:
        return list(_polygon_center(cells_by_id[str(cell_id)].polygon_px))
    return None


def _draw_error_marker(draw: ImageDraw.ImageDraw, point: list[float], radius: int = 6) -> None:
    x, y = [float(value) for value in point]
    draw.line([(x - radius, y - radius), (x + radius, y + radius)], fill=ERROR_COLOR, width=3)
    draw.line([(x - radius, y + radius), (x + radius, y - radius)], fill=ERROR_COLOR, width=3)


def _issue_label(issue: dict[str, Any]) -> str:
    for key in ("label", "category", "error_type", "reason", "event", "result"):
        value = issue.get(key)
        if value:
            return str(value)
    return "error"


def _first_issue_value(issue: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = issue.get(key)
        if value is not None and value != "":
            return value
    return None


def _draw_issue_legend(draw: ImageDraw.ImageDraw, errors: list[dict[str, Any]]) -> None:
    unique_labels = []
    for issue in errors:
        label = _issue_label(issue)
        if label not in unique_labels:
            unique_labels.append(label)
    if not unique_labels:
        return
    y = 4
    for label in unique_labels[:8]:
        draw.rectangle([4, y + 2, 12, y + 10], fill=ERROR_COLOR)
        _draw_label(draw, (16, y), label, ERROR_COLOR)
        y += 13


def _polygon_center(polygon: list[list[float]]) -> tuple[float, float]:
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    return sum(xs) / len(xs), sum(ys) / len(ys)
