from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

from .schemas import CalibrationArtifact, CalibrationConfig, ErrorSummary
from .transforms import apply_homography


def load_calibration_config(path: str | Path) -> CalibrationConfig:
    data = _load_mapping(path)
    return CalibrationConfig.from_dict(data)


def estimate_calibration_artifact(
    config: CalibrationConfig,
    calibration_id: str = "calibration_v0",
    created_at: str | None = None,
    tool_offset_mm: list[float] | None = None,
) -> CalibrationArtifact:
    if len(config.target_points_px) < 4:
        raise ValueError("At least four image/tray point pairs are required for homography estimation")
    created = _parse_or_now(created_at)
    image_to_tray = estimate_homography(config.target_points_px, config.target_points_tray_mm)
    tray_to_robot = _estimate_tray_to_robot(config)
    residuals = calibration_residuals(config, image_to_tray)
    error_summary = _error_summary(residuals)
    px_per_mm_x, px_per_mm_y = _px_per_mm(config)
    valid_until = created + timedelta(hours=config.valid_hours)
    return CalibrationArtifact(
        calibration_id=calibration_id,
        created_at=created.isoformat(),
        camera_id=config.camera_id,
        tray_type=config.tray_type,
        image_to_tray_homography=image_to_tray,
        tray_to_robot_transform=tray_to_robot,
        tool_offset_mm=tool_offset_mm or [0.0, 0.0, 0.0],
        px_per_mm_x=px_per_mm_x,
        px_per_mm_y=px_per_mm_y,
        valid_until=valid_until.isoformat(),
        error_summary_mm=error_summary,
        metadata={
            "grid_rows": config.grid_rows,
            "grid_cols": config.grid_cols,
            "target_points": len(config.target_points_px),
            "estimator": "numpy_dlt_v0",
        },
    )


def estimate_homography(
    source_points: list[list[float]],
    destination_points: list[list[float]],
) -> list[list[float]]:
    if len(source_points) != len(destination_points):
        raise ValueError("source_points and destination_points must have the same length")
    if len(source_points) < 4:
        raise ValueError("At least four point pairs are required")
    rows = []
    for source, destination in zip(source_points, destination_points):
        x, y = float(source[0]), float(source[1])
        u, v = float(destination[0]), float(destination[1])
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    matrix = vh[-1].reshape(3, 3)
    if abs(matrix[2, 2]) < 1e-12:
        raise ValueError("Estimated homography is degenerate")
    matrix = matrix / matrix[2, 2]
    return [[float(value) for value in row] for row in matrix.tolist()]


def calibration_residuals(
    config: CalibrationConfig,
    image_to_tray_homography: list[list[float]],
) -> list[dict[str, Any]]:
    rows = []
    for index, (point_px, expected_mm) in enumerate(zip(config.target_points_px, config.target_points_tray_mm)):
        predicted = apply_homography(point_px, image_to_tray_homography)
        error = math.hypot(predicted[0] - expected_mm[0], predicted[1] - expected_mm[1])
        rows.append(
            {
                "point_index": index,
                "image_px": point_px,
                "expected_tray_mm": expected_mm,
                "predicted_tray_mm": predicted,
                "error_mm": error,
            }
        )
    return rows


def write_error_map(
    config_path: str | Path,
    calibration_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    config = load_calibration_config(config_path)
    artifact = CalibrationArtifact.from_json(calibration_path)
    residuals = calibration_residuals(config, artifact.image_to_tray_homography)
    payload = {
        "calibration_id": artifact.calibration_id,
        "camera_id": artifact.camera_id,
        "tray_type": artifact.tray_type,
        "tray_size_mm": config.tray_size_mm,
        "grid_rows": config.grid_rows,
        "grid_cols": config.grid_cols,
        "rows": residuals,
        "error_summary_mm": _error_summary(residuals).to_dict(),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() in {".html", ".htm"}:
        output.write_text(_error_map_html(payload), encoding="utf-8")
    else:
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _error_map_html(payload: dict[str, Any]) -> str:
    tray_size = payload.get("tray_size_mm") or [1.0, 1.0]
    tray_width = max(float(tray_size[0]), 1.0)
    tray_height = max(float(tray_size[1]), 1.0)
    rows = payload.get("rows", [])
    max_error = max((float(row.get("error_mm", 0.0) or 0.0) for row in rows), default=0.0)
    width = 720
    height = max(360, int(width * tray_height / tray_width))
    scale_x = width / tray_width
    scale_y = height / tray_height
    grid = _error_map_grid(payload, width, height)
    points = []
    for row in rows:
        expected = row.get("expected_tray_mm") or [0.0, 0.0]
        predicted = row.get("predicted_tray_mm") or expected
        error = float(row.get("error_mm", 0.0) or 0.0)
        x = float(expected[0]) * scale_x
        y = float(expected[1]) * scale_y
        px = float(predicted[0]) * scale_x
        py = float(predicted[1]) * scale_y
        color = _error_color(error, max_error)
        points.append(
            f'<line x1="{x:.2f}" y1="{y:.2f}" x2="{px:.2f}" y2="{py:.2f}" stroke="#5b6470" stroke-width="1" />'
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="{color}" stroke="#20252b" stroke-width="1">'
            f"<title>point {row.get('point_index')}: {error:.3f} mm</title></circle>"
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3" fill="#111827">'
            f"<title>predicted point {row.get('point_index')}</title></circle>"
        )
    table = _error_rows_table(rows)
    summary = payload.get("error_summary_mm", {})
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Calibration Error Map: {escape(str(payload.get("calibration_id", "")))}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2528; }}
    .meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 8px 16px; margin-bottom: 18px; }}
    .map {{ border: 1px solid #c8d0d8; max-width: 100%; background: #fafbfc; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 18px; font-size: 13px; }}
    td, th {{ border: 1px solid #ddd; padding: 6px 8px; text-align: right; }}
    td:first-child, th:first-child {{ text-align: left; }}
    th {{ background: #f2f2ed; }}
    code {{ font-family: Consolas, monospace; }}
  </style>
</head>
<body>
  <h1>Calibration Error Map</h1>
  <section class="meta">
    <div>Calibration: <code>{escape(str(payload.get("calibration_id")))}</code></div>
    <div>Camera: <code>{escape(str(payload.get("camera_id")))}</code></div>
    <div>Tray: <code>{escape(str(payload.get("tray_type")))}</code></div>
    <div>P95 error: <code>{escape(str(summary.get("p95")))}</code> mm</div>
    <div>RMS error: <code>{escape(str(summary.get("rms")))}</code> mm</div>
    <div>Max error: <code>{escape(str(summary.get("max")))}</code> mm</div>
  </section>
  <svg class="map" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="Calibration residual heatmap">
    <rect x="0" y="0" width="{width}" height="{height}" fill="#fbfcfd" />
    {grid}
    {"".join(points)}
  </svg>
  <p>Colored circles mark expected control points; black dots mark predicted tray coordinates. Darker red means higher residual error.</p>
  {table}
</body>
</html>
"""


def _error_map_grid(payload: dict[str, Any], width: int, height: int) -> str:
    grid_rows = max(int(payload.get("grid_rows") or 1), 1)
    grid_cols = max(int(payload.get("grid_cols") or 1), 1)
    parts = []
    for col in range(1, grid_cols):
        x = width * col / grid_cols
        parts.append(f'<line x1="{x:.2f}" y1="0" x2="{x:.2f}" y2="{height}" stroke="#e5e9ef" stroke-width="1" />')
    for row in range(1, grid_rows):
        y = height * row / grid_rows
        parts.append(f'<line x1="0" y1="{y:.2f}" x2="{width}" y2="{y:.2f}" stroke="#e5e9ef" stroke-width="1" />')
    return "\n".join(parts)


def _error_color(error: float, max_error: float) -> str:
    if max_error <= 0:
        return "#2ca25f"
    ratio = max(0.0, min(1.0, error / max_error))
    red = int(44 + ratio * 211)
    green = int(162 - ratio * 107)
    blue = int(95 - ratio * 59)
    return f"#{red:02x}{green:02x}{blue:02x}"


def _error_rows_table(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{escape(str(row.get('point_index')))}</td>"
            f"<td>{escape(str(row.get('expected_tray_mm')))}</td>"
            f"<td>{escape(str(row.get('predicted_tray_mm')))}</td>"
            f"<td>{float(row.get('error_mm', 0.0) or 0.0):.6f}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Point</th><th>Expected tray mm</th><th>Predicted tray mm</th><th>Error mm</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def _estimate_tray_to_robot(config: CalibrationConfig) -> list[list[float]]:
    if len(config.robot_reference_points_mm) >= 4 and len(config.robot_reference_points_mm) == len(config.target_points_tray_mm):
        robot_xy = [[point[0], point[1]] for point in config.robot_reference_points_mm]
        return estimate_homography(config.target_points_tray_mm, robot_xy)
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _error_summary(residuals: list[dict[str, Any]]) -> ErrorSummary:
    values = np.asarray([float(row["error_mm"]) for row in residuals], dtype=np.float64)
    if values.size == 0:
        return ErrorSummary()
    return ErrorSummary(
        p50=float(np.percentile(values, 50)),
        p95=float(np.percentile(values, 95)),
        p99=float(np.percentile(values, 99)),
        rms=float(math.sqrt(float(np.mean(values**2)))),
        max=float(np.max(values)),
    )


def _px_per_mm(config: CalibrationConfig) -> tuple[float, float]:
    px = np.asarray(config.target_points_px, dtype=np.float64)
    mm = np.asarray(config.target_points_tray_mm, dtype=np.float64)
    px_range = np.ptp(px, axis=0)
    mm_range = np.ptp(mm, axis=0)
    x = float(px_range[0] / mm_range[0]) if mm_range[0] > 0 else 1.0
    y = float(px_range[1] / mm_range[1]) if mm_range[1] > 0 else 1.0
    return max(x, 1e-9), max(y, 1e-9)


def _parse_or_now(value: str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _load_mapping(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    text = source.read_text(encoding="utf-8-sig")
    if source.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to load calibration config files") from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Calibration config must be a mapping: {source}")
    return data
