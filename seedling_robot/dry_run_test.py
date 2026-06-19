from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .adapters.dry_run_serial import DryRunSerialAdapter
from .protocol import SpeedProfile


@dataclass(frozen=True)
class DryRunControlPoint:
    point_id: str
    expected_mm: list[float]

    def __post_init__(self) -> None:
        if len(self.expected_mm) != 3:
            raise ValueError("expected_mm must contain [x, y, z]")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DryRunControlPoint":
        return cls(point_id=str(data["point_id"]), expected_mm=[float(value) for value in data["expected_mm"]])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_dry_run_control_points(
    points_path: str | Path,
    output_path: str | Path,
    command_log_path: str | Path | None = None,
    tolerance_mm: float = 1.0,
) -> dict[str, Any]:
    points = load_control_points(points_path)
    adapter = DryRunSerialAdapter(command_log_path=command_log_path)
    adapter.connect()
    adapter.home()
    rows = []
    for point in points:
        result = adapter.move_to(point.expected_mm, SpeedProfile())
        actual = result.end_mm
        error = _distance(point.expected_mm, actual)
        rows.append(
            {
                "point_id": point.point_id,
                "expected_mm": point.expected_mm,
                "actual_mm": actual,
                "error_mm": error,
                "ok": error <= tolerance_mm,
            }
        )
    errors = [row["error_mm"] for row in rows]
    payload = {
        "ok": all(row["ok"] for row in rows),
        "points": rows,
        "tolerance_mm": tolerance_mm,
        "error_summary_mm": _summary(errors),
        "command_log": str(command_log_path) if command_log_path else None,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_control_points(path: str | Path) -> list[DryRunControlPoint]:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    raw_points = data.get("points") if isinstance(data, dict) else data
    if not isinstance(raw_points, list):
        raise ValueError("Control points file must contain a list or {'points': [...]}")
    return [DryRunControlPoint.from_dict(item) for item in raw_points]


def _distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "mean": sum(values) / len(values),
        "p50": _percentile(ordered, 50),
        "p95": _percentile(ordered, 95),
        "p99": _percentile(ordered, 99),
        "max": ordered[-1],
    }


def _percentile(ordered_values: list[float], percentile: float) -> float | None:
    if not ordered_values:
        return None
    if len(ordered_values) == 1:
        return ordered_values[0]
    position = (len(ordered_values) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered_values[int(position)]
    weight = position - lower
    return ordered_values[lower] * (1.0 - weight) + ordered_values[upper] * weight
