from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schemas import CalibrationArtifact


ERROR_BUDGET_SCHEMA_VERSION = "error_budget_v0_1"
ERROR_BUDGET_COMPONENTS = (
    "e_detection",
    "e_grid",
    "e_calibration",
    "e_mechanics",
    "e_focus",
    "e_latency",
    "e_biological_target",
)


@dataclass(frozen=True)
class ErrorBudgetReport:
    components_mm: dict[str, float | None]
    total_error_mm: float | None
    complete: bool
    ok: bool
    missing_components: list[str] = field(default_factory=list)
    max_total_mm: float | None = None
    contributions: list[dict[str, float | str]] = field(default_factory=list)
    schema_version: str = ERROR_BUDGET_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "components_mm": self.components_mm,
            "total_error_mm": self.total_error_mm,
            "complete": self.complete,
            "ok": self.ok,
            "missing_components": self.missing_components,
            "max_total_mm": self.max_total_mm,
            "contributions": self.contributions,
            "metadata": self.metadata,
        }

    def to_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_error_budget_components(path: str | Path) -> dict[str, float | None]:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"Error budget input must be a JSON object: {path}")
    raw_components = data.get("components_mm", data)
    if not isinstance(raw_components, dict):
        raise ValueError("Error budget input must contain a `components_mm` object")
    return _normalize_components(raw_components)


def compute_error_budget(
    components_mm: dict[str, float | None],
    *,
    calibration: CalibrationArtifact | None = None,
    max_total_mm: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> ErrorBudgetReport:
    components = _normalize_components(components_mm)
    report_metadata = dict(metadata or {})
    if calibration is not None:
        report_metadata["calibration_id"] = calibration.calibration_id
        if components.get("e_calibration") is None and calibration.error_summary_mm.p95 is not None:
            components["e_calibration"] = float(calibration.error_summary_mm.p95)
            report_metadata["e_calibration_source"] = "calibration.error_summary_mm.p95"
    if max_total_mm is not None:
        max_total_mm = float(max_total_mm)
        if max_total_mm < 0.0:
            raise ValueError("max_total_mm must be non-negative")
    missing = [name for name in ERROR_BUDGET_COMPONENTS if components.get(name) is None]
    present_values = {name: value for name, value in components.items() if value is not None}
    variance_sum = sum(float(value) ** 2 for value in present_values.values())
    total = math.sqrt(variance_sum) if present_values else None
    complete = not missing
    ok = complete and total is not None and (max_total_mm is None or total <= max_total_mm)
    contributions = _contributions(present_values, variance_sum)
    return ErrorBudgetReport(
        components_mm=components,
        total_error_mm=total,
        complete=complete,
        ok=ok,
        missing_components=missing,
        max_total_mm=max_total_mm,
        contributions=contributions,
        metadata=report_metadata,
    )


def write_error_budget_report(
    components_path: str | Path,
    output_path: str | Path,
    *,
    calibration_path: str | Path | None = None,
    max_total_mm: float | None = None,
) -> ErrorBudgetReport:
    calibration = CalibrationArtifact.from_json(calibration_path) if calibration_path else None
    report = compute_error_budget(
        load_error_budget_components(components_path),
        calibration=calibration,
        max_total_mm=max_total_mm,
        metadata={"components_path": str(components_path), **({"calibration_path": str(calibration_path)} if calibration_path else {})},
    )
    report.to_json(output_path)
    return report


def _normalize_components(raw_components: dict[str, Any]) -> dict[str, float | None]:
    unknown = sorted(set(raw_components) - set(ERROR_BUDGET_COMPONENTS))
    if unknown:
        raise ValueError(f"Unknown error budget component(s): {', '.join(unknown)}")
    components: dict[str, float | None] = {name: None for name in ERROR_BUDGET_COMPONENTS}
    for name, value in raw_components.items():
        if value in {None, ""}:
            components[name] = None
            continue
        numeric = float(value)
        if numeric < 0.0:
            raise ValueError(f"{name} must be non-negative")
        components[name] = numeric
    return components


def _contributions(present_values: dict[str, float | None], variance_sum: float) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for name in ERROR_BUDGET_COMPONENTS:
        value = present_values.get(name)
        if value is None:
            continue
        variance = float(value) ** 2
        rows.append(
            {
                "component": name,
                "value_mm": float(value),
                "variance_mm2": variance,
                "share": variance / variance_sum if variance_sum > 0.0 else 0.0,
            }
        )
    return rows
