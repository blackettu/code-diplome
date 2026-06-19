from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .schemas import CalibrationArtifact, CalibrationValidationResult


@dataclass(frozen=True)
class CalibrationValidator:
    expected_tray_type: str | None = None
    max_error_p95_mm: float | None = 2.0
    max_error_rms_mm: float | None = None
    now: datetime | None = None

    def validate(self, artifact: CalibrationArtifact) -> CalibrationValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        now = self.now or datetime.now(timezone.utc)

        if self.expected_tray_type and artifact.tray_type != self.expected_tray_type:
            errors.append(
                f"tray_type mismatch: expected {self.expected_tray_type!r}, got {artifact.tray_type!r}"
            )
        if artifact.valid_until:
            valid_until = _parse_datetime(artifact.valid_until)
            if valid_until < now:
                errors.append(f"calibration expired at {artifact.valid_until}")
        else:
            warnings.append("valid_until is missing")

        p95 = artifact.error_summary_mm.p95
        if self.max_error_p95_mm is not None:
            if p95 is None:
                errors.append("error_summary_mm.p95 is required")
            elif p95 > self.max_error_p95_mm:
                errors.append(f"error_summary_mm.p95={p95} exceeds limit {self.max_error_p95_mm}")

        rms = artifact.error_summary_mm.rms
        if self.max_error_rms_mm is not None:
            if rms is None:
                errors.append("error_summary_mm.rms is required")
            elif rms > self.max_error_rms_mm:
                errors.append(f"error_summary_mm.rms={rms} exceeds limit {self.max_error_rms_mm}")

        if artifact.px_per_mm_x <= 0.0 or artifact.px_per_mm_y <= 0.0:
            errors.append("px_per_mm_x and px_per_mm_y must be positive")

        return CalibrationValidationResult(
            ok=not errors,
            calibration_id=artifact.calibration_id,
            errors=errors,
            warnings=warnings,
            checked_at=now.isoformat(),
        )


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed
