from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HIL_REVIEW_SCHEMA_VERSION = "hardware_in_loop_review_v0_1"
SAFE_TOOL_PROFILES = {"pointer_only"}
SAFE_MODES = {"hardware_in_loop_pointer", "dry_run_pointer"}


@dataclass(frozen=True)
class HardwareInLoopReview:
    review_id: str
    reviewer: str
    approved_at: str
    expires_at: str
    gantry_id: str
    allowed_modes: list[str] = field(default_factory=lambda: ["hardware_in_loop_pointer"])
    allowed_tool_profiles: list[str] = field(default_factory=lambda: ["pointer_only"])
    calibration_id: str | None = None
    max_error_p95_mm: float = 2.0
    positioning_error_p95_mm: float | None = None
    interlock_required: bool = True
    limit_switch_required: bool = True
    emergency_stop_tested: bool = False
    allow_real_actuation: bool = False
    notes: str = ""
    schema_version: str = HIL_REVIEW_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _parse_datetime(self.approved_at, "approved_at")
        _parse_datetime(self.expires_at, "expires_at")
        if self.max_error_p95_mm < 0:
            raise ValueError("max_error_p95_mm must be non-negative")
        if self.positioning_error_p95_mm is not None and self.positioning_error_p95_mm < 0:
            raise ValueError("positioning_error_p95_mm must be non-negative")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HardwareInLoopReview":
        payload = dict(data)
        schema_version = payload.pop("schema_version", HIL_REVIEW_SCHEMA_VERSION)
        if schema_version != HIL_REVIEW_SCHEMA_VERSION:
            raise ValueError(f"unsupported HIL review schema_version: {schema_version}")
        return cls(**payload)

    @classmethod
    def from_json(cls, path: str | Path) -> "HardwareInLoopReview":
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError(f"HIL review artifact must be a JSON object: {path}")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def validate_hil_review(
    review: HardwareInLoopReview,
    now: datetime | None = None,
    required_mode: str | None = None,
    required_tool_profile: str | None = None,
    require_positioning_error: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    now = now or datetime.now(timezone.utc)
    if _parse_datetime(review.expires_at, "expires_at") < now:
        errors.append("hil_review_expired")
    if not review.reviewer:
        errors.append("reviewer_required")
    if review.allow_real_actuation:
        errors.append("real_actuation_not_allowed_in_current_project")
    unsupported_modes = sorted(set(review.allowed_modes) - SAFE_MODES)
    if unsupported_modes:
        errors.append(f"unsupported_modes:{','.join(unsupported_modes)}")
    if required_mode is not None and required_mode not in review.allowed_modes:
        errors.append(f"mode_not_authorized:{required_mode}")
    unsupported_profiles = sorted(set(review.allowed_tool_profiles) - SAFE_TOOL_PROFILES)
    if unsupported_profiles:
        errors.append(f"unsupported_tool_profiles:{','.join(unsupported_profiles)}")
    if required_tool_profile is not None and required_tool_profile not in review.allowed_tool_profiles:
        errors.append(f"tool_profile_not_authorized:{required_tool_profile}")
    if review.interlock_required is not True:
        errors.append("interlock_must_be_required")
    if review.limit_switch_required is not True:
        errors.append("limit_switch_must_be_required")
    if review.positioning_error_p95_mm is None:
        if require_positioning_error:
            errors.append("positioning_error_p95_mm_required")
        else:
            warnings.append("positioning_error_p95_mm_missing")
    elif review.positioning_error_p95_mm > review.max_error_p95_mm:
        errors.append("positioning_error_p95_mm_exceeds_limit")
    if not review.emergency_stop_tested:
        warnings.append("emergency_stop_test_not_recorded")
    if review.calibration_id is None:
        warnings.append("calibration_id_missing")
    return {
        "ok": not errors,
        "review_id": review.review_id,
        "gantry_id": review.gantry_id,
        "emergency_stop_tested": review.emergency_stop_tested,
        "interlock_required": review.interlock_required,
        "limit_switch_required": review.limit_switch_required,
        "allowed_modes": list(review.allowed_modes),
        "allowed_tool_profiles": list(review.allowed_tool_profiles),
        "allow_real_actuation": review.allow_real_actuation,
        "calibration_id": review.calibration_id,
        "max_error_p95_mm": review.max_error_p95_mm,
        "positioning_error_p95_mm": review.positioning_error_p95_mm,
        "errors": errors,
        "warnings": warnings,
    }


def _parse_datetime(value: str, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO datetime") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed
