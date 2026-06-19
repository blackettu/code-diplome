from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


POST_ACTION_OUTCOMES = {
    "target_removed",
    "target_survived",
    "crop_damage",
    "regrowth",
    "no_change",
    "not_visible",
    "uncertain",
}
DEFAULT_REQUIRED_HOURS = (24.0, 48.0, 72.0)


@dataclass(frozen=True)
class PostActionObservation:
    command_id: str
    target_id: str
    scene_id: str
    tray_id: str
    observed_at: str
    hours_after_action: float
    outcome: str
    image_ref: str | None = None
    confidence: float = 1.0
    observer_id: str | None = None
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "command_id", str(self.command_id))
        object.__setattr__(self, "target_id", str(self.target_id))
        object.__setattr__(self, "scene_id", str(self.scene_id))
        object.__setattr__(self, "tray_id", str(self.tray_id))
        object.__setattr__(self, "observed_at", str(self.observed_at))
        object.__setattr__(self, "hours_after_action", float(self.hours_after_action))
        object.__setattr__(self, "outcome", str(self.outcome))
        object.__setattr__(self, "confidence", float(self.confidence))
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.hours_after_action < 0.0:
            raise ValueError("hours_after_action must be non-negative")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PostActionObservation":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_post_action_observations(path: str | Path) -> list[PostActionObservation]:
    return [PostActionObservation.from_dict(row) for row in _read_jsonl(path)]


def write_post_action_observations(path: str | Path, rows: Iterable[PostActionObservation]) -> None:
    _write_jsonl(path, [row.to_dict() for row in rows])


def audit_post_action_observations(
    path: str | Path,
    required_hours: Iterable[float] = DEFAULT_REQUIRED_HOURS,
) -> dict[str, Any]:
    rows, read_errors = _read_observations_with_errors(path)
    required = [float(value) for value in required_hours]
    errors = list(read_errors)
    warnings: list[str] = []
    outcome_counts: Counter[str] = Counter()
    by_command: dict[str, set[float]] = defaultdict(set)
    seen: set[tuple[str, float]] = set()

    for index, row in enumerate(rows, 1):
        location = f"{path}:{index}"
        if row.outcome not in POST_ACTION_OUTCOMES:
            errors.append(f"{location}: unknown outcome {row.outcome!r}")
        if not row.command_id:
            errors.append(f"{location}: command_id is required")
        if not row.target_id:
            errors.append(f"{location}: target_id is required")
        if not row.observed_at:
            errors.append(f"{location}: observed_at is required")
        key = (row.command_id, row.hours_after_action)
        if key in seen:
            errors.append(f"{location}: duplicate observation for {row.command_id} at {row.hours_after_action:g}h")
        seen.add(key)
        by_command[row.command_id].add(row.hours_after_action)
        outcome_counts[row.outcome] += 1
        if row.outcome in {"uncertain", "not_visible"}:
            warnings.append(f"{location}: outcome requires manual review")
        if row.image_ref is None:
            warnings.append(f"{location}: image_ref is missing")

    missing = _missing_required_observations(by_command, required)
    if missing:
        warnings.append(f"{len(missing)} command/hour follow-up observations are missing")

    summary = summarize_post_action_observations(rows, required_hours=required)
    return {
        "ok": not errors,
        "rows": len(rows),
        "commands": len(by_command),
        "required_hours": required,
        "errors": errors,
        "warnings": warnings,
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "missing_required_observations": missing,
        "summary": summary,
    }


def summarize_post_action_observations(
    observations: Iterable[PostActionObservation],
    required_hours: Iterable[float] = DEFAULT_REQUIRED_HOURS,
) -> dict[str, Any]:
    rows = list(observations)
    required = [float(value) for value in required_hours]
    by_command: dict[str, list[PostActionObservation]] = defaultdict(list)
    outcome_counts: Counter[str] = Counter()
    observer_ids = sorted({row.observer_id for row in rows if row.observer_id})
    for row in rows:
        by_command[row.command_id].append(row)
        outcome_counts[row.outcome] += 1

    latest = [_latest_observation(command_rows) for command_rows in by_command.values()]
    latest_outcomes = Counter(row.outcome for row in latest)
    missing = _missing_required_observations(
        {command_id: {row.hours_after_action for row in command_rows} for command_id, command_rows in by_command.items()},
        required,
    )
    command_count = len(by_command)
    complete = command_count - len({item["command_id"] for item in missing})
    return {
        "ok": True,
        "rows": len(rows),
        "commands": command_count,
        "required_hours": required,
        "complete_commands": complete,
        "incomplete_commands": command_count - complete,
        "coverage_rate": complete / command_count if command_count else 0.0,
        "observer_ids": observer_ids,
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "latest_outcome_counts": dict(sorted(latest_outcomes.items())),
        "delayed_success_rate": latest_outcomes.get("target_removed", 0) / command_count if command_count else 0.0,
        "crop_damage_rate": latest_outcomes.get("crop_damage", 0) / command_count if command_count else 0.0,
        "regrowth_rate": latest_outcomes.get("regrowth", 0) / command_count if command_count else 0.0,
        "uncertain_rate": (
            latest_outcomes.get("uncertain", 0) + latest_outcomes.get("not_visible", 0)
        ) / command_count if command_count else 0.0,
        "missing_required_observations": missing,
    }


def summarize_post_action_file(
    path: str | Path,
    output_path: str | Path | None = None,
    required_hours: Iterable[float] = DEFAULT_REQUIRED_HOURS,
) -> dict[str, Any]:
    audit = audit_post_action_observations(path, required_hours=required_hours)
    payload = {
        "ok": audit["ok"],
        "source": str(path),
        **audit["summary"],
        "errors": audit["errors"],
        "warnings": audit["warnings"],
    }
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def parse_required_hours(value: str | None) -> list[float]:
    if value is None or not value.strip():
        return list(DEFAULT_REQUIRED_HOURS)
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _latest_observation(rows: list[PostActionObservation]) -> PostActionObservation:
    return max(rows, key=lambda row: row.hours_after_action)


def _missing_required_observations(
    by_command: dict[str, set[float]],
    required_hours: list[float],
) -> list[dict[str, Any]]:
    missing = []
    for command_id, observed_hours in sorted(by_command.items()):
        for required in required_hours:
            if not any(abs(observed - required) < 0.001 for observed in observed_hours):
                missing.append({"command_id": command_id, "hours_after_action": required})
    return missing


def _read_observations_with_errors(path: str | Path) -> tuple[list[PostActionObservation], list[str]]:
    rows = []
    errors = []
    for line_number, data in enumerate(_read_jsonl(path), 1):
        try:
            rows.append(PostActionObservation.from_dict(data))
        except (TypeError, ValueError) as exc:
            errors.append(f"{path}:{line_number}: {exc}")
    return rows, errors


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
