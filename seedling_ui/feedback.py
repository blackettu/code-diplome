from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALID_FEEDBACK_PRIORITIES = {"low", "normal", "high", "urgent"}
DEFAULT_PRIORITY_BY_ERROR_TYPE = {
    "crop_damage": "urgent",
    "unsafe_action": "urgent",
    "wrong_target": "high",
    "false_positive_target": "high",
    "missed_target": "high",
    "bad_cell_state": "normal",
    "bad_bbox": "normal",
}


@dataclass(frozen=True)
class AnnotationFeedback:
    image_id: str
    error_type: str
    comment: str
    target_id: str | None = None
    object_id: str | None = None
    cell_id: str | None = None
    operator_id: str | None = None
    priority: str | None = None
    proposed_correction: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.priority is not None and self.priority not in VALID_FEEDBACK_PRIORITIES:
            raise ValueError(f"feedback priority must be one of {sorted(VALID_FEEDBACK_PRIORITIES)}")
        if not isinstance(self.proposed_correction, dict):
            raise ValueError("proposed_correction must be a JSON object")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnnotationFeedback":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FeedbackAnnotationTask:
    task_id: str
    image_id: str
    reason: str
    comment: str = ""
    target_id: str | None = None
    object_id: str | None = None
    cell_id: str | None = None
    operator_id: str | None = None
    source_feedback_created_at: str | None = None
    priority: str = "normal"
    proposed_correction: dict[str, Any] = field(default_factory=dict)
    status: str = "open"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.priority not in VALID_FEEDBACK_PRIORITIES:
            raise ValueError(f"annotation task priority must be one of {sorted(VALID_FEEDBACK_PRIORITIES)}")
        if not isinstance(self.proposed_correction, dict):
            raise ValueError("proposed_correction must be a JSON object")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeedbackAnnotationTask":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def append_feedback(path: str | Path, feedback: AnnotationFeedback) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(feedback.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def read_feedback(path: str | Path) -> list[AnnotationFeedback]:
    feedback_path = Path(path)
    if not feedback_path.exists():
        return []
    rows = []
    for line_number, line in enumerate(feedback_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        data = json.loads(line)
        if not isinstance(data, dict):
            raise ValueError(f"{feedback_path}:{line_number}: feedback row must be a JSON object")
        rows.append(AnnotationFeedback.from_dict(data))
    return rows


def feedback_to_annotation_tasks(feedback_rows: list[AnnotationFeedback]) -> list[FeedbackAnnotationTask]:
    tasks = []
    for index, row in enumerate(feedback_rows, 1):
        task_id = _task_id(row, index)
        tasks.append(
            FeedbackAnnotationTask(
                task_id=task_id,
                image_id=row.image_id,
                reason=row.error_type,
                comment=row.comment,
                target_id=row.target_id,
                object_id=row.object_id,
                cell_id=row.cell_id,
                operator_id=row.operator_id,
                source_feedback_created_at=row.created_at,
                priority=_task_priority(row),
                proposed_correction=dict(row.proposed_correction),
                metadata=dict(row.metadata),
            )
        )
    return tasks


def write_annotation_tasks(path: str | Path, tasks: list[FeedbackAnnotationTask]) -> list[FeedbackAnnotationTask]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return tasks


def write_annotation_tasks_from_feedback(
    feedback_path: str | Path,
    output_path: str | Path,
) -> list[FeedbackAnnotationTask]:
    tasks = feedback_to_annotation_tasks(read_feedback(feedback_path))
    return write_annotation_tasks(output_path, tasks)


def _task_id(row: AnnotationFeedback, index: int) -> str:
    parts = [
        "feedback",
        _slug(row.image_id),
        _slug(row.target_id or row.object_id or row.cell_id or "unlinked"),
        f"{index:06d}",
    ]
    return "_".join(part for part in parts if part)


def _task_priority(row: AnnotationFeedback) -> str:
    return row.priority or DEFAULT_PRIORITY_BY_ERROR_TYPE.get(row.error_type, "normal")


def _slug(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "_" for char in str(value)]
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "item"
