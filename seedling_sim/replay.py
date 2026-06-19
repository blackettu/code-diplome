from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ReplayStep:
    step_index: int
    event_type: str
    payload: dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReplayLog:
    replay_id: str
    scene_id: str
    steps: list[ReplayStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReplayLog":
        steps = [step if isinstance(step, ReplayStep) else ReplayStep(**step) for step in data.get("steps", [])]
        return cls(
            replay_id=str(data["replay_id"]),
            scene_id=str(data["scene_id"]),
            steps=steps,
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ReplayLog":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"ReplayLog must be a JSON object: {path}")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "replay_id": self.replay_id,
            "scene_id": self.scene_id,
            "steps": [step.to_dict() for step in self.steps],
            "metadata": self.metadata,
        }

    def to_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass
class ReplayLogger:
    replay_id: str
    scene_id: str
    log: ReplayLog = field(init=False)

    def __post_init__(self) -> None:
        self.log = ReplayLog(replay_id=self.replay_id, scene_id=self.scene_id)

    def append(self, event_type: str, payload: dict[str, Any]) -> ReplayStep:
        step = ReplayStep(step_index=len(self.log.steps) + 1, event_type=event_type, payload=payload)
        self.log.steps.append(step)
        return step

    def append_execution(self, execution_result: Any) -> ReplayStep:
        payload = execution_result.to_dict() if hasattr(execution_result, "to_dict") else dict(execution_result)
        return self.append("robot_execution", payload)

    def to_json(self, path: str | Path) -> None:
        self.log.to_json(path)
