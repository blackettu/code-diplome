from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


RL_METRIC_COLUMNS = [
    "phase",
    "episode",
    "step",
    "policy",
    "event",
    "reward",
    "critical_error",
    "review_targets",
    "review_rate",
    "movement_mm",
    "distance_error_mm",
    "total_distance_mm",
    "mean_distance_error_mm",
]


@dataclass
class RLMetricsLogger:
    jsonl_path: str | Path
    csv_path: str | Path
    rows: list[dict[str, Any]] = field(default_factory=list)
    total_steps: int = 0
    critical_errors: int = 0
    review_targets: int = 0
    total_distance_mm: float = 0.0
    distance_error_sum_mm: float = 0.0
    distance_error_count: int = 0

    def __post_init__(self) -> None:
        self.jsonl_path = Path(self.jsonl_path)
        self.csv_path = Path(self.csv_path)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.write_text("", encoding="utf-8")

    def log_step(
        self,
        *,
        phase: str,
        episode: int | None,
        step: int | None,
        reward: float,
        info: dict[str, Any] | None = None,
        policy: str | None = None,
    ) -> dict[str, Any]:
        info = info or {}
        event = str(info.get("event") or "")
        outcome = info.get("outcome") if isinstance(info.get("outcome"), dict) else {}
        critical = event in {"unsafe_target", "invalid_target_index", "invalid_action", "evaluator_max_steps_reached"}
        if bool(outcome.get("crop_damage")):
            critical = True
        review_count = int(info.get("reviewed_targets", 0) or 0)
        outcome_info = outcome.get("info") if isinstance(outcome.get("info"), dict) else {}
        movement_mm = _maybe_float(outcome.get("movement_mm"))
        if movement_mm is None:
            movement_mm = _maybe_float(outcome_info.get("movement_mm"))
        distance_error_mm = _maybe_float(outcome.get("distance_error_mm"))

        self.total_steps += 1
        if critical:
            self.critical_errors += 1
        self.review_targets += review_count
        if movement_mm is not None:
            self.total_distance_mm += movement_mm
        if distance_error_mm is not None:
            self.distance_error_sum_mm += distance_error_mm
            self.distance_error_count += 1

        row = {
            "phase": phase,
            "episode": episode,
            "step": step,
            "policy": policy,
            "event": event,
            "reward": float(reward),
            "critical_error": int(critical),
            "review_targets": review_count,
            "review_rate": self.review_targets / max(self.total_steps, 1),
            "movement_mm": movement_mm,
            "distance_error_mm": distance_error_mm,
            "total_distance_mm": self.total_distance_mm,
            "mean_distance_error_mm": (
                self.distance_error_sum_mm / self.distance_error_count if self.distance_error_count else None
            ),
        }
        self.rows.append(row)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")
        return row

    def log_event(self, *, phase: str, event: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        row = {
            "phase": phase,
            "episode": None,
            "step": None,
            "policy": None,
            "event": event,
            "reward": 0.0,
            "critical_error": 0,
            "review_targets": 0,
            "review_rate": self.review_targets / max(self.total_steps, 1),
            "movement_mm": None,
            "distance_error_mm": None,
            "total_distance_mm": self.total_distance_mm,
            "mean_distance_error_mm": (
                self.distance_error_sum_mm / self.distance_error_count if self.distance_error_count else None
            ),
            "metadata": metadata or {},
        }
        self.rows.append(row)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")
        return row

    def close(self) -> None:
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RL_METRIC_COLUMNS)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({column: row.get(column) for column in RL_METRIC_COLUMNS})

    def summary(self) -> dict[str, Any]:
        return {
            "steps": self.total_steps,
            "critical_errors": self.critical_errors,
            "review_targets": self.review_targets,
            "review_rate": self.review_targets / max(self.total_steps, 1),
            "total_distance_mm": self.total_distance_mm,
            "mean_distance_error_mm": self.distance_error_sum_mm / self.distance_error_count
            if self.distance_error_count
            else None,
            "jsonl_path": str(self.jsonl_path),
            "csv_path": str(self.csv_path),
        }


def make_sb3_metric_callback(logger: RLMetricsLogger) -> Any:
    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError:
        return None

    class _SB3MetricCallback(BaseCallback):  # pragma: no cover - depends on optional SB3 runtime
        def __init__(self) -> None:
            super().__init__()
            self._step = 0

        def _on_step(self) -> bool:
            rewards = self.locals.get("rewards", [])
            infos = self.locals.get("infos", [])
            for index, info in enumerate(infos):
                reward = rewards[index] if index < len(rewards) else 0.0
                logger.log_step(phase="train", episode=None, step=self._step, reward=float(reward), info=info)
                self._step += 1
            return True

        def _on_training_end(self) -> None:
            logger.close()

    return _SB3MetricCallback()


def _maybe_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
