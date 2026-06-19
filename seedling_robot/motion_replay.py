from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from seedling_robot.adapters.simulator import SimulatorRobotAdapter
from seedling_robot.adapters.base import RobotAdapter
from seedling_robot.protocol import MotionCommand, MotionResult


def replay_motion_commands(
    adapter: RobotAdapter,
    commands: list[MotionCommand],
    *,
    stop_on_failure: bool = True,
) -> list[MotionResult]:
    results: list[MotionResult] = []
    adapter.connect()
    adapter.home()
    for command in commands:
        result = adapter.move_to(command.point_mm, command.speed)
        results.append(result)
        if stop_on_failure and not result.ok:
            break
    return results


def load_motion_commands(path: str | Path) -> list[MotionCommand]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    else:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
        rows = _command_rows(payload)
    commands = []
    for index, row in enumerate(rows, 1):
        payload = _command_payload(row, index)
        if payload is not None:
            commands.append(MotionCommand.from_dict(payload))
    return commands


def run_motion_replay(
    commands_path: str | Path,
    output_path: str | Path,
    *,
    stop_on_failure: bool = True,
) -> dict[str, Any]:
    commands = load_motion_commands(commands_path)
    adapter = SimulatorRobotAdapter()
    results = replay_motion_commands(adapter, commands, stop_on_failure=stop_on_failure)
    payload = {
        "ok": all(result.ok for result in results) and len(results) == len(commands),
        "mode": "simulation",
        "commands": len(commands),
        "executed": len(results),
        "stop_on_failure": stop_on_failure,
        "stopped_on_failure": len(results) < len(commands),
        "results": [
            {
                "command_id": command.command_id,
                "command": command.to_dict(),
                "result": result.to_dict(),
            }
            for command, result in zip(commands, results)
        ],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _command_rows(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("motion_commands", "commands"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return rows
    raise ValueError("motion replay input must be a JSON list or contain a commands/motion_commands list")


def _command_payload(row: Any, index: int) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        raise ValueError("motion command row must be a JSON object")
    if "command_id" in row and "point_mm" in row:
        return row
    command = row.get("command")
    if isinstance(command, dict) and "command_id" in command and "point_mm" in command:
        return command
    if row.get("kind") == "move_to" and isinstance(row.get("point_mm"), list):
        return {
            "command_id": str(row.get("command_id") or f"move_to_{index:06d}"),
            "point_mm": row["point_mm"],
            "speed": row.get("speed") if isinstance(row.get("speed"), dict) else {},
            "mode": str(row.get("mode", "dry_run_pointer")),
            "metadata": {"source_log_kind": "move_to"},
        }
    if row.get("kind") in {"home", "mark_or_act", "emergency_stop"}:
        return None
    raise ValueError("motion command row must contain command_id/point_mm, a nested command object or move_to log fields")
