from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from seedling_decision.safety_gate import SafetyGate, SafetyLimits
from seedling_robot.adapters.dry_run_serial import DryRunSerialAdapter
from seedling_robot.hil_runner import load_action_plan, load_scene_state
from seedling_robot.protocol import RobotExecutionResult
from seedling_sim.replay import ReplayLogger


AdapterFactory = Callable[..., DryRunSerialAdapter]


def run_dry_run_plan(
    scene_path: str | Path,
    plan_path: str | Path,
    out_path: str | Path,
    *,
    command_log_path: str | Path | None = None,
    replay_path: str | Path | None = None,
    operator_confirmed: bool = False,
    interlock_ok: bool = True,
    limit_switch_ok: bool = True,
    adapter_factory: AdapterFactory = DryRunSerialAdapter,
) -> dict[str, Any]:
    scene = load_scene_state(scene_path)
    plan = load_action_plan(plan_path)
    replay_logger = ReplayLogger(replay_id=f"dry_run_{plan.plan_id}", scene_id=scene.scene_id) if replay_path else None
    adapter = adapter_factory(
        command_log_path=command_log_path,
        replay_logger=replay_logger,
        interlock_ok=interlock_ok,
        limit_switch_ok=limit_switch_ok,
    )
    gate = SafetyGate(
        SafetyLimits(
            mode="dry_run_pointer",
            allow_real_action=False,
            require_operator_confirmation=not operator_confirmed,
        )
    )
    adapter.connect()
    adapter.home()

    results: list[RobotExecutionResult] = []
    aborted = False
    abort_reason: str | None = None
    for command in plan.commands:
        result = adapter.execute_command(command, scene, safety_gate=gate)
        results.append(result)
        if not result.ok:
            aborted = True
            abort_reason = "command_failed"
            adapter.emergency_stop()
            break

    if replay_logger is not None and replay_path is not None:
        replay_logger.to_json(replay_path)

    skipped = len(plan.commands) - len(results)
    payload = {
        "ok": not aborted and skipped == 0 and all(result.ok for result in results),
        "mode": "dry_run_pointer",
        "scene_id": scene.scene_id,
        "plan_id": plan.plan_id,
        "policy_id": plan.policy_id,
        "aborted": aborted,
        "abort_reason": abort_reason,
        "commands": len(plan.commands),
        "executed": sum(1 for result in results if result.ok),
        "blocked": sum(1 for result in results if not result.ok) + skipped,
        "skipped": skipped,
        "operator_confirmed": operator_confirmed,
        "interlock_ok": interlock_ok,
        "limit_switch_ok": limit_switch_ok,
        "command_log": str(command_log_path) if command_log_path else None,
        "replay": str(replay_path) if replay_path else None,
        "results": [result.to_dict() for result in results],
    }
    _write_report(out_path, payload)
    return payload


def _write_report(out_path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
