from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from seedling_core.schemas import ActionPlan, SceneState
from seedling_decision.safety_gate import SafetyGate, SafetyLimits
from seedling_robot.adapters.real_gantry_serial import RealGantrySerialAdapter
from seedling_robot.hil_safety import HardwareInLoopReview, validate_hil_review
from seedling_robot.protocol import RobotExecutionResult
from seedling_sim.replay import ReplayLogger


AdapterFactory = Callable[..., RealGantrySerialAdapter]


def run_hil_pointer_plan(
    scene_path: str | Path,
    plan_path: str | Path,
    review_path: str | Path,
    port: str,
    out_path: str | Path,
    *,
    command_log_path: str | Path | None = None,
    replay_path: str | Path | None = None,
    allow_hardware: bool = False,
    operator_confirmed: bool = False,
    baudrate: int = 115200,
    timeout_s: float = 1.0,
    interlock_ok: bool = True,
    limit_switch_ok: bool = True,
    adapter_factory: AdapterFactory = RealGantrySerialAdapter,
) -> dict[str, Any]:
    scene = load_scene_state(scene_path)
    plan = load_action_plan(plan_path)
    review = HardwareInLoopReview.from_json(review_path)
    review_validation = validate_hil_review(
        review,
        required_mode="hardware_in_loop_pointer",
        required_tool_profile="pointer_only",
        require_positioning_error=True,
    )
    if not review_validation["ok"]:
        payload = {
            "ok": False,
            "mode": "hardware_in_loop_pointer",
            "scene_id": scene.scene_id,
            "plan_id": plan.plan_id,
            "policy_id": plan.policy_id,
            "review": review_validation,
            "aborted": True,
            "abort_reason": "invalid_hil_review",
            "commands": len(plan.commands),
            "executed": 0,
            "blocked": len(plan.commands),
            "command_log": str(command_log_path) if command_log_path else None,
            "replay": str(replay_path) if replay_path else None,
            "results": [],
        }
        _write_report(out_path, payload)
        return payload
    replay_logger = ReplayLogger(replay_id=f"hil_{plan.plan_id}", scene_id=scene.scene_id) if replay_path else None
    adapter = adapter_factory(
        port=port,
        review=review,
        baudrate=baudrate,
        timeout_s=timeout_s,
        allow_hardware=allow_hardware,
        command_log_path=command_log_path,
        replay_logger=replay_logger,
        interlock_ok=interlock_ok,
        limit_switch_ok=limit_switch_ok,
    )
    gate = SafetyGate(
        SafetyLimits(
            mode="hardware_in_loop_pointer",
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
        "ok": review_validation["ok"] and not aborted and skipped == 0 and all(result.ok for result in results),
        "mode": "hardware_in_loop_pointer",
        "scene_id": scene.scene_id,
        "plan_id": plan.plan_id,
        "policy_id": plan.policy_id,
        "review": review_validation,
        "aborted": aborted,
        "abort_reason": abort_reason,
        "commands": len(plan.commands),
        "executed": sum(1 for result in results if result.ok),
        "blocked": sum(1 for result in results if not result.ok) + skipped,
        "skipped": skipped,
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


def load_scene_state(path: str | Path) -> SceneState:
    data = _load_json(path)
    if isinstance(data.get("scenes"), list):
        scenes = [item for item in data["scenes"] if isinstance(item, dict)]
        if not scenes:
            raise ValueError(f"Scene bundle contains no scenes: {path}")
        data = scenes[0]
    return SceneState.from_dict(data)


def load_action_plan(path: str | Path) -> ActionPlan:
    data = _load_json(path)
    if isinstance(data.get("plan"), dict):
        data = data["plan"]
    if not isinstance(data, dict):
        raise ValueError(f"ActionPlan file must contain a JSON object: {path}")
    return ActionPlan.from_dict(data)


def _load_json(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data
