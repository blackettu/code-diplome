from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from seedling_sim.replay import ReplayLog


CRITICAL_EVENTS = {
    "unsafe_target",
    "invalid_target_index",
    "invalid_action",
    "evaluator_max_steps_reached",
}


def evaluate_replay_logs(replay_paths: list[str | Path], output_path: str | Path | None = None) -> dict[str, Any]:
    paths = [Path(path) for path in replay_paths]
    logs = [ReplayLog.from_json(path) for path in paths]
    robot_events = 0
    rl_step_events = 0
    blocked = 0
    allowed = 0
    actions = 0
    target_steps = 0
    successes = 0
    crop_damage = 0
    reviews = 0
    rewards: list[float] = []
    distance_errors: list[float] = []
    movement_values: list[float] = []
    critical_events: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    replay_summaries: list[dict[str, Any]] = []

    for path, log in zip(paths, logs):
        replay_summary = {
            "path": str(path),
            "replay_id": log.replay_id,
            "scene_id": log.scene_id,
            "steps": len(log.steps),
            "robot_execution_events": 0,
            "rl_step_events": 0,
            "reward_total": 0.0,
            "critical_events": 0,
            "target_steps": 0,
            "successes": 0,
            "crop_damage": 0,
            "reviews": 0,
            "blocked": 0,
            "allowed": 0,
            "distance_mm": 0.0,
        }
        previous_point: list[float] | None = None
        for step in log.steps:
            if step.event_type not in {"robot_execution", "rl_step"}:
                continue
            payload = step.payload if isinstance(step.payload, dict) else {}
            info = payload.get("info") if isinstance(payload.get("info"), dict) else payload
            outcome = _outcome_from_payload(payload)
            event = str(payload.get("event") or info.get("event") or step.event_type)
            reward = _first_float(payload.get("reward"), outcome.get("reward"))
            if reward is not None:
                rewards.append(reward)
                replay_summary["reward_total"] += reward
            if step.event_type == "robot_execution":
                robot_events += 1
                replay_summary["robot_execution_events"] += 1
                decision = payload.get("safety_decision", {}) if isinstance(payload.get("safety_decision"), dict) else {}
                allowed_value = _allowed_from_payload(payload, decision)
                if allowed_value is True:
                    allowed += 1
                    replay_summary["allowed"] += 1
                elif allowed_value is False:
                    blocked += 1
                    replay_summary["blocked"] += 1
                    _add_reasons(reason_counts, decision, payload)
            else:
                rl_step_events += 1
                replay_summary["rl_step_events"] += 1

            actions += 1
            if event == "review":
                reviewed = int(info.get("reviewed_targets", payload.get("reviewed_targets", 0)) or 0)
                reviews += reviewed
                replay_summary["reviews"] += reviewed
            if event == "target_step" or outcome:
                target_steps += 1
                replay_summary["target_steps"] += 1
                if outcome.get("success"):
                    successes += 1
                    replay_summary["successes"] += 1
                if outcome.get("crop_damage"):
                    crop_damage += 1
                    replay_summary["crop_damage"] += 1

            distance_error = _first_float(outcome.get("distance_error_mm"))
            if distance_error is not None:
                distance_errors.append(distance_error)
            movement = _movement_from_outcome(outcome, previous_point)
            if movement is not None:
                movement_values.append(movement)
                replay_summary["distance_mm"] += movement
            previous_point = _point_from_outcome(outcome) or previous_point

            is_critical = event in CRITICAL_EVENTS or bool(outcome.get("crop_damage"))
            if step.event_type == "robot_execution" and _allowed_from_payload(payload, payload.get("safety_decision", {})) is False:
                is_critical = True
            if is_critical:
                critical = {
                    "replay_id": log.replay_id,
                    "scene_id": log.scene_id,
                    "step_index": step.step_index,
                    "event": event,
                    "reasons": _reasons_from_payload(payload),
                    "target_id": payload.get("target_id") or outcome.get("target_id"),
                }
                critical_events.append(critical)
                replay_summary["critical_events"] += 1
        replay_summaries.append(replay_summary)

    reward_total = sum(rewards)
    payload = {
        "ok": True,
        "replays": len(logs),
        "robot_execution_events": robot_events,
        "rl_step_events": rl_step_events,
        "action_events": actions,
        "allowed": allowed,
        "blocked": blocked,
        "block_rate": blocked / robot_events if robot_events else 0.0,
        "block_reason_counts": dict(sorted(reason_counts.items())),
        "target_steps": target_steps,
        "successful_targets": successes,
        "successful_target_rate": successes / target_steps if target_steps else 0.0,
        "crop_damage": crop_damage,
        "reviewed_targets": reviews,
        "review_rate": reviews / actions if actions else 0.0,
        "critical_events": critical_events,
        "critical_events_count": len(critical_events),
        "critical_error_rate": len(critical_events) / actions if actions else 0.0,
        "reward_total": reward_total if rewards else None,
        "reward_mean": sum(rewards) / len(rewards) if rewards else None,
        "total_distance_mm": sum(movement_values) if movement_values else None,
        "mean_distance_error_mm": sum(distance_errors) / len(distance_errors) if distance_errors else None,
        "max_distance_error_mm": max(distance_errors) if distance_errors else None,
        "replay_summaries": replay_summaries,
    }
    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _outcome_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
    outcome = payload.get("outcome")
    if isinstance(outcome, dict):
        return outcome
    outcome = info.get("outcome")
    if isinstance(outcome, dict):
        return outcome
    return {}


def _allowed_from_payload(payload: dict[str, Any], decision: Any) -> bool | None:
    if payload.get("ok") is False:
        return False
    if isinstance(decision, dict) and decision.get("allowed") is False:
        return False
    if isinstance(decision, dict) and "allowed" in decision:
        return bool(decision.get("allowed"))
    if "ok" in payload:
        return bool(payload.get("ok"))
    return None


def _movement_from_outcome(outcome: dict[str, Any], previous_point: list[float] | None) -> float | None:
    info = outcome.get("info") if isinstance(outcome.get("info"), dict) else {}
    movement = _first_float(outcome.get("movement_mm"), info.get("movement_mm"))
    if movement is not None:
        return movement
    point = _point_from_outcome(outcome)
    if point is not None and previous_point is not None:
        return _distance(previous_point, point)
    return None


def _point_from_outcome(outcome: dict[str, Any]) -> list[float] | None:
    value = outcome.get("actual_point_mm") or outcome.get("commanded_point_mm")
    if not isinstance(value, list | tuple) or len(value) < 2:
        return None
    return [float(value[0]), float(value[1])]


def _add_reasons(counts: Counter[str], decision: dict[str, Any], payload: dict[str, Any]) -> None:
    for reason in _reasons_from_payload(payload):
        counts[reason] += 1


def _reasons_from_payload(payload: dict[str, Any]) -> list[str]:
    collected: list[str] = []
    decision = payload.get("safety_decision")
    if isinstance(decision, dict):
        collected.extend(_reasons_from_decision(decision))
    reasons = payload.get("reasons")
    if isinstance(reasons, list):
        collected.extend(str(reason) for reason in reasons)
    elif reasons:
        collected.append(str(reasons))
    collected.extend(_adapter_block_reasons(payload))
    return sorted(dict.fromkeys(collected))


def _adapter_block_reasons(payload: dict[str, Any]) -> list[str]:
    outcome = _outcome_from_payload(payload)
    tool_result = outcome.get("tool_result") if isinstance(outcome.get("tool_result"), dict) else {}
    profile = tool_result.get("profile") if isinstance(tool_result.get("profile"), dict) else {}
    message = str(payload.get("message") or tool_result.get("message") or "")
    if profile.get("profile_id") and profile.get("profile_id") != "pointer_only":
        return ["unsupported_tool_profile"]
    if "only allows pointer_only" in message:
        return ["unsupported_tool_profile"]
    return []


def _reasons_from_decision(decision: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    value = decision.get("reasons")
    if isinstance(value, list):
        reasons.extend(str(reason) for reason in value)
    elif value:
        reasons.append(str(value))
    result = decision.get("result")
    if result:
        reasons.append(str(result))
    return sorted(dict.fromkeys(reasons))


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value in {None, ""}:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _distance(left: list[float], right: list[float]) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))
