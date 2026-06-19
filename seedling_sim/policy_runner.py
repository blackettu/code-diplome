from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from seedling_sim.logical_tray import LogicalTraySimulator
from seedling_sim.replay import ReplayLogger, ReplayLog
from seedling_sim.schemas import SimScene, SimTarget


POLICY_ALIASES = {
    "rule_based_v0": "risk_aware_rule",
    "nearest_neighbor_v0": "nearest_neighbor",
    "route_planning_v0": "route_planning",
}

SUPPORTED_POLICIES = {
    "noop",
    "raster_scan",
    "nearest_neighbor",
    "route_planning",
    "risk_aware_rule",
    *POLICY_ALIASES,
}


def run_policy_on_scene(
    scene: SimScene,
    policy: str,
    max_steps: int | None = None,
    seed: int = 42,
    replay_path: str | Path | None = None,
) -> tuple[dict[str, Any], ReplayLog, SimScene]:
    canonical_policy = _canonical_policy(policy)
    simulator = LogicalTraySimulator(seed=seed)
    state = simulator.reset(scene)
    logger = ReplayLogger(replay_id=f"{state.scene_id}_{canonical_policy}", scene_id=state.scene_id)
    logger.log.metadata.update({"policy": canonical_policy, "requested_policy": policy, "seed": seed})
    ordered_targets = _ordered_targets(state, canonical_policy)
    step_limit = max_steps if max_steps is not None else len(ordered_targets)
    total_reward = 0.0
    successes = 0
    crop_damage = 0
    command_order: list[str] = []
    logger.append(
        "policy_start",
        {
            "policy": canonical_policy,
            "requested_policy": policy,
            "candidate_targets": len(ordered_targets),
            "max_steps": step_limit,
        },
    )
    for target in ordered_targets[:step_limit]:
        outcome = simulator.step_target(target)
        outcome_dict = outcome.to_dict()
        total_reward += outcome.reward
        successes += int(outcome.success)
        crop_damage += int(outcome.crop_damage)
        command_order.append(outcome.target_id)
        logger.append(
            "robot_execution",
            {
                "policy": canonical_policy,
                "target_id": outcome.target_id,
                "ok": not outcome.crop_damage,
                "outcome": outcome_dict,
            },
        )
        if outcome.crop_damage:
            break
    remaining_targets = [target.target_id for target in state.targets if not target.processed]
    summary = {
        "ok": True,
        "scene_id": state.scene_id,
        "policy": canonical_policy,
        "requested_policy": policy,
        "steps": len(command_order),
        "reward_total": total_reward,
        "successes": successes,
        "crop_damage": crop_damage,
        "remaining_targets": len(remaining_targets),
        "command_target_order": command_order,
    }
    logger.append("policy_stop", summary)
    if replay_path:
        logger.to_json(replay_path)
    return summary, logger.log, state


def compare_policies_on_scenes(
    scenes: list[SimScene],
    policies: list[str],
    max_steps: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        for policy in policies:
            summary, _, _ = run_policy_on_scene(scene, policy, max_steps=max_steps, seed=seed)
            rows.append(
                {
                    "scene_id": summary["scene_id"],
                    "policy": summary["policy"],
                    "requested_policy": summary["requested_policy"],
                    "steps": summary["steps"],
                    "reward_total": summary["reward_total"],
                    "successes": summary["successes"],
                    "crop_damage": summary["crop_damage"],
                    "remaining_targets": summary["remaining_targets"],
                    "command_target_order": ",".join(summary["command_target_order"]),
                }
            )
    return rows


def load_sim_scenes(path: str | Path) -> list[SimScene]:
    source = Path(path)
    if source.is_file():
        return [SimScene.from_json(source)]
    if not source.exists():
        raise FileNotFoundError(f"Scene path not found: {source}")
    scenes: list[SimScene] = []
    for candidate in sorted(source.glob("*.json")):
        try:
            scenes.append(SimScene.from_json(candidate))
        except Exception:
            continue
    if not scenes:
        raise ValueError(f"No SimScene JSON files found in {source}")
    return scenes


def write_policy_comparison(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".html":
        from seedling_reports.html import write_experiment_html_report

        write_experiment_html_report("Simulation Policy Comparison", output, [("Policies", rows)])
        json_path = output.with_suffix(".json")
        json_path.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    output.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


def _canonical_policy(policy: str) -> str:
    normalized = str(policy)
    if normalized not in SUPPORTED_POLICIES:
        raise ValueError(f"Unsupported simulation policy {policy!r}; supported: {sorted(SUPPORTED_POLICIES)}")
    return POLICY_ALIASES.get(normalized, normalized)


def _ordered_targets(scene: SimScene, policy: str) -> list[SimTarget]:
    safe_targets = [target for target in scene.targets if _safe_to_act(target)]
    if policy == "noop":
        return []
    if policy == "raster_scan":
        plants = {plant.object_id: plant for plant in scene.plants}
        return sorted(
            safe_targets,
            key=lambda target: (
                plants[target.object_id].row if target.object_id in plants else 9999,
                plants[target.object_id].col if target.object_id in plants else 9999,
                target.target_id,
            ),
        )
    if policy == "nearest_neighbor":
        start = scene.robot_position_mm
        return sorted(safe_targets, key=lambda target: (_distance(start[:2], target.point_mm), target.target_id))
    if policy in {"route_planning", "risk_aware_rule"}:
        if policy == "risk_aware_rule":
            safe_targets = sorted(
                safe_targets,
                key=lambda target: (target.uncertainty_radius_mm, target.target_id),
            )
        return _greedy_route(safe_targets, scene.robot_position_mm)
    raise ValueError(f"Unsupported policy: {policy}")


def _safe_to_act(target: SimTarget) -> bool:
    return (
        not target.processed
        and target.target_type in {"remove_extra_crop", "remove_weed"}
        and target.uncertainty_radius_mm <= 2.0
        and (target.min_distance_to_keep_mm is None or target.min_distance_to_keep_mm >= 5.0)
    )


def _greedy_route(targets: list[SimTarget], start_position_mm: list[float]) -> list[SimTarget]:
    remaining = list(targets)
    current = start_position_mm[:2]
    route: list[SimTarget] = []
    while remaining:
        target = min(remaining, key=lambda item: (_distance(current, item.point_mm), item.target_id))
        route.append(target)
        current = list(target.point_mm)
        remaining.remove(target)
    return route


def _distance(a: list[float], b: list[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
