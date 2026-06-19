from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from seedling_core.schemas import ActionPlan, SceneState
from seedling_decision.policies import (
    HumanReviewPolicy,
    NearestNeighborPolicy,
    NoOpPolicy,
    RasterScanPolicy,
    RiskAwareRulePolicy,
    RoutePlanningPolicy,
)

from .html import write_experiment_html_report


POLICY_FACTORIES = {
    "noop": NoOpPolicy,
    "human_review": HumanReviewPolicy,
    "raster_scan": RasterScanPolicy,
    "nearest_neighbor": NearestNeighborPolicy,
    "route_planning": RoutePlanningPolicy,
    "risk_aware_rule": RiskAwareRulePolicy,
}


def compare_policies_on_scene(
    scene: SceneState,
    policy_names: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for policy_name in policy_names:
        plan = policy_plan_for_scene(scene, policy_name)
        rows.append(
            {
                "scene_id": scene.scene_id,
                "policy_selector": policy_name,
                "policy": plan.policy_id,
                "commands": len(plan.commands),
                "route_distance_mm": _route_distance_mm(plan.commands),
                "review_targets": len(plan.review_target_ids),
                "blocked_targets": len(plan.blocked_target_ids),
                "rl_action_kind": plan.metadata.get("rl_action_kind"),
                "command_target_order": ",".join(command.target_id for command in plan.commands),
                "review_target_ids": ",".join(plan.review_target_ids),
                "blocked_target_ids": ",".join(plan.blocked_target_ids),
                "review_reasons": json.dumps(
                    plan.metadata.get("review_reasons_by_target", {}),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "blocked_reasons": json.dumps(
                    plan.metadata.get("blocked_reasons_by_target", {}),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
    return rows


def policy_plan_for_scene(scene: SceneState, policy_name: str) -> ActionPlan:
    if policy_name in {"rl_no_model", "rl_policy_adapter"}:
        from seedling_rl import RLPolicyAdapter

        return RLPolicyAdapter(policy_id=policy_name).propose_plan(scene)
    if policy_name.startswith("rl_checkpoint:"):
        from seedling_rl import RLPolicyAdapter

        checkpoint = policy_name.split(":", 1)[1]
        if not checkpoint:
            raise ValueError("rl_checkpoint policy selector requires a checkpoint path")
        return RLPolicyAdapter.from_sb3(checkpoint, policy_id="rl_policy_adapter").propose_plan(scene)
    policy_cls = POLICY_FACTORIES.get(policy_name)
    if policy_cls is None:
        available = sorted([*POLICY_FACTORIES, "rl_no_model", "rl_policy_adapter", "rl_checkpoint:<path>"])
        raise KeyError(f"Unknown policy {policy_name!r}; available: {available}")
    policy = policy_cls()
    return policy.propose_plan(scene)


def _route_distance_mm(commands: list[Any]) -> float:
    if not commands:
        return 0.0
    cumulative = commands[-1].metadata.get("cumulative_route_distance_mm")
    if cumulative is not None:
        return float(cumulative)
    return sum(float(command.metadata.get("leg_distance_mm", 0.0)) for command in commands)


def compare_policy_scenario_file(
    scene_path: str | Path,
    policy_names: list[str],
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    scene = load_scene_state(scene_path)
    rows = compare_policies_on_scene(scene, policy_names)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "scenario_compare.json").write_text(
        json.dumps({"scene_id": scene.scene_id, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_experiment_html_report("Scenario Policy Comparison", output / "scenario_compare.html", [("Policies", rows)])
    return rows


def write_policy_plan_file(
    scene_path: str | Path,
    policy_name: str,
    output_path: str | Path,
) -> ActionPlan:
    scene = load_scene_state(scene_path)
    plan = policy_plan_for_scene(scene, policy_name)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "scene_id": scene.scene_id,
                "policy": policy_name,
                "plan": plan.to_dict(),
                "commands": len(plan.commands),
                "review_targets": len(plan.review_target_ids),
                "blocked_targets": len(plan.blocked_target_ids),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return plan


def load_scene_state(path: str | Path) -> SceneState:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(data, dict) and "scenes" in data:
        scenes = data.get("scenes")
        if not isinstance(scenes, list) or not scenes:
            raise ValueError(f"Scene file contains no scenes: {path}")
        data = scenes[0]
    if not isinstance(data, dict):
        raise ValueError(f"SceneState file must contain a JSON object: {path}")
    return SceneState.from_dict(data)
