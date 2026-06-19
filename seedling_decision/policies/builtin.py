from __future__ import annotations

import math
from dataclasses import dataclass, field

from seedling_core.schemas import ActionPlan, ActionTarget, SceneState
from seedling_decision.action_masks import ActionMaskBuilder

from .base import DecisionPolicy


@dataclass
class NoOpPolicy(DecisionPolicy):
    policy_id: str = "noop"

    def propose_plan(self, scene: SceneState) -> ActionPlan:
        review_ids = [target.target_id for target in scene.targets]
        return ActionPlan(
            plan_id=f"{self.policy_id}_{scene.scene_id}",
            policy_id=self.policy_id,
            commands=[],
            review_target_ids=review_ids,
            blocked_target_ids=[],
            metadata={
                "review_reasons_by_target": {
                    target_id: ["noop_policy_requires_review"]
                    for target_id in review_ids
                }
            },
        )


@dataclass
class HumanReviewPolicy(DecisionPolicy):
    policy_id: str = "human_review"
    action_mask_builder: ActionMaskBuilder = field(default_factory=ActionMaskBuilder)

    def propose_plan(self, scene: SceneState) -> ActionPlan:
        mask = self.action_mask_builder.build(scene, self.processed_target_ids)
        reasons = mask.reasons_by_target()
        review_ids = [
            target.target_id
            for target in scene.targets
            if target.human_review_required or reasons.get(target.target_id)
        ]
        return ActionPlan(
            plan_id=f"{self.policy_id}_{scene.scene_id}",
            policy_id=self.policy_id,
            commands=[],
            review_target_ids=review_ids,
            blocked_target_ids=[],
            metadata={
                "review_reasons_by_target": {
                    target_id: reasons.get(target_id) or ["human_review_required"]
                    for target_id in review_ids
                }
            },
        )


@dataclass
class RasterScanPolicy(DecisionPolicy):
    policy_id: str = "raster_scan"
    action_mask_builder: ActionMaskBuilder = field(default_factory=ActionMaskBuilder)

    def propose_plan(self, scene: SceneState) -> ActionPlan:
        return _build_plan(self, scene, self._ordered_targets(scene), self.action_mask_builder)

    def _ordered_targets(self, scene: SceneState) -> list[ActionTarget]:
        cell_order = {cell.cell_id: (cell.row, cell.col) for cell in scene.cells}
        return sorted(
            scene.targets,
            key=lambda target: (*cell_order.get(target.cell_id, (9999, 9999)), target.target_id),
        )


@dataclass
class NearestNeighborPolicy(DecisionPolicy):
    policy_id: str = "nearest_neighbor"
    action_mask_builder: ActionMaskBuilder = field(default_factory=ActionMaskBuilder)

    def propose_plan(self, scene: SceneState) -> ActionPlan:
        ordered = sorted(scene.targets, key=lambda target: _target_distance_from_robot(target, scene.robot.position_mm))
        return _build_plan(self, scene, ordered, self.action_mask_builder)


@dataclass
class RoutePlanningPolicy(DecisionPolicy):
    policy_id: str = "route_planning"
    action_mask_builder: ActionMaskBuilder = field(default_factory=ActionMaskBuilder)

    def propose_plan(self, scene: SceneState) -> ActionPlan:
        mask = self.action_mask_builder.build(scene, self.processed_target_ids)
        valid_ids = set(mask.valid_target_ids())
        reasons = mask.reasons_by_target()
        valid_targets = [target for target in scene.targets if target.target_id in valid_ids]
        ordered, leg_distances = _greedy_route(valid_targets, scene.robot.position_mm)
        commands = []
        cumulative_distance = 0.0
        for index, target in enumerate(ordered, 1):
            cumulative_distance += leg_distances[index - 1]
            command = self._command_for_target(target, index)
            command.metadata.update(
                {
                    "route_planning_method": "greedy_nearest_neighbor",
                    "route_index": index,
                    "leg_distance_mm": leg_distances[index - 1],
                    "cumulative_route_distance_mm": cumulative_distance,
                }
            )
            commands.append(command)
        review_ids, blocked_ids = _review_and_blocked_ids(mask)
        return ActionPlan(
            plan_id=f"{self.policy_id}_{scene.scene_id}",
            policy_id=self.policy_id,
            commands=commands,
            review_target_ids=review_ids,
            blocked_target_ids=blocked_ids,
            metadata=_reason_metadata(reasons, review_ids, blocked_ids),
        )


@dataclass
class RiskAwareRulePolicy(DecisionPolicy):
    policy_id: str = "risk_aware_rule"
    action_mask_builder: ActionMaskBuilder = field(
        default_factory=lambda: ActionMaskBuilder(
            max_target_uncertainty_mm=2.0,
            min_distance_to_keep_mm=5.0,
            max_risk_score=0.25,
            require_calibration_valid=True,
            require_robot_homed=True,
        )
    )

    def propose_plan(self, scene: SceneState) -> ActionPlan:
        ordered = sorted(
            scene.targets,
            key=lambda target: (
                target.risk_score,
                _target_distance_from_robot(target, scene.robot.position_mm),
                target.target_id,
            ),
        )
        return _build_plan(self, scene, ordered, self.action_mask_builder)


def _build_plan(
    policy: DecisionPolicy,
    scene: SceneState,
    ordered_targets: list[ActionTarget],
    action_mask_builder: ActionMaskBuilder,
) -> ActionPlan:
    mask = action_mask_builder.build(scene, policy.processed_target_ids)
    valid_ids = set(mask.valid_target_ids())
    reasons = mask.reasons_by_target()
    commands = [
        policy._command_for_target(target, index)
        for index, target in enumerate(ordered_targets, 1)
        if target.target_id in valid_ids
    ]
    review_ids = [
        target_id
        for target_id, target_reasons in reasons.items()
        if "human_review_required" in target_reasons
    ]
    blocked_ids = [
        target_id
        for target_id, target_reasons in reasons.items()
        if target_reasons and target_id not in review_ids
    ]
    return ActionPlan(
        plan_id=f"{policy.policy_id}_{scene.scene_id}",
        policy_id=policy.policy_id,
        commands=commands,
        review_target_ids=review_ids,
        blocked_target_ids=blocked_ids,
        metadata=_reason_metadata(reasons, review_ids, blocked_ids),
    )


def _target_distance_from_robot(target: ActionTarget, robot_position_mm: list[float]) -> float:
    point = _target_point_mm(target)
    return math.hypot(point[0] - robot_position_mm[0], point[1] - robot_position_mm[1])


def _target_point_mm(target: ActionTarget) -> list[float]:
    point = target.robot_point_mm
    if point is None and target.action_point_mm is not None:
        point = [target.action_point_mm[0], target.action_point_mm[1], 0.0]
    if point is None:
        point = [target.action_point_px[0], target.action_point_px[1], 0.0]
    return point


def _greedy_route(
    targets: list[ActionTarget],
    start_position_mm: list[float],
) -> tuple[list[ActionTarget], list[float]]:
    remaining = list(targets)
    current = list(start_position_mm)
    route: list[ActionTarget] = []
    leg_distances: list[float] = []
    while remaining:
        next_target = min(
            remaining,
            key=lambda target: (_target_distance_from_robot(target, current), target.target_id),
        )
        distance = _target_distance_from_robot(next_target, current)
        route.append(next_target)
        leg_distances.append(distance)
        current = _target_point_mm(next_target)
        remaining.remove(next_target)
    return route, leg_distances


def _review_and_blocked_ids(mask) -> tuple[list[str], list[str]]:
    reasons = mask.reasons_by_target()
    review_ids = [
        target_id
        for target_id, target_reasons in reasons.items()
        if "human_review_required" in target_reasons
    ]
    blocked_ids = [
        target_id
        for target_id, target_reasons in reasons.items()
        if target_reasons and target_id not in review_ids
    ]
    return review_ids, blocked_ids


def _reason_metadata(
    reasons: dict[str, list[str]],
    review_ids: list[str],
    blocked_ids: list[str],
) -> dict[str, dict[str, list[str]]]:
    return {
        "review_reasons_by_target": {
            target_id: list(reasons.get(target_id, []))
            for target_id in review_ids
        },
        "blocked_reasons_by_target": {
            target_id: list(reasons.get(target_id, []))
            for target_id in blocked_ids
        },
    }
