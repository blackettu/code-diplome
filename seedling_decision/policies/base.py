from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from seedling_core.schemas import ActionCommand, ActionPlan, ActionTarget, SceneState


@dataclass
class DecisionPolicy(ABC):
    policy_id: str
    processed_target_ids: set[str] = field(default_factory=set)

    def reset(self, scene: SceneState) -> None:
        self.processed_target_ids.clear()

    @abstractmethod
    def propose_plan(self, scene: SceneState) -> ActionPlan:
        raise NotImplementedError

    def next_action(self, scene: SceneState) -> ActionCommand | None:
        plan = self.propose_plan(scene)
        return plan.commands[0] if plan.commands else None

    def _command_for_target(self, target: ActionTarget, sequence: int) -> ActionCommand:
        coordinate_source = "robot_point_mm"
        robot_point = target.robot_point_mm
        if robot_point is None and target.action_point_mm is not None:
            robot_point = [target.action_point_mm[0], target.action_point_mm[1], 0.0]
            coordinate_source = "action_point_mm"
        if robot_point is None:
            robot_point = [target.action_point_px[0], target.action_point_px[1], 0.0]
            coordinate_source = "action_point_px"
        return ActionCommand(
            command_id=f"{self.policy_id}_cmd_{sequence:04d}",
            target_id=target.target_id,
            action_type="move_and_mark",
            robot_point_mm=robot_point,
            tool_profile="pointer_only",
            requires_operator_confirmation=target.human_review_required,
            safety_gate_result="not_evaluated",
            created_by_policy=self.policy_id,
            metadata={
                "target_type": target.target_type,
                "coordinate_source": coordinate_source,
            },
        )
