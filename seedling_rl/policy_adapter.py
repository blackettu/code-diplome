from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from seedling_core.schemas import ActionPlan, SceneState
from seedling_decision.action_masks import ActionMaskBuilder
from seedling_decision.policies.base import DecisionPolicy


@dataclass
class RLPolicyAdapter(DecisionPolicy):
    """Connect a trained RL policy through the DecisionPolicy API.

    The adapter never executes commands. It selects a target index and returns an
    ActionPlan that still has to pass SafetyGate before any robot adapter sees it.
    """

    policy_id: str = "rl_policy_adapter"
    model: Any | None = None
    max_targets: int = 128
    recurrent: bool = False
    action_mask_builder: ActionMaskBuilder = field(default_factory=ActionMaskBuilder)
    _recurrent_state: Any | None = field(default=None, init=False, repr=False)
    _episode_start: bool = field(default=True, init=False, repr=False)

    @classmethod
    def from_sb3(
        cls,
        checkpoint: str,
        policy_id: str = "rl_policy_adapter",
        max_targets: int = 128,
        recurrent: bool | None = None,
    ) -> "RLPolicyAdapter":
        model = _load_sb3_model(checkpoint)
        return cls(
            policy_id=policy_id,
            model=model,
            max_targets=max_targets,
            recurrent=_is_recurrent_model(model) if recurrent is None else recurrent,
        )

    def reset(self, scene: SceneState) -> None:
        super().reset(scene)
        self._recurrent_state = None
        self._episode_start = True

    def propose_plan(self, scene: SceneState) -> ActionPlan:
        mask = self.action_mask_builder.build(scene, self.processed_target_ids)
        reasons = mask.reasons_by_target()
        if self.model is None:
            review_ids = [target.target_id for target in scene.targets if target.human_review_required]
            return self._empty_plan(
                scene,
                review_target_ids=review_ids,
                metadata=_metadata(action_kind="no_model", reasons=reasons, review_ids=review_ids),
            )
        observation = self._observation(scene)
        if self.recurrent:
            action, self._recurrent_state = self.model.predict(
                observation,
                state=self._recurrent_state,
                episode_start=np.asarray([self._episode_start], dtype=bool),
                deterministic=True,
            )
            self._episode_start = False
        else:
            action, _ = self.model.predict(observation, deterministic=True)
        action_index = int(np.asarray(action).item())
        if action_index == self.max_targets:
            review_ids = [target.target_id for target in scene.targets if target.human_review_required]
            return self._empty_plan(
                scene,
                review_target_ids=review_ids,
                metadata=_metadata(
                    action_index=action_index,
                    action_kind="review",
                    reasons=reasons,
                    review_ids=review_ids,
                ),
            )
        if action_index == self.max_targets + 1:
            return self._empty_plan(
                scene,
                metadata=_metadata(action_index=action_index, action_kind="stop", reasons=reasons),
            )
        if action_index > self.max_targets + 1:
            return self._empty_plan(
                scene,
                metadata=_metadata(
                    action_index=action_index,
                    action_kind="invalid_action_index",
                    reasons=reasons,
                    extra={"invalid_action_index": action_index},
                ),
            )

        ordered_targets = sorted(scene.targets, key=lambda target: target.target_id)
        if action_index >= len(ordered_targets):
            return self._empty_plan(
                scene,
                metadata=_metadata(
                    action_index=action_index,
                    action_kind="target_index_out_of_range",
                    reasons=reasons,
                    extra={"target_count": len(ordered_targets)},
                ),
            )
        valid_ids = set(mask.valid_target_ids())
        target = ordered_targets[action_index]
        if target.target_id not in valid_ids:
            return self._empty_plan(
                scene,
                blocked_target_ids=[target.target_id],
                metadata=_metadata(
                    action_index=action_index,
                    action_kind="blocked_target",
                    reasons=reasons,
                    blocked_ids=[target.target_id],
                ),
            )
        return ActionPlan(
            plan_id=f"{self.policy_id}_{scene.scene_id}",
            policy_id=self.policy_id,
            commands=[self._command_for_target(target, 1)],
            review_target_ids=[],
            blocked_target_ids=[],
            metadata=_metadata(
                action_index=action_index,
                action_kind="target_command",
                reasons=reasons,
                extra={"selected_target_id": target.target_id},
            ),
        )

    def _empty_plan(
        self,
        scene: SceneState,
        review_target_ids: list[str] | None = None,
        blocked_target_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ActionPlan:
        return ActionPlan(
            plan_id=f"{self.policy_id}_{scene.scene_id}",
            policy_id=self.policy_id,
            commands=[],
            review_target_ids=review_target_ids or [],
            blocked_target_ids=blocked_target_ids or [],
            metadata=metadata or {},
        )

    def _observation(self, scene: SceneState) -> dict[str, np.ndarray]:
        cell_tensor = np.zeros((scene.tray.grid_rows, scene.tray.grid_cols, 11), dtype=np.float32)
        cell_by_id = {cell.cell_id: cell for cell in scene.cells}
        target_features = np.zeros((self.max_targets, 8), dtype=np.float32)
        target_mask = np.zeros((self.max_targets,), dtype=np.int8)
        action_mask = np.zeros((self.max_targets + 2,), dtype=np.int8)
        ordered_targets = sorted(scene.targets, key=lambda target: target.target_id)
        valid_ids = set(self.action_mask_builder.build(scene, self.processed_target_ids).valid_target_ids())
        for cell in scene.cells:
            if 0 <= cell.row < scene.tray.grid_rows and 0 <= cell.col < scene.tray.grid_cols:
                cell_tensor[cell.row, cell.col, 0] = len(cell.object_ids)
                cell_tensor[cell.row, cell.col, 3] = 1.0 if cell.state == "multiple_crop" else 0.0
                cell_tensor[cell.row, cell.col, 5] = cell.state_confidence
                cell_tensor[cell.row, cell.col, 9] = 0.0 if cell.human_review_required else 1.0
                cell_tensor[cell.row, cell.col, 10] = 1.0 if cell.human_review_required else 0.0
        for index, target in enumerate(ordered_targets[: self.max_targets]):
            point = target.action_point_mm or target.action_point_px
            cell = cell_by_id.get(target.cell_id)
            if cell is not None and 0 <= cell.row < scene.tray.grid_rows and 0 <= cell.col < scene.tray.grid_cols:
                cell_tensor[cell.row, cell.col, 6] = min(
                    cell_tensor[cell.row, cell.col, 6] or 999.0,
                    target.min_distance_to_keep_mm or 999.0,
                )
                cell_tensor[cell.row, cell.col, 7] = max(
                    cell_tensor[cell.row, cell.col, 7],
                    target.uncertainty_radius_mm or 0.0,
                )
            target_features[index] = np.array(
                [
                    point[0],
                    point[1],
                    1.0 if target.target_type == "remove_extra_crop" else 0.0,
                    1.0 if target.target_type == "remove_weed" else 0.0,
                    1.0 if target.human_review_required else 0.0,
                    1.0 - target.risk_score,
                    target.uncertainty_radius_mm or 0.0,
                    1.0 if target.target_id in self.processed_target_ids else 0.0,
                ],
                dtype=np.float32,
            )
            target_mask[index] = 1
            action_mask[index] = 1 if target.target_id in valid_ids else 0
        action_mask[self.max_targets] = 1 if any(target.human_review_required for target in scene.targets) else 0
        action_mask[self.max_targets + 1] = 1
        return {
            "cell_tensor": cell_tensor,
            "target_features": target_features,
            "target_mask": target_mask,
            "robot_state": np.array(scene.robot.position_mm, dtype=np.float32),
            "action_mask": action_mask,
        }


def _load_sb3_model(checkpoint: str) -> Any:
    try:
        from sb3_contrib import MaskablePPO
    except ImportError:
        MaskablePPO = None
    try:
        from sb3_contrib import RecurrentPPO
    except ImportError:
        RecurrentPPO = None
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        if MaskablePPO is None and RecurrentPPO is None:
            raise RuntimeError("stable-baselines3 or sb3-contrib is required to load RL checkpoints") from exc
        PPO = None
    errors: list[Exception] = []
    for cls in [RecurrentPPO, MaskablePPO, PPO]:
        if cls is None:
            continue
        try:
            return cls.load(checkpoint)
        except Exception as exc:  # pragma: no cover - depends on external checkpoint format
            errors.append(exc)
    raise RuntimeError(f"Could not load RL checkpoint {checkpoint!r}: {errors}")


def _is_recurrent_model(model: Any) -> bool:
    class_name = model.__class__.__name__.lower()
    policy_class = str(getattr(model, "policy_class", "")).lower()
    policy = getattr(model, "policy", None)
    policy_name = policy.__class__.__name__.lower() if policy is not None else ""
    return "recurrent" in class_name or "lstm" in class_name or "recurrent" in policy_class or "lstm" in policy_name


def _metadata(
    action_kind: str,
    reasons: dict[str, list[str]],
    action_index: int | None = None,
    review_ids: list[str] | None = None,
    blocked_ids: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    review_ids = review_ids or []
    blocked_ids = blocked_ids or []
    payload: dict[str, Any] = {
        "rl_action_kind": action_kind,
        "review_reasons_by_target": {
            target_id: list(reasons.get(target_id, [])) or ["human_review_required"]
            for target_id in review_ids
        },
        "blocked_reasons_by_target": {
            target_id: list(reasons.get(target_id, [])) or ["blocked_by_action_mask"]
            for target_id in blocked_ids
        },
    }
    if action_index is not None:
        payload["rl_action_index"] = int(action_index)
    if extra:
        payload.update(extra)
    return payload
