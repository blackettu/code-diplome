from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from seedling_sim import (
    ActuatorErrorModel,
    LogicalTraySimulator,
    SceneGeneratorConfig,
    SimScene,
    SimSceneDetectionNoiseModel,
    SimSceneGenerator,
)
from seedling_sim.schemas import SimTarget
from seedling_rl.reward import RewardConfig

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised only in environments without gymnasium
    gym = None
    spaces = None


@dataclass(frozen=True)
class TrayEnvConfig:
    grid_rows: int = 11
    grid_cols: int = 11
    cell_size_mm: tuple[float, float] = (33.0, 33.0)
    max_targets: int = 128
    max_steps: int = 256
    seed: int = 42
    scene_generator: SceneGeneratorConfig = field(default_factory=SceneGeneratorConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    actuator_xy_sigma_mm: float = 0.8
    actuator_drift_sigma_mm: float = 0.2
    actuator_zone_error_prob: float = 0.0
    actuator_zone_error_mm: float = 0.0
    noise_bbox_center_sigma_mm: float = 0.0
    noise_classification_error_prob: float = 0.0
    noise_missed_detection_prob: float = 0.0
    noise_false_positive_prob: float = 0.0
    render_mode: str = "ansi"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrayEnvConfig":
        generator_data = data.get("scene_generator", {})
        generator = SceneGeneratorConfig.from_dict(
            {
                "grid_rows": data.get("grid_rows", 11),
                "grid_cols": data.get("grid_cols", 11),
                "cell_size_mm": data.get("cell_size_mm", [33.0, 33.0]),
                **(generator_data if isinstance(generator_data, dict) else {}),
                "seed": data.get("seed", 42),
            }
        )
        actuator = data.get("actuator", {}) if isinstance(data.get("actuator", {}), dict) else {}
        noise = data.get("noise", {}) if isinstance(data.get("noise", {}), dict) else {}
        return cls(
            grid_rows=int(data.get("grid_rows", 11)),
            grid_cols=int(data.get("grid_cols", 11)),
            cell_size_mm=tuple(float(value) for value in data.get("cell_size_mm", [33.0, 33.0])),
            max_targets=int(data.get("max_targets", 128)),
            max_steps=int(data.get("max_steps", 256)),
            seed=int(data.get("seed", 42)),
            scene_generator=generator,
            reward=RewardConfig.from_dict(data.get("reward") if isinstance(data.get("reward"), dict) else None),
            actuator_xy_sigma_mm=float(actuator.get("xy_sigma_mm", 0.8)),
            actuator_drift_sigma_mm=float(actuator.get("drift_sigma_mm", 0.2)),
            actuator_zone_error_prob=float(actuator.get("zone_error_prob", 0.0)),
            actuator_zone_error_mm=float(actuator.get("zone_error_mm", 0.0)),
            noise_bbox_center_sigma_mm=float(noise.get("bbox_center_sigma_mm", 0.0)),
            noise_classification_error_prob=float(noise.get("classification_error_prob", 0.0)),
            noise_missed_detection_prob=float(noise.get("missed_detection_prob", 0.0)),
            noise_false_positive_prob=float(noise.get("false_positive_prob", 0.0)),
            render_mode=str(data.get("render_mode", "ansi")),
        )


class _FallbackEnv:
    metadata = {"render_modes": ["human", "rgb_array", "ansi"]}


BaseEnv = gym.Env if gym is not None else _FallbackEnv


class SeedlingTrayEnv(BaseEnv):  # type: ignore[misc, valid-type]
    metadata = {"render_modes": ["human", "rgb_array", "ansi"]}

    def __init__(self, config: TrayEnvConfig | dict[str, Any] | None = None):
        super().__init__()
        if config is None:
            config = TrayEnvConfig()
        if isinstance(config, dict):
            config = TrayEnvConfig.from_dict(config)
        self.config = config
        self.generator = SimSceneGenerator(config.scene_generator)
        self.simulator = LogicalTraySimulator(
            actuator_model=ActuatorErrorModel(
                xy_sigma_mm=config.actuator_xy_sigma_mm,
                drift_sigma_mm=config.actuator_drift_sigma_mm,
                zone_error_prob=config.actuator_zone_error_prob,
                zone_error_mm=config.actuator_zone_error_mm,
                seed=config.seed,
            ),
            correct_remove_reward=config.reward.correct_remove,
            crop_damage_penalty=config.reward.crop_damage,
            miss_penalty=config.reward.missed_target,
            move_mm_penalty=config.reward.move_mm_penalty,
            seed=config.seed,
        )
        self.scene: SimScene | None = None
        self._episode_index = 0
        self._attempts_by_target: dict[str, int] = {}
        self._last_info: dict[str, Any] = {}
        self.render_mode = config.render_mode
        if spaces is not None:
            self.action_space = spaces.Discrete(config.max_targets + 2)
            self.observation_space = spaces.Dict(
                {
                    "cell_tensor": spaces.Box(
                        low=0.0,
                        high=np.inf,
                        shape=(config.grid_rows, config.grid_cols, 11),
                        dtype=np.float32,
                    ),
                    "target_features": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(config.max_targets, 8),
                        dtype=np.float32,
                    ),
                    "target_mask": spaces.MultiBinary(config.max_targets),
                    "robot_state": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                    "action_mask": spaces.MultiBinary(config.max_targets + 2),
                }
            )

    @property
    def review_action(self) -> int:
        return self.config.max_targets

    @property
    def stop_action(self) -> int:
        return self.config.max_targets + 1

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if gym is not None:
            super().reset(seed=seed)
        if seed is not None:
            self.generator = SimSceneGenerator(
                SceneGeneratorConfig(
                    grid_rows=self.config.scene_generator.grid_rows,
                    grid_cols=self.config.scene_generator.grid_cols,
                    cell_size_mm=self.config.scene_generator.cell_size_mm,
                    p_empty=self.config.scene_generator.p_empty,
                    p_single_crop=self.config.scene_generator.p_single_crop,
                    p_multiple_crop=self.config.scene_generator.p_multiple_crop,
                    p_weed_present=self.config.scene_generator.p_weed_present,
                    p_unknown_present=self.config.scene_generator.p_unknown_present,
                    max_extra_crops=self.config.scene_generator.max_extra_crops,
                    seed=seed,
                )
            )
        self._episode_index += 1
        effective_seed = seed if seed is not None else self.config.seed + self._episode_index
        if options and isinstance(options.get("scene"), SimScene):
            scene = options["scene"]
        else:
            scene = self.generator.generate(f"rl_scene_{self._episode_index:06d}")
        scene = self._apply_detection_noise(scene, effective_seed)
        self.scene = self.simulator.reset(scene)
        self._attempts_by_target = {}
        observation = self._observation()
        info = {
            "scene_id": self.scene.scene_id,
            "action_mask": observation["action_mask"],
            "targets": len(self.scene.targets),
        }
        self._last_info = info
        return observation, info

    def step(self, action: int):
        if self.scene is None:
            raise RuntimeError("SeedlingTrayEnv.reset() must be called before step()")
        action = int(action)
        reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {"action": action}

        valid_targets = self._ordered_targets()
        if action == self.stop_action:
            remaining = [target for target in valid_targets if not target.processed]
            reward += self.config.reward.episode_complete if not remaining else self.config.reward.stop_with_remaining_target
            terminated = True
            info["event"] = "stop"
            info["remaining_targets"] = len(remaining)
        elif action == self.review_action:
            review_targets = [target for target in valid_targets if self._requires_review(target)]
            reward += self.config.reward.correct_review if review_targets else self.config.reward.unsafe_action
            for target in review_targets:
                target.processed = True
            info["event"] = "review"
            info["reviewed_targets"] = len(review_targets)
        elif 0 <= action < self.config.max_targets:
            target = valid_targets[action] if action < len(valid_targets) else None
            if target is None:
                reward += self.config.reward.unsafe_action
                terminated = True
                info["event"] = "invalid_target_index"
            elif target.processed:
                reward += self.config.reward.repeated_action
                info["event"] = "repeated_target"
            elif not self._safe_to_act(target):
                reward += self.config.reward.unsafe_action
                terminated = True
                info["event"] = "unsafe_target"
            else:
                outcome = self.simulator.step_target(target)
                reward += outcome.reward
                if outcome.crop_damage:
                    terminated = True
                self._attempts_by_target[target.target_id] = self._attempts_by_target.get(target.target_id, 0) + 1
                info.update({"event": "target_step", "outcome": outcome.to_dict()})
        else:
            reward += self.config.reward.unsafe_action
            terminated = True
            info["event"] = "invalid_action"

        if self.scene.step_index >= self.config.max_steps:
            truncated = True
        if not [target for target in self._ordered_targets() if not target.processed and self._safe_to_act(target)]:
            terminated = True
        observation = self._observation()
        info["action_mask"] = observation["action_mask"]
        self._last_info = info
        return observation, float(reward), terminated, truncated, info

    def render(self):
        if self.scene is None:
            return "" if self.config.render_mode == "ansi" else None
        if self.config.render_mode == "ansi":
            return _render_ansi(self.scene)
        if self.config.render_mode == "human":
            text = _render_ansi(self.scene)
            print(text)
            return None
        if self.config.render_mode == "rgb_array":
            return _render_rgb_array(self.scene)
        return _render_ansi(self.scene)

    def action_masks(self) -> np.ndarray:
        return self._observation()["action_mask"]

    def _observation(self) -> dict[str, np.ndarray]:
        assert self.scene is not None
        cell_tensor = np.zeros((self.config.grid_rows, self.config.grid_cols, 11), dtype=np.float32)
        targets = self._ordered_targets()
        target_features = np.zeros((self.config.max_targets, 8), dtype=np.float32)
        target_mask = np.zeros((self.config.max_targets,), dtype=np.int8)
        action_mask = np.zeros((self.config.max_targets + 2,), dtype=np.int8)

        crop_counts: dict[tuple[int, int], int] = {}
        weed_counts: dict[tuple[int, int], int] = {}
        unknown_counts: dict[tuple[int, int], int] = {}
        max_conf: dict[tuple[int, int], float] = {}
        for plant in self.scene.plants:
            key = (plant.row, plant.col)
            max_conf[key] = max(max_conf.get(key, 0.0), plant.confidence)
            if plant.class_name in {"crop_seedling", "seedling", "seedlings"}:
                crop_counts[key] = crop_counts.get(key, 0) + 1
            elif plant.class_name == "weed":
                weed_counts[key] = weed_counts.get(key, 0) + 1
            else:
                unknown_counts[key] = unknown_counts.get(key, 0) + 1
        target_by_object = {target.object_id: target for target in targets}
        for row in range(self.config.grid_rows):
            for col in range(self.config.grid_cols):
                key = (row, col)
                cell_tensor[row, col, 0] = crop_counts.get(key, 0)
                cell_tensor[row, col, 1] = weed_counts.get(key, 0)
                cell_tensor[row, col, 2] = unknown_counts.get(key, 0)
                cell_tensor[row, col, 3] = 1.0 if crop_counts.get(key, 0) > 1 else 0.0
                cell_tensor[row, col, 4] = 1.0 if weed_counts.get(key, 0) > 0 else 0.0
                cell_tensor[row, col, 5] = max_conf.get(key, 0.0)
                related = [
                    target
                    for plant in self.scene.plants
                    if plant.row == row and plant.col == col and (target := target_by_object.get(plant.object_id)) is not None
                ]
                cell_tensor[row, col, 6] = min((target.min_distance_to_keep_mm or 999.0 for target in related), default=999.0)
                cell_tensor[row, col, 7] = max((target.uncertainty_radius_mm for target in related), default=0.0)
                cell_tensor[row, col, 8] = max((self._attempts_by_target.get(target.target_id, 0) for target in related), default=0)
                cell_tensor[row, col, 9] = 1.0 if any(self._safe_to_act(target) for target in related) else 0.0
                cell_tensor[row, col, 10] = 1.0 if any(self._requires_review(target) for target in related) else 0.0

        for index, target in enumerate(targets[: self.config.max_targets]):
            plant = self.scene.plant_by_id(target.object_id)
            target_mask[index] = 0 if target.processed else 1
            action_mask[index] = 1 if self._safe_to_act(target) and not target.processed else 0
            target_features[index] = np.array(
                [
                    target.point_mm[0],
                    target.point_mm[1],
                    1.0 if plant and plant.class_name in {"crop_seedling", "seedling", "seedlings"} else 0.0,
                    1.0 if plant and plant.class_name == "weed" else 0.0,
                    1.0 if plant and plant.class_name in {"unknown_plant", "unknown"} else 0.0,
                    plant.confidence if plant else 0.0,
                    target.uncertainty_radius_mm,
                    float(self._attempts_by_target.get(target.target_id, 0)),
                ],
                dtype=np.float32,
            )
        action_mask[self.review_action] = 1 if any(self._requires_review(target) and not target.processed for target in targets) else 0
        action_mask[self.stop_action] = 1
        return {
            "cell_tensor": cell_tensor,
            "target_features": target_features,
            "target_mask": target_mask,
            "robot_state": np.array(self.scene.robot_position_mm, dtype=np.float32),
            "action_mask": action_mask,
        }

    def _ordered_targets(self) -> list[SimTarget]:
        assert self.scene is not None
        return sorted(self.scene.targets, key=lambda target: target.target_id)

    def _safe_to_act(self, target: SimTarget) -> bool:
        return (
            not target.processed
            and target.target_type in {"remove_extra_crop", "remove_weed"}
            and target.uncertainty_radius_mm <= 2.0
            and (target.min_distance_to_keep_mm is None or target.min_distance_to_keep_mm >= 5.0)
        )

    def _requires_review(self, target: SimTarget) -> bool:
        return target.target_type == "human_review_required" or target.uncertainty_radius_mm > 2.0

    def _apply_detection_noise(self, scene: SimScene, seed: int) -> SimScene:
        model = SimSceneDetectionNoiseModel(
            bbox_center_sigma_mm=self.config.noise_bbox_center_sigma_mm,
            classification_error_prob=self.config.noise_classification_error_prob,
            missed_detection_prob=self.config.noise_missed_detection_prob,
            false_positive_prob=self.config.noise_false_positive_prob,
            seed=seed,
        )
        return model.apply(scene)


def _render_ansi(scene: SimScene) -> str:
    cells = [["." for _ in range(scene.grid_cols)] for _ in range(scene.grid_rows)]
    for plant in scene.plants:
        marker = "x" if plant.removed else ("w" if plant.class_name == "weed" else ("?" if plant.class_name == "unknown_plant" else "c"))
        cells[plant.row][plant.col] = marker
    return "\n".join("".join(row) for row in cells)


def _render_rgb_array(scene: SimScene) -> np.ndarray:
    image = np.full((scene.grid_rows, scene.grid_cols, 3), 240, dtype=np.uint8)
    for plant in scene.plants:
        if plant.removed:
            color = [128, 128, 128]
        elif plant.class_name == "weed":
            color = [190, 100, 40]
        elif plant.class_name == "unknown_plant":
            color = [140, 90, 180]
        else:
            color = [40, 140, 80]
        image[plant.row, plant.col] = color
    return image
