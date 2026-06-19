from __future__ import annotations

import unittest

from seedling_rl import SeedlingTrayEnv, TrayEnvConfig, evaluate_baseline_suite
from seedling_sim import SceneGeneratorConfig, SimSceneGenerator, render_scene_html, render_scene_svg

try:
    from gymnasium.utils.env_checker import check_env as gym_check_env
except ImportError:  # pragma: no cover - depends on optional RL extras
    gym_check_env = None


class RLEnvTests(unittest.TestCase):
    def test_scene_generator_creates_targets_and_round_trips(self) -> None:
        generator = SimSceneGenerator(
            SceneGeneratorConfig(
                grid_rows=2,
                grid_cols=2,
                p_empty=0.0,
                p_single_crop=0.0,
                p_multiple_crop=1.0,
                p_weed_present=0.0,
                seed=42,
            )
        )

        scene = generator.generate("scene_test")
        restored = scene.from_dict(scene.to_dict())

        self.assertEqual(restored.to_dict(), scene.to_dict())
        self.assertGreater(len(scene.targets), 0)

    def test_static_renderer_outputs_svg_and_html(self) -> None:
        scene = SimSceneGenerator(
            SceneGeneratorConfig(grid_rows=1, grid_cols=1, p_empty=0.0, p_single_crop=0.0, p_multiple_crop=1.0, seed=1)
        ).generate("scene_render")

        svg = render_scene_svg(scene)
        html = render_scene_html(scene)

        self.assertIn("<svg", svg)
        self.assertIn("scene_render", html)

    def test_seedling_tray_env_reset_step_and_action_mask(self) -> None:
        config = TrayEnvConfig(
            grid_rows=2,
            grid_cols=2,
            max_targets=16,
            max_steps=10,
            seed=42,
            scene_generator=SceneGeneratorConfig(
                grid_rows=2,
                grid_cols=2,
                p_empty=0.0,
                p_single_crop=0.0,
                p_multiple_crop=1.0,
                p_weed_present=0.0,
                seed=42,
            ),
            actuator_xy_sigma_mm=0.0,
            actuator_drift_sigma_mm=0.0,
        )
        env = SeedlingTrayEnv(config)

        observation, info = env.reset(seed=42)
        action_mask = observation["action_mask"]
        first_action = next(index for index, value in enumerate(action_mask) if int(value) == 1 and index < config.max_targets)
        next_observation, reward, terminated, truncated, step_info = env.step(first_action)

        self.assertEqual(info["targets"], len(env.scene.targets))
        self.assertIn("cell_tensor", observation)
        self.assertIn("action_mask", next_observation)
        self.assertFalse(truncated)
        self.assertEqual(step_info["event"], "target_step")
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)

    def test_tray_env_config_applies_detection_noise_block(self) -> None:
        config = TrayEnvConfig.from_dict(
            {
                "grid_rows": 1,
                "grid_cols": 1,
                "max_targets": 4,
                "seed": 7,
                "scene_generator": {
                    "p_empty": 0.0,
                    "p_single_crop": 1.0,
                    "p_multiple_crop": 0.0,
                    "p_weed_present": 0.0,
                },
                "noise": {
                    "missed_detection_prob": 1.0,
                    "false_positive_prob": 1.0,
                },
            }
        )
        env = SeedlingTrayEnv(config)

        observation, info = env.reset(seed=7)

        assert env.scene is not None
        self.assertEqual(info["targets"], 1)
        self.assertEqual(len(env.scene.plants), 1)
        self.assertIn("false_positive_noise", env.scene.plants[0].attributes)
        self.assertEqual(env.scene.targets[0].target_type, "human_review_required")
        self.assertEqual(observation["action_mask"][env.review_action], 1)
        self.assertEqual(env.scene.metadata["detection_noise"]["false_positive_prob"], 1.0)

    @unittest.skipIf(gym_check_env is None, "gymnasium is not installed")
    def test_seedling_tray_env_passes_gymnasium_checker(self) -> None:
        config = TrayEnvConfig(
            grid_rows=2,
            grid_cols=2,
            max_targets=16,
            max_steps=10,
            seed=42,
            scene_generator=SceneGeneratorConfig(
                grid_rows=2,
                grid_cols=2,
                p_empty=0.0,
                p_single_crop=0.0,
                p_multiple_crop=1.0,
                p_weed_present=0.0,
                seed=42,
            ),
            actuator_xy_sigma_mm=0.0,
            actuator_drift_sigma_mm=0.0,
            render_mode="ansi",
        )

        gym_check_env(SeedlingTrayEnv(config), skip_render_check=True)

    def test_baseline_suite_returns_policy_metrics(self) -> None:
        config = TrayEnvConfig(
            grid_rows=2,
            grid_cols=2,
            max_targets=16,
            max_steps=10,
            seed=42,
            scene_generator=SceneGeneratorConfig(
                grid_rows=2,
                grid_cols=2,
                p_empty=0.0,
                p_single_crop=0.0,
                p_multiple_crop=1.0,
                p_weed_present=0.0,
                seed=42,
            ),
            actuator_xy_sigma_mm=0.0,
            actuator_drift_sigma_mm=0.0,
        )

        results = evaluate_baseline_suite(
            config,
            episodes=2,
            policies=["noop", "raster_scan", "nearest_neighbor", "route_planning"],
        )

        self.assertEqual([item.policy for item in results], ["noop", "raster_scan", "nearest_neighbor", "route_planning"])
        self.assertEqual(results[0].episodes, 2)
        self.assertGreaterEqual(results[1].actions_per_tray, 1.0)
        self.assertGreaterEqual(results[3].actions_per_tray, 1.0)


if __name__ == "__main__":
    unittest.main()
