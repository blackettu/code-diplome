from __future__ import annotations

import json
import time
import tempfile
import unittest
from pathlib import Path

from seedling_rl import train_from_config
from seedling_rl import SeedlingTrayEnv, TrayEnvConfig
from seedling_sim import SceneGeneratorConfig, SimSceneGenerator
from seedling_vision import batch_predict
from seedling_vision.adapters import MockDetector


class PerformanceSmokeTests(unittest.TestCase):
    def test_batch_predict_mock_detector_does_not_regress_grossly(self) -> None:
        detector = MockDetector(image_size_px=[64, 64])
        detector.load("mock://detector")
        images = [f"tray_{index:03d}.jpg" for index in range(100)]

        started = time.perf_counter()
        results = batch_predict(detector, images)
        elapsed = time.perf_counter() - started

        self.assertEqual(len(results), 100)
        self.assertLess(elapsed, 5.0)

    def test_scene_generation_and_env_reset_are_bounded(self) -> None:
        config = SceneGeneratorConfig(grid_rows=4, grid_cols=4, seed=42)
        generator = SimSceneGenerator(config)
        env = SeedlingTrayEnv(TrayEnvConfig(grid_rows=4, grid_cols=4, max_targets=64, seed=42))

        started = time.perf_counter()
        scenes = generator.generate_many(50)
        for episode in range(20):
            env.reset(seed=episode)
        elapsed = time.perf_counter() - started

        self.assertEqual(len(scenes), 50)
        self.assertLess(elapsed, 10.0)

    def test_rl_training_dry_run_path_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_config = root / "tray_env.yaml"
            env_config.write_text(
                "\n".join(
                    [
                        "grid_rows: 2",
                        "grid_cols: 2",
                        "cell_size_mm: [33.0, 33.0]",
                        "max_targets: 8",
                        "max_steps: 8",
                        "seed: 7",
                    ]
                ),
                encoding="utf-8",
            )
            config = root / "rl.yaml"
            run_dir = root / "rl_run"
            config.write_text(
                "\n".join(
                    [
                        "algorithm: ppo",
                        f"env_config: {env_config.as_posix()}",
                        "total_timesteps: 16",
                        "seed: 7",
                        "policy: MultiInputPolicy",
                        "learning_rate: 0.0003",
                        "n_steps: 8",
                        "batch_size: 8",
                        "gamma: 0.99",
                        "outputs:",
                        f"  run_dir: {run_dir.as_posix()}",
                    ]
                ),
                encoding="utf-8",
            )

            started = time.perf_counter()
            result = train_from_config(config, dry_run=True)
            elapsed = time.perf_counter() - started

            summary = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(result["ok"])
            self.assertTrue(summary["dry_run"])
            self.assertTrue((run_dir / "run_snapshot.json").exists())
            self.assertTrue((run_dir / "artifact_registry.json").exists())
            self.assertLess(elapsed, 5.0)

    def test_rl_episode_stepping_is_bounded(self) -> None:
        env = SeedlingTrayEnv(TrayEnvConfig(grid_rows=4, grid_cols=4, max_targets=64, max_steps=25, seed=42))

        started = time.perf_counter()
        steps = 0
        for episode in range(20):
            observation, _ = env.reset(seed=episode)
            for _ in range(env.config.max_steps):
                action = next(index for index, value in enumerate(observation["action_mask"]) if int(value) == 1)
                observation, _, terminated, truncated, _ = env.step(action)
                steps += 1
                if terminated or truncated:
                    break
        elapsed = time.perf_counter() - started

        self.assertGreater(steps, 20)
        self.assertLess(elapsed, 10.0)


if __name__ == "__main__":
    unittest.main()
