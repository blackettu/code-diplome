from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from seedling_rl.envs import SeedlingTrayEnv, TrayEnvConfig


@dataclass(frozen=True)
class VectorizedEnvSpec:
    n_envs: int
    seed: int
    backend: str
    env_config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def vectorized_env_spec(config: TrayEnvConfig, n_envs: int = 1, backend: str = "dummy_vec_env") -> VectorizedEnvSpec:
    if n_envs <= 0:
        raise ValueError("n_envs must be positive")
    return VectorizedEnvSpec(n_envs=n_envs, seed=config.seed, backend=backend, env_config=_config_dict(config))


def make_vectorized_env(config: TrayEnvConfig, n_envs: int = 1):
    try:
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:
        raise RuntimeError("stable-baselines3 is required to create vectorized envs") from exc
    return DummyVecEnv([_env_factory(config, index) for index in range(n_envs)])


def _env_factory(config: TrayEnvConfig, index: int) -> Callable[[], SeedlingTrayEnv]:
    def factory() -> SeedlingTrayEnv:
        env_config = TrayEnvConfig.from_dict(_config_dict(config))
        env_config = _with_seed(env_config, config.seed + index)
        return SeedlingTrayEnv(env_config)

    return factory


def _with_seed(config: TrayEnvConfig, seed: int) -> TrayEnvConfig:
    data = _config_dict(config)
    data["seed"] = seed
    return TrayEnvConfig.from_dict(data)


def _config_dict(config: TrayEnvConfig) -> dict[str, Any]:
    return {
        "grid_rows": config.grid_rows,
        "grid_cols": config.grid_cols,
        "cell_size_mm": list(config.cell_size_mm),
        "max_targets": config.max_targets,
        "max_steps": config.max_steps,
        "seed": config.seed,
        "scene_generator": config.scene_generator.__dict__,
        "reward": config.reward.__dict__,
        "actuator": {
            "xy_sigma_mm": config.actuator_xy_sigma_mm,
            "drift_sigma_mm": config.actuator_drift_sigma_mm,
            "zone_error_prob": config.actuator_zone_error_prob,
            "zone_error_mm": config.actuator_zone_error_mm,
        },
        "noise": {
            "bbox_center_sigma_mm": config.noise_bbox_center_sigma_mm,
            "classification_error_prob": config.noise_classification_error_prob,
            "missed_detection_prob": config.noise_missed_detection_prob,
            "false_positive_prob": config.noise_false_positive_prob,
        },
        "render_mode": config.render_mode,
    }
