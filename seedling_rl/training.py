from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from seedling_core.config import load_config_file
from seedling_core.run_snapshot import save_command_snapshot
from seedling_reports.registry import write_run_registry
from seedling_rl.baseline_eval import BaselineEvaluationResult, select_baseline_action
from seedling_rl.callbacks import RLMetricsLogger, make_sb3_metric_callback
from seedling_rl.envs import SeedlingTrayEnv, TrayEnvConfig
from seedling_rl.vectorized_env import make_vectorized_env
from seedling_sim import ReplayLogger


@dataclass(frozen=True)
class RLTrainingConfig:
    algorithm: str
    env_config: str
    total_timesteps: int
    seed: int
    policy: str
    learning_rate: float
    n_steps: int
    batch_size: int
    gamma: float
    run_dir: str
    tensorboard_log: str | None = None
    n_envs: int = 1
    recurrent: bool = False

    @classmethod
    def from_file(cls, path: str | Path) -> "RLTrainingConfig":
        data = load_config_file(path)
        outputs = data.get("outputs", {}) if isinstance(data.get("outputs"), dict) else {}
        algorithm = str(data.get("algorithm", "ppo"))
        return cls(
            algorithm=algorithm,
            env_config=str(data.get("env_config", "configs/simulation/tray_env_v0.yaml")),
            total_timesteps=int(data.get("total_timesteps", 10000)),
            seed=int(data.get("seed", 42)),
            policy=str(data.get("policy", "MultiInputPolicy")),
            learning_rate=float(data.get("learning_rate", 0.0003)),
            n_steps=int(data.get("n_steps", 2048)),
            batch_size=int(data.get("batch_size", 256)),
            gamma=float(data.get("gamma", 0.99)),
            run_dir=str(outputs.get("run_dir", "runs/rl/default")),
            tensorboard_log=str(outputs["tensorboard_dir"]) if outputs.get("tensorboard_dir") else None,
            n_envs=int(data.get("n_envs", 1)),
            recurrent=_bool_config(data.get("recurrent"), default=_is_recurrent_algorithm(algorithm)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def train_from_config(config_path: str | Path, dry_run: bool = False) -> dict[str, Any]:
    config = RLTrainingConfig.from_file(config_path)
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = run_dir / "rl_config.json"
    config_snapshot.write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    metric_logger = RLMetricsLogger(run_dir / "rl_metrics.jsonl", run_dir / "rl_metrics.csv")
    if dry_run:
        metric_logger.log_event(phase="train", event="dry_run", metadata=config.to_dict())
        metric_logger.close()
        payload = {
            "ok": True,
            "dry_run": True,
            "config": config.to_dict(),
            "run_dir": str(run_dir),
            "metric_log_path": str(metric_logger.jsonl_path),
            "metric_table_path": str(metric_logger.csv_path),
            "metric_summary": metric_logger.summary(),
        }
        train_summary = run_dir / "train_summary.json"
        training_log = run_dir / "training_log.jsonl"
        _write_jsonl(training_log, [{"event": "dry_run", "config": config.to_dict()}])
        snapshot = save_command_snapshot(
            run_dir,
            "seedling-rl:train",
            command_args={"config": str(config_path), "dry_run": dry_run},
            input_paths=[config_path, config.env_config],
            output_paths=[config_snapshot, train_summary, training_log, metric_logger.jsonl_path, metric_logger.csv_path],
            metadata={"algorithm": config.algorithm, "seed": config.seed, "dry_run": dry_run},
        )
        payload["run_snapshot"] = snapshot
        _write_json(train_summary, payload)
        write_run_registry(
            run_dir,
            "rl-train:dry-run",
            [
                config_snapshot,
                train_summary,
                training_log,
                metric_logger.jsonl_path,
                metric_logger.csv_path,
                snapshot,
            ],
        )
        return payload

    model_cls = _algorithm_class(config.algorithm)
    env_config = TrayEnvConfig.from_dict(load_config_file(config.env_config))
    env = _training_env(config, env_config)
    model = model_cls(
        config.policy,
        env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        gamma=config.gamma,
        seed=config.seed,
        tensorboard_log=config.tensorboard_log,
        verbose=0,
    )
    callback = make_sb3_metric_callback(metric_logger)
    if callback is not None:
        model.learn(total_timesteps=config.total_timesteps, callback=callback)
    else:
        model.learn(total_timesteps=config.total_timesteps)
        metric_logger.log_event(phase="train", event="train_complete_without_metric_callback")
        metric_logger.close()
    checkpoint_path = run_dir / "model.zip"
    model.save(str(checkpoint_path))
    payload = {
        "ok": True,
        "dry_run": False,
        "algorithm": config.algorithm,
        "total_timesteps": config.total_timesteps,
        "checkpoint": str(checkpoint_path),
        "run_dir": str(run_dir),
        "metric_log_path": str(metric_logger.jsonl_path),
        "metric_table_path": str(metric_logger.csv_path),
        "metric_summary": metric_logger.summary(),
    }
    train_summary = run_dir / "train_summary.json"
    training_log = run_dir / "training_log.jsonl"
    _write_jsonl(training_log, [{"event": "train_complete", "summary": payload}])
    snapshot = save_command_snapshot(
        run_dir,
        "seedling-rl:train",
        command_args={"config": str(config_path), "dry_run": dry_run},
        input_paths=[config_path, config.env_config],
        output_paths=[config_snapshot, checkpoint_path, train_summary, training_log, metric_logger.jsonl_path, metric_logger.csv_path],
        metadata={"algorithm": config.algorithm, "seed": config.seed, "dry_run": dry_run},
    )
    payload["run_snapshot"] = snapshot
    _write_json(train_summary, payload)
    write_run_registry(
        run_dir,
        "rl-train",
        [
            config_snapshot,
            checkpoint_path,
            train_summary,
            training_log,
            metric_logger.jsonl_path,
            metric_logger.csv_path,
            snapshot,
        ],
    )
    return payload


def evaluate_checkpoint(
    checkpoint: str | Path | None,
    env_config_path: str | Path,
    episodes: int,
    out_dir: str | Path,
    algorithm: str = "ppo",
    baseline: str | None = None,
) -> dict[str, Any]:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    env_config = TrayEnvConfig.from_dict(load_config_file(env_config_path))
    if baseline:
        replay_dir = output / "replays"
        metric_logger = RLMetricsLogger(output / "rl_metrics.jsonl", output / "rl_metrics.csv")
        result, replay_paths, critical_events = evaluate_baseline_with_replays(
            baseline,
            env_config,
            episodes,
            replay_dir,
            metric_logger=metric_logger,
        )
        metric_logger.close()
        critical_path = output / "critical_events.json"
        _write_json(critical_path, {"critical_events": critical_events})
        payload = {
            "ok": True,
            "mode": "baseline",
            "episodes": episodes,
            "results": [result.to_dict()],
            "replay_dir": str(replay_dir),
            "replay_paths": replay_paths,
            "critical_events_path": str(critical_path),
            "critical_events": critical_events,
            "metric_log_path": str(metric_logger.jsonl_path),
            "metric_table_path": str(metric_logger.csv_path),
            "metric_summary": metric_logger.summary(),
        }
        metrics_path = output / "rl_eval_metrics.json"
        snapshot = save_command_snapshot(
            output,
            "seedling-rl:evaluate",
            command_args={
                "config": str(env_config_path),
                "checkpoint": str(checkpoint) if checkpoint else None,
                "algorithm": algorithm,
                "baseline": baseline,
                "episodes": episodes,
                "out": str(out_dir),
            },
            input_paths=[env_config_path],
            output_paths=[metrics_path, critical_path, metric_logger.jsonl_path, metric_logger.csv_path, *replay_paths],
            metadata={"mode": "baseline", "episodes": episodes, "baseline": baseline},
        )
        payload["run_snapshot"] = snapshot
        _write_json(metrics_path, payload)
        write_run_registry(
            output,
            "rl-evaluate:baseline",
            [
                metrics_path,
                critical_path,
                metric_logger.jsonl_path,
                metric_logger.csv_path,
                *replay_paths,
                snapshot,
            ],
        )
        return payload
    if checkpoint is None:
        raise ValueError("checkpoint is required unless baseline is provided")
    model_cls = _algorithm_class(algorithm)
    model = model_cls.load(str(checkpoint))
    recurrent = _is_recurrent_algorithm(algorithm)
    replay_dir = output / "replays"
    metric_logger = RLMetricsLogger(output / "rl_metrics.jsonl", output / "rl_metrics.csv")
    metrics = evaluate_model(
        model,
        env_config,
        episodes,
        replay_dir=replay_dir,
        metric_logger=metric_logger,
        recurrent=recurrent,
    )
    metric_logger.close()
    critical_path = output / "critical_events.json"
    _write_json(critical_path, {"critical_events": metrics.get("critical_events", [])})
    payload = {
        "ok": True,
        "mode": "checkpoint",
        "checkpoint": str(checkpoint),
        "recurrent": recurrent,
        **metrics,
        "metric_log_path": str(metric_logger.jsonl_path),
        "metric_table_path": str(metric_logger.csv_path),
        "metric_summary": metric_logger.summary(),
    }
    metrics_path = output / "rl_eval_metrics.json"
    snapshot = save_command_snapshot(
        output,
        "seedling-rl:evaluate",
        command_args={
            "config": str(env_config_path),
            "checkpoint": str(checkpoint) if checkpoint else None,
            "algorithm": algorithm,
            "baseline": baseline,
            "episodes": episodes,
            "out": str(out_dir),
        },
        input_paths=[env_config_path, checkpoint],
        output_paths=[metrics_path, critical_path, metric_logger.jsonl_path, metric_logger.csv_path, *metrics.get("replay_paths", [])],
        metadata={"mode": "checkpoint", "episodes": episodes, "algorithm": algorithm, "recurrent": recurrent},
    )
    payload["run_snapshot"] = snapshot
    _write_json(metrics_path, payload)
    write_run_registry(
        output,
        "rl-evaluate",
        [
            metrics_path,
            critical_path,
            metric_logger.jsonl_path,
            metric_logger.csv_path,
            checkpoint,
            *metrics.get("replay_paths", []),
            snapshot,
        ],
    )
    return payload


def evaluate_model(
    model: Any,
    env_config: TrayEnvConfig,
    episodes: int,
    replay_dir: str | Path | None = None,
    metric_logger: RLMetricsLogger | None = None,
    recurrent: bool = False,
) -> dict[str, Any]:
    env = SeedlingTrayEnv(env_config)
    rewards: list[float] = []
    critical_errors = 0
    actions = 0
    successes = 0
    target_steps = 0
    reviews = 0
    replay_paths: list[str] = []
    critical_events: list[dict[str, Any]] = []
    for episode in range(episodes):
        obs, reset_info = env.reset(seed=env.config.seed + episode)
        recurrent_state = None
        episode_start = np.ones((1,), dtype=bool)
        logger = ReplayLogger(replay_id=f"checkpoint_episode_{episode:04d}", scene_id=str(reset_info["scene_id"]))
        logger.log.metadata.update({"mode": "checkpoint", "episode": episode})
        done = False
        episode_reward = 0.0
        episode_actions = 0
        while not done and episode_actions < env.config.max_steps:
            if recurrent:
                action, recurrent_state = model.predict(
                    obs,
                    state=recurrent_state,
                    episode_start=episode_start,
                    deterministic=True,
                )
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_reward += float(reward)
            actions += 1
            episode_actions += 1
            _append_rl_replay_step(logger, episode, int(action), reward, terminated, truncated, info)
            if metric_logger is not None:
                metric_logger.log_step(
                    phase="evaluate",
                    episode=episode,
                    step=episode_actions,
                    reward=float(reward),
                    info=info,
                    policy="checkpoint",
                )
            if info.get("event") in {"unsafe_target", "invalid_target_index", "invalid_action"}:
                critical_errors += 1
                critical_events.append(_critical_event(episode, int(action), info))
            if info.get("event") == "review":
                reviews += int(info.get("reviewed_targets", 0))
            if info.get("event") == "target_step":
                target_steps += 1
                outcome = info.get("outcome", {})
                if outcome.get("success"):
                    successes += 1
                if outcome.get("crop_damage"):
                    critical_errors += 1
                    critical_events.append(_critical_event(episode, int(action), info))
            done = terminated or truncated
            episode_start = np.array([done], dtype=bool)
        if not done and episode_actions >= env.config.max_steps:
            critical_errors += 1
            critical_info = {"event": "evaluator_max_steps_reached", "max_steps": env.config.max_steps}
            critical_events.append(_critical_event(episode, -1, critical_info))
            logger.append("rl_truncated", _jsonable(critical_info))
            if metric_logger is not None:
                metric_logger.log_step(
                    phase="evaluate",
                    episode=episode,
                    step=episode_actions,
                    reward=0.0,
                    info=critical_info,
                    policy="checkpoint",
                )
        rewards.append(episode_reward)
        if replay_dir is not None:
            replay_paths.append(_write_episode_replay(logger, replay_dir, "checkpoint", episode))
    return {
        "episodes": episodes,
        "reward_mean": sum(rewards) / len(rewards) if rewards else 0.0,
        "critical_error_rate": critical_errors / max(actions, 1),
        "successful_target_rate": successes / target_steps if target_steps else 0.0,
        "review_rate": reviews / max(actions, 1),
        "actions_per_tray": actions / max(episodes, 1),
        "total_distance_mm": metric_logger.summary()["total_distance_mm"] if metric_logger is not None else None,
        "mean_distance_error_mm": metric_logger.summary()["mean_distance_error_mm"] if metric_logger is not None else None,
        "actions": actions,
        "replay_dir": str(replay_dir) if replay_dir is not None else None,
        "replay_paths": replay_paths,
        "critical_events": critical_events,
    }


def evaluate_baseline_with_replays(
    policy: str,
    config: TrayEnvConfig | dict[str, object] | None = None,
    episodes: int = 10,
    replay_dir: str | Path | None = None,
    metric_logger: RLMetricsLogger | None = None,
) -> tuple[BaselineEvaluationResult, list[str], list[dict[str, Any]]]:
    env = SeedlingTrayEnv(config)
    rewards: list[float] = []
    successes = 0
    targets = 0
    critical = 0
    reviews = 0
    actions = 0
    details: list[dict[str, object]] = []
    replay_paths: list[str] = []
    critical_events: list[dict[str, Any]] = []
    for episode in range(episodes):
        obs, reset_info = env.reset(seed=env.config.seed + episode)
        logger = ReplayLogger(replay_id=f"{policy}_episode_{episode:04d}", scene_id=str(reset_info["scene_id"]))
        logger.log.metadata.update({"mode": "baseline", "policy": policy, "episode": episode})
        done = False
        episode_reward = 0.0
        episode_actions = 0
        while not done and episode_actions < env.config.max_steps:
            action = select_baseline_action(policy, env, obs)
            obs, reward, terminated, truncated, step_info = env.step(action)
            episode_reward += reward
            episode_actions += 1
            _append_rl_replay_step(logger, episode, action, reward, terminated, truncated, step_info)
            if metric_logger is not None:
                metric_logger.log_step(
                    phase="evaluate",
                    episode=episode,
                    step=episode_actions,
                    reward=float(reward),
                    info=step_info,
                    policy=policy,
                )
            event = step_info.get("event")
            if event == "target_step":
                outcome = step_info["outcome"]
                targets += 1
                if outcome.get("success"):
                    successes += 1
                if outcome.get("crop_damage"):
                    critical += 1
                    critical_events.append(_critical_event(episode, action, step_info))
            elif event == "review":
                reviews += int(step_info.get("reviewed_targets", 0))
            elif event in {"unsafe_target", "invalid_target_index", "invalid_action"}:
                critical += 1
                critical_events.append(_critical_event(episode, action, step_info))
            done = terminated or truncated
        if not done and episode_actions >= env.config.max_steps:
            critical += 1
            critical_info = {"event": "evaluator_max_steps_reached", "max_steps": env.config.max_steps}
            critical_events.append(_critical_event(episode, -1, critical_info))
            logger.append("rl_truncated", _jsonable(critical_info))
            if metric_logger is not None:
                metric_logger.log_step(
                    phase="evaluate",
                    episode=episode,
                    step=episode_actions,
                    reward=0.0,
                    info=critical_info,
                    policy=policy,
                )
        rewards.append(episode_reward)
        actions += episode_actions
        replay_path = _write_episode_replay(logger, replay_dir, policy, episode) if replay_dir is not None else None
        if replay_path:
            replay_paths.append(replay_path)
        details.append(
            {
                "episode": episode,
                "reward": episode_reward,
                "actions": episode_actions,
                "replay_path": replay_path,
            }
        )
    return (
        BaselineEvaluationResult(
            policy=policy,
            episodes=episodes,
            reward_mean=sum(rewards) / len(rewards) if rewards else 0.0,
            successful_target_rate=successes / targets if targets else 0.0,
            critical_error_rate=critical / max(actions, 1),
            review_rate=reviews / max(actions, 1),
            actions_per_tray=actions / max(episodes, 1),
            total_distance_mm=metric_logger.summary()["total_distance_mm"] if metric_logger is not None else None,
            mean_distance_error_mm=metric_logger.summary()["mean_distance_error_mm"] if metric_logger is not None else None,
            details=details,
        ),
        replay_paths,
        critical_events,
    )


def _algorithm_class(algorithm: str) -> Any:
    normalized = _normalized_algorithm(algorithm)
    if normalized in {"maskable_ppo", "maskableppo"}:
        try:
            from sb3_contrib import MaskablePPO
        except ImportError as exc:
            raise RuntimeError("sb3-contrib is required for MaskablePPO training") from exc
        return MaskablePPO
    if normalized in {"recurrent_ppo", "recurrentppo"}:
        try:
            from sb3_contrib import RecurrentPPO
        except ImportError as exc:
            raise RuntimeError("sb3-contrib is required for RecurrentPPO training") from exc
        return RecurrentPPO
    if normalized == "ppo":
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise RuntimeError("stable-baselines3 is required for PPO training") from exc
        return PPO
    raise ValueError(f"Unsupported RL algorithm: {algorithm}")


def _is_recurrent_algorithm(algorithm: str) -> bool:
    return _normalized_algorithm(algorithm) in {"recurrent_ppo", "recurrentppo"}


def _normalized_algorithm(algorithm: str) -> str:
    return algorithm.lower().replace("-", "_")


def _bool_config(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Invalid boolean config value: {value!r}")


def _training_env(config: RLTrainingConfig, env_config: TrayEnvConfig) -> Any:
    if config.n_envs > 1:
        return make_vectorized_env(env_config, n_envs=config.n_envs)
    return SeedlingTrayEnv(env_config)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _append_rl_replay_step(
    logger: ReplayLogger,
    episode: int,
    action: int,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
) -> None:
    logger.append(
        "rl_step",
        {
            "episode": episode,
            "action": int(action),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "event": info.get("event"),
            "info": _jsonable(info),
        },
    )


def _write_episode_replay(logger: ReplayLogger, replay_dir: str | Path, policy: str, episode: int) -> str:
    output_dir = Path(replay_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_path = output_dir / f"{policy}_episode_{episode:04d}.json"
    logger.to_json(replay_path)
    return str(replay_path)


def _critical_event(episode: int, action: int, info: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode": int(episode),
        "action": int(action),
        "event": info.get("event"),
        "info": _jsonable(info),
    }


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
