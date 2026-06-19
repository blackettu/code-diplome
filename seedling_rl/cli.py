from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from seedling_core.config import load_config_file
from seedling_core.run_snapshot import save_command_snapshot
from seedling_rl.baseline_eval import evaluate_baseline_suite
from seedling_rl.curriculum import write_curriculum_plan
from seedling_rl.envs import SeedlingTrayEnv, TrayEnvConfig
from seedling_rl.offline_replay import evaluate_replay_logs
from seedling_rl.sweep import build_sweep_plan, write_sweep_stability_report
from seedling_rl.training import evaluate_checkpoint, train_from_config
from seedling_rl.vectorized_env import vectorized_env_spec


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m seedling_rl")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_env = subparsers.add_parser("check-env", help="Reset and step SeedlingTrayEnv once.")
    check_env.add_argument("--config", default="configs/simulation/tray_env_v0.yaml")

    evaluate = subparsers.add_parser("evaluate-baselines", help="Evaluate baseline policies in the simulator.")
    evaluate.add_argument("--config", default="configs/simulation/tray_env_v0.yaml")
    evaluate.add_argument("--episodes", type=int, default=10)
    evaluate.add_argument("--out", default=None)

    train = subparsers.add_parser("train", help="Train PPO/MaskablePPO/RecurrentPPO from YAML config.")
    train.add_argument("--config", default="configs/rl/maskable_ppo_v0.yaml")
    train.add_argument("--dry-run", action="store_true", help="Validate config and write run metadata without SB3.")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained checkpoint or baseline policy.")
    eval_parser.add_argument("--config", default="configs/simulation/tray_env_v0.yaml")
    eval_parser.add_argument("--checkpoint", default=None)
    eval_parser.add_argument("--algorithm", default="ppo")
    eval_parser.add_argument("--baseline", default=None)
    eval_parser.add_argument("--episodes", type=int, default=10)
    eval_parser.add_argument("--out", default="runs/rl/eval")

    vector = subparsers.add_parser("vector-env", help="Build vectorized env metadata without importing SB3.")
    vector.add_argument("--config", default="configs/simulation/tray_env_v0.yaml")
    vector.add_argument("--n-envs", type=int, default=1)

    curriculum = subparsers.add_parser("curriculum", help="Write default curriculum plan.")
    curriculum.add_argument("--config", default="configs/simulation/tray_env_v0.yaml")
    curriculum.add_argument("--out", required=True)
    curriculum.add_argument("--episodes-per-stage", type=int, default=100)

    replay_eval = subparsers.add_parser("offline-replay-eval", help="Evaluate replay logs without stepping env.")
    replay_eval.add_argument("--replays", nargs="+", required=True)
    replay_eval.add_argument("--out", required=True)

    sweep = subparsers.add_parser("sweep", help="Build hyperparameter sweep plan.")
    sweep.add_argument("--config", required=True)
    sweep.add_argument("--out", required=True)

    sweep_report = subparsers.add_parser("sweep-report", help="Summarize sweep stability across seeds/configs.")
    sweep_report.add_argument("--plan", required=True)
    sweep_report.add_argument("--metrics", nargs="+", required=True)
    sweep_report.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    if args.command == "check-env":
        config = TrayEnvConfig.from_dict(load_config_file(args.config))
        env = SeedlingTrayEnv(config)
        observation, info = env.reset(seed=config.seed)
        first_valid = next(index for index, value in enumerate(observation["action_mask"]) if int(value) == 1)
        _, reward, terminated, truncated, step_info = env.step(first_valid)
        payload = {
            "ok": True,
            "scene_id": info["scene_id"],
            "first_action": first_valid,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "event": step_info.get("event"),
        }
    elif args.command == "evaluate-baselines":
        config = TrayEnvConfig.from_dict(load_config_file(args.config))
        results = [item.to_dict() for item in evaluate_baseline_suite(config, episodes=args.episodes)]
        payload = {"ok": True, "episodes": args.episodes, "results": results}
        if args.out:
            output = Path(args.out)
            output.parent.mkdir(parents=True, exist_ok=True)
            inputs = [args.config]
            outputs = [output]
            metadata = {"episodes": args.episodes}
            snapshot = _write_rl_snapshot(
                output.parent,
                "seedling-rl:evaluate-baselines",
                args,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
            )
            payload["run_snapshot"] = snapshot
            output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            payload["artifact_registry"] = _write_rl_registry(
                output.parent,
                "seedling-rl:evaluate-baselines",
                inputs=inputs,
                outputs=[*outputs, *([snapshot] if snapshot else [])],
                metadata=metadata,
            )
    elif args.command == "train":
        payload = train_from_config(args.config, dry_run=args.dry_run)
        payload["run_snapshot"] = _existing_snapshot(payload.get("run_dir"))
        payload["artifact_registry"] = _existing_registry(payload.get("run_dir"))
    elif args.command == "evaluate":
        payload = evaluate_checkpoint(
            checkpoint=args.checkpoint,
            env_config_path=args.config,
            episodes=args.episodes,
            out_dir=args.out,
            algorithm=args.algorithm,
            baseline=args.baseline,
        )
        payload["run_snapshot"] = _existing_snapshot(args.out)
        payload["artifact_registry"] = _existing_registry(args.out)
    elif args.command == "vector-env":
        config = TrayEnvConfig.from_dict(load_config_file(args.config))
        payload = {"ok": True, "vectorized_env": vectorized_env_spec(config, n_envs=args.n_envs).to_dict()}
    elif args.command == "curriculum":
        config = TrayEnvConfig.from_dict(load_config_file(args.config))
        stages = write_curriculum_plan(config, args.out, episodes_per_stage=args.episodes_per_stage)
        inputs = [args.config]
        outputs = [args.out]
        metadata = {"episodes_per_stage": args.episodes_per_stage, "stages": len(stages)}
        snapshot = _write_rl_snapshot(
            Path(args.out).parent,
            "seedling-rl:curriculum",
            args,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
        payload = {
            "ok": True,
            "out": args.out,
            "stages": len(stages),
            "run_snapshot": snapshot,
            "artifact_registry": _write_rl_registry(
                Path(args.out).parent,
                "seedling-rl:curriculum",
                inputs=inputs,
                outputs=[*outputs, *([snapshot] if snapshot else [])],
                metadata=metadata,
            ),
        }
    elif args.command == "offline-replay-eval":
        payload = evaluate_replay_logs(args.replays, output_path=args.out)
        inputs = args.replays
        outputs = [args.out]
        metadata = {"replays": len(args.replays)}
        snapshot = _write_rl_snapshot(
            Path(args.out).parent,
            "seedling-rl:offline-replay-eval",
            args,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
        payload["run_snapshot"] = snapshot
        payload["artifact_registry"] = _write_rl_registry(
            Path(args.out).parent,
            "seedling-rl:offline-replay-eval",
            inputs=inputs,
            outputs=[*outputs, *([snapshot] if snapshot else [])],
            metadata=metadata,
        )
    elif args.command == "sweep":
        payload = build_sweep_plan(args.config, output_path=args.out)
        inputs = [args.config]
        outputs = [args.out]
        metadata = {"runs": payload.get("count")}
        snapshot = _write_rl_snapshot(
            Path(args.out).parent,
            "seedling-rl:sweep",
            args,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
        payload["run_snapshot"] = snapshot
        payload["artifact_registry"] = _write_rl_registry(
            Path(args.out).parent,
            "seedling-rl:sweep",
            inputs=inputs,
            outputs=[*outputs, *([snapshot] if snapshot else [])],
            metadata=metadata,
        )
    elif args.command == "sweep-report":
        payload = write_sweep_stability_report(args.plan, args.metrics, args.out)
        inputs = [args.plan, *args.metrics]
        outputs = [payload["summary_path"], payload["csv_path"]]
        metadata = {"metrics": len(args.metrics), "groups": payload.get("group_count")}
        snapshot = _write_rl_snapshot(
            args.out,
            "seedling-rl:sweep-report",
            args,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
        payload["run_snapshot"] = snapshot
        payload["artifact_registry"] = _write_rl_registry(
            args.out,
            "seedling-rl:sweep-report",
            inputs=inputs,
            outputs=[*outputs, *([snapshot] if snapshot else [])],
            metadata=metadata,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _write_rl_registry(
    run_dir: str | Path,
    command: str,
    *,
    inputs: list[str | Path],
    outputs: list[str | Path],
    metadata: dict[str, Any] | None = None,
) -> str | None:
    try:
        from seedling_reports.registry import ArtifactRecord, write_run_registry_records

        artifacts = [
            *(ArtifactRecord.from_path(path, artifact_type=_artifact_type(path), role="input", command=command) for path in inputs),
            *(ArtifactRecord.from_path(path, artifact_type=_artifact_type(path), role="output", command=command) for path in outputs),
        ]
        write_run_registry_records(
            run_dir,
            command,
            artifacts,
            metadata=metadata,
            run_id=_registry_run_id(run_dir, command, outputs),
        )
        return str(Path(run_dir) / "artifact_registry.json")
    except Exception:
        return None


def _write_rl_snapshot(
    run_dir: str | Path,
    command: str,
    args: argparse.Namespace,
    *,
    inputs: list[str | Path],
    outputs: list[str | Path],
    metadata: dict[str, Any] | None = None,
) -> str | None:
    try:
        return save_command_snapshot(
            run_dir,
            command,
            command_args=_namespace_payload(args),
            input_paths=inputs,
            output_paths=outputs,
            metadata=metadata,
            snapshot_name=_snapshot_name(command, outputs),
        )
    except Exception:
        return None


def _snapshot_name(command: str, outputs: list[str | Path]) -> str:
    stem = Path(outputs[0]).stem if outputs else "".join(char if char.isalnum() else "_" for char in command).strip("_")
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in stem).strip("_")
    return f"{safe or 'command'}.run_snapshot.json"


def _namespace_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {key: _jsonable(value) for key, value in vars(args).items()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _existing_registry(run_dir: object) -> str | None:
    if not run_dir:
        return None
    path = Path(str(run_dir)) / "artifact_registry.json"
    return str(path) if path.exists() else None


def _existing_snapshot(run_dir: object) -> str | None:
    if not run_dir:
        return None
    path = Path(str(run_dir)) / "run_snapshot.json"
    return str(path) if path.exists() else None


def _artifact_type(path: str | Path) -> str:
    candidate = Path(path)
    name = candidate.name
    if candidate.is_dir():
        return "directory"
    if name == "run_snapshot.json" or name.endswith(".run_snapshot.json"):
        return "run_snapshot"
    if name.endswith(".json"):
        return name.removesuffix(".json")
    if name.endswith(".jsonl"):
        return name.removesuffix(".jsonl")
    if name.endswith(".csv"):
        return name.removesuffix(".csv")
    if name.endswith(".yaml") or name.endswith(".yml"):
        return "config"
    return candidate.suffix.lstrip(".") or "artifact"


def _registry_run_id(run_dir: str | Path, command: str, outputs: list[str | Path]) -> str:
    directory = Path(run_dir).name or "run"
    output_stem = Path(outputs[0]).stem if outputs else "stdout"
    command_slug = "".join(char if char.isalnum() else "_" for char in command).strip("_")
    return f"{directory}_{command_slug}_{output_stem}"
