from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from seedling_core.run_snapshot import save_command_snapshot
from seedling_core.config import load_config_file

from .domain_randomization import DOMAIN_RANDOMIZATION_PRESETS, domain_randomization_preset
from .image_backed import sim_scene_from_scene_state_file
from .logical_tray import LogicalTraySimulator
from .policy_runner import compare_policies_on_scenes, load_sim_scenes, run_policy_on_scene, write_policy_comparison
from .renderers import render_scene_html, render_scene_png, render_scene_svg
from .scene_generator import SceneGeneratorConfig, SimSceneGenerator
from .schemas import SimScene


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m seedling_sim")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-scene", help="Validate a SimScene JSON file.")
    validate.add_argument("--scene", required=True)

    generate = subparsers.add_parser("generate-scenes", help="Generate logical SimScene JSON files.")
    generate.add_argument("--config", default="configs/simulation/tray_env_v0.yaml")
    generate.add_argument("--count", type=int, default=1)
    generate.add_argument("--out", required=True)
    generate.add_argument("--prefix", default="scene")

    render = subparsers.add_parser("render-scene", help="Render a SimScene as SVG or HTML.")
    render.add_argument("--scene", required=True)
    render.add_argument("--out", required=True)
    render.add_argument("--format", choices=["svg", "html"], default=None)

    image_backed = subparsers.add_parser("image-backed-scene", help="Convert SceneState JSON to image-backed SimScene.")
    image_backed.add_argument("--scene-state", required=True)
    image_backed.add_argument("--out", required=True)
    image_backed.add_argument("--cell-size-mm", default=None)

    synthetic = subparsers.add_parser("render-synthetic", help="Render a SimScene to synthetic PNG.")
    synthetic.add_argument("--scene", required=True)
    synthetic.add_argument("--out", required=True)
    synthetic.add_argument("--width-px", type=int, default=720)
    synthetic.add_argument("--preset", default="greenhouse_default")
    synthetic.add_argument("--seed", type=int, default=42)

    presets = subparsers.add_parser("randomization-presets", help="List domain randomization presets.")

    step = subparsers.add_parser("step", help="Run one logical simulator step for a target.")
    step.add_argument("--scene", required=True)
    step.add_argument("--target-id", required=True)
    step.add_argument("--seed", type=int, default=42)

    run_policy = subparsers.add_parser("run-policy", help="Run a deterministic policy on one SimScene and write replay JSON.")
    run_policy.add_argument("--scene", required=True)
    run_policy.add_argument("--policy", required=True)
    run_policy.add_argument("--out", required=True)
    run_policy.add_argument("--max-steps", type=int, default=None)
    run_policy.add_argument("--seed", type=int, default=42)

    compare = subparsers.add_parser("compare-policies", help="Compare deterministic policies on one scene or a scene directory.")
    compare.add_argument("--scenes", required=False)
    compare.add_argument("--config", default=None)
    compare.add_argument("--policies", nargs="+", default=None)
    compare.add_argument("--out", required=True)
    compare.add_argument("--max-steps", type=int, default=None)
    compare.add_argument("--seed", type=int, default=42)

    play = subparsers.add_parser("play-policy", help="Run a policy replay and render/print it for inspection.")
    play.add_argument("--scene", required=True)
    play.add_argument("--policy", required=True)
    play.add_argument("--out", default=None)
    play.add_argument("--render", choices=["human", "html", "json"], default="human")
    play.add_argument("--env-config", default=None)
    play.add_argument("--max-steps", type=int, default=None)
    play.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)
    if args.command == "validate-scene":
        scene = SimScene.from_json(args.scene)
        print(json.dumps({"ok": True, "scene_id": scene.scene_id}, ensure_ascii=False, indent=2))
    elif args.command == "generate-scenes":
        raw_config = load_config_file(args.config)
        scene_config = raw_config.get("scene_generator", {})
        if not isinstance(scene_config, dict):
            scene_config = {}
        config = SceneGeneratorConfig.from_dict(
            {
                "grid_rows": raw_config.get("grid_rows", 11),
                "grid_cols": raw_config.get("grid_cols", 11),
                "cell_size_mm": raw_config.get("cell_size_mm", [33.0, 33.0]),
                **scene_config,
            }
        )
        generator = SimSceneGenerator(config)
        output_dir = Path(args.out)
        output_dir.mkdir(parents=True, exist_ok=True)
        scenes = generator.generate_many(args.count, prefix=args.prefix)
        output_paths = []
        for scene in scenes:
            output_path = output_dir / f"{scene.scene_id}.json"
            scene.to_json(output_path)
            output_paths.append(output_path)
        inputs = [args.config]
        outputs = output_paths
        metadata = {"count": len(scenes), "prefix": args.prefix}
        snapshot = _write_sim_snapshot(
            output_dir,
            "seedling-sim:generate-scenes",
            args,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
        payload = {
            "ok": True,
            "count": len(scenes),
            "out": str(output_dir),
            "run_snapshot": snapshot,
            "artifact_registry": _write_sim_registry(
                output_dir,
                "seedling-sim:generate-scenes",
                inputs=inputs,
                outputs=[*outputs, *([snapshot] if snapshot else [])],
                metadata=metadata,
            ),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.command == "render-scene":
        scene = SimScene.from_json(args.scene)
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output_format = args.format or output.suffix.lstrip(".").lower() or "html"
        if output_format == "svg":
            output.write_text(render_scene_svg(scene), encoding="utf-8")
        else:
            output.write_text(render_scene_html(scene), encoding="utf-8")
        inputs = [args.scene]
        outputs = [output]
        metadata = {"format": output_format, "scene_id": scene.scene_id}
        snapshot = _write_sim_snapshot(
            output.parent,
            "seedling-sim:render-scene",
            args,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
        payload = {
            "ok": True,
            "out": str(output),
            "run_snapshot": snapshot,
            "artifact_registry": _write_sim_registry(
                output.parent,
                "seedling-sim:render-scene",
                inputs=inputs,
                outputs=[*outputs, *([snapshot] if snapshot else [])],
                metadata=metadata,
            ),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.command == "image-backed-scene":
        scene = sim_scene_from_scene_state_file(
            args.scene_state,
            output_path=args.out,
            cell_size_mm=_csv_floats(args.cell_size_mm, 2) if args.cell_size_mm else None,
        )
        inputs = [args.scene_state]
        outputs = [args.out]
        metadata = {"scene_id": scene.scene_id, "cell_size_mm": _csv_floats(args.cell_size_mm, 2) if args.cell_size_mm else None}
        snapshot = _write_sim_snapshot(
            Path(args.out).parent,
            "seedling-sim:image-backed-scene",
            args,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
        payload = {
            "ok": True,
            "out": args.out,
            "scene_id": scene.scene_id,
            "run_snapshot": snapshot,
            "artifact_registry": _write_sim_registry(
                Path(args.out).parent,
                "seedling-sim:image-backed-scene",
                inputs=inputs,
                outputs=[*outputs, *([snapshot] if snapshot else [])],
                metadata=metadata,
            ),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.command == "render-synthetic":
        scene = SimScene.from_json(args.scene)
        render_scene_png(
            scene,
            args.out,
            width_px=args.width_px,
            randomization=domain_randomization_preset(args.preset),
            seed=args.seed,
        )
        inputs = [args.scene]
        outputs = [args.out]
        metadata = {"preset": args.preset, "seed": args.seed, "width_px": args.width_px, "scene_id": scene.scene_id}
        snapshot = _write_sim_snapshot(
            Path(args.out).parent,
            "seedling-sim:render-synthetic",
            args,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
        payload = {
            "ok": True,
            "out": args.out,
            "preset": args.preset,
            "run_snapshot": snapshot,
            "artifact_registry": _write_sim_registry(
                Path(args.out).parent,
                "seedling-sim:render-synthetic",
                inputs=inputs,
                outputs=[*outputs, *([snapshot] if snapshot else [])],
                metadata=metadata,
            ),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.command == "randomization-presets":
        payload = {
            "ok": True,
            "presets": {name: preset.to_dict() for name, preset in DOMAIN_RANDOMIZATION_PRESETS.items()},
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "step":
        scene = SimScene.from_json(args.scene)
        simulator = LogicalTraySimulator(seed=args.seed)
        simulator.reset(scene)
        target = next(item for item in scene.targets if item.target_id == args.target_id)
        outcome = simulator.step_target(target).to_dict()
        print(json.dumps(outcome, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "run-policy":
        scene = SimScene.from_json(args.scene)
        summary, _, _ = run_policy_on_scene(
            scene,
            args.policy,
            max_steps=args.max_steps,
            seed=args.seed,
            replay_path=args.out,
        )
        inputs = [args.scene]
        outputs = [args.out]
        metadata = {"policy": args.policy, "max_steps": args.max_steps, "seed": args.seed}
        snapshot = _write_sim_snapshot(
            Path(args.out).parent,
            "seedling-sim:run-policy",
            args,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
        payload = {
            "ok": True,
            "out": args.out,
            **summary,
            "run_snapshot": snapshot,
            "artifact_registry": _write_sim_registry(
                Path(args.out).parent,
                "seedling-sim:run-policy",
                inputs=inputs,
                outputs=[*outputs, *([snapshot] if snapshot else [])],
                metadata=metadata,
            ),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "compare-policies":
        scene_path, policies = _compare_inputs(args)
        scenes = load_sim_scenes(scene_path)
        rows = compare_policies_on_scenes(
            scenes,
            policies,
            max_steps=args.max_steps,
            seed=args.seed,
        )
        write_policy_comparison(rows, args.out)
        output = Path(args.out)
        comparison_outputs = [output, *([output.with_suffix(".json")] if output.suffix.lower() == ".html" else [])]
        inputs = [scene_path]
        metadata = {"policies": policies, "max_steps": args.max_steps, "seed": args.seed}
        snapshot = _write_sim_snapshot(
            output.parent,
            "seedling-sim:compare-policies",
            args,
            inputs=inputs,
            outputs=comparison_outputs,
            metadata=metadata,
        )
        payload = {
            "ok": True,
            "out": args.out,
            "scenes": len(scenes),
            "rows": len(rows),
            "run_snapshot": snapshot,
            "artifact_registry": _write_sim_registry(
                output.parent,
                "seedling-sim:compare-policies",
                inputs=inputs,
                outputs=[*comparison_outputs, *([snapshot] if snapshot else [])],
                metadata=metadata,
            ),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "play-policy":
        scene = SimScene.from_json(args.scene)
        summary, replay, final_scene = run_policy_on_scene(
            scene,
            args.policy,
            max_steps=args.max_steps,
            seed=args.seed,
            replay_path=args.out if args.render == "json" and args.out else None,
        )
        if args.render == "html":
            if not args.out:
                raise ValueError("--out is required for --render html")
            output = Path(args.out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(render_scene_html(final_scene, replay), encoding="utf-8")
        elif args.render == "human":
            print(render_scene_html(final_scene, replay) if args.out and str(args.out).endswith(".html") else _ansi_replay(summary, replay))
        if args.render != "human":
            inputs = [args.scene]
            outputs = [args.out] if args.out else []
            metadata = {"policy": args.policy, "render": args.render, "env_config": args.env_config, "seed": args.seed}
            snapshot = (
                _write_sim_snapshot(
                    Path(args.out).parent,
                    "seedling-sim:play-policy",
                    args,
                    inputs=inputs,
                    outputs=outputs,
                    metadata=metadata,
                )
                if args.out
                else None
            )
            payload = {
                "ok": True,
                "out": args.out,
                **summary,
                "run_snapshot": snapshot,
                "artifact_registry": _write_sim_registry(
                    Path(args.out).parent,
                    "seedling-sim:play-policy",
                    inputs=inputs,
                    outputs=[*outputs, *([snapshot] if snapshot else [])],
                    metadata=metadata,
                ) if args.out else None,
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _csv_floats(value: str, expected_length: int) -> list[float]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != expected_length:
        raise ValueError(f"Expected {expected_length} comma-separated floats")
    return [float(part) for part in parts]


def _compare_inputs(args: argparse.Namespace) -> tuple[str, list[str]]:
    scene_path = args.scenes
    policies = args.policies
    if args.config:
        raw_config = load_config_file(args.config)
        scene_path = scene_path or raw_config.get("scenes") or raw_config.get("scene")
        policies = policies or raw_config.get("policies")
    if not scene_path:
        raise ValueError("--scenes or --config with scenes/scene is required")
    if not policies:
        policies = ["raster_scan", "route_planning", "risk_aware_rule"]
    return str(scene_path), [str(policy) for policy in policies]


def _ansi_replay(summary: dict[str, object], replay) -> str:
    lines = [
        f"scene={summary['scene_id']} policy={summary['policy']} steps={summary['steps']} reward={summary['reward_total']}",
    ]
    for step in replay.steps:
        lines.append(f"{step.step_index:03d} {step.event_type} {step.payload}")
    return "\n".join(lines)


def _write_sim_registry(
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


def _write_sim_snapshot(
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


def _artifact_type(path: str | Path) -> str:
    candidate = Path(path)
    name = candidate.name
    suffix = candidate.suffix.lower()
    if candidate.is_dir():
        return "directory"
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}:
        return "image"
    if name == "run_snapshot.json" or name.endswith(".run_snapshot.json"):
        return "run_snapshot"
    if name.endswith(".json"):
        return name.removesuffix(".json")
    if name.endswith(".jsonl"):
        return name.removesuffix(".jsonl")
    if name.endswith(".csv"):
        return name.removesuffix(".csv")
    if name.endswith(".html"):
        return "html_report"
    if name.endswith(".svg"):
        return "svg_report"
    if name.endswith(".yaml") or name.endswith(".yml"):
        return "config"
    return candidate.suffix.lstrip(".") or "artifact"


def _registry_run_id(run_dir: str | Path, command: str, outputs: list[str | Path]) -> str:
    directory = Path(run_dir).name or "run"
    output_stem = Path(outputs[0]).stem if outputs else "stdout"
    command_slug = "".join(char if char.isalnum() else "_" for char in command).strip("_")
    return f"{directory}_{command_slug}_{output_stem}"
