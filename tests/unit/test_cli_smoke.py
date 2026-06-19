from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from seedling_core.config import ConfigError
from seedling_core.run_snapshot import config_hash as core_config_hash
from seedling_core.run_snapshot import register_run_artifacts
from seedling_core.run_snapshot import save_command_snapshot
from seedling_core.run_snapshot import save_run_snapshot as core_save_run_snapshot
from seedling_experiments.config import config_hash, save_run_snapshot, validate_experiment_config


ROOT = Path(__file__).resolve().parents[2]


class CliSmokeTests(unittest.TestCase):
    def test_run_snapshot_records_config_hash_and_command_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            config = {"dataset": {"raw_root": "raw", "prepared_root": "prepared"}}
            source_config = Path(tmp) / "config.json"
            source_config.write_text(json.dumps(config), encoding="utf-8")
            command_args = {"command": "prepare", "config": str(source_config)}

            save_run_snapshot(output, config, "prepare", command_args=command_args)

            snapshot = json.loads((output / "run_snapshot.json").read_text(encoding="utf-8"))
            self.assertEqual(snapshot["config_hash"], config_hash(config))
            self.assertEqual(snapshot["command_args"], command_args)
            self.assertIn("python", snapshot["environment"])
            self.assertIn("platform", snapshot["environment"])
            registry = json.loads((output / "artifact_registry.json").read_text(encoding="utf-8"))
            run = registry["runs"][0]
            self.assertEqual(run["metadata"]["config_hash"], config_hash(config))
            self.assertEqual(run["metadata"]["command_args"], command_args)
            self.assertEqual(
                [(artifact["artifact_type"], artifact["role"]) for artifact in run["artifacts"]],
                [("config", "input"), ("config", "output"), ("run_snapshot", "output")],
            )
            self.assertEqual(run["artifacts"][0]["path"], str(source_config))
            self.assertTrue(run["artifacts"][0]["sha256"])

    def test_core_run_snapshot_service_matches_legacy_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            config = {"dataset": {"raw_root": "raw", "prepared_root": "prepared"}}
            source_config = Path(tmp) / "config.json"
            source_config.write_text(json.dumps(config), encoding="utf-8")

            core_save_run_snapshot(output, config, "seedling-core:test", command_args={"config": str(source_config)})

            snapshot = json.loads((output / "run_snapshot.json").read_text(encoding="utf-8"))
            self.assertEqual(snapshot["config_hash"], core_config_hash(config))
            self.assertEqual(snapshot["config_hash"], config_hash(config))
            self.assertIn("environment", snapshot)
            self.assertTrue((output / "artifact_registry.json").exists())

    def test_core_command_snapshot_records_inputs_outputs_and_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report"
            source = Path(tmp) / "runs"
            source.mkdir()
            report = output / "report_summary.json"

            snapshot_path = save_command_snapshot(
                output,
                "seedling-reports:build",
                command_args={"command": "build", "root": str(source), "out": str(output)},
                input_paths=[source],
                output_paths=[report],
                metadata={"rows": 1},
            )

            snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
            self.assertEqual(snapshot["schema_version"], "command_run_snapshot_v0_1")
            self.assertEqual(snapshot["command"], "seedling-reports:build")
            self.assertEqual(snapshot["command_args"]["root"], str(source))
            self.assertEqual(snapshot["inputs"], [str(source)])
            self.assertEqual(snapshot["outputs"], [str(report)])
            self.assertEqual(snapshot["metadata"], {"rows": 1})
            self.assertEqual(snapshot["config_hash"], core_config_hash(snapshot["config"]))

    def test_core_register_run_artifacts_replaces_snapshot_record_with_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "run"
            input_path = root / "images"
            input_path.mkdir()
            output_file = output / "predictions.json"
            config = {"prediction": {"images": str(input_path), "output_dir": str(output)}}

            core_save_run_snapshot(output, config, "predict", command_args={"config": str(root / "config.json")})
            output_file.write_text('{"images":[]}', encoding="utf-8")
            register_run_artifacts(
                output,
                "predict",
                config=config,
                command_args={"config": str(root / "config.json")},
                input_paths=[input_path],
                output_paths=[output_file],
            )

            registry = json.loads((output / "artifact_registry.json").read_text(encoding="utf-8"))
            self.assertEqual(len(registry["runs"]), 1)
            run = registry["runs"][0]
            self.assertEqual(run["metadata"]["config_hash"], core_config_hash(config))
            artifacts = {(artifact["artifact_type"], artifact["role"], Path(artifact["path"]).name) for artifact in run["artifacts"]}
            self.assertIn(("directory", "input", "images"), artifacts)
            self.assertIn(("config", "output", "config.yaml"), artifacts)
            self.assertIn(("run_snapshot", "output", "run_snapshot.json"), artifacts)
            self.assertIn(("predictions", "output", "predictions.json"), artifacts)

    def test_architecture_cli_modules_show_help(self) -> None:
        modules = [
            "seedling_calibration",
            "seedling_data",
            "seedling_experiments",
            "seedling_reports",
            "seedling_rl",
            "seedling_robot",
            "seedling_sim",
            "seedling_ui",
            "seedling_vision",
        ]
        for module in modules:
            with self.subTest(module=module):
                completed = subprocess.run(
                    [sys.executable, "-B", "-m", module, "--help"],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                self.assertIn("usage:", completed.stdout)

    def test_requirement_splits_match_pyproject_dependency_groups(self) -> None:
        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 compatibility
            self.skipTest("tomllib is required for pyproject dependency checks")

        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        project = pyproject["project"]
        dependencies = _dependency_names(project["dependencies"])
        optional = {
            name: _dependency_names(values)
            for name, values in project["optional-dependencies"].items()
        }

        self.assertEqual(_requirement_names(ROOT / "requirements/base.txt"), dependencies)
        self.assertEqual(_requirement_names(ROOT / "requirements/vision.txt"), dependencies | optional["vision"])
        self.assertEqual(
            _requirement_names(ROOT / "requirements/training.txt"),
            dependencies | optional["vision"] | optional["training"],
        )
        self.assertEqual(_requirement_names(ROOT / "requirements/rl.txt"), dependencies | optional["rl"])
        self.assertEqual(_requirement_names(ROOT / "requirements/robot.txt"), dependencies | optional["robot"])
        self.assertEqual(_requirement_names(ROOT / "requirements/dev.txt"), dependencies | optional["dev"])

    def test_unified_experiment_sim_smoke_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "sim_smoke.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_experiments",
                    "sim",
                    "--config",
                    str(ROOT / "configs/simulation/tray_env_v0.yaml"),
                    "--out",
                    str(output),
                    "--seed",
                    "7",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(completed.stdout)
            written = json.loads(output.read_text(encoding="utf-8"))
            registry = json.loads((output.parent / "artifact_registry.json").read_text(encoding="utf-8"))
            self.assertTrue(payload["ok"])
            self.assertEqual(written["command"], "sim")
            self.assertIn(written["event"], {"target_step", "review", "stop"})
            self.assertEqual(registry["runs"][0]["command"], "seedling-experiments:sim")
            self.assertEqual(
                [artifact["role"] for artifact in registry["runs"][0]["artifacts"]],
                ["input", "output"],
            )

    def test_prepare_cli_snapshot_records_command_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            prepared = root / "prepared"
            config_path = root / "config.json"
            (raw / "images").mkdir(parents=True)
            (raw / "labels").mkdir(parents=True)
            Image.new("RGB", (8, 8), "green").save(raw / "images" / "tray001.png")
            (raw / "labels" / "tray001.txt").write_text("", encoding="utf-8")
            config = {
                "dataset": {
                    "raw_root": str(raw),
                    "prepared_root": str(prepared),
                    "split": {"train": 1.0, "val": 0.0, "test": 0.0, "seed": 1},
                    "class_names": ["container", "seedlings"],
                }
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_experiments",
                    "prepare",
                    "--config",
                    str(config_path),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

            snapshot = json.loads((prepared / "run_snapshot.json").read_text(encoding="utf-8"))
            summary = json.loads((prepared / "prepare_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(snapshot["config_hash"], config_hash(config))
            self.assertEqual(snapshot["command_args"]["command"], "prepare")
            self.assertEqual(snapshot["command_args"]["config"], str(config_path))
            self.assertTrue((prepared / "raw_dataset_audit.json").exists())
            self.assertTrue((prepared / "train_augmentation_manifest.csv").exists())
            self.assertIn("raw_audit", summary)
            self.assertEqual(summary["augmentation"]["created_images"], 0)
            registry = json.loads((prepared / "artifact_registry.json").read_text(encoding="utf-8"))
            run = registry["runs"][0]
            artifacts = {(artifact["artifact_type"], artifact["role"], Path(artifact["path"]).name) for artifact in run["artifacts"]}
            self.assertEqual(run["metadata"]["config_hash"], config_hash(config))
            self.assertIn(("config", "input", "config.json"), artifacts)
            self.assertIn(("directory", "input", "raw"), artifacts)
            self.assertIn(("run_snapshot", "output", "run_snapshot.json"), artifacts)
            self.assertIn(("raw_dataset_audit", "output", "raw_dataset_audit.json"), artifacts)
            self.assertIn(("train_augmentation_manifest", "output", "train_augmentation_manifest.csv"), artifacts)
            self.assertIn(("prepare_summary", "output", "prepare_summary.json"), artifacts)

    def test_experiment_config_validation_checks_existing_input_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            config = {
                "prediction": {
                    "model": "missing_model.pt",
                    "images": "images",
                }
            }

            with self.assertRaisesRegex(ConfigError, "prediction.model.*path does not exist"):
                validate_experiment_config(config, "predict", base_dir=root, check_paths=True)

            model = root / "model.pt"
            model.write_text("placeholder", encoding="utf-8")
            config["prediction"]["model"] = "model.pt"
            validate_experiment_config(config, "predict", base_dir=root, check_paths=True)

    def test_unified_experiment_rl_dry_run_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "rl_config.json"
            output_path = tmp_path / "rl_summary.json"
            run_dir = tmp_path / "rl_run"
            config_path.write_text(
                json.dumps(
                    {
                        "algorithm": "maskable_ppo",
                        "env_config": str(ROOT / "configs/simulation/tray_env_v0.yaml"),
                        "total_timesteps": 100,
                        "seed": 42,
                        "policy": "MultiInputPolicy",
                        "outputs": {"run_dir": str(run_dir)},
                    }
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_experiments",
                    "rl",
                    "--config",
                    str(config_path),
                    "--dry-run",
                    "--out",
                    str(output_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(completed.stdout)
            written = json.loads(output_path.read_text(encoding="utf-8"))
            registry = json.loads((output_path.parent / "artifact_registry.json").read_text(encoding="utf-8"))
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["dry_run"])
            self.assertEqual(written["run_dir"], str(run_dir))
            self.assertTrue((run_dir / "training_log.jsonl").exists())
            self.assertEqual(registry["runs"][0]["command"], "seedling-experiments:rl:train")
            self.assertEqual(
                [artifact["role"] for artifact in registry["runs"][0]["artifacts"]],
                ["input", "output"],
            )

    def test_unified_experiment_rl_baseline_eval_writes_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "rl_config.json"
            output_path = tmp_path / "rl_baselines.json"
            config_path.write_text(
                json.dumps(
                    {
                        "env_config": str(ROOT / "configs/simulation/tray_env_v0.yaml"),
                        "eval": {"compare_baselines": ["noop", "route_planning"]},
                    }
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_experiments",
                    "rl",
                    "--config",
                    str(config_path),
                    "--mode",
                    "evaluate-baselines",
                    "--episodes",
                    "1",
                    "--out",
                    str(output_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(completed.stdout)
            registry = json.loads((output_path.parent / "artifact_registry.json").read_text(encoding="utf-8"))
            self.assertTrue(payload["ok"])
            self.assertEqual([item["policy"] for item in payload["results"]], ["noop", "route_planning"])
            self.assertEqual(registry["runs"][0]["command"], "seedling-experiments:rl:evaluate-baselines")
            self.assertEqual(
                [artifact["role"] for artifact in registry["runs"][0]["artifacts"]],
                ["input", "output"],
            )


def _dependency_names(values: list[str]) -> set[str]:
    return {_dependency_name(value) for value in values}


def _requirement_names(path: Path, seen: set[Path] | None = None) -> set[str]:
    seen = seen or set()
    path = path.resolve()
    if path in seen:
        return set()
    seen.add(path)
    names: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-r "):
            names.update(_requirement_names(path.parent / line[3:].strip(), seen))
            continue
        names.add(_dependency_name(line))
    return names


def _dependency_name(value: str) -> str:
    return re.split(r"[<>=!~\[]", value, maxsplit=1)[0].strip().lower().replace("_", "-")


if __name__ == "__main__":
    unittest.main()
