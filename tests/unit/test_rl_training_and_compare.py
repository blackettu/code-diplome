from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from seedling_core.schemas import ActionTarget, CellState, RobotState, SafetyState, SceneState, TrayState
from seedling_reports.compare import compare_runs
from seedling_rl.curriculum import default_curriculum
from seedling_rl.envs import TrayEnvConfig
from seedling_rl.offline_replay import evaluate_replay_logs
from seedling_rl.policy_adapter import RLPolicyAdapter
from seedling_rl.sweep import build_sweep_plan, build_sweep_stability_report, write_sweep_stability_report
from seedling_rl.training import RLTrainingConfig, evaluate_checkpoint, evaluate_model, train_from_config
from seedling_rl.vectorized_env import vectorized_env_spec
from seedling_sim import ReplayLogger


ROOT = Path(__file__).resolve().parents[2]


class _FakeModel:
    def __init__(self, action: int):
        self.action = action
        self.seen_observation = None

    def predict(self, observation, deterministic: bool = True):
        self.seen_observation = observation
        return self.action, None


class _FakeRecurrentModel:
    def __init__(self, actions: list[int]):
        self.actions = actions
        self.calls: list[dict[str, object]] = []

    def predict(
        self,
        observation,
        state: object | None = None,
        episode_start: object | None = None,
        deterministic: bool = True,
    ):
        self.calls.append(
            {
                "observation": observation,
                "state": state,
                "episode_start": episode_start.tolist() if hasattr(episode_start, "tolist") else episode_start,
                "deterministic": deterministic,
            }
        )
        action = self.actions[min(len(self.calls) - 1, len(self.actions) - 1)]
        return action, f"state_{len(self.calls)}"


class _FakeTrainAlgorithm:
    env = None
    kwargs: dict[str, object] = {}
    learned_timesteps = None

    def __init__(self, policy: str, env: object, **kwargs: object) -> None:
        self.policy = policy
        self.env = env
        self.kwargs = kwargs
        type(self).env = env
        type(self).kwargs = kwargs

    def learn(self, total_timesteps: int, callback: object | None = None) -> None:
        type(self).learned_timesteps = total_timesteps

    def save(self, path: str) -> None:
        Path(path).write_text("fake checkpoint", encoding="utf-8")


class RLTrainingAndCompareTests(unittest.TestCase):
    def test_rl_train_dry_run_writes_summary_and_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_config = Path(tmp) / "tray_env.yaml"
            env_config.write_text("grid_rows: 1\ngrid_cols: 1\ncell_size_mm: [33.0, 33.0]\n", encoding="utf-8")
            run_dir = Path(tmp) / "rl_run"
            rl_config = Path(tmp) / "rl.yaml"
            rl_config.write_text(
                "\n".join(
                    [
                        "algorithm: ppo",
                        f"env_config: {env_config.as_posix()}",
                        "total_timesteps: 10",
                        "seed: 7",
                        "policy: MultiInputPolicy",
                        "outputs:",
                        f"  run_dir: {run_dir.as_posix()}",
                    ]
                ),
                encoding="utf-8",
            )

            result = train_from_config(rl_config, dry_run=True)

            self.assertTrue(result["ok"])
            self.assertTrue(result["dry_run"])
            self.assertTrue((run_dir / "train_summary.json").exists())
            self.assertTrue((run_dir / "training_log.jsonl").exists())
            self.assertTrue((run_dir / "rl_metrics.jsonl").exists())
            self.assertTrue((run_dir / "rl_metrics.csv").exists())
            self.assertTrue((run_dir / "artifact_registry.json").exists())
            self.assertTrue((run_dir / "run_snapshot.json").exists())
            summary = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
            snapshot = json.loads((run_dir / "run_snapshot.json").read_text(encoding="utf-8"))
            registry = json.loads((run_dir / "artifact_registry.json").read_text(encoding="utf-8"))
            self.assertEqual(result["run_snapshot"], str(run_dir / "run_snapshot.json"))
            self.assertEqual(summary["run_snapshot"], str(run_dir / "run_snapshot.json"))
            self.assertEqual(snapshot["command"], "seedling-rl:train")
            self.assertTrue(snapshot["metadata"]["dry_run"])
            self.assertIn(str(run_dir / "run_snapshot.json"), {artifact["path"] for artifact in registry["runs"][0]["artifacts"]})
            self.assertIn("metric_summary", summary)
            self.assertIn("total_distance_mm", summary["metric_summary"])

    def test_rl_train_uses_vectorized_env_when_n_envs_gt_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_config = Path(tmp) / "tray_env.yaml"
            env_config.write_text("grid_rows: 1\ngrid_cols: 1\ncell_size_mm: [33.0, 33.0]\n", encoding="utf-8")
            run_dir = Path(tmp) / "vector_rl_run"
            rl_config = Path(tmp) / "rl_vector.yaml"
            rl_config.write_text(
                "\n".join(
                    [
                        "algorithm: ppo",
                        f"env_config: {env_config.as_posix()}",
                        "total_timesteps: 12",
                        "seed: 7",
                        "policy: MultiInputPolicy",
                        "n_envs: 3",
                        "outputs:",
                        f"  run_dir: {run_dir.as_posix()}",
                    ]
                ),
                encoding="utf-8",
            )
            fake_vec_env = object()

            with (
                patch("seedling_rl.training._algorithm_class", return_value=_FakeTrainAlgorithm),
                patch("seedling_rl.training.make_vectorized_env", return_value=fake_vec_env) as make_vec,
            ):
                result = train_from_config(rl_config, dry_run=False)

            make_vec.assert_called_once()
            self.assertEqual(make_vec.call_args.kwargs["n_envs"], 3)
            self.assertIs(_FakeTrainAlgorithm.env, fake_vec_env)
            self.assertEqual(_FakeTrainAlgorithm.learned_timesteps, 12)
            self.assertTrue(result["ok"])
            self.assertFalse(result["dry_run"])
            self.assertEqual(json.loads((run_dir / "rl_config.json").read_text(encoding="utf-8"))["n_envs"], 3)
            self.assertTrue((run_dir / "model.zip").exists())
            self.assertEqual(result["run_snapshot"], str(run_dir / "run_snapshot.json"))
            self.assertTrue((run_dir / "run_snapshot.json").exists())

    def test_recurrent_ppo_train_dry_run_records_recurrent_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_config = Path(tmp) / "tray_env.yaml"
            env_config.write_text("grid_rows: 1\ngrid_cols: 1\ncell_size_mm: [33.0, 33.0]\n", encoding="utf-8")
            run_dir = Path(tmp) / "recurrent_rl_run"
            rl_config = Path(tmp) / "recurrent_rl.yaml"
            rl_config.write_text(
                "\n".join(
                    [
                        "algorithm: recurrent_ppo",
                        f"env_config: {env_config.as_posix()}",
                        "policy: MultiInputLstmPolicy",
                        "total_timesteps: 10",
                        "recurrent: true",
                        "outputs:",
                        f"  run_dir: {run_dir.as_posix()}",
                    ]
                ),
                encoding="utf-8",
            )

            result = train_from_config(rl_config, dry_run=True)

            self.assertTrue(result["dry_run"])
            self.assertTrue(result["config"]["recurrent"])
            self.assertEqual(result["config"]["algorithm"], "recurrent_ppo")

    def test_recurrent_ppo_example_config_loads(self) -> None:
        config = RLTrainingConfig.from_file(ROOT / "configs" / "rl" / "recurrent_ppo_v0.yaml")

        self.assertEqual(config.algorithm, "recurrent_ppo")
        self.assertEqual(config.policy, "MultiInputLstmPolicy")
        self.assertTrue(config.recurrent)
        self.assertEqual(config.run_dir, "runs/rl/recurrent_ppo_v0_seed42")

    def test_training_config_parses_recurrent_boolean_strings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "rl.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "algorithm: ppo",
                        'recurrent: "false"',
                    ]
                ),
                encoding="utf-8",
            )
            recurrent_config_path = Path(tmp) / "recurrent_rl.yaml"
            recurrent_config_path.write_text("algorithm: recurrent_ppo\n", encoding="utf-8")

            config = RLTrainingConfig.from_file(config_path)
            recurrent_config = RLTrainingConfig.from_file(recurrent_config_path)

        self.assertFalse(config.recurrent)
        self.assertTrue(recurrent_config.recurrent)

    def test_rl_evaluate_baseline_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_config = Path(tmp) / "tray_env.yaml"
            env_config.write_text(
                "\n".join(
                    [
                        "grid_rows: 1",
                        "grid_cols: 1",
                        "cell_size_mm: [33.0, 33.0]",
                        "max_targets: 8",
                        "seed: 1",
                        "scene_generator:",
                        "  p_empty: 0.0",
                        "  p_single_crop: 0.0",
                        "  p_multiple_crop: 1.0",
                        "  p_weed_present: 0.0",
                    ]
                ),
                encoding="utf-8",
            )
            out = Path(tmp) / "eval"

            result = evaluate_checkpoint(None, env_config, episodes=1, out_dir=out, baseline="raster_scan")

            self.assertEqual(result["mode"], "baseline")
            self.assertTrue((out / "rl_eval_metrics.json").exists())
            self.assertTrue((out / "critical_events.json").exists())
            self.assertTrue((out / "rl_metrics.jsonl").exists())
            self.assertTrue((out / "rl_metrics.csv").exists())
            self.assertEqual(result["run_snapshot"], str(out / "run_snapshot.json"))
            self.assertTrue((out / "run_snapshot.json").exists())
            self.assertTrue((out / "replays" / "raster_scan_episode_0000.json").exists())
            self.assertEqual(len(result["replay_paths"]), 1)
            self.assertIn("metric_summary", result)
            self.assertIsNotNone(result["results"][0]["total_distance_mm"])
            self.assertIsNotNone(result["results"][0]["mean_distance_error_mm"])
            replay = json.loads((out / "replays" / "raster_scan_episode_0000.json").read_text(encoding="utf-8"))
            self.assertEqual(replay["metadata"]["policy"], "raster_scan")

    def test_evaluate_model_writes_checkpoint_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = TrayEnvConfig(
                grid_rows=1,
                grid_cols=1,
                max_targets=8,
                max_steps=4,
                seed=1,
            )
            replay_dir = Path(tmp) / "replays"
            model = _FakeModel(action=0)

            metrics = evaluate_model(model, config, episodes=1, replay_dir=replay_dir)

            self.assertEqual(metrics["episodes"], 1)
            self.assertTrue((replay_dir / "checkpoint_episode_0000.json").exists())
            self.assertEqual(len(metrics["replay_paths"]), 1)

    def test_evaluate_model_preserves_recurrent_state(self) -> None:
        config = TrayEnvConfig(
            grid_rows=1,
            grid_cols=1,
            max_targets=8,
            max_steps=4,
            seed=1,
        )
        model = _FakeRecurrentModel(actions=[0, 9])

        metrics = evaluate_model(model, config, episodes=1, recurrent=True)

        self.assertEqual(metrics["episodes"], 1)
        self.assertGreaterEqual(len(model.calls), 2)
        self.assertEqual(model.calls[0]["episode_start"], [True])
        self.assertIsNone(model.calls[0]["state"])
        self.assertEqual(model.calls[1]["episode_start"], [False])
        self.assertEqual(model.calls[1]["state"], "state_1")

    def test_rl_policy_adapter_returns_command_for_model_action(self) -> None:
        scene = SceneState(
            scene_id="scene_001",
            image_ref="image.jpg",
            dataset_version="zks",
            ontology_version="ontology",
            image_size_px=[100, 100],
            tray=TrayState("tray", 1, 1, bbox_xyxy_px=[0, 0, 100, 100]),
            robot=RobotState(position_mm=[0, 0, 0], homed=True),
            cells=[
                CellState(
                    cell_id="r00_c00",
                    row=0,
                    col=0,
                    polygon_px=[[0, 0], [100, 0], [100, 100], [0, 100]],
                    state="multiple_crop",
                )
            ],
            targets=[
                ActionTarget(
                    target_id="target_001",
                    cell_id="r00_c00",
                    object_id="obj_001",
                    target_type="remove_extra_crop",
                    action_point_px=[10, 10],
                    action_point_mm=[10, 10],
                    robot_point_mm=[10, 10, 0],
                    uncertainty_radius_mm=1.0,
                    min_distance_to_keep_mm=6.0,
                )
            ],
            safety=SafetyState(calibration_valid=True),
        )
        model = _FakeModel(action=0)

        plan = RLPolicyAdapter(model=model, max_targets=8).propose_plan(scene)

        self.assertEqual([command.target_id for command in plan.commands], ["target_001"])
        self.assertIn("cell_tensor", model.seen_observation)
        self.assertEqual(plan.metadata["rl_action_kind"], "target_command")
        self.assertEqual(plan.metadata["selected_target_id"], "target_001")

    def test_rl_policy_adapter_reports_blocked_action_mask_reasons(self) -> None:
        scene = _rl_policy_scene()
        scene.targets[0].min_distance_to_keep_mm = 1.0
        model = _FakeModel(action=0)

        plan = RLPolicyAdapter(model=model, max_targets=8).propose_plan(scene)

        self.assertEqual(plan.commands, [])
        self.assertEqual(plan.blocked_target_ids, ["target_001"])
        self.assertEqual(plan.metadata["rl_action_kind"], "blocked_target")
        self.assertIn("too_close_to_keep_seedling", plan.metadata["blocked_reasons_by_target"]["target_001"])

    def test_rl_policy_adapter_review_action_surfaces_review_reasons(self) -> None:
        scene = _rl_policy_scene()
        scene.targets[0].human_review_required = True
        model = _FakeModel(action=8)

        plan = RLPolicyAdapter(model=model, max_targets=8).propose_plan(scene)

        self.assertEqual(plan.commands, [])
        self.assertEqual(plan.review_target_ids, ["target_001"])
        self.assertEqual(plan.metadata["rl_action_kind"], "review")
        self.assertIn("human_review_required", plan.metadata["review_reasons_by_target"]["target_001"])

    def test_rl_policy_adapter_preserves_recurrent_state_until_reset(self) -> None:
        scene = _rl_policy_scene()
        model = _FakeRecurrentModel(actions=[0, 9, 9])
        adapter = RLPolicyAdapter(model=model, max_targets=8, recurrent=True)

        first = adapter.propose_plan(scene)
        second = adapter.propose_plan(scene)
        adapter.reset(scene)
        third = adapter.propose_plan(scene)

        self.assertEqual([command.target_id for command in first.commands], ["target_001"])
        self.assertEqual(second.metadata["rl_action_kind"], "stop")
        self.assertEqual(third.metadata["rl_action_kind"], "stop")
        self.assertEqual(model.calls[0]["episode_start"], [True])
        self.assertIsNone(model.calls[0]["state"])
        self.assertEqual(model.calls[1]["episode_start"], [False])
        self.assertEqual(model.calls[1]["state"], "state_1")
        self.assertEqual(model.calls[2]["episode_start"], [True])
        self.assertIsNone(model.calls[2]["state"])

    def test_rl_policy_adapter_loader_accepts_recurrent_ppo(self) -> None:
        from seedling_rl.policy_adapter import _load_sb3_model

        class FailingLoader:
            @classmethod
            def load(cls, checkpoint: str) -> object:
                raise ValueError(f"{cls.__name__} cannot load {checkpoint}")

        class RecurrentLoader:
            @classmethod
            def load(cls, checkpoint: str) -> object:
                return {"loader": cls.__name__, "checkpoint": checkpoint}

        sb3_contrib = types.ModuleType("sb3_contrib")
        sb3_contrib.RecurrentPPO = RecurrentLoader
        sb3_contrib.MaskablePPO = FailingLoader
        stable_baselines3 = types.ModuleType("stable_baselines3")
        stable_baselines3.PPO = FailingLoader

        with patch.dict(
            sys.modules,
            {
                "sb3_contrib": sb3_contrib,
                "stable_baselines3": stable_baselines3,
            },
        ):
            model = _load_sb3_model("checkpoint.zip")

        self.assertEqual(model, {"loader": "RecurrentLoader", "checkpoint": "checkpoint.zip"})

    def test_compare_runs_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run1 = Path(tmp) / "run1"
            run2 = Path(tmp) / "run2"
            run1.mkdir()
            run2.mkdir()
            for index, run in enumerate([run1, run2], 1):
                with (run / "task_level_results.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["cell_accuracy", "target_recall"])
                    writer.writeheader()
                    writer.writerow({"cell_accuracy": 0.8 + index / 10, "target_recall": 0.5})
                (run / "rl_eval_metrics.json").write_text(
                    json.dumps({"reward_mean": index, "critical_error_rate": 0.0, "successful_target_rate": 1.0}),
                    encoding="utf-8",
                )
            out = Path(tmp) / "compare"
            cli_out = Path(tmp) / "compare_cli"

            rows = compare_runs([run1, run2], out)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_reports",
                    "compare-runs",
                    "--runs",
                    str(run1),
                    str(run2),
                    "--out",
                    str(cli_out),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout)
            registry = json.loads((cli_out / "artifact_registry.json").read_text(encoding="utf-8"))

            self.assertEqual(len(rows), 2)
            self.assertTrue(payload["artifact_registry"])
            self.assertEqual(payload["runs"], 2)
            self.assertEqual(registry["runs"][0]["command"], "seedling-reports:compare-runs")
            self.assertTrue((out / "compare_runs.csv").exists())
            self.assertTrue((out / "compare_runs.html").exists())
            self.assertTrue((out / "compare_runs_summary.json").exists())
            self.assertEqual(rows[0]["rl_reward_mean"], 1.0)

    def test_compare_runs_reads_generated_rl_results_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "report"
            run.mkdir()
            with (run / "rl_results.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "policy",
                        "reward_mean",
                        "critical_error_rate",
                        "successful_target_rate",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "policy": "route_planning",
                        "reward_mean": 11.0,
                        "critical_error_rate": 0.1,
                        "successful_target_rate": 0.9,
                    }
                )
            out = Path(tmp) / "compare"

            rows = compare_runs([run], out)

            self.assertEqual(rows[0]["rl_reward_mean"], 11.0)
            self.assertEqual(rows[0]["rl_critical_error_rate"], 0.1)
            self.assertEqual(rows[0]["rl_successful_target_rate"], 0.9)

    def test_compare_runs_reads_direct_rl_artifacts_without_report_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            offline = Path(tmp) / "offline_run"
            baseline = Path(tmp) / "baseline_run"
            offline.mkdir()
            baseline.mkdir()
            (offline / "offline_replay_eval.json").write_text(
                json.dumps(
                    {
                        "reward_mean": 4.0,
                        "critical_error_rate": 0.25,
                        "successful_target_rate": 0.5,
                        "critical_events_count": 2,
                        "allowed": 3,
                        "blocked": 1,
                        "block_rate": 0.25,
                        "crop_damage": 1,
                    }
                ),
                encoding="utf-8",
            )
            (baseline / "unified_rl_baselines.json").write_text(
                json.dumps(
                    {
                        "mode": "evaluate-baselines",
                        "results": [
                            {
                                "policy": "route_planning",
                                "reward_mean": 12.0,
                                "critical_error_rate": 0.0,
                                "successful_target_rate": 1.0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            out = Path(tmp) / "compare"

            rows = compare_runs([offline, baseline], out, thresholds={"max_rl_block_rate": 0.1})

            self.assertEqual(rows[0]["rl_reward_mean"], 4.0)
            self.assertEqual(rows[0]["rl_critical_events_count"], 2.0)
            self.assertEqual(rows[0]["rl_block_rate"], 0.25)
            self.assertEqual(rows[0]["rl_crop_damage_count"], 1.0)
            self.assertEqual(rows[0]["threshold_status"], "fail")
            self.assertEqual(rows[1]["rl_reward_mean"], 12.0)
            self.assertEqual(rows[1]["rl_successful_target_rate"], 1.0)

    def test_compare_runs_reads_hardware_and_safety_results_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "report"
            run.mkdir()
            with (run / "hardware_dry_run_results.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["mode", "move_success_rate", "positioning_error_p95_mm", "safety_blocks"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "mode": "dry_run_pointer",
                        "move_success_rate": 1.0,
                        "positioning_error_p95_mm": 1.2,
                        "safety_blocks": 0,
                    }
                )
                writer.writerow(
                    {
                        "mode": "hardware_in_loop_pointer",
                        "move_success_rate": 0.8,
                        "positioning_error_p95_mm": "",
                        "safety_blocks": 1,
                    }
                )
            with (run / "safety_results.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "unsafe_action_attempted_count",
                        "unsafe_action_blocked_count",
                        "unsafe_action_escape_count",
                        "interlock_failure_count",
                        "calibration_expired_blocks",
                        "real_action_guard_blocks",
                        "unsupported_tool_profile_blocks",
                        "aborted_runs",
                        "skipped_commands",
                        "review_required_count",
                        "forbidden_zone_violation_count",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "unsafe_action_attempted_count": 2,
                        "unsafe_action_blocked_count": 1,
                        "unsafe_action_escape_count": 1,
                        "interlock_failure_count": 0,
                        "calibration_expired_blocks": 1,
                        "real_action_guard_blocks": 1,
                        "unsupported_tool_profile_blocks": 1,
                        "aborted_runs": 1,
                        "skipped_commands": 2,
                        "review_required_count": 0,
                        "forbidden_zone_violation_count": 0,
                    }
                )
            out = Path(tmp) / "compare"

            rows = compare_runs(
                [run],
                out,
                thresholds={
                    "min_hardware_move_success_rate": 0.9,
                    "max_safety_unsafe_action_escape_count": 0.0,
                    "max_safety_real_action_guard_blocks": 0.0,
                    "max_safety_unsupported_tool_profile_blocks": 0.0,
                    "max_safety_aborted_runs": 0.0,
                    "max_safety_skipped_commands": 0.0,
                },
            )

            self.assertEqual(rows[0]["hardware_move_success_rate_min"], 0.8)
            self.assertEqual(rows[0]["hardware_positioning_error_p95_mm_max"], 1.2)
            self.assertEqual(rows[0]["hardware_safety_blocks"], 1.0)
            self.assertEqual(rows[0]["safety_unsafe_action_escape_count"], 1.0)
            self.assertEqual(rows[0]["safety_real_action_guard_blocks"], 1.0)
            self.assertEqual(rows[0]["safety_unsupported_tool_profile_blocks"], 1.0)
            self.assertEqual(rows[0]["safety_aborted_runs"], 1.0)
            self.assertEqual(rows[0]["safety_skipped_commands"], 2.0)
            self.assertEqual(rows[0]["threshold_status"], "fail")
            self.assertIn("hardware_move_success_rate_min", rows[0]["threshold_failures"])
            self.assertIn("safety_real_action_guard_blocks", rows[0]["threshold_failures"])
            self.assertIn("safety_unsupported_tool_profile_blocks", rows[0]["threshold_failures"])
            self.assertIn("safety_aborted_runs", rows[0]["threshold_failures"])
            self.assertIn("safety_skipped_commands", rows[0]["threshold_failures"])

    def test_compare_runs_reads_direct_hardware_and_safety_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "report"
            run.mkdir()
            (run / "dry_run_control_points.json").write_text(
                json.dumps(
                    {
                        "ok": False,
                        "points": [
                            {"point_id": "p1", "error_mm": 0.0, "ok": True},
                            {"point_id": "p2", "error_mm": 2.0, "ok": False},
                        ],
                        "error_summary_mm": {"count": 2},
                    }
                ),
                encoding="utf-8",
            )
            (run / "dry_run_plan_report.json").write_text(
                json.dumps(
                    {
                        "ok": False,
                        "mode": "dry_run_pointer",
                        "commands": 2,
                        "executed": 1,
                        "blocked": 1,
                        "aborted": True,
                        "skipped": 1,
                        "results": [
                            {"ok": True, "safety_decision": {"allowed": True, "result": "ALLOW_DRY_RUN", "reasons": []}},
                            {
                                "ok": False,
                                "safety_decision": {
                                    "allowed": False,
                                    "result": "BLOCK_REVIEW_REQUIRED",
                                    "reasons": ["operator_confirmation_required"],
                                },
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (run / "hil_pointer_report.json").write_text(
                json.dumps(
                    {
                        "ok": False,
                        "mode": "hardware_in_loop_pointer",
                        "commands": 1,
                        "executed": 0,
                        "blocked": 1,
                        "aborted": True,
                        "skipped": 0,
                        "results": [
                            {
                                "ok": False,
                                "message": "HIL adapter only allows pointer_only",
                                "safety_decision": {
                                    "allowed": True,
                                    "result": "ALLOW_HARDWARE_IN_LOOP_POINTER",
                                    "reasons": [],
                                },
                                "outcome": {"tool_result": {"profile": {"profile_id": "laser"}}},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            out = Path(tmp) / "compare"

            rows = compare_runs(
                [run],
                out,
                thresholds={
                    "min_hardware_move_success_rate": 0.9,
                    "max_safety_unsupported_tool_profile_blocks": 0.0,
                    "max_safety_aborted_runs": 0.0,
                    "max_safety_skipped_commands": 0.0,
                },
            )

            row = rows[0]
            self.assertEqual(row["hardware_modes"], "dry_run_pointer,hardware_in_loop_pointer")
            self.assertEqual(row["hardware_move_success_rate_min"], 0.0)
            self.assertEqual(row["hardware_positioning_error_p95_mm_max"], 1.9)
            self.assertEqual(row["hardware_safety_blocks"], 2.0)
            self.assertEqual(row["safety_unsafe_action_attempted_count"], 3.0)
            self.assertEqual(row["safety_unsafe_action_blocked_count"], 2.0)
            self.assertEqual(row["safety_unsupported_tool_profile_blocks"], 1.0)
            self.assertEqual(row["safety_aborted_runs"], 2.0)
            self.assertEqual(row["safety_skipped_commands"], 1.0)
            self.assertEqual(row["safety_review_required_count"], 2.0)
            self.assertEqual(row["threshold_status"], "fail")
            self.assertIn("hardware_move_success_rate_min", row["threshold_failures"])
            self.assertIn("safety_unsupported_tool_profile_blocks", row["threshold_failures"])

    def test_compare_runs_reads_post_action_results_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "report"
            run.mkdir()
            with (run / "post_action_results.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "coverage_rate",
                        "delayed_success_rate",
                        "crop_damage_rate",
                        "regrowth_rate",
                        "incomplete_commands",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "coverage_rate": 0.75,
                        "delayed_success_rate": 0.5,
                        "crop_damage_rate": 0.25,
                        "regrowth_rate": 0.1,
                        "incomplete_commands": 1,
                    }
                )
            out = Path(tmp) / "compare"

            rows = compare_runs(
                [run],
                out,
                thresholds={
                    "min_post_action_coverage_rate": 0.9,
                    "max_post_action_crop_damage_rate": 0.1,
                },
            )

            self.assertEqual(rows[0]["post_action_coverage_rate"], 0.75)
            self.assertEqual(rows[0]["post_action_delayed_success_rate"], 0.5)
            self.assertEqual(rows[0]["post_action_crop_damage_rate"], 0.25)
            self.assertEqual(rows[0]["threshold_status"], "fail")
            self.assertIn("post_action_coverage_rate", rows[0]["threshold_failures"])

    def test_compare_runs_reads_error_budget_results_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "report"
            run.mkdir()
            with (run / "error_budget_results.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["complete", "ok", "total_error_mm"],
                )
                writer.writeheader()
                writer.writerow({"complete": "true", "ok": "true", "total_error_mm": 2.4})
                writer.writerow({"complete": "false", "ok": "false", "total_error_mm": 3.2})
            out = Path(tmp) / "compare"

            rows = compare_runs(
                [run],
                out,
                thresholds={
                    "max_error_budget_total_mm": 3.0,
                    "require_error_budget_complete": 0.0,
                },
            )

            self.assertEqual(rows[0]["error_budget_total_mm_max"], 3.2)
            self.assertEqual(rows[0]["error_budget_complete_count"], 1.0)
            self.assertEqual(rows[0]["error_budget_incomplete_count"], 1.0)
            self.assertEqual(rows[0]["error_budget_ok_count"], 1.0)
            self.assertEqual(rows[0]["threshold_status"], "fail")
            self.assertIn("error_budget_total_mm_max", rows[0]["threshold_failures"])
            self.assertIn("error_budget_incomplete_count", rows[0]["threshold_failures"])

    def test_compare_runs_reads_direct_error_budget_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            (run / "calibration_a").mkdir(parents=True)
            (run / "calibration_b").mkdir()
            (run / "calibration_a" / "error_budget.json").write_text(
                json.dumps({"complete": True, "ok": True, "total_error_mm": 2.4}),
                encoding="utf-8",
            )
            (run / "calibration_b" / "error_budget.json").write_text(
                json.dumps({"complete": False, "ok": False, "total_error_mm": 3.2}),
                encoding="utf-8",
            )
            out = Path(tmp) / "compare"

            rows = compare_runs(
                [run],
                out,
                thresholds={
                    "max_error_budget_total_mm": 3.0,
                    "require_error_budget_complete": 0.0,
                },
            )

            self.assertEqual(rows[0]["error_budget_total_mm_max"], 3.2)
            self.assertEqual(rows[0]["error_budget_complete_count"], 1.0)
            self.assertEqual(rows[0]["error_budget_incomplete_count"], 1.0)
            self.assertEqual(rows[0]["error_budget_ok_count"], 1.0)
            self.assertEqual(rows[0]["threshold_status"], "fail")

    def test_compare_runs_reads_run_snapshot_results_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "report"
            run.mkdir()
            with (run / "run_snapshot_results.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["command", "config_hash", "has_command_args"])
                writer.writeheader()
                writer.writerow({"command": "prepare", "config_hash": "abc123", "has_command_args": "True"})
                writer.writerow({"command": "predict", "config_hash": "", "has_command_args": "False"})
            out = Path(tmp) / "compare"

            rows = compare_runs(
                [run],
                out,
                thresholds={
                    "require_run_snapshots": 1.0,
                    "max_run_snapshot_missing_config_hash": 0.0,
                    "max_run_snapshot_missing_command_args": 0.0,
                },
            )

            self.assertEqual(rows[0]["run_snapshot_count"], 2.0)
            self.assertEqual(rows[0]["run_snapshot_with_config_hash_count"], 1.0)
            self.assertEqual(rows[0]["run_snapshot_missing_config_hash_count"], 1.0)
            self.assertEqual(rows[0]["run_snapshot_with_command_args_count"], 1.0)
            self.assertEqual(rows[0]["run_snapshot_missing_command_args_count"], 1.0)
            self.assertEqual(rows[0]["threshold_status"], "fail")
            self.assertIn("run_snapshot_missing_config_hash_count", rows[0]["threshold_failures"])
            self.assertIn("run_snapshot_missing_command_args_count", rows[0]["threshold_failures"])

    def test_compare_runs_reads_direct_run_snapshot_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            (run / "prepare").mkdir(parents=True)
            (run / "predict").mkdir()
            (run / "prepare" / "run_snapshot.json").write_text(
                json.dumps({"command": "prepare", "config_hash": "abc123", "command_args": {"config": "prepare.yaml"}}),
                encoding="utf-8",
            )
            (run / "predict" / "run_snapshot.json").write_text(
                json.dumps({"command": "predict", "config_hash": "", "command_args": {}}),
                encoding="utf-8",
            )
            (run / "offline_viewer.run_snapshot.json").write_text(
                json.dumps({"command": "seedling-ui:offline-viewer", "config_hash": "viewer123", "command_args": {"out": "viewer.html"}}),
                encoding="utf-8",
            )
            out = Path(tmp) / "compare"

            rows = compare_runs(
                [run],
                out,
                thresholds={
                    "require_run_snapshots": 1.0,
                    "max_run_snapshot_missing_config_hash": 0.0,
                    "max_run_snapshot_missing_command_args": 0.0,
                },
            )

            self.assertEqual(rows[0]["run_snapshot_count"], 3.0)
            self.assertEqual(rows[0]["run_snapshot_with_config_hash_count"], 2.0)
            self.assertEqual(rows[0]["run_snapshot_missing_config_hash_count"], 1.0)
            self.assertEqual(rows[0]["run_snapshot_with_command_args_count"], 2.0)
            self.assertEqual(rows[0]["run_snapshot_missing_command_args_count"], 1.0)
            self.assertEqual(rows[0]["threshold_status"], "fail")

    def test_compare_runs_applies_thresholds_and_reads_rl_seed_stability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "report"
            run.mkdir()
            with (run / "task_level_results.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["cell_accuracy", "edge_cell_accuracy", "corner_cell_accuracy", "target_recall"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "cell_accuracy": 0.9,
                        "edge_cell_accuracy": 0.72,
                        "corner_cell_accuracy": 0.5,
                        "target_recall": 0.7,
                    }
                )
            (run / "rl_eval_metrics.json").write_text(
                json.dumps(
                    {
                        "reward_mean": 11.0,
                        "critical_error_rate": 0.0,
                        "successful_target_rate": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            (run / "sweep_stability_summary.json").write_text(
                json.dumps(
                    {
                        "groups": [
                            {
                                "group_id": "ppo_lr_0003",
                                "seed_count": 2,
                                "metrics": {
                                    "reward_mean": {"mean": 11.0, "std": 0.5},
                                    "critical_error_rate": {"max": 0.0},
                                    "successful_target_rate": {"mean": 1.0},
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            out = Path(tmp) / "compare"

            rows = compare_runs(
                [run],
                out,
                thresholds={
                    "min_edge_cell_accuracy": 0.8,
                    "min_corner_cell_accuracy": 0.6,
                    "min_target_recall": 0.8,
                    "max_rl_critical_error_rate": 0.01,
                    "min_rl_seed_count": 2,
                    "max_rl_reward_std": 1.0,
                },
            )

            self.assertEqual(rows[0]["rl_seed_count"], 2)
            self.assertEqual(rows[0]["rl_reward_mean_stability_std"], 0.5)
            self.assertEqual(rows[0]["edge_cell_accuracy"], 0.72)
            self.assertEqual(rows[0]["corner_cell_accuracy"], 0.5)
            self.assertEqual(rows[0]["threshold_status"], "fail")
            self.assertIn("edge_cell_accuracy", rows[0]["threshold_failures"])
            self.assertIn("corner_cell_accuracy", rows[0]["threshold_failures"])
            self.assertIn("target_recall", rows[0]["threshold_failures"])
            summary = json.loads((out / "compare_runs_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["failed"], 1)

    def test_vectorized_env_spec_and_curriculum_plan(self) -> None:
        config = TrayEnvConfig(grid_rows=1, grid_cols=1, seed=7)

        spec = vectorized_env_spec(config, n_envs=3)
        stages = default_curriculum(config, episodes_per_stage=5)

        self.assertEqual(spec.n_envs, 3)
        self.assertEqual(spec.seed, 7)
        self.assertEqual(len(stages), 3)
        self.assertEqual(stages[0].episodes, 5)

    def test_offline_replay_evaluation_counts_blocked_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "replay.json"
            out = Path(tmp) / "offline_eval.json"
            logger = ReplayLogger(replay_id="replay_001", scene_id="scene_001")
            logger.append(
                "robot_execution",
                {
                    "safety_decision": {"allowed": False, "result": "BLOCK_CALIBRATION", "reasons": ["calibration_required"]},
                    "outcome": {"reward": -1.0},
                },
            )
            logger.to_json(replay_path)

            payload = evaluate_replay_logs([replay_path], output_path=out)

            self.assertEqual(payload["blocked"], 1)
            self.assertEqual(payload["block_rate"], 1.0)
            self.assertEqual(payload["critical_events_count"], 1)
            self.assertEqual(payload["block_reason_counts"]["calibration_required"], 1)
            self.assertEqual(payload["block_reason_counts"]["BLOCK_CALIBRATION"], 1)
            self.assertTrue(out.exists())

    def test_offline_replay_evaluation_counts_adapter_failures_as_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "adapter_block_replay.json"
            logger = ReplayLogger(replay_id="replay_adapter_block", scene_id="scene_001")
            logger.append(
                "robot_execution",
                {
                    "ok": False,
                    "safety_decision": {
                        "allowed": True,
                        "result": "ALLOW_DRY_RUN",
                        "reasons": [],
                    },
                    "outcome": {
                        "tool_result": {
                            "ok": False,
                            "profile": {"profile_id": "laser", "dwell_ms": 0, "metadata": {}},
                            "message": "dry-run adapter only allows pointer_only",
                        }
                    },
                    "message": "dry-run adapter only allows pointer_only",
                },
            )
            logger.to_json(replay_path)

            payload = evaluate_replay_logs([replay_path])

            self.assertEqual(payload["allowed"], 0)
            self.assertEqual(payload["blocked"], 1)
            self.assertEqual(payload["block_reason_counts"]["unsupported_tool_profile"], 1)
            self.assertEqual(payload["critical_events_count"], 1)
            self.assertIn("unsupported_tool_profile", payload["critical_events"][0]["reasons"])

    def test_offline_replay_evaluation_summarizes_rl_step_success_and_distance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "rl_replay.json"
            logger = ReplayLogger(replay_id="replay_001", scene_id="scene_001")
            logger.append(
                "rl_step",
                {
                    "episode": 0,
                    "action": 0,
                    "reward": 9.0,
                    "event": "target_step",
                    "info": {
                        "event": "target_step",
                        "outcome": {
                            "target_id": "target_001",
                            "success": True,
                            "crop_damage": False,
                            "distance_error_mm": 0.5,
                            "actual_point_mm": [3.0, 4.0],
                            "info": {"movement_mm": 5.0},
                        },
                    },
                },
            )
            logger.append("rl_step", {"episode": 0, "action": 1, "reward": 0.0, "event": "review", "info": {"reviewed_targets": 2}})
            logger.to_json(replay_path)

            payload = evaluate_replay_logs([replay_path])

            self.assertEqual(payload["rl_step_events"], 2)
            self.assertEqual(payload["target_steps"], 1)
            self.assertEqual(payload["successful_target_rate"], 1.0)
            self.assertEqual(payload["reviewed_targets"], 2)
            self.assertEqual(payload["review_rate"], 1.0)
            self.assertEqual(payload["reward_total"], 9.0)
            self.assertEqual(payload["total_distance_mm"], 5.0)
            self.assertEqual(payload["mean_distance_error_mm"], 0.5)

    def test_sweep_plan_expands_grid_and_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sweep_config = Path(tmp) / "sweep.yaml"
            output = Path(tmp) / "sweep_plan.json"
            sweep_config.write_text(
                "\n".join(
                    [
                        "base:",
                        "  algorithm: ppo",
                        "grid:",
                        "  learning_rate: [0.001, 0.0003]",
                        "  gamma: [0.95]",
                        "seeds: [1, 2]",
                    ]
                ),
                encoding="utf-8",
            )

            payload = build_sweep_plan(sweep_config, output)

            self.assertEqual(payload["count"], 4)
            self.assertTrue(output.exists())

    def test_sweep_stability_report_groups_runs_across_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sweep_config = Path(tmp) / "sweep.yaml"
            plan_path = Path(tmp) / "sweep_plan.json"
            report_dir = Path(tmp) / "sweep_report"
            sweep_config.write_text(
                "\n".join(
                    [
                        "base:",
                        "  algorithm: ppo",
                        "grid:",
                        "  learning_rate: [0.001]",
                        "seeds: [1, 2]",
                    ]
                ),
                encoding="utf-8",
            )
            build_sweep_plan(sweep_config, plan_path)
            metrics_paths = []
            for run_id, reward in [("sweep_001_seed_001", 10.0), ("sweep_001_seed_002", 14.0)]:
                metrics_dir = Path(tmp) / run_id
                metrics_dir.mkdir()
                metrics_path = metrics_dir / "rl_eval_metrics.json"
                metrics_path.write_text(
                    json.dumps(
                        {
                            "ok": True,
                            "mode": "baseline",
                            "results": [
                                {
                                    "policy": "route_planning",
                                    "reward_mean": reward,
                                    "critical_error_rate": 0.0,
                                    "successful_target_rate": 1.0,
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                metrics_paths.append(metrics_path)

            payload = build_sweep_stability_report(plan_path, metrics_paths)
            written = write_sweep_stability_report(plan_path, metrics_paths, report_dir)

            self.assertEqual(payload["group_count"], 1)
            group = payload["groups"][0]
            self.assertEqual(group["seed_count"], 2)
            self.assertEqual(group["metrics"]["reward_mean"]["mean"], 12.0)
            self.assertEqual(group["metrics"]["reward_mean"]["std"], 2.0)
            self.assertEqual(group["metrics"]["critical_error_rate"]["max"], 0.0)
            self.assertTrue(Path(written["summary_path"]).exists())
            self.assertTrue(Path(written["csv_path"]).exists())
            self.assertTrue((report_dir / "artifact_registry.json").exists())

    def test_rl_extra_cli_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_config = Path(tmp) / "tray_env.yaml"
            baselines = Path(tmp) / "baselines.json"
            curriculum = Path(tmp) / "curriculum.json"
            replay_path = Path(tmp) / "replay.json"
            offline = Path(tmp) / "offline.json"
            sweep_config = Path(tmp) / "sweep.yaml"
            sweep_out = Path(tmp) / "sweep.json"
            sweep_report = Path(tmp) / "sweep_report"
            sweep_metrics_dir = Path(tmp) / "sweep_001_seed_001"
            sweep_metrics = sweep_metrics_dir / "rl_eval_metrics.json"
            env_config.write_text("grid_rows: 1\ngrid_cols: 1\ncell_size_mm: [33.0, 33.0]\n", encoding="utf-8")
            logger = ReplayLogger(replay_id="replay_001", scene_id="scene_001")
            logger.append("robot_execution", {"safety_decision": {"allowed": True}, "outcome": {"reward": 1.0}})
            logger.to_json(replay_path)
            sweep_config.write_text("base: {algorithm: ppo}\ngrid: {gamma: [0.95]}\nseeds: [1]\n", encoding="utf-8")
            sweep_metrics_dir.mkdir()
            sweep_metrics.write_text(
                json.dumps({"ok": True, "mode": "checkpoint", "reward_mean": 1.0, "critical_error_rate": 0.0}),
                encoding="utf-8",
            )

            commands = [
                [sys.executable, "-B", "-m", "seedling_rl", "vector-env", "--config", str(env_config), "--n-envs", "2"],
                [sys.executable, "-B", "-m", "seedling_rl", "evaluate-baselines", "--config", str(env_config), "--episodes", "1", "--out", str(baselines)],
                [sys.executable, "-B", "-m", "seedling_rl", "curriculum", "--config", str(env_config), "--out", str(curriculum)],
                [sys.executable, "-B", "-m", "seedling_rl", "offline-replay-eval", "--replays", str(replay_path), "--out", str(offline)],
                [sys.executable, "-B", "-m", "seedling_rl", "sweep", "--config", str(sweep_config), "--out", str(sweep_out)],
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_rl",
                    "sweep-report",
                    "--plan",
                    str(sweep_out),
                    "--metrics",
                    str(sweep_metrics),
                    "--out",
                    str(sweep_report),
                ],
            ]
            payloads: dict[str, dict[str, object]] = {}
            for command in commands:
                with self.subTest(command=command):
                    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
                    payload = json.loads(completed.stdout)
                    subcommand = command[4]
                    payloads[subcommand] = payload
                    self.assertTrue(payload["ok"])
                    if subcommand != "vector-env":
                        self.assertTrue(payload["artifact_registry"])
                        self.assertTrue(Path(payload["artifact_registry"]).exists())
                        self.assertTrue(payload["run_snapshot"])
                        self.assertTrue(Path(str(payload["run_snapshot"])).exists())

            registry = json.loads((Path(tmp) / "artifact_registry.json").read_text(encoding="utf-8"))
            self.assertEqual(
                {run["command"] for run in registry["runs"]},
                {
                    "seedling-rl:evaluate-baselines",
                    "seedling-rl:curriculum",
                    "seedling-rl:offline-replay-eval",
                    "seedling-rl:sweep",
                },
            )
            self.assertEqual(payloads["evaluate-baselines"]["run_snapshot"], str(Path(tmp) / "baselines.run_snapshot.json"))
            self.assertEqual(payloads["curriculum"]["run_snapshot"], str(Path(tmp) / "curriculum.run_snapshot.json"))
            self.assertEqual(payloads["offline-replay-eval"]["run_snapshot"], str(Path(tmp) / "offline.run_snapshot.json"))
            self.assertEqual(payloads["sweep"]["run_snapshot"], str(Path(tmp) / "sweep.run_snapshot.json"))
            for snapshot_name in (
                "baselines.run_snapshot.json",
                "curriculum.run_snapshot.json",
                "offline.run_snapshot.json",
                "sweep.run_snapshot.json",
            ):
                self.assertTrue((Path(tmp) / snapshot_name).exists())
            sweep_report_registry = json.loads((sweep_report / "artifact_registry.json").read_text(encoding="utf-8"))
            self.assertEqual(payloads["sweep-report"]["run_snapshot"], str(sweep_report / "sweep_stability_summary.run_snapshot.json"))
            self.assertEqual(
                {run["command"] for run in sweep_report_registry["runs"]},
                {"rl-sweep-report", "seedling-rl:sweep-report"},
            )
            cli_report_run = next(run for run in sweep_report_registry["runs"] if run["command"] == "seedling-rl:sweep-report")
            self.assertIn(str(sweep_report / "sweep_stability_summary.run_snapshot.json"), {artifact["path"] for artifact in cli_report_run["artifacts"]})


def _rl_policy_scene() -> SceneState:
    return SceneState(
        scene_id="scene_001",
        image_ref="image.jpg",
        dataset_version="zks",
        ontology_version="ontology",
        image_size_px=[100, 100],
        tray=TrayState("tray", 1, 1, bbox_xyxy_px=[0, 0, 100, 100]),
        robot=RobotState(position_mm=[0, 0, 0], homed=True),
        cells=[
            CellState(
                cell_id="r00_c00",
                row=0,
                col=0,
                polygon_px=[[0, 0], [100, 0], [100, 100], [0, 100]],
                state="multiple_crop",
            )
        ],
        targets=[
            ActionTarget(
                target_id="target_001",
                cell_id="r00_c00",
                object_id="obj_001",
                target_type="remove_extra_crop",
                action_point_px=[10, 10],
                action_point_mm=[10, 10],
                robot_point_mm=[10, 10, 0],
                uncertainty_radius_mm=1.0,
                min_distance_to_keep_mm=6.0,
            )
        ],
        safety=SafetyState(calibration_valid=True),
    )


if __name__ == "__main__":
    unittest.main()
