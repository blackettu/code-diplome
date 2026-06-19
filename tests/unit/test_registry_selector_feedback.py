from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from seedling_core.registry import ComponentRegistry, ModelRegistry
from seedling_ui.feedback import (
    AnnotationFeedback,
    feedback_to_annotation_tasks,
)


ROOT = Path(__file__).resolve().parents[2]


class RegistrySelectorAndFeedbackTests(unittest.TestCase):
    def test_component_registry_selects_baseline_detector(self) -> None:
        registry = ComponentRegistry.from_file(ROOT / "configs/registry/models_v0_1.yaml")

        selected = registry.select(kind="detector", name="baseline_green")

        self.assertEqual(selected.entry_id, "detector_baseline_green_v0")
        self.assertEqual(selected.backend, "seedling_vision.adapters.BaselineGreenDetector")
        self.assertIn("hsv", selected.tags)

    def test_selector_cli_emits_selected_entry(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "seedling_ui",
                "selector",
                "--registry",
                str(ROOT / "configs/registry/policies_v0_1.yaml"),
                "--kind",
                "policy",
                "--name",
                "risk_aware_rule",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        payload = json.loads(completed.stdout)

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["entry"]["entry_id"], "policy_risk_aware_rule_v0")

    def test_component_registry_selects_route_planning_policy(self) -> None:
        registry = ComponentRegistry.from_file(ROOT / "configs/registry/policies_v0_1.yaml")

        selected = registry.select(kind="policy", name="route_planning")

        self.assertEqual(selected.entry_id, "policy_route_planning_v0")
        self.assertEqual(selected.backend, "seedling_decision.policies.RoutePlanningPolicy")
        self.assertIn("tsp_like", selected.tags)
        self.assertEqual(selected.metadata["safety_level"], "dry_run")

    def test_component_registry_validates_default_registries(self) -> None:
        model_components = ComponentRegistry.from_file(ROOT / "configs/registry/models_v0_1.yaml")
        policy_components = ComponentRegistry.from_file(ROOT / "configs/registry/policies_v0_1.yaml")

        model_validation = model_components.validate()
        policy_validation = policy_components.validate()

        self.assertTrue(model_validation["ok"], model_validation)
        self.assertTrue(policy_validation["ok"], policy_validation)
        self.assertEqual(model_validation["entries"], 4)
        self.assertEqual(policy_validation["entries"], 7)
        self.assertEqual(model_validation["warnings"], [])
        self.assertEqual(policy_validation["warnings"], [])

    def test_component_registry_blocks_duplicate_backend_and_unsafe_rl_policy(self) -> None:
        registry = ComponentRegistry.from_dict(
            {
                "version": "test",
                "entries": [
                    {
                        "entry_id": "policy_bad_v0",
                        "kind": "policy",
                        "name": "bad_rl",
                        "backend": "missing_backend.module.Policy",
                        "status": "draft",
                        "tags": ["rl"],
                        "metadata": {
                            "safety_level": "supervised",
                            "requires_safety_gate": False,
                            "direct_hardware_access": True,
                            "requires_action_mask": False,
                        },
                    },
                    {
                        "entry_id": "policy_bad_v0",
                        "kind": "policy",
                        "name": "bad_rl",
                        "backend": "seedling_decision.policies.NoOpPolicy",
                        "status": "draft",
                        "metadata": {"safety_level": "dry_run", "requires_safety_gate": True},
                    },
                ],
            }
        )

        validation = registry.validate()
        errors = "\n".join(validation["errors"])
        warnings = "\n".join(validation["warnings"])

        self.assertFalse(validation["ok"])
        self.assertIn("duplicate entry_id", errors)
        self.assertIn("duplicate kind/name selector", errors)
        self.assertIn("backend module is not importable", errors)
        self.assertIn("rl policy component must remain offline_only/simulation_only/dry_run", errors)
        self.assertIn("metadata.direct_hardware_access=false", errors)
        self.assertIn("metadata.requires_action_mask=true", errors)
        self.assertIn("metadata.requires_safety_gate=true", warnings)

    def test_typed_model_registry_validates_required_metadata(self) -> None:
        registry = ModelRegistry.from_file(ROOT / "configs/registry/model_registry_v0_1.yaml")

        validation = registry.validate()
        selected = registry.select(model_id="route_planning_policy_v0")
        recurrent = registry.select(model_id="recurrent_ppo_v0_seed42")

        self.assertTrue(validation["ok"], validation)
        self.assertEqual(validation["models"], 4)
        self.assertEqual(selected.model_type, "policy")
        self.assertEqual(selected.safety_level, "dry_run")
        self.assertEqual(selected.output_schema, "ActionPlan")
        self.assertEqual(recurrent.model_type, "rl_policy")
        self.assertEqual(recurrent.safety_level, "simulation_only")
        self.assertTrue(recurrent.metadata["recurrent"])

    def test_typed_model_registry_blocks_production_candidate_without_review(self) -> None:
        registry = ModelRegistry.from_dict(
            {
                "version": "test",
                "models": [
                    {
                        "model_id": "unsafe_production_model",
                        "model_type": "policy",
                        "framework": "test",
                        "artifact_uri": "builtin://test",
                        "input_schema": "SceneState",
                        "output_schema": "ActionPlan",
                        "status": "validated",
                        "safety_level": "production_candidate",
                        "calibration_requirements": {"required": True},
                    }
                ],
            }
        )

        validation = registry.validate()

        self.assertFalse(validation["ok"])
        self.assertIn("production_candidate requires separate safety review", "\n".join(validation["errors"]))

    def test_typed_model_registry_blocks_rl_direct_hardware_access(self) -> None:
        registry = ModelRegistry.from_dict(
            {
                "version": "test",
                "models": [
                    {
                        "model_id": "unsafe_rl",
                        "model_type": "rl_policy",
                        "framework": "sb3",
                        "artifact_uri": "runs/rl/model.zip",
                        "input_schema": "SeedlingTrayEnvObservationV1",
                        "output_schema": "ActionIndexV1",
                        "status": "validated",
                        "safety_level": "supervised",
                        "calibration_requirements": {"required": True},
                        "metadata": {
                            "direct_hardware_access": True,
                            "requires_action_mask": False,
                        },
                    }
                ],
            }
        )

        validation = registry.validate()
        errors = "\n".join(validation["errors"])

        self.assertFalse(validation["ok"])
        self.assertIn("rl_policy must remain offline_only/simulation_only/dry_run", errors)
        self.assertIn("metadata.direct_hardware_access=false", errors)
        self.assertIn("metadata.requires_action_mask=true", errors)

    def test_model_registry_cli_validates_and_selects_record(self) -> None:
        registry_path = ROOT / "configs/registry/model_registry_v0_1.yaml"

        validation = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "seedling_ui",
                "model-registry",
                "--registry",
                str(registry_path),
                "--validate",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        selected = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "seedling_ui",
                "model-registry",
                "--registry",
                str(registry_path),
                "--model-id",
                "baseline_green_detector_v0",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertTrue(json.loads(validation.stdout)["ok"])
        self.assertEqual(json.loads(selected.stdout)["model"]["model_type"], "detector")

    def test_selector_cli_validates_component_registry(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "seedling_ui",
                "selector",
                "--registry",
                str(ROOT / "configs/registry/policies_v0_1.yaml"),
                "--validate",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        payload = json.loads(completed.stdout)

        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["entries"], 7)
        self.assertEqual(payload["warnings"], [])

    def test_feedback_rows_convert_to_annotation_tasks(self) -> None:
        row = AnnotationFeedback(
            image_id="tray001.jpg",
            target_id="target_001",
            object_id="obj_001",
            cell_id="r00_c00",
            error_type="wrong_target",
            comment="operator rejected target",
            operator_id="operator_a",
            proposed_correction={"target_type": "human_review_required"},
        )

        tasks = feedback_to_annotation_tasks([row])

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].image_id, "tray001.jpg")
        self.assertEqual(tasks[0].reason, "wrong_target")
        self.assertEqual(tasks[0].target_id, "target_001")
        self.assertEqual(tasks[0].priority, "high")
        self.assertEqual(tasks[0].proposed_correction["target_type"], "human_review_required")
        self.assertTrue(tasks[0].task_id.startswith("feedback_tray001_jpg_target_001_"))

    def test_feedback_priority_override_is_preserved_in_annotation_task(self) -> None:
        row = AnnotationFeedback(
            image_id="tray001.jpg",
            cell_id="r00_c00",
            error_type="bad_cell_state",
            comment="cell should be weed only",
            priority="urgent",
            proposed_correction={"cell_state": "weed_only"},
        )

        tasks = feedback_to_annotation_tasks([row])

        self.assertEqual(tasks[0].priority, "urgent")
        self.assertEqual(tasks[0].proposed_correction, {"cell_state": "weed_only"})

    def test_feedback_cli_writes_annotation_task_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            feedback_path = Path(tmp) / "feedback.jsonl"
            tasks_path = Path(tmp) / "annotation_tasks.jsonl"
            add = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_ui",
                    "feedback",
                    "--feedback",
                    str(feedback_path),
                    "--image-id",
                    "tray001.jpg",
                    "--target-id",
                    "target_001",
                    "--error-type",
                    "missed_target",
                    "--comment",
                    "needs relabel",
                    "--priority",
                    "urgent",
                    "--proposed-correction-json",
                    json.dumps({"target_type": "remove_weed", "cell_id": "r00_c00"}),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            summary = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_ui",
                    "feedback",
                    "--feedback",
                    str(feedback_path),
                    "--summary",
                    "--annotation-tasks-out",
                    str(tasks_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            add_payload = json.loads(add.stdout)
            summary_payload = json.loads(summary.stdout)
            registry = json.loads((Path(tmp) / "artifact_registry.json").read_text(encoding="utf-8"))
            add_snapshot = json.loads((Path(tmp) / "feedback.run_snapshot.json").read_text(encoding="utf-8"))
            tasks_snapshot = json.loads((Path(tmp) / "annotation_tasks.run_snapshot.json").read_text(encoding="utf-8"))

            self.assertTrue(add_payload["artifact_registry"])
            self.assertTrue(summary_payload["artifact_registry"])
            self.assertEqual(add_payload["run_snapshot"], str(Path(tmp) / "feedback.run_snapshot.json"))
            self.assertEqual(summary_payload["run_snapshot"], str(Path(tmp) / "annotation_tasks.run_snapshot.json"))
            self.assertEqual(add_snapshot["command"], "seedling-ui:feedback:add")
            self.assertEqual(tasks_snapshot["command"], "seedling-ui:feedback:annotation-tasks")
            self.assertEqual(summary_payload["annotation_tasks"], 1)
            self.assertTrue(tasks_path.exists())
            saved = json.loads(tasks_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(saved["reason"], "missed_target")
            self.assertEqual(saved["priority"], "urgent")
            self.assertEqual(saved["proposed_correction"]["target_type"], "remove_weed")
            self.assertEqual(saved["status"], "open")
            self.assertEqual(
                {run["command"] for run in registry["runs"]},
                {"seedling-ui:feedback:add", "seedling-ui:feedback:annotation-tasks"},
            )
            tasks_run = next(run for run in registry["runs"] if run["command"] == "seedling-ui:feedback:annotation-tasks")
            self.assertEqual([artifact["role"] for artifact in tasks_run["artifacts"]], ["input", "output", "output"])
            self.assertIn(str(Path(tmp) / "annotation_tasks.run_snapshot.json"), {artifact["path"] for artifact in tasks_run["artifacts"]})


@unittest.skipIf(importlib.util.find_spec("cv2") is None, "opencv-python is not installed")
class BaselineGreenDetectorTests(unittest.TestCase):
    def test_baseline_green_detector_finds_green_region(self) -> None:
        from seedling_core.schemas import InferenceContext
        from seedling_vision.adapters import BaselineGreenDetector

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "green.png"
            image = Image.new("RGB", (64, 64), "black")
            draw = ImageDraw.Draw(image)
            draw.rectangle([20, 18, 42, 45], fill=(0, 180, 0))
            image.save(image_path)

            detector = BaselineGreenDetector(config={"min_seedling_area": 20})
            detector.load("baseline://green")
            result = detector.predict(image_path, InferenceContext(image_id="green.png"))

            self.assertEqual(result.image_ref, "green.png")
            self.assertEqual(result.image_size_px, [64, 64])
            self.assertGreaterEqual(len(result.detections), 1)
            self.assertEqual(result.detections[0].class_name, "crop_seedling")


if __name__ == "__main__":
    unittest.main()
