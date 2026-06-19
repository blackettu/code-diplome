from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from seedling_data.annotations import (
    ActionPointAnnotation,
    CellAnnotation,
    audit_action_points,
    audit_cell_annotations,
    write_action_points,
    write_cell_annotations,
)
from seedling_data.changelog import audit_dataset_changelog
from seedling_data.duplicates import find_cross_split_duplicates
from seedling_data.manifests import build_image_manifest
from seedling_data.post_action import (
    PostActionObservation,
    audit_post_action_observations,
    summarize_post_action_file,
    write_post_action_observations,
)
from seedling_data.registry import add_dataset, validate_dataset_registry, write_dataset_summary


ROOT = Path(__file__).resolve().parents[2]


class DataRegistryAnnotationTests(unittest.TestCase):
    def test_cell_and_action_annotation_audits_accept_valid_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cells = root / "cell_annotations.jsonl"
            actions = root / "action_points.jsonl"
            write_cell_annotations(
                cells,
                [
                    CellAnnotation(
                        image_id="tray001",
                        cell_id="r00_c00",
                        row=0,
                        col=0,
                        state="multiple_crop",
                        object_ids=["obj_001", "obj_002"],
                        keep_object_id="obj_001",
                    )
                ],
            )
            write_action_points(
                actions,
                [
                    ActionPointAnnotation(
                        image_id="tray001",
                        cell_id="r00_c00",
                        object_id="obj_002",
                        target_type="remove_extra_crop",
                        action_point_px=[12, 15],
                        uncertainty_radius_px=2,
                        min_distance_to_keep_px=5,
                    )
                ],
            )

            cell_audit = audit_cell_annotations(
                cells,
                ontology_path=ROOT / "configs/ontology/ontology_v0_1.yaml",
                grid_rows=11,
                grid_cols=11,
            )
            action_audit = audit_action_points(
                actions,
                ontology_path=ROOT / "configs/ontology/ontology_v0_1.yaml",
                cell_annotations_path=cells,
            )

            self.assertTrue(cell_audit["ok"], cell_audit)
            self.assertTrue(action_audit["ok"], action_audit)
            self.assertEqual(cell_audit["state_counts"]["multiple_crop"], 1)
            self.assertEqual(action_audit["target_counts"]["remove_extra_crop"], 1)

    def test_action_annotation_audit_requires_review_for_unsafe_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            actions = root / "action_points.jsonl"
            write_action_points(
                actions,
                [
                    ActionPointAnnotation(
                        image_id="tray001",
                        cell_id="r00_c00",
                        object_id="obj_close",
                        target_type="remove_extra_crop",
                        action_point_px=[12, 15],
                        uncertainty_radius_px=9,
                        min_distance_to_keep_px=2,
                    ),
                    ActionPointAnnotation(
                        image_id="tray001",
                        cell_id="r00_c01",
                        object_id="obj_forbidden",
                        target_type="remove_weed",
                        action_point_px=[22, 15],
                        forbidden_zone_ids=["keep_seedling_zone"],
                    ),
                    ActionPointAnnotation(
                        image_id="tray001",
                        cell_id="r00_c02",
                        object_id="obj_review",
                        target_type="human_review_required",
                        action_point_px=[32, 15],
                    ),
                ],
            )

            audit = audit_action_points(
                actions,
                ontology_path=ROOT / "configs/ontology/ontology_v0_1.yaml",
                min_safe_distance_px=5,
                max_uncertainty_px=4,
            )

            self.assertFalse(audit["ok"])
            self.assertEqual(audit["unsafe_distance_rows"], 1)
            self.assertEqual(audit["high_uncertainty_rows"], 1)
            self.assertEqual(audit["forbidden_zone_rows"], 1)
            self.assertIn("requires human_review_required=true", "\n".join(audit["errors"]))

    def test_action_annotation_audit_accepts_review_flags_for_unsafe_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            actions = Path(tmp) / "action_points.jsonl"
            write_action_points(
                actions,
                [
                    ActionPointAnnotation(
                        image_id="tray001",
                        cell_id="r00_c00",
                        object_id="obj_close",
                        target_type="remove_extra_crop",
                        action_point_px=[12, 15],
                        uncertainty_radius_px=9,
                        min_distance_to_keep_px=2,
                        forbidden_zone_ids=["keep_seedling_zone"],
                        human_review_required=True,
                    )
                ],
            )

            audit = audit_action_points(
                actions,
                ontology_path=ROOT / "configs/ontology/ontology_v0_1.yaml",
                min_safe_distance_px=5,
                max_uncertainty_px=4,
            )

            self.assertTrue(audit["ok"], audit)
            self.assertEqual(audit["review_required_rows"], 1)
            self.assertEqual(audit["unsafe_distance_rows"], 1)
            self.assertEqual(audit["high_uncertainty_rows"], 1)

    def test_action_annotation_audit_requires_review_for_crop_and_weed_cells(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cells = root / "cell_annotations.jsonl"
            actions = root / "action_points.jsonl"
            write_cell_annotations(
                cells,
                [
                    CellAnnotation(
                        image_id="tray001",
                        cell_id="r00_c00",
                        row=0,
                        col=0,
                        state="crop_and_weed",
                        object_ids=["crop_001", "weed_001"],
                    )
                ],
            )
            write_action_points(
                actions,
                [
                    ActionPointAnnotation(
                        image_id="tray001",
                        cell_id="r00_c00",
                        object_id="weed_001",
                        target_type="remove_weed",
                        action_point_px=[12, 15],
                    )
                ],
            )

            audit = audit_action_points(
                actions,
                ontology_path=ROOT / "configs/ontology/ontology_v0_1.yaml",
                cell_annotations_path=cells,
            )

            self.assertFalse(audit["ok"])
            self.assertEqual(audit["crop_and_weed_action_rows"], 1)
            self.assertIn("crop_and_weed action point requires", "\n".join(audit["errors"]))

            write_action_points(
                actions,
                [
                    ActionPointAnnotation(
                        image_id="tray001",
                        cell_id="r00_c00",
                        object_id="weed_001",
                        target_type="remove_weed",
                        action_point_px=[12, 15],
                        human_review_required=True,
                    )
                ],
            )
            reviewed_audit = audit_action_points(
                actions,
                ontology_path=ROOT / "configs/ontology/ontology_v0_1.yaml",
                cell_annotations_path=cells,
            )
            self.assertTrue(reviewed_audit["ok"], reviewed_audit)
            self.assertEqual(reviewed_audit["crop_and_weed_action_rows"], 1)

    def test_action_annotation_audit_cli_supports_safety_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            actions = Path(tmp) / "action_points.jsonl"
            write_action_points(
                actions,
                [
                    ActionPointAnnotation(
                        image_id="tray001",
                        cell_id="r00_c00",
                        object_id="obj_close",
                        target_type="remove_extra_crop",
                        action_point_px=[12, 15],
                        uncertainty_radius_px=9,
                        min_distance_to_keep_px=2,
                        human_review_required=True,
                    )
                ],
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_data",
                    "audit-action-points",
                    "--path",
                    str(actions),
                    "--ontology",
                    str(ROOT / "configs/ontology/ontology_v0_1.yaml"),
                    "--min-safe-distance-px",
                    "5",
                    "--max-uncertainty-px",
                    "4",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(completed.stdout)

            self.assertTrue(payload["ok"], payload)
            self.assertEqual(payload["unsafe_distance_rows"], 1)
            self.assertEqual(payload["high_uncertainty_rows"], 1)

    def test_dataset_registry_add_validate_and_summarize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            manifests = root / "manifests"
            manifests.mkdir(parents=True)
            _write_changelog(root / "DATASET_CHANGELOG.md", "tiny_v0")
            image_path = root / "tray001.png"
            Image.new("RGB", (8, 8), "green").save(image_path)
            labels = root / "labels"
            labels.mkdir()
            (labels / "tray001.txt").write_text(
                "1 0.5 0.5 0.2 0.2\n2 0.4 0.4 0.1 0.1\n",
                encoding="utf-8",
            )
            write_cell_annotations(
                manifests / "cell_annotations.jsonl",
                [
                    CellAnnotation(
                        image_id="tray001",
                        cell_id="r00_c00",
                        row=0,
                        col=0,
                        state="crop_and_weed",
                    )
                ],
            )
            write_action_points(
                manifests / "action_points.jsonl",
                [
                    ActionPointAnnotation(
                        image_id="tray001",
                        cell_id="r00_c00",
                        object_id="weed_001",
                        target_type="remove_weed",
                        action_point_px=[4, 4],
                    )
                ],
            )
            manifest_path = manifests / "image_manifest.csv"
            _write_manifest(
                manifest_path,
                [
                    {
                        "image_id": "tray001",
                        "file_path": str(image_path),
                        "sha256": _sha256(image_path),
                        "split": "train",
                    }
                ],
            )
            registry_path = Path(tmp) / "dataset_registry.yaml"
            summary_path = Path(tmp) / "summary.md"
            html_summary_path = Path(tmp) / "summary.html"

            entry = add_dataset(
                registry_path=registry_path,
                dataset_root=root,
                name="tiny_v0",
                ontology_version="ontology_v0_1",
            )
            validation = validate_dataset_registry(registry_path, dataset="tiny_v0")
            summary = write_dataset_summary(registry_path, dataset="tiny_v0", output_path=summary_path)
            html_summary = write_dataset_summary(registry_path, dataset="tiny_v0", output_path=html_summary_path)

            self.assertEqual(entry.dataset_version, "tiny_v0")
            self.assertEqual(entry.changelog_path, "DATASET_CHANGELOG.md")
            self.assertTrue(validation["ok"], validation)
            self.assertTrue(validation["datasets"][0]["changelog"]["ok"], validation)
            self.assertTrue(summary_path.exists())
            self.assertTrue(summary["changelog"]["ok"], summary)
            self.assertEqual(summary["manifest"]["images"], 1)
            self.assertEqual(summary["manifest"]["class_counts"], {"1": 1, "2": 1})
            self.assertEqual(summary["manifest"]["cell_annotations"]["counts"]["crop_and_weed"], 1)
            self.assertEqual(summary["manifest"]["action_points"]["counts"]["remove_weed"], 1)
            markdown = summary_path.read_text(encoding="utf-8")
            self.assertIn("## Class Distribution", markdown)
            self.assertIn("## Cell Distribution", markdown)
            self.assertEqual(html_summary["manifest"]["class_counts"], {"1": 1, "2": 1})
            html = html_summary_path.read_text(encoding="utf-8")
            self.assertIn("<table>", html)
            self.assertIn("crop_and_weed", html)
            self.assertIn("remove_weed", html)

    def test_dataset_changelog_audit_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            root.mkdir()
            changelog = root / "DATASET_CHANGELOG.md"
            _write_changelog(changelog, "tiny_v0")

            audit = audit_dataset_changelog(dataset_root=root, dataset_version="tiny_v0")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_data",
                    "audit-changelog",
                    "--dataset-root",
                    str(root),
                    "--dataset-version",
                    "tiny_v0",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout)

            self.assertTrue(audit["ok"], audit)
            self.assertTrue(payload["ok"], payload)
            changelog.write_text(
                "# Dataset Changelog\n\n## tiny_v0\n\n- Status: draft / frozen / archived\n",
                encoding="utf-8",
            )
            incomplete = audit_dataset_changelog(dataset_root=root, dataset_version="tiny_v0")
            self.assertFalse(incomplete["ok"])
            self.assertIn("placeholder", "\n".join(incomplete["errors"]))

    def test_dataset_changelog_accepts_source_dataset_root_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            root.mkdir()
            changelog = root / "DATASET_CHANGELOG.md"
            _write_changelog(changelog, "tiny_v0")
            text = changelog.read_text(encoding="utf-8")
            changelog.write_text(
                text.replace("- Source dataset root: dataset", "- Source dataset root: data/raw/tiny_v0"),
                encoding="utf-8",
            )

            audit = audit_dataset_changelog(dataset_root=root, dataset_version="tiny_v0")

            self.assertTrue(audit["ok"], audit)
            self.assertEqual(audit["fields"]["Source dataset root"], "data/raw/tiny_v0")

    def test_post_action_observation_audit_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            observations = Path(tmp) / "post_action_observations.jsonl"
            write_post_action_observations(
                observations,
                [
                    _post_action("cmd_001", 24, "target_survived"),
                    _post_action("cmd_001", 48, "target_survived"),
                    _post_action("cmd_001", 72, "target_removed"),
                    _post_action("cmd_002", 24, "crop_damage"),
                    _post_action("cmd_002", 48, "crop_damage"),
                    _post_action("cmd_002", 72, "crop_damage"),
                ],
            )
            summary_path = Path(tmp) / "post_action_summary.json"

            audit = audit_post_action_observations(observations)
            summary = summarize_post_action_file(observations, summary_path)

            self.assertTrue(audit["ok"], audit)
            self.assertEqual(audit["summary"]["coverage_rate"], 1.0)
            self.assertEqual(audit["summary"]["observer_ids"], ["tester"])
            self.assertEqual(summary["commands"], 2)
            self.assertEqual(summary["observer_ids"], ["tester"])
            self.assertEqual(summary["delayed_success_rate"], 0.5)
            self.assertEqual(summary["crop_damage_rate"], 0.5)
            self.assertTrue(summary_path.exists())

    def test_post_action_audit_flags_missing_windows_and_cli_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            observations = Path(tmp) / "post_action_observations.jsonl"
            summary_path = Path(tmp) / "post_action_summary.json"
            write_post_action_observations(
                observations,
                [
                    _post_action("cmd_001", 24, "uncertain"),
                    _post_action("cmd_001", 72, "target_removed"),
                ],
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_data",
                    "summarize-post-action",
                    "--path",
                    str(observations),
                    "--out",
                    str(summary_path),
                    "--required-hours",
                    "24,48,72",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout)

            self.assertTrue(payload["ok"], payload)
            self.assertEqual(payload["coverage_rate"], 0.0)
            self.assertEqual(payload["incomplete_commands"], 1)
            self.assertTrue(payload["artifact_registry"])
            self.assertTrue(summary_path.exists())
            snapshot_path = Path(tmp) / "post_action_summary.run_snapshot.json"
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_snapshot"], str(snapshot_path))
            self.assertEqual(snapshot["command"], "seedling-data:summarize-post-action")
            self.assertEqual(snapshot["metadata"]["required_hours"], [24, 48, 72])
            registry = json.loads((Path(tmp) / "artifact_registry.json").read_text(encoding="utf-8"))
            self.assertEqual(registry["runs"][0]["command"], "seedling-data:summarize-post-action")
            self.assertEqual(
                [artifact["role"] for artifact in registry["runs"][0]["artifacts"]],
                ["input", "output", "output"],
            )
            self.assertIn(
                ("run_snapshot", "output", str(snapshot_path)),
                {
                    (artifact["artifact_type"], artifact["role"], artifact["path"])
                    for artifact in registry["runs"][0]["artifacts"]
                },
            )

    def test_registry_cli_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            (root / "manifests").mkdir(parents=True)
            _write_changelog(root / "DATASET_CHANGELOG.md", "tiny_v0")
            (root / "manifests" / "image_manifest.csv").write_text(
                "image_id,file_path,sha256,split\n",
                encoding="utf-8",
            )
            registry_path = Path(tmp) / "registry.yaml"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_data",
                    "registry",
                    "add",
                    "--registry",
                    str(registry_path),
                    "--dataset-root",
                    str(root),
                    "--name",
                    "tiny_v0",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(completed.stdout)

            self.assertTrue(payload["ok"])
            self.assertTrue(payload["artifact_registry"])
            add_snapshot_path = Path(tmp) / "registry.run_snapshot.json"
            add_snapshot = json.loads(add_snapshot_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_snapshot"], str(add_snapshot_path))
            self.assertEqual(add_snapshot["command"], "seedling-data:registry:add")
            self.assertEqual(add_snapshot["metadata"]["dataset"], "tiny_v0")
            self.assertEqual(payload["entry"]["dataset_version"], "tiny_v0")
            summary_path = Path(tmp) / "dataset_summary.md"
            summarize = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_data",
                    "registry",
                    "summarize",
                    "--registry",
                    str(registry_path),
                    "--dataset",
                    "tiny_v0",
                    "--out",
                    str(summary_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            summary_payload = json.loads(summarize.stdout)
            registry = json.loads((Path(tmp) / "artifact_registry.json").read_text(encoding="utf-8"))

            self.assertTrue(summary_payload["ok"])
            self.assertTrue(summary_payload["artifact_registry"])
            summary_snapshot_path = Path(tmp) / "dataset_summary.run_snapshot.json"
            summary_snapshot = json.loads(summary_snapshot_path.read_text(encoding="utf-8"))
            self.assertEqual(summary_payload["run_snapshot"], str(summary_snapshot_path))
            self.assertEqual(summary_snapshot["command"], "seedling-data:registry:summarize")
            self.assertTrue(summary_path.exists())
            self.assertEqual(
                {run["command"] for run in registry["runs"]},
                {"seedling-data:registry:add", "seedling-data:registry:summarize"},
            )
            add_run = next(run for run in registry["runs"] if run["command"] == "seedling-data:registry:add")
            summary_run = next(run for run in registry["runs"] if run["command"] == "seedling-data:registry:summarize")
            self.assertIn(str(add_snapshot_path), {artifact["path"] for artifact in add_run["artifacts"]})
            self.assertIn(str(summary_snapshot_path), {artifact["path"] for artifact in summary_run["artifacts"]})

    def test_image_manifest_cli_writes_artifact_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            images = root / "images"
            images.mkdir(parents=True)
            Image.new("RGB", (8, 8), "green").save(images / "tray001.png")
            manifest_path = root / "manifests" / "image_manifest.csv"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_data",
                    "image-manifest",
                    "--dataset-root",
                    str(root),
                    "--images-dir",
                    "images",
                    "--output",
                    str(manifest_path),
                    "--session-id",
                    "session_cli",
                    "--tray-id",
                    "tray_cli",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(completed.stdout)
            registry = json.loads((manifest_path.parent / "artifact_registry.json").read_text(encoding="utf-8"))
            with manifest_path.open("r", encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))

            self.assertEqual(payload["images"], 1)
            self.assertTrue(payload["artifact_registry"])
            snapshot_path = manifest_path.parent / "image_manifest.run_snapshot.json"
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_snapshot"], str(snapshot_path))
            self.assertEqual(snapshot["command"], "seedling-data:image-manifest")
            self.assertEqual(snapshot["metadata"]["session_id"], "session_cli")
            self.assertEqual(row["session_id"], "session_cli")
            self.assertEqual(row["tray_id"], "tray_cli")
            self.assertEqual(registry["runs"][0]["command"], "seedling-data:image-manifest")
            self.assertEqual(
                [artifact["role"] for artifact in registry["runs"][0]["artifacts"]],
                ["input", "input", "output", "output"],
            )
            self.assertIn(
                ("run_snapshot", "output", str(snapshot_path)),
                {
                    (artifact["artifact_type"], artifact["role"], artifact["path"])
                    for artifact in registry["runs"][0]["artifacts"]
                },
            )

    def test_image_manifest_generator_fills_group_session_and_tray_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            images = root / "session_01" / "images"
            images.mkdir(parents=True)
            Image.new("RGB", (8, 8), "green").save(images / "tray_a_frame001.png")

            rows = build_image_manifest(
                root,
                images_dir=images,
                group_regex=r"(?P<tray_id>tray_[a-z])_frame\d+",
            )

            self.assertEqual(rows[0]["group_id"], "tray_a")
            self.assertEqual(rows[0]["tray_id"], "tray_a")
            self.assertEqual(rows[0]["session_id"], "session_01")
            self.assertEqual(rows[0]["width"], "8")
            self.assertEqual(rows[0]["height"], "8")
            self.assertTrue(rows[0]["sha256"])

    def test_duplicate_check_finds_cross_split_identical_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_image = root / "train.png"
            test_image = root / "test.png"
            Image.new("RGB", (8, 8), "green").save(train_image)
            Image.new("RGB", (8, 8), "green").save(test_image)
            manifest_path = root / "image_manifest.csv"
            _write_manifest(
                manifest_path,
                [
                    {
                        "image_id": "train",
                        "file_path": str(train_image),
                        "sha256": _sha256(train_image),
                        "split": "train",
                    },
                    {
                        "image_id": "test",
                        "file_path": str(test_image),
                        "sha256": _sha256(test_image),
                        "split": "test",
                    },
                ],
            )

            result = find_cross_split_duplicates(manifest_path)

            self.assertFalse(result["ok"])
            self.assertEqual(len(result["exact_sha256_duplicates"]), 1)
            self.assertGreaterEqual(len(result["near_perceptual_duplicates"]), 1)
            report_path = root / "near_duplicates.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "seedling_data",
                    "near-duplicates",
                    "--manifest",
                    str(manifest_path),
                    "--out",
                    str(report_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout)
            registry = json.loads((root / "artifact_registry.json").read_text(encoding="utf-8"))
            snapshot_path = root / "near_duplicates.run_snapshot.json"
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

            self.assertFalse(payload["ok"])
            self.assertTrue(payload["artifact_registry"])
            self.assertEqual(payload["run_snapshot"], str(snapshot_path))
            self.assertEqual(snapshot["command"], "seedling-data:near-duplicates")
            self.assertEqual(snapshot["metadata"]["max_hamming"], 4)
            self.assertEqual(registry["runs"][0]["command"], "seedling-data:near-duplicates")
            self.assertIn(str(snapshot_path), {artifact["path"] for artifact in registry["runs"][0]["artifacts"]})

    def test_duplicate_check_reports_missing_manifest_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "image_manifest.csv"
            missing = root / "missing.png"
            _write_manifest(
                manifest_path,
                [
                    {
                        "image_id": "missing",
                        "file_path": str(missing),
                        "sha256": "abc",
                        "split": "test",
                    }
                ],
            )

            result = find_cross_split_duplicates(manifest_path)

            self.assertFalse(result["ok"])
            self.assertEqual(result["missing_files"], [{"image": "missing", "file_path": str(missing), "split": "test"}])

    def test_duplicate_check_rejects_manifest_without_split_hash_or_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            Image.new("RGB", (8, 8), "green").save(image)
            manifest_path = root / "image_manifest.csv"
            _write_manifest(
                manifest_path,
                [
                    {
                        "image_id": "missing_split_and_hash",
                        "file_path": str(image),
                    },
                    {
                        "image_id": "missing_path",
                        "sha256": _sha256(image),
                        "split": "train",
                    },
                ],
            )

            result = find_cross_split_duplicates(manifest_path)

            self.assertFalse(result["ok"])
            self.assertEqual(
                result["missing_split_rows"],
                [{"image": "missing_split_and_hash", "file_path": str(image), "split": ""}],
            )
            self.assertEqual(
                result["missing_sha256_rows"],
                [{"image": "missing_split_and_hash", "file_path": str(image), "split": ""}],
            )
            self.assertEqual(
                result["missing_file_path_rows"],
                [{"image": "missing_path", "file_path": "", "split": "train"}],
            )


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "file_path", "sha256", "split"])
        writer.writeheader()
        writer.writerows(rows)


def _write_changelog(path: Path, dataset_version: str) -> None:
    path.write_text(
        f"""# Dataset Changelog

## {dataset_version}

- Status: draft
- Date: 2026-06-17
- Author: test
- Source dataset root: dataset
- Ontology version: ontology_v0_1
- Annotation guide version: annotation_guide_v0_1
- Split version: split_v0
- Calibration version: calibration_v0

### Added

- Initial tiny fixture.

### Changed

- None.

### Removed

- None.

### Known Issues

- Fixture only.

### Validation

- python -B -m unittest tests.unit.test_data_registry_annotations
""",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


def _post_action(command_id: str, hours: float, outcome: str) -> PostActionObservation:
    return PostActionObservation(
        command_id=command_id,
        target_id=f"target_{command_id[-3:]}",
        scene_id="scene_001",
        tray_id="tray_001",
        observed_at=f"2026-06-18T{int(hours) % 24:02d}:00:00+00:00",
        hours_after_action=hours,
        outcome=outcome,
        image_ref=f"after_{int(hours)}h.png",
        observer_id="tester",
    )


if __name__ == "__main__":
    unittest.main()
