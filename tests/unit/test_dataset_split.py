from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from seedling_experiments.dataset import make_grouped_split


class DatasetSplitTests(unittest.TestCase):
    def test_grouped_split_uses_image_manifest_group_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            images = source / "images"
            labels = source / "labels"
            output = root / "prepared"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)

            image_groups = {
                "alpha_001.png": "tray_alpha",
                "alpha_002.png": "tray_alpha",
                "beta_001.png": "tray_beta",
                "beta_002.png": "tray_beta",
            }
            for image_name in image_groups:
                Image.new("RGB", (8, 8), "green").save(images / image_name)
                (labels / f"{Path(image_name).stem}.txt").write_text("", encoding="utf-8")

            manifest = root / "image_manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["image_id", "file_path", "group_id"])
                writer.writeheader()
                for image_name, group_id in image_groups.items():
                    writer.writerow(
                        {
                            "image_id": Path(image_name).stem,
                            "file_path": str((images / image_name).resolve()),
                            "group_id": group_id,
                        }
                    )

            summary = make_grouped_split(
                source,
                output,
                train_ratio=0.5,
                val_ratio=0.25,
                test_ratio=0.25,
                seed=1,
                metadata_path=manifest,
            )

            group_splits: dict[str, set[str]] = {}
            with (output / "split_manifest.csv").open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    group_splits.setdefault(row["group"], set()).add(row["split"])

            self.assertEqual(summary["metadata_path"], str(manifest.resolve()))
            self.assertEqual(set(group_splits), {"tray_alpha", "tray_beta"})
            self.assertTrue(all(len(splits) == 1 for splits in group_splits.values()))


if __name__ == "__main__":
    unittest.main()
