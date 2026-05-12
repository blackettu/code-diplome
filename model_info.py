"""Print YOLO model information and validation metrics."""

import argparse

from seedling_experiments.config import load_config
from seedling_experiments.train import validate_yolo_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a YOLO model and print metrics.")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON experiment config.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = parser.parse_args()
    metrics = validate_yolo_from_config(load_config(args.config), split=args.split)
    print(metrics)


if __name__ == "__main__":
    main()
