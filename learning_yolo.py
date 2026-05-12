"""Legacy entry point for YOLO training.

Prefer:
    py -m seedling_experiments train --config configs/example_experiment.yaml
"""

import argparse

from seedling_experiments.config import load_config
from seedling_experiments.train import train_yolo_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO from an experiment config.")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON experiment config.")
    args = parser.parse_args()
    train_yolo_from_config(load_config(args.config))


if __name__ == "__main__":
    main()
