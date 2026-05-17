from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffusion_training import TrainConfig, train_diffusion_model


def parse_args() -> argparse.Namespace:
    defaults = TrainConfig(data_source="")
    parser = argparse.ArgumentParser(description="Train an unconditional DDPM prior on PDE snapshot NPZ data.")
    for field, value in asdict(defaults).items():
        arg = "--" + field.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(arg, action=argparse.BooleanOptionalAction, default=value)
        elif isinstance(value, int):
            parser.add_argument(arg, type=int, default=value)
        elif isinstance(value, float):
            parser.add_argument(arg, type=float, default=value)
        else:
            parser.add_argument(arg, type=str, default=value)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_source:
        raise SystemExit("--data-source is required")
    config = TrainConfig(**vars(args))
    best_path = train_diffusion_model(config)
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
