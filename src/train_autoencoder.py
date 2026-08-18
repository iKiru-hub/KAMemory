"""Train the paper's spatial+sensory autoencoder from one settings object.

Examples, from the repository root:

    python src/train_autoencoder.py
    python src/train_autoencoder.py --epochs 400 --lr 0.0005 --name ae_factorial_01
    python src/train_autoencoder.py --config my_settings.json
    python src/train_autoencoder.py --settings-json '{"seed": 9, "epochs": 20}' --no-save

The reusable Python API is ``run_autoencoder_experiment(settings_dict)`` in
``kamemory.autoencoder_experiment``.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from kamemory.autoencoder_experiment import (
    load_settings_json,
    run_autoencoder_experiment,
)
from kamemory.io import PATHS


DEFAULT_CONFIG = PATHS.configs / "autoencoder_factorial.json"


def _merge(base: dict, update: dict) -> dict:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--settings-json",
        default=None,
        help="inline JSON object merged over --config",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--name", help="checkpoint directory name")
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def settings_from_args(args: argparse.Namespace) -> dict:
    settings = load_settings_json(args.config)
    if args.settings_json:
        inline = json.loads(args.settings_json)
        if not isinstance(inline, dict):
            raise TypeError("--settings-json must decode to an object")
        settings = _merge(settings, inline)
    overrides: dict = {"training": {}, "save": {}}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.epochs is not None:
        overrides["training"]["epochs"] = args.epochs
    if args.lr is not None:
        overrides["training"]["learning_rate"] = args.lr
    if args.batch_size is not None:
        overrides["training"]["batch_size"] = args.batch_size
    if args.device is not None:
        overrides["device"] = args.device
    if args.name is not None:
        overrides["save"]["name"] = args.name
    if args.output_directory is not None:
        overrides["save"]["directory"] = str(args.output_directory)
    if args.no_save:
        overrides["save"]["enabled"] = False
    return _merge(settings, overrides)


def main() -> int:
    settings = settings_from_args(parse_args())
    result = run_autoencoder_experiment(settings)
    report = result["report"]
    test = report["metrics"]["trained"]["test"]
    reference = report["metrics"].get("reference_checkpoint_test")
    print("\nfinal held-out metrics")
    print(json.dumps(test, indent=2, sort_keys=True))
    if reference is not None:
        print("\nreference checkpoint held-out metrics")
        print(json.dumps(reference, indent=2, sort_keys=True))
    if result["session_path"] is not None:
        print(f"\nsaved checkpoint: {result['session_path']}")
        print(
            "reload with: "
            f"load_autoencoder_session({str(result['session_path'])!r})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
