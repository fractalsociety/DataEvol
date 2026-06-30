from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import BASE_MODEL, EXPERTS, prepare_local_adapter_training, run_local_adapter_training_from_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or run DataEvol local expert adapter training.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Write datasets, manifest, and train_adapters.py.")
    prepare.add_argument("--output", default=".dataevol/local_models")
    prepare.add_argument("--base-model", default=BASE_MODEL)
    prepare.add_argument("--expert", action="append", choices=EXPERTS)
    prepare.add_argument("--count", type=int, default=24)
    prepare.add_argument("--iters", type=int, default=2)

    train = subparsers.add_parser("train", help="Train adapters from a manifest.")
    train.add_argument("--manifest", default=".dataevol/local_models/adapter_training_manifest.json")
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--timeout", type=int, default=1800)

    args = parser.parse_args()
    if args.command == "prepare":
        plan = prepare_local_adapter_training(
            args.output,
            base_model=args.base_model,
            experts=tuple(args.expert or EXPERTS),
            count=args.count,
            iters=args.iters,
        )
        result = {"ok": True, "status": "prepared", **plan.to_dict()}
    else:
        result = run_local_adapter_training_from_manifest(
            Path(args.manifest),
            execute=not args.dry_run,
            timeout=args.timeout,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
