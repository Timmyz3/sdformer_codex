"""Create reproducibility configs after the final candidate is frozen."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import yaml


DEFAULT_SEEDS = (6701, 6702, 6703)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--seed", action="append", type=int, default=[])
    args = parser.parse_args()

    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8")) or {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seed or DEFAULT_SEEDS:
        config = deepcopy(base)
        name = f"{args.prefix}_seed{seed}"
        config["experiment"] = name
        config.setdefault("runtime", {}).update({"seed": seed, "deterministic": True})
        config["note"] = (
            f"Reviewer reproducibility repeat seed={seed}; architecture and optimization "
            f"are inherited unchanged from {args.base_config}."
        )
        path = args.output_dir / f"{name}.yml"
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        rows.append({"seed": seed, "config": str(path), "status": "generated_not_run"})
        print(path)

    manifest = args.output_dir / f"{args.prefix}_seed_manifest.json"
    manifest.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
