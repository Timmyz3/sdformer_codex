"""Template evaluation entrypoint for an SDFormerFlow neuron experiment."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, extra_args = parser.parse_known_args()

    experiment_root = Path(__file__).resolve().parents[1]
    repo_root = experiment_root.parents[1]
    baseline_root = repo_root / "third_party" / "SDformerFlow"
    overlay_root = experiment_root / "overlay"
    baseline_entry = baseline_root / "eval_DSEC_flow_SNN.py"

    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(baseline_root))
    sys.path.insert(0, str(overlay_root))
    sys.argv = [
        str(baseline_entry),
        "--config",
        str(Path(args.config).resolve()),
        *extra_args,
    ]

    os.chdir(baseline_root)
    runpy.run_path(str(baseline_entry), run_name="__main__")


if __name__ == "__main__":
    main()
