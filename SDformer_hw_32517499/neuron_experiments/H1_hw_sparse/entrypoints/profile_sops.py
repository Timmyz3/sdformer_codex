"""H1 profile entrypoint — installs sparse gates before checkpoint load."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


LOAD_CHECKPOINT_ANCHOR = "    model = load_model(args.prev_runid, model, device, remap)"
PROFILE_LOAD_PATCH = """    from models.STSwinNet_SNN.sparse_gate import (
        config_from_dict,
        install_sparse_gates,
        install_hw_sparse_gates,
    )
    sg_cfg = config_from_dict(config.get("sparse_gate"))
    if sg_cfg.enabled:
        if sg_cfg.use_hardware_neuron:
            install_hw_sparse_gates(model, config.get("sparse_gate"))
        else:
            install_sparse_gates(model, config.get("sparse_gate"))
    model = load_model(args.prev_runid, model, device, remap)
"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--split", default="valid")
    parser.add_argument("--num-samples", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args, extra_args = parser.parse_known_args()

    experiment_root = Path(__file__).resolve().parents[1]
    repo_root = _repo_root()
    baseline_root = repo_root / "third_party" / "SDformerFlow"
    overlay_root = experiment_root / "overlay"
    profile_script = repo_root / "tools" / "profile_sops.py"

    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(baseline_root))
    sys.path.insert(0, str(overlay_root))
    sys.path.insert(0, str(repo_root / "tools"))

    os.chdir(baseline_root)

    # Read and optionally patch the profile script
    source = profile_script.read_text()
    if LOAD_CHECKPOINT_ANCHOR in source:
        source = source.replace(LOAD_CHECKPOINT_ANCHOR, PROFILE_LOAD_PATCH, 1)

    sys.argv = [
        str(profile_script),
        "--config", str(Path(args.config).resolve()),
    ]
    if args.checkpoint:
        sys.argv += ["--checkpoint", str(Path(args.checkpoint).resolve())]
    if args.output_dir:
        sys.argv += ["--output-dir", str(Path(args.output_dir).resolve())]
    sys.argv += ["--split", args.split]
    sys.argv += ["--num-samples", str(args.num_samples)]
    sys.argv += ["--batch-size", str(args.batch_size)]
    sys.argv += ["--num-workers", str(args.num_workers)]
    sys.argv += ["--device", args.device]

    code = compile(source, str(profile_script), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(profile_script)})


if __name__ == "__main__":
    main()
