"""G1 SOP/profile entrypoint.

This wraps tools/profile_sops.py and installs sparse gates before checkpoint
loading so G1 checkpoints restore with the correct module structure.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


INIT_ANCHOR = """    model.to(device)
    model.init_weights()
    if args.checkpoint:
        model = load_model(str(args.checkpoint), model, device, remap=args.remap, test=True)
"""

INIT_PATCH = """    model.to(device)
    model.init_weights()
    if config.get("sparse_gate", {}).get("enabled", False):
        from models.STSwinNet_SNN.sparse_gate import install_sparse_gates, sparse_gate_summary
        installed_sparse_gates = install_sparse_gates(model, config.get("sparse_gate"))
        print(f"[G1] installed sparse gates before profile load: {installed_sparse_gates}")
        print(f"[G1] sparse gate summary before profile load: {sparse_gate_summary(model)}")
    if args.checkpoint:
        model = load_model(str(args.checkpoint), model, device, remap=args.remap, test=True)
        if config.get("sparse_gate", {}).get("enabled", False):
            from models.STSwinNet_SNN.sparse_gate import sparse_gate_summary
            print(f"[G1] sparse gate summary after profile load: {sparse_gate_summary(model)}")
"""


def main() -> None:
    experiment_root = Path(__file__).resolve().parents[1]
    repo_root = experiment_root.parents[1]
    baseline_root = repo_root / "third_party" / "SDformerFlow"
    overlay_root = experiment_root / "overlay"
    profile_script = repo_root / "tools" / "profile_sops.py"

    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(baseline_root))
    sys.path.insert(0, str(overlay_root))
    source = profile_script.read_text()
    if INIT_ANCHOR not in source:
        raise RuntimeError("Could not patch tools/profile_sops.py for G1 sparse gate loading")
    source = source.replace(INIT_ANCHOR, INIT_PATCH, 1)
    code = compile(source, str(profile_script), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(profile_script)})


if __name__ == "__main__":
    main()
