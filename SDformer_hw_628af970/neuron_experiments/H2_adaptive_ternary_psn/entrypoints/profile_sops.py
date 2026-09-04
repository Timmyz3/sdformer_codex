"""H2 SOP/profile entrypoint.

Wraps tools/profile_sops.py and installs AdaptiveTernaryPSN before checkpoint
loading so H2 checkpoints restore with the correct module structure.
"""

from __future__ import annotations

import sys
from pathlib import Path


INIT_ANCHOR = """    model.to(device)
    model.init_weights()
    if args.checkpoint:
        model = load_model(str(args.checkpoint), model, device, remap=args.remap, test=True)
"""

INIT_PATCH = """    model.to(device)
    model.init_weights()
    if config.get("adaptive_ternary_psn", {}).get("enabled", False):
        from models.STSwinNet_SNN.adaptive_ternary import (
            adaptive_ternary_summary,
            install_adaptive_ternary_qk,
        )
        installed_h2 = install_adaptive_ternary_qk(model, config.get("adaptive_ternary_psn"))
        print(f"[H2] installed AdaptiveTernaryPSN before profile load: {len(installed_h2)} modules")
        print(f"[H2] summary before profile load: {adaptive_ternary_summary(model)}")
    if args.checkpoint:
        model = load_model(str(args.checkpoint), model, device, remap=args.remap, test=True)
        if config.get("adaptive_ternary_psn", {}).get("enabled", False):
            from models.STSwinNet_SNN.adaptive_ternary import adaptive_ternary_summary
            print(f"[H2] summary after profile load: {adaptive_ternary_summary(model)}")
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
        raise RuntimeError("Could not patch tools/profile_sops.py for H2 AdaptiveTernaryPSN loading")
    source = source.replace(INIT_ANCHOR, INIT_PATCH, 1)
    code = compile(source, str(profile_script), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(profile_script)})


if __name__ == "__main__":
    main()
