"""H7 SOP/profile entrypoint with ATLIFTernaryPSN installation before checkpoint load."""

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
    if config.get("atlif_ternary_psn", {}).get("enabled", False):
        from models.STSwinNet_SNN.atlif_ternary_psn import atlif_ternary_summary, install_atlif_ternary_psn
        installed_h6 = install_atlif_ternary_psn(model, config.get("atlif_ternary_psn"))
        print(f"[H7] installed ATLIFTernaryPSN before profile load: {len(installed_h6)} modules")
        print(f"[H7] summary before profile load: {atlif_ternary_summary(model)}")
    if args.checkpoint:
        model = load_model(str(args.checkpoint), model, device, remap=args.remap, test=True)
        if config.get("atlif_ternary_psn", {}).get("enabled", False):
            from models.STSwinNet_SNN.atlif_ternary_psn import atlif_ternary_summary
            print(f"[H7] summary after profile load: {atlif_ternary_summary(model)}")
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
        raise RuntimeError("Could not patch tools/profile_sops.py for H7 ATLIFTernaryPSN loading")
    source = source.replace(INIT_ANCHOR, INIT_PATCH, 1)
    code = compile(source, str(profile_script), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(profile_script)})


if __name__ == "__main__":
    main()
