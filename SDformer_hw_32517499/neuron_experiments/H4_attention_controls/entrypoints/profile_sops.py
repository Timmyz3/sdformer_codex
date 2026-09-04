"""H4 SOP/profile entrypoint with Q/K attention control installation."""

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
    if args.checkpoint:
        model = load_model(str(args.checkpoint), model, device, remap=args.remap, test=True)
    if config.get("qk_control", {}).get("enabled", False):
        from models.STSwinNet_SNN.qk_control import install_qk_control
        installed_h4 = install_qk_control(model, config.get("qk_control"))
        print(f"[H4] installed Q/K control after profile load: {len(installed_h4)} modules")
        print(f"[H4] targets: {installed_h4[:8]}{' ...' if len(installed_h4) > 8 else ''}")
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
        raise RuntimeError("Could not patch tools/profile_sops.py for H4 Q/K control loading")
    source = source.replace(INIT_ANCHOR, INIT_PATCH, 1)
    code = compile(source, str(profile_script), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(profile_script)})


if __name__ == "__main__":
    main()
