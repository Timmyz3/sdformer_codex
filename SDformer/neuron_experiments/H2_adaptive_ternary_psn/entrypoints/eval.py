"""H2 evaluation entrypoint that installs AdaptiveTernaryPSN before loading."""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path


LOAD_MODEL_ANCHOR = """    model_source = args.checkpoint if args.checkpoint else args.runid
    model = load_model(model_source, model, device, remap = remap, test = True) # delete the relative positioning bias and index
"""

LOAD_MODEL_PATCH = """    from models.STSwinNet_SNN.adaptive_ternary import (
        adaptive_ternary_summary,
        install_adaptive_ternary_qk,
    )
    installed_h2 = install_adaptive_ternary_qk(model, config.get("adaptive_ternary_psn"))
    if installed_h2:
        print(f"[H2] installed AdaptiveTernaryPSN before load: {len(installed_h2)} modules")
    model_source = args.checkpoint if args.checkpoint else args.runid
    model = load_model(model_source, model, device, remap = remap, test = True) # delete the relative positioning bias and index
    if installed_h2:
        print(f"[H2] summary after load: {adaptive_ternary_summary(model)}")
"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _install_optional_mlflow_stub() -> None:
    disabled = os.getenv("SDFORMER_USE_MLFLOW", "1").lower() in {"0", "false", "no"}
    if not disabled:
        return
    try:
        __import__("mlflow")
    except ModuleNotFoundError:
        sys.modules["mlflow"] = types.ModuleType("mlflow")


def _patch_source(source: str, baseline_entry: Path) -> str:
    if LOAD_MODEL_ANCHOR not in source:
        raise RuntimeError(f"Could not patch {baseline_entry}: missing load_model anchor")
    return source.replace(LOAD_MODEL_ANCHOR, LOAD_MODEL_PATCH, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, extra_args = parser.parse_known_args()

    experiment_root = Path(__file__).resolve().parents[1]
    repo_root = _repo_root()
    baseline_root = repo_root / "third_party" / "SDformerFlow"
    overlay_root = experiment_root / "overlay"
    baseline_entry = baseline_root / "eval_DSEC_flow_SNN.py"

    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(baseline_root))
    sys.path.insert(0, str(overlay_root))
    sys.argv = [str(baseline_entry), "--config", str(Path(args.config).resolve()), *extra_args]

    _install_optional_mlflow_stub()
    os.chdir(baseline_root)
    source = _patch_source(baseline_entry.read_text(), baseline_entry)
    code = compile(source, str(baseline_entry), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(baseline_entry)})


if __name__ == "__main__":
    main()

