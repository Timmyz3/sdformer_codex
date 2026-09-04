"""Token mixer training entrypoint for SDFormerFlow.

Replaces QKV self-attention with token mixing (Conv/MLP/Pool/Identity).

Usage:
    python -m autoresearch_sparsity.entrypoints.train_tokenmix \
        --config configs/train_tokenmix_conv.yml \
        --prev_runid experiments/checkpoints/.../checkpoint_epoch59.pth
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# ── Source patch: install token mixers BEFORE model construction ──────

MODEL_BUILD_ANCHOR = """    if config["swin_transformer"]["use_arc"][0]:
        model = eval(config["model"]["name"])(config["model"].copy(), config["swin_transformer"].copy())
    else:
        model = eval(config["model"]["name"])(config["model"].copy())"""

MODEL_BUILD_PATCH = """    from token_mixers import install_token_mixers
    install_token_mixers(config.get("token_mixer", {}).get("type", "conv"))
    if config["swin_transformer"]["use_arc"][0]:
        model = eval(config["model"]["name"])(config["model"].copy(), config["swin_transformer"].copy())
    else:
        model = eval(config["model"]["name"])(config["model"].copy())"""


# ── Paths ────────────────────────────────────────────────────────────────

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _absolutize_path_args(extra_args: list[str]) -> list[str]:
    path_flags = {"--save_path", "--resume", "--finetune", "--path_mlflow", "--prev_runid"}
    normalized = list(extra_args)
    index = 0
    while index < len(normalized):
        item = normalized[index]
        if item in path_flags and index + 1 < len(normalized):
            normalized[index + 1] = str(Path(normalized[index + 1]).resolve())
            index += 2
            continue
        matched = next((flag for flag in path_flags if item.startswith(f"{flag}=")), None)
        if matched is not None:
            value = item.split("=", 1)[1]
            normalized[index] = f"{matched}={Path(value).resolve()}"
        index += 1
    return normalized


def _patch_source(source: str, baseline_entry: Path) -> str:
    if MODEL_BUILD_ANCHOR not in source:
        raise RuntimeError(f"Anchor not found in {baseline_entry}")
    return source.replace(MODEL_BUILD_ANCHOR, MODEL_BUILD_PATCH, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, extra_args = parser.parse_known_args()

    repo_root = _repo_root()
    baseline_root = repo_root / "third_party" / "SDformerFlow"
    sparsity_overlay = repo_root / "autoresearch_sparsity" / "overlay"
    baseline_entry = baseline_root / "train_flow_parallel_supervised_SNN.py"

    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(baseline_root))
    sys.path.insert(0, str(sparsity_overlay))

    sys.argv = [
        str(baseline_entry),
        "--config",
        str(Path(args.config).resolve()),
        *_absolutize_path_args(extra_args),
    ]

    # Disable MLflow if requested
    disabled = os.getenv("SDFORMER_USE_MLFLOW", "1").lower() in {"0", "false", "no"}
    if disabled:
        import types
        try:
            __import__("mlflow")
        except ModuleNotFoundError:
            sys.modules["mlflow"] = types.ModuleType("mlflow")

    os.chdir(baseline_root)
    source = _patch_source(baseline_entry.read_text(), baseline_entry)
    code = compile(source, str(baseline_entry), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(baseline_entry)})


if __name__ == "__main__":
    main()
