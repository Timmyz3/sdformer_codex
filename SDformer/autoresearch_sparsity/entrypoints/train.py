"""Sparsity-aware training entrypoint for SDFormerFlow.

Patches the upstream training script to insert sparsity preprocessing
(timestep budget, token pruning, window pruning) into the data pipeline.

Usage:
    python -m autoresearch_sparsity.entrypoints.train \
        --config autoresearch_sparsity/configs/train_ts_token.yml \
        --prev_runid experiments/checkpoints/.../checkpoint_epoch59.pth
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# ── Patch anchors ────────────────────────────────────────────────────────
# These must exactly match the upstream source at:
# third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py

MODEL_FORWARD_ANCHOR = """                pred_list = model(chunk.to(device))
                pred = pred_list["flow"]"""

MODEL_FORWARD_PATCH = """                # ── sparsity preprocessing (autoresearch_sparsity) ──
                if config.get("sparsity", {}).get("enabled", False):
                    from sparse_preprocess import build_sparsity_pipeline
                    _sp_pipe = globals().get("_ar_sparse_pipeline")
                    if _sp_pipe is None:
                        _sp_pipe = build_sparsity_pipeline(config)
                        globals()["_ar_sparse_pipeline"] = _sp_pipe
                    if _sp_pipe is not None:
                        _sp_pipe.train(True)
                        chunk, _sp_stats = _sp_pipe(chunk)
                pred_list = model(chunk.to(device))
                pred = pred_list["flow"]"""

VALID_FORWARD_ANCHOR = """                        pred_list = model(chunk.to(device))
                        pred = pred_list["flow"][-1]"""

VALID_FORWARD_PATCH = """                        # ── sparsity preprocessing (autoresearch_sparsity, validation) ──
                        if config.get("sparsity", {}).get("enabled", False):
                            from sparse_preprocess import build_sparsity_pipeline
                            _sp_pipe = globals().get("_ar_sparse_pipeline")
                            if _sp_pipe is None:
                                _sp_pipe = build_sparsity_pipeline(config)
                                globals()["_ar_sparse_pipeline"] = _sp_pipe
                            if _sp_pipe is not None:
                                _sp_pipe.train(False)
                                chunk, _sp_stats = _sp_pipe(chunk)
                        pred_list = model(chunk.to(device))
                        pred = pred_list["flow"][-1]"""

VALIDATION_GATE_ANCHOR = """        if epoch % config["test"]["n_valid"] == 0:"""

VALIDATION_GATE_PATCH = """        if (not config.get("runtime", {}).get("skip_train_validation", False)) and epoch % config["test"]["n_valid"] == 0:"""

TRAIN_STEP_ANCHOR = """            sample += 1
            train_sample_count += chunk.shape[0]"""

TRAIN_STEP_PATCH = """            sample += 1
            train_sample_count += chunk.shape[0]
            _max_train_steps = config.get("runtime", {}).get("max_train_steps", None)
            if _max_train_steps is not None and sample >= int(_max_train_steps):
                print(f"[SPARSE] stopping train epoch early at max_train_steps={_max_train_steps}")
                break"""

EPOCH_STATS_ANCHOR = """        print(
            f"Epoch stats: lr={optimizer.param_groups[0]['lr']:.6g}, "
            f"epoch_time_sec={epoch_time_sec:.2f}, train_step_time_sec={train_step_time_sec:.4f}, "
            f"train_samples_per_sec={train_samples_per_sec:.4f}, max_gpu_mem_gib={max_gpu_mem_gib:.3f}"
        )"""

EPOCH_STATS_PATCH = """        print(
            f"Epoch stats: lr={optimizer.param_groups[0]['lr']:.6g}, "
            f"epoch_time_sec={epoch_time_sec:.2f}, train_step_time_sec={train_step_time_sec:.4f}, "
            f"train_samples_per_sec={train_samples_per_sec:.4f}, max_gpu_mem_gib={max_gpu_mem_gib:.3f}"
        )
        if config.get("sparsity", {}).get("enabled", False):
            print(f"[SPARSE] pipeline active: {config['sparsity']}")"""


# ── Path resolution ──────────────────────────────────────────────────────

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
    for anchor, replacement in (
        (MODEL_FORWARD_ANCHOR, MODEL_FORWARD_PATCH),
        (VALID_FORWARD_ANCHOR, VALID_FORWARD_PATCH),
        (VALIDATION_GATE_ANCHOR, VALIDATION_GATE_PATCH),
        (TRAIN_STEP_ANCHOR, TRAIN_STEP_PATCH),
        (EPOCH_STATS_ANCHOR, EPOCH_STATS_PATCH),
    ):
        if anchor not in source:
            raise RuntimeError(
                f"Could not patch {baseline_entry}: missing anchor {anchor[:80]!r}"
            )
        source = source.replace(anchor, replacement, 1)
    return source


# ── Main ──────────────────────────────────────────────────────────────────

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

    # Local experiment folders need deterministic checkpoint files; MLflow model
    # logging diverts saves into mlruns and makes rapid screening unable to
    # profile the just-trained checkpoint.
    force_local_save = os.getenv("SPARSE_FORCE_LOCAL_SAVE", "1").lower() not in {"0", "false", "no"}
    if force_local_save:
        os.environ.setdefault("SDFORMER_USE_MLFLOW", "0")
        os.environ.setdefault("SDFORMER_MLFLOW_MODEL_LOGGING", "0")
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
