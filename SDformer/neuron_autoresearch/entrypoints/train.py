"""Autoresearch training entrypoint — extends H1 source-patching with FSN support.

Supports: HardSparseGate, HardwareSparseNeuron (GTCN), FusedSparseNeuron (FSN).
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# Source-patch anchors and replacements (based on H1 patterns)
# ---------------------------------------------------------------------------

PIN_MEMORY_ANCHOR = """"pin_memory": True,
"""

PIN_MEMORY_PATCH = """"pin_memory": bool(config["loader"].get("pin_memory", True)),
"""

LOAD_MODEL_ANCHOR = """    model = load_model(args.prev_runid, model, device, remap)
"""

LOAD_MODEL_PATCH = """    model = load_model(args.prev_runid, model, device, remap)
    from models.STSwinNet_SNN.sparse_gate import (
        config_from_dict,
        install_sparse_gates,
        install_hw_sparse_gates,
        install_fsn_gates,
        sparse_gate_summary,
    )
    sg_cfg = config_from_dict(config.get("sparse_gate"))
    if sg_cfg.enabled:
        if sg_cfg.use_fsn:
            installed = install_fsn_gates(model, config.get("sparse_gate"))
            print(f"[AR] installed FSN gates ({sg_cfg.stage_selection}): {len(installed)} gates, levels={sg_cfg.fsn_num_levels}, signed={sg_cfg.fsn_signed}")
        elif sg_cfg.use_hardware_neuron:
            installed = install_hw_sparse_gates(model, config.get("sparse_gate"))
            print(f"[AR] installed HW sparse gates ({sg_cfg.stage_selection}): {len(installed)} gates")
        else:
            installed = install_sparse_gates(model, config.get("sparse_gate"))
            print(f"[AR] installed sparse gates: {installed}")
        print(f"[AR] sparse gate summary after install: {sparse_gate_summary(model)}")
"""

LOSS_ANCHOR = """                # print("loss: ", curr_loss.item())

                if np.isnan(curr_loss.item()):
                    raise
"""

LOSS_PATCH = """                from models.STSwinNet_SNN.sparse_gate import (
                    sparse_gate_regularization,
                    threshold_regularization_loss,
                )
                gate_penalty = sparse_gate_regularization(model, config.get("sparse_gate"))
                if gate_penalty is not None:
                    curr_loss = curr_loss + gate_penalty / num_acc_steps
                threshold_penalty = threshold_regularization_loss(model, config.get("sparse_gate"))
                if threshold_penalty is not None:
                    curr_loss = curr_loss + threshold_penalty / num_acc_steps
                # print("loss: ", curr_loss.item())

                if np.isnan(curr_loss.item()):
                    raise
"""

EPOCH_STATS_ANCHOR = """        print(
            f"Epoch stats: lr={optimizer.param_groups[0]['lr']:.6g}, "
            f"epoch_time_sec={epoch_time_sec:.2f}, train_step_time_sec={train_step_time_sec:.4f}, "
            f"train_samples_per_sec={train_samples_per_sec:.4f}, max_gpu_mem_gib={max_gpu_mem_gib:.3f}"
        )
"""

EPOCH_STATS_PATCH = """        print(
            f"Epoch stats: lr={optimizer.param_groups[0]['lr']:.6g}, "
            f"epoch_time_sec={epoch_time_sec:.2f}, train_step_time_sec={train_step_time_sec:.4f}, "
            f"train_samples_per_sec={train_samples_per_sec:.4f}, max_gpu_mem_gib={max_gpu_mem_gib:.3f}"
        )
        if config.get("sparse_gate", {}).get("enabled", False):
            from models.STSwinNet_SNN.sparse_gate import sparse_gate_summary
            print(f"[AR] sparse gate summary: {sparse_gate_summary(model)}")
"""

MODEL_TRAIN_ANCHOR = """        model.train()
"""

MODEL_TRAIN_PATCH = """        model.train()
        if config.get("sparse_gate", {}).get("freeze_backbone", False):
            model.eval()
"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _install_optional_mlflow_stub() -> None:
    disabled = os.getenv("SDFORMER_USE_MLFLOW", "1").lower() in {"0", "false", "no"}
    if not disabled:
        return
    try:
        __import__("mlflow")
    except ModuleNotFoundError:
        sys.modules["mlflow"] = types.ModuleType("mlflow")


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
        (PIN_MEMORY_ANCHOR, PIN_MEMORY_PATCH),
        (LOAD_MODEL_ANCHOR, LOAD_MODEL_PATCH),
        (MODEL_TRAIN_ANCHOR, MODEL_TRAIN_PATCH),
        (LOSS_ANCHOR, LOSS_PATCH),
        (EPOCH_STATS_ANCHOR, EPOCH_STATS_PATCH),
    ):
        if anchor not in source:
            raise RuntimeError(
                f"Could not patch {baseline_entry}: missing anchor {anchor[:60]!r}"
            )
        source = source.replace(anchor, replacement, 1)
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, extra_args = parser.parse_known_args()

    repo_root = _repo_root()
    baseline_root = repo_root / "third_party" / "SDformerFlow"

    # Use H1 overlay (contains sparse_gate.py with all three install functions)
    h1_overlay = repo_root / "neuron_experiments" / "H1_hw_sparse" / "overlay"
    baseline_entry = baseline_root / "train_flow_parallel_supervised_SNN.py"

    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(baseline_root))
    sys.path.insert(0, str(h1_overlay))

    sys.argv = [
        str(baseline_entry),
        "--config",
        str(Path(args.config).resolve()),
        *_absolutize_path_args(extra_args),
    ]

    _install_optional_mlflow_stub()
    os.chdir(baseline_root)
    source = _patch_source(baseline_entry.read_text(), baseline_entry)
    code = compile(source, str(baseline_entry), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(baseline_entry)})


if __name__ == "__main__":
    main()
