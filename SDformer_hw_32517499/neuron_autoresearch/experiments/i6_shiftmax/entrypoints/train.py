"""I6 = H6a + Shiftmax. Reuses H6 patches + inserts shiftmax in attention."""
from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

# --- Identical to H6 train.py ---
PIN_MEMORY_ANCHOR = """"pin_memory": True,
"""
PIN_MEMORY_PATCH = """"pin_memory": bool(config["loader"].get("pin_memory", True)),
"""

LOAD_MODEL_ANCHOR = """    model = load_model(args.prev_runid, model, device, remap)
"""
LOAD_MODEL_PATCH = """    model = load_model(args.prev_runid, model, device, remap)
    from models.STSwinNet_SNN.atlif_ternary_psn import apply_trainable_mode, atlif_ternary_summary, install_atlif_ternary_psn
    installed_h6 = install_atlif_ternary_psn(model, config.get("atlif_ternary_psn"))
    if installed_h6:
        print(f"[I6] installed ATLIFTernaryPSN: {len(installed_h6)} modules")
        print(f"[I6] targets: {installed_h6[:8]}{' ...' if len(installed_h6) > 8 else ''}")
        print(f"[I6] trainable: {apply_trainable_mode(model, config.get('atlif_ternary_psn'))}")
        print(f"[I6] summary after install: {atlif_ternary_summary(model)}")
    if config.get("shiftmax", {}).get("enabled", False):
        print(f"[I6] Shiftmax enabled")
"""

LOSS_ANCHOR = """                # print("loss: ", curr_loss.item())

                if np.isnan(curr_loss.item()):
                    raise
"""
LOSS_PATCH = """                from models.STSwinNet_SNN.atlif_ternary_psn import regularize_activity
                h6_penalty = regularize_activity(model, config.get("atlif_ternary_psn"))
                if h6_penalty is not None:
                    curr_loss = curr_loss + h6_penalty / num_acc_steps
                # print("loss: ", curr_loss.item())

                if np.isnan(curr_loss.item()):
                    raise
"""

SCALER_STEP_ANCHOR = """                    scaler.step(optimizer)
                    scaler.update()
"""
SCALER_STEP_PATCH = """                    scaler.step(optimizer)
                    from models.STSwinNet_SNN.atlif_ternary_psn import threshold_update
                    h6_update_stats = threshold_update(model, optimizer.param_groups[0]["lr"], config.get("atlif_ternary_psn"))
                    h6_log_interval = int(config.get("atlif_ternary_psn", {}).get("log_interval_steps", 0) or 0)
                    if h6_log_interval > 0 and (sample + 1) % h6_log_interval == 0:
                        print(f"[I6] step {sample + 1} update: {h6_update_stats}")
                    scaler.update()
"""

OPTIMIZER_STEP_ANCHOR = """                    optimizer.step()

                # zero grad
"""
OPTIMIZER_STEP_PATCH = """                    optimizer.step()
                    from models.STSwinNet_SNN.atlif_ternary_psn import threshold_update
                    h6_update_stats = threshold_update(model, optimizer.param_groups[0]["lr"], config.get("atlif_ternary_psn"))
                    h6_log_interval = int(config.get("atlif_ternary_psn", {}).get("log_interval_steps", 0) or 0)
                    if h6_log_interval > 0 and (sample + 1) % h6_log_interval == 0:
                        print(f"[I6] step {sample + 1} update: {h6_update_stats}")

                # zero grad
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
        if config.get("atlif_ternary_psn", {}).get("enabled", False):
            from models.STSwinNet_SNN.atlif_ternary_psn import atlif_ternary_summary
            print(f"[I6] ATLIFTernaryPSN summary: {atlif_ternary_summary(model)}")
"""

TRAIN_STEP_ANCHOR = """            sample += 1
            train_sample_count += chunk.shape[0]
"""
TRAIN_STEP_PATCH = """            sample += 1
            train_sample_count += chunk.shape[0]
            max_train_steps = int(config.get("runtime", {}).get("max_train_steps", 0) or 0)
            if max_train_steps > 0 and sample >= max_train_steps:
                print(f"[I6] stopping train epoch early at max_train_steps={max_train_steps}")
"""

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


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
        (LOSS_ANCHOR, LOSS_PATCH),
        (SCALER_STEP_ANCHOR, SCALER_STEP_PATCH),
        (OPTIMIZER_STEP_ANCHOR, OPTIMIZER_STEP_PATCH),
        (EPOCH_STATS_ANCHOR, EPOCH_STATS_PATCH),
        (TRAIN_STEP_ANCHOR, TRAIN_STEP_PATCH),
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
    i6_overlay = repo_root / "neuron_autoresearch" / "experiments" / "i6_shiftmax" / "overlay"
    h6_overlay = repo_root / "neuron_experiments" / "H6_attention_ternary_binary_highsops" / "overlay"
    baseline_entry = baseline_root / "train_flow_parallel_supervised_SNN.py"

    # Install shiftmax import hook before any model imports
    from pathlib import Path as _Path
    import yaml as _yaml
    with open(_Path(args.config).resolve()) as _f:
        _cfg = _yaml.safe_load(_f)
    if _cfg.get("shiftmax", {}).get("enabled", False):
        import importlib.util as _iu
        _spec = _iu.spec_from_file_location(
            "shiftmax_hook", str(i6_overlay / "shiftmax_hook.py")
        )
        _mod = _iu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _mod.install_shiftmax_hook(repo_root)
        print("[I6] shiftmax import hook installed")

    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(baseline_root))
    sys.path.insert(0, str(i6_overlay))
    sys.path.insert(0, str(h6_overlay))

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
