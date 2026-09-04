"""I17 = S1+S5 symmetric ternary Q/K + H9a compat_qk_product + angular loss.

Stack: install ATLIFTernaryPSN → patch with symmetric_ternary (S1+S5) →
       install bsa_attention (H9a compat_qk_product) → angular loss via h9_losses.
"""
from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

PIN_MEMORY_ANCHOR = """"pin_memory": True,
"""
PIN_MEMORY_PATCH = """"pin_memory": bool(config["loader"].get("pin_memory", True)),
"""

LOAD_MODEL_ANCHOR = """    model = load_model(args.prev_runid, model, device, remap)
"""
LOAD_MODEL_PATCH = """    from models.STSwinNet_SNN.bsa_attention import register_shiftmax_pickle_compat
    register_shiftmax_pickle_compat()
    if bool(config.get("runtime", {}).get("load_full_model", False)) and args.prev_runid and os.path.isfile(args.prev_runid):
        model = torch.load(args.prev_runid, map_location=device, weights_only=False)
        model.to(device)
        print("I17 full model restored from local checkpoint " + args.prev_runid + "\\n")
    else:
        model = load_model(args.prev_runid, model, device, remap)

    from models.STSwinNet_SNN.atlif_ternary_psn import apply_trainable_mode, atlif_ternary_summary, install_atlif_ternary_psn
    installed = install_atlif_ternary_psn(model, config.get("atlif_ternary_psn"))
    if installed:
        print(f"[I17] ATLIFTernaryPSN: {len(installed)} modules")

    # S1+S5: symmetric ternary (neg_thre=thre, out=±thre)
    from neuron_autoresearch.attention.symmetric_ternary_psn import install_symmetric_ternary
    sym_installed = install_symmetric_ternary(model, config)
    if sym_installed:
        print(f"[I17] SymmetricTernary (S1+S5): {len(sym_installed)} modules")

    # BSA attention (compat_qk_product)
    from models.STSwinNet_SNN.bsa_attention import install_shiftmax_attention, shiftmax_attention_summary
    bsa_ok = install_shiftmax_attention(model, config.get("bsa_attention"))
    if bsa_ok:
        print(f"[I17] Shiftmax attention: {len(bsa_ok)} modules")

    print(f"[I17] trainable: {apply_trainable_mode(model, config.get('atlif_ternary_psn'))}")
    print(f"[I17] ATLIF: {atlif_ternary_summary(model)}")
    if bsa_ok:
        print(f"[I17] Attn: {shiftmax_attention_summary(model)}")
"""

LOSS_ANCHOR = """                # print("loss: ", curr_loss.item())

                if np.isnan(curr_loss.item()):
                    raise
"""
LOSS_PATCH = """                from models.STSwinNet_SNN.atlif_ternary_psn import regularize_activity
                penalty = regularize_activity(model, config.get("atlif_ternary_psn"))
                if penalty is not None:
                    curr_loss = curr_loss + penalty / num_acc_steps
                if np.isnan(curr_loss.item()):
                    raise
"""

SCALER_STEP_ANCHOR = """                    scaler.step(optimizer)
                    scaler.update()
"""
SCALER_STEP_PATCH = """                    scaler.step(optimizer)
                    from models.STSwinNet_SNN.atlif_ternary_psn import threshold_update
                    stats = threshold_update(model, optimizer.param_groups[0]["lr"], config.get("atlif_ternary_psn"))
                    if (sample + 1) % 20 == 0:
                        print(f"[I17] step {sample+1}: thr_mean={stats.get('threshold_mean',0):.4f} thr_max={stats.get('threshold_max',0):.4f} act_mean={stats.get('activity_mean',0):.4f}")
                    scaler.update()
"""

OPTIMIZER_STEP_ANCHOR = """                    optimizer.step()

                # zero grad
"""
OPTIMIZER_STEP_PATCH = """                    optimizer.step()
                    from models.STSwinNet_SNN.atlif_ternary_psn import threshold_update
                    threshold_update(model, optimizer.param_groups[0]["lr"], config.get("atlif_ternary_psn"))
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
            s = atlif_ternary_summary(model)
            print(f"[I17] thr={s['threshold_mean']:.3f} act={s['activity_mean']:.3f} pos={s.get('pos_mean',0):.3f} neg={s.get('neg_mean',0):.4f}")
"""

TRAIN_STEP_ANCHOR = """            sample += 1
            train_sample_count += chunk.shape[0]
"""
TRAIN_STEP_PATCH = """            sample += 1
            train_sample_count += chunk.shape[0]
            max_train_steps = int(config.get("runtime", {}).get("max_train_steps", 0) or 0)
            if max_train_steps > 0 and sample >= max_train_steps:
                print(f"[I17] stopping early at max_train_steps={max_train_steps}")
"""

SAVE_ANCHOR = """            should_save_model = epoch_loss < best_loss or epoch == config["loader"]["n_epochs"] - 1
"""
SAVE_PATCH = """            should_save_model = (
                epoch_loss < best_loss or epoch == config["loader"]["n_epochs"] - 1
            )
"""

STATE_SAVE_ANCHOR = """                    state_path = checkpoint_path.replace(".pth", "_state_dict.pth")
                    torch.save(
                        {
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict() if scheduler else None,
                            "epoch": epoch,
                            "scaler": scaler.state_dict() if scaler else None,
                        },
                        state_path,
                    )
                    print(f"Local checkpoint saved to {checkpoint_path}")
                    print(f"Local training state saved to {state_path}")
"""
STATE_SAVE_PATCH = """                    if not bool(config.get("runtime", {}).get("skip_state_save", False)):
                        state_path = checkpoint_path.replace(".pth", "_state_dict.pth")
                        torch.save(
                            {
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict() if scheduler else None,
                                "epoch": epoch,
                                "scaler": scaler.state_dict() if scaler else None,
                            },
                            state_path,
                        )
                        print(f"Local training state saved to {state_path}")
                    print(f"Local checkpoint saved to {checkpoint_path}")
"""

MLFLOW_MODEL_LOGGING_ANCHOR = """                if use_ml_flow and use_mlflow_model_logging:
"""
MLFLOW_MODEL_LOGGING_PATCH = """                if (
                    use_ml_flow
                    and use_mlflow_model_logging
                    and not config.get("runtime", {}).get("use_mlflow_model_logging", True) is False
                ):
"""

LOSS_FUNCTION_ANCHOR = """    # Define the loss function
    loss_function = flow_loss_supervised(config,device)
"""
LOSS_FUNCTION_PATCH = """    # Define the loss function
    loss_function = flow_loss_supervised(config,device)
    from models.STSwinNet_SNN.h9_losses import maybe_replace_flow_loss
    loss_function = maybe_replace_flow_loss(loss_function, config, device)
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
        (SAVE_ANCHOR, SAVE_PATCH),
        (STATE_SAVE_ANCHOR, STATE_SAVE_PATCH),
        (MLFLOW_MODEL_LOGGING_ANCHOR, MLFLOW_MODEL_LOGGING_PATCH),
        (LOSS_FUNCTION_ANCHOR, LOSS_FUNCTION_PATCH),
    ):
        if anchor not in source:
            raise RuntimeError(f"anchor missing: {anchor[:60]!r}")
        source = source.replace(anchor, replacement, 1)
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, extra_args = parser.parse_known_args()

    repo_root = _repo_root()
    baseline_root = repo_root / "third_party" / "SDformerFlow"
    h9_overlay = repo_root / "neuron_experiments" / "H9_bipolar_self_attention" / "overlay"
    baseline_entry = baseline_root / "train_flow_parallel_supervised_SNN.py"

    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(baseline_root))
    sys.path.insert(0, str(h9_overlay))

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
