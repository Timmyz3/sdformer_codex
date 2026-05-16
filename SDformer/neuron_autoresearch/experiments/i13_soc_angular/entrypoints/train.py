"""I13 = H9h base + T5 SOC attention + angular loss + progressive schedule.

SOC (Sign-Only Consensus): decouples theta (sparsity) from sign (attention).
Pure popcount-based token gating — zero exponentiation, zero multiplication.
"""
from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

# --- Source patches (H9-compatible) ---

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
        print("I13 full model restored from local checkpoint " + args.prev_runid + "\\n")
    else:
        model = load_model(args.prev_runid, model, device, remap)
    from models.STSwinNet_SNN.atlif_ternary_psn import apply_trainable_mode, atlif_ternary_summary, install_atlif_ternary_psn
    from neuron_autoresearch.attention.soc_attention import install_soc_attention, soc_attention_summary
    installed_h9 = install_atlif_ternary_psn(model, config.get("atlif_ternary_psn"))
    if installed_h9:
        print(f"[I13] installed ATLIFTernaryPSN: {len(installed_h9)} modules")
    installed_soc = install_soc_attention(model, config.get("soc_attention", {}).get("enabled", True))
    if installed_soc:
        print(f"[I13] installed SOC attention: {len(installed_soc)} modules")
    if installed_h9 or installed_soc:
        print(f"[I13] trainable: {apply_trainable_mode(model, config.get('atlif_ternary_psn'))}")
        print(f"[I13] ATLIF summary: {atlif_ternary_summary(model)}")
        print(f"[I13] SOC summary: {soc_attention_summary(model)}")
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
                    h8_update_stats = threshold_update(model, optimizer.param_groups[0]["lr"], config.get("atlif_ternary_psn"))
                    log_interval = int(config.get("atlif_ternary_psn", {}).get("log_interval_steps", 0) or 0)
                    if log_interval > 0 and (sample + 1) % log_interval == 0:
                        print(f"[I13] step {sample + 1} update: {h8_update_stats}")
                    scaler.update()
"""

OPTIMIZER_STEP_ANCHOR = """                    optimizer.step()

                # zero grad
"""
OPTIMIZER_STEP_PATCH = """                    optimizer.step()
                    from models.STSwinNet_SNN.atlif_ternary_psn import threshold_update
                    h8_update_stats = threshold_update(model, optimizer.param_groups[0]["lr"], config.get("atlif_ternary_psn"))
                    log_interval = int(config.get("atlif_ternary_psn", {}).get("log_interval_steps", 0) or 0)
                    if log_interval > 0 and (sample + 1) % log_interval == 0:
                        print(f"[I13] step {sample + 1} update: {h8_update_stats}")

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
            from neuron_autoresearch.attention.soc_attention import soc_attention_summary
            print(f"[I13] ATLIF: {atlif_ternary_summary(model)}  SOC: {soc_attention_summary(model)}")
"""

TRAIN_STEP_ANCHOR = """            sample += 1
            train_sample_count += chunk.shape[0]
"""
TRAIN_STEP_PATCH = """            sample += 1
            train_sample_count += chunk.shape[0]
            max_train_steps = int(config.get("runtime", {}).get("max_train_steps", 0) or 0)
            if max_train_steps > 0 and sample >= max_train_steps:
                print(f"[I13] stopping early at max_train_steps={max_train_steps}")
"""

SAVE_ANCHOR = """            should_save_model = epoch_loss < best_loss or epoch == config["loader"]["n_epochs"] - 1
"""
SAVE_PATCH = """            should_save_model = (
                epoch_loss < best_loss or epoch == config["loader"]["n_epochs"] - 1
            )
"""

MLFLOW_MODEL_LOGGING_ANCHOR = """                if use_ml_flow and use_mlflow_model_logging:
"""
MLFLOW_MODEL_LOGGING_PATCH = """                if (
                    use_ml_flow
                    and use_mlflow_model_logging
                    and not config.get("runtime", {}).get("use_mlflow_model_logging", True) is False
                ):
"""

STATE_SAVE_ANCHOR = """                    state_path = checkpoint_path.replace(".pth", "_state_dict.pth")
"""
STATE_SAVE_PATCH = """                    if not bool(config.get("runtime", {}).get("skip_state_save", False)):
                        state_path = checkpoint_path.replace(".pth", "_state_dict.pth")
                    else:
                        state_path = None
"""

# --- Angular loss patch (enable the commented-out angular loss in baseline) ---
ANG_LOSS_ANCHOR = """                # ang_loss = self.angular_loss_function(flow, gt_flow, mask, num_valid_px)
                # curr_loss += self.lambda_mod * mod_loss + self.lambda_ang * ang_loss"""
ANG_LOSS_PATCH = """                if self.lambda_ang > 0:
                    ang_loss = self.angular_loss_function(flow, gt_flow, mask, num_valid_px)
                    curr_loss += self.lambda_mod * mod_loss + self.lambda_ang * ang_loss
                else:
                    curr_loss += self.lambda_mod * mod_loss"""


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
        (MLFLOW_MODEL_LOGGING_ANCHOR, MLFLOW_MODEL_LOGGING_PATCH),
        (STATE_SAVE_ANCHOR, STATE_SAVE_PATCH),
    ):
        if anchor not in source:
            raise RuntimeError(f"anchor missing: {anchor[:60]!r}")
        source = source.replace(anchor, replacement, 1)
    return source


def _install_angular_loss_hook(repo_root: Path):
    """Import hook: intercept flow_supervised to enable angular loss without modifying files."""
    import importlib
    import importlib.util

    target_file = str(
        repo_root / "third_party" / "SDformerFlow" / "loss" / "flow_supervised.py"
    )

    class AngularLoader(importlib.machinery.SourceFileLoader):
        def get_source(self, fullname):
            source = super().get_source(fullname)
            if ANG_LOSS_ANCHOR in source:
                source = source.replace(ANG_LOSS_ANCHOR, ANG_LOSS_PATCH, 1)
            return source

    class AngularFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != "loss.flow_supervised":
                return None
            loader = AngularLoader(fullname, target_file)
            return importlib.util.spec_from_loader(fullname, loader, origin=target_file)

    sys.meta_path.insert(0, AngularFinder())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, extra_args = parser.parse_known_args()

    repo_root = _repo_root()
    baseline_root = repo_root / "third_party" / "SDformerFlow"
    h9_overlay = repo_root / "neuron_experiments" / "H9_bipolar_self_attention" / "overlay"
    baseline_entry = baseline_root / "train_flow_parallel_supervised_SNN.py"

    _install_angular_loss_hook(repo_root)

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
