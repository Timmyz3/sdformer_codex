"""H9 training entrypoint for PSN-ATLIF ternary attention plus Shiftmax."""

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

SEED_ANCHOR = """    config = config_parser.combine_entries(config)

    runtime_cfg = config.get("runtime", {})
"""

SEED_PATCH = """    config = config_parser.combine_entries(config)

    runtime_cfg = config.get("runtime", {})
    h9_seed = runtime_cfg.get("seed")
    if h9_seed is not None:
        import numpy as h9_numpy
        h9_seed = int(h9_seed)
        random.seed(h9_seed)
        h9_numpy.random.seed(h9_seed)
        torch.manual_seed(h9_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(h9_seed)
        h9_deterministic = bool(runtime_cfg.get("deterministic", False))
        torch.backends.cudnn.deterministic = h9_deterministic
        if h9_deterministic:
            torch.backends.cudnn.benchmark = False
        print(f"[runtime] seed={h9_seed}, deterministic={h9_deterministic}")
"""

LOAD_MODEL_ANCHOR = """    model = load_model(args.prev_runid, model, device, remap)
"""

LOAD_MODEL_PATCH = """    from models.STSwinNet_SNN.bsa_attention import register_shiftmax_pickle_compat
    register_shiftmax_pickle_compat()
    from models.STSwinNet_SNN.atlif_ternary_psn import apply_trainable_mode, atlif_ternary_summary, install_atlif_ternary_psn
    from models.STSwinNet_SNN.bsa_attention import install_shiftmax_attention, set_shiftmax_attention_step, shiftmax_attention_summary, sync_independent_value_branch_from_k

    def _h9_is_overlay_key(key):
        markers = (".linear_v.", ".bn_v.", ".sn_v.", "._h9_match_code_weight", "._h9_lc4_coefficients", "._h9_cf10_beta", ".spiking_neuron.thresh", ".spiking_neuron.center", ".temporal_factor_left", ".temporal_factor_right")
        return any(marker in key for marker in markers)

    def _h9_is_match_candidate_key(key):
        return any(marker in key for marker in ("._h9_match_code_weight", "._h9_lc4_coefficients", "._h9_cf10_beta"))

    def _h9_load_model_with_audit(prev_runid, model, device, remap=None):
        if prev_runid and os.path.isfile(prev_runid):
            from utils.utils import _extract_pretrained_state_dict, load_pretrained_interpolate, load_model as _baseline_load_model, remap_pretrained_keys_swin
            pretrained_model = torch.load(prev_runid, map_location=device, weights_only=False)
            pretrained_dict = _extract_pretrained_state_dict(pretrained_model, test=False)
            if remap == "v2":
                print(">>>>>>>>>> Remapping pre-trained keys for SWIN ..........")
                pretrained_dict = remap_pretrained_keys_swin(model, pretrained_dict)
            elif remap == "v1":
                load_pretrained_interpolate(model, pretrained_dict)
                print("[H9] remap=v1 interpolation complete; applying interpolated state dict")
            overlay_checkpoint_keys = [key for key in pretrained_dict.keys() if _h9_is_overlay_key(key)]
            match_candidate_checkpoint_keys = [key for key in pretrained_dict.keys() if _h9_is_match_candidate_key(key)]
            overlay_v_checkpoint_keys = [
                key for key in pretrained_dict.keys()
                if any(marker in key for marker in (".linear_v.", ".bn_v.", ".sn_v."))
            ]
            # ── NTX-11 window compat: drop keys whose shape changed (e.g., positional_encoding) ──
            _model_keys = dict(model.named_parameters())
            _dropped_window_keys = []
            for _k in list(pretrained_dict.keys()):
                if _k in _model_keys and _model_keys[_k].shape != pretrained_dict[_k].shape:
                    if any(marker in _k for marker in (".temporal_factor_left", ".temporal_factor_right")):
                        raise RuntimeError(
                            "[M29] factor checkpoint rank/shape does not match config: "
                            + _k
                        )
                    del pretrained_dict[_k]
                    _dropped_window_keys.append(_k)
            if _dropped_window_keys:
                print(f"[H9] dropped {len(_dropped_window_keys)} shape-mismatched keys (window_size changed): {_dropped_window_keys[:5]}")
            # ── end window compat ──
            incompatible = model.load_state_dict(pretrained_dict, strict=False)
            missing = list(getattr(incompatible, "missing_keys", []))
            unexpected = list(getattr(incompatible, "unexpected_keys", []))
            overlay_missing = [key for key in missing if _h9_is_overlay_key(key)]
            match_code_missing = [key for key in missing if "._h9_match_code_weight" in key]
            round3_aux_missing = [key for key in missing if "._h9_lc4_coefficients" in key]
            round4_aux_missing = [key for key in missing if "._h9_cf10_beta" in key]
            match_candidate_missing = [key for key in missing if _h9_is_match_candidate_key(key)]
            non_candidate_overlay_missing = [key for key in overlay_missing if not _h9_is_match_candidate_key(key)]
            overlay_unexpected = [key for key in unexpected if _h9_is_overlay_key(key)]
            print(
                f"[H9] load audit: checkpoint_overlay_keys={len(overlay_checkpoint_keys)}, "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
            if missing:
                print(f"[H9] missing keys sample: {missing[:12]}")
            if unexpected:
                print(f"[H9] unexpected keys sample: {unexpected[:12]}")
            if overlay_unexpected:
                raise RuntimeError(
                    "[H9] overlay checkpoint keys were not registered before load: "
                    + str(overlay_unexpected[:20])
                )
            if overlay_checkpoint_keys and non_candidate_overlay_missing:
                raise RuntimeError(
                    "[H9] checkpoint contains overlay parameters but matching model keys are missing: "
                    + str(non_candidate_overlay_missing[:20])
                )
            if match_candidate_checkpoint_keys and match_candidate_missing:
                raise RuntimeError(
                    "[H9] Match-Code candidate checkpoint is incomplete for the current model: "
                    + str(match_candidate_missing[:20])
                )
            if match_candidate_missing and not match_candidate_checkpoint_keys:
                match_modes = {
                    "binary_de9_match_code", "de9_match_code",
                    "binary_mc49_match_code", "mc49_match_code",
                    "binary_ax17_match_code", "ax17_match_code",
                    "binary_pc9_patch_match_code", "pc9_patch_match_code", "h76_pc9",
                    "binary_lc4_match_code", "lc4_match_code", "h77_lc4",
                    "binary_g4_match_code", "g4_match_code", "h78_g4",
                    "binary_cf10_match_code", "cf10_match_code", "h79_cf10",
                    "binary_dn9_match_code", "dn9_match_code", "h80_dn9",
                }
                current_mode = str(config.get("bsa_attention", {}).get("mode", ""))
                if current_mode not in match_modes:
                    raise RuntimeError("[H9] unexpected missing Match-Code candidate weights outside a Match-Code config")
                if match_code_missing:
                    print(f"[H9] initialized new Match-Code weights: {len(match_code_missing)}")
                if round3_aux_missing:
                    print(f"[H9] initialized new Round3 auxiliary parameters: {len(round3_aux_missing)}")
                if round4_aux_missing:
                    print(f"[H9] initialized new Round4 auxiliary parameters: {len(round4_aux_missing)}")
            if not overlay_v_checkpoint_keys:
                synced_v = sync_independent_value_branch_from_k(model, config.get("bsa_attention"))
                if synced_v:
                    print(f"[H9] initialized independent V from loaded K branches: {synced_v} modules")
            del pretrained_model
            torch.cuda.empty_cache()
            print("Model restored from local checkpoint " + prev_runid + "\\n")
            return model
        from utils.utils import load_model as _baseline_load_model
        return _baseline_load_model(prev_runid, model, device, remap)

    if bool(config.get("runtime", {}).get("load_full_model", False)) and args.prev_runid and os.path.isfile(args.prev_runid):
        model = torch.load(args.prev_runid, map_location=device, weights_only=False)
        model.to(device)
        print("H9 full model restored from local checkpoint " + args.prev_runid + "\\n")
    else:
        installed_h9_preload = install_atlif_ternary_psn(model, config.get("atlif_ternary_psn"))
        installed_h9_bsa_preload = install_shiftmax_attention(model, config.get("bsa_attention"))
        if installed_h9_preload:
            print(f"[H9] installed ATLIFTernaryPSN before load: {len(installed_h9_preload)} modules")
            print(f"[H9] preload neuron targets: {installed_h9_preload[:8]}{' ...' if len(installed_h9_preload) > 8 else ''}")
        if installed_h9_bsa_preload:
            print(f"[H9] installed attention before load: {len(installed_h9_bsa_preload)} modules")
            print(f"[H9] preload attention targets: {installed_h9_bsa_preload[:8]}{' ...' if len(installed_h9_bsa_preload) > 8 else ''}")
        model = _h9_load_model_with_audit(args.prev_runid, model, device, remap)
    installed_h9 = install_atlif_ternary_psn(model, config.get("atlif_ternary_psn"))
    if installed_h9:
        print(f"[H9] installed ATLIFTernaryPSN: {len(installed_h9)} modules")
        print(f"[H9] neuron targets: {installed_h9[:8]}{' ...' if len(installed_h9) > 8 else ''}")
    installed_h9_bsa = install_shiftmax_attention(model, config.get("bsa_attention"))
    if installed_h9_bsa:
        print(f"[H9] installed Shiftmax attention: {len(installed_h9_bsa)} modules")
        print(f"[H9] attention targets: {installed_h9_bsa[:8]}{' ...' if len(installed_h9_bsa) > 8 else ''}")

    from models.STSwinNet_SNN.pattern_paft import install_pattern_paft
    installed_m71_paft = install_pattern_paft(
        model, config.get("pattern_paft"), args.prev_runid)
    if installed_m71_paft:
        print(f"[M71] installed hardware-weighted PAFT hooks: {installed_m71_paft}")

    # ---- SimpleTernaryPSN (PSN+ternary, no ATLIF) ----
    from models.STSwinNet_SNN.simple_ternary_installer import install_simple_ternary_psn
    installed_st = install_simple_ternary_psn(model, config)
    if installed_st:
        print(f"[ST] installed SimpleTernaryPSN: {len(installed_st)} modules, "
              f"theta_init={config.get('simple_ternary_psn', {}).get('theta_init', 1.0)}")
    if installed_h9 or installed_h9_bsa:
        print(f"[H9] trainable: {apply_trainable_mode(model, config.get('atlif_ternary_psn'))}")
        print(f"[H9] neuron summary after install: {atlif_ternary_summary(model)}")
        print(f"[H9] attention summary after install: {shiftmax_attention_summary(model)}")
"""

LOSS_ANCHOR = """                # print("loss: ", curr_loss.item())

                if np.isnan(curr_loss.item()):
                    raise
"""

LOSS_PATCH = """                from models.STSwinNet_SNN.atlif_ternary_psn import regularize_activity
                h6_penalty = regularize_activity(model, config.get("atlif_ternary_psn"))
                if h6_penalty is not None:
                    curr_loss = curr_loss + h6_penalty / num_acc_steps
                from models.STSwinNet_SNN.bsa_attention import (
                    regularize_class_stability,
                    regularize_member_jaccard,
                    regularize_source_gate_cardinality,
                )
                h9_flow_loss_before_gate_cardinality = curr_loss.detach()
                h9_class_stability_penalty = regularize_class_stability(
                    model, config.get("bsa_attention")
                )
                if h9_class_stability_penalty is not None:
                    curr_loss = curr_loss + h9_class_stability_penalty / num_acc_steps
                h9_member_jaccard_penalty = regularize_member_jaccard(
                    model, config.get("bsa_attention")
                )
                if h9_member_jaccard_penalty is not None:
                    curr_loss = curr_loss + h9_member_jaccard_penalty / num_acc_steps
                from models.STSwinNet_SNN.bsa_attention import regularize_row_jaccard
                h9_row_jaccard_penalty = regularize_row_jaccard(
                    model, config.get("bsa_attention")
                )
                if h9_row_jaccard_penalty is not None:
                    curr_loss = curr_loss + h9_row_jaccard_penalty / num_acc_steps
                from models.STSwinNet_SNN.bsa_attention import regularize_h85_delta
                h9_h85_penalty = regularize_h85_delta(model, config.get("bsa_attention"))
                if h9_h85_penalty is not None:
                    curr_loss = curr_loss + h9_h85_penalty / num_acc_steps
                from models.STSwinNet_SNN.bsa_attention import regularize_h86_member_tv
                h9_h86_penalty = regularize_h86_member_tv(model, config.get("bsa_attention"))
                if h9_h86_penalty is not None:
                    curr_loss = curr_loss + h9_h86_penalty / num_acc_steps
                h9_gate_cardinality_penalty = regularize_source_gate_cardinality(
                    model, config.get("bsa_attention")
                )
                if h9_gate_cardinality_penalty is not None:
                    curr_loss = curr_loss + h9_gate_cardinality_penalty / num_acc_steps
                    h9_gate_log_interval = int(
                        config.get("bsa_attention", {}).get(
                            "source_gate_cardinality_log_interval_steps", 0
                        )
                        or 0
                    )
                    if h9_gate_log_interval > 0 and (sample + 1) % h9_gate_log_interval == 0:
                        h9_gate_weight = float(
                            config.get("bsa_attention", {}).get(
                                "source_gate_cardinality_regularization_weight", 0.0
                            )
                            or 0.0
                        )
                        h9_gate_proxy = h9_gate_cardinality_penalty.detach() / h9_gate_weight
                        print(
                            f"[H9-GC] step {sample + 1}: "
                            f"flow_loss={h9_flow_loss_before_gate_cardinality.item():.9g}, "
                            f"unweighted_proxy={h9_gate_proxy.item():.9g}, "
                            f"weighted_penalty={h9_gate_cardinality_penalty.detach().item():.9g}"
                        )
                from models.STSwinNet_SNN.pattern_paft import regularize_pattern_paft
                m71_pattern_paft_penalty = regularize_pattern_paft(
                    model, config.get("pattern_paft")
                )
                if m71_pattern_paft_penalty is not None:
                    curr_loss = curr_loss + m71_pattern_paft_penalty / num_acc_steps
                # print("loss: ", curr_loss.item())

                if np.isnan(curr_loss.item()):
                    raise
"""

SCALER_STEP_ANCHOR = """                    scaler.step(optimizer)
                    scaler.update()
"""

SCALER_STEP_PATCH = """                    from models.STSwinNet_SNN.h28_optimizer import apply_lr_warmup, lr_warmup_factor
                    h40_global_step = epoch * len(train_dataloader) + sample + 1
                    h40_lrs = apply_lr_warmup(optimizer, h40_global_step, config)
                    from models.STSwinNet_SNN.h28_optimizer import freeze_threshold_gradients
                    h40_frozen_threshold_grads = freeze_threshold_gradients(model, h40_global_step, config)
                    scaler.step(optimizer)
                    from models.STSwinNet_SNN.atlif_ternary_psn import threshold_update
                    h40_warmup_factor = lr_warmup_factor(h40_global_step, config)
                    h8_threshold_lr = float(config.get("atlif_ternary_psn", {}).get("threshold_base_lr", optimizer.param_groups[0]["lr"]))
                    if h40_warmup_factor is not None:
                        h8_threshold_lr *= h40_warmup_factor
                    h8_threshold_cfg = dict(config.get("atlif_ternary_psn", {}) or {})
                    h8_threshold_cfg["_global_step"] = h40_global_step
                    h8_update_stats = threshold_update(model, h8_threshold_lr, h8_threshold_cfg)
                    h6_log_interval = int(config.get("atlif_ternary_psn", {}).get("log_interval_steps", 0) or 0)
                    if h6_log_interval > 0 and (sample + 1) % h6_log_interval == 0:
                        if h40_lrs is not None:
                            print(f"[H40] step {sample + 1} lr_warmup: {h40_lrs}")
                        if h40_frozen_threshold_grads:
                            print(f"[H40] step {sample + 1} frozen threshold gradients: {h40_frozen_threshold_grads}")
                        print(f"[H9] step {sample + 1} update: {h8_update_stats}")
                    scaler.update()
"""

OPTIMIZER_STEP_ANCHOR = """                    optimizer.step()

                # zero grad
"""

OPTIMIZER_STEP_PATCH = """                    from models.STSwinNet_SNN.h28_optimizer import apply_lr_warmup, lr_warmup_factor
                    h40_global_step = epoch * len(train_dataloader) + sample + 1
                    h40_lrs = apply_lr_warmup(optimizer, h40_global_step, config)
                    from models.STSwinNet_SNN.h28_optimizer import freeze_threshold_gradients
                    h40_frozen_threshold_grads = freeze_threshold_gradients(model, h40_global_step, config)
                    optimizer.step()
                    from models.STSwinNet_SNN.atlif_ternary_psn import threshold_update
                    h40_warmup_factor = lr_warmup_factor(h40_global_step, config)
                    h8_threshold_lr = float(config.get("atlif_ternary_psn", {}).get("threshold_base_lr", optimizer.param_groups[0]["lr"]))
                    if h40_warmup_factor is not None:
                        h8_threshold_lr *= h40_warmup_factor
                    h8_threshold_cfg = dict(config.get("atlif_ternary_psn", {}) or {})
                    h8_threshold_cfg["_global_step"] = h40_global_step
                    h8_update_stats = threshold_update(model, h8_threshold_lr, h8_threshold_cfg)
                    h6_log_interval = int(config.get("atlif_ternary_psn", {}).get("log_interval_steps", 0) or 0)
                    if h6_log_interval > 0 and (sample + 1) % h6_log_interval == 0:
                        if h40_lrs is not None:
                            print(f"[H40] step {sample + 1} lr_warmup: {h40_lrs}")
                        if h40_frozen_threshold_grads:
                            print(f"[H40] step {sample + 1} frozen threshold gradients: {h40_frozen_threshold_grads}")
                        print(f"[H9] step {sample + 1} update: {h8_update_stats}")

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
            from models.STSwinNet_SNN.atlif_ternary_psn import atlif_temporal_factor_diagnostics, atlif_ternary_summary
            from models.STSwinNet_SNN.bsa_attention import shiftmax_attention_summary
            print(f"[H9] ATLIFTernaryPSN summary: {atlif_ternary_summary(model)}")
            print(f"[M29] temporal factor diagnostics: {atlif_temporal_factor_diagnostics(model)}")
            print(f"[H9] Shiftmax attention summary: {shiftmax_attention_summary(model)}")
        if config.get("pattern_paft", {}).get("enabled", False):
            from models.STSwinNet_SNN.pattern_paft import pattern_paft_summary
            print(f"[M71] PAFT summary: {pattern_paft_summary(model)}")
"""

MLFLOW_METRIC_STEP_ANCHOR = """            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=epoch)
            mlflow.log_metric("epoch_time_sec", epoch_time_sec, step=epoch)
            mlflow.log_metric("train_step_time_sec", train_step_time_sec, step=epoch)
            mlflow.log_metric("train_samples_per_sec", train_samples_per_sec, step=epoch)
            mlflow.log_metric("max_gpu_mem_gib", max_gpu_mem_gib, step=epoch)
"""

MLFLOW_METRIC_STEP_PATCH = """            h9_epoch_offset = int(config.get("runtime", {}).get("epoch_offset", 0) or 0)
            h9_log_epoch = epoch + h9_epoch_offset
            mlflow.log_metric("train_loss", epoch_loss, step=h9_log_epoch)
            mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=h9_log_epoch)
            mlflow.log_metric("epoch_time_sec", epoch_time_sec, step=h9_log_epoch)
            mlflow.log_metric("train_step_time_sec", train_step_time_sec, step=h9_log_epoch)
            mlflow.log_metric("train_samples_per_sec", train_samples_per_sec, step=h9_log_epoch)
            mlflow.log_metric("max_gpu_mem_gib", max_gpu_mem_gib, step=h9_log_epoch)
"""

TRAIN_STEP_ANCHOR = """            sample += 1
            train_sample_count += chunk.shape[0]
"""

TRAIN_STEP_PATCH = """            sample += 1
            train_sample_count += chunk.shape[0]
            max_train_steps = int(config.get("runtime", {}).get("max_train_steps", 0) or 0)
            if max_train_steps > 0 and sample >= max_train_steps:
                print(f"[H9] stopping train epoch early at max_train_steps={max_train_steps}")
                break
"""

SAVE_ANCHOR = """            should_save_model = epoch_loss < best_loss or epoch == config["loader"]["n_epochs"] - 1
"""

SAVE_PATCH = """            force_save_epochs = set(int(item) for item in config.get("runtime", {}).get("force_save_epochs", []) or [])
            save_only_force_epochs = bool(config.get("runtime", {}).get("save_only_force_epochs", False))
            if save_only_force_epochs:
                should_save_model = (
                    epoch in force_save_epochs
                    or epoch == config["loader"]["n_epochs"] - 1
                )
            else:
                should_save_model = (
                    epoch_loss < best_loss
                    or epoch == config["loader"]["n_epochs"] - 1
                    or epoch in force_save_epochs
                )
            should_save_model = should_save_model and not bool(config.get("runtime", {}).get("skip_save", False))
"""

MLFLOW_MODEL_LOGGING_ANCHOR = """                if use_ml_flow and use_mlflow_model_logging:
"""

MLFLOW_MODEL_LOGGING_PATCH = """                if (
                    use_ml_flow
                    and use_mlflow_model_logging
                    and bool(config.get("runtime", {}).get("use_mlflow_model_logging", False))
                ):
"""

MODEL_CHECKPOINT_SAVE_ANCHOR = """                    torch.save(model, checkpoint_path)
"""

MODEL_CHECKPOINT_SAVE_PATCH = """                    # Save only tensor state here.  Runtime-installed PAFT hooks may
                    # contain local Python callables, so pickling the whole module is
                    # neither portable nor guaranteed to succeed.  load_model()
                    # explicitly accepts this model_state_dict container.
                    torch.save(
                        {"model_state_dict": model.state_dict()},
                        checkpoint_path,
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

STATE_SAVE_PATCH = """                    state_save_epochs = set(int(item) for item in config.get("runtime", {}).get("state_save_epochs", []) or [])
                    should_save_state = (
                        not state_save_epochs
                        or epoch in state_save_epochs
                        or epoch == config["loader"]["n_epochs"] - 1
                    )
                    if should_save_state and not bool(config.get("runtime", {}).get("skip_state_save", False)):
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

CHECKPOINT_PATH_ANCHOR = """                    checkpoint_path = args.save_path.format(epoch)
"""

CHECKPOINT_PATH_PATCH = """                    h9_epoch_offset = int(config.get("runtime", {}).get("epoch_offset", 0) or 0)
                    checkpoint_path = args.save_path.format(epoch + h9_epoch_offset)
"""

LOSS_FUNCTION_ANCHOR = """    # Define the loss function
    loss_function = flow_loss_supervised(config,device)
"""

LOSS_FUNCTION_PATCH = """    # Define the loss function
    loss_function = flow_loss_supervised(config,device)
    from models.STSwinNet_SNN.h9_losses import maybe_replace_flow_loss
    loss_function = maybe_replace_flow_loss(loss_function, config, device)
    from models.STSwinNet_SNN.h55_teacher import build_teacher_model, teacher_forward
    h55_teacher_model = build_teacher_model(config, device, remap)
"""

OPTIMIZER_ANCHOR = """    # optimizers
    if config["optimizer"]["name"] == 'AdamW':
        optimizer = eval(config["optimizer"]["name"])(model.parameters(), lr=config["optimizer"]["lr"],weight_decay=config["optimizer"]["wd"])

    else:
        optimizer = eval(config["optimizer"]["name"])(model.parameters(), lr=config["optimizer"]["lr"])
"""

OPTIMIZER_PATCH = """    # optimizers
    from models.STSwinNet_SNN.h28_optimizer import build_optimizer, describe_optimizer_groups
    optimizer = build_optimizer(model, config)
    print(f"[H28] optimizer groups: {describe_optimizer_groups(optimizer)}")
"""

TEACHER_FORWARD_ANCHOR = """                pred_list = model(chunk.to(device))
                pred = pred_list["flow"]

                #backward pass only the last flow pred
                if config["metrics"]["mask_events"]:
                    # event_mask = torch.unsqueeze(torch.sum(chunk, dim=1).bool(), dim=1)
                    event_mask = torch.sum(torch.sum(chunk, dim=1),dim=1, keepdim=True).bool()
                    curr_loss = loss_function(pred, label, mask*event_mask, gamma = config["loss"]["gamma"])/num_acc_steps
                else:
                    curr_loss = loss_function(pred, label, mask, gamma = config["loss"]["gamma"])/num_acc_steps
"""

TEACHER_FORWARD_PATCH = """                h9_global_step = epoch * len(train_dataloader) + sample + 1
                set_shiftmax_attention_step(model, h9_global_step)
                h55_teacher_pred = teacher_forward(h55_teacher_model, chunk.to(device), config)
                pred_list = model(chunk.to(device))
                pred = pred_list["flow"]

                #backward pass only the last flow pred
                if config["metrics"]["mask_events"]:
                    # event_mask = torch.unsqueeze(torch.sum(chunk, dim=1).bool(), dim=1)
                    event_mask = torch.sum(torch.sum(chunk, dim=1),dim=1, keepdim=True).bool()
                    if hasattr(loss_function, "set_teacher_prediction"):
                        loss_function.set_teacher_prediction(h55_teacher_pred)
                    curr_loss = loss_function(pred, label, mask*event_mask, gamma = config["loss"]["gamma"])/num_acc_steps
                else:
                    if hasattr(loss_function, "set_teacher_prediction"):
                        loss_function.set_teacher_prediction(h55_teacher_pred)
                    curr_loss = loss_function(pred, label, mask, gamma = config["loss"]["gamma"])/num_acc_steps
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


def _absolutize_path_args(extra_args: list[str]) -> list[str]:
    path_flags = {"--save_path", "--path_mlflow", "--prev_runid"}
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
        (SEED_ANCHOR, SEED_PATCH),
        (LOAD_MODEL_ANCHOR, LOAD_MODEL_PATCH),
        (LOSS_ANCHOR, LOSS_PATCH),
        (SCALER_STEP_ANCHOR, SCALER_STEP_PATCH),
        (OPTIMIZER_STEP_ANCHOR, OPTIMIZER_STEP_PATCH),
        (EPOCH_STATS_ANCHOR, EPOCH_STATS_PATCH),
        (MLFLOW_METRIC_STEP_ANCHOR, MLFLOW_METRIC_STEP_PATCH),
        (TRAIN_STEP_ANCHOR, TRAIN_STEP_PATCH),
        (SAVE_ANCHOR, SAVE_PATCH),
        (MLFLOW_MODEL_LOGGING_ANCHOR, MLFLOW_MODEL_LOGGING_PATCH),
        (MODEL_CHECKPOINT_SAVE_ANCHOR, MODEL_CHECKPOINT_SAVE_PATCH),
        (STATE_SAVE_ANCHOR, STATE_SAVE_PATCH),
        (CHECKPOINT_PATH_ANCHOR, CHECKPOINT_PATH_PATCH),
        (LOSS_FUNCTION_ANCHOR, LOSS_FUNCTION_PATCH),
        (OPTIMIZER_ANCHOR, OPTIMIZER_PATCH),
        (TEACHER_FORWARD_ANCHOR, TEACHER_FORWARD_PATCH),
    ):
        if anchor not in source:
            raise RuntimeError(f"Could not patch {baseline_entry}: missing anchor {anchor[:60]!r}")
        source = source.replace(anchor, replacement, 1)
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, extra_args = parser.parse_known_args()

    experiment_root = Path(__file__).resolve().parents[1]
    repo_root = _repo_root()
    baseline_root = repo_root / "third_party" / "SDformerFlow"
    overlay_root = experiment_root / "overlay"
    baseline_entry = baseline_root / "train_flow_parallel_supervised_SNN.py"

    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(baseline_root))
    sys.path.insert(0, str(overlay_root))
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
