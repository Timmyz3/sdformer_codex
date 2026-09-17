#!/usr/bin/env python3
"""Dedicated additive entrypoint; legacy H9 train.py and hardware stay intact."""
from pathlib import Path
import sys

import train as h9


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"refresh source anchor count != 1: {old[:100]!r}")
    return source.replace(old, new, 1)


original_patch = h9._patch_source


def patch_source(source, entry):
    source = original_patch(source, entry)
    source = replace_once(source, "    # optimizers\n", """    from models.STSwinNet_SNN.refresh_training import RefreshTraining, dense_export
    if args.resume:
        raise ValueError("Refresh initial trials use matched fresh optimizers, not --resume")
    if not args.finetune or not args.prev_runid:
        raise ValueError("Refresh requires full-resolution fine-tuning from a bound checkpoint")
    refresh = RefreshTraining(model, config, device)

    # optimizers
""")
    source = replace_once(source,
        "                h55_teacher_pred = teacher_forward(h55_teacher_model, chunk.to(device), config)\n"
        "                pred_list = model(chunk.to(device))\n",
        "                h55_teacher_pred = None\n"
        "                refresh_chunk = refresh.prepare(chunk, h9_global_step)\n"
        "                pred_list = model(refresh_chunk)\n")
    source = replace_once(source,
        "                from models.STSwinNet_SNN.atlif_ternary_psn import regularize_activity\n",
        "                curr_loss = curr_loss + refresh.penalty(pred, label, mask, h9_global_step) / num_acc_steps\n"
        "                if not torch.isfinite(curr_loss):\n"
        "                    raise FloatingPointError('non-finite refresh loss')\n"
        "                from models.STSwinNet_SNN.atlif_ternary_psn import regularize_activity\n")
    source = replace_once(source, "            sample += 1\n            train_sample_count += chunk.shape[0]\n",
        "            refresh.report(model, h9_global_step)\n            sample += 1\n"
        "            train_sample_count += chunk.shape[0]\n")
    source = replace_once(source, '{"model_state_dict": model.state_dict()},',
        '{"model_state_dict": dense_export(model)},')
    source = replace_once(source, '"optimizer": optimizer.state_dict(),',
        '"refresh_training_model_state": model.state_dict() if refresh.delay_names else None,\n'
        '                                "optimizer": optimizer.state_dict(),')
    # Use the existing standard evaluator after the run, not the trainer's
    # small running-BN validation screen; it also keeps all RNG streams paired.
    source = replace_once(source, '        if epoch % config["test"]["n_valid"] == 0:',
        '        if False:  # refresh queue runs standard valid825 on dense exports')
    return source


if __name__ == "__main__":
    h9._patch_source = patch_source
    h9.main()
