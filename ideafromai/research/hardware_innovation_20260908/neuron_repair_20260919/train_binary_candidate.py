"""Process-local training entry: correct AMP clipping, optional binary theta rule.

No source-tree overlay is edited. Pass normal H9 training arguments after --rule.
All new checkpoints must use an experiment-specific --save_path.
"""
import argparse
import importlib.util
from pathlib import Path
import sys


def patch_training_source(source, rule, theta_epsilon=None, neuron_fp32=False):
    old_clip = '''            if config["loss"]["clip_grad"] is not None:
                torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), config["loss"]["clip_grad"])
'''
    boundary = '''            if ((sample + 1) % num_acc_steps == 0) or (sample + 1 == len(train_dataloader)):
'''
    if '# [H9 AMP] Unscale before clipping at the optimizer boundary.' in source:
        # The normal H9 entry already owns the fix. Never unscale twice.
        assert old_clip not in source and source.count('scaler.unscale_(optimizer)') == 1
    else:
        assert source.count(old_clip) == 1 and source.count(boundary) == 1
        source = source.replace(old_clip, '', 1)
        source = source.replace(boundary, boundary + '''                if scaler is not None:
                    scaler.unscale_(optimizer)
                if config["loss"]["clip_grad"] is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["loss"]["clip_grad"])
''', 1)
    anchor = '    for epoch in range(epoch_initial, config["loader"]["n_epochs"]):'
    assert source.count(anchor) == 1
    if neuron_fp32:
        source = source.replace(anchor,
            '    from neuron_precision import install as _install_neuron_fp32\n'
            '    print("[neuron-fp32] installed", len(_install_neuron_fp32(model)), flush=True)\n' + anchor, 1)
    if theta_epsilon is not None:
        source = source.replace(anchor,
            '    _theta_groups = [g for g in optimizer.param_groups if g.get("name") == "atlif_threshold"]\n'
            '    assert len(_theta_groups) == 1, "theta-eps requires the explicit ATLIF parameter group"\n'
            f'    _theta_groups[0]["eps"] = {theta_epsilon!r}\n' + anchor, 1)
    if rule != 'official':
        source = source.replace(anchor,
            '    from binary_surrogate import install as _install_binary_candidate\n'
            f'    print("[binary-candidate] installed", _install_binary_candidate(model, {rule!r}), flush=True)\n' + anchor, 1)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--rule', choices=['official', 'margin', 'normalized'], required=True)
    parser.add_argument('--theta-epsilon', type=float)
    parser.add_argument('--neuron-fp32', action='store_true')
    args, remainder = parser.parse_known_args()
    path = args.repo / 'neuron_experiments/H9_bipolar_self_attention/entrypoints/train.py'
    spec = importlib.util.spec_from_file_location('h9_binary_candidate_train', path)
    h9 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h9)
    original_patch = h9._patch_source
    h9._patch_source = lambda source, entry: patch_training_source(
        original_patch(source, entry), args.rule, args.theta_epsilon, args.neuron_fp32)
    sys.argv = [str(path), *remainder]
    h9.main()


if __name__ == '__main__':
    main()
