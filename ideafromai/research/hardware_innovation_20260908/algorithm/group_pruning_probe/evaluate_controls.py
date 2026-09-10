"""Re-evaluate saved strong controls in the same process settings as grouping.

The additional zero-hidden axis deletes all target FC1/PSN hidden neurons and
FC2 input columns, retaining real FC2 bias, BN2 and the outer shortcut. It is
an algorithm ablation, not a free hardware bypass or a chosen deployment.
"""
import argparse
import copy
from pathlib import Path

import torch

from run_group_pruning_probe import base, probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    system = probe.load_system(args)
    source = args.root/'algorithm/pruning_probe'
    out = args.root/'algorithm/group_pruning_probe/controls'
    out.mkdir(parents=True, exist_ok=True)
    variants = {name: torch.load(source/(name+'.pt'), map_location='cpu', weights_only=False)
                for name in ('original_trained', 'row_2of4_trained', 'hidden_H_half_trained')}
    zero = copy.deepcopy(variants['original_trained'])
    zero['weight_int8'].zero_()
    zero['weight_mask'].zero_()
    zero['hidden_keep'].zero_()
    zero['hidden_indices'] = torch.empty(0, dtype=torch.long)
    zero.update(remaining_H=0, nonzero_W=0, mask_slots=0,
                mask_rule='delete all s2b3 hidden neurons and corresponding FC2 columns; bias/BN2/shortcut retained')
    variants['zero_hidden'] = zero
    for name, record in variants.items():
        base.save_record(out, name, record)
    base.evaluate(args, system, variants, out, {})


if __name__ == '__main__':
    main()
