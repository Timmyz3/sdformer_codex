"""Create a separate CPU cache contract without touching the queued probe."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REMOTE = Path('/data/flow_a/code/research_20261013/motion_groups')
RUN = Path('/data/flow_a/runs/motion_groups_20261013')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True, type=int)
    cli = parser.parse_args()
    assert cli.revision >= 2
    out = RUN / ('coarse_cpu_pilot_v' + str(cli.revision))
    source = json.loads((HERE / 'cpu_smoke_contract.json').read_text())
    argv = source['base_args']
    args = dict(zip(argv[::2], argv[1::2]))
    args['--output'] = str(out / 'cache_report.json')
    args['--wall-minutes'] = '30'
    fit = HERE.parent / 'recovery/manifests/svs_fit20.json'
    pins = {p: v for p, v in source['source_pins'].items() if not p.endswith('run_pair_probe.py')}
    for name in ('collect_coarse_cache.py', 'coarse_head.py'):
        pins[str(REMOTE / name)] = sha(HERE / name)
    value = {'schema': 'flow_coarse_cache_contract_v1', 'CPU_only': True,
             'source_pins': pins, 'breadth_root': source['breadth_root'],
             'base_args': [v for item in args.items() for v in item],
             'fit_manifest': str(REMOTE / 'coarse_fit20_membership.json'),
             'fit_manifest_sha256': sha(fit), 'output_directory': str(out), 'stages': [0, 2],
             'flow_contract': {
                 'direction': 'forward_t_start_to_t_end_rectified_left_event_camera',
                 'time_interval': 'same-target forward GT pair; per-target timestamps bound separately',
                 'units': 'original_image_pixels'},
             'feature_label_geometry': 'logical_cell_support_origin0_v1',
             'dev_selection': 'existing gate_dev32 sequence round-robin first8; not final-valid',
             'fit_selection': 'reuse fixed SVS20 MEMBERS ONLY, not its scores or BN setting',
             'BN_policy': 'MLP_CMA176; outside MLP current-stat; fixed before features',
             'backbone_seen': True, 'GPU_required': False,
             'training_recipe_predeclared': {
                 'models': ['constant component median', 'linear C->2', 'C->16 GELU->2'],
                 'optimizer_for_trainable_heads': 'AdamW', 'learning_rate': 0.003,
                 'weight_decay': 0.0001, 'seed': 20261013, 'steps_per_trainable_head': 1000,
                 'batch': 'all valid fit tiles', 'loss': 'valid-tile-equal component L1',
                 'descriptor_normalization': 'fit-only population mean/std; folded into first FC',
                 'selection': 'fixed endpoint; no development early stopping or hyperparameter selection',
                 'evaluation_primary': 'within frame valid-tile average, then equal-frame mean',
                 'direction_threshold_original_pixels': 1.0,
                 'direction_abstention': 'prediction below1px is wrong on GT-eligible tiles; denominator unchanged'}}
    path = HERE / ('coarse_cache_contract_v' + str(cli.revision) + '.json')
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')
    print(json.dumps({'contract_sha256': sha(path), 'output_directory': str(out)}))


if __name__ == '__main__':
    main()
