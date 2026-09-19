"""Describe the removed temporal matrices; no approximation or speed claim."""
import argparse
import json
from pathlib import Path
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint', type=Path)
    args = p.parse_args()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)['model_state_dict']
    rows = []
    for name, theta in state.items():
        if not name.endswith('.spiking_neuron.thresh'):
            continue
        prefix = name.removesuffix('.thresh')
        path = prefix.removesuffix('.spiking_neuron')
        if path.endswith(('.sn2_q', '.attn_sn')):
            continue
        w, bias = state[prefix + '.weight'].double(), state[prefix + '.bias'].double()
        rows.append(dict(name=prefix, T=w.shape[0], theta=float(theta),
                         squared_energy=float(w.square().sum()),
                         strict_upper_squared_energy=float(w.triu(1).square().sum()),
                         bias_min=float(bias.min()), bias_max=float(bias.max())))
    groups = {}
    for T in [2, 10]:
        subset = [r for r in rows if r['T'] == T]
        groups[str(T)] = dict(modules=len(subset),
            future_axis_weight_energy_fraction=sum(r['strict_upper_squared_energy'] for r in subset)
                                              / sum(r['squared_energy'] for r in subset))
    result = dict(scope='Exact ep34 weights; upper triangle is future on the supplied PSN axis. '
                  'T2 axis is legacy layout, not necessarily chronological. Weight energy is not quality or runtime.',
                  checkpoint=str(args.checkpoint), groups=groups, layers=rows)
    Path(__file__).with_name('REMOVED_PSN.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(groups, indent=2))


if __name__ == '__main__':
    main()
