"""A cheaper model control: retain codebook-active source time rows only.

No nearest-code projection. The original T10 continuous PSN inputs remain,
and all retained gates keep their actual theta. Same saved integer FC1/PSN,
real FC2/BN2/shortcut, and coarse readout as the S2 temporal-code student.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from run_bn_probe import build_model, input_frame, set_bn_mode, ALL_FC1, save_json, read_names
from evaluate_stage2_deployment import install_saved_consumers, S2, tag, CoarseReady


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('code-root', 'config', 'checkpoint', 'data', 'calibration', 'parameters', 'codebooks', 'samples', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--full-valid', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'capture').mkdir(exist_ok=True)
    model, cfg, _, _ = build_model(args)
    model.requires_grad_(False)
    from spikingjelly.activation_based import functional
    stats = torch.load(args.calibration, map_location='cpu', weights_only=False)
    params = torch.load(args.parameters, map_location='cpu', weights_only=False)
    set_bn_mode(model, stats, ALL_FC1)
    install_saved_consumers(model, params)
    modules, current, rows = dict(model.named_modules()), {}, []
    active_rows, handles = {}, []
    with np.load(args.codebooks) as book:
        for prefix in S2:
            dictionary = book[tag(prefix)+'_dictionary']
            active = np.any(dictionary != dictionary[0], axis=0)
            mask = torch.from_numpy(active).cuda().reshape(10, 1, 1, 1, 1)
            active_rows[tag(prefix)] = np.flatnonzero(active).tolist()

            def keep(module, inputs, output, selected=mask, name=tag(prefix)):
                result = output*selected
                if current['capture_name']:
                    g = result.ne(0).reshape(10, -1, result.shape[-1])
                    current.setdefault('sources', {})[name] = {
                        'active_gates': int(g.sum()), 'total_gates': g.numel(),
                        'nonzero_trajectories': int(g.any(0).sum())}
                    np.savez_compressed(args.output/'capture'/(Path(current['capture_name']).stem+'_'+name+'.npz'),
                        gate_bits=np.packbits(g.cpu().numpy(), axis=-1, bitorder='little'),
                        shape=np.array(g.shape), active_rows=selected.reshape(-1).cpu().numpy())
                return result

            handles.append(modules[prefix+'sn1.spiking_neuron'].register_forward_hook(keep))

    def readout(module, inputs, output):
        current['flow'] = output.detach().sum(0)
        raise CoarseReady()

    handles.append(modules['sttmultires_unet.preds.2'].register_forward_hook(readout))
    capture_names = json.loads(args.samples.read_text())['valid'][:10]
    names = read_names(args.data, 'valid') if args.full_valid else capture_names
    with torch.no_grad():
        for index, name in enumerate(names):
            current['capture_name'] = name if name in capture_names else None
            functional.reset_net(model)
            x, label, mask = input_frame(args.data, name)
            try:
                model(x)
            except CoarseReady:
                pred = F.interpolate(current.pop('flow'), size=(480, 640), mode='bilinear', align_corners=False)
            error = torch.linalg.vector_norm((pred-label).permute(0, 2, 3, 1)[mask], dim=1)
            row = {'file': name, 'valid_pixels': error.numel(),
                   'AEE': float(error.double().mean()), 'source_statistics': current.pop('sources', {})}
            rows.append(row)
            if (index+1) % 50 == 0 or index+1 == len(names):
                save_json(args.output/'frames.json', rows)
                print('VALID', index+1, float(np.mean([r['AEE'] for r in rows])), flush=True)
            del x, label, mask, pred, error
    for handle in handles:
        handle.remove()
    summary = {'frames': len(rows), 'AEE_frame_mean': float(np.mean([r['AEE'] for r in rows])),
               'valid_pixels': sum(r['valid_pixels'] for r in rows),
               'active_source_time_rows': active_rows,
               'scope': 'new-model control; ordinary fixed source time-row pruning, no K8 projection or validation fitting',
               'upstream': 'original full T10 sn1 runs; only emitted rows are masked here, so no GPU timing saving is claimed',
               'downstream': 'same all12 saved integer FC1/sn2, real FC2/BN2/shortcut and preds2 bilinear'}
    save_json(args.output/'frames.json', rows)
    save_json(args.output/'summary.json', summary)
    print('COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
