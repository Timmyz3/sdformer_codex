"""Single-pass evaluation of saved signed12/signed8 six-source parameters.

Default is official full validation. --capture10 instead runs the fixed
diverse10 and saves 64 complete T10 source vectors per module and frame.
There is no recalibration, training, or full-validation FP32 reference cache.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import types

import numpy as np
import torch
import torch.nn.functional as F

from run_bn_probe import ALL_FC1, build_model, input_frame, read_names, save_json, set_bn_mode
from evaluate_stage2_deployment import S2, CoarseReady, install_saved_consumers, summarize, tag
from train_class_shift_probe import install_class_consumers, load_variant
from probe_direct_code_integer import XMIN, XMAX, integer_codes


def summarize_sources(counters):
    modules = {}
    total_elements = total_clipped = total_outside = 0
    for prefix, counts in counters.items():
        elements = counts['input_elements']
        clipped, outside = int(counts['clipped']), int(counts['outside'])
        modules[tag(prefix)] = {'input_elements': elements, 'clipped_elements': clipped,
                               'outside_representable_range_elements': outside,
                               'clipping_fraction': clipped/elements,
                               'outside_representable_range_fraction': outside/elements,
                               'class_histogram': counts['class_histogram'].cpu().tolist()}
        total_elements += elements
        total_clipped += clipped
        total_outside += outside
    return {'modules': modules, 'total_input_elements': total_elements,
            'total_clipped_elements': total_clipped,
            'total_outside_representable_range_elements': total_outside,
            'clipping_fraction': total_clipped/total_elements,
            'outside_representable_range_fraction': total_outside/total_elements}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--capture10', action='store_true')
    args = parser.parse_args()
    alg = args.root/'algorithm'
    out = alg/'direct_code_integer/deployment'
    out.mkdir(parents=True, exist_ok=True)
    stem = 'valid10' if args.capture10 else 'valid825'
    old = json.loads((alg/'stage2_temporal_codes/run.json').read_text())
    args.code_root = args.root/'code/SDformer'
    args.config, args.checkpoint = Path(old['config']), Path(old['checkpoint'])
    args.data = args.root.parent/'sdformer_codex/SDformer/data/Datasets/DSEC/saved_flow_data'
    names = (json.loads((alg/'samples.json').read_text())['valid'][:10] if args.capture10
             else read_names(args.data, 'valid'))
    model, _, _, _ = build_model(args)
    model.requires_grad_(False)
    from spikingjelly.activation_based import functional
    modules = dict(model.named_modules())
    stats = torch.load(alg/'valid825_cal32/train_calibration.pt', map_location='cpu', weights_only=False)
    old_params = torch.load(alg/'stage2_temporal_codes/integer_parameters.pt', map_location='cpu', weights_only=False)
    set_bn_mode(model, stats, ALL_FC1)
    gpu = install_saved_consumers(model, old_params)
    current = {'cache': None}
    install_class_consumers(model, gpu, alg, current)
    consumers = torch.load(alg/'stage2_class_shift/consumers.pt', map_location='cpu', weights_only=False)
    load_variant(gpu, consumers['power2_trained'])
    parameters = torch.load(alg/'direct_code_integer/parameters.pt', map_location='cpu', weights_only=False)
    book = np.load(alg/'stage2_temporal_codes/codebooks.npz')
    counters = {}
    capture = out/'capture10'
    if args.capture10:
        capture.mkdir(parents=True, exist_ok=True)

    for prefix in S2:
        exported = parameters[prefix]
        record = {key: value.cuda() if torch.is_tensor(value) else value for key, value in exported.items()}
        dictionary = torch.from_numpy(book[tag(prefix)+'_dictionary']).cuda().float()
        counters[prefix] = {'input_elements': 0,
                            'clipped': torch.zeros((), dtype=torch.int64, device='cuda'),
                            'outside': torch.zeros((), dtype=torch.int64, device='cuda'),
                            'class_histogram': torch.zeros(8, dtype=torch.int64, device='cuda')}

        def source(self, x, p=prefix, quantized=record, words=dictionary, q=gpu[prefix]):
            shape = x.shape
            channels = q['w'].shape[1]
            vectors = x.reshape(shape[0], -1, channels).permute(1, 2, 0)
            codes, clipped, outside = integer_codes(vectors, quantized)
            counts = counters[p]
            counts['input_elements'] += vectors.numel()
            counts['clipped'] += clipped
            counts['outside'] += outside
            counts['class_histogram'] += torch.bincount(codes.reshape(-1), minlength=8)
            if args.capture10:
                flattened = vectors.reshape(-1, shape[0])
                selected = torch.linspace(0, flattened.shape[0]-1, min(64, flattened.shape[0]),
                                          device=x.device).round().long()
                input_q = (flattened[selected]/quantized['input_step']).round().clamp(XMIN, XMAX).short()
                expected = codes.reshape(-1)[selected]
                inverse_mapping = torch.argsort(quantized['mapping'])
                # expected_class includes the learned address-to-class map;
                # expected_address is the raw three-predicate bit pattern.
                np.savez_compressed(capture/(f"v{current['index']:03d}_{Path(current['file']).stem}_{tag(p)}.npz"),
                    codes=codes.cpu().numpy().astype(np.uint8),
                    input_int12=input_q.cpu().numpy(),
                    expected_class=expected.cpu().numpy().astype(np.uint8),
                    expected_address=inverse_mapping[expected].cpu().numpy().astype(np.uint8),
                    flat_index=selected.cpu().numpy(),
                    position=(selected//channels).cpu().numpy(),
                    channel=(selected%channels).cpu().numpy(),
                    source_shape=np.array(shape), module=np.array(p),
                    input_step=np.float64(quantized['input_step']),
                    weight_int8=quantized['weight_int8'].cpu().numpy(),
                    threshold_int32=quantized['threshold_int32'].cpu().numpy(),
                    variable_rows=quantized['variable_rows'].cpu().numpy(),
                    constant_gates=quantized['constant_gates'].cpu().numpy(),
                    address_to_class=quantized['mapping'].cpu().numpy().astype(np.uint8),
                    class_to_spikes=words.cpu().numpy().astype(np.uint8),
                    theta_source=q['theta_source'].cpu().numpy())
            return (words[codes].permute(2, 0, 1)*q['theta_source']).reshape(shape)

        neuron = modules[prefix+'sn1.spiking_neuron']
        neuron.forward = types.MethodType(source, neuron)

    def stop(module, inputs, output):
        current['flow'] = output.detach().sum(0)
        raise CoarseReady()

    modules['sttmultires_unet.preds.2'].register_forward_hook(stop)
    save_json(out/(stem+'_run.json'), {
        'model': 'six-source bits3 signed12/signed8 student; saved train32 calibration',
        'parameters': str(alg/'direct_code_integer/parameters.pt'), 'modules': S2, 'frames': names,
        'source_arithmetic': 'int64 reference in original T order, initialized with minus tau; source parameters already checked against Acc24',
        'preserved': 'class lookup, continuous theta amplitudes, power2-trained class consumer, FC2, dynamic BN2, shortcuts',
        'readout': 'preds.2 time sum; bilinear480x640, align_corners=False, no flow scaling',
        'clipping': 'count actual rounded values outside signed12 before clamp; also report raw scaled values outside representable interval',
        'allow_tf32_matmul': bool(torch.backends.cuda.matmul.allow_tf32),
        'allow_tf32_cudnn': bool(torch.backends.cudnn.allow_tf32),
        'capture': ('complete codes[P,C] for same-model hardware service, plus 64 uniformly spaced flattened (position,channel) indices per module per frame; all T inputs retained; mapped class and raw decision address both saved'
                    if args.capture10 else 'none; single-pass full validation, no FP32 reference cache'),
        'limits': 'new integer-source model, no calibration or fitting in this evaluator, no hardware-cycle claim'})
    rows = []
    with torch.no_grad():
        for index, filename in enumerate(names):
            current['file'], current['index'] = filename, index
            functional.reset_net(model)
            x, label, mask = input_frame(args.data, filename)
            try:
                model(x)
            except CoarseReady:
                pred = F.interpolate(current.pop('flow'), (480, 640), mode='bilinear', align_corners=False)
            error = torch.linalg.vector_norm(pred.permute(0, 2, 3, 1)[mask]
                                            -label.permute(0, 2, 3, 1)[mask], dim=1)
            total, pixels = float(error.double().sum()), error.numel()
            rows.append({'file': filename, 'valid_pixels': pixels, 'aee_sum': total, 'AEE': total/pixels})
            if args.capture10 or (index+1) % 50 == 0 or index+1 == len(names):
                source_stats = summarize_sources(counters)
                result = summarize(rows, len(rows) == len(names))
                result['source_clipping_fraction'] = source_stats['clipping_fraction']
                result['source_clipped_elements'] = source_stats['total_clipped_elements']
                result['source_input_elements'] = source_stats['total_input_elements']
                save_json(out/(stem+'_frames.json'), rows)
                save_json(out/(stem+'_summary.json'), result)
                save_json(out/(stem+'_source_statistics.json'), source_stats)
                print('PROGRESS', index+1, len(names), json.dumps(result), flush=True)
            del x, label, mask, pred, error
    print('COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
