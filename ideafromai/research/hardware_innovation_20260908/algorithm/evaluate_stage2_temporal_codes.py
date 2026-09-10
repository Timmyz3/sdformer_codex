"""Train-only K8 temporal-code projection at all six S2 FFN sources.

Reuse saved native train32 BN calibration. Install the existing integer FC1
and sn2 compiler at all twelve FFNs; the rest of the model stays FP32. Collect
only CUDA 1024-bin source-signature histograms on those same training frames.
Compare this teacher and simultaneous S2 K8 projection on the fixed valid10.
No optimization, wide activation cache, second full-U check, or hardware timing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_bn_probe import (ALL_FC1, MLPS, build_model, input_frame, read_names,
                          save_json, set_bn_mode)
from run_integer_fc1_probe import install_integer_consumers
import numpy as np
import torch
import torch.nn.functional as F

S2 = [prefix for prefix in MLPS if '.layers.2.' in prefix]
T = 10


def tag(prefix):
    return 's2b' + prefix.split('.swin_blocks.')[1].split('.')[0]


def row_description(matrix):
    groups = {}
    for i, row in enumerate(matrix):
        groups.setdefault(tuple(int(v) for v in row), []).append(i)
    return {'time_row_classes': list(groups.values()),
            'equal_time_row_groups': [g for g in groups.values() if len(g) > 1],
            'zero_time_rows': np.flatnonzero(np.all(matrix == 0, axis=1)).tolist()}


def signed_width(matrix):
    lo, hi = int(matrix.min()), int(matrix.max())
    bits = 1
    while lo < -(1 << (bits-1)) or hi >= (1 << (bits-1)):
        bits += 1
    return {'min': lo, 'max': hi, 'minimum_signed_bits_for_observed_coefficients': bits}


def hist_description(counts):
    bits = ((np.arange(1024)[:, None] >> np.arange(T)) & 1).astype(np.int64)
    total = int(counts.sum())
    return {'trajectories': total, 'observed_signatures': int(np.count_nonzero(counts)),
            'zero_signature_count': int(counts[0]),
            'zero_signature_fraction': float(counts[0]/total),
            'time_firing_fraction': (counts @ bits / total).tolist(),
            'mean_gate_activity': float((counts @ bits).sum()/(total*T)),
            **row_description(bits[counts > 0].T)}


class SourceHistograms:
    def __init__(self, model, params):
        modules = dict(model.named_modules())
        self.theta = {p: params[p]['theta_source'].cuda().float() for p in S2}
        self.maps, self.shapes, self.residual = {}, {}, {}
        self.project = False
        self.reset()
        self.handles = [modules[p+'sn1.spiking_neuron'].register_forward_hook(self.hook(p)) for p in S2]

    def reset(self):
        self.counts = {p: torch.zeros(1024, dtype=torch.int64, device='cuda') for p in S2}

    def hook(self, prefix):
        def collect(module, inputs, output):
            assert output.shape[0] == T
            self.shapes.setdefault(prefix, list(output.shape))
            gate = output != 0
            if prefix not in self.residual:
                self.residual[prefix] = float((output-gate*self.theta[prefix]).abs().max())
                assert self.residual[prefix] == 0, 'This source cannot be represented by its actual scalar theta*g.'
            shifts = torch.arange(T, device=output.device, dtype=torch.int64)
            shifts = shifts.reshape(T, *([1]*(output.ndim-1)))
            words = torch.sum(gate.to(torch.int64) << shifts, dim=0)
            self.counts[prefix].add_(torch.bincount(words.reshape(-1), minlength=1024))
            if self.project:
                projected_words = self.maps[prefix][words]
                projected = ((projected_words.unsqueeze(0) >> shifts) & 1).to(output.dtype)
                return projected*self.theta[prefix]
        return collect

    def cpu_counts(self):
        return {p: counts.cpu().numpy().copy() for p, counts in self.counts.items()}

    def close(self):
        for handle in self.handles:
            handle.remove()


def make_codebooks(args, histograms, params, counts):
    records, arrays, word_maps, selected = {}, {}, {}, {}
    for prefix in S2:
        name = tag(prefix)
        count = counts[prefix]
        order = sorted(range(1, 1024), key=lambda word: (-int(count[word]), word))
        words = np.array([0]+order[:7], dtype=np.int64)
        # First dictionary index breaks Hamming-distance ties, as in the S0 pilot.
        mapping = np.array([int(words[min(range(8), key=lambda j: ((word ^ int(words[j])).bit_count(), j))])
                            for word in range(1024)], dtype=np.int64)
        dictionary = ((words[:, None] >> np.arange(T)) & 1).astype(np.int64)
        projected_counts = np.zeros(1024, dtype=np.int64)
        np.add.at(projected_counts, mapping, count)
        record = params[prefix]
        aq = record['temporal_int16'].numpy().astype(np.int64)
        transformed = aq @ dictionary.T
        w = record['weight_int8']
        records[name] = {
            'module': prefix, 'source_shape': histograms.shapes[prefix],
            'C': int(w.shape[1]), 'H': int(w.shape[0]),
            'theta_source': float(record['theta_source']),
            'theta_output': float(record['theta_output']),
            'theta_g_max_residual_first_train_frame': histograms.residual[prefix],
            'words_lsb_is_time0': words.tolist(),
            'dictionary_order': 'zero then descending train frequency, numeric word breaks count ties',
            'projection': 'nearest Hamming word; first dictionary index breaks distance ties',
            'selected_train_counts': count[words].tolist(),
            'train_exact_dictionary_coverage': float(count[words].sum()/count.sum()),
            'teacher_train_source': hist_description(count),
            'projected_teacher_train_source': hist_description(projected_counts),
            'dictionary_time_rows': row_description(dictionary.T),
            'A': {'meaning': 'integer sn2 temporal matrix, after existing calibrated-BN compiler',
                  'shape': list(aq.shape), 'stored_signed_bits': 16,
                  'fractional_bits': record['temporal_fractional_bits'],
                  **signed_width(aq), **row_description(aq)},
            'B': {'meaning': 'A @ dictionary.T; column0 is the zero signature',
                  'shape': list(transformed.shape), 'stored_signed_bits': 32,
                  'nonzero_coefficients': int(np.count_nonzero(transformed)),
                  'nonzero_columns': int(np.count_nonzero(np.any(transformed != 0, axis=0))),
                  **signed_width(transformed), **row_description(transformed)},
            'W_storage_signed_bits': 8,
            'source_producer': 'original full-T10 FP32 sn1 is executed, then its actual theta*g is projected',
            'time_rank_limit': 'zero plus seven source words span at most seven time dimensions; original sn2 A is retained'}
        arrays[name+'_train_histogram'] = count
        arrays[name+'_words'] = words.astype(np.uint16)
        arrays[name+'_word_map'] = mapping.astype(np.uint16)
        arrays[name+'_dictionary'] = dictionary.astype(np.uint8)
        arrays[name+'_A_int16'] = aq.astype(np.int16)
        arrays[name+'_B_int32'] = transformed.astype(np.int32)
        histograms.maps[prefix] = torch.from_numpy(mapping).cuda()
        word_maps[prefix], selected[prefix] = mapping, words
    np.savez_compressed(args.output/'codebooks.npz', **arrays)
    save_json(args.output/'codebooks.json', records)
    return records, word_maps, selected


def flow_metric(pred, label, mask):
    error = torch.linalg.vector_norm(
        pred.permute(0, 2, 3, 1)[mask]-label.permute(0, 2, 3, 1)[mask], dim=1)
    total = float(error.double().sum())
    return {'AEE': total/error.numel(), 'aee_sum': total}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('code-root', 'config', 'checkpoint', 'data', 'samples', 'calibration'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'stage2_temporal_codes')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    calibration_run = json.loads(args.calibration.with_name('run.json').read_text())
    train = calibration_run['calibration_train']
    valid = json.loads(args.samples.read_text())['valid'][:10]
    official_train, official_valid = set(read_names(args.data, 'train')), set(read_names(args.data, 'valid'))
    assert len(train) == 32 and len(set(train)) == 32
    assert set(train) <= official_train and not set(train) & official_valid
    assert len(valid) == 10 and set(valid) <= official_valid
    model, cfg, installed, attention = build_model(args)
    model.requires_grad_(False)
    from spikingjelly.activation_based import functional
    stats = torch.load(args.calibration, map_location='cpu', weights_only=False)
    set_bn_mode(model, stats, ALL_FC1)
    params, descriptions = install_integer_consumers(model, stats, prefixes=MLPS)
    histograms = SourceHistograms(model, params)
    save_json(args.output/'run.json', {
        'kind': 'S2_six_FFN_train_only_temporal_code_projection_generalization_probe',
        'checkpoint': str(args.checkpoint), 'config': str(args.config),
        'calibration': str(args.calibration), 'calibration_reused_without_refitting': True,
        'train_codebook_frames': train, 'valid': valid, 'optimization_updates': 0,
        'integer_consumer_prefixes': MLPS, 'projected_source_prefixes': S2,
        'teacher': 'all12 FC1/sn2 use existing integer compiler; all other arithmetic remains its original type',
        'BN': 'ALL_FC1 BN1 constants from native train32; other BN including BN2 keep current-frame statistics',
        'attention_hardware_quant_enabled': cfg['bsa_attention']['hardware_quant_enabled'],
        'motion_alpha': cfg['bsa_attention']['binary_motion_xor_alpha'],
        'installed_atlif': len(installed), 'attention_blocks': len(attention),
        'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
        'tf32_cudnn': torch.backends.cudnn.allow_tf32,
        'dictionary_rule': 'per module, zero plus top7 nonzero word IDs in train32 teacher; Hamming tie to first index',
        'validation_does_not_update_dictionary': True,
        'numeric_evaluation': 'actual projected theta*g -> compiled gW -> full A -> thresholds -> real FC2/BN2/shortcut',
        'readouts': 'native final and preds2 sum(T) then bilinear480x640, align_corners=False, no vector rescaling',
        'limits': 'new lossy model, ten validation frames, no training, no RTL/cycle/PPA or frozen-FP32 equivalence claim'})

    # Only six short CUDA histograms survive each forward. No large source cache.
    with torch.no_grad():
        for i, name in enumerate(train, 1):
            functional.reset_net(model)
            x, _, _ = input_frame(args.data, name, targets=False)
            model(x)
            del x
            print('TRAIN_SIGNATURES', i, name, flush=True)
    train_counts = histograms.cpu_counts()
    codebooks, word_maps, selected = make_codebooks(args, histograms, params, train_counts)
    torch.save(params, args.output/'integer_parameters.pt')
    save_json(args.output/'integer_descriptions.json', descriptions)
    current = {}

    def coarse_hook(module, inputs, output):
        current['coarse'] = output.detach().sum(0)

    handle = dict(model.named_modules())['sttmultires_unet.preds.2'].register_forward_hook(coarse_hook)
    rows, summaries, validation_counts = [], {}, {}
    for variant in ('all12_integer_teacher', 'S2_six_sources_code8'):
        histograms.reset()
        histograms.project = variant == 'S2_six_sources_code8'
        values = []
        with torch.no_grad():
            for name in valid:
                functional.reset_net(model)
                x, label, mask = input_frame(args.data, name)
                final = model(x)['flow'][-1]
                coarse = F.interpolate(current.pop('coarse'), size=(480, 640), mode='bilinear', align_corners=False)
                row = {'variant': variant, 'file': name, 'valid_pixels': int(mask.sum()),
                       'final': flow_metric(final, label, mask),
                       'coarse_bilinear': flow_metric(coarse, label, mask)}
                values.append(row)
                rows.append(row)
                print('FRAME', json.dumps(row), flush=True)
                del x, label, mask, final, coarse
        summary = {key: {'AEE_frame_mean': float(np.mean([r[key]['AEE'] for r in values])),
                         'AEE_pixel_mean': sum(r[key]['aee_sum'] for r in values)/sum(r['valid_pixels'] for r in values)}
                   for key in ('final', 'coarse_bilinear')}
        module_stats = {}
        for prefix, counts in histograms.cpu_counts().items():
            emitted = counts.copy()
            if histograms.project:
                emitted.fill(0)
                np.add.at(emitted, word_maps[prefix], counts)
            module_stats[tag(prefix)] = {'raw_source': hist_description(counts),
                                        'emitted_source': hist_description(emitted),
                                        'raw_exact_dictionary_coverage': float(counts[selected[prefix]].sum()/counts.sum())}
            validation_counts[variant+'_'+tag(prefix)+'_raw'] = counts
            validation_counts[variant+'_'+tag(prefix)+'_emitted'] = emitted
        summary['source_statistics'] = module_stats
        summaries[variant] = summary
        save_json(args.output/'frames.json', rows)
    handle.remove()
    histograms.close()
    np.savez_compressed(args.output/'validation_histograms.npz', **validation_counts)
    for key in ('final', 'coarse_bilinear'):
        summaries['S2_six_sources_code8'][key]['delta_AEE_frame_mean_vs_teacher'] = (
            summaries['S2_six_sources_code8'][key]['AEE_frame_mean']-
            summaries['all12_integer_teacher'][key]['AEE_frame_mean'])
    save_json(args.output/'summary.json', {
        'frames_per_variant': len(valid), 'train_codebook_frames': len(train),
        'AEE': summaries, 'codebooks_file': 'codebooks.json',
        'scope': 'simultaneous six-S2 projection against same all12 integer-consumer teacher; no validation fitting',
        'limits': 'all original sn1 producers still execute; no producer saving measured; no duplicate full-U equivalence sweep; ten-frame opportunity gate only'})
    print('COMPLETE', json.dumps({k: {r: v[r] for r in ('final', 'coarse_bilinear')}
                                  for k, v in summaries.items()}), flush=True)


if __name__ == '__main__':
    main()
