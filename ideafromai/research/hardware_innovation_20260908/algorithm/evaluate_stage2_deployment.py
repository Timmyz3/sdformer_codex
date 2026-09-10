"""Paired valid825: saved all12 integer consumers, then six S2 code consumers.

Both modes stop at the supervised preds.2 readout. The student computes seven
class partial sums and consumes B=A@D directly; FC2, BN2, and shortcuts remain.
Only the first student frame compares each S2 U against projected-gW then A.
Capture student codes and raw source histograms for the first ten official
validation frames, without saving Y/U or collecting training frames again.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import types

from run_bn_probe import (ALL_FC1, MLPS, build_model, input_frame,
                          read_names, save_json, set_bn_mode)
import numpy as np
import torch
import torch.nn.functional as F

S2 = [p for p in MLPS if '.layers.2.' in p]
TEACHER = 'all12_integer_teacher'
STUDENT = 'S2_six_sources_code8_direct_B'


class CoarseReady(Exception):
    pass


def tag(prefix):
    return 's2b'+prefix.split('.swin_blocks.')[1].split('.')[0]


def output_gate(u, q):
    gate = torch.where(q['positive'][None, None, :],
                       u >= q['tau'][:, None, :], u <= q['tau'][:, None, :])
    return torch.where(q['variable'][None, None, :], gate, q['fixed'][:, None, :])


def install_saved_consumers(model, params):
    """Install the exact exported arithmetic of run_integer_fc1_probe.py."""
    modules, gpu = dict(model.named_modules()), {}
    for prefix in MLPS:
        r = params[prefix]
        q = {'w': r['weight_int8'].cuda().float(),
             'a': r['temporal_int16'].cuda().double(),
             'tau': r['threshold_int64'].cuda().double(),
             'positive': r['positive_gain'].cuda(),
             'variable': (~r['constant_channels']).cuda(),
             'fixed': r['constant_gate'].cuda(),
             'theta_source': r['theta_source'].cuda().float(),
             'theta_output': r['theta_output'].cuda().float()}
        gpu[prefix] = q

        def linear(self, x, compiled=q):
            return F.linear(x/compiled['theta_source'], compiled['w'])

        def passthrough(self, x):
            return x

        def temporal(self, x, compiled=q):
            shape = x.shape
            u = (compiled['a'] @ x.double().reshape(shape[0], -1)).reshape(shape[0], -1, shape[-1])
            return output_gate(u, compiled).reshape(shape).float()*compiled['theta_output']

        fc, bn, sn = (modules[prefix+'fc1'], modules[prefix+'bn1.norm_layer'],
                      modules[prefix+'sn2.spiking_neuron'])
        assert tuple(q['w'].shape) == tuple(fc.weight.shape)
        fc.forward = types.MethodType(linear, fc)
        bn.forward = types.MethodType(passthrough, bn)
        sn.forward = types.MethodType(temporal, sn)
    return gpu


def install_code_consumers(model, params, gpu, args, current, checks, summary):
    modules = dict(model.named_modules())
    (args.output/'capture').mkdir(parents=True, exist_ok=True)
    with np.load(args.codebooks) as book:
        for prefix in S2:
            name, q = tag(prefix), gpu[prefix]
            words = book[name+'_words'].astype(np.int64)
            mapping = book[name+'_word_map'].astype(np.int64)
            dictionary = book[name+'_dictionary'].astype(np.int64)
            aq = params[prefix]['temporal_int16'].numpy().astype(np.int64)
            b = book[name+'_B_int32'].astype(np.int64)
            assert np.array_equal(aq, book[name+'_A_int16'])
            assert np.array_equal(b, aq @ dictionary.T)
            assert words[0] == 0 and np.all(b[:, 0] == 0)
            word_to_code = np.full(1024, -1, dtype=np.int64)
            word_to_code[words] = np.arange(len(words))
            code_map = word_to_code[mapping]
            assert np.all(code_map >= 0)
            q['code_map'] = torch.from_numpy(code_map).cuda()
            q['dictionary'] = torch.from_numpy(dictionary).cuda().float()
            q['b'] = torch.from_numpy(b).cuda().double()
            q['words'] = words
            q['K'] = len(words)
            q['C'], q['H'] = int(q['w'].shape[1]), int(q['w'].shape[0])

            def forward(self, x, compiled=q, module_name=prefix, module_tag=name):
                # Preserve the original producer and the real MLP consumer path.
                if self.norm_layer in ('LN', 'GN'):
                    x = self.norm(x.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
                source = self.drop1(self.sn1(x))
                shape, t, c = source.shape, source.shape[0], compiled['C']
                assert shape[-1] == c and t == compiled['a'].shape[0]
                raw_gate = source != 0
                shifts = torch.arange(t, device=source.device, dtype=torch.int64)
                shifts = shifts.reshape(t, *([1]*(source.ndim-1)))
                raw_words = torch.sum(raw_gate.to(torch.int64) << shifts, dim=0)
                codes = compiled['code_map'][raw_words].reshape(-1, c)
                membership = F.one_hot(codes, num_classes=compiled['K'])[:, :, 1:].float().permute(0, 2, 1)
                partials = membership @ compiled['w'].T
                u = torch.einsum('tk,pkh->tph', compiled['b'][:, 1:], partials.double())

                if module_name not in checks:
                    projected_g = compiled['dictionary'][codes].permute(2, 0, 1)
                    yi = F.linear(projected_g, compiled['w'])
                    direct = (compiled['a'] @ yi.double().reshape(t, -1)).reshape_as(u)
                    check = {'file': current['file'], 'values': u.numel(),
                             'mismatches': int(u.ne(direct).sum()),
                             'max_abs_difference': float((u-direct).abs().max()),
                             'raw_theta_g_max_residual': float((source-raw_gate*compiled['theta_source']).abs().max()),
                             'comparison': 'same projected g: classS->B versus ordinary gW->A; no unprojected-model equivalence claim'}
                    checks[module_name] = check
                    print('FIRST_FRAME_U', module_tag, json.dumps(check), flush=True)
                    if check['mismatches'] or check['raw_theta_g_max_residual']:
                        summary['status'] = 'stopped_on_first_frame_integer_check'
                        save_json(args.output/'summary.json', summary)
                        raise RuntimeError('S2 integer implementation check failed: '+module_name)
                    del projected_g, yi, direct

                if current['valid_index'] < 10:
                    counts = torch.bincount(raw_words.reshape(-1), minlength=1024)
                    filename = f"v{current['valid_index']:03d}_{Path(current['file']).stem}_{module_tag}.npz"
                    np.savez_compressed(args.output/'capture'/filename,
                        codes=codes.cpu().numpy().astype(np.uint8),
                        raw_source_signature_histogram=counts.cpu().numpy(),
                        source_shape=np.array(shape, dtype=np.int32),
                        theta_source=np.float32(compiled['theta_source'].item()),
                        theta_output=np.float32(compiled['theta_output'].item()),
                        dictionary_words=compiled['words'].astype(np.uint16))

                hidden = output_gate(u, compiled).reshape(*shape[:-1], compiled['H']).float()*compiled['theta_output']
                out = self.fc2(self.drop2(hidden))
                if self.norm_layer in ('BN', 'BNTT', 'tdBN', 'IN'):
                    out = self.bn2(out.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
                return out

            mlp = modules[prefix.rstrip('.')]
            mlp.forward = types.MethodType(forward, mlp)


def summarize(rows, complete):
    return {'frames': len(rows), 'complete': complete,
            'valid_pixels': sum(r['valid_pixels'] for r in rows),
            'AEE_frame_mean': float(np.mean([r['AEE'] for r in rows])),
            'AEE_pixel_mean': sum(r['aee_sum'] for r in rows)/sum(r['valid_pixels'] for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('code-root', 'config', 'checkpoint', 'data', 'calibration', 'codebooks', 'parameters'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'stage2_deployment_valid825')
    parser.add_argument('--limit', type=int, choices=(1, 825), default=825,
                        help='1 is only a first-frame implementation check; normal evaluation is 825.')
    parser.add_argument('--capture-samples', type=Path,
                        help='Student-only ten-frame capture on the existing diverse sample list, after full825 evaluation.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = read_names(args.data, 'valid')
    assert len(names) == 825
    if args.capture_samples:
        names = json.loads(args.capture_samples.read_text())['valid'][:10]
        modes = (STUDENT,)
    else:
        names = names[:args.limit]
        modes = (TEACHER, STUDENT)
    params = torch.load(args.parameters, map_location='cpu', weights_only=False)
    stats = torch.load(args.calibration, map_location='cpu', weights_only=False)
    model, cfg, installed, attention = build_model(args)
    model.requires_grad_(False)
    from spikingjelly.activation_based import functional
    set_bn_mode(model, stats, ALL_FC1)
    gpu = install_saved_consumers(model, params)
    current, checks, rows = {}, {}, []
    summary = {
        'status': 'running', 'results': {}, 'integer_U_checks': checks,
        'checkpoint': str(args.checkpoint), 'config': str(args.config),
        'calibration': str(args.calibration), 'parameters': str(args.parameters), 'codebooks': str(args.codebooks),
        'modes': list(modes), 'expected_frames_per_mode': len(names),
        'validation_scope': ('diverse ten-frame hardware capture' if args.capture_samples
                             else 'full825' if len(names) == 825 else 'first-frame implementation check only'),
        'integer_consumer_prefixes': MLPS, 'student_source_prefixes': S2,
        'calibration_and_dictionary_refitting': False, 'optimization_updates': 0,
        'readout': 'preds.2 sum(T) -> bilinear480x640 align_corners=False; no flow-vector rescaling; both stop before D3',
        'teacher': 'saved all12 FC1/sn2 integer consumers, fixed ALL_FC1 BN1; remaining network keeps its original arithmetic',
        'student': 'original FP32 full-T sn1 -> fixed train-only code -> seven shared S classes -> B -> actual theta_output*g -> original FC2/BN2/shortcut',
        'installed_atlif': len(installed), 'attention_blocks': len(attention),
        'attention_hardware_quant_enabled': cfg['bsa_attention']['hardware_quant_enabled'],
        'motion_alpha': cfg['bsa_attention']['binary_motion_xor_alpha'],
        'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
        'tf32_cudnn': torch.backends.cudnn.allow_tf32,
        'capture': {'mode': STUDENT, 'files': names[:10], 'modules': S2,
                    'format': 'codes[P,C] uint8 indices in saved dictionary; raw unprojected source-g 1024-bin counts; source shape and theta',
                    'raw_source_context': 'actual sn1 under this student prefix, before its local projection; not an independent native-network capture',
                    'no_Y_or_U_saved': True},
        'claim': 'new-model AEE for the declared validation scope and first-frame integer implementation checks; GPU class-membership emulation is not hardware cycles/PPA or free producer work'}
    save_json(args.output/'summary.json', summary)

    def stop_at_coarse(module, inputs, output):
        current['flow'] = output.detach().sum(0)
        raise CoarseReady()

    handle = dict(model.named_modules())['sttmultires_unet.preds.2'].register_forward_hook(stop_at_coarse)
    for mode in modes:
        if mode == STUDENT:
            install_code_consumers(model, params, gpu, args, current, checks, summary)
        mode_rows = []
        with torch.no_grad():
            for i, name in enumerate(names):
                current['valid_index'], current['file'] = i, name
                functional.reset_net(model)
                x, label, mask = input_frame(args.data, name)
                try:
                    model(x)
                except CoarseReady:
                    pred = F.interpolate(current.pop('flow'), size=(480, 640), mode='bilinear', align_corners=False)
                error = torch.linalg.vector_norm(
                    pred.permute(0, 2, 3, 1)[mask]-label.permute(0, 2, 3, 1)[mask], dim=1)
                total, count = float(error.double().sum()), error.numel()
                row = {'mode': mode, 'valid_index': i, 'file': name, 'valid_pixels': count,
                       'AEE': total/count, 'aee_sum': total}
                rows.append(row)
                mode_rows.append(row)
                if i == 0 and mode == STUDENT:
                    save_json(args.output/'summary.json', summary)
                if (i+1) % 50 == 0 or i+1 == len(names):
                    summary['results'][mode] = summarize(mode_rows, i+1 == len(names))
                    save_json(args.output/'frames.json', rows)
                    save_json(args.output/'summary.json', summary)
                    print('VALID', mode, i+1, json.dumps(summary['results'][mode]), flush=True)
                del x, label, mask, pred, error
        # Publish a completed stage before changing the next execution mode.
        summary['results'][mode] = summarize(mode_rows, True)
        save_json(args.output/'frames.json', rows)
        save_json(args.output/'summary.json', summary)
    handle.remove()
    summary['status'] = 'complete'
    if TEACHER in summary['results']:
        summary['student_delta_AEE_frame_mean'] = (summary['results'][STUDENT]['AEE_frame_mean']-
                                                  summary['results'][TEACHER]['AEE_frame_mean'])
    save_json(args.output/'summary.json', summary)
    print('COMPLETE', json.dumps(summary['results']), flush=True)


if __name__ == '__main__':
    main()
