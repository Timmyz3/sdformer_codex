"""Evaluate one spatial-R16 integer function through the live residual consumer."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
BASE = Path(os.environ.get('R0_STREAM_BASE', str(STAGE.parent)))
sys.path.insert(0, str(BASE / 'r8_consumer_fusion_20260914/data'))
sys.path.insert(0, str(STAGE / 'spatial_r16_integer'))
from model_access import load_parent
from torch_integer import SpatialR16Integer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=('diverse', 'valid'), required=True)
    ap.add_argument('--variant', choices=('q13', 'q11', 'moment', 'native_tap', 'unconstrained'), default='q13')
    ap.add_argument('--capture-sequences', action='store_true')
    opt = ap.parse_args()
    import torch
    import cupy
    import torch.nn.functional as F

    output = HERE if opt.variant == 'q13' else HERE / opt.variant
    output.mkdir(parents=True, exist_ok=True)
    if opt.variant in ('moment', 'native_tap', 'unconstrained'):
        factor_dir = STAGE / 'spatial_winograd_pruning' / opt.variant
    else:
        factor_dir = STAGE / ('spatial_r16_integer' if opt.variant == 'q13' else 'spatial_winograd_inputs')
    args, net = load_parent(output / ('deployed_' + opt.split))
    args.split = opt.split
    from run_bn_probe import read_names
    from evaluate_branch_control import evaluate_axis
    if opt.variant == 'unconstrained':
        from phase3_endpoint import Phase3Integer
        endpoint = Phase3Integer(factor_dir / 'factors.npz', device='cuda')
    else:
        endpoint = SpatialR16Integer(factor_dir / 'factors.npz', device='cuda')
    gold = np.load(factor_dir / 'gold_tiles.npz')
    ref = np.load(BASE / 'r8_consumer_fusion_20260914/data/consumer_first8.npz')
    r0name = net.BLOCK.rsplit('.', 1)[0] + '.0'
    r0 = net.modules[r0name]
    conv = net.modules[r0name + '.conv2.0']
    bn = net.modules[r0name + '.norm2.norm_layer']
    assert tuple(conv.stride) == (1, 1) and tuple(conv.padding) == (1, 1)
    assert not bn.training and bn.track_running_stats
    for key, tensor in [('bn_gamma', bn.weight), ('bn_beta', bn.bias),
                        ('bn_running_mean', bn.running_mean), ('bn_running_var', bn.running_var)]:
        assert np.array_equal(ref[key], tensor.detach().cpu().numpy()), key
    rows, state, captured, capture_indices = [], {}, [], set()
    original = conv.forward

    def flat(x):
        return x[:, 0] if x.ndim == 5 else x

    def pre(module, inputs):
        state['identity'] = flat(inputs[0]).detach()

    def forward(x):
        g = flat(x)
        first_gold = not rows and opt.split == 'diverse'
        capture = len(rows) in capture_indices
        if first_gold:
            diagnostic_z = F.conv2d(g.ne(0).double(), endpoint.q1_fp64, padding=(1, 0))
            state['cudnn_diagnostic'] = dict(
                noninteger_max_abs=float((diagnostic_z - diagnostic_z.round()).abs().max()),
                noninteger_values=int((diagnostic_z != diagnostic_z.round()).sum()),
                outside_static_bound=int(((diagnostic_z < endpoint.z_lower) | (diagnostic_z > endpoint.z_upper)).sum()))
            del diagnostic_z
        # cuDNN may choose a transformed FP64 convolution with noninteger
        # roundoff. This oracle needs direct integer-valued products/sums.
        # The context is local: all other model operators retain their setup.
        with torch.backends.cudnn.flags(enabled=False):
            if first_gold or capture:
                p, z = endpoint.raw_p(g, return_z=True)
                state['source'] = g.detach()
                state['z'] = z
            else:
                p = endpoint.raw_p(g)
        state['p'] = p
        y = (p.double() * endpoint.output_scale + endpoint.bias).to(x.dtype)
        return y[:, None] if x.ndim == 5 else y

    def check_tiles(identity, values):
        origins = gold['output_origin_yx']
        def extract(tensor):
            return torch.stack([tensor[:, :, int(y):int(y)+2, int(x):int(x)+2]
                                for y, x in origins]).cpu().numpy()
        for key, field in [('p', 'p_int'), ('J', 'J_q20'), ('wide', 'wide_int64'), ('i24', 'i24')]:
            assert np.array_equal(extract(values[key]), gold[field]), key
        actual_identity = extract(identity).astype(np.float32).view(np.uint32)
        assert np.array_equal(actual_identity, gold['identity_fp32_bits']), 'identity bits'
        zpad = F.pad(state.pop('z'), (1, 1, 0, 0))
        ztiles = torch.stack([zpad[:, :, int(y):int(y)+2, int(x):int(x)+4]
                              for y, x in origins]).cpu().numpy()
        assert np.array_equal(ztiles, gold['z_halo_int']), 'Z'
        srcpad = F.pad(state.pop('source').ne(0), (1, 1, 1, 1))
        src = torch.stack([srcpad[:, :, int(y):int(y)+4, int(x):int(x)+4]
                           for y, x in origins]).cpu().numpy()
        words = np.zeros(src.shape[:1] + src.shape[2:], dtype=np.uint16)
        for t in range(10):
            words |= src[:, t].astype(np.uint16) << t
        assert np.array_equal(words, gold['source_words']), 'source words'
        return dict(tiles=len(origins), raw_values=int(gold['p_int'].size), differences=0,
                    checked=['live_source', 'identity_bits', 'Z', 'p', 'J', 'wide', 'I24'])

    def post(module, inputs, original_output):
        identity, p = state.pop('identity'), state.pop('p')
        values = endpoint.consume(p, identity, return_intermediates=True)
        i24, j, wide = values['i24'], values['J'], values['wide']
        jr = (identity.double() * (1 << 20)).round()
        rec = dict(frame_index=len(rows), identity_saturations=int((jr != j).sum()),
                   i24_min=int(i24.min()), i24_max=int(i24.max()),
                   p_min=int(p.min()), p_max=int(p.max()), wide_min=int(wide.min()), wide_max=int(wide.max()))
        if not rows and opt.split == 'diverse':
            rec['cudnn_diagnostic'] = state.pop('cudnn_diagnostic')
            rec['live_gold'] = check_tiles(identity, values)
            print('LIVE_SPATIAL_GOLD_OK', json.dumps(rec), flush=True)
        if len(rows) in capture_indices:
            zpad = F.pad(state.pop('z'), (1, 1, 0, 0))
            source = F.pad(state.pop('source').ne(0), (1, 1, 1, 1))
            for tile in (128, 9664):
                y, x = 2*(tile//160), 2*(tile%160)
                src = source[:, :, y:y+4, x:x+4].cpu().numpy()
                words = np.zeros((96, 4, 4), dtype=np.uint16)
                for t in range(10):
                    words |= src[t].astype(np.uint16) << t
                item = dict(frame_index=len(rows), file=names[len(rows)], tile_id=tile,
                            source_words=words, output_origin_yx=np.array([y, x], np.int32),
                            z_halo_int=zpad[:, :, y:y+2, x:x+4].cpu().numpy().astype(np.int16),
                            identity_fp32_bits=identity[:, :, y:y+2, x:x+2].cpu().numpy().view(np.uint32))
                for key, field in [('p', 'p_int'), ('J', 'J_q20'), ('wide', 'wide_int64'), ('i24', 'i24')]:
                    item[field] = values[key][:, :, y:y+2, x:x+2].cpu().numpy().astype(np.int64 if key == 'wide' else np.int32)
                captured.append(item)
        rows.append(rec)
        state['expected_i24'] = i24
        value = endpoint.reader_value(i24)
        return value[:, None] if original_output.ndim == 5 else value

    def reader(module, inputs, output):
        assert torch.equal(net.helper.i, state.pop('expected_i24').double()), 'live I24 reader differs'

    hooks = [r0.register_forward_pre_hook(pre), r0.register_forward_hook(post),
             net.helper.source.register_forward_hook(reader)]
    conv.forward = forward
    names = (read_names(args.data, 'valid') if opt.split == 'valid' else
             json.loads((BASE / 'algorithm/samples.json').read_text())['valid'][:10])
    assert len(names) == (825 if opt.split == 'valid' else 10)
    if opt.capture_sequences:
        assert opt.split == 'valid'
        seen = set()
        for index, name in enumerate(names):
            sequence = name.rsplit('_', 1)[0]
            if sequence not in seen:
                seen.add(sequence)
                capture_indices.add(index)
    threshold = 1.447936665574317 if opt.split == 'valid' else 1.4699681144337489
    report = dict(complete=False, split=opt.split, variant=opt.variant, frames=names, python=sys.version,
                  torch=torch.__version__, cupy=cupy.__version__, gpu=torch.cuda.get_device_name(0),
                  TF32_matmul=torch.backends.cuda.matmul.allow_tf32, TF32_cudnn=torch.backends.cudnn.allow_tf32,
                  target=r0name + '.conv2.0', function=f'spatial R16 q1i8 -> Z15 -> {opt.variant} -> p32 -> J20+BN -> I24',
                  arithmetic='FP64 convolution of exact integers; no intermediate quantizer; signed64 endpoint RNE26/satI24',
                  oracle_cudnn=False,
                  old_FP_BN_add_executed_as_ignored_shadow=True, fullnet_bittrue=False,
                  training=False, format_search=False, matched_environment_NB0_frame_AEE=threshold,
                  pruning_policy=opt.variant if opt.variant in ('moment', 'native_tap', 'unconstrained') else None,
                  phase_dependent_operator=opt.variant == 'unconstrained',
                  NB0_note='local upstream NB0 ep29 full output; candidate matched-dense student with existing coarse head',
                  per_frame_integer_checks=rows)
    def save():
        (output / ('deployed_' + opt.split + '.json')).write_text(json.dumps(report, indent=2) + '\n')
    save()
    try:
        result = evaluate_axis(args, net.model, net.current, names, 'spatial_integer', progress_tag='SPATIAL_INTEGER')
        report.update(complete=True, result=result, below_matched_NB0=result['AEE_frame_mean'] < threshold)
        if captured:
            fields = ['source_words', 'output_origin_yx', 'z_halo_int', 'identity_fp32_bits', 'p_int', 'J_q20', 'wide_int64', 'i24']
            arrays = {field: np.stack([row[field] for row in captured]) for field in fields}
            np.savez_compressed(output / 'sequence_tiles.npz', **arrays)
            manifest = [{key: row[key] for key in ('frame_index', 'file', 'tile_id')} for row in captured]
            (output / 'sequence_tiles.json').write_text(json.dumps(manifest, indent=2) + '\n')
            report['live_sequence_capture'] = dict(sequences=len(capture_indices), tiles=len(captured),
                                                 origins='fixed edge tile128 and interior tile9664; no adaptive selection')
        save()
    except Exception as exc:
        report['error'] = repr(exc)
        save()
        raise
    finally:
        conv.forward = original
        for hook in hooks:
            hook.remove()
        gold.close()
        net.close()


if __name__ == '__main__':
    main()
