"""Independent NumPy-int64 versus IntegerLatentPair CPU numerical checks.

Real source image patches include global boundaries and an active interior.
The pure integer reference does not call compiler functional()/predict().
No GPU, parameter changes, integer-network AEE or hardware timing claim.
"""
from fractions import Fraction
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from integer_adapter import IntegerLatentPair

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent.parent/'joint_completion_20260909/full_capture4/capture/00_optimized_prefix_train16_row34_packed_word_exact/000_zurich_city_09_a_0001/gates.npz'
NAMES = ('shared56_u8_vq5', 'private56_u8_vq5', 'shared48_u8_vq5')


def read_arrays(path):
    with np.load(path) as data:
        return {key: data[key].copy() for key in data.files}


def group(values):
    # T,C,H,W -> G=H*(W/4),T,P4,C; original horizontal positions.
    t, c, height, width = values.shape
    return values.reshape(t, c, height, width//4, 4).transpose(2, 3, 0, 4, 1).reshape(-1, t, 4, c)


def image(values, height, width):
    return values.reshape(height, width//4, 10, 4, 96).transpose(2, 4, 0, 1, 3).reshape(10, 96, height, width)


def source_columns(bits):
    padded = np.pad(bits, ((0, 0), (0, 0), (1, 1), (1, 1)))
    height, width = bits.shape[-2:]
    # Explicit c,kh,kw order, independent of torch unfold/Conv2d.
    columns = np.stack([padded[:, c, kh:kh+height, kw:kw+width]
        for c in range(96) for kh in range(3) for kw in range(3)], -1)
    return columns.astype(np.int64)


def reference(arrays, bits):
    source = source_columns(bits)
    shared = int(arrays['shared_rank'])
    z = source @ arrays['u_int8'].astype(np.int64)
    v, a = arrays['integer_v_align_coeff'], arrays['integer_a_q14'].astype(np.int64)
    ys = z[..., :shared] @ v[:shared]
    yt = z[..., shared:] @ v[shared:]
    y = ys+yt
    yg, sg = group(y.transpose(0, 3, 1, 2)), group(ys.transpose(0, 3, 1, 2))
    empty_image = ~source.any(-1)
    empty = group(empty_image[:, None])[:, :, :, 0]
    full_u = np.einsum('ts,gsph->gtph', a, yg, dtype=np.int64)
    shared_u = np.einsum('ts,gsph->gtph', a, sg, dtype=np.int64)
    full = full_u >= arrays['integer_full_threshold'][None, :, None, :]
    pos, neg, codes = predicates(arrays, shared_u, empty)
    accepted = pos | neg
    conditional = np.where(accepted, pos, full)
    return dict(z=z, y=yg, shared_y=sg, full_u=full_u, shared_u=shared_u,
        empty=empty, full=full, accepted=accepted, conditional=conditional,
        codes=codes, columns=source)


def predicates(arrays, shared_u, empty):
    pos, neg = np.zeros_like(shared_u, dtype=bool), np.zeros_like(shared_u, dtype=bool)
    codes = []
    for t in range(10):
        code = np.zeros(empty.shape[::2], np.int64)  # G,P
        # Deliberately scalar bit construction to cross-check adapter packing.
        for bit, source_t in enumerate(arrays[f'integer_empty_indices_t{t}']):
            code += empty[:, int(source_t)].astype(np.int64)*(1 << bit)
        pos[:, t] = shared_u[:, t] >= arrays[f'integer_threshold_pos_t{t}'][code]
        neg[:, t] = shared_u[:, t] <= arrays[f'integer_threshold_neg_t{t}'][code]
        codes.append(code)
    return pos, neg, codes


def independent_domains_and_thresholds(arrays):
    u, v, a = arrays['u_int8'].astype(np.int64), arrays['integer_v_align_coeff'], arrays['integer_a_q14'].astype(np.int64)
    zlo, zhi = np.minimum(u, 0).sum(0), np.maximum(u, 0).sum(0)
    zabs = np.maximum(-zlo, zhi)
    yabs = zabs @ np.abs(v)
    uabs = np.abs(a).sum(1)[:, None]*yabs[None, :]
    thresholds = [arrays['integer_full_threshold']]
    thresholds += [arrays[f'integer_threshold_{kind}_t{t}'] for t in range(10) for kind in ('pos', 'neg')]
    threshold_abs = max(int(np.abs(value).max()) for value in thresholds)
    maximum = max(int(uabs.max()), threshold_abs)
    full_errors = conditional_errors = 0
    def f(value):
        return Fraction.from_float(float(value))
    def p2(value):
        return Fraction(2**value) if value >= 0 else Fraction(1, 2**(-value))
    def ceil(value):
        return -((-value.numerator)//value.denominator)
    for h in range(96):
        kappa = f(arrays['bn_scale'][h])*p2(int(arrays['integer_y_exponent'][h])-14)
        if kappa <= 0:
            raise ValueError('The supplied deployment explicitly compiles positive gains only.')
        for t in range(10):
            offset = f(arrays['bn_bias'][h])*Fraction(int(a[t].sum()), 16384)+f(arrays['temporal_bias'][t])-f(arrays['theta_output'])
            full_errors += ceil(-offset/kappa) != int(arrays['integer_full_threshold'][t, h])
            means, radii = arrays[f'integer_predictor_mean_t{t}'], arrays[f'integer_predictor_radius_t{t}']
            for i in range(len(means)):
                center = offset+f(means[i, h])
                positive = ceil((f(radii[i, h])-center)/kappa)
                negative = ((-f(radii[i, h])-center)/kappa).__floor__()
                conditional_errors += (positive != int(arrays[f'integer_threshold_pos_t{t}'][i, h]) or
                    negative != int(arrays[f'integer_threshold_neg_t{t}'][i, h]))
    return dict(Zi_all_source_min=int(zlo.min()), Zi_all_source_max=int(zhi.max()),
        Zi_FP32_prefix_exact=bool(max(int(-zlo.min()), int(zhi.max())) < 2**24),
        Yi_any_reduction_absolute_bound=int(yabs.max()),
        Ui_any_reduction_absolute_bound=int(uabs.max()),
        threshold_max_absolute=threshold_abs, all_FP64_integer_operations_below_2pow53=maximum < 2**53,
        full_BN_once_threshold_errors=int(full_errors), conditional_threshold_errors=int(conditional_errors),
        formula='margin=bn_scale*2^(y_exp-14)*Ui + bn_bias*sum(Aq_row)/16384 + temporal_bias - theta_output; one BN affine only')


def directed_table_checks(arrays):
    # Synthetic thresholds only test interface control. They are never saved
    # into the actual compiled model or mistaken for reachable real margins.
    synthetic = {key: value.copy() for key, value in arrays.items()}
    synthetic['integer_full_threshold'].fill(1)
    for t in range(10):
        count = len(synthetic[f'integer_threshold_pos_t{t}'])
        pattern = np.arange(count)[:, None]
        bit = np.arange(96)[None, :] % max(len(synthetic[f'integer_empty_indices_t{t}']), 1)
        active = ((pattern >> bit) & 1).astype(bool)
        synthetic[f'integer_threshold_pos_t{t}'] = np.where(active, 0, 1).astype(np.int64)
        synthetic[f'integer_threshold_neg_t{t}'] = np.where(active, -1, 0).astype(np.int64)
    # Every full ten-bit source-empty pattern appears, distributed over P4.
    empty = ((np.arange(1024)[:, None] >> np.arange(10)) & 1).reshape(256, 4, 10).transpose(0, 2, 1).astype(bool)
    zeros = np.zeros((256, 10, 4, 96), np.int64)
    pair = IntegerLatentPair(synthetic, 'cpu', True)
    gate, accept = pair.forward_groups(torch.from_numpy(zeros).double(), torch.from_numpy(zeros).double(), torch.from_numpy(empty))
    pos, neg, _ = predicates(synthetic, zeros, empty)
    errors = int(np.count_nonzero(gate.numpy() != pos))+int(np.count_nonzero(accept.numpy() != (pos | neg)))
    # Force simultaneous positive/negative acceptance with a negative full
    # fallback, so selecting the wrong priority changes every output bit.
    for t in range(10):
        synthetic[f'integer_threshold_pos_t{t}'].fill(0)
        synthetic[f'integer_threshold_neg_t{t}'].fill(0)
    pair = IntegerLatentPair(synthetic, 'cpu', True)
    gate, accept = pair.forward_groups(torch.from_numpy(zeros[:1]).double(), torch.from_numpy(zeros[:1]).double(), torch.from_numpy(empty[:1]))
    return dict(all_empty_patterns=1024, table_gate_and_accept_checks=int(2*zeros.size),
        table_bit_order_errors=errors, positive_priority_tie_cases=gate.numel(),
        positive_priority_errors=int((~gate | ~accept).sum()),
        scope='Directed interface-only threshold substitutions in memory; actual compiled files unchanged')


def directed_theta_check(arrays, source):
    # Amplitude and decision threshold are separate interfaces. Preserve
    # the integer decision function, while supplying a nonunit source theta
    # and a distinct nonunit output amplitude; no saved model is changed.
    synthetic = {key: value.copy() for key, value in arrays.items()}
    synthetic['theta_source'] = np.array(.25)
    synthetic['theta_output'] = np.array(.375)
    sample = source[:, :, :4, :4]
    reference_ = reference(synthetic, sample)
    pair = IntegerLatentPair(synthetic, 'cpu', True)
    displayed = pair.conv_forward(torch.from_numpy(sample).float().unsqueeze(1)*.25)
    actual = pair.neuron_forward(displayed).numpy()[:, 0]
    expected = image(reference_['conditional'].astype(np.float32)*.375, 4, 4)
    return dict(source_theta=.25, output_theta=.375, gates=expected.size,
        output_amplitude_and_gate_errors=int(np.count_nonzero(actual != expected)),
        source_amplitude_residual=pair.numeric_checks['source_theta_residual'],
        scope='Directed protocol test with fixed integer predicates; tau is not silently replaced with output theta; actual compiled model unchanged')


def main():
    torch.set_num_threads(4)
    with np.load(SOURCE) as saved:
        shape = tuple(saved['source_gate_shape'])
        bits = np.unpackbits(saved['source_gate_bits'], bitorder=str(saved['bitorder'].item()), count=int(np.prod(shape))).reshape(shape)
        name, theta = str(saved['frame_name'].item()), float(saved['theta_source'])
    # Pick activity using inputs only, solely to exercise nonzero arithmetic.
    count = bits.sum((0, 1))
    iy, ix = np.unravel_index(np.argmax(count[5:-5, 8:-8]), count[5:-5, 8:-8].shape)
    iy, ix = iy+5, ix+8
    y0 = max(1, min(iy-4, 229))
    x0 = max(4, min((ix//4)*4-4, 304))
    regions = [('top_left', 0, 0), ('bottom_right', 230, 308), ('active_interior', int(y0), int(x0))]
    result = dict(complete=False, source=str(SOURCE), frame=name,
        scope='Three real input 10x12 spatial patches, T10/C96, three integer students, CPU adapter full/conditional versus independent NumPy int64. Patch outer crop edge has zero extension; native-valid inner outputs and actual global boundaries are separately identified.',
        models={})
    original_conv2d = F.conv2d
    with torch.no_grad():
        for student in NAMES:
            arrays = read_arrays(HERE/'integer_factors'/(student+'.npz'))
            if float(arrays['theta_source']) != theta:
                raise ValueError('Captured theta differs from the folded source coefficient contract')
            rows = []
            for label, sy, sx in regions:
                sample = bits[:, :, sy:sy+10, sx:sx+12].copy()
                ref = reference(arrays, sample)
                height, width = sample.shape[-2:]
                x = torch.from_numpy(sample).float().unsqueeze(1)*theta
                # Native-valid subregion excludes artificial crop borders;
                # true image boundary still uses the actual zero padding.
                native = np.ones((height, width), bool)
                if sy: native[0] = False
                if sy+height < shape[-2]: native[-1] = False
                if sx: native[:, 0] = False
                if sx+width < shape[-1]: native[:, -1] = False
                record = dict(region=label, source_y=sy, source_x=sx,
                    source_shape=list(sample.shape), active_source_bits=int(sample.sum()),
                    native_valid_output_pixels=int(native.sum()), differences={}, modes={})
                for conditional in (False, True):
                    pair = IntegerLatentPair(arrays, 'cpu', conditional)
                    captured_z = []
                    def tap(input_, weight_, *args, **kwargs):
                        value = original_conv2d(input_, weight_, *args, **kwargs)
                        captured_z.append(value.clone())
                        return value
                    with patch('integer_adapter.F.conv2d', side_effect=tap):
                        display = pair.conv_forward(x)
                    actual_z = captured_z[0].permute(0, 2, 3, 1).numpy()
                    actual_y = group(pair.full_y[:, 0].numpy())
                    actual_s = group(pair.shared_y[:, 0].numpy())
                    actual_empty = group(pair.empty[:, None].numpy())[:, :, :, 0]
                    actual_u = torch.einsum('ts,gsph->gtph', pair.a, torch.from_numpy(actual_y)).numpy()
                    actual_su = torch.einsum('ts,gsph->gtph', pair.a, torch.from_numpy(actual_s)).numpy()
                    actual_gate, actual_accept = pair.forward_groups(torch.from_numpy(actual_y), torch.from_numpy(actual_s), torch.from_numpy(actual_empty))
                    expected = ref['conditional'] if conditional else ref['full']
                    expected_accept = ref['accepted'] if conditional else np.zeros_like(expected)
                    diffs = dict(Zi=int(np.count_nonzero(actual_z != ref['z'])), Yi=int(np.count_nonzero(actual_y != ref['y'])),
                        shared_Yi=int(np.count_nonzero(actual_s != ref['shared_y'])),
                        Aq_Ui=int(np.count_nonzero(actual_u != ref['full_u'])),
                        shared_Aq_Ui=int(np.count_nonzero(actual_su != ref['shared_u'])),
                        source_empty=int(np.count_nonzero(actual_empty != ref['empty'])),
                        group_gate=int(np.count_nonzero(actual_gate.numpy() != expected)),
                        group_accept=int(np.count_nonzero(actual_accept.numpy() != expected_accept)))
                    # The real wrapper presents a BN-transformed view. The
                    # neuron must consume its stored integer state instead.
                    bn_view = display*torch.tensor(arrays['bn_scale']).float()[None, None, :, None, None]+torch.tensor(arrays['bn_bias']).float()[None, None, :, None, None]
                    native_output = pair.neuron_forward(bn_view).numpy()[:, 0]
                    native_expected = image(expected.astype(np.float32)*float(arrays['theta_output']), height, width)
                    diffs['native_TCHW_gate'] = int(np.count_nonzero(native_output != native_expected))
                    diffs['native_valid_region_gate'] = int(np.count_nonzero((native_output != native_expected)[..., native]))
                    # A deliberately changed displayed view cannot double
                    # apply normalization, because it is not the operand.
                    pair.full_y = torch.from_numpy(image(ref['y'], height, width)).double().unsqueeze(1)
                    pair.shared_y = torch.from_numpy(image(ref['shared_y'], height, width)).double().unsqueeze(1)
                    pair.empty = torch.from_numpy((~ref['columns'].any(-1)))
                    ignored = pair.neuron_forward(bn_view*7+3).numpy()[:, 0]
                    diffs['displayed_BN_applied_again'] = int(np.count_nonzero(ignored != native_expected))
                    mode = 'conditional' if conditional else 'full'
                    record['modes'][mode] = dict(gates=expected.size, accepted=int(expected_accept.sum()),
                        Zi_max_abs=int(np.abs(ref['z']).max()), Yi_max_abs=int(np.abs(ref['y']).max()),
                        Ui_max_abs=int(np.abs(ref['full_u']).max()), differences=diffs,
                        theta_checks=pair.numeric_checks)
                    if any(diffs.values()):
                        raise AssertionError(student+' '+label+' '+mode+' '+json.dumps(diffs))
                rows.append(record)
            domain = independent_domains_and_thresholds(arrays)
            directed = directed_table_checks(arrays)
            theta_check = directed_theta_check(arrays, bits[:, :, int(y0):int(y0)+10, int(x0):int(x0)+12])
            if domain['full_BN_once_threshold_errors'] or domain['conditional_threshold_errors'] or not domain['all_FP64_integer_operations_below_2pow53'] or directed['table_bit_order_errors'] or directed['positive_priority_errors'] or theta_check['output_amplitude_and_gate_errors'] or theta_check['source_amplitude_residual']:
                raise AssertionError(student+': threshold/domain/interface failure')
            result['models'][student] = dict(real_regions=rows, static_domain_and_BN=domain,
                directed=directed, directed_theta=theta_check)
            (HERE/'integer_factors/adapter_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
            print('PASS', student, json.dumps(dict(real_gates=sum(x['modes']['full']['gates'] for x in rows), domain=domain)), flush=True)
    result['complete'] = True
    (HERE/'integer_factors/adapter_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
