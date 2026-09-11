"""Actual two-window chain check. Full-domain BN data are an oracle, not free HW.

Integer boundaries require zero differences. FP32 contractions report rounding
differences from the CUDA implementation; no bit-equivalence or new AEE claim.
"""
from pathlib import Path
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from consumer_service import rne24, read_npz

T, C = 10, 96


def difference(actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    d = a.astype(np.float64)-b.astype(np.float64)
    return dict(values=int(a.size), differences=int(np.count_nonzero(d)),
                max_abs=float(np.max(np.abs(d))), rms=float(np.sqrt(np.mean(d*d))),
                reference_max_abs=float(np.max(np.abs(b))))


def source(identity, q):
    shape = identity.shape
    value = identity.astype(np.int64).reshape(T, -1)
    if 'As_q16' in q:
        value = rne24(q['As_q16'].astype(np.int64) @ value, int(q['As_exponent']))
    else:
        for layer in range(4):
            i, j = q['lifting_matchings'][layer].T
            x, y = value[i].copy(), value[j].copy()
            first = rne24((x << 12)+q['lifting_q12'][layer, :, 0, None]*y, 12)
            second = rne24((y << 12)+q['lifting_q12'][layer, :, 1, None]*first, 12)
            value[i], value[j] = first, second
        value = value[q['source_permutation'].astype(int)]
    return compare(value.reshape(shape), q, 'source')


def compare(value, q, prefix):
    dims = (T,)+(1,)*(value.ndim-1)
    const = q[prefix+'_constant'].reshape(dims)
    threshold = q[prefix+'_threshold'].reshape(dims)
    direction = q[prefix+'_direction'].reshape(dims)
    return np.where(const >= 0, const.astype(bool),
                    np.where(direction > 0, value >= threshold, value <= threshold))


def patch(gates, origin, y, x):
    """Full K864 with true image padding, never spike(biased zero input)."""
    result = np.zeros((C, 3, 3, T), np.float32)
    oy, ox = origin
    for kh in range(3):
        for kw in range(3):
            sy, sx = y+kh-1, x+kw-1
            if 0 <= sy < 240 and 0 <= sx < 320:
                ly, lx = sy-oy, sx-ox
                if not (0 <= ly < gates.shape[2] and 0 <= lx < gates.shape[3]):
                    raise ValueError('Missing an in-image halo value')
                result[:, kh, kw] = gates[:, :, ly, lx].T
    return result.reshape(864, T)


def tf32_round(x):
    value = np.asarray(x, np.float32)
    bits = value.view(np.uint32)
    rounded = (bits+np.uint32(4095)+((bits >> np.uint32(13)) & np.uint32(1))) & np.uint32(0xffffe000)
    return rounded.view(np.float32)


def fpdot(w, x, tensorfloat=False):
    w, x = np.asarray(w, np.float32), np.asarray(x, np.float32)
    if tensorfloat:
        w, x = tf32_round(w), tf32_round(x)
    return np.einsum('hk,kt->ht', w, x, optimize=False)


def normalize(x, p, prefix, mean=None, var=None):
    mean = p[prefix+'_mean'] if mean is None else mean
    var = p[prefix+'_var'] if var is None else var
    shape = (1, C)+(1,)*(x.ndim-2)
    inv = np.float32(1)/np.sqrt(np.float32(var)+np.float32(p[prefix+'_eps']))
    scale = np.float32(p[prefix+'_gamma']*inv)
    bias = np.float32(p[prefix+'_beta']-np.float32(mean*scale))
    return np.float32(np.float32(x*scale.reshape(shape))+bias.reshape(shape))


def window(data, q, p, label):
    geo = json.loads(str(data['window_geometry_json']))[label]
    get = lambda name: data[label+'_'+name]
    gate_origin, source_origin = geo['gate_origin'], geo['source_origin']
    h, w = geo['gate_shape']
    source_gate = source(get('I24'), q)
    checks = {'source_gate': difference(source_gate, get('sn1_gate'))}
    rank = int(p['preview_shared_rank'])
    z = np.empty((T, rank, h, w), np.float32)
    raw = np.empty((T, C, h, w), np.float32)
    tail = np.empty_like(raw)
    for y in range(h):
        for x in range(w):
            g = patch(source_gate, source_origin, gate_origin[0]+y, gate_origin[1]+x)
            latent = fpdot(p['preview_u'].T, np.float32(g*float(p['preview_theta_source'])), bool(p['TF32_cudnn']))
            z[:, :, y, x] = latent[:rank].T
            raw[:, :, y, x] = fpdot(p['preview_v'][:rank].T, latent[:rank], bool(p['TF32_cudnn'])).T
            tail[:, :, y, x] = fpdot(p['preview_v'][rank:].T, latent[rank:], bool(p['TF32_cudnn'])).T
    checks['preview_Z_FP32'] = difference(z, get('preview_Z_shared'))
    checks['preview_shared_FP32'] = difference(raw, get('preview_shared_raw'))
    checks['preview_tail_FP32'] = difference(tail, get('preview_tail_raw'))
    ybn = normalize(np.float32(raw+tail), p, 'bn1')
    checks['BN1_FP32'] = difference(ybn, get('preview_BN1_Y'))
    membrane = fpdot(p['preview_A'], ybn.reshape(T, -1))
    membrane = np.float32(np.float32(membrane+p['preview_b'][:, None])-np.float32(p['preview_theta_output']))
    sn2 = (membrane >= 0).reshape(T, C, h, w)
    checks['sn2_gate'] = difference(sn2, get('sn2_gate'))
    checks['sn2_min_abs_margin'] = float(np.min(np.abs(membrane)))
    iy, ix = gate_origin[0]-source_origin[0], gate_origin[1]-source_origin[1]
    updated = get('I24')[:, :, iy:iy+h, ix:ix+w].astype(np.int64).copy()
    continuous = np.empty((T, C, 4, 4), np.int64)
    out_y, out_x = geo['output_origin']
    for dy in range(4):
        for dx in range(4):
            y, x = 2*(out_y+dy), 2*(out_x+dx)
            g = patch(sn2, gate_origin, y, x).astype(np.int64)
            latent = rne24(q['U_conv2_theta_q16'].astype(np.int64) @ g,
                           int(q['U_conv2_theta_exponent'])-14)
            branch = rne24(q['F_q16'].astype(np.int64) @ latent, int(q['F_exponent']))
            ly, lx = y-gate_origin[0], x-gate_origin[1]
            merged = updated[:, :, ly, lx].T+branch+q['BN2_constant_q24'][:, None]
            updated[:, :, ly, lx] = rne24(merged, 0).T
            latent = rne24(q['U_ped_q16'].astype(np.int64) @ updated[:, :, ly, lx].T,
                           int(q['U_ped_exponent']))
            ped = rne24(q['V_ped_q16'].astype(np.int64) @ latent, int(q['V_ped_exponent']))
            continuous[:, :, dy, dx] = rne24(ped+q['PED_bias_q24'][:, None], 0).T
    checks['updated_I24'] = difference(updated, get('updated_I24'))
    checks['PED_continuous_q24'] = difference(continuous, get('continuous_q24'))
    proj_gate = compare(updated[q['consumer_permutation'].astype(int)], q, 'consumer')
    checks['proj_gate'] = difference(proj_gate, get('proj_gate'))
    conv = np.empty((T, C, 4, 4), np.float32)
    for dy in range(4):
        for dx in range(4):
            g = patch(proj_gate, gate_origin, 2*(out_y+dy), 2*(out_x+dx))
            conv[:, :, dy, dx] = fpdot(p['proj_weight_fp32'].reshape(C, -1),
                np.float32(g*float(p['proj_theta_output'])), bool(p['TF32_cudnn'])).T
    if bool(p['proj_has_bias']):
        conv += p['proj_bias_fp32'][None, :, None, None]
    checks['native_projection_FP32'] = difference(conv, get('proj_conv_fp32'))
    norm = normalize(conv, p, 'proj_bn', data['proj_bn_actual_domain_mean'],
                     data['proj_bn_actual_domain_var'])
    checks['native_BN_with_actual_global_stats_FP32'] = difference(norm, get('proj_norm_fp32'))
    final = np.float32(norm+np.float32(continuous.astype(np.float32)*np.float32(2**-14)))
    checks['final_PED_with_actual_global_stats_FP32'] = difference(final, get('ped_output_fp32'))
    integer_keys = ['source_gate', 'sn2_gate', 'updated_I24', 'PED_continuous_q24', 'proj_gate']
    checks['integer_and_gate_all_equal'] = all(checks[k]['differences'] == 0 for k in integer_keys)
    checks['geometry'] = geo
    checks['uses_global_BN_statistics_oracle'] = True
    checks['local_closed_hardware_endpoint'] = False
    return checks, dict(sn1=source_gate, sn2=sn2, proj=proj_gate, updated=updated, continuous=continuous)


def global_bn(data, p):
    x = data['proj_bn_full_input_fp32']
    mean = x.mean(axis=(0, 2, 3), dtype=np.float64)
    # A stable mathematical check, not the proposed hardware reduction order.
    var = np.mean((x.astype(np.float64)-mean[None, :, None, None])**2, axis=(0, 2, 3))
    norm = normalize(x, p, 'proj_bn', data['proj_bn_actual_domain_mean'], data['proj_bn_actual_domain_var'])
    return dict(actual_domain=list(x.shape), values=int(x.size), values_per_channel=int(x.size//C),
        track_running_stats=bool(p['proj_bn_track_running_stats']),
        running_statistics_elements=int(p['proj_bn_mean'].size),
        actual_batch_statistics=bool(data['proj_bn_uses_batch_statistics']),
        mean_vs_FP64_check=difference(data['proj_bn_actual_domain_mean'], mean),
        variance_vs_FP64_check=difference(data['proj_bn_actual_domain_var'], var),
        output_using_actual_stats=difference(norm, data['proj_bn_full_output_fp32']),
        raw_projection_FP32_bytes=int(x.nbytes),
        scope='Actual full domain captured unchanged. Independent double reduction is a numerical check only; no hardware statistics unit or free fixed BN is claimed.')


def main():
    result = dict(scope=__doc__, axes={}, complete_hardware_chain=False,
                  new_training=False, new_quantization=False, new_valid825=False,
                  FP_mode='TF32-rounded multiplicands plus NumPy FP32 contraction; diagnostic approximation, not proven cuDNN reduction-order identity.')
    for axis in ('ordinary', 'lifting_raw'):
        path = HERE/'capture'/axis
        data = read_npz(path/'000_zurich_city_09_a_0001.npz')
        q, p = read_npz(path/'parameters.npz'), read_npz(path/'live_parameters.npz')
        flags = json.loads((HERE/'capture/result.json').read_text())
        p['TF32_cudnn'] = flags['TF32_cudnn']
        row = dict(global_BN=global_bn(data, p), windows={})
        for label in ('corner', 'interior'):
            row['windows'][label], _ = window(data, q, p, label)
            print(axis, label, row['windows'][label]['integer_and_gate_all_equal'], flush=True)
        result['axes'][axis] = row
    (HERE/'numerical_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
