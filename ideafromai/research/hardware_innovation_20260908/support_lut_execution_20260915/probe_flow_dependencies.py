#!/opt/anaconda3/bin/python3.12
"""Actual D2 geometry + synthetic masks; CPU dependency probe, not inference/RTL.

Reads existing checkpoint/profile/config/calibration through a NumPy-only reader.
No model execution, trained mask, optical-flow quality claim, or timing estimate.
"""
from pathlib import Path
import ast
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
REPO = BASE.parents[2]
SOURCE = REPO / 'SDformer/third_party/SDformerFlow/models/STSwinNet_SNN'
OVERLAY = REPO / 'SDformer/neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN'
PROFILE = BASE / 'open_fusion_execution/major_operator_fusions_20260913/root_owned/profile.json'
INCOMING = REPO / 'SDformer/hw_autoresearch_nts07/system_handoff/incoming/m2041_ep34_quant_binding_inputs'
CHECKPOINT = INCOMING / 'checkpoint_epoch34.pth'
CONFIG = INCOMING / 'dsec_c12_alpha0125_ep29_resume5_20260830.yml'
CALIBRATION = BASE / 'algorithm/patch_probe/patch_train_calibration.pt'
sys.path.insert(0, str(REPO / 'ideafromai/research/mechanism_rebuild_gh_20260906/scripts'))
from checkpoint_numpy import read_checkpoint


def histogram(x):
    a, n = np.unique(x, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(a, n)}


def source_contract():
    rows = json.loads(PROFILE.read_text())['rows']
    modules = {r['module']: r for r in rows}
    d = modules['sttmultires_unet.decoders.2.deconv.0']
    t, ci, ih, iw = d['left_shape']
    _, co, oh, ow = d['output_shape']
    assert (t, ci, ih, iw, co, oh, ow) == (10, 386, 60, 80, 96, 120, 160)
    assert d['right_shape'] == [386, 96, 3, 3]
    # Extract stride/padding policy from the actual constructor, not old prose.
    tree = ast.parse((SOURCE / 'Spiking_modules.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SpikingTransposeDecoderLayer')
    convs = [n for n in ast.walk(cls) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == 'ConvTranspose2d']
    kws = [{k.arg: ast.unparse(k.value) for k in n.keywords} for n in convs]
    assert any(k.get('stride') == '2' and k.get('padding') == 'padding'
               and k.get('output_padding') == '1' for k in kws)
    assert 'padding = kernel_size // 2' in ast.unparse(cls)
    cfg = CONFIG.read_text()
    assert 'kernel_size: 3' in cfg and 'num_steps: 10' in cfg
    assert 'spike_norm: BN' in cfg and 'center_mode: zero' in cfg
    requirements = REPO / 'SDformer/requirements.txt'
    assert 'spikingjelly==0.0.0.0.14' in requirements.read_text()
    state = read_checkpoint(CHECKPOINT)['model_state_dict']
    names = [
        'sttmultires_unet.decoders.2.deconv.0.weight',
        'sttmultires_unet.decoders.2.sn.spiking_neuron.weight',
        'sttmultires_unet.preds.1.conv.0.weight',
        'sttmultires_unet.preds.2.conv.0.weight',
        'sttmultires_unet.preds.2.sn.spiking_neuron.weight',
    ]
    params = {n: {'shape': list(state[n].shape), 'nonzero': int(np.count_nonzero(state[n])),
                  'elements': int(state[n].size)} for n in names}
    assert all(v['nonzero'] == v['elements'] for v in params.values())
    del state
    calibration_keys = list(read_checkpoint(CALIBRATION))
    assert len(calibration_keys) == 4 and not any('decoders.' in k or 'preds.' in k for k in calibration_keys)
    loader = (BASE / 'algorithm/run_bn_probe.py').read_text()
    assert 'set_bn_mode(model)' in loader
    assert 'module.track_running_stats = False' in loader
    return (t, ci, ih, iw, co, oh, ow), {
        'profile': str(PROFILE), 'checkpoint': str(CHECKPOINT), 'configuration': str(CONFIG),
        'model_source': str(SOURCE / 'Spiking_STSwinNet.py'),
        'decoder_source': str(SOURCE / 'Spiking_modules.py'),
        'PSN_source': str(OVERLAY / 'atlif_ternary_psn/atlif_ternary_psn.py'),
        'parameters': params, 'patch_calibration_keys': calibration_keys,
        'BN_mode_evidence': 'build_model set_bn_mode resets all BN to no-running-stats; only FC1 and four patch BNs are later calibrated; D2 stays current-domain',
        'BN_library_contract': {'requirements': str(requirements), 'version': '0.0.0.0.14',
            'batchnorm_source': 'https://raw.githubusercontent.com/fangwei123456/spikingjelly/0.0.0.0.14/spikingjelly/activation_based/layer.py',
            'flatten_T_B_source': 'https://raw.githubusercontent.com/fangwei123456/spikingjelly/0.0.0.0.14/spikingjelly/activation_based/functional.py'},
        'runtime_model_not_instantiated': True,
    }


def make_edges(ih, iw, oh, ow):
    source, target, tap = [], [], []
    for y in range(ih):
        for x in range(iw):
            for ky in range(3):
                oy = 2*y + ky - 1
                for kx in range(3):
                    ox = 2*x + kx - 1
                    if 0 <= oy < oh and 0 <= ox < ow:
                        source.append(y*iw+x); target.append(oy*ow+ox); tap.append(3*ky+kx)
    source, target, tap = map(np.asarray, (source, target, tap))
    assert len(source) == (3*ih-1)*(3*iw-1)
    # Independent output-gather enumeration checks parity and cropped edges.
    gathered = []
    for oy in range(oh):
        for ox in range(ow):
            for ky in range(3):
                ynum = oy + 1 - ky
                if ynum % 2: continue
                iy = ynum // 2
                for kx in range(3):
                    xnum = ox + 1 - kx
                    if xnum % 2: continue
                    ix = xnum // 2
                    if 0 <= iy < ih and 0 <= ix < iw:
                        gathered.append((iy*iw+ix, oy*ow+ox, 3*ky+kx))
    assert sorted(gathered) == sorted(zip(source.tolist(), target.tolist(), tap.tolist()))
    return source, target, tap


def bilinear_predecessor_mask(mask, oh, ow):
    """Actual 4x align_corners=False inverse footprint; no flow-value scaling."""
    fh, fw = mask.shape
    assert (fh, fw) == (4*oh, 4*ow)
    ys, xs = np.nonzero(mask)
    # floor((x+.5)/4-.5), exact integer numerator; both taps have nonzero weights.
    y0, x0 = (2*ys+1-4)//8, (2*xs+1-4)//8
    low = np.zeros((oh, ow), bool)
    for dy in (0, 1):
        for dx in (0, 1):
            low[np.clip(y0+dy, 0, oh-1), np.clip(x0+dx, 0, ow-1)] = True
    return low


def main():
    dims, evidence = source_contract()
    t, ci, ih, iw, co, oh, ow = dims
    src, dst, taps = make_edges(ih, iw, oh, ow)
    source_degree = np.bincount(src, minlength=ih*iw)
    target_degree = np.bincount(dst, minlength=oh*ow)
    def analyze(name, mask, final_pixels=None):
        chosen = mask.reshape(-1)[dst]
        used = np.bincount(src[chosen], minlength=ih*iw)
        requested_source = used > 0
        forward = np.bincount(dst[requested_source[src]], minlength=oh*ow) > 0
        active = bool(mask.any())
        return {
            'name': name, 'mask_kind': 'synthetic_not_GT_not_learned_not_real_event_activity',
            'selected_final_480x640_pixels': final_pixels,
            'needed_pred2_120x160_positions': int(mask.sum()),
            'fixed_BN_geometric_source_positions': int(requested_source.sum()),
            'fixed_BN_selected_spatial_edges': int(chosen.sum()),
            'fixed_BN_dense_scalar_coefficient_terms': int(chosen.sum()*ci*co*t),
            'fixed_BN_source_continuous_values_before_PSN': int(requested_source.sum()*ci*t),
            'fixed_BN_shared_source_positions_with_selected_and_unselected_children': int(np.count_nonzero((used > 0) & (used < source_degree))),
            'fixed_BN_forward_all_taps_footprint_positions': int(forward.sum()),
            'fixed_BN_outside_selected_footprint': int(np.count_nonzero(forward & ~mask.reshape(-1))),
            'selected_children_per_source_histogram': histogram(used[requested_source]),
            'current_dynamic_BN_exact_source_positions': ih*iw if active else 0,
            'current_dynamic_BN_exact_Y_positions_per_channel': oh*ow if active else 0,
            'current_dynamic_BN_exact_Y_values_all_channels_T': oh*ow*co*t if active else 0,
            'current_dynamic_BN_exact_dense_scalar_coefficient_terms': len(src)*ci*co*t if active else 0,
            'temporal_input_support_per_requested_source_channel': t if active else 0,
            'required_output_channels_for_two_component_flow': co if active else 0,
        }
    cases=[]
    def low_case(name, rects=None, points=None, phase=None, dense=False):
        m=np.zeros((oh,ow),bool)
        if dense:m[:]=True
        for r in rects or []:m[r[0]:r[1],r[2]:r[3]]=True
        for y,x in points or []:m[y,x]=True
        if phase is not None:m[phase[0]::2,phase[1]::2]=True
        cases.append(analyze(name,m));return m
    low_case('empty');low_case('full',dense=True)
    low_case('single_center_even_even',points=[(60,80)])
    low_case('single_center_odd_odd',points=[(61,81)])
    low_case('single_bottom_right',points=[(119,159)])
    low_case('center_2x2_all_phases',rects=[(60,62,80,82)])
    a=low_case('center_8x8',rects=[(56,64,72,80)])
    c=low_case('adjacent_8x8',rects=[(56,64,80,88)])
    cases.append(analyze('two_adjacent_8x8_union',a|c))
    low_case('odd_odd_phase_only',phase=(1,1))
    rng=np.random.default_rng(20260915)
    cases.append(analyze('low_random_10percent',rng.random((oh,ow))<.1))
    for name, rect in [('final_center_32x32',(224,256,288,320)),('final_center_64x64',(224,288,288,352)),('final_corner_32x32',(0,32,0,32))]:
        m=np.zeros((480,640),bool);m[rect[0]:rect[1],rect[2]:rect[3]]=True
        cases.append(analyze(name,bilinear_predecessor_mask(m,oh,ow),int(m.sum())))
    m=rng.random((480,640))<.1
    cases.append(analyze('final_random_10percent',bilinear_predecessor_mask(m,oh,ow),int(m.sum())))
    byname={c['name']:c for c in cases}
    assert byname['single_center_even_even']['fixed_BN_geometric_source_positions']==1
    assert byname['single_center_odd_odd']['fixed_BN_geometric_source_positions']==4
    assert byname['single_bottom_right']['fixed_BN_geometric_source_positions']==1
    assert byname['odd_odd_phase_only']['fixed_BN_geometric_source_positions']==ih*iw
    result={
        'scope':'CPU exact topology/support closure with synthetic masks; not new flow inference, accuracy, measured latency, or RTL',
        'evidence':evidence,'T':t,'B':1,'input_shape':[t,1,ci,ih,iw],
        'output_shape':[t,1,co,oh,ow],'kernel':[3,3],'stride':[2,2],
        'padding':[1,1],'output_padding':[1,1],'total_spatial_edges':int(len(src)),
        'source_output_degree_histogram':histogram(source_degree),
        'output_predecessor_degree_histogram':histogram(target_degree),
        'one_source_spike_spatial_channel_consumers_max':int(source_degree.max()*co),
        'one_pre_PSN_source_scalar_temporal_spatial_channel_consumers_max':int(t*source_degree.max()*co),
        'dynamic_BN_values_per_output_channel':int(t*oh*ow),
        'union_saved_source_positions_for_adjacent_tiles':byname['center_8x8']['fixed_BN_geometric_source_positions']+byname['adjacent_8x8']['fixed_BN_geometric_source_positions']-byname['two_adjacent_8x8_union']['fixed_BN_geometric_source_positions'],
        'cases':cases,
        'checks':{'scatter_equals_independent_gather':True,'dense_valid_tap_formula':True,'phase_corner_and_full_source_checks':True,'checkpoint_weight_support_exactly_dense':True},
        'limitations':['Spatial edge counts include every nonzero coefficient, not activation sparsity or numeric cancellation.',
        'Dynamic BN closure is dependency closure for exact selected outputs; alternative algebra for exact statistics is not timed here.',
        'Fixed BN columns are a proposed changed-function counterfactual, not a deployed fact.',
        'Mask policy, decision latency, cache misses, bank grants, photometric confidence, and AEE are not measured.',
        'No GT mask enters generation; final interpolation predecessor masks are exact geometric dependencies only.']}
    out=HERE/'probe_flow_dependencies.json';out.write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n')
    print(json.dumps({'result':str(out),'cases':len(cases),'spatial_edges':len(src),
                      'source_degree':result['source_output_degree_histogram'],
                      'output_degree':result['output_predecessor_degree_histogram'],
                      'status':'PASS'},ensure_ascii=False))


if __name__=='__main__':main()
