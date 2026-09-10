"""Selected checks of the compiled integer patch students.

The GPU executes dense Conv1 as an exact-range integer-valued reference.  The
controller masks unavailable columns only in the neuron decision, not GPU work.
No sparse runtime or original-FP32 equivalence is claimed.
"""
import argparse
import io
import json
from pathlib import Path
import sys
import time
import types
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_patch_probe import probe, RES

VARIANTS = ('row34', 'common3_diagonal_34')
ORDERS = {'row34': [0, 2, 5, 7, 8, 1, 3, 6, 4, 9],
          'common3_diagonal_34': [2, 3, 7, 0, 5, 8, 1, 6, 4, 9]}


def prepare_contract(arrays, device, contexts=240*80*12):
    c = {k: torch.from_numpy(np.array(v)).to(device) for k, v in arrays.items()
         if v.dtype.kind in 'biuf'}
    c['A'] = c['temporal_int16'].double()
    c['pos'] = c['threshold_positive'].double()
    c['neg'] = c['threshold_negative'].double()
    c['map'] = c['state_to_entry'].long()
    c['full'] = c['full_entry'].long()
    bits = 1 << torch.arange(10, device=device)
    support = c['temporal_int16'].ne(0)
    deps = (support.long()*bits).sum(1)
    lutmask = torch.arange(1024, device=device)[:, None].bitwise_and(bits).ne(0)
    need = torch.zeros(1024, device=device, dtype=torch.long)
    for t in range(10):
        need |= lutmask[:, t].long()*deps[t]
    # N order: native P4 spatial group first, then one of 12 consecutive H8.
    c.update(bits=bits, deps=deps, need=need, popcount=lutmask.sum(1),
        channels=(torch.arange(contexts, device=device)[:, None] % 12)*8+
                 torch.arange(32, device=device)[None]//4)
    return c


def integer_decisions(u, entry, c):
    """Ui and INT64-derived thresholds are exact integers represented in FP64."""
    h = c['channels'][:, None, :]
    row = torch.arange(10, device=u.device)[None, :, None]
    positive = c['pos'][entry[:, :, None], h]
    negative = c['neg'][entry[:, :, None], h]
    sign = c['positive_gain'][h]
    yes = torch.where(sign > 0, u >= positive, u <= positive)
    no = torch.where(sign > 0, u <= negative, u >= negative)
    constant = c['constant_gate'][row, h]
    yes = torch.where(sign == 0, constant, yes)
    no = torch.where(sign == 0, ~constant, no)
    return yes, no


def replay_integer(y, zero, c, mode, order, check_integer=False):
    """Same five-Y release/need planner; full function is only an error label."""
    n = len(y)
    full_u = torch.matmul(c['A'], y)
    full_entry = c['full'][None].expand(n, -1)
    full_gate, full_no = integer_decisions(full_u, full_entry, c)
    checks = {}
    if check_integer:
        residual = float((full_u-full_u.round()).abs().max())
        checks = dict(U_integer_residual=residual, U_min=float(full_u.min()),
                      U_max=float(full_u.max()), U_abs_bound=int(c['U_abs_bound'].max()))
        if residual != 0 or bool((full_u.abs() > c['U_abs_bound'][:, c['channels']].permute(1, 0, 2)).any()):
            raise RuntimeError('integer Ui violates compiled integer/range contract')
        if not bool((full_gate ^ full_no).all()):
            raise RuntimeError('full-entry thresholds do not partition the integer domain')
    del full_u, full_no
    if mode == 'exact':
        return full_gate, 0, checks
    seen = zero.clone()
    unresolved = torch.ones_like(full_gate)
    answer = torch.zeros_like(full_gate)
    for _ in range(11):
        if not bool(unresolved.any()):
            break
        observed = seen[:, None].bitwise_and(c['bits']).ne(0)
        partial = torch.matmul(c['A'], y*observed[:, :, None])
        positive, negative = integer_decisions(partial, c['map'][seen], c)
        decide = unresolved & (positive | negative)
        answer = torch.where(decide, positive, answer)
        unresolved &= ~decide
        active_rows = unresolved.any(2).long().mul(c['bits']).sum(1)
        dependency = c['need'][active_rows]
        retained = dependency.bitwise_and(seen).bitwise_and(~zero)
        needed = dependency.bitwise_and(~seen)
        free = 5-c['popcount'][retained]
        if bool(((free <= 0) & needed.ne(0)).any()):
            raise RuntimeError('integer completion exceeds the common five-Y capacity')
        batch = torch.zeros_like(seen)
        for t in order:
            take = needed.bitwise_and(1 << t).ne(0) & (free > 0)
            batch |= take.long() << t
            free -= take.long()
        seen |= batch
    if bool(unresolved.any()):
        raise RuntimeError('integer completion left unresolved gates')
    return answer, int((answer != full_gate).sum()), checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--split', choices=('diverse', 'valid'), default='diverse')
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--modes', nargs='+', choices=('exact', 'individual'), default=['exact', 'individual'])
    parser.add_argument('--output-directory', type=Path)
    args = parser.parse_args()
    system = probe.load_system(args)
    from run_bn_probe import input_frame, read_names
    from evaluate_stage2_deployment import CoarseReady, summarize
    from spikingjelly.activation_based import functional
    model, modules, _, current, sources, _, _, _ = system
    probe.install_sources(system, sources)
    current['count_codes'] = False
    patch = args.root/'algorithm/patch_probe'
    here = patch/'partial_completion'
    out = args.output_directory or here/('integer_valid10' if args.split == 'diverse' else 'integer_valid'+str(args.count))
    out.mkdir(parents=True, exist_ok=True)
    for name, values in torch.load(patch/'patch_train_calibration.pt',
                                  map_location='cpu', weights_only=False).items():
        bn = modules[name]
        bn.track_running_stats = True
        bn.running_mean = values['mean'].to(bn.weight)
        bn.running_var = values['var'].to(bn.weight)
    arrays = {name: dict(np.load(here/'integer_deployment'/(name+'.npz')))
              for name in VARIANTS}
    for key in ('weight_int8', 'theta_source', 'theta_output'):
        assert np.array_equal(arrays[VARIANTS[0]][key], arrays[VARIANTS[1]][key])
    contracts = {name: prepare_contract(arrays[name], 'cuda') for name in args.variants}
    common = arrays[VARIANTS[0]]
    source_theta = float(common['theta_source'])
    output_theta = float(common['theta_output'])
    weight = torch.tensor(common['weight_int8'].reshape(96, -1), device='cuda', dtype=torch.float32)
    source_bound = torch.tensor(common['Y_abs_bound'], device='cuda')
    names = (read_names(args.data, 'valid') if args.split == 'valid' else
             json.loads((args.root/'algorithm/samples.json').read_text())['valid'])[:args.count]
    full_valid = args.split == 'valid' and len(names) == 825
    captured = []
    if args.split == 'diverse':
        previous = torch.load(here/'capture.pt', map_location='cpu', weights_only=False)
        assert names[:4] == previous['metadata']['valid'][:min(len(names), 4)]
        group_ids = previous['groups'].long().cuda()
        positions = previous['positions'].long().cuda()
        del previous
        sampled_contexts = (group_ids[:, None]*12+torch.arange(12, device='cuda')[None]).reshape(-1)
        yy, xx = positions//320, positions%320
        captured = [dict(file=np.array(name), group_ids=group_ids.cpu().numpy()) for name in names[:4]]
    state = {}
    numeric_checks = {}

    def integer_conv(self, x):
        assert tuple(x.shape) == (10, 1, 96, 240, 320)
        unit = x[:, 0].float()/source_theta
        live = unit.ne(0)
        spatial = live.any(1)
        halo = F.max_pool2d(spatial[:, None].float(), 3, stride=1, padding=1).bool()[:, 0]
        empty = ~halo.reshape(10, 240, 80, 4).any(-1)
        zero_spatial = (empty.long()*(1 << torch.arange(10, device=x.device))[:, None, None]).sum(0)
        state['zero'] = zero_spatial[:, :, None].expand(240, 80, 12).reshape(-1)
        i = state['frame']
        if i < len(captured) and 'source_gate_words' not in captured[i]:
            padded = F.pad(live, (1, 1, 1, 1))
            words = torch.zeros((64, 96, 3, 3, 4), device=x.device, dtype=torch.int16)
            bits = (1 << torch.arange(10, device=x.device))[:, None, None, None]
            for kh in range(3):
                for kw in range(3):
                    support = padded[:, :, yy+kh, xx+kw]
                    words[:, :, kh, kw] = (support.long()*bits).sum(0).permute(1, 0, 2).short()
            captured[i]['source_gate_words'] = words.reshape(64, 864, 4).cpu().numpy()
            captured[i]['zero_source_word'] = zero_spatial.flatten()[group_ids].short().cpu().numpy()
        columns = F.unfold(unit, kernel_size=3, padding=1, stride=1)
        yi = torch.matmul(weight[None], columns)
        if 'conv' not in numeric_checks:
            r = float((yi-yi.round()).abs().max())
            u = float((unit-unit.round()).abs().max())
            numeric_checks['conv'] = dict(Y_integer_residual=r, source_unit_residual=u,
                source_unit_min=float(unit.min()), source_unit_max=float(unit.max()),
                Y_min=float(yi.min()), Y_max=float(yi.max()), Y_abs_bound=int(source_bound.max()))
            if r != 0 or u != 0 or float(unit.min()) < 0 or float(unit.max()) > 1:
                raise RuntimeError('Conv input or output is not the compiled integer function')
            if bool((yi.abs().amax((0, 2)) > source_bound).any()):
                raise RuntimeError('integer Conv exceeds the compiled Yi bound')
            print('INTEGER_CONV_CHECK', json.dumps(numeric_checks['conv']), flush=True)
        return yi.reshape(10, 1, 96, 240, 320)

    def integer_neuron(self, x):
        y = x[:, 0].reshape(10, 12, 8, 240, 80, 4).permute(3, 4, 1, 0, 2, 5).reshape(-1, 10, 32).double()
        name, mode = state['variant'], state['mode']
        gate, wrong, checks = replay_integer(y, state['zero'], contracts[name], mode,
            ORDERS[name], check_integer=name not in numeric_checks)
        if checks:
            numeric_checks[name] = checks
            print('INTEGER_U_CHECK', name, json.dumps(checks), flush=True)
        state['wrong'] = wrong
        i = state['frame']
        if i < len(captured):
            small_y = y[sampled_contexts].int().cpu().numpy()
            if 'Yi' not in captured[i]:
                captured[i]['Yi'] = small_y
            else:
                assert np.array_equal(captured[i]['Yi'], small_y)
            captured[i]['gate_'+name+'_'+mode] = gate[sampled_contexts].cpu().numpy()
        spikes = gate.reshape(240, 80, 12, 10, 8, 4).permute(3, 2, 4, 0, 1, 5).reshape_as(x)
        state['conv2_source_active_scalars'] = int(spikes.sum())
        state['conv2_active_terms'] = int((spikes.sum((0, 1, 2), dtype=torch.int64)*state['fanout']).sum())*96
        return spikes.to(x.dtype)*output_theta

    modules[RES+'1.conv1.0'].forward = types.MethodType(integer_conv, modules[RES+'1.conv1.0'])
    modules[RES+'1.norm1'].forward = types.MethodType(lambda self, x: x, modules[RES+'1.norm1'])
    neuron = modules[RES+'1.sn2.spiking_neuron']
    assert neuron.center_mode == 'zero'
    neuron.forward = types.MethodType(integer_neuron, neuron)
    state['fanout'] = F.conv2d(torch.ones(1, 1, 240, 320, device='cuda'),
        torch.ones(1, 1, 3, 3, device='cuda'), padding=1)[0, 0].long()
    results, skipped = {}, {}
    with torch.no_grad():
        for name in args.variants:
            for mode in args.modes:
                axis = name+'_'+mode
                if mode == 'individual' and name+'_exact' in results and results[name+'_exact']['AEE_frame_mean'] > 1.259:
                    skipped[axis] = 'own exact diverse10 AEE exceeded the fixed 1.259 gate'
                    continue
                rows, started = [], time.monotonic()
                for i, file in enumerate(names):
                    state.update(variant=name, mode=mode, frame=i)
                    functional.reset_net(model)
                    x, label, valid = input_frame(args.data, file)
                    try:
                        model(x)
                    except CoarseReady:
                        pred = F.interpolate(current.pop('flow'), (480, 640), mode='bilinear', align_corners=False)
                    error = torch.linalg.vector_norm(pred.permute(0, 2, 3, 1)[valid]-label.permute(0, 2, 3, 1)[valid], dim=1)
                    total, pixels = float(error.double().sum()), error.numel()
                    rows.append(dict(file=file, valid_pixels=pixels, aee_sum=total, AEE=total/pixels,
                        gate_difference_vs_own_full_integer=state['wrong'],
                        conv2_source_active_scalars=state['conv2_source_active_scalars'],
                        conv2_active_terms=state['conv2_active_terms']))
                    if len(names) <= 10:
                        print('INTEGER_EVAL', axis, i+1, len(names), 'AEE', total/pixels, flush=True)
                    elif (i+1) % 50 == 0 or i+1 == len(names):
                        progress = dict(summarize(rows, i+1 == len(names)), axis=axis,
                                        wall_seconds=time.monotonic()-started)
                        probe.save_json(out/(axis+'_summary.json'), progress)
                        probe.save_json(out/(axis+'_frames.json'), rows)
                        print('INTEGER_PROGRESS', axis, i+1, len(names), json.dumps(progress), flush=True)
                summary = dict(summarize(rows, full_valid), complete=True, axis=axis,
                    wall_seconds=time.monotonic()-started,
                    numeric='INT8 BN-gain-folded W / integer Yi in FP32 / INT16 Q14 A with FP64 exact-range Ui / INT64-derived threshold tables',
                    source_theta=source_theta, output_theta=output_theta,
                    normalization='r1 norm1 wrapper bypassed; BN gain is in W and BN offset is in compiled thresholds',
                    first_batch_size=5 if mode == 'individual' else 10,
                    Y_capacity=5 if mode == 'individual' else 10,
                    time_order=ORDERS[name] if mode == 'individual' else None,
                    gamma=3 if mode == 'individual' else None,
                    target='r1 Conv1 + sn2 only; true Conv2/fixed norm2/shortcut and coarse full network',
                    evaluation_split=args.split, requested_count=args.count,
                    selection=('fixed gamma3 promoted after exact diverse10 AEE <=1.259; exploratory validation, no validation fitting of compiled constants'
                               if mode == 'individual' else
                               'fixed full-integer function; complete validation control, no new parameter or threshold fitting'),
                    claim='new quantized numerical student, dense GPU function reference, no sparse runtime claim',
                    tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                    tf32_cudnn=torch.backends.cudnn.allow_tf32)
                summary['conv2_active_terms_mean'] = sum(r['conv2_active_terms'] for r in rows)/len(rows)
                results[axis] = summary
                probe.save_json(out/(axis+'_summary.json'), summary)
                probe.save_json(out/(axis+'_frames.json'), rows)
                print('INTEGER_SUMMARY', json.dumps(summary), flush=True)
    for i, sample in enumerate(captured):
        # The requested array name "file" conflicts with np.savez's path
        # argument, so write the ordinary NPZ (ZIP of NPY arrays) explicitly.
        with zipfile.ZipFile(out/f'capture_{i:02d}.npz', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for key, value in sample.items():
                buffer = io.BytesIO()
                np.save(buffer, value, allow_pickle=False)
                archive.writestr(key+'.npy', buffer.getvalue())
    probe.save_json(out/'run.json', dict(complete=True, evaluated_axes=list(results), skipped_axes=skipped,
        frames=names, captured_frames=names[:len(captured)], numeric_checks=numeric_checks,
        capture_layout='Yi/gate N=(sampled_P4_group*12+H8_group), [N,10,32], lane=local_h*4+p; source_gate_words [64,864,4], k=((c*3)+kh)*3+kw, bit=t; padded coordinates y+kh-1,x+kw-1',
        capture_shapes=dict(Yi=[768,10,32],source_gate_words=[64,864,4],group_ids=[64],zero_source_word=[64]),
        source_and_Y_capture=('one common copy; identical compiled W/source theta and Yi across captured axes' if captured else
                              'no new capture in this full-valid pass; common inputs are in integer_valid10'),
        parameter_directory=str(here/'integer_deployment')))
    print('INTEGER_DONE', json.dumps({k:v['AEE_frame_mean'] for k,v in results.items()}), flush=True)


if __name__ == '__main__':
    main()
