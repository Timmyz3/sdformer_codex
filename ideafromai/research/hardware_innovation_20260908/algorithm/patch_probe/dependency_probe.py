"""Train32 moment fitting of patch temporal dependencies; fixed-BN valid10 probe.

This changes only r1.sn2's temporal A and bias. Masks define a further new
student, D * neuron(BN(I * conv1(z))), I_s = OR_t(D_t & nonzero(A_ts)).
Dense GPU forwards implement the reference function, not sparse execution.
"""
import argparse
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from run_patch_probe import probe, RES
from production_mask_probe import scores, BLOCK

PREFIX = RES+'1.'


def sparse_fit(a, bias, mean, cov, support):
    result = np.zeros_like(a)
    for row in range(10):
        cols = np.flatnonzero(support[row])
        result[row, cols] = np.linalg.lstsq(cov[np.ix_(cols, cols)],
            cov[cols] @ a[row], rcond=1e-10)[0]
    fitted_bias = bias+(a-result) @ mean
    error = np.einsum('ij,jk,ik->i', a-result, cov, a-result).clip(0)
    return result, fitted_bias, error


def grouped_support(groups):
    support = np.zeros((10, 10), bool)
    for group in groups:
        support[np.ix_(group, group)] = True
    return support


def fit_all(a, bias, mean, cov, demand, active_terms):
    variants = {}
    def add(name, support, groups=None, note=None, fitted=None):
        w, b, error = sparse_fit(a, bias, mean, cov, support) if fitted is None else fitted
        # Deployment uses these FP32 coefficients, rather than the LS doubles.
        w, b = w.astype(np.float32), b.astype(np.float32)
        variants[name] = dict(weight=w, bias=b, support=w != 0,
            allocated_connections=int(support.sum()), actual_connections=int(np.count_nonzero(w)),
            matrix_rank=int(np.linalg.matrix_rank(w)), singular_values=np.linalg.svd(w, compute_uv=False).tolist(),
            training_membrane_MSE=float(error.mean()), per_row_MSE=error.tolist(),
            groups=groups, note=note)
    add('native', np.ones((10, 10), bool), fitted=(a, bias, np.zeros(10)), note='unchanged temporal neuron')
    causal = np.array([[0 <= t-s < 4 for s in range(10)] for t in range(10)])
    add('masked_k4', causal, note='known causal order-4 masked PSN; 34 independent coefficient slots')
    # A 4-tap shared sliding kernel, with ten independently fitted row biases.
    gram, rhs = np.zeros((4, 4)), np.zeros(4)
    for t in range(10):
        lags = np.arange(min(4, t+1))
        cols = t-lags
        gram[np.ix_(lags, lags)] += cov[np.ix_(cols, cols)]
        rhs[lags] += cov[cols] @ a[t]
    kernel = np.linalg.lstsq(gram, rhs, rcond=1e-10)[0]
    sliding = np.zeros_like(a)
    for t in range(10):
        for lag in range(min(4, t+1)):
            sliding[t, t-lag] = kernel[lag]
    b = bias+(a-sliding) @ mean
    err = np.einsum('ij,jk,ik->i', a-sliding, cov, a-sliding).clip(0)
    add('sliding_k4', causal, fitted=(sliding, b, err),
        note='known shared 4-tap temporal kernel with ten fitted biases; not a reproduction of end-to-end PSN training')
    variants['sliding_k4']['shared_kernel'] = kernel.tolist()
    best = {}
    for row in range(10):
        other = [s for s in range(10) if s != row]
        for width in (3, 4):
            options = []
            for extra in itertools.combinations(other, width-1):
                cols = np.array(sorted((row,)+extra))
                coef = np.linalg.lstsq(cov[np.ix_(cols, cols)], cov[cols] @ a[row], rcond=1e-10)[0]
                delta = a[row].copy()
                delta[cols] -= coef
                options.append((max(0., float(delta @ cov @ delta)), tuple(cols)))
            best[row, width] = min(options)
    four = sorted(range(10), key=lambda t: (-(best[t, 3][0]-best[t, 4][0]), t))[:4]
    support = np.zeros((10, 10), bool)
    for row in range(10):
        support[row, best[row, 4 if row in four else 3][1]] = True
    add('row34', support, note='diagonal retained; six 3-entry rows and four 4-entry rows chosen by train MSE')
    variants['row34']['four_entry_rows'] = four
    add('continuous334', grouped_support([[0,1,2], [3,4,5], [6,7,8,9]]),
        groups=[[0,1,2], [3,4,5], [6,7,8,9]], note='ordinary contiguous temporal groups')
    # Precompute each possible 3/4-set's distortion and actual train closure work.
    cache = {}
    for size in (3, 4):
        for group in itertools.combinations(range(10), size):
            selected = np.any(demand[:, group], axis=1)
            work = float((active_terms[:, group].sum(1)*selected).sum())
            error = 0.
            for row in group:
                cols = np.array(group)
                coef = np.linalg.lstsq(cov[np.ix_(cols, cols)], cov[cols] @ a[row], rcond=1e-10)[0]
                delta = a[row].copy()
                delta[cols] -= coef
                error += max(0., float(delta @ cov @ delta))
            cache[group] = (error, work)
    partitions = []
    for four_group in itertools.combinations(range(10), 4):
        remain = [t for t in range(10) if t not in four_group]
        for three in itertools.combinations(remain, 3):
            other = tuple(t for t in remain if t not in three)
            if three > other:
                continue
            groups = (three, other, four_group)
            partitions.append((sum(cache[g][0] for g in groups), sum(cache[g][1] for g in groups), groups))
    minimum = min(partitions, key=lambda x: (x[0], x[2]))
    joint = min((x for x in partitions if x[0] <= minimum[0]*1.10+1e-15), key=lambda x: (x[1], x[0], x[2]))
    for name, selected, note in (
        ('fit334', minimum, 'all 2100 partitions, minimum train membrane MSE'),
        ('closure334', joint, 'minimum train active conv terms after demand closure, subject to MSE <= 1.10 times best group MSE')):
        groups = [list(g) for g in selected[2]]
        add(name, grouped_support(groups), groups=groups, note=note)
        variants[name]['selection_train_closed_conv1_active_terms'] = selected[1]
    return variants


def serializable(value):
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mask-aee-gap', type=float, default=.06)
    parser.add_argument('--no-mask-network', action='store_true')
    args = parser.parse_args()
    out = args.root/'algorithm/patch_probe/dependency'
    out.mkdir(parents=True, exist_ok=True)
    system = probe.load_system(args)
    from run_bn_probe import input_frame
    from evaluate_stage2_deployment import CoarseReady, summarize
    from spikingjelly.activation_based import functional
    model, modules, _, current, sources, _, _, _ = system
    probe.install_sources(system, sources)
    current['count_codes'] = False
    fixed = torch.load(args.root/'algorithm/patch_probe/patch_train_calibration.pt', map_location='cpu', weights_only=False)
    for name, values in fixed.items():
        module = modules[name]
        module.track_running_stats = True
        module.running_mean = values['mean'].to(module.weight)
        module.running_var = values['var'].to(module.weight)
    neuron = modules[PREFIX+'sn2.spiking_neuron']
    if neuron.temporal_factor_rank:
        raise ValueError('this experiment fits the full temporal matrix')
    a, bias = neuron.weight.detach().double().cpu().numpy(), neuron.bias.detach().flatten().double().cpu().numpy()
    train = json.loads((args.root/'algorithm/direct_code_integer/run.json').read_text())['train']
    valid = json.loads((args.root/'algorithm/samples.json').read_text())['valid'][:10]
    run = dict(train=train, valid=valid, target=PREFIX+'sn2.spiking_neuron', fixed_BN=list(fixed),
        theta=float(neuron.thresh), center_mode=neuron.center_mode, center=neuron.center.cpu().tolist(),
        output_mode=neuron.output_mode, threshold_mode=neuron.threshold_mode,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
        fitting='full train32 spatial/channel/time moments; no gradient training or validation fitting',
        demand='existing sn1 amplitude score, B8 plus one-pixel halo, fixed train median',
        mask_network_gate=f'only axes with unmasked valid10 AEE <= native + {args.mask_aee_gap}; exploratory stopping, no fitted model selection',
        claim='new FP32 students and operation/state counts; GPU remains dense; not cycles or ep34 equivalence')
    probe.save_json(out/'run.json', run)
    total = torch.zeros(10, device='cuda', dtype=torch.float64)
    second = torch.zeros((10,10), device='cuda', dtype=torch.float64)
    count = 0
    captured, score_list, active_list, recovery_samples = [], [], [], []
    sample_positions = torch.linspace(0, 240*320-1, 512, device='cuda').round().long()
    observed_sample = {}
    class Captured(Exception):
        pass
    def input_capture(module, inputs):
        x = inputs[0].detach()
        score_list.append(scores(x).cpu().numpy())
        source_count = x.ne(0).sum(2).float()
        terms = F.conv2d(source_count, torch.ones(1,1,3,3, device=x.device), padding=1)
        active_list.append((F.avg_pool2d(terms, BLOCK, stride=BLOCK)*BLOCK**2*96).cpu().numpy())
    def membrane_observer(h, theta):
        observed_sample['membrane'] = h.detach().reshape(10,96,-1)[:,:,sample_positions].cpu()
    old_observer = getattr(neuron, '_h9_calibration_observer', None)
    neuron._h9_calibration_observer = membrane_observer
    def moment_capture(module, inputs, output):
        nonlocal count
        flat = inputs[0].detach().flatten(1).double()
        total.add_(flat.sum(1))
        second.addmm_(flat, flat.T)
        count += flat.shape[1]
        recovery_samples.append(dict(
            input=inputs[0].detach().reshape(10,96,-1)[:,:,sample_positions].cpu(),
            membrane=observed_sample.pop('membrane'),
            gate=output.detach().reshape(10,96,-1)[:,:,sample_positions].ne(0).cpu()))
        raise Captured()
    h1 = modules[PREFIX+'conv1.0'].register_forward_pre_hook(input_capture)
    h2 = neuron.register_forward_hook(moment_capture)
    started = time.monotonic()
    with torch.no_grad():
        for i, name in enumerate(train):
            functional.reset_net(model)
            x, _, _ = input_frame(args.data, name, targets=False)
            try:
                model(x)
            except Captured:
                captured.append(name)
            if (i+1)%8 == 0:
                print('TRAIN_MOMENTS', i+1, count, round(time.monotonic()-started, 2), flush=True)
    h1.remove()
    h2.remove()
    neuron._h9_calibration_observer = old_observer
    torch.save(dict(frames=captured, positions=sample_positions.cpu(), samples=recovery_samples,
        theta=neuron.thresh.detach().cpu(),
        note='actual full-forward FP32 teacher membrane and gates, uniform train-only 512 spatial positions, all C96/T10'),
        out/'recovery_train_samples.pt')
    del recovery_samples
    mean = total.cpu().numpy()/count
    cov = second.cpu().numpy()/count-np.outer(mean, mean)
    sc = np.stack(score_list)
    threshold = float(np.quantile(sc, .5))
    demand = sc > threshold
    active_terms = np.stack(active_list)
    variants = fit_all(a, bias, mean, cov, demand, active_terms)
    np.savez_compressed(out/'train_moments.npz', mean=mean, covariance=cov, count=count,
        scores=sc, active_conv1_terms=active_terms, demand=demand)
    probe.save_json(out/'fit.json', dict(count=count, threshold=threshold,
        covariance_eigenvalues=np.linalg.eigvalsh(cov).tolist(), variants=serializable(variants)))
    torch.save({name: dict(weight=torch.from_numpy(v['weight']), bias=torch.from_numpy(v['bias']))
                for name,v in variants.items()}, out/'parameters.pt')
    closure = {}
    for name,v in variants.items():
        closed = np.einsum('ts,ntbhw->nsbhw', v['support'].astype(np.int16), demand.astype(np.int16)) > 0
        connections = np.einsum('ts,ntbhw->', v['support'].astype(np.int64), demand.astype(np.int64))*BLOCK**2*96
        closure[name] = dict(demanded_time_tiles_fraction=float(demand.mean()),
            required_input_time_tiles_fraction=float(closed.mean()),
            closed_conv1_active_terms_mean=float((active_terms*closed).sum()/len(train)),
            parent_conv1_active_terms_mean=float(active_terms.sum()/len(train)),
            selected_PSN_connections_mean=float(connections/len(train)),
            note='train-mask closure only; no oracle labels, no estimate of recognition error or cycles')
    probe.save_json(out/'train_closure.json', closure)
    print('FIT', json.dumps({n:{k:v[k] for k in ('matrix_rank','actual_connections','training_membrane_MSE','groups')} for n,v in variants.items()}), flush=True)

    original_conv = modules[PREFIX+'conv1.0'].forward
    original_neuron = neuron.forward
    axis, masking, frame_counts = 'native', False, []
    support = torch.ones((10,10), device='cuda')
    demand_pixels = None
    def conv_forward(x):
        nonlocal demand_pixels
        score = scores(x)
        d = score > threshold if masking else torch.ones_like(score, dtype=torch.bool)
        closed = (support.T @ d.flatten(1).float()).reshape_as(d).ne(0)
        ip = closed.repeat_interleave(BLOCK,-2).repeat_interleave(BLOCK,-1)
        demand_pixels = d.repeat_interleave(BLOCK,-2).repeat_interleave(BLOCK,-1)
        t,b,c,h,w = x.shape
        source_count = x.detach().ne(0).sum(2).float()
        terms = F.conv2d(source_count, torch.ones(1,1,3,3,device=x.device), padding=1)
        live = terms.ne(0)&ip
        required = F.max_pool2d(ip.float(),3,stride=1,padding=1).bool()
        words = required.reshape(t,b,h,w//4,4).any(-1)
        g = x.detach().ne(0)&required[:,:,None]
        nonempty = g.reshape(t,b,c//16,16,h,w//4,4).any(3).any(-1)
        row_inputs = (support @ live.flatten(1).float()).reshape_as(live)
        per_tile_outputs = d.sum(0)
        frame_counts.append(dict(
            requested_output_time_tile_fraction=float(d.float().mean()),
            required_input_time_tile_fraction=float(closed.float().mean()),
            required_space_tile_fraction=float(closed.any(0).float().mean()),
            conv1_active_terms=int((terms*ip).sum())*96,
            parent_conv1_active_terms=int(terms.sum())*96,
            PSN_connections_with_known_rawY_zero_skip=int((row_inputs*demand_pixels).sum())*96,
            PSN_requested_output_gates=int(demand_pixels.sum())*96,
            halo_C16_P4_geometric_source_words=int(words.sum())*(c//16),
            halo_C16_P4_nonempty_source_words=int(nonempty.sum()),
            input_score_elements_examined=x.numel() if masking else 0,
            bare_one_tile_selected_U_FP32_bytes_mean=float(per_tile_outputs.float().mean())*BLOCK**2*96*4,
            bare_one_tile_selected_U_FP32_bytes_peak=int(per_tile_outputs.max())*BLOCK**2*96*4,
            bare_one_tile_selected_Y_FP32_bytes_mean=float(closed.sum(0).float().mean())*BLOCK**2*96*4))
        y = original_conv(x)
        return y*ip[:,:,None].to(y.dtype) if masking else y
    def neuron_forward(x):
        z = original_neuron(x)
        return z*demand_pixels[:,:,None].to(z.dtype) if masking else z
    def second_conv(module, inputs):
        g = inputs[0].detach().ne(0)
        h,w = g.shape[-2:]
        fanout = F.conv2d(torch.ones(1,1,h,w,device=g.device), torch.ones(1,1,3,3,device=g.device),padding=1)[0,0]
        frame_counts[-1]['conv2_active_terms'] = int((g.sum((0,1,2))*fanout).sum())*96
        frame_counts[-1]['conv2_source_gate_fraction'] = float(g.float().mean())
    modules[PREFIX+'conv1.0'].forward = conv_forward
    neuron.forward = neuron_forward
    hook = modules[PREFIX+'conv2.0'].register_forward_pre_hook(second_conv)
    results = {}
    with torch.no_grad():
        for masked in (False, True):
            if masked and args.no_mask_network:
                break
            masking = masked
            for name, variant in variants.items():
                axis = name
                key = name+('_demand' if masked else '_plain')
                if masked and results[name+'_plain']['AEE_frame_mean'] > results['native_plain']['AEE_frame_mean']+args.mask_aee_gap:
                    results[key] = dict(stopped='unmasked AEE exceeds predeclared native+gap; fit and static train closure retained')
                    continue
                neuron.weight.copy_(torch.from_numpy(variant['weight']).to(neuron.weight))
                neuron.bias.copy_(torch.from_numpy(variant['bias']).reshape_as(neuron.bias).to(neuron.bias))
                support = torch.from_numpy(variant['support']).cuda().float()
                rows, frame_counts = [], []
                for i,name_frame in enumerate(valid):
                    functional.reset_net(model)
                    x,label,mask = input_frame(args.data,name_frame)
                    try:
                        model(x)
                    except CoarseReady:
                        pred = F.interpolate(current.pop('flow'), (480,640),mode='bilinear',align_corners=False)
                    error = torch.linalg.vector_norm(pred.permute(0,2,3,1)[mask]-label.permute(0,2,3,1)[mask],dim=1)
                    total_error,pixels = float(error.double().sum()),error.numel()
                    rows.append(dict(file=name_frame, valid_pixels=pixels,aee_sum=total_error,AEE=total_error/pixels))
                results[key] = dict(**summarize(rows,False),
                    counts={k:(max(v[k] for v in frame_counts) if k.endswith('_peak')
                               else sum(v[k] for v in frame_counts)/len(frame_counts)) for k in frame_counts[0]},
                    claim='algorithm AEE and operation/state counts only; no sparse GPU or hardware timing')
                probe.save_json(out/(key+'_frames.json'),[dict(**r,**c) for r,c in zip(rows,frame_counts)])
                probe.save_json(out/'valid10_summary.json',results)
                print('VALID10',key,json.dumps(results[key]),flush=True)
    hook.remove()
    probe.save_json(out/'valid10_summary.json',results)
    run.update(complete=True, train_capture_completed=captured, threshold=threshold,
        wall_seconds=time.monotonic()-started,
        state_scope='one B8xB8 spatial tile, FP32 bare cells; does not include source, packet, lane ports or pipeline; no cycle claim',
        rank_scope='numerical rank of stored FP32 A before D; diag(D)A has rank no greater than requested output rows',
        zero_column_scope='common strongest control: fixed BN affine offset folded into PSN bias, allowing known raw-Y-zero columns to skip')
    probe.save_json(out/'run.json',run)


if __name__ == '__main__':
    main()
