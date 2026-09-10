"""Bounded train32/valid10 NR4-cost probe; no production changes.

Three competitors start at the SAME saved signed12/signed8 bits3 source:
original local distillation, plus ordinary spike-rate cost, or plus NR4
nonempty-destination cost. Only the three source projections and their source
decision thresholds train; class dictionary/map, consumer W/B/tau, theta, and
input/weight quantization scales remain fixed. Consumer tau is NOT theta.

Capture preserves real P4-aligned contiguous spatial groups and ALL 384 source
channels with complete T10 inputs. The differentiable NR proxy first packs
currently hard-nonempty source rows in stable channel order, using a detached
selection; its within-group expectation assumes independent categorical draws.
It is NOT the expectation through differentiable NRV repacking. Exact hard NR4
counts (and the uncompressed fixed-four-channel proxy) are reported separately.
Counts precede output-specific weight-zero compaction; they are not cycles.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import types

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ALGORITHM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ALGORITHM))
AXES = ('distill', 'spike_cost', 'nr4_cost')
P = NR = 4
WUNIT, TUNIT = 128.0, 262144.0


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def ste_round(x):
    return x + (x.round()-x).detach()


class QuantizedProducer(nn.Module):
    def __init__(self, record, margin_scale):
        super().__init__()
        self.weight = nn.Parameter(record['weight_int8'].float()/WUNIT)
        self.threshold = nn.Parameter(record['threshold_int32'].float()/TUNIT)
        self.register_buffer('mapping', record['mapping'].long())
        self.register_buffer('margin_scale', margin_scale.float().clamp_min(1))
        self.register_buffer('binary', ((torch.arange(8, device=self.weight.device)[:, None]
                                       >> torch.arange(3, device=self.weight.device)) & 1).float())

    def forward(self, xq):
        w = ste_round(self.weight*WUNIT).clamp(-128, 127)
        tau = ste_round(self.threshold*TUNIT)
        # Integer-valued FP32 products/sums are exact within the admitted range;
        # do not use a TF32 GEMM to implement integer source predicates.
        margin = -tau + torch.zeros((*xq.shape[:-1], 3), device=xq.device)
        for t in range(10):
            margin = margin+xq[..., t, None]*w[:, t]
        logits = margin/self.margin_scale
        logprob = (F.logsigmoid(logits)[..., None, :]*self.binary
                   + F.logsigmoid(-logits)[..., None, :]*(1-self.binary)).sum(-1)
        probability = logprob[..., torch.argsort(self.mapping)].exp()
        address = ((margin >= 0).long() << torch.arange(3, device=xq.device)).sum(-1)
        return probability, self.mapping[address], margin

    def export(self, base):
        out = copy.deepcopy(base)
        w = (self.weight.detach().cpu()*WUNIT).round().clamp(-128, 127).long()
        tau = (self.threshold.detach().cpu()*TUNIT).round().long()
        lower = torch.where(w >= 0, w*(-2048), w*2047)
        upper = torch.where(w >= 0, w*2047, w*(-2048))
        lo = torch.cat((torch.zeros(3, 1, dtype=torch.int64), lower.cumsum(1)), 1)-tau[:, None]
        hi = torch.cat((torch.zeros(3, 1, dtype=torch.int64), upper.cumsum(1)), 1)-tau[:, None]
        if int(lo.min()) < -(1 << 23) or int(hi.max()) >= (1 << 23):
            raise ValueError('trained source prefix exceeds Acc24')
        out.update(weight_int8=w.to(torch.int8), threshold_int32=tau.to(torch.int32),
                   unclamped_thresholds=tau.tolist(), variable_rows=torch.ones(3, dtype=torch.bool),
                   constant_gates=torch.zeros(3, dtype=torch.bool),
                   dot_min_int64=lower.sum(1), dot_max_int64=upper.sum(1),
                   prefix_min_int64=lo, prefix_max_int64=hi,
                   training='fixed-grid source-only QAT; round-even W8 and integer source decision thresholds')
        return out


def pack_groups(probability, codes):
    """[B,P4,C,8] -> stable current-hard NRV groups [B,P4,G,NR4,8]."""
    b, positions, channels, _ = probability.shape
    assert positions == P and channels % NR == 0
    live = (codes != 0).any(1)
    channel = torch.arange(channels, device=codes.device).expand(b, -1)
    order = torch.argsort(channel+channels*(~live), dim=1, stable=True)
    indices = order[:, None, :, None].expand(b, P, channels, 8)
    packed = probability.gather(2, indices)
    selected_live = live.gather(1, order)[:, None, :, None]
    zero = F.one_hot(torch.zeros((), dtype=torch.long, device=codes.device), 8).to(probability.dtype)
    packed = torch.where(selected_live, packed, zero)
    return packed.reshape(b, P, channels//NR, NR, 8), live


def regularizers(probability, codes, dictionary):
    grouped, live = pack_groups(probability, codes)
    # Sum over positions, groups, classes; mean over real spatial P4 blocks.
    soft_nr = (1-(1-grouped[..., 1:]).prod(3)).sum((1, 2, 3)).mean()
    raw = probability.reshape(probability.shape[0], P, -1, NR, 8)
    fixed_channel_soft = (1-(1-raw[..., 1:]).prod(3)).sum((1, 2, 3)).mean()
    spike_cost = (probability @ dictionary.sum(1)).mean()/dictionary.shape[1]
    hard = F.one_hot(codes, 8).to(probability.dtype)
    hard_groups, _ = pack_groups(hard, codes)
    hard_nr = hard_groups[..., 1:].sum(3).ne(0).sum((1, 2, 3)).float().mean()
    hard_rate = dictionary[codes].float().mean()
    return soft_nr, spike_cost, hard_nr, hard_rate, fixed_channel_soft, live.sum(1).float().mean()


def time_decode(dictionary):
    rows = []
    for column in np.asarray(dictionary).T:
        if np.any(column) and not any(np.array_equal(column, r) for r in rows):
            rows.append(column)
    return np.stack(rows, axis=1)


def count_nr(codes, decode):
    """Same P4/NR4/empty-row rule as psn/nrv_window_opportunity.py."""
    hist = np.zeros(5, np.int64)
    retained = packets = 0
    for start in range(0, len(codes), P):
        rows = codes[start:start+P].T
        rows = rows[np.any(rows != 0, axis=1)]
        retained += len(rows)
        if not len(rows):
            continue
        rows = np.pad(rows, ((0, (-len(rows)) % NR), (0, 0)))
        grouped = rows.reshape(-1, NR, P)
        members = decode[grouped].sum(1, dtype=np.int64)
        hist += np.bincount(members.ravel(), minlength=5)
        packets += len(grouped)
    return {'scalar_events': int(hist @ np.arange(5)),
            'merged_destinations': int(hist[1:].sum()), 'member_histogram': hist[1:].tolist(),
            'retained_source_rows': retained, 'nr4_packets': packets}


def self_check():
    generator = torch.Generator().manual_seed(9210)
    codes = torch.randint(0, 8, (3, P, 12), generator=generator)
    codes[0, :, :5] = 0
    codes[2] = 0
    probabilities = F.one_hot(codes, 8).float().requires_grad_()
    dictionary = ((torch.arange(8)[:, None] >> torch.arange(10)) & 1).float()
    cost, _, hard, _, _, _ = regularizers(probabilities, codes, dictionary)
    direct = sum(count_nr(x.numpy(), np.eye(8, dtype=np.uint8)[:, 1:])['merged_destinations']
                 for x in codes)/len(codes)
    assert abs(float(cost)-direct) < 1e-5 and float(cost) == float(hard)
    cost.backward()
    assert torch.isfinite(probabilities.grad).all()
    rec = {'weight_int8': torch.randint(-128, 128, (3, 10), generator=generator).to(torch.int8),
           'threshold_int32': torch.tensor([36078, 39619, 24721]), 'mapping': torch.randperm(8, generator=generator)}
    producer = QuantizedProducer(rec, torch.ones(3))
    xq = torch.randint(-2048, 2048, (30, 10), generator=generator).float()
    _, observed, margin = producer(xq)
    expected_margin = xq.long() @ rec['weight_int8'].long().T-rec['threshold_int32']
    expected = rec['mapping'][((expected_margin >= 0).long() << torch.arange(3)).sum(-1)]
    assert torch.equal(margin.long(), expected_margin) and torch.equal(observed, expected)
    print('SELF_CHECK_PASS hard-NRV agreement, zero rows, stable compaction, gradient, integer predicates', flush=True)


def load_system(args):
    from run_bn_probe import ALL_FC1, build_model, set_bn_mode
    from evaluate_stage2_deployment import S2, CoarseReady, install_saved_consumers, tag
    from train_class_shift_probe import install_class_consumers, load_variant
    alg = args.root/'algorithm'
    old = json.loads((alg/'stage2_temporal_codes/run.json').read_text())
    args.code_root = args.root/'code/SDformer'
    args.config, args.checkpoint = Path(old['config']), Path(old['checkpoint'])
    args.data = args.root.parent/'sdformer_codex/SDformer/data/Datasets/DSEC/saved_flow_data'
    model, _, _, _ = build_model(args)
    model.requires_grad_(False)
    modules = dict(model.named_modules())
    stats = torch.load(alg/'valid825_cal32/train_calibration.pt', map_location='cpu', weights_only=False)
    params = torch.load(alg/'stage2_temporal_codes/integer_parameters.pt', map_location='cpu', weights_only=False)
    set_bn_mode(model, stats, ALL_FC1)
    gpu = install_saved_consumers(model, params)
    current = {'cache': None, 'capture': False}
    install_class_consumers(model, gpu, alg, current)
    consumers = torch.load(alg/'stage2_class_shift/consumers.pt', map_location='cpu', weights_only=False)
    load_variant(gpu, consumers['power2_trained'])
    parameters = torch.load(alg/'direct_code_integer/parameters.pt', map_location='cpu', weights_only=False)
    book = np.load(alg/'stage2_temporal_codes/codebooks.npz')
    dictionaries = {p: torch.from_numpy(book[tag(p)+'_dictionary']).cuda().float() for p in S2}
    native = {p: modules[p+'sn1.spiking_neuron'].forward for p in S2}
    records = {}

    def stop(module, inputs, output):
        current['flow'] = output.detach().sum(0)
        raise CoarseReady()

    modules['sttmultires_unet.preds.2'].register_forward_hook(stop)
    return model, modules, gpu, current, parameters, dictionaries, native, records


def install_sources(system, parameters, capture=False):
    from evaluate_stage2_deployment import S2, tag
    from probe_direct_code_integer import integer_codes
    model, modules, gpu, current, _, dictionaries, native, records = system
    for prefix in S2:
        record = {k: v.cuda() if torch.is_tensor(v) else v for k, v in parameters[prefix].items()}

        def source(self, x, p=prefix, q=record):
            shape = x.shape
            vectors = x.reshape(shape[0], -1, gpu[p]['w'].shape[1]).permute(1, 2, 0)
            codes, _, _ = integer_codes(vectors, q)
            if capture:
                assert vectors.shape == (1200, 384, 10)
                starts = torch.linspace(0, len(vectors)//P-1, current['blocks'], device=x.device).round().long()*P
                positions = starts[:, None]+torch.arange(P, device=x.device)[None]
                xq = (vectors[positions]/q['input_step']).round().clamp(-2048, 2047).short()
                # Native frozen PSN targets on the SAME integer-student input;
                # all training labels are generated without consulting validation.
                original = native[p](x).detach().ne(0).reshape(shape[0], -1, 384)
                raw = (original.long() << torch.arange(10, device=x.device)[:, None, None]).sum(0)
                target = gpu[p]['map'][raw][positions]
                records.setdefault(p, []).append({'xq': xq.cpu(), 'target': target.cpu().byte(),
                                                  'start_codes': codes[positions].cpu().byte(),
                                                  'position_indices': positions.cpu(), 'source_shape': list(shape)})
            if current.get('count_codes'):
                a = codes.cpu().numpy().astype(np.uint8)
                d = dictionaries[p].cpu().numpy().astype(np.uint8)
                if current.get('save_codes_dir') is not None:
                    directory = Path(current['save_codes_dir'])
                    directory.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(directory/(tag(p)+'.npz'), codes=a)
                current['code_counts'][tag(p)] = {
                    'shape': list(a.shape), 'class_histogram': np.bincount(a.ravel(), minlength=8).tolist(),
                    'spike_ones': int(d[a].sum()), 'spike_elements': int(a.size*10),
                    'class': count_nr(a, np.eye(8, dtype=np.uint8)[:, 1:]),
                    'time': count_nr(a, time_decode(d))}
            return (dictionaries[p][codes].permute(2, 0, 1)*gpu[p]['theta_source']).reshape(shape)

        neuron = modules[prefix+'sn1.spiking_neuron']
        neuron.forward = types.MethodType(source, neuron)


def capture_train(args, system, out):
    from run_bn_probe import input_frame
    from evaluate_stage2_deployment import S2, CoarseReady, tag
    from spikingjelly.activation_based import functional
    model, _, _, current, parameters, _, _, records = system
    current['blocks'] = args.blocks
    install_sources(system, parameters, capture=True)
    train = json.loads((args.root/'algorithm/direct_code_integer/run.json').read_text())['train']
    with torch.no_grad():
        for i, filename in enumerate(train):
            functional.reset_net(model)
            x, _, _ = input_frame(args.data, filename, targets=False)
            try:
                model(x)
            except CoarseReady:
                current.pop('flow')
            del x
            print('CAPTURE', i+1, filename, flush=True)
    for p in S2:
        rows = records.pop(p)
        cache = {key: torch.stack([r[key] for r in rows]) for key in ('xq', 'target', 'start_codes')}
        cache.update(frames=train, position_indices=rows[0]['position_indices'], source_shape=rows[0]['source_shape'])
        torch.save(cache, out/(tag(p)+'_train.pt'))
    save_json(out/'capture.json', {'frames': train, 'blocks_per_frame': args.blocks, 'P': P, 'C': 384, 'T': 10,
              'shape_per_module': [len(train), args.blocks, P, 384, 10],
              'input': 'saved integer-student signed12 quantizer, fixed scales; actual upstream full network',
              'positions': 'P4-aligned contiguous positions; uniformly distributed block starts; no independent channel sampling',
              'targets': 'native sn1 PSN then fixed K8 map on each same captured tensor',
              'start': 'all six saved integer bits3 sources active; original consumers and dynamic BN2 remain'})


def fit(args, system, out):
    from evaluate_stage2_deployment import S2, tag
    from train_mfpsn_probe import ATanSpike
    _, modules, gpu, _, initial, dictionaries, _, _ = system
    variants = {name: {} for name in AXES}
    history, normalization, weight_statistics = {}, {}, {}
    for module_index, p in enumerate(S2):
        cache = torch.load(out/(tag(p)+'_train.pt'), map_location='cpu', weights_only=False)
        xq, targets = cache['xq'].cuda().float(), cache['target'].cuda().long()
        assert xq.shape[1:] == (args.blocks, P, 384, 10)
        base = {k: v.cuda() if torch.is_tensor(v) else v for k, v in initial[p].items()}
        with torch.no_grad():
            base_margin = -base['threshold_int32'].float()+torch.zeros((*xq.shape[:-1], 3), device='cuda')
            for t in range(10):
                base_margin += xq[..., t, None]*base['weight_int8'].float()[:, t]
            scale = base_margin.std((0, 1, 2, 3)).clamp_min(1)
            del base_margin
        b = (gpu[p]['B'].float() @ gpu[p]['E'].T)[:, 1:]
        w, tau = gpu[p]['w'].float(), gpu[p]['tau'].float()
        w2 = modules[p+'fc2'].weight.detach()
        dictionary = dictionaries[p]
        with torch.no_grad():
            sums = torch.zeros((10, w.shape[0]), dtype=torch.float64, device='cuda')
            squares = torch.zeros_like(sums)
            count = 0
            for frame_codes in targets:
                reference_partial = F.one_hot(frame_codes.reshape(-1, 384), 8)[..., 1:].float().permute(0, 2, 1) @ w.T
                reference_margin = torch.einsum('tk,pkh->tph', b, reference_partial)-tau[:, None]
                sums += reference_margin.double().sum(1)
                squares += reference_margin.double().square().sum(1)
                count += reference_margin.shape[1]
            sigma = (squares/count-(sums/count).square()).clamp_min(1).sqrt().float()[:, None]
            del sums, squares, reference_partial, reference_margin
        weight_statistics[tag(p)] = {
            'fc1_integer_W': {'zero': int(w.eq(0).sum()), 'elements': w.numel(), 'zero_rate': float(w.eq(0).float().mean())},
            'fc2_actual_float_W': {'zero': int(w2.eq(0).sum()), 'elements': w2.numel(), 'zero_rate': float(w2.eq(0).float().mean())},
            'fixed_onehot7_B': {'zero': int(b.eq(0).sum()), 'elements': b.numel(), 'zero_rate': float(b.eq(0).float().mean())}}
        starting = QuantizedProducer(base, scale).cuda()
        with torch.no_grad():
            probe = xq[:4].reshape(-1, P, 384, 10)
            prob, code, _ = starting(probe)
            costs = regularizers(prob, code, dictionary)
            nr_den, spike_den = float(costs[0].clamp_min(1)), float(costs[1].clamp_min(.001))
            check = starting(xq)[1].cpu().byte()
            assert torch.equal(check, cache['start_codes']), 'QAT initialization changed starting integer source'
        normalization[tag(p)] = {'margin_std': scale.cpu().tolist(), 'soft_nr4_initial': nr_den,
                                'soft_spike_rate_initial': spike_den,
                                'normalizer_data': 'first four fixed training frames only'}
        print('START', tag(p), json.dumps(normalization[tag(p)]), flush=True)
        history[tag(p)] = {}
        for axis in AXES:
            producer = copy.deepcopy(starting)
            optimizer = torch.optim.Adam(producer.parameters(), lr=args.lr)
            generator = torch.Generator(device='cuda').manual_seed(9210+module_index)
            trace = []
            for step in range(args.steps):
                frame = step % xq.shape[0]
                selected = torch.randperm(args.blocks, device='cuda', generator=generator)[:4]
                inputs, labels = xq[frame, selected], targets[frame, selected]
                probability, codes, _ = producer(inputs)
                ce = F.nll_loss(probability.clamp_min(1e-12).log().reshape(-1, 8), labels.reshape(-1))
                assignment = F.one_hot(codes, 8).float()+(probability-probability.detach())
                partial = assignment.reshape(-1, 384, 8)[..., 1:].permute(0, 2, 1) @ w.T
                margin = torch.einsum('tk,pkh->tph', b, partial)-tau[:, None]
                with torch.no_grad():
                    target_partial = F.one_hot(labels.reshape(-1, 384), 8)[..., 1:].float().permute(0, 2, 1) @ w.T
                    target_margin = torch.einsum('tk,pkh->tph', b, target_partial)-tau[:, None]
                    hidden_target = target_margin.ge(0).float()*gpu[p]['theta_output']
                    reference_fc2 = F.linear(hidden_target, w2)
                normalized = margin/sigma
                consumer = F.mse_loss(normalized, target_margin/sigma)
                hidden = ATanSpike.apply(normalized)*gpu[p]['theta_output']
                fc2_loss = F.mse_loss(F.linear(hidden, w2), reference_fc2)/reference_fc2.var(unbiased=False).clamp_min(.01)
                soft_nr, soft_spike, hard_nr, hard_spike, fixed_soft, live = regularizers(probability, codes, dictionary)
                penalty = (soft_nr/nr_den if axis == 'nr4_cost' else soft_spike/spike_den
                           if axis == 'spike_cost' else soft_nr*0)
                loss = ce+.25*consumer+.1*fc2_loss+args.regularizer_weight*penalty
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if step % 64 == 0 or step+1 == args.steps:
                    row = {'step': step+1, 'loss': float(loss), 'ce': float(ce),
                           'consumer_margin_mse': float(consumer), 'fc2_mse': float(fc2_loss),
                           'regularizer': float(penalty), 'soft_nr4_per_P4': float(soft_nr),
                           'hard_nr4_per_P4': float(hard_nr), 'fixed_uncompressed_soft_nr4': float(fixed_soft),
                           'hard_spike_rate': float(hard_spike), 'soft_spike_rate': float(soft_spike),
                           'retained_rows_per_P4': float(live),
                           'code_accuracy': float(codes.eq(labels).float().mean())}
                    trace.append(row)
                    print('FIT', tag(p), axis, json.dumps(row), flush=True)
            variants[axis][p] = producer.export(initial[p])
            history[tag(p)][axis] = trace
            torch.save(variants, out/'parameters.pt')
            save_json(out/'training.json', history)
            del optimizer, producer
        del xq, targets, cache, starting, probe, prob, code, check
        torch.cuda.empty_cache()
    save_json(out/'normalization.json', normalization)
    save_json(out/'teacher_weight_sparsity.json', weight_statistics)


def evaluate(args, system, out):
    from run_bn_probe import input_frame
    from evaluate_stage2_deployment import CoarseReady, summarize
    from spikingjelly.activation_based import functional
    model, _, _, current, initial, _, _, _ = system
    parameters = torch.load(out/'parameters.pt', map_location='cpu', weights_only=False)
    if not getattr(args, 'skip_start', False):
        parameters = {'start_integer': initial, **parameters}
    valid = json.loads((args.root/'algorithm/samples.json').read_text())['valid'][:10]
    summaries = {}
    current['count_codes'] = True
    for axis, params in parameters.items():
        install_sources(system, params)
        rows = []
        with torch.no_grad():
            for i, filename in enumerate(valid):
                functional.reset_net(model)
                current['code_counts'] = {}
                if args.save_codes:
                    current['save_codes_dir'] = out/'codes'/axis/Path(filename).stem
                x, label, mask = input_frame(args.data, filename)
                try:
                    model(x)
                except CoarseReady:
                    pred = F.interpolate(current.pop('flow'), (480, 640), mode='bilinear', align_corners=False)
                error = torch.linalg.vector_norm(pred.permute(0, 2, 3, 1)[mask]
                                                -label.permute(0, 2, 3, 1)[mask], dim=1)
                total, pixels = float(error.double().sum()), error.numel()
                rows.append({'file': filename, 'valid_pixels': pixels, 'aee_sum': total,
                             'AEE': total/pixels, 'sources': current['code_counts']})
                save_json(out/(axis+'_valid10_frames.json'), rows)
                print('VALID', axis, i+1, rows[-1]['AEE'], flush=True)
                del x, label, mask, pred, error
        result = summarize(rows, True)
        sources = [source for r in rows for source in r['sources'].values()]
        result['source_counts'] = {route: {key: sum(s[route][key] for s in sources)
                                          for key in ('scalar_events', 'merged_destinations', 'retained_source_rows', 'nr4_packets')}
                                   for route in ('class', 'time')}
        result['spike_ones'] = sum(s['spike_ones'] for s in sources)
        result['spike_elements'] = sum(s['spike_elements'] for s in sources)
        result['spike_rate'] = result['spike_ones']/result['spike_elements']
        result['zero_source_codes'] = sum(s['class_histogram'][0] for s in sources)
        result['source_codes'] = sum(sum(s['class_histogram']) for s in sources)
        result['source_zero_rate'] = result['zero_source_codes']/result['source_codes']
        result['scope'] = 'valid10 actual network; complete 60 source codes; source-control NR4 opportunity, not cycles'
        summaries[axis] = result
        save_json(out/'summary.json', summaries)
        print('COMPLETE', axis, json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--stage', choices=('all', 'capture', 'train', 'evaluate'), default='all')
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--blocks', type=int, default=32)
    parser.add_argument('--lr', type=float, default=.002)
    parser.add_argument('--regularizer-weight', type=float, default=.1)
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--save-codes', action='store_true', help='Persist complete validation source codes for the CPU service model.')
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.root is None:
        parser.error('--root is required')
    out = args.root/'algorithm/nrv_cost_probe'
    out.mkdir(parents=True, exist_ok=True)
    if args.stage in ('all', 'capture'):
        save_json(out/'run.json', {'axes': list(AXES), 'same_start': 'direct_code_integer/parameters.pt',
                  'steps_each': args.steps, 'lr': args.lr, 'regularizer_weight': args.regularizer_weight,
                  'blocks_per_train_frame': args.blocks, 'minibatch_blocks': 4, 'P': P, 'NR': NR,
                  'training_scope': 'only source projection and source decision threshold; no consumer tau/B/W/dictionary/map/theta change',
                  'quantization': 'fixed signed12 input scale; signed8 weight grid; integer source threshold; STE round',
                  'distillation': 'CE+.25 normalized consumer-margin MSE+.1 normalized true FC2 MSE; native sn1 targets on same student input',
                  'proxy': 'independent categorical expectation within detached current-hard NRV groups',
                  'regularizer_scale': 'each proxy divided by its initial mean on first four training frames; same lambda',
                  'unpriced': ['output-specific weight-zero packing', 'queue/metadata/generation', 'weight supply', 'state/PSN service', 'global bandwidth'],
                  'selection': 'fixed settings, train32 only; no valid-based fit or checkpoint selection; no full825'})
    system = load_system(args)
    if args.stage in ('all', 'capture'):
        capture_train(args, system, out)
    if args.stage in ('all', 'train'):
        fit(args, system, out)
    if args.stage in ('all', 'evaluate'):
        evaluate(args, system, out)


if __name__ == '__main__':
    main()
