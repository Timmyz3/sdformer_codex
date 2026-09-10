"""Fit direct source-code producers in all six S2 FFNs, using train frames only.

The fixed teacher is the existing power2-trained K8 consumer model. One pass
captures the actual inputs and target codes at all six original sn1 neurons.
Each producer is fitted locally; validation installs all six replacements at
once. This is a changed FP32 model, not a GustavSNN reproduction or ep34 proof.
The scalar control is a learned uniform eight-level temporal quantizer, not
the complete SpikePack neuron. No GPU work is performed merely on import.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import types

import numpy as np
import torch
import torch.nn.functional as F

from run_bn_probe import ALL_FC1, build_model, input_frame, save_json, set_bn_mode
from evaluate_stage2_deployment import S2, CoarseReady, install_saved_consumers, summarize, tag
from train_class_shift_probe import install_class_consumers, load_variant
from train_direct_code_producer import DirectProducer
from train_mfpsn_probe import ATanSpike


def initialize_producers(x, target, source):
    """Use training data to initialize all controls and the three-bit model."""
    mean = x.mean((0, 1, 2))
    scale = x.std((0, 1, 2)).clamp_min(1e-4)
    a = source.temporal_effective_weight().detach()
    bias = source.bias.detach().reshape(a.shape[0])-source.thresh.detach()
    if source.center_mode != 'zero':
        bias = bias-source.center.detach().reshape(a.shape[0])
    native_bits = F.linear(x, a, bias) >= 0
    bit_shifts = torch.arange(3, device=x.device)
    best = None
    for rows in itertools.combinations(range(a.shape[0]), 3):
        address = (native_bits[..., list(rows)].long() << bit_shifts).sum(-1)
        counts = torch.bincount((address*8+target).reshape(-1), minlength=64).reshape(8, 8)
        correct = int(counts.max(1).values.sum())
        if best is None or correct > best[0]:
            best = correct, rows, counts
    correct, rows, counts = best
    retained = DirectProducer('rows3', mean, scale, a[list(rows)]*scale,
                              bias[list(rows)]+a[list(rows)]@mean, counts.argmax(1))
    count_array = counts.cpu().numpy()
    permutation = max(itertools.permutations(range(8)),
                      key=lambda p: sum(int(count_array[i, p[i]]) for i in range(8)))
    bits = DirectProducer('bits3', mean, scale, retained.weight.detach(), retained.bias.detach(),
                          torch.tensor(permutation, device=x.device))
    normalized = (x-mean)/scale
    centers = torch.stack([normalized[target == c].mean(0) if (target == c).any()
                           else torch.zeros(x.shape[-1], device=x.device) for c in range(8)])
    prior = torch.bincount(target.flatten(), minlength=8).float().clamp_min(1)
    scores = DirectProducer('scores8', mean, scale, centers,
                            -.5*centers.square().sum(1)+(prior/prior.sum()).log())
    _, _, right = torch.linalg.svd(centers-centers.mean(0), full_matrices=False)
    direction = right[0]
    projection = centers @ direction
    order = projection.argsort()
    gain = 7/(projection.max()-projection.min()).clamp_min(.01)
    scalar = DirectProducer('scalar8', mean, scale, (direction*gain)[None],
                            (-projection.min()*gain)[None], order)
    notes = {'rows3_selected': rows, 'rows3_train_accuracy': correct/target.numel(),
             'bits3_initial_code_permutation': permutation,
             'scalar8_initial_code_order': order.tolist(),
             'source_temporal_matrix_shape': list(a.shape)}
    return {'rows3': retained, 'bits3': bits, 'scores8': scores, 'scalar8': scalar}, notes


def fit_module(x, target, source, q, w2, steps, seed, label):
    with torch.no_grad():
        producers, notes = initialize_producers(x, target, source)
        # Ordinary onehot7 is the strongest existing same-output class route.
        b = (q['B'].float() @ q['E'].T)[:, 1:]
        w, tau = q['w'].float(), q['tau'].float()
        teacher_margin = []
        for codes in target:
            partial = F.one_hot(codes, 8)[..., 1:].float().permute(0, 2, 1) @ w.T
            teacher_margin.append(torch.einsum('tk,pkh->tph', b, partial)-tau[:, None])
        teacher_margin = torch.stack(teacher_margin)
        sigma = teacher_margin.std((0, 2)).clamp_min(1)
    traces = {}
    for name in ('bits3', 'scores8', 'scalar8'):
        producer = producers[name]
        optimizer = torch.optim.Adam(producer.parameters(), lr=.012)
        # Identical minibatches for all three trained competitors in this module.
        generator = torch.Generator(device=x.device).manual_seed(seed)
        trace = []
        for step in range(steps):
            frame = step % x.shape[0]
            positions = torch.randperm(x.shape[1], device=x.device, generator=generator)[:16]
            inputs, codes = x[frame, positions], target[frame, positions]
            probability = producer.probabilities(inputs)
            ce = F.nll_loss(probability.clamp_min(1e-12).log().reshape(-1, 8), codes.reshape(-1))
            hard = F.one_hot(producer.codes(inputs), 8).float()
            # Parentheses preserve an exactly hard forward in floating point.
            assignment = hard.detach()+(probability-probability.detach())
            partial = assignment[..., 1:].permute(0, 2, 1) @ w.T
            margin = torch.einsum('tk,pkh->tph', b, partial)-tau[:, None]
            expected = teacher_margin[frame].index_select(1, positions)
            normalized_margin = margin/sigma[:, None]
            normalized_target = expected/sigma[:, None]
            consumer = F.mse_loss(normalized_margin, normalized_target)
            hidden = ATanSpike.apply(normalized_margin)*q['theta_output']
            hidden_target = (normalized_target >= 0).float()*q['theta_output']
            reference_fc2 = F.linear(hidden_target, w2)
            fc2_loss = F.mse_loss(F.linear(hidden, w2), reference_fc2)
            fc2_loss = fc2_loss/reference_fc2.var(unbiased=False).clamp_min(.01)
            loss = ce+.25*consumer+.1*fc2_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 128 == 0 or step+1 == steps:
                with torch.no_grad():
                    row = {'step': step+1, 'loss': float(loss), 'cross_entropy': float(ce),
                           'consumer_margin_mse': float(consumer), 'fc2_mse': float(fc2_loss),
                           'code_accuracy': float((producer.codes(inputs) == codes).float().mean()),
                           'hidden_gate_disagreement': float(((margin >= 0) != (expected >= 0)).float().mean())}
                trace.append(row)
                print('FIT', label, name, json.dumps(row), flush=True)
        traces[name] = trace
    return producers, notes, traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=512)
    parser.add_argument('--positions', type=int, default=128)
    args = parser.parse_args()
    alg, out = args.root/'algorithm', args.root/'algorithm/direct_code_stage2'
    out.mkdir(parents=True, exist_ok=True)
    old = json.loads((alg/'stage2_temporal_codes/run.json').read_text())
    args.code_root = args.root/'code/SDformer'
    args.config, args.checkpoint = Path(old['config']), Path(old['checkpoint'])
    args.data = args.root.parent/'sdformer_codex/SDformer/data/Datasets/DSEC/saved_flow_data'
    train = old['train_codebook_frames']
    valid = json.loads((alg/'samples.json').read_text())['valid'][:10]
    model, _, _, _ = build_model(args)
    model.requires_grad_(False)
    from spikingjelly.activation_based import functional
    modules = dict(model.named_modules())
    stats = torch.load(alg/'valid825_cal32/train_calibration.pt', map_location='cpu', weights_only=False)
    params = torch.load(alg/'stage2_temporal_codes/integer_parameters.pt', map_location='cpu', weights_only=False)
    set_bn_mode(model, stats, ALL_FC1)
    gpu = install_saved_consumers(model, params)
    current = {'cache': None}
    install_class_consumers(model, gpu, alg, current)
    records = torch.load(alg/'stage2_class_shift/consumers.pt', map_location='cpu', weights_only=False)
    load_variant(gpu, records['power2_trained'])
    sources = {p: modules[p+'sn1.spiking_neuron'] for p in S2}
    teacher_sources = {p: module.forward for p, module in sources.items()}
    teacher_mlps = {p: modules[p.rstrip('.')].forward for p in S2}
    cache = {p: {'x': [], 'codes': []} for p in S2}

    def collect(module, inputs, output, prefix):
        q = gpu[prefix]
        t, c = inputs[0].shape[0], q['w'].shape[1]
        x = inputs[0].detach().reshape(t, -1, c)
        g = output.detach().ne(0).reshape_as(x)
        raw = (g.long() << torch.arange(t, device=x.device)[:, None, None]).sum(0)
        count = min(args.positions, x.shape[1])
        pos = torch.linspace(0, x.shape[1]-1, count, device=x.device).round().long()
        cache[prefix]['x'].append(x[:, pos].permute(1, 2, 0).cpu())
        cache[prefix]['codes'].append(q['map'][raw[pos]].cpu())

    handles = [source.register_forward_hook(
        lambda module, inputs, output, p=p: collect(module, inputs, output, p))
        for p, source in sources.items()]

    def stop(module, inputs, output):
        current['flow'] = output.detach().sum(0)
        raise CoarseReady()

    modules['sttmultires_unet.preds.2'].register_forward_hook(stop)
    with torch.no_grad():
        for index, filename in enumerate(train):
            functional.reset_net(model)
            x, _, _ = input_frame(args.data, filename, targets=False)
            try:
                model(x)
            except CoarseReady:
                current.pop('flow')
            del x
            print('CAPTURE', index+1, filename, flush=True)
    for handle in handles:
        handle.remove()
    notes = {'modules': S2, 'train': train, 'valid': valid,
             'positions_per_train_frame': args.positions, 'steps_each': args.steps,
             'teacher': 'power2_trained class consumer; six S2 K8 sources, all12 integer consumers, coarse preds2',
             'changed': 'all six S2 sn1 producers replaced together during complete-network validation',
             'preserved': 'same codebooks, W, B, tau, theta amplitudes, real FC2, dynamic BN2 and shortcuts',
             'training': 'independent local fits on one common teacher capture; same minibatches per format',
             'selection': 'training-only rows/map/permutation/initialization; fixed steps; no validation selection',
             'controls': {'rows3': 'three original PSN rows and train-only address map in every S2 source',
                          'zero_code': 'all six sources emit code0; retain fixed-threshold hidden gates, FC2, BN2 and shortcuts',
                          'identity': 'return zero from all six entire MLP branches; outer identity shortcut remains'},
             'scalar8': 'learned temporal projection plus uniform eight-level quantizer; not complete SpikePack',
             'limits': 'FP32 source coefficients, local pre-BN2 FC2 distillation, new model, valid10 only; no hardware cycle claim',
             'initialization': {}}
    save_json(out/'run.json', notes)
    trained, traces = {}, {}
    for index, prefix in enumerate(S2):
        values = cache.pop(prefix)
        x = torch.stack(values['x']).cuda()
        target = torch.stack(values['codes']).cuda().long()
        del values
        producers, initialization, history = fit_module(
            x, target, sources[prefix], gpu[prefix], modules[prefix+'fc2'].weight.detach(),
            args.steps, 8091+index, tag(prefix))
        trained[prefix], traces[tag(prefix)] = producers, history
        notes['initialization'][tag(prefix)] = initialization
        save_json(out/'run.json', notes)
        save_json(out/'training.json', traces)
        torch.save({p: {name: producer.compiled() for name, producer in collection.items()}
                    for p, collection in trained.items()}, out/'producers.pt')
        del x, target
    book = np.load(alg/'stage2_temporal_codes/codebooks.npz')
    dictionaries = {p: torch.from_numpy(book[tag(p)+'_dictionary']).cuda().float() for p in S2}
    summaries = {}
    for name in ('teacher', 'rows3', 'bits3', 'scores8', 'scalar8', 'zero_code', 'identity'):
        for prefix in S2:
            sources[prefix].forward = teacher_sources[prefix]
            mlp = modules[prefix.rstrip('.')]
            mlp.forward = teacher_mlps[prefix]
            if name == 'identity':
                def skip(self, x):
                    return torch.zeros_like(x)
                mlp.forward = types.MethodType(skip, mlp)
            elif name == 'zero_code':
                # The class consumer is still called and applies its actual tau.
                assert not bool(dictionaries[prefix][0].count_nonzero())
                def zero(self, x):
                    return torch.zeros_like(x)
                sources[prefix].forward = types.MethodType(zero, sources[prefix])
            elif name != 'teacher':
                def direct(self, x, producer=trained[prefix][name],
                           dictionary=dictionaries[prefix], q=gpu[prefix]):
                    shape = x.shape
                    t, c = shape[0], q['w'].shape[1]
                    vectors = x.reshape(t, -1, c).permute(1, 2, 0)
                    codes = producer.codes(vectors)
                    # Software adapter for unmodified consumers. A hardware
                    # path may pass these class IDs without expanding T gates.
                    return (dictionary[codes].permute(2, 0, 1)*q['theta_source']).reshape(shape)
                sources[prefix].forward = types.MethodType(direct, sources[prefix])
        frame_rows = []
        with torch.no_grad():
            for index, filename in enumerate(valid):
                functional.reset_net(model)
                x, label, mask = input_frame(args.data, filename)
                try:
                    model(x)
                except CoarseReady:
                    pred = F.interpolate(current.pop('flow'), (480, 640), mode='bilinear', align_corners=False)
                error = torch.linalg.vector_norm(pred.permute(0, 2, 3, 1)[mask]
                                                -label.permute(0, 2, 3, 1)[mask], dim=1)
                total, pixels = float(error.double().sum()), error.numel()
                row = {'file': filename, 'valid_pixels': pixels, 'aee_sum': total, 'AEE': total/pixels}
                frame_rows.append(row)
                print('FRAME', name, index+1, json.dumps(row), flush=True)
                del x, label, mask, pred, error
        summaries[name] = summarize(frame_rows, True)
        save_json(out/(name+'_valid10_frames.json'), frame_rows)
        save_json(out/'summary.json', summaries)
        print('COMPLETE', name, json.dumps(summaries[name]), flush=True)


if __name__ == '__main__':
    main()
