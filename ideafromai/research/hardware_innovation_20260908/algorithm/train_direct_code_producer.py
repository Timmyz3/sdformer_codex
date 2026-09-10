"""A source-producer probe for an eventual local temporal-batch pipeline.

The existing K8 model first computes ten full PSN decisions, then a 1024-entry
map. Replace only S2 block0's source with three learned temporal decisions and
an eight-entry map. Controls: three original temporal rows, eight learned
linear class scores, and one learned scalar with eight-level quantization.
The scalar control uses SpikePack Eq8-9's direct scalar quantization idea, with
a learned temporal vector and train-only class reorder, not its full model.
All use the same fixed class consumer and real FC2/BN2.
This is a changed model, not GustavSNN reproduction or original-PSN equivalence.
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import types
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from run_bn_probe import ALL_FC1, build_model, input_frame, save_json, set_bn_mode
from evaluate_stage2_deployment import S2, CoarseReady, install_saved_consumers, summarize
from train_class_shift_probe import install_class_consumers, load_variant
from train_mfpsn_probe import ATanSpike


class DirectProducer(nn.Module):
    def __init__(self, kind, mean, scale, weight, bias, mapping=None):
        super().__init__()
        self.kind = kind
        self.register_buffer('mean', mean)
        self.register_buffer('scale', scale)
        self.weight = nn.Parameter(weight.clone())
        self.bias = nn.Parameter(bias.clone())
        if mapping is None:
            mapping = torch.arange(8, device=weight.device)
        self.register_buffer('mapping', mapping)
        self.register_buffer('binary', ((torch.arange(8, device=weight.device)[:, None]
                                       >> torch.arange(3, device=weight.device)) & 1).float())

    def logits(self, x):
        return F.linear((x-self.mean)/self.scale, self.weight, self.bias)

    def probabilities(self, x):
        logits = self.logits(x)
        if self.kind == 'scores8':
            return logits.softmax(-1)
        if self.kind == 'scalar8':
            probability = (-(logits-torch.arange(8, device=x.device)).square()).softmax(-1)
            return probability[..., torch.argsort(self.mapping)]
        logp = (F.logsigmoid(logits)[..., None, :]*self.binary
                + F.logsigmoid(-logits)[..., None, :]*(1-self.binary)).sum(-1)
        # Every mapping used for learned bits is a bijection.
        reverse = torch.argsort(self.mapping)
        return logp[..., reverse].exp()

    def codes(self, x):
        logits = self.logits(x)
        if self.kind == 'scores8':
            return logits.argmax(-1)
        if self.kind == 'scalar8':
            address = logits[..., 0].round().long().clamp(0, 7)
            return self.mapping[address]
        address = ((logits >= 0).long() << torch.arange(3, device=x.device)).sum(-1)
        return self.mapping[address]

    def compiled(self):
        w = self.weight/self.scale
        b = self.bias-(self.weight*self.mean/self.scale).sum(1)
        return {'kind': self.kind, 'weight': w.detach().cpu(), 'bias': b.detach().cpu(),
                'mapping': self.mapping.detach().cpu(), 'numeric': 'FP32; no integer/PPA claim'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--steps', type=int, default=512)
    p.add_argument('--positions', type=int, default=128)
    args = p.parse_args()
    alg = args.root/'algorithm'
    out = alg/'direct_code_producer'
    out.mkdir(parents=True, exist_ok=True)
    old = json.loads((alg/'stage2_temporal_codes/run.json').read_text())
    args.code_root = args.root/'code/SDformer'
    args.config, args.checkpoint = Path(old['config']), Path(old['checkpoint'])
    args.data = args.root.parent/'sdformer_codex/SDformer/data/Datasets/DSEC/saved_flow_data'
    train = old['train_codebook_frames']
    valid = json.loads((alg/'samples.json').read_text())['valid'][:10]
    torch.manual_seed(8091)
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
    prefix, q = S2[0], gpu[S2[0]]
    source = modules[prefix+'sn1.spiking_neuron']
    teacher_forward = source.forward
    samples_x, samples_code = [], []
    def collect(module, inputs, output):
        x = inputs[0].detach().reshape(10, -1, 384)
        g = output.detach().ne(0).reshape_as(x)
        raw = (g.long() << torch.arange(10, device=x.device)[:, None, None]).sum(0)
        positions = torch.linspace(0, x.shape[1]-1, args.positions, device=x.device).round().long()
        samples_x.append(x[:, positions].permute(1, 2, 0).cpu())
        samples_code.append(q['map'][raw[positions]].cpu())

    handle = source.register_forward_hook(collect)
    def stop(module, inputs, output):
        current['flow'] = output.detach().sum(0)
        raise CoarseReady()
    modules['sttmultires_unet.preds.2'].register_forward_hook(stop)
    with torch.no_grad():
        for i, name in enumerate(train):
            functional.reset_net(model)
            x, _, _ = input_frame(args.data, name, targets=False)
            try:
                model(x)
            except CoarseReady:
                current.pop('flow')
            del x
            print('CAPTURE', i+1, name, flush=True)
    handle.remove()
    X, target = torch.stack(samples_x).cuda(), torch.stack(samples_code).cuda().long()
    del samples_x, samples_code
    mean, scale = X.mean((0, 1, 2)), X.std((0, 1, 2)).clamp_min(1e-4)
    # Native sn1 has one shared T10 matrix and its own continuous theta.
    A = source.temporal_effective_weight().detach()
    bias = source.bias.detach().reshape(10)-source.thresh.detach()
    if source.center_mode != 'zero':
        bias = bias-source.center.detach().reshape(10)
    with torch.no_grad():
        native_bits = F.linear(X, A, bias) >= 0
        best = None
        # Strong no-training control: choose three real PSN rows and an optimal
        # per-address class on training frames only. Never consult validation.
        for rows in itertools.combinations(range(10), 3):
            address = (native_bits[..., list(rows)].long()
                       << torch.arange(3, device=X.device)).sum(-1)
            contingency = torch.bincount((address*8+target).reshape(-1), minlength=64).reshape(8, 8)
            mapping = contingency.argmax(1)
            correct = int(contingency.max(1).values.sum())
            if best is None or correct > best[0]:
                best = correct, rows, mapping
        correct, rows, mapping = best
        retained = DirectProducer('rows3', mean, scale, A[list(rows)]*scale,
                                  bias[list(rows)]+A[list(rows)]@mean, mapping)
        # Train a bijective three-bit classifier from the same selected rows.
        # Choose the label permutation maximizing the training contingency.
        address = (native_bits[..., list(rows)].long()
                   << torch.arange(3, device=X.device)).sum(-1)
        counts = torch.bincount((address*8+target).reshape(-1), minlength=64).reshape(8, 8).cpu().numpy()
        permutations = itertools.permutations(range(8))
        permutation = max(permutations, key=lambda v: sum(int(counts[a, v[a]]) for a in range(8)))
        learnt = DirectProducer('bits3', mean, scale, retained.weight.detach(), retained.bias.detach(),
                                torch.tensor(permutation, device=X.device))
        # Eight-class linear LDA initialization uses the same training samples.
        normalized = (X-mean)/scale
        centers = torch.stack([normalized[target == c].mean(0) if (target == c).any()
                               else torch.zeros(10, device=X.device) for c in range(8)])
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
        del native_bits, normalized
        # Ordinary seven-class coefficients, the stronger existing datapath.
        B = (q['B'].float() @ q['E'].T)[:, 1:]
        W = q['w'].float()
        tau = q['tau'].float()
        W2 = modules[prefix+'fc2'].weight.detach()
        teacher_cache = []
        for codes in target:
            partials = F.one_hot(codes, 8)[..., 1:].float().permute(0, 2, 1) @ W.T
            margin = torch.einsum('tk,pkh->tph', B, partials)-tau[:, None]
            teacher_cache.append(margin)
        teacher_margin = torch.stack(teacher_cache)
        sigma = teacher_margin.std((0, 2)).clamp_min(1)
        del teacher_cache

    notes = {'module': prefix, 'train': train, 'valid': valid, 'positions_per_train_frame': args.positions,
        'steps_each': args.steps, 'teacher': 'power2_trained class consumer; all six S2 codes, all12 integer consumers, coarse preds2',
        'changed': 'only S2 block0 sn1 producer; direct learned temporal code decisions',
        'unchanged_consumers': 'same codebook, W, B, tau, continuous theta amplitudes, FC2, current-frame BN2 and shortcut',
        'rows3_selected': rows, 'rows3_train_accuracy': correct/target.numel(),
        'bits3_initial_code_permutation': permutation, 'scalar8_initial_code_order': order.tolist(),
        'scalar8_prior': 'SpikePack arXiv2501.14484v2 Eq8-9; learned time vector and code reorder adapter, not full reproduction',
        'limits': 'local distillation; FP32 source coefficients; not original ep34 or full GustavSNN; valid10 only'}
    save_json(out/'run.json', notes)
    training = {}
    for name, producer in [('bits3', learnt), ('scores8', scores), ('scalar8', scalar)]:
        optimizer = torch.optim.Adam(producer.parameters(), lr=.012)
        trace = []
        for step in range(args.steps):
            frame = step % len(train)
            positions = torch.randperm(args.positions, device=X.device)[:16]
            inputs, codes = X[frame, positions], target[frame, positions]
            probability = producer.probabilities(inputs)
            ce = F.nll_loss(probability.clamp_min(1e-12).log().reshape(-1, 8), codes.reshape(-1))
            # Hard forward and soft gradient: measure the implemented code,
            # not a dense mixture of eight classes at inference.
            hard = F.one_hot(producer.codes(inputs), 8).float()
            assignment = hard.detach()+probability-probability.detach()
            partial = assignment[..., 1:].permute(0, 2, 1) @ W.T
            margin = torch.einsum('tk,pkh->tph', B, partial)-tau[:, None]
            expected = teacher_margin[frame, :, positions]
            normalized_margin, normalized_target = margin/sigma[:, None], expected/sigma[:, None]
            # A smooth local margin objective accounts for W/B consumer cost.
            consumer = F.mse_loss(normalized_margin, normalized_target)
            hidden = ATanSpike.apply(normalized_margin)*q['theta_output']
            hidden_target = (normalized_target >= 0).float()*q['theta_output']
            reference_fc2 = F.linear(hidden_target, W2)
            fc2_loss = F.mse_loss(F.linear(hidden, W2), reference_fc2)/reference_fc2.var(unbiased=False).clamp_min(.01)
            loss = ce+.25*consumer+.1*fc2_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 128 == 0 or step+1 == args.steps:
                with torch.no_grad():
                    predicted = producer.codes(inputs)
                    row = {'step': step+1, 'loss': float(loss), 'cross_entropy': float(ce),
                           'consumer_margin_mse': float(consumer), 'fc2_mse': float(fc2_loss), 'code_accuracy': float((predicted == codes).float().mean()),
                           'hidden_gate_disagreement': float(((margin >= 0) != (expected >= 0)).float().mean())}
                trace.append(row)
                print('FIT', name, json.dumps(row), flush=True)
        training[name] = trace
    save_json(out/'training.json', training)
    producers = {'teacher': None, 'rows3': retained, 'bits3': learnt, 'scores8': scores, 'scalar8': scalar}
    torch.save({name: producer.compiled() for name, producer in producers.items() if producer is not None}, out/'producers.pt')
    book = np.load(alg/'stage2_temporal_codes/codebooks.npz')
    dictionary = torch.from_numpy(book['s2b0_dictionary']).cuda().float()
    summaries = {}
    for name, producer in producers.items():
        if producer is None:
            source.forward = teacher_forward
        else:
            def direct(self, x, trained=producer):
                shape = x.shape
                vector = x.reshape(10, -1, 384).permute(1, 2, 0)
                code = trained.codes(vector)
                # Software adapter only. The hardware connection carries the
                # three-bit code directly and does not expand/encode T10.
                return (dictionary[code].permute(2, 0, 1)*q['theta_source']).reshape(shape)
            source.forward = types.MethodType(direct, source)
        rows_out = []
        with torch.no_grad():
            for i, filename in enumerate(valid):
                functional.reset_net(model)
                x, label, mask = input_frame(args.data, filename)
                try:
                    model(x)
                except CoarseReady:
                    pred = F.interpolate(current.pop('flow'), (480, 640), mode='bilinear', align_corners=False)
                error = torch.linalg.vector_norm(pred.permute(0, 2, 3, 1)[mask]-label.permute(0, 2, 3, 1)[mask], dim=1)
                total, pixels = float(error.double().sum()), error.numel()
                row = {'file': filename, 'valid_pixels': pixels, 'aee_sum': total, 'AEE': total/pixels}
                rows_out.append(row)
                print('FRAME', name, i+1, json.dumps(row), flush=True)
                del x, label, mask, pred, error
        summaries[name] = summarize(rows_out, True)
        save_json(out/(name+'_valid10_frames.json'), rows_out)
        save_json(out/'summary.json', summaries)
        print('COMPLETE', name, json.dumps(summaries[name]), flush=True)


if __name__ == '__main__':
    main()
